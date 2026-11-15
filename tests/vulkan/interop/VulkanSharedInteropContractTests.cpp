#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/VulkanResources.h"
#include "Vulkan/VulkanTestHooks.h"

#include <catch2/catch_test_macros.hpp>

#ifdef _WIN32
#include <Windows.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint64_t bufferSize = 4'096;
    constexpr std::uint32_t textureWidth = 4;
    constexpr std::uint32_t textureHeight = 4;
    constexpr std::uint32_t bytesPerPixel = 4;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t transferSize = static_cast<std::uint64_t>(rowPitch) * textureHeight;

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

    [[nodiscard]] constexpr std::byte payloadByte(std::uint8_t seed, std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((seed + index * 37U + index / 17U * 53U) & 0xFFU);
    }

    void fillUpload(rachis::Device& device, rachis::BufferHandle buffer, std::uint8_t seed)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
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
        for (std::uint64_t index = 0; index < mapped->size; ++index)
        {
            CAPTURE(seed, index);
            REQUIRE(mapped->data[index] == payloadByte(seed, index));
        }
        REQUIRE(device.unmapBuffer(buffer).has_value());
    }

    [[nodiscard]] rachis::TimelinePoint copyBuffer(rachis::Device& device,
                                                   rachis::BufferHandle source,
                                                   rachis::BufferHandle destination,
                                                   rachis::QueueType queueType = rachis::QueueType::copy)
    {
        auto command = rachis::test::acquireCommandBuffer(device, queueType);
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

    [[nodiscard]] constexpr std::byte texelByte(std::uint8_t seed,
                                                std::uint32_t x,
                                                std::uint32_t y,
                                                std::uint32_t channel) noexcept
    {
        return static_cast<std::byte>((seed + x * 29U + y * 47U + channel * 61U) & 0xFFU);
    }

    void fillTextureUpload(rachis::Device& device, rachis::BufferHandle buffer, std::uint8_t seed)
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

    void requireTextureReadback(rachis::Device& device, rachis::BufferHandle readback, std::uint8_t seed)
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

    [[nodiscard]] rachis::CommandBuffer recordTextureReadback(rachis::Device& device,
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
        return std::move(*command);
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

    [[nodiscard]] rachis::TextureBarrier ownershipTextureBarrier(rachis::TextureHandle texture,
                                                                 rachis::OwnershipTransfer ownership) noexcept
    {
        const bool release = ownership == rachis::OwnershipTransfer::releaseExternal;
        return rachis::TextureBarrier{
            .texture = texture,
            .stagesBefore = release ? rachis::PipelineStage::all : rachis::PipelineStage::none,
            .stagesAfter = release ? rachis::PipelineStage::none : rachis::PipelineStage::all,
            .accessBefore = release ? rachis::Access::common : rachis::Access::none,
            .accessAfter = release ? rachis::Access::none : rachis::Access::common,
            .layoutBefore = rachis::TextureLayout::common,
            .layoutAfter = rachis::TextureLayout::common,
            .ownership = ownership,
        };
    }

    [[nodiscard]] rachis::BufferBarrier ownershipBufferBarrier(rachis::BufferHandle buffer,
                                                               rachis::OwnershipTransfer ownership) noexcept
    {
        const bool release = ownership == rachis::OwnershipTransfer::releaseExternal;
        return rachis::BufferBarrier{
            .buffer = buffer,
            .stagesBefore = release ? rachis::PipelineStage::all : rachis::PipelineStage::none,
            .stagesAfter = release ? rachis::PipelineStage::none : rachis::PipelineStage::all,
            .accessBefore = release ? rachis::Access::common : rachis::Access::none,
            .accessAfter = release ? rachis::Access::none : rachis::Access::common,
            .ownership = ownership,
        };
    }

    void requireInvalidArgument(const rachis::Status& status)
    {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == rachis::ErrorCode::invalidArgument);
    }

    [[nodiscard]] bool rejectsTextureImport(rachis::Device& device,
                                            rachis::SharedHandleView handle,
                                            const rachis::TextureDesc& desc)
    {
        auto imported = device.importSharedTexture(handle, desc);
        if (imported)
        {
            REQUIRE(device.destroyTexture(*imported).has_value());
            return false;
        }
        return imported.error().code == rachis::ErrorCode::invalidArgument;
    }

#ifdef _WIN32
    template<typename Call>
    [[nodiscard]] auto callWithBadAlloc(rachis::vulkan::VulkanBadAllocPoint point, Call&& call)
    {
        rachis::vulkan::injectVulkanBadAlloc(point, true);
        auto result = call();
        rachis::vulkan::injectVulkanBadAlloc(point, false);
        return result;
    }

    [[nodiscard]] DWORD processHandleCount()
    {
        DWORD count = 0;
        REQUIRE(GetProcessHandleCount(GetCurrentProcess(), &count) != FALSE);
        return count;
    }

    [[nodiscard]] DWORD settledProcessHandleCount()
    {
        constexpr int stableSamples = 20;
        auto count = processHandleCount();
        for (int stable = 0; stable < stableSamples;)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            const auto next = processHandleCount();
            stable = next == count ? stable + 1 : 0;
            count = next;
        }
        return count;
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

    [[nodiscard]] rachis::TimelinePoint clearThroughView(rachis::Device& device,
                                                         rachis::TextureHandle texture,
                                                         rachis::TextureViewHandle view,
                                                         const rachis::ClearColor& color)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = view,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = color,
            },
        };
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::all,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::common,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::common,
                                    })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        return *submitted;
    }

    void requireSrgbEncodedHalfGray(rachis::Device& device, rachis::BufferHandle readback)
    {
        constexpr int encodedHalf = 188;
        auto mapped = device.mapBuffer(readback);
        REQUIRE(mapped.has_value());
        for (std::uint32_t y = 0; y < textureHeight; ++y)
        {
            for (std::uint32_t x = 0; x < textureWidth; ++x)
            {
                const auto offset =
                    static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * bytesPerPixel;
                CAPTURE(x, y, offset);
                for (std::uint32_t channel = 0; channel < 3; ++channel)
                {
                    CAPTURE(channel);
                    const auto value = std::to_integer<int>(mapped->data[offset + channel]);
                    CHECK(value >= encodedHalf - 1);
                    CHECK(value <= encodedHalf + 1);
                }
                CHECK(std::to_integer<int>(mapped->data[offset + 3]) == 255);
            }
        }
        REQUIRE(device.unmapBuffer(readback).has_value());
    }

    class BarrierCapture final
    {
    public:
        BarrierCapture() noexcept
        {
            rachis::vulkan::captureVulkanResourceBarriers(&barriers_);
        }

        ~BarrierCapture()
        {
            rachis::vulkan::captureVulkanResourceBarriers(nullptr);
        }

        BarrierCapture(const BarrierCapture&) = delete;
        BarrierCapture& operator=(const BarrierCapture&) = delete;
        BarrierCapture(BarrierCapture&&) = delete;
        BarrierCapture& operator=(BarrierCapture&&) = delete;

        [[nodiscard]] std::vector<rachis::vulkan::VulkanCapturedBarrier> take()
        {
            return std::exchange(barriers_, {});
        }

    private:
        std::vector<rachis::vulkan::VulkanCapturedBarrier> barriers_;
    };

    [[nodiscard]] rachis::TimelinePoint transferOwnership(rachis::Device& device,
                                                          rachis::OwnershipTransfer ownership,
                                                          std::span<const rachis::TextureHandle> textures,
                                                          rachis::BufferHandle buffer)
    {
        std::vector<rachis::TextureBarrier> textureBarriers;
        for (const auto texture : textures)
        {
            textureBarriers.push_back(ownershipTextureBarrier(texture, ownership));
        }
        const std::array bufferBarriers{ownershipBufferBarrier(buffer, ownership)};
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .resourceBarriers(*command,
                                      rachis::ResourceBarrierBatch{
                                          .bufferBarriers = bufferBarriers,
                                          .textureBarriers = textureBarriers,
                                      })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        return *submitted;
    }

    void requireNoQueueFamilyTransfers(BarrierCapture& capture)
    {
        for (const auto& barrier : capture.take())
        {
            CAPTURE(barrier.image, barrier.srcQueueFamilyIndex, barrier.dstQueueFamilyIndex);
            CHECK(barrier.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
            CHECK(barrier.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
        }
    }

    void requireExternalTransfer(const rachis::vulkan::VulkanCapturedBarrier& barrier,
                                 rachis::OwnershipTransfer ownership)
    {
        CAPTURE(barrier.image,
                barrier.sharingMode,
                barrier.recordingFamily,
                barrier.srcQueueFamilyIndex,
                barrier.dstQueueFamilyIndex);
        REQUIRE(barrier.recordingFamily != VK_QUEUE_FAMILY_IGNORED);
        const auto localFamily =
            barrier.sharingMode == VK_SHARING_MODE_CONCURRENT ? VK_QUEUE_FAMILY_IGNORED : barrier.recordingFamily;
        if (ownership == rachis::OwnershipTransfer::releaseExternal)
        {
            CHECK(barrier.srcQueueFamilyIndex == localFamily);
            CHECK(barrier.dstQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL);
        }
        else
        {
            CHECK(barrier.srcQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL);
            CHECK(barrier.dstQueueFamilyIndex == localFamily);
        }
        if (barrier.image)
        {
            CHECK(barrier.oldLayout == VK_IMAGE_LAYOUT_GENERAL);
            CHECK(barrier.newLayout == VK_IMAGE_LAYOUT_GENERAL);
        }
    }

    void requireOwnershipTransfers(BarrierCapture& capture, rachis::OwnershipTransfer ownership)
    {
        const auto barriers = capture.take();
        REQUIRE(barriers.size() == 3);
        CHECK_FALSE(barriers[0].image);
        CHECK(barriers[1].image);
        CHECK(barriers[1].sharingMode == VK_SHARING_MODE_EXCLUSIVE);
        CHECK(barriers[2].image);
        for (const auto& barrier : barriers)
        {
            requireExternalTransfer(barrier, ownership);
        }
    }
#endif

} // namespace

TEST_CASE("Vulkan shared buffers use explicit cross-device timeline handoffs",
          "[size:medium][rhi][vulkan][interop][buffer][timeline]")
{
    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto consumer = std::move(*consumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedBufferHandles,
                                            "sharedBufferHandles"))
    {
        return;
    }

    const rachis::BufferDesc sharedDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan shared interop buffer",
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
        .label = "Vulkan shared producer upload",
    });
    auto producerReadback = producer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan shared producer readback",
    });
    auto consumerUpload = consumer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan shared consumer upload",
    });
    auto consumerReadback = consumer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan shared consumer readback",
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
    if (!bufferHandle)
    {
        INFO(bufferHandle.error().message);
        INFO(bufferHandle.error().nativeCode);
    }
    REQUIRE(bufferHandle.has_value());
    auto mismatchDesc = sharedDesc;
    mismatchDesc.size += 1;
    CHECK_FALSE(consumer.importSharedBuffer(bufferHandle->view(), mismatchDesc).has_value());
    auto incompatibleUsageDesc = sharedDesc;
    incompatibleUsageDesc.usage = incompatibleUsageDesc.usage | rachis::BufferUsage::storage;
    CHECK_FALSE(consumer.importSharedBuffer(bufferHandle->view(), incompatibleUsageDesc).has_value());
    auto importedBuffer = consumer.importSharedBuffer(bufferHandle->view(), sharedDesc);
    REQUIRE(importedBuffer.has_value());
    const auto staleBufferView = bufferHandle->view();
    bufferHandle->reset();
    bufferHandle->reset();
    CHECK_FALSE(consumer.importSharedBuffer(staleBufferView, sharedDesc).has_value());
    CHECK_FALSE(consumer.importSharedBuffer(bufferHandle->view(), sharedDesc).has_value());
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

    auto producerReadCommand = rachis::test::acquireCommandBuffer(producer, rachis::QueueType::copy);
    REQUIRE(producerReadCommand.has_value());
    REQUIRE(producer
                .copyBuffer(*producerReadCommand,
                            rachis::BufferCopy{
                                .source = *sharedBuffer,
                                .destination = *producerReadback,
                                .size = bufferSize,
                            })
                .has_value());
    REQUIRE(producer.destroyBuffer(*sharedBuffer).has_value());
    auto producerReadPoint = producer.submit(std::move(*producerReadCommand));
    if (!producerReadPoint)
    {
        INFO(producerReadPoint.error().message);
    }
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

TEST_CASE("Vulkan shared indirect buffers export and import device-address-capable memory",
          "[size:medium][rhi][vulkan][interop][buffer][indirect]")
{
    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto consumer = std::move(*consumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedBufferHandles,
                                            "sharedBufferHandles"))
    {
        return;
    }

    const rachis::BufferDesc indirectDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::indirect | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan shared indirect buffer",
    };
    auto sharedBuffer = producer.createSharedBuffer(indirectDesc);
    if (!sharedBuffer)
    {
        INFO(sharedBuffer.error().message);
    }
    REQUIRE(sharedBuffer.has_value());
    auto bufferHandle = producer.exportSharedBuffer(*sharedBuffer);
    REQUIRE(bufferHandle.has_value());
    auto importedBuffer = consumer.importSharedBuffer(bufferHandle->view(), indirectDesc);
    if (!importedBuffer)
    {
        INFO(importedBuffer.error().message);
    }
    REQUIRE(importedBuffer.has_value());
    bufferHandle->reset();

    REQUIRE(consumer.destroyBuffer(*importedBuffer).has_value());
    REQUIRE(producer.destroyBuffer(*sharedBuffer).has_value());
    producer.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(consumer);
}

#ifdef _WIN32
TEST_CASE("Vulkan shared buffer exports are independent NT handles", "[size:medium][rhi][vulkan][interop][buffer]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(device).sharedBufferHandles,
                                            "sharedBufferHandles"))
    {
        return;
    }

    auto sharedBuffer = device.createSharedBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan twice-exported shared buffer",
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

TEST_CASE("Vulkan shared buffer re-exports name their buffer after an earlier export closes",
          "[size:medium][rhi][vulkan][interop][buffer][timeline]")
{
    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto firstConsumerCreated = rachis::createDevice(vulkanDesc());
    auto secondConsumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(firstConsumerCreated) ||
        rachis::test::skipWithoutCreatedDevice(secondConsumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto firstConsumer = std::move(*firstConsumerCreated);
    auto secondConsumer = std::move(*secondConsumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedBufferHandles,
                                            "sharedBufferHandles"))
    {
        return;
    }

    const rachis::BufferDesc sharedDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan re-exported shared buffer",
    };
    auto producerUpload = producer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan re-export producer upload",
    });
    auto producerReadback = producer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan re-export producer readback",
    });
    auto consumerUpload = secondConsumer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan re-export consumer upload",
    });
    auto consumerReadback = secondConsumer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan re-export consumer readback",
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
#endif

TEST_CASE("Vulkan shared textures use explicit cross-device timeline handoffs",
          "[size:medium][rhi][vulkan][interop][texture][timeline]")
{
    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto consumer = std::move(*consumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedTextureHandles,
                                            "sharedTextureHandles"))
    {
        return;
    }

    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::renderTarget,
        .label = "Vulkan shared interop texture",
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
        .label = "Vulkan shared texture producer upload",
    });
    auto producerReadback = producer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan shared texture producer readback",
    });
    auto consumerUpload = consumer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan shared texture consumer upload",
    });
    auto consumerReadback = consumer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan shared texture consumer readback",
    });
    auto sharedTexture = producer.createSharedTexture(textureDesc);
    REQUIRE(producerUpload.has_value());
    REQUIRE(producerReadback.has_value());
    REQUIRE(consumerUpload.has_value());
    REQUIRE(consumerReadback.has_value());
    REQUIRE(sharedTexture.has_value());

    auto textureHandle = producer.exportSharedTexture(*sharedTexture);
    if (!textureHandle)
    {
        INFO(textureHandle.error().message);
        INFO(textureHandle.error().nativeCode);
    }
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
    const auto staleTextureView = textureHandle->view();
    textureHandle->reset();
    textureHandle->reset();
    CHECK_FALSE(consumer.importSharedTexture(staleTextureView, textureDesc).has_value());
    CHECK_FALSE(consumer.importSharedTexture(textureHandle->view(), textureDesc).has_value());

    constexpr std::uint8_t producerSeed = 17;
    fillTextureUpload(producer, *producerUpload, producerSeed);
    const auto producerPoint = uploadTexture(producer, *producerUpload, *sharedTexture);
    auto sharedProducerPoint = producer.exportTimelinePoint(producerPoint);
    REQUIRE(sharedProducerPoint.has_value());
    auto consumerReady = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, sharedProducerPoint->view());
    REQUIRE(consumerReady.has_value());
    sharedProducerPoint->handle.reset();

    auto consumerRead = recordTextureReadback(consumer, *importedTexture, *consumerReadback);
    auto consumerReadPoint = consumer.submit(std::move(consumerRead));
    REQUIRE(consumerReadPoint.has_value());
    REQUIRE(consumer.wait(*consumerReadPoint).has_value());
    requireTextureReadback(consumer, *consumerReadback, producerSeed);

    constexpr std::uint8_t consumerSeed = 193;
    fillTextureUpload(consumer, *consumerUpload, consumerSeed);
    const auto consumerPoint = uploadTexture(consumer, *consumerUpload, *importedTexture);
    auto sharedConsumerPoint = consumer.exportTimelinePoint(consumerPoint);
    REQUIRE(sharedConsumerPoint.has_value());
    auto producerReady = producer.enqueueSharedTimelineWait(rachis::QueueType::graphics, sharedConsumerPoint->view());
    REQUIRE(producerReady.has_value());
    sharedConsumerPoint->handle.reset();

    auto producerRead = recordTextureReadback(producer, *sharedTexture, *producerReadback);
    REQUIRE(producer.destroyTexture(*sharedTexture).has_value());
    auto producerReadPoint = producer.submit(std::move(producerRead));
    REQUIRE(producerReadPoint.has_value());
    REQUIRE(consumer.destroyTexture(*importedTexture).has_value());
    REQUIRE(producer.wait(*producerReadPoint).has_value());
    requireTextureReadback(producer, *producerReadback, consumerSeed);

    REQUIRE(producer.destroyBuffer(*producerUpload).has_value());
    REQUIRE(producer.destroyBuffer(*producerReadback).has_value());
    REQUIRE(consumer.destroyBuffer(*consumerUpload).has_value());
    REQUIRE(consumer.destroyBuffer(*consumerReadback).has_value());
    producer.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(consumer);
}

TEST_CASE("Vulkan ownership barriers require shared resources and the common layout",
          "[size:medium][rhi][vulkan][interop][barrier]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto buffer = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan ownership barrier ordinary buffer",
    });
    auto texture = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Vulkan ownership barrier ordinary texture",
    });
    REQUIRE(buffer.has_value());
    REQUIRE(texture.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const auto unknownOwnership = static_cast<rachis::OwnershipTransfer>(3);
    for (const auto ownership :
         {rachis::OwnershipTransfer::releaseExternal, rachis::OwnershipTransfer::acquireExternal, unknownOwnership})
    {
        CAPTURE(static_cast<int>(ownership));
        requireInvalidArgument(device.bufferBarrier(*command, ownershipBufferBarrier(*buffer, ownership)));
        requireInvalidArgument(device.textureBarrier(*command, ownershipTextureBarrier(*texture, ownership)));
        const std::array bufferBarriers{ownershipBufferBarrier(*buffer, ownership)};
        const std::array textureBarriers{ownershipTextureBarrier(*texture, ownership)};
        requireInvalidArgument(device.resourceBarriers(*command,
                                                       rachis::ResourceBarrierBatch{
                                                           .bufferBarriers = bufferBarriers,
                                                           .textureBarriers = textureBarriers,
                                                       }));
    }

    const auto& capabilities = rachis::test::requireCapabilities(device);
    if (capabilities.sharedTextureHandles && capabilities.sharedBufferHandles)
    {
        auto sharedBuffer = device.createSharedBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Vulkan ownership barrier shared buffer",
        });
        auto sharedTexture = device.createSharedTexture({
            .width = textureWidth,
            .height = textureHeight,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "Vulkan ownership barrier shared texture",
        });
        REQUIRE(sharedBuffer.has_value());
        REQUIRE(sharedTexture.has_value());

        requireInvalidArgument(device.bufferBarrier(*command, ownershipBufferBarrier(*sharedBuffer, unknownOwnership)));
        requireInvalidArgument(
            device.textureBarrier(*command, ownershipTextureBarrier(*sharedTexture, unknownOwnership)));
        auto fromRenderTarget = ownershipTextureBarrier(*sharedTexture, rachis::OwnershipTransfer::releaseExternal);
        fromRenderTarget.layoutBefore = rachis::TextureLayout::renderTarget;
        requireInvalidArgument(device.textureBarrier(*command, fromRenderTarget));
        auto toRenderTarget = ownershipTextureBarrier(*sharedTexture, rachis::OwnershipTransfer::acquireExternal);
        toRenderTarget.layoutAfter = rachis::TextureLayout::renderTarget;
        toRenderTarget.accessAfter = rachis::Access::renderTarget;
        requireInvalidArgument(device.textureBarrier(*command, toRenderTarget));
        auto fromUndefined = ownershipTextureBarrier(*sharedTexture, rachis::OwnershipTransfer::acquireExternal);
        fromUndefined.layoutBefore = rachis::TextureLayout::undefined;
        requireInvalidArgument(device.textureBarrier(*command, fromUndefined));

        REQUIRE(device
                    .bufferBarrier(*command,
                                   ownershipBufferBarrier(*sharedBuffer, rachis::OwnershipTransfer::acquireExternal))
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    ownershipTextureBarrier(*sharedTexture, rachis::OwnershipTransfer::acquireExternal))
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        REQUIRE(device.destroyTexture(*sharedTexture).has_value());
        REQUIRE(device.destroyBuffer(*sharedBuffer).has_value());
    }
    else
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *texture,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
    }

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    device.collectGarbage();
    requireCleanDiagnostics(device);
}

#ifdef _WIN32
TEST_CASE("Vulkan shared texture imports require the exporter's view formats",
          "[size:medium][rhi][vulkan][interop][texture][timeline]")
{
    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto consumer = std::move(*consumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedTextureHandles,
                                            "sharedTextureHandles"))
    {
        return;
    }

    const std::array srgbViewFormats{rachis::Format::rgba8UnormSrgb};
    const rachis::TextureDesc plainDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::renderTarget,
        .label = "Vulkan shared texture without view formats",
    };
    auto srgbDesc = plainDesc;
    srgbDesc.viewFormats = srgbViewFormats;
    srgbDesc.label = "Vulkan shared texture with an sRGB view format";

    auto producerUpload = producer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan view-format shared texture upload",
    });
    auto producerReadback = producer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan view-format shared texture readback",
    });
    auto plainTexture = producer.createSharedTexture(plainDesc);
    auto srgbTexture = producer.createSharedTexture(srgbDesc);
    REQUIRE(producerUpload.has_value());
    REQUIRE(producerReadback.has_value());
    REQUIRE(plainTexture.has_value());
    REQUIRE(srgbTexture.has_value());
    auto plainExport = producer.exportSharedTexture(*plainTexture);
    auto srgbExport = producer.exportSharedTexture(*srgbTexture);
    REQUIRE(plainExport.has_value());
    REQUIRE(srgbExport.has_value());

    CHECK(rejectsTextureImport(consumer, plainExport->view(), srgbDesc));
    CHECK(rejectsTextureImport(consumer, srgbExport->view(), plainDesc));
    auto plainImport = consumer.importSharedTexture(plainExport->view(), plainDesc);
    auto srgbImport = consumer.importSharedTexture(srgbExport->view(), srgbDesc);
    REQUIRE(plainImport.has_value());
    REQUIRE(srgbImport.has_value());
    plainExport->reset();
    srgbExport->reset();

    constexpr std::uint8_t producerSeed = 59;
    fillTextureUpload(producer, *producerUpload, producerSeed);
    REQUIRE(producer.wait(uploadTexture(producer, *producerUpload, *srgbTexture)).has_value());
    auto srgbView = consumer.createTextureView({
        .texture = *srgbImport,
        .usage = rachis::TextureViewUsage::renderTarget,
        .format = rachis::Format::rgba8UnormSrgb,
        .label = "Vulkan imported shared texture sRGB view",
    });
    REQUIRE(srgbView.has_value());
    const auto cleared = clearThroughView(consumer, *srgbImport, *srgbView, {0.5F, 0.5F, 0.5F, 1.0F});
    handOff(consumer, cleared, producer, rachis::QueueType::graphics);
    auto readPoint = producer.submit(recordTextureReadback(producer, *srgbTexture, *producerReadback));
    REQUIRE(readPoint.has_value());
    REQUIRE(producer.wait(*readPoint).has_value());
    requireSrgbEncodedHalfGray(producer, *producerReadback);

    REQUIRE(consumer.destroyTextureView(*srgbView).has_value());
    REQUIRE(consumer.destroyTexture(*srgbImport).has_value());
    REQUIRE(consumer.destroyTexture(*plainImport).has_value());
    REQUIRE(producer.destroyTexture(*srgbTexture).has_value());
    REQUIRE(producer.destroyTexture(*plainTexture).has_value());
    REQUIRE(producer.destroyBuffer(*producerUpload).has_value());
    REQUIRE(producer.destroyBuffer(*producerReadback).has_value());
    producer.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(consumer);
}

TEST_CASE("Vulkan shared texture exports are independent NT handles", "[size:medium][rhi][vulkan][interop][texture]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(device).sharedTextureHandles,
                                            "sharedTextureHandles"))
    {
        return;
    }

    auto sharedTexture = device.createSharedTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan twice-exported shared texture",
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

TEST_CASE("Vulkan shared texture re-exports name their texture after an earlier export closes",
          "[size:medium][rhi][vulkan][interop][texture][timeline]")
{
    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto firstConsumerCreated = rachis::createDevice(vulkanDesc());
    auto secondConsumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(firstConsumerCreated) ||
        rachis::test::skipWithoutCreatedDevice(secondConsumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto firstConsumer = std::move(*firstConsumerCreated);
    auto secondConsumer = std::move(*secondConsumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedTextureHandles,
                                            "sharedTextureHandles"))
    {
        return;
    }

    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::renderTarget,
        .label = "Vulkan re-exported shared texture",
    };
    auto producerUpload = producer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan texture re-export producer upload",
    });
    auto producerReadback = producer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan texture re-export producer readback",
    });
    auto consumerUpload = secondConsumer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan texture re-export consumer upload",
    });
    auto consumerReadback = secondConsumer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan texture re-export consumer readback",
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
    fillTextureUpload(producer, *producerUpload, exportedSeed);
    REQUIRE(producer.wait(uploadTexture(producer, *producerUpload, *exportedTexture)).has_value());
    auto firstExport = producer.exportSharedTexture(*exportedTexture);
    REQUIRE(firstExport.has_value());
    auto firstImport = firstConsumer.importSharedTexture(firstExport->view(), textureDesc);
    REQUIRE(firstImport.has_value());
    fillTextureUpload(producer, *producerUpload, otherSeed);
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
    auto consumerReadPoint =
        secondConsumer.submit(recordTextureReadback(secondConsumer, *reimported, *consumerReadback));
    REQUIRE(consumerReadPoint.has_value());
    REQUIRE(secondConsumer.wait(*consumerReadPoint).has_value());
    requireTextureReadback(secondConsumer, *consumerReadback, exportedSeed);

    fillTextureUpload(secondConsumer, *consumerUpload, consumerSeed);
    const auto consumerPoint = uploadTexture(secondConsumer, *consumerUpload, *reimported);
    handOff(secondConsumer, consumerPoint, producer, rachis::QueueType::graphics);
    auto otherReadPoint = producer.submit(recordTextureReadback(producer, *otherTexture, *producerReadback));
    REQUIRE(otherReadPoint.has_value());
    REQUIRE(producer.wait(*otherReadPoint).has_value());
    requireTextureReadback(producer, *producerReadback, otherSeed);
    auto exportedReadPoint = producer.submit(recordTextureReadback(producer, *exportedTexture, *producerReadback));
    REQUIRE(exportedReadPoint.has_value());
    REQUIRE(producer.wait(*exportedReadPoint).has_value());
    requireTextureReadback(producer, *producerReadback, consumerSeed);

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

TEST_CASE("Vulkan shared resources move to and from the external queue family at each handoff",
          "[size:medium][rhi][vulkan][interop][barrier][timeline]")
{
    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto consumer = std::move(*consumerCreated);
    const auto& capabilities = rachis::test::requireCapabilities(producer);
    if (rachis::test::skipWithoutCapability(capabilities.sharedTextureHandles && capabilities.sharedBufferHandles,
                                            "sharedTextureHandles and sharedBufferHandles"))
    {
        return;
    }

    const rachis::TextureDesc exclusiveDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Vulkan render-target-only shared texture",
    };
    const rachis::TextureDesc copyableDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::renderTarget,
        .label = "Vulkan copyable shared texture",
    };
    const rachis::BufferDesc bufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan ownership shared buffer",
    };

    auto producerExclusive = producer.createSharedTexture(exclusiveDesc);
    auto producerCopyable = producer.createSharedTexture(copyableDesc);
    auto producerShared = producer.createSharedBuffer(bufferDesc);
    REQUIRE(producerExclusive.has_value());
    REQUIRE(producerCopyable.has_value());
    REQUIRE(producerShared.has_value());
    auto exclusiveExport = producer.exportSharedTexture(*producerExclusive);
    auto copyableExport = producer.exportSharedTexture(*producerCopyable);
    auto bufferExport = producer.exportSharedBuffer(*producerShared);
    REQUIRE(exclusiveExport.has_value());
    REQUIRE(copyableExport.has_value());
    REQUIRE(bufferExport.has_value());
    auto consumerExclusive = consumer.importSharedTexture(exclusiveExport->view(), exclusiveDesc);
    auto consumerCopyable = consumer.importSharedTexture(copyableExport->view(), copyableDesc);
    auto consumerShared = consumer.importSharedBuffer(bufferExport->view(), bufferDesc);
    REQUIRE(consumerExclusive.has_value());
    REQUIRE(consumerCopyable.has_value());
    REQUIRE(consumerShared.has_value());
    exclusiveExport->reset();
    copyableExport->reset();
    bufferExport->reset();

    auto producerView = producer.createTextureView({
        .texture = *producerExclusive,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Vulkan producer render-target-only shared view",
    });
    auto consumerView = consumer.createTextureView({
        .texture = *consumerExclusive,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Vulkan consumer render-target-only shared view",
    });
    REQUIRE(producerView.has_value());
    REQUIRE(consumerView.has_value());

    const auto createStaging = [](rachis::Device& device, std::uint64_t size, bool upload) {
        auto staging = device.createBuffer({
            .size = size,
            .usage = upload ? rachis::BufferUsage::copySource : rachis::BufferUsage::copyDestination,
            .memoryType = upload ? rachis::MemoryType::upload : rachis::MemoryType::readback,
            .label = upload ? "Vulkan ownership upload" : "Vulkan ownership readback",
        });
        REQUIRE(staging.has_value());
        return *staging;
    };
    const auto producerUpload = createStaging(producer, bufferSize, true);
    const auto producerReadback = createStaging(producer, bufferSize, false);
    const auto producerTextureUpload = createStaging(producer, transferSize, true);
    const auto producerTextureReadback = createStaging(producer, transferSize, false);
    const auto consumerUpload = createStaging(consumer, bufferSize, true);
    const auto consumerReadback = createStaging(consumer, bufferSize, false);
    const auto consumerTextureUpload = createStaging(consumer, transferSize, true);
    const auto consumerTextureReadback = createStaging(consumer, transferSize, false);

    BarrierCapture capture;
    const std::array producerTextures{*producerExclusive, *producerCopyable};
    const std::array consumerTextures{*consumerExclusive, *consumerCopyable};
    constexpr auto release = rachis::OwnershipTransfer::releaseExternal;
    constexpr auto acquire = rachis::OwnershipTransfer::acquireExternal;

    constexpr std::uint8_t producerSeed = 23;
    fillUpload(producer, producerUpload, producerSeed);
    fillTextureUpload(producer, producerTextureUpload, producerSeed);
    REQUIRE(
        producer.wait(copyBuffer(producer, producerUpload, *producerShared, rachis::QueueType::graphics)).has_value());
    REQUIRE(producer.wait(uploadTexture(producer, producerTextureUpload, *producerCopyable)).has_value());
    REQUIRE(producer.wait(clearThroughView(producer, *producerExclusive, *producerView, {0.25F, 0.5F, 0.75F, 1.0F}))
                .has_value());
    requireNoQueueFamilyTransfers(capture);
    const auto producerReleased = transferOwnership(producer, release, producerTextures, *producerShared);
    requireOwnershipTransfers(capture, release);
    handOff(producer, producerReleased, consumer, rachis::QueueType::graphics);

    REQUIRE(consumer.wait(transferOwnership(consumer, acquire, consumerTextures, *consumerShared)).has_value());
    requireOwnershipTransfers(capture, acquire);
    REQUIRE(consumer.wait(copyBuffer(consumer, *consumerShared, consumerReadback, rachis::QueueType::graphics))
                .has_value());
    requireReadback(consumer, consumerReadback, producerSeed);
    auto consumerRead = consumer.submit(recordTextureReadback(consumer, *consumerCopyable, consumerTextureReadback));
    REQUIRE(consumerRead.has_value());
    REQUIRE(consumer.wait(*consumerRead).has_value());
    requireTextureReadback(consumer, consumerTextureReadback, producerSeed);

    constexpr std::uint8_t consumerSeed = 151;
    fillUpload(consumer, consumerUpload, consumerSeed);
    fillTextureUpload(consumer, consumerTextureUpload, consumerSeed);
    REQUIRE(
        consumer.wait(copyBuffer(consumer, consumerUpload, *consumerShared, rachis::QueueType::graphics)).has_value());
    REQUIRE(consumer.wait(uploadTexture(consumer, consumerTextureUpload, *consumerCopyable)).has_value());
    REQUIRE(consumer.wait(clearThroughView(consumer, *consumerExclusive, *consumerView, {0.75F, 0.5F, 0.25F, 1.0F}))
                .has_value());
    requireNoQueueFamilyTransfers(capture);
    const auto consumerReleased = transferOwnership(consumer, release, consumerTextures, *consumerShared);
    requireOwnershipTransfers(capture, release);
    handOff(consumer, consumerReleased, producer, rachis::QueueType::graphics);

    REQUIRE(producer.wait(transferOwnership(producer, acquire, producerTextures, *producerShared)).has_value());
    requireOwnershipTransfers(capture, acquire);
    REQUIRE(producer.wait(copyBuffer(producer, *producerShared, producerReadback, rachis::QueueType::graphics))
                .has_value());
    requireReadback(producer, producerReadback, consumerSeed);
    auto producerRead = producer.submit(recordTextureReadback(producer, *producerCopyable, producerTextureReadback));
    REQUIRE(producerRead.has_value());
    REQUIRE(producer.wait(*producerRead).has_value());
    requireTextureReadback(producer, producerTextureReadback, consumerSeed);
    requireNoQueueFamilyTransfers(capture);

    REQUIRE(consumer.destroyTextureView(*consumerView).has_value());
    REQUIRE(producer.destroyTextureView(*producerView).has_value());
    REQUIRE(consumer.destroyTexture(*consumerExclusive).has_value());
    REQUIRE(consumer.destroyTexture(*consumerCopyable).has_value());
    REQUIRE(consumer.destroyBuffer(*consumerShared).has_value());
    REQUIRE(producer.destroyTexture(*producerExclusive).has_value());
    REQUIRE(producer.destroyTexture(*producerCopyable).has_value());
    REQUIRE(producer.destroyBuffer(*producerShared).has_value());
    for (const auto staging : {producerUpload, producerReadback, producerTextureUpload, producerTextureReadback})
    {
        REQUIRE(producer.destroyBuffer(staging).has_value());
    }
    for (const auto staging : {consumerUpload, consumerReadback, consumerTextureUpload, consumerTextureReadback})
    {
        REQUIRE(consumer.destroyBuffer(staging).has_value());
    }
    producer.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(consumer);
}

TEST_CASE("Vulkan shared resource creation and import release native objects when allocation fails",
          "[size:medium][rhi][vulkan][interop][buffer][texture]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(vulkanDesc(&errors));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipWithoutCapability(capabilities.sharedBufferHandles && capabilities.sharedTextureHandles,
                                            "sharedBufferHandles and sharedTextureHandles"))
    {
        return;
    }
    constexpr auto keepAlive = rachis::vulkan::VulkanBadAllocPoint::nativeKeepAlive;

    const rachis::BufferDesc bufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan shared buffer with a failing keep-alive allocation",
    };
    const auto failedBuffer = callWithBadAlloc(keepAlive, [&] { return device.createSharedBuffer(bufferDesc); });
    REQUIRE_FALSE(failedBuffer.has_value());
    CHECK(failedBuffer.error().code == rachis::ErrorCode::outOfMemory);
    auto buffer = device.createSharedBuffer(bufferDesc);
    REQUIRE(buffer.has_value());
    auto bufferExport = device.exportSharedBuffer(*buffer);
    REQUIRE(bufferExport.has_value());
    const auto failedBufferImport =
        callWithBadAlloc(keepAlive, [&] { return device.importSharedBuffer(bufferExport->view(), bufferDesc); });
    REQUIRE_FALSE(failedBufferImport.has_value());
    CHECK(failedBufferImport.error().code == rachis::ErrorCode::outOfMemory);
    auto importedBuffer = device.importSharedBuffer(bufferExport->view(), bufferDesc);
    REQUIRE(importedBuffer.has_value());

    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan shared texture with a failing keep-alive allocation",
    };
    const auto failedTexture = callWithBadAlloc(keepAlive, [&] { return device.createSharedTexture(textureDesc); });
    REQUIRE_FALSE(failedTexture.has_value());
    CHECK(failedTexture.error().code == rachis::ErrorCode::outOfMemory);
    auto texture = device.createSharedTexture(textureDesc);
    REQUIRE(texture.has_value());
    auto textureExport = device.exportSharedTexture(*texture);
    REQUIRE(textureExport.has_value());
    const auto failedTextureImport =
        callWithBadAlloc(keepAlive, [&] { return device.importSharedTexture(textureExport->view(), textureDesc); });
    REQUIRE_FALSE(failedTextureImport.has_value());
    CHECK(failedTextureImport.error().code == rachis::ErrorCode::outOfMemory);
    auto importedTexture = device.importSharedTexture(textureExport->view(), textureDesc);
    REQUIRE(importedTexture.has_value());

    bufferExport->reset();
    textureExport->reset();
    REQUIRE(device.destroyTexture(*importedTexture).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*importedBuffer).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    device.collectGarbage();
    requireCleanDiagnostics(device);
    device = {};
    CHECK(errors.load() == 0);
}

TEST_CASE("Vulkan shared exports keep the handle count and the import registry when allocation fails",
          "[size:medium][rhi][vulkan][interop][buffer][texture]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(vulkanDesc(&errors));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipWithoutCapability(capabilities.sharedBufferHandles && capabilities.sharedTextureHandles,
                                            "sharedBufferHandles and sharedTextureHandles"))
    {
        return;
    }
    constexpr auto registry = rachis::vulkan::VulkanBadAllocPoint::sharedExportRegistry;

    const rachis::BufferDesc bufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan shared buffer with a failing export registration",
    };
    auto mismatchedBufferDesc = bufferDesc;
    mismatchedBufferDesc.usage = rachis::BufferUsage::copySource;
    auto buffer = device.createSharedBuffer(bufferDesc);
    REQUIRE(buffer.has_value());
    const auto handlesBeforeBufferExport = settledProcessHandleCount();
    const auto failedBufferExport = callWithBadAlloc(registry, [&] { return device.exportSharedBuffer(*buffer); });
    REQUIRE_FALSE(failedBufferExport.has_value());
    CHECK(failedBufferExport.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(settledProcessHandleCount() == handlesBeforeBufferExport);
    auto bufferExport = device.exportSharedBuffer(*buffer);
    REQUIRE(bufferExport.has_value());
    auto mismatchedBuffer = device.importSharedBuffer(bufferExport->view(), mismatchedBufferDesc);
    REQUIRE_FALSE(mismatchedBuffer.has_value());
    CHECK(mismatchedBuffer.error().code == rachis::ErrorCode::invalidArgument);

    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan shared texture with a failing export registration",
    };
    auto mismatchedTextureDesc = textureDesc;
    mismatchedTextureDesc.usage = rachis::TextureUsage::copySource;
    auto texture = device.createSharedTexture(textureDesc);
    REQUIRE(texture.has_value());
    const auto handlesBeforeTextureExport = settledProcessHandleCount();
    const auto failedTextureExport = callWithBadAlloc(registry, [&] { return device.exportSharedTexture(*texture); });
    REQUIRE_FALSE(failedTextureExport.has_value());
    CHECK(failedTextureExport.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(settledProcessHandleCount() == handlesBeforeTextureExport);
    auto textureExport = device.exportSharedTexture(*texture);
    REQUIRE(textureExport.has_value());
    CHECK(rejectsTextureImport(device, textureExport->view(), mismatchedTextureDesc));

    bufferExport->reset();
    textureExport->reset();
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    device.collectGarbage();
    requireCleanDiagnostics(device);
    device = {};
    CHECK(errors.load() == 0);
}

TEST_CASE("Vulkan shared imports release the identity handle and native objects when registration fails",
          "[size:medium][rhi][vulkan][interop][buffer][texture]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(vulkanDesc(&errors));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipWithoutCapability(capabilities.sharedBufferHandles && capabilities.sharedTextureHandles,
                                            "sharedBufferHandles and sharedTextureHandles"))
    {
        return;
    }
    constexpr auto registry = rachis::vulkan::VulkanBadAllocPoint::sharedImportRegistry;
    const auto& liveAllocations = rachis::vulkan::test::liveNativeKeepAlives();

    const rachis::BufferDesc bufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan shared buffer with a failing import registration",
    };
    auto buffer = device.createSharedBuffer(bufferDesc);
    REQUIRE(buffer.has_value());
    auto bufferExport = device.exportSharedBuffer(*buffer);
    REQUIRE(bufferExport.has_value());
    const auto handlesBeforeBufferImport = settledProcessHandleCount();
    const auto allocationsBeforeBufferImport = liveAllocations.load();
    const auto failedBufferImport =
        callWithBadAlloc(registry, [&] { return device.importSharedBuffer(bufferExport->view(), bufferDesc); });
    REQUIRE_FALSE(failedBufferImport.has_value());
    CHECK(failedBufferImport.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(liveAllocations.load() == allocationsBeforeBufferImport);
    CHECK(settledProcessHandleCount() == handlesBeforeBufferImport);
    auto importedBuffer = device.importSharedBuffer(bufferExport->view(), bufferDesc);
    REQUIRE(importedBuffer.has_value());
    CHECK(liveAllocations.load() == allocationsBeforeBufferImport + 1);

    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan shared texture with a failing import registration",
    };
    auto texture = device.createSharedTexture(textureDesc);
    REQUIRE(texture.has_value());
    auto textureExport = device.exportSharedTexture(*texture);
    REQUIRE(textureExport.has_value());
    const auto handlesBeforeTextureImport = settledProcessHandleCount();
    const auto allocationsBeforeTextureImport = liveAllocations.load();
    const auto failedTextureImport =
        callWithBadAlloc(registry, [&] { return device.importSharedTexture(textureExport->view(), textureDesc); });
    REQUIRE_FALSE(failedTextureImport.has_value());
    CHECK(failedTextureImport.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(liveAllocations.load() == allocationsBeforeTextureImport);
    CHECK(settledProcessHandleCount() == handlesBeforeTextureImport);
    auto importedTexture = device.importSharedTexture(textureExport->view(), textureDesc);
    REQUIRE(importedTexture.has_value());
    CHECK(liveAllocations.load() == allocationsBeforeTextureImport + 1);

    bufferExport->reset();
    textureExport->reset();
    REQUIRE(device.destroyTexture(*importedTexture).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*importedBuffer).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    device.collectGarbage();
    requireCleanDiagnostics(device);
    device = {};
    CHECK(errors.load() == 0);
}
#endif

TEST_CASE("Vulkan re-exports of imported shared buffers return independent handles to the shared memory",
          "[size:medium][rhi][vulkan][interop][buffer][reexport]")
{
    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto relayCreated = rachis::createDevice(vulkanDesc());
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(relayCreated) || rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto relay = std::move(*relayCreated);
    auto consumer = std::move(*consumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedBufferHandles,
                                            "sharedBufferHandles"))
    {
        return;
    }

    const rachis::BufferDesc sharedDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan re-exported shared buffer",
    };
    auto producerUpload = producer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan re-export producer upload",
    });
    auto consumerReadback = consumer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan re-export consumer readback",
    });
    auto sharedBuffer = producer.createSharedBuffer(sharedDesc);
    REQUIRE(producerUpload.has_value());
    REQUIRE(consumerReadback.has_value());
    REQUIRE(sharedBuffer.has_value());

    constexpr std::uint8_t producerSeed = 71;
    fillUpload(producer, *producerUpload, producerSeed);
    const auto producerPoint = copyBuffer(producer, *producerUpload, *sharedBuffer);
    auto sharedProducerPoint = producer.exportTimelinePoint(producerPoint);
    REQUIRE(sharedProducerPoint.has_value());
    REQUIRE(consumer.enqueueSharedTimelineWait(rachis::QueueType::copy, sharedProducerPoint->view()).has_value());
    sharedProducerPoint->handle.reset();

    auto producerHandle = producer.exportSharedBuffer(*sharedBuffer);
    REQUIRE(producerHandle.has_value());
    auto relayBuffer = relay.importSharedBuffer(producerHandle->view(), sharedDesc);
    REQUIRE(relayBuffer.has_value());

    auto firstReexport = relay.exportSharedBuffer(*relayBuffer);
    if (!firstReexport)
    {
        INFO(firstReexport.error().message);
    }
    REQUIRE(firstReexport.has_value());
    CHECK(firstReexport->view().value != producerHandle->view().value);
    firstReexport->reset();
    auto importedFromProducer = consumer.importSharedBuffer(producerHandle->view(), sharedDesc);
    REQUIRE(importedFromProducer.has_value());
    producerHandle->reset();

    auto secondReexport = relay.exportSharedBuffer(*relayBuffer);
    REQUIRE(secondReexport.has_value());
    REQUIRE(producer.destroyBuffer(*sharedBuffer).has_value());
    auto importedFromRelay = consumer.importSharedBuffer(secondReexport->view(), sharedDesc);
    if (!importedFromRelay)
    {
        INFO(importedFromRelay.error().message);
    }
    REQUIRE(importedFromRelay.has_value());
    secondReexport->reset();

    const auto consumerReadPoint = copyBuffer(consumer, *importedFromRelay, *consumerReadback);
    REQUIRE(consumer.wait(consumerReadPoint).has_value());
    requireReadback(consumer, *consumerReadback, producerSeed);

    REQUIRE(consumer.destroyBuffer(*importedFromProducer).has_value());
    REQUIRE(consumer.destroyBuffer(*importedFromRelay).has_value());
    REQUIRE(relay.destroyBuffer(*relayBuffer).has_value());
    REQUIRE(producer.destroyBuffer(*producerUpload).has_value());
    REQUIRE(consumer.destroyBuffer(*consumerReadback).has_value());
    REQUIRE(producer.wait(producerPoint).has_value());
    producer.collectGarbage();
    relay.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(relay);
    requireCleanDiagnostics(consumer);
}

TEST_CASE("Vulkan re-exports of imported shared textures return independent handles to the shared memory",
          "[size:medium][rhi][vulkan][interop][texture][reexport]")
{
    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto relayCreated = rachis::createDevice(vulkanDesc());
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(relayCreated) || rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto relay = std::move(*relayCreated);
    auto consumer = std::move(*consumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedTextureHandles,
                                            "sharedTextureHandles"))
    {
        return;
    }

    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::renderTarget,
        .label = "Vulkan re-exported shared texture",
    };
    auto producerUpload = producer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan texture re-export producer upload",
    });
    auto consumerReadback = consumer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan texture re-export consumer readback",
    });
    auto sharedTexture = producer.createSharedTexture(textureDesc);
    REQUIRE(producerUpload.has_value());
    REQUIRE(consumerReadback.has_value());
    REQUIRE(sharedTexture.has_value());

    constexpr std::uint8_t producerSeed = 113;
    fillTextureUpload(producer, *producerUpload, producerSeed);
    const auto producerPoint = uploadTexture(producer, *producerUpload, *sharedTexture);
    auto sharedProducerPoint = producer.exportTimelinePoint(producerPoint);
    REQUIRE(sharedProducerPoint.has_value());
    REQUIRE(consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, sharedProducerPoint->view()).has_value());
    sharedProducerPoint->handle.reset();

    auto producerHandle = producer.exportSharedTexture(*sharedTexture);
    REQUIRE(producerHandle.has_value());
    auto relayTexture = relay.importSharedTexture(producerHandle->view(), textureDesc);
    REQUIRE(relayTexture.has_value());

    auto firstReexport = relay.exportSharedTexture(*relayTexture);
    if (!firstReexport)
    {
        INFO(firstReexport.error().message);
    }
    REQUIRE(firstReexport.has_value());
    CHECK(firstReexport->view().value != producerHandle->view().value);
    firstReexport->reset();
    auto importedFromProducer = consumer.importSharedTexture(producerHandle->view(), textureDesc);
    REQUIRE(importedFromProducer.has_value());
    producerHandle->reset();

    auto secondReexport = relay.exportSharedTexture(*relayTexture);
    REQUIRE(secondReexport.has_value());
    REQUIRE(producer.destroyTexture(*sharedTexture).has_value());
    auto importedFromRelay = consumer.importSharedTexture(secondReexport->view(), textureDesc);
    if (!importedFromRelay)
    {
        INFO(importedFromRelay.error().message);
    }
    REQUIRE(importedFromRelay.has_value());
    secondReexport->reset();

    auto consumerRead = recordTextureReadback(consumer, *importedFromRelay, *consumerReadback);
    auto consumerReadPoint = consumer.submit(std::move(consumerRead));
    REQUIRE(consumerReadPoint.has_value());
    REQUIRE(consumer.wait(*consumerReadPoint).has_value());
    requireTextureReadback(consumer, *consumerReadback, producerSeed);

    REQUIRE(consumer.destroyTexture(*importedFromProducer).has_value());
    REQUIRE(consumer.destroyTexture(*importedFromRelay).has_value());
    REQUIRE(relay.destroyTexture(*relayTexture).has_value());
    REQUIRE(producer.destroyBuffer(*producerUpload).has_value());
    REQUIRE(consumer.destroyBuffer(*consumerReadback).has_value());
    REQUIRE(producer.wait(producerPoint).has_value());
    producer.collectGarbage();
    relay.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(relay);
    requireCleanDiagnostics(consumer);
}

TEST_CASE("Vulkan re-exports of imported shared textures require the exporter's view formats",
          "[size:medium][rhi][vulkan][interop][texture][reexport]")
{
    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto relayCreated = rachis::createDevice(vulkanDesc());
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(relayCreated) || rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto relay = std::move(*relayCreated);
    auto consumer = std::move(*consumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedTextureHandles,
                                            "sharedTextureHandles"))
    {
        return;
    }

    const std::array srgbViewFormats{rachis::Format::rgba8UnormSrgb};
    const rachis::TextureDesc plainDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::renderTarget,
        .label = "Vulkan re-exported shared texture without view formats",
    };
    auto srgbDesc = plainDesc;
    srgbDesc.viewFormats = srgbViewFormats;
    srgbDesc.label = "Vulkan re-exported shared texture with an sRGB view format";

    auto sharedTexture = producer.createSharedTexture(srgbDesc);
    REQUIRE(sharedTexture.has_value());
    auto producerHandle = producer.exportSharedTexture(*sharedTexture);
    REQUIRE(producerHandle.has_value());
    auto relayTexture = relay.importSharedTexture(producerHandle->view(), srgbDesc);
    REQUIRE(relayTexture.has_value());
    producerHandle->reset();
    auto reexport = relay.exportSharedTexture(*relayTexture);
    REQUIRE(reexport.has_value());
    REQUIRE(producer.destroyTexture(*sharedTexture).has_value());

    CHECK(rejectsTextureImport(consumer, reexport->view(), plainDesc));
    auto imported = consumer.importSharedTexture(reexport->view(), srgbDesc);
    INFO((imported ? std::string_view{} : std::string_view{imported.error().message}));
    REQUIRE(imported.has_value());
    reexport->reset();

    REQUIRE(consumer.destroyTexture(*imported).has_value());
    REQUIRE(relay.destroyTexture(*relayTexture).has_value());
    producer.collectGarbage();
    relay.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(relay);
    requireCleanDiagnostics(consumer);
}

#if RACHIS_ENABLE_D3D12
namespace
{

    struct SharedFence final
    {
        rachis::Device device;
        rachis::SharedTimelinePoint shared;
        std::uint64_t signaledValue = 0;

        [[nodiscard]] rachis::SharedTimelinePointView at(std::uint64_t value) const noexcept
        {
            return rachis::SharedTimelinePointView{
                .handle = shared.handle.view(),
                .value = value,
            };
        }

        void signal(std::uint64_t value)
        {
            while (signaledValue < value)
            {
                auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
                REQUIRE(command.has_value());
                auto submitted = device.submit(std::move(*command));
                REQUIRE(submitted.has_value());
                signaledValue = submitted->value();
            }
        }
    };

    [[nodiscard]] SharedFence createSharedFence(const rachis::Device& vulkan)
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::d3d12,
            .adapterPreference = rachis::AdapterPreference::highPerformance,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
        });
        REQUIRE(created.has_value());
        const auto vulkanAdapter = rachis::test::requireAdapterInfo(vulkan);
        const auto d3d12Adapter = rachis::test::requireAdapterInfo(*created);
        if (d3d12Adapter.vendorId != vulkanAdapter.vendorId || d3d12Adapter.deviceId != vulkanAdapter.deviceId)
        {
            SKIP("The Direct3D 12 fence device and the Vulkan devices use different adapters.");
        }

        SharedFence fence{.device = std::move(*created)};
        auto command = rachis::test::acquireCommandBuffer(fence.device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        auto first = fence.device.submit(std::move(*command));
        REQUIRE(first.has_value());
        REQUIRE(fence.device.wait(*first).has_value());
        auto shared = fence.device.exportTimelinePoint(*first);
        REQUIRE(shared.has_value());
        fence.shared = std::move(*shared);
        fence.signaledValue = first->value();
        return fence;
    }

    [[nodiscard]] rachis::TimelinePointStatus requireStatus(const rachis::Result<rachis::TimelinePointStatus>& status)
    {
        REQUIRE(status.has_value());
        return *status;
    }

    [[nodiscard]] rachis::Result<rachis::Device> createDeviceWithQueueTimelinesAt(std::uint64_t value)
    {
        rachis::vulkan::test::initialQueueTimelineValue().store(value, std::memory_order_release);
        auto created = rachis::createDevice(vulkanDesc());
        rachis::vulkan::test::initialQueueTimelineValue().store(0, std::memory_order_release);
        return created;
    }

} // namespace

TEST_CASE("Vulkan exported timeline points complete with the point and not after later work on its queue",
          "[size:medium][requires:HardwareD3D12][rhi][vulkan][interop][timeline]")
{
    using namespace std::chrono_literals;
    using rachis::TimelinePointStatus;

    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto consumer = std::move(*consumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedTimelineHandles,
                                            "sharedTimelineHandles"))
    {
        return;
    }
    auto fence = createSharedFence(producer);

    constexpr std::array queueTypes{
        rachis::QueueType::graphics,
        rachis::QueueType::compute,
        rachis::QueueType::copy,
    };
    for (const auto queueType : queueTypes)
    {
        for (const bool laterWork : {false, true})
        {
            CAPTURE(static_cast<int>(queueType), laterWork);
            const auto target = fence.signaledValue + 1;

            auto command = rachis::test::acquireCommandBuffer(producer, queueType);
            REQUIRE(command.has_value());
            auto point = producer.submit(std::move(*command));
            REQUIRE(point.has_value());
            if (!laterWork)
            {
                REQUIRE(producer.wait(*point).has_value());
                CHECK(requireStatus(producer.queryTimelinePoint(*point)) == TimelinePointStatus::complete);
            }

            auto gate = producer.enqueueSharedTimelineWait(queueType, fence.at(target));
            REQUIRE(gate.has_value());
            std::optional<rachis::TimelinePoint> later;
            if (laterWork)
            {
                auto laterCommand = rachis::test::acquireCommandBuffer(producer, queueType);
                REQUIRE(laterCommand.has_value());
                auto laterPoint = producer.submit(std::move(*laterCommand));
                REQUIRE(laterPoint.has_value());
                later = *laterPoint;
            }
            const auto pendingPoint = later.value_or(*gate);

            auto exported = producer.exportTimelinePoint(*point);
            REQUIRE(exported.has_value());
            CHECK(exported->value == point->value());
            auto exportedPending = producer.exportTimelinePoint(pendingPoint);
            REQUIRE(exportedPending.has_value());
            CHECK(exportedPending->value == pendingPoint.value());
            auto vulkanConsumed = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, exported->view());
            REQUIRE(vulkanConsumed.has_value());
            auto d3d12Consumed = fence.device.enqueueSharedTimelineWait(rachis::QueueType::compute, exported->view());
            REQUIRE(d3d12Consumed.has_value());
            auto d3d12Pending =
                fence.device.enqueueSharedTimelineWait(rachis::QueueType::compute, exportedPending->view());
            REQUIRE(d3d12Pending.has_value());
            exported->handle.reset();
            exportedPending->handle.reset();

            const auto vulkanStatus = requireStatus(consumer.wait(*vulkanConsumed, 2s));
            const auto d3d12Status = requireStatus(
                fence.device.wait(*d3d12Consumed, vulkanStatus == TimelinePointStatus::complete ? 2s : 0s));
            const auto d3d12PendingStatus = requireStatus(fence.device.wait(*d3d12Pending, 50ms));
            const auto gateStatus = requireStatus(producer.queryTimelinePoint(*gate));
            fence.signal(target);
            CHECK(vulkanStatus == TimelinePointStatus::complete);
            CHECK(d3d12Status == TimelinePointStatus::complete);
            CHECK(d3d12PendingStatus == TimelinePointStatus::pending);
            CHECK(gateStatus == TimelinePointStatus::pending);

            CHECK(requireStatus(producer.wait(*gate, 5s)) == TimelinePointStatus::complete);
            if (later)
            {
                CHECK(requireStatus(producer.wait(*later, 5s)) == TimelinePointStatus::complete);
            }
            CHECK(requireStatus(consumer.wait(*vulkanConsumed, 5s)) == TimelinePointStatus::complete);
            CHECK(requireStatus(fence.device.wait(*d3d12Consumed, 5s)) == TimelinePointStatus::complete);
            CHECK(requireStatus(fence.device.wait(*d3d12Pending, 5s)) == TimelinePointStatus::complete);
        }
    }

    REQUIRE(producer.waitIdle().has_value());
    REQUIRE(consumer.waitIdle().has_value());
    REQUIRE(fence.device.waitIdle().has_value());
    producer.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(consumer);
    const auto fenceDiagnostics = fence.device.diagnosticCounts();
    CHECK(fenceDiagnostics.errors == 0);
    CHECK(fenceDiagnostics.corruptions == 0);
}

TEST_CASE("Vulkan exported timeline points from one queue complete independently when waited back to back",
          "[size:medium][requires:HardwareD3D12][rhi][vulkan][interop][timeline]")
{
    using namespace std::chrono_literals;
    using rachis::TimelinePointStatus;

    auto producerCreated = rachis::createDevice(vulkanDesc());
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto consumer = std::move(*consumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedTimelineHandles,
                                            "sharedTimelineHandles"))
    {
        return;
    }
    auto fence = createSharedFence(producer);

    constexpr std::array queueTypes{
        rachis::QueueType::graphics,
        rachis::QueueType::compute,
        rachis::QueueType::copy,
    };
    for (const auto queueType : queueTypes)
    {
        CAPTURE(static_cast<int>(queueType));
        const auto target = fence.signaledValue + 1;

        auto first = producer.enqueueSharedTimelineWait(queueType, fence.at(target));
        REQUIRE(first.has_value());
        auto second = producer.enqueueSharedTimelineWait(queueType, fence.at(target + 1));
        REQUIRE(second.has_value());
        auto third = producer.enqueueSharedTimelineWait(queueType, fence.at(target + 2));
        REQUIRE(third.has_value());
        auto exportedFirst = producer.exportTimelinePoint(*first);
        REQUIRE(exportedFirst.has_value());
        auto exportedSecond = producer.exportTimelinePoint(*second);
        REQUIRE(exportedSecond.has_value());
        auto exportedThird = producer.exportTimelinePoint(*third);
        REQUIRE(exportedThird.has_value());
        CHECK(exportedFirst->value == first->value());
        CHECK(exportedSecond->value == second->value());
        CHECK(exportedThird->value == third->value());

        auto d3d12First = fence.device.enqueueSharedTimelineWait(rachis::QueueType::compute, exportedFirst->view());
        REQUIRE(d3d12First.has_value());
        auto d3d12Second = fence.device.enqueueSharedTimelineWait(rachis::QueueType::compute, exportedSecond->view());
        REQUIRE(d3d12Second.has_value());
        auto vulkanSecond = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, exportedSecond->view());
        REQUIRE(vulkanSecond.has_value());

        fence.signal(target);
        const auto firstStatus = requireStatus(producer.wait(*first, 5s));
        const auto d3d12FirstStatus = requireStatus(fence.device.wait(*d3d12First, 5s));
        const auto d3d12SecondStatus = requireStatus(fence.device.wait(*d3d12Second, 50ms));
        const auto vulkanSecondStatus = requireStatus(consumer.queryTimelinePoint(*vulkanSecond));

        auto d3d12FirstAgain = fence.device.enqueueSharedTimelineWait(rachis::QueueType::copy, exportedFirst->view());
        REQUIRE(d3d12FirstAgain.has_value());
        auto d3d12Third = fence.device.enqueueSharedTimelineWait(rachis::QueueType::copy, exportedThird->view());
        REQUIRE(d3d12Third.has_value());
        const auto d3d12FirstAgainStatus = requireStatus(fence.device.wait(*d3d12FirstAgain, 5s));
        const auto d3d12ThirdStatus = requireStatus(fence.device.wait(*d3d12Third, 50ms));
        const auto secondStatus = requireStatus(producer.queryTimelinePoint(*second));
        fence.signal(target + 2);
        CHECK(firstStatus == TimelinePointStatus::complete);
        CHECK(d3d12FirstStatus == TimelinePointStatus::complete);
        CHECK(d3d12SecondStatus == TimelinePointStatus::pending);
        CHECK(vulkanSecondStatus == TimelinePointStatus::pending);
        CHECK(d3d12FirstAgainStatus == TimelinePointStatus::complete);
        CHECK(d3d12ThirdStatus == TimelinePointStatus::pending);
        CHECK(secondStatus == TimelinePointStatus::pending);

        CHECK(requireStatus(producer.wait(*third, 5s)) == TimelinePointStatus::complete);
        CHECK(requireStatus(consumer.wait(*vulkanSecond, 5s)) == TimelinePointStatus::complete);
        CHECK(requireStatus(fence.device.wait(*d3d12Second, 5s)) == TimelinePointStatus::complete);
        CHECK(requireStatus(fence.device.wait(*d3d12Third, 5s)) == TimelinePointStatus::complete);
    }

    REQUIRE(producer.waitIdle().has_value());
    REQUIRE(consumer.waitIdle().has_value());
    REQUIRE(fence.device.waitIdle().has_value());
    producer.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(consumer);
    const auto fenceDiagnostics = fence.device.diagnosticCounts();
    CHECK(fenceDiagnostics.errors == 0);
    CHECK(fenceDiagnostics.corruptions == 0);
}

TEST_CASE("Vulkan exported timeline points stay within the value difference limit on a timeline past 32 bits",
          "[size:medium][requires:HardwareD3D12][rhi][vulkan][interop][timeline]")
{
    using namespace std::chrono_literals;
    using rachis::TimelinePointStatus;

    constexpr std::uint64_t seededValue = (std::uint64_t{1} << 32U) + 5U;
    auto producerCreated = createDeviceWithQueueTimelinesAt(seededValue);
    auto consumerCreated = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(producerCreated) ||
        rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto producer = std::move(*producerCreated);
    auto consumer = std::move(*consumerCreated);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(producer).sharedTimelineHandles,
                                            "sharedTimelineHandles"))
    {
        return;
    }
    auto fence = createSharedFence(producer);

    constexpr std::array queueTypes{
        rachis::QueueType::graphics,
        rachis::QueueType::compute,
        rachis::QueueType::copy,
    };
    for (const auto queueType : queueTypes)
    {
        CAPTURE(static_cast<int>(queueType));
        const auto target = fence.signaledValue + 1;

        auto command = rachis::test::acquireCommandBuffer(producer, queueType);
        REQUIRE(command.has_value());
        auto complete = producer.submit(std::move(*command));
        REQUIRE(complete.has_value());
        CHECK(complete->value() > seededValue);
        REQUIRE(requireStatus(producer.wait(*complete, 5s)) == TimelinePointStatus::complete);

        auto gate = producer.enqueueSharedTimelineWait(queueType, fence.at(target));
        REQUIRE(gate.has_value());
        auto laterCommand = rachis::test::acquireCommandBuffer(producer, queueType);
        REQUIRE(laterCommand.has_value());
        auto pending = producer.submit(std::move(*laterCommand));
        REQUIRE(pending.has_value());

        auto exportedComplete = producer.exportTimelinePoint(*complete);
        REQUIRE(exportedComplete.has_value());
        CHECK(exportedComplete->value == complete->value());
        auto exportedPending = producer.exportTimelinePoint(*pending);
        REQUIRE(exportedPending.has_value());
        CHECK(exportedPending->value == pending->value());

        auto vulkanComplete = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, exportedComplete->view());
        REQUIRE(vulkanComplete.has_value());
        auto vulkanPending = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, exportedPending->view());
        REQUIRE(vulkanPending.has_value());
        auto d3d12Complete =
            fence.device.enqueueSharedTimelineWait(rachis::QueueType::compute, exportedComplete->view());
        REQUIRE(d3d12Complete.has_value());
        auto d3d12Pending = fence.device.enqueueSharedTimelineWait(rachis::QueueType::compute, exportedPending->view());
        REQUIRE(d3d12Pending.has_value());

        const auto vulkanCompleteStatus = requireStatus(consumer.wait(*vulkanComplete, 2s));
        const auto d3d12CompleteStatus = requireStatus(fence.device.wait(*d3d12Complete, 2s));
        const auto vulkanPendingStatus = requireStatus(consumer.queryTimelinePoint(*vulkanPending));
        const auto d3d12PendingStatus = requireStatus(fence.device.wait(*d3d12Pending, 50ms));
        const auto gateStatus = requireStatus(producer.queryTimelinePoint(*gate));
        fence.signal(target);
        CHECK(vulkanCompleteStatus == TimelinePointStatus::complete);
        CHECK(d3d12CompleteStatus == TimelinePointStatus::complete);
        CHECK(vulkanPendingStatus == TimelinePointStatus::pending);
        CHECK(d3d12PendingStatus == TimelinePointStatus::pending);
        CHECK(gateStatus == TimelinePointStatus::pending);

        CHECK(requireStatus(producer.wait(*pending, 5s)) == TimelinePointStatus::complete);
        CHECK(requireStatus(consumer.wait(*vulkanPending, 5s)) == TimelinePointStatus::complete);
        CHECK(requireStatus(fence.device.wait(*d3d12Pending, 5s)) == TimelinePointStatus::complete);
    }

    REQUIRE(producer.waitIdle().has_value());
    REQUIRE(consumer.waitIdle().has_value());
    REQUIRE(fence.device.waitIdle().has_value());
    producer.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(consumer);
    const auto fenceDiagnostics = fence.device.diagnosticCounts();
    CHECK(fenceDiagnostics.errors == 0);
    CHECK(fenceDiagnostics.corruptions == 0);
}
#endif
