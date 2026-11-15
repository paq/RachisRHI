#include "Rachis/Device.h"
#include "RhiTestSupport.h"
#if RACHIS_ENABLE_METAL
#include "Metal/MetalTestHooks.h"
#endif

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <string_view>
#include <thread>

namespace
{

#if RACHIS_ENABLE_METAL
    constexpr std::uint64_t bufferSize = 256;

    void captureErrors(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        if (message.severity == rachis::DiagnosticSeverity::error ||
            message.severity == rachis::DiagnosticSeverity::corruption)
        {
            static_cast<std::atomic_uint64_t*>(userData)->fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] rachis::DeviceDesc metalDesc(std::atomic_uint64_t* errors = nullptr) noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
            .diagnosticCallback = errors != nullptr ? &captureErrors : nullptr,
            .diagnosticUserData = errors,
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
        return static_cast<std::byte>((index * 29U + 7U) & 0xFFU);
    }
#endif

} // namespace

#if RACHIS_ENABLE_METAL

TEST_CASE("Metal rejected submissions keep retained GPU identity until a later commit",
          "[size:large][rhi][metal][lifecycle][reject][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal lifecycle reject source",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal lifecycle reject destination",
    });
    REQUIRE(source);
    REQUIRE(destination);

    std::array<std::byte, bufferSize> payload{};
    for (std::uint64_t index = 0; index < payload.size(); ++index)
    {
        payload[index] = pattern(index);
    }
    REQUIRE(device.writeBuffer(*source, 0, payload).has_value());

    auto graphics = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    auto copy = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(graphics);
    REQUIRE(copy);
    REQUIRE(device.copyBuffer(*copy, {.source = *source, .destination = *destination, .size = bufferSize}).has_value());
    REQUIRE(device.destroyBuffer(*source).has_value());

    const auto retained = device.captureResourceMemorySnapshot();
    REQUIRE(retained);
    const auto* retainedSource = findByLabel(*retained, "Metal lifecycle reject source");
    REQUIRE(retainedSource != nullptr);
    CHECK_FALSE(retainedSource->logicalAlive);
    const auto allocationId = retainedSource->allocationId;

    std::array mixed{std::move(*graphics), std::move(*copy)};
    auto rejected = device.submit(mixed);
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(mixed[0].isValid());
    CHECK(mixed[1].isValid());

    const auto afterReject = device.captureResourceMemorySnapshot();
    REQUIRE(afterReject);
    const auto* stillRetained = findByLabel(*afterReject, "Metal lifecycle reject source");
    REQUIRE(stillRetained != nullptr);
    CHECK(stillRetained->allocationId == allocationId);
    CHECK_FALSE(stillRetained->logicalAlive);

    mixed[0] = rachis::CommandBuffer{};
    auto submitted = device.submit(std::move(mixed[1]));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*destination, 0, actual).has_value());
    CHECK(actual == payload);
    REQUIRE(device.destroyBuffer(*destination).has_value());
    device.collectGarbage();
    const auto released = device.captureResourceMemorySnapshot();
    REQUIRE(released);
    CHECK(findByLabel(*released, "Metal lifecycle reject source") == nullptr);
}

TEST_CASE("Metal command reset drops recording retainers and closes the copy encoder",
          "[size:large][rhi][metal][lifecycle][reset][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal lifecycle reset source",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal lifecycle reset destination",
    });
    REQUIRE(source);
    REQUIRE(destination);

    auto command = rachis::test::acquireCommandBuffer(
        device, rachis::QueueType::copy, rachis::CommandPoolFlag::allowIndividualReset);
    REQUIRE(command);
    REQUIRE(
        device.copyBuffer(*command, {.source = *source, .destination = *destination, .size = bufferSize}).has_value());
    REQUIRE(device.destroyBuffer(*source).has_value());
    const auto retained = device.captureResourceMemorySnapshot();
    REQUIRE(retained);
    REQUIRE(findByLabel(*retained, "Metal lifecycle reset source") != nullptr);

    REQUIRE(device.resetCommandBuffer(*command).has_value());
    device.collectGarbage();
    const auto afterReset = device.captureResourceMemorySnapshot();
    REQUIRE(afterReset);
    CHECK(findByLabel(*afterReset, "Metal lifecycle reset source") == nullptr);

    auto replacement = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal lifecycle reset replacement",
    });
    REQUIRE(replacement);
    std::array<std::byte, bufferSize> payload{};
    payload.fill(std::byte{0x3C});
    REQUIRE(device.writeBuffer(*replacement, 0, payload).has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *replacement, .destination = *destination, .size = bufferSize})
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*destination, 0, actual).has_value());
    CHECK(actual == payload);
    REQUIRE(device.destroyBuffer(*replacement).has_value());
    REQUIRE(device.destroyBuffer(*destination).has_value());
}

TEST_CASE("Metal unretained recordings do not keep GPU identity after logical destroy",
          "[size:large][rhi][metal][lifecycle][unretained]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal lifecycle unretained source",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal lifecycle unretained destination",
    });
    REQUIRE(source);
    REQUIRE(destination);

    auto command = rachis::test::acquireCommandBuffer(
        device, rachis::QueueType::copy, rachis::CommandPoolFlag::unretainedResources);
    REQUIRE(command);
    REQUIRE(
        device.copyBuffer(*command, {.source = *source, .destination = *destination, .size = bufferSize}).has_value());
    REQUIRE(device.destroyBuffer(*source).has_value());
    device.collectGarbage();

    const auto snapshot = device.captureResourceMemorySnapshot();
    REQUIRE(snapshot);
    CHECK(findByLabel(*snapshot, "Metal lifecycle unretained source") == nullptr);
    REQUIRE(findByLabel(*snapshot, "Metal lifecycle unretained destination") != nullptr);

    command = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyBuffer(*destination).has_value());
}

TEST_CASE("Metal copy encoder stays open across a producer-consumer barrier on one command",
          "[size:large][rhi][metal][lifecycle][encoder][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal lifecycle encoder source",
    });
    auto temp = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Metal lifecycle encoder temp",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal lifecycle encoder destination",
    });
    REQUIRE(source);
    REQUIRE(temp);
    REQUIRE(destination);

    std::array<std::byte, bufferSize> payload{};
    for (std::uint64_t index = 0; index < payload.size(); ++index)
    {
        payload[index] = pattern(index + 3);
    }
    REQUIRE(device.writeBuffer(*source, 0, payload).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    REQUIRE(device.copyBuffer(*command, {.source = *source, .destination = *temp, .size = bufferSize}).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *temp,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(
        device.copyBuffer(*command, {.source = *temp, .destination = *destination, .size = bufferSize}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*destination, 0, actual).has_value());
    CHECK(actual == payload);
    REQUIRE(device.destroyBuffer(*source).has_value());
    REQUIRE(device.destroyBuffer(*temp).has_value());
    REQUIRE(device.destroyBuffer(*destination).has_value());
}

TEST_CASE("Metal device close drains retained work without invoking a freed diagnostic callback",
          "[size:large][rhi][metal][lifecycle][callback][close]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(metalDesc(&errors));
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal lifecycle close source",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal lifecycle close destination",
    });
    REQUIRE(source);
    REQUIRE(destination);
    std::array<std::byte, bufferSize> payload{};
    payload.fill(std::byte{0xA1});
    REQUIRE(device.writeBuffer(*source, 0, payload).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    REQUIRE(
        device.copyBuffer(*command, {.source = *source, .destination = *destination, .size = bufferSize}).has_value());
    REQUIRE(device.destroyBuffer(*source).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    device = rachis::Device{};
    CHECK(errors.load() == 0);
}

TEST_CASE("Metal unsubmitted command buffers outliving Device detach retained resources and open scopes cleanly",
          "[size:large][rhi][metal][lifecycle][command-pool]")
{
    std::atomic_uint64_t errors{0};
    rachis::CommandBuffer liveGraphicsCommand{};
    rachis::CommandBuffer destroyedPoolCopyCommand{};
    {
        auto created = rachis::createDevice(metalDesc(&errors));
        REQUIRE(created.has_value());
        auto device = std::move(*created);

        auto graphicsPool = device.createCommandPool({
            .queueType = rachis::QueueType::graphics,
            .maxCommandBuffers = 1,
            .label = "Live graphics pool",
        });
        auto copyPool = device.createCommandPool({
            .queueType = rachis::QueueType::copy,
            .maxCommandBuffers = 1,
            .label = "Destroyed copy pool",
        });
        REQUIRE(graphicsPool);
        REQUIRE(copyPool);

        auto acquiredGraphics = device.acquireCommandBuffer(*graphicsPool);
        auto acquiredCopy = device.acquireCommandBuffer(*copyPool);
        REQUIRE(acquiredGraphics);
        REQUIRE(acquiredCopy);
        liveGraphicsCommand = std::move(*acquiredGraphics);
        destroyedPoolCopyCommand = std::move(*acquiredCopy);

        auto upload = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Post-device upload",
        });
        auto destination = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Post-device destination",
        });
        auto target = device.createTexture({
            .width = 16,
            .height = 16,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "Post-device render target",
        });
        REQUIRE(upload);
        REQUIRE(destination);
        REQUIRE(target);

        auto targetView = device.createTextureView({
            .texture = *target,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension = rachis::TextureViewDimension::texture2D,
            .format = rachis::Format::rgba8Unorm,
        });
        REQUIRE(targetView);

        REQUIRE(device
                    .copyBuffer(destroyedPoolCopyCommand,
                                {.source = *upload, .destination = *destination, .size = bufferSize})
                    .has_value());
        REQUIRE(device.beginDebugGroup(destroyedPoolCopyCommand, {.label = "Abandoned copy group"}).has_value());
        REQUIRE(device.destroyCommandPool(*copyPool).has_value());

        REQUIRE(device.beginDebugGroup(liveGraphicsCommand, {.label = "Abandoned render group"}).has_value());
        REQUIRE(device
                    .textureBarrier(liveGraphicsCommand,
                                    {
                                        .texture = *target,
                                        .stagesBefore = rachis::PipelineStage::none,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *targetView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.2F, 0.4F, 0.6F, 1.0F},
            },
        };
        REQUIRE(device.beginRendering(liveGraphicsCommand, {.colorAttachments = attachments}).has_value());

        REQUIRE(device.destroyTextureView(*targetView).has_value());
        REQUIRE(device.destroyTexture(*target).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        REQUIRE(device.destroyBuffer(*destination).has_value());
    }

    CHECK(liveGraphicsCommand.isValid());
    CHECK(liveGraphicsCommand.queueType() == rachis::QueueType::graphics);
    CHECK(destroyedPoolCopyCommand.isValid());
    CHECK(destroyedPoolCopyCommand.queueType() == rachis::QueueType::copy);

    liveGraphicsCommand = {};
    destroyedPoolCopyCommand = {};
    CHECK(errors.load() == 0);
}

TEST_CASE("Metal teardown preserves in-flight commit retentions after an earlier commit feedback failure",
          "[size:large][rhi][metal][lifecycle][gpu][regression]")
{
    auto firstFeedbackGate = std::make_shared<rachis::metal::MetalCommitFeedbackGate>();
    auto secondFeedbackGate = std::make_shared<rachis::metal::MetalCommitFeedbackGate>();
    struct ScopedGateRelease final
    {
        std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> first;
        std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> second;
        ~ScopedGateRelease() noexcept
        {
            if (first != nullptr)
            {
                first->release();
            }
            if (second != nullptr)
            {
                second->release();
            }
        }
    } const releaseOnExit{firstFeedbackGate, secondFeedbackGate};

    std::weak_ptr<void> inFlightDestinationObserver;
    rachis::CommandBuffer unsubmittedSiblingCommand;

    {
        auto created = rachis::createDevice(metalDesc());
        REQUIRE(created.has_value());
        auto device = std::move(*created);

        struct UnwindGateRelease final
        {
            int initialUncaught = std::uncaught_exceptions();
            std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> first;
            std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> second;
            ~UnwindGateRelease() noexcept
            {
                if (std::uncaught_exceptions() > initialUncaught)
                {
                    if (first != nullptr)
                    {
                        first->release();
                    }
                    if (second != nullptr)
                    {
                        second->release();
                    }
                }
            }
        } const releaseOnUnwind{.first = firstFeedbackGate, .second = secondFeedbackGate};

        auto pool = device.createCommandPool({
            .queueType = rachis::QueueType::graphics,
            .maxCommandBuffers = 3,
            .label = "Post-failure teardown pool",
        });
        REQUIRE(pool);

        auto firstCmd = device.acquireCommandBuffer(*pool);
        auto secondCmd = device.acquireCommandBuffer(*pool);
        auto thirdCmd = device.acquireCommandBuffer(*pool);
        REQUIRE(firstCmd);
        REQUIRE(secondCmd);
        REQUIRE(thirdCmd);
        unsubmittedSiblingCommand = std::move(*thirdCmd);

        auto source = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Post-failure source",
        });
        auto destination = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Post-failure destination",
        });
        REQUIRE(source);
        REQUIRE(destination);
        inFlightDestinationObserver = rachis::metal::boundBufferOwnerObserver(*destination);
        REQUIRE_FALSE(inFlightDestinationObserver.expired());

        REQUIRE(device.insertDebugMarker(*firstCmd, {.label = "Failing first commit"}).has_value());
        rachis::metal::holdNextCommitFeedback(firstFeedbackGate);
        rachis::metal::failNextCommitFeedback();
        auto firstSubmitted = device.submit(std::move(*firstCmd));
        REQUIRE(firstSubmitted);

        REQUIRE(device.copyBuffer(*secondCmd, {.source = *source, .destination = *destination, .size = bufferSize})
                    .has_value());
        rachis::metal::holdNextCommitFeedback(secondFeedbackGate);
        auto secondSubmitted = device.submit(std::move(*secondCmd));
        REQUIRE(secondSubmitted);

        REQUIRE(device.destroyBuffer(*source).has_value());
        REQUIRE(device.destroyBuffer(*destination).has_value());
        REQUIRE_FALSE(inFlightDestinationObserver.expired());

        REQUIRE(firstFeedbackGate->waitForEntry(std::chrono::seconds(2)));
        firstFeedbackGate->release();

        auto waitedFirst = device.wait(*firstSubmitted, std::chrono::seconds(2));
        REQUIRE_FALSE(waitedFirst.has_value());
        CHECK(waitedFirst.error().code == rachis::ErrorCode::deviceLost);
        CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);
    }

    CHECK_FALSE(inFlightDestinationObserver.expired());
    CHECK(unsubmittedSiblingCommand.isValid());
    unsubmittedSiblingCommand = {};

    secondFeedbackGate->release();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!inFlightDestinationObserver.expired() && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(inFlightDestinationObserver.expired());
}

#endif
