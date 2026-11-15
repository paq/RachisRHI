#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifndef RACHIS_TESTING
#error Metal device-loss injection must stay test-only.
#endif

namespace
{

#if RACHIS_ENABLE_METAL
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

    [[nodiscard]] std::vector<std::byte> loadBytes(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        return stream ? bytes : std::vector<std::byte>{};
    }
#endif

} // namespace

#if RACHIS_ENABLE_METAL

TEST_CASE("injected Metal device loss is terminal and repeats the original error",
          "[size:large][rhi][metal][device-lost][loss]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);
    CHECK_FALSE(device.deviceLossError().has_value());

    auto healthy = device.captureDeviceFaultReport();
    REQUIRE(healthy);
    CHECK(healthy->status == rachis::DeviceFaultStatus::healthy);
    CHECK(device.diagnosticCounts().errors == 0);

    auto rejected = device.createBuffer({});
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);

    auto injected = rachis::metal::injectMetalDeviceLoss();
    REQUIRE_FALSE(injected);
    CHECK(injected.error().code == rachis::ErrorCode::deviceLost);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);

    const auto first = device.deviceLossError();
    REQUIRE(first.has_value());
    CHECK(first->code == rachis::ErrorCode::deviceLost);
    CHECK(first->message == injected.error().message);

    auto again = rachis::metal::injectMetalDeviceLoss(rachis::DeviceFaultStatus::hung);
    REQUIRE_FALSE(again);
    CHECK(again.error().message == first->message);

    auto createdAfter = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copyDestination,
        .label = "lost-device buffer",
    });
    REQUIRE_FALSE(createdAfter);
    CHECK(createdAfter.error().message == first->message);

    auto idle = device.waitIdle();
    REQUIRE_FALSE(idle);
    CHECK(idle.error().message == first->message);

    auto report = device.captureDeviceFaultReport();
    REQUIRE(report);
    CHECK(report->status == rachis::DeviceFaultStatus::removed);
    CHECK(report->nativeReason != 0);
    REQUIRE_FALSE(report->breadcrumbs.empty());

    auto repeated = device.captureDeviceFaultReport();
    REQUIRE(repeated);
    CHECK(repeated->status == report->status);
    CHECK(repeated->nativeReason == report->nativeReason);
}

TEST_CASE("injected Metal device loss rejects host buffer writes, reads and maps before handle checks",
          "[size:large][rhi][metal][device-lost][loss]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto upload = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "lost-device upload",
    });
    REQUIRE(upload);
    auto readback = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "lost-device readback",
    });
    REQUIRE(readback);
    auto stale = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "lost-device stale upload",
    });
    REQUIRE(stale);
    REQUIRE(device.destroyBuffer(*stale));
    std::array<std::byte, 16> bytes{};
    REQUIRE(device.writeBuffer(*upload, 0, bytes));
    REQUIRE(device.readBuffer(*readback, 0, bytes));

    auto injected = rachis::metal::injectMetalDeviceLoss();
    REQUIRE_FALSE(injected);
    const auto first = device.deviceLossError();
    REQUIRE(first.has_value());

    auto written = device.writeBuffer(*upload, 0, bytes);
    REQUIRE_FALSE(written);
    CHECK(written.error().code == rachis::ErrorCode::deviceLost);
    CHECK(written.error().message == first->message);
    auto read = device.readBuffer(*readback, 0, bytes);
    REQUIRE_FALSE(read);
    CHECK(read.error().code == rachis::ErrorCode::deviceLost);
    CHECK(read.error().message == first->message);
    auto mapped = device.mapBuffer(*upload);
    REQUIRE_FALSE(mapped);
    CHECK(mapped.error().code == rachis::ErrorCode::deviceLost);
    CHECK(mapped.error().message == first->message);

    auto staleWrite = device.writeBuffer(*stale, 0, bytes);
    REQUIRE_FALSE(staleWrite);
    CHECK(staleWrite.error().code == rachis::ErrorCode::deviceLost);
    auto staleRead = device.readBuffer(*stale, 0, bytes);
    REQUIRE_FALSE(staleRead);
    CHECK(staleRead.error().code == rachis::ErrorCode::deviceLost);
    auto staleMap = device.mapBuffer(*stale);
    REQUIRE_FALSE(staleMap);
    CHECK(staleMap.error().code == rachis::ErrorCode::deviceLost);
}

TEST_CASE("injected Metal loss rejects texture view and sampler creation",
          "[size:large][rhi][metal][device-lost][loss][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto texture = device.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "lost-device view source",
    });
    REQUIRE(texture);
    const auto baseline = device.captureBackendAllocationSnapshot();

    auto injected = rachis::metal::injectMetalDeviceLoss();
    REQUIRE_FALSE(injected);
    const auto loss = device.deviceLossError();
    REQUIRE(loss.has_value());

    auto view = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "lost-device view",
    });
    REQUIRE_FALSE(view);
    CHECK(view.error().code == rachis::ErrorCode::deviceLost);
    CHECK(view.error().message == loss->message);

    auto sampler = device.createSampler({.label = "lost-device sampler"});
    REQUIRE_FALSE(sampler);
    CHECK(sampler.error().code == rachis::ErrorCode::deviceLost);
    CHECK(sampler.error().message == loss->message);
    CHECK(device.captureBackendAllocationSnapshot().liveDescriptorCount == baseline.liveDescriptorCount);
}

TEST_CASE("injected Metal loss rejects public waits and drains the healthy GPU at teardown",
          "[size:large][rhi][metal][device-lost][loss][teardown]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(metalDesc(&errors));
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);

    auto injected = rachis::metal::injectMetalDeviceLoss();
    REQUIRE_FALSE(injected);
    auto waited = device.wait(*submitted);
    REQUIRE_FALSE(waited);
    CHECK(waited.error().code == rachis::ErrorCode::deviceLost);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);

    auto afterSubmit = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE_FALSE(afterSubmit);
    CHECK(afterSubmit.error().code == rachis::ErrorCode::deviceLost);
    rachis::metal::resetTeardownQueueDrainCount();
    device = {};
    CHECK(rachis::metal::teardownQueueDrainCount() == 1);
    CHECK(errors.load() == 1);
}

TEST_CASE("Metal commit feedback makes GPU execution errors terminal",
          "[size:large][rhi][metal][device-lost][gpu][regression]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(metalDesc(&errors));
    REQUIRE(created);
    auto device = std::move(*created);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    rachis::metal::failNextCommitFeedback();
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);

    auto firstWait = device.wait(*submitted, std::chrono::seconds(5));
    REQUIRE_FALSE(firstWait);
    CHECK(firstWait.error().code == rachis::ErrorCode::deviceLost);
    REQUIRE(device.operationalState() == rachis::DeviceOperationalState::lost);
    const auto first = device.deviceLossError();
    REQUIRE(first);
    CHECK(first->nativeCode != 0);

    const auto feedbackDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (device.completedValue(rachis::QueueType::copy) < submitted->value() &&
           std::chrono::steady_clock::now() < feedbackDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(device.completedValue(rachis::QueueType::copy) >= submitted->value());
    CHECK(errors.load() == 1);
    CHECK(device.diagnosticCounts().errors == 1);

    auto queried = device.queryTimelinePoint(*submitted);
    REQUIRE_FALSE(queried);
    CHECK(queried.error().code == rachis::ErrorCode::deviceLost);
    auto waited = device.wait(*submitted, std::chrono::seconds(30));
    REQUIRE_FALSE(waited);
    CHECK(waited.error().nativeCode == first->nativeCode);
    auto idle = device.waitIdle();
    REQUIRE_FALSE(idle);
    CHECK(idle.error().nativeCode == first->nativeCode);
    auto afterFailure = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE_FALSE(afterFailure);
    auto report = device.captureDeviceFaultReport();
    REQUIRE(report);
    CHECK(report->status == rachis::DeviceFaultStatus::driverInternalError);
    CHECK(report->nativeReason == first->nativeCode);
    REQUIRE_FALSE(report->breadcrumbs.empty());

    auto injectedAfterNative = rachis::metal::injectMetalDeviceLoss(rachis::DeviceFaultStatus::hung);
    REQUIRE_FALSE(injectedAfterNative);
    CHECK(injectedAfterNative.error().code == rachis::ErrorCode::deviceLost);
    CHECK(injectedAfterNative.error().nativeCode == first->nativeCode);
    CHECK(injectedAfterNative.error().message == first->message);

    auto reportAfterInject = device.captureDeviceFaultReport();
    REQUIRE(reportAfterInject);
    CHECK(reportAfterInject->status == rachis::DeviceFaultStatus::driverInternalError);
    CHECK(reportAfterInject->nativeReason == first->nativeCode);
    CHECK(errors.load() == 1);
    CHECK(device.diagnosticCounts().errors == 1);

    rachis::metal::resetTeardownQueueDrainCount();
    device = {};
    CHECK(rachis::metal::teardownQueueDrainCount() == 0);
}

TEST_CASE("injected Metal loss followed by in-flight commit feedback failure keeps first loss report and skips "
          "teardown drain",
          "[size:large][rhi][metal][device-lost][gpu][regression]")
{
    auto firstGate = std::make_shared<rachis::metal::MetalCommitFeedbackGate>();
    auto secondGate = std::make_shared<rachis::metal::MetalCommitFeedbackGate>();
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
    } const releaseOnExit{firstGate, secondGate};

    std::atomic_uint64_t errors{0};
    {
        auto created = rachis::createDevice(metalDesc(&errors));
        REQUIRE(created);
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
        } const releaseOnUnwind{.first = firstGate, .second = secondGate};

        auto firstCmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(firstCmd);
        rachis::metal::holdNextCommitFeedback(firstGate);
        rachis::metal::failNextCommitFeedback();
        auto firstSubmitted = device.submit(std::move(*firstCmd));
        REQUIRE(firstSubmitted);

        auto secondCmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(secondCmd);
        rachis::metal::holdNextCommitFeedback(secondGate);
        auto secondSubmitted = device.submit(std::move(*secondCmd));
        REQUIRE(secondSubmitted);

        REQUIRE(firstGate->waitForEntry(std::chrono::seconds(2)));

        auto injected = rachis::metal::injectMetalDeviceLoss(rachis::DeviceFaultStatus::removed);
        REQUIRE_FALSE(injected);
        CHECK(injected.error().code == rachis::ErrorCode::deviceLost);
        CHECK(errors.load() == 1);

        firstGate->release();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (device.completedValue(rachis::QueueType::copy) < firstSubmitted->value() &&
               std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        REQUIRE(device.completedValue(rachis::QueueType::copy) >= firstSubmitted->value());

        const auto loss = device.deviceLossError();
        REQUIRE(loss);
        CHECK(loss->message == injected.error().message);
        const auto report = device.captureDeviceFaultReport();
        REQUIRE(report);
        CHECK(report->status == rachis::DeviceFaultStatus::removed);
        CHECK(errors.load() == 1);
        CHECK(device.diagnosticCounts().errors == 1);

        rachis::metal::resetTeardownQueueDrainCount();
        device = {};
        CHECK(rachis::metal::teardownQueueDrainCount() == 0);
    }
    secondGate->release();
}

TEST_CASE("Metal resetCommandPool on a lost device returns the loss error",
          "[size:large][rhi][metal][device-lost][loss][command-pool][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 2,
        .label = "Lost-device reset pool",
    });
    REQUIRE(pool);
    auto recording = device.acquireCommandBuffer(*pool);
    REQUIRE(recording);
    REQUIRE(device.insertDebugMarker(*recording, {.label = "Recorded before loss"}).has_value());

    auto injected = rachis::metal::injectMetalDeviceLoss();
    REQUIRE_FALSE(injected);
    auto reset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(reset);
    CHECK(reset.error().code == rachis::ErrorCode::deviceLost);
    CHECK(reset.error().message == injected.error().message);
    CHECK(recording->isValid());
}

TEST_CASE("Metal copyBuffer on a lost device returns the loss error",
          "[size:large][rhi][metal][device-lost][loss][transfer][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto upload = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Lost-device copy source",
    });
    auto readback = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Lost-device copy destination",
    });
    REQUIRE(upload);
    REQUIRE(readback);
    auto recording = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(recording);
    REQUIRE(device.copyBuffer(*recording, {.source = *upload, .destination = *readback, .size = 32}).has_value());

    auto injected = rachis::metal::injectMetalDeviceLoss();
    REQUIRE_FALSE(injected);
    auto copied = device.copyBuffer(*recording, {.source = *upload, .destination = *readback, .size = 64});
    REQUIRE_FALSE(copied);
    CHECK(copied.error().code == rachis::ErrorCode::deviceLost);
    CHECK(copied.error().message == injected.error().message);
    CHECK(recording->isValid());
}

TEST_CASE("Metal barriers and alias activations on a lost device return the loss error",
          "[size:large][rhi][metal][device-lost][loss][barrier][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const rachis::BufferDesc placedBufferDesc{
        .size = 256,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Lost-device aliased buffer",
    };
    const auto bufferRequirements = device.queryBufferMemoryRequirements(placedBufferDesc);
    REQUIRE(bufferRequirements);
    const auto bufferHeap = device.createResourceHeap({
        .size = bufferRequirements->size,
        .alignment = bufferRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
        .label = "Lost-device aliased buffer heap",
    });
    REQUIRE(bufferHeap);
    const auto firstBuffer = device.createPlacedBuffer({.heap = *bufferHeap}, placedBufferDesc);
    const auto secondBuffer = device.createPlacedBuffer({.heap = *bufferHeap}, placedBufferDesc);
    REQUIRE(firstBuffer);
    REQUIRE(secondBuffer);

    const rachis::TextureDesc textureDesc{
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Lost-device aliased texture",
    };
    const auto textureRequirements = device.queryTextureMemoryRequirements(textureDesc);
    REQUIRE(textureRequirements);
    const auto textureHeap = device.createResourceHeap({
        .size = textureRequirements->size,
        .alignment = textureRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
        .label = "Lost-device aliased texture heap",
    });
    REQUIRE(textureHeap);
    const auto firstTexture = device.createPlacedTexture({.heap = *textureHeap}, textureDesc);
    const auto secondTexture = device.createPlacedTexture({.heap = *textureHeap}, textureDesc);
    REQUIRE(firstTexture);
    REQUIRE(secondTexture);

    const rachis::BufferAliasActivation bufferActivation{
        .before = *firstBuffer,
        .after = *secondBuffer,
        .stagesBefore = rachis::PipelineStage::all,
        .stagesAfter = rachis::PipelineStage::copy,
        .accessBefore = rachis::Access::common,
        .accessAfter = rachis::Access::copyDestination,
    };
    const rachis::TextureAliasActivation textureActivation{
        .before = *firstTexture,
        .after = *secondTexture,
        .stagesBefore = rachis::PipelineStage::all,
        .stagesAfter = rachis::PipelineStage::copy,
        .accessBefore = rachis::Access::common,
        .accessAfter = rachis::Access::copyDestination,
        .layoutBefore = rachis::TextureLayout::common,
        .layoutAfter = rachis::TextureLayout::copyDestination,
    };
    const rachis::BufferBarrier bufferBarrier{
        .buffer = *secondBuffer,
        .stagesBefore = rachis::PipelineStage::copy,
        .stagesAfter = rachis::PipelineStage::copy,
        .accessBefore = rachis::Access::copyDestination,
        .accessAfter = rachis::Access::copySource,
    };
    const rachis::TextureBarrier textureBarrier{
        .texture = *secondTexture,
        .stagesBefore = rachis::PipelineStage::copy,
        .stagesAfter = rachis::PipelineStage::copy,
        .accessBefore = rachis::Access::copyDestination,
        .accessAfter = rachis::Access::copySource,
        .layoutBefore = rachis::TextureLayout::copyDestination,
        .layoutAfter = rachis::TextureLayout::copySource,
    };
    const std::array bufferBarriers{bufferBarrier};
    const std::array textureBarriers{textureBarrier};
    const rachis::ResourceBarrierBatch batch{.bufferBarriers = bufferBarriers, .textureBarriers = textureBarriers};

    auto recording = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(recording);
    REQUIRE(device.activateBufferAlias(*recording, bufferActivation).has_value());
    REQUIRE(device.activateTextureAlias(*recording, textureActivation).has_value());
    REQUIRE(device.resourceBarriers(*recording, batch).has_value());

    auto injected = rachis::metal::injectMetalDeviceLoss();
    REQUIRE_FALSE(injected);
    auto batched = device.resourceBarriers(*recording, batch);
    REQUIRE_FALSE(batched);
    CHECK(batched.error().code == rachis::ErrorCode::deviceLost);
    CHECK(batched.error().message == injected.error().message);
    auto buffered = device.bufferBarrier(*recording, bufferBarrier);
    REQUIRE_FALSE(buffered);
    CHECK(buffered.error().code == rachis::ErrorCode::deviceLost);
    CHECK(buffered.error().message == injected.error().message);
    auto textured = device.textureBarrier(*recording, textureBarrier);
    REQUIRE_FALSE(textured);
    CHECK(textured.error().code == rachis::ErrorCode::deviceLost);
    CHECK(textured.error().message == injected.error().message);
    auto bufferAliased = device.activateBufferAlias(*recording, bufferActivation);
    REQUIRE_FALSE(bufferAliased);
    CHECK(bufferAliased.error().code == rachis::ErrorCode::deviceLost);
    CHECK(bufferAliased.error().message == injected.error().message);
    auto textureAliased = device.activateTextureAlias(*recording, textureActivation);
    REQUIRE_FALSE(textureAliased);
    CHECK(textureAliased.error().code == rachis::ErrorCode::deviceLost);
    CHECK(textureAliased.error().message == injected.error().message);
    CHECK(recording->isValid());
}

TEST_CASE("Metal teardown after a commit feedback failure waits for later in-flight feedback",
          "[size:large][rhi][metal][device-lost][gpu][teardown][regression]")
{
    constexpr std::uint64_t bufferSize = 256;
    auto firstGate = std::make_shared<rachis::metal::MetalCommitFeedbackGate>();
    auto secondGate = std::make_shared<rachis::metal::MetalCommitFeedbackGate>();
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
    } const releaseOnExit{firstGate, secondGate};

    std::atomic_uint64_t errors{0};
    std::atomic_bool secondReleased{false};
    std::weak_ptr<void> inFlightDestination;
    std::thread releaser;
    {
        auto created = rachis::createDevice(metalDesc(&errors));
        REQUIRE(created);
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
        } const releaseOnUnwind{.first = firstGate, .second = secondGate};

        auto source = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Lost teardown source",
        });
        auto destination = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Lost teardown destination",
        });
        REQUIRE(source);
        REQUIRE(destination);
        inFlightDestination = rachis::metal::boundBufferOwnerObserver(*destination);
        REQUIRE_FALSE(inFlightDestination.expired());

        auto firstCmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(firstCmd);
        rachis::metal::holdNextCommitFeedback(firstGate);
        rachis::metal::failNextCommitFeedback();
        auto firstSubmitted = device.submit(std::move(*firstCmd));
        REQUIRE(firstSubmitted);

        auto secondCmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(secondCmd);
        REQUIRE(device.copyBuffer(*secondCmd, {.source = *source, .destination = *destination, .size = bufferSize})
                    .has_value());
        rachis::metal::holdNextCommitFeedback(secondGate);
        auto secondSubmitted = device.submit(std::move(*secondCmd));
        REQUIRE(secondSubmitted);
        REQUIRE(device.destroyBuffer(*source).has_value());
        REQUIRE(device.destroyBuffer(*destination).has_value());

        REQUIRE(firstGate->waitForEntry(std::chrono::seconds(2)));
        firstGate->release();
        auto waitedFirst = device.wait(*firstSubmitted, std::chrono::seconds(2));
        REQUIRE_FALSE(waitedFirst);
        CHECK(waitedFirst.error().code == rachis::ErrorCode::deviceLost);
        REQUIRE(device.operationalState() == rachis::DeviceOperationalState::lost);
        REQUIRE(secondGate->waitForEntry(std::chrono::seconds(2)));
        CHECK_FALSE(inFlightDestination.expired());

        releaser = std::thread([gate = secondGate, &secondReleased] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            secondReleased.store(true, std::memory_order_release);
            gate->release();
        });
        device = {};
        CHECK(secondReleased.load(std::memory_order_acquire));
        CHECK(inFlightDestination.expired());
    }
    if (releaser.joinable())
    {
        releaser.join();
    }
    CHECK(errors.load() == 1);
}

TEST_CASE("Metal destroy and release calls succeed after device loss and retire their handles",
          "[size:large][rhi][metal][device-lost][loss]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(metalDesc(&errors));
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto bytecode = loadBytes(std::filesystem::path(RACHIS_METALLIB_DIR) / "BootstrapCube.cs.pnms");
    REQUIRE_FALSE(bytecode.empty());

    const rachis::BufferDesc storageDesc{
        .size = 256,
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Metal lost-device storage buffer",
    };
    auto mapped = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal lost-device mapped buffer",
    });
    auto storage = device.createBuffer(storageDesc);
    REQUIRE(mapped);
    REQUIRE(storage);
    REQUIRE(device.mapBuffer(*mapped));
    const auto requirements = device.queryBufferMemoryRequirements(storageDesc);
    REQUIRE(requirements);
    auto heap = device.createResourceHeap({.size = requirements->size * 2, .label = "Metal lost-device heap"});
    REQUIRE(heap);
    auto texture = device.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Metal lost-device texture",
    });
    REQUIRE(texture);
    auto view = device.createTextureView({.texture = *texture});
    auto sampler = device.createSampler({});
    REQUIRE(view);
    REQUIRE(sampler);

    const std::array entries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries});
    REQUIRE(bindingLayout);
    const std::array poolSizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::readOnlyStorageBuffer, .descriptorCount = 1},
        rachis::BindingPoolSize{.type = rachis::BindingType::storageBuffer, .descriptorCount = 1},
    };
    auto bindingPool = device.createBindingPool({.maxBindingSets = 1, .sizes = poolSizes});
    REQUIRE(bindingPool);
    const std::array writes{
        rachis::BindingWrite{
            .binding = 1,
            .resource = rachis::BufferBinding{.buffer = *storage, .size = 256, .structureByteStride = 4},
        },
        rachis::BindingWrite{
            .binding = 2,
            .resource = rachis::BufferBinding{.buffer = *storage, .size = 256, .structureByteStride = 4},
        },
    };
    auto bindingSet = device.createBindingSet({.layout = *bindingLayout, .writes = writes, .pool = *bindingPool});
    REQUIRE(bindingSet);
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(pipelineLayout);

    const std::array libraryEntries{rachis::ShaderLibraryEntryDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "computeMain",
    }};
    auto library = device.createShaderLibrary({.entries = libraryEntries});
    const std::array<rachis::ShaderBindingRequirement, 2> shaderBindings{
        rachis::ShaderBindingRequirement{
            .set = 0,
            .binding = 1,
            .type = rachis::ShaderResourceType::readOnlyStorageBuffer,
            .count = 1,
        },
        rachis::ShaderBindingRequirement{
            .set = 0,
            .binding = 2,
            .type = rachis::ShaderResourceType::storageBuffer,
            .count = 1,
        },
    };
    const std::array shaderPush{rachis::ShaderPushConstantRequirement{.offset = 0, .size = 16}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}, shaderBindings, shaderPush),
    });
    REQUIRE(library);
    REQUIRE(shader);
    auto cache = device.createPipelineCache();
    REQUIRE(cache);
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cache = *cache,
        .cacheKey = {.high = 1, .low = 133},
    });
    auto compilation = device.beginComputePipelineCompilation({
        .layout = *pipelineLayout,
        .computeShader = *shader,
    });
    REQUIRE(pipeline);
    REQUIRE(compilation);

    auto commandPool = device.createCommandPool({.queueType = rachis::QueueType::copy, .maxCommandBuffers = 2});
    auto queryPool = device.createTimestampQueryPool({.queueType = rachis::QueueType::graphics, .queryCount = 2});
    REQUIRE(commandPool);
    REQUIRE(queryPool);

    REQUIRE_FALSE(rachis::metal::injectMetalDeviceLoss());
    REQUIRE(device.operationalState() == rachis::DeviceOperationalState::lost);
    auto rejected = device.createBuffer({.size = 16, .usage = rachis::BufferUsage::copySource});
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().code == rachis::ErrorCode::deviceLost);

    const auto released = [](const char* call, const rachis::Status& status) {
        INFO(call << ": " << (status ? std::string{} : status.error().message));
        CHECK(status.has_value());
    };
    released("destroyPipelineCompilation", device.destroyPipelineCompilation(*compilation));
    released("destroyPipeline", device.destroyPipeline(*pipeline));
    released("destroyPipelineCache", device.destroyPipelineCache(*cache));
    released("destroyShaderModule", device.destroyShaderModule(*shader));
    released("destroyShaderLibrary", device.destroyShaderLibrary(*library));
    released("destroyPipelineLayout", device.destroyPipelineLayout(*pipelineLayout));
    released("destroyBindingSet", device.destroyBindingSet(*bindingSet));
    released("destroyBindingPool", device.destroyBindingPool(*bindingPool));
    released("destroyBindingLayout", device.destroyBindingLayout(*bindingLayout));
    released("destroyTextureView", device.destroyTextureView(*view));
    released("destroyTexture", device.destroyTexture(*texture));
    released("destroySampler", device.destroySampler(*sampler));
    released("destroyResourceHeap", device.destroyResourceHeap(*heap));
    released("unmapBuffer", device.unmapBuffer(*mapped));
    released("destroyBuffer(mapped)", device.destroyBuffer(*mapped));
    released("destroyBuffer(storage)", device.destroyBuffer(*storage));
    released("destroyQueryPool", device.destroyQueryPool(*queryPool));
    released("destroyCommandPool", device.destroyCommandPool(*commandPool));

    auto again = device.destroyBuffer(*storage);
    REQUIRE_FALSE(again);
    CHECK(again.error().code == rachis::ErrorCode::invalidArgument);
    auto setAgain = device.destroyBindingSet(*bindingSet);
    REQUIRE_FALSE(setAgain);
    CHECK(setAgain.error().code == rachis::ErrorCode::invalidArgument);
    auto pipelineAgain = device.destroyPipeline(*pipeline);
    REQUIRE_FALSE(pipelineAgain);
    CHECK(pipelineAgain.error().code == rachis::ErrorCode::invalidArgument);
    device = {};
    CHECK(errors.load() == 1);
}

#endif
