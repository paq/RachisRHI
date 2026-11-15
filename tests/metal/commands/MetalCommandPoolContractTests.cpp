#include "Internal/CommandBufferAccess.h"
#include "Internal/CommandPoolRetirementTestHook.h"
#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{

#if RACHIS_ENABLE_METAL
    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    template<typename T>
    [[nodiscard]] std::string errorMessage(const rachis::Result<T>& result)
    {
        return result ? std::string{} : result.error().message;
    }

    [[nodiscard]] std::vector<std::byte> loadPnmsBytes(std::string_view metallibName)
    {
        const auto path =
            std::filesystem::path(RACHIS_METALLIB_DIR) / std::filesystem::path(metallibName).replace_extension(".pnms");
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

    struct AutoreleaseMeasurement final
    {
        std::string operation;
        std::size_t autoreleasedObjects = 0;
        bool succeeded = false;
    };
#endif

} // namespace

#if RACHIS_ENABLE_METAL

TEST_CASE("Metal command pools are bounded and recycle abandoned recordings", "[size:large][rhi][metal][command-pool]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto zeroCapacity = device.createCommandPool(rachis::CommandPoolDesc{});
    REQUIRE_FALSE(zeroCapacity);
    CHECK(zeroCapacity.error().code == rachis::ErrorCode::invalidArgument);

    auto invalidQueue = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = static_cast<rachis::QueueType>(99),
        .maxCommandBuffers = 1,
    });
    REQUIRE_FALSE(invalidQueue);
    CHECK(invalidQueue.error().code == rachis::ErrorCode::invalidArgument);

    const std::string oversizedLabel(rachis::maxDebugLabelBytes + 1, 'x');
    auto oversized = device.createCommandPool(rachis::CommandPoolDesc{
        .maxCommandBuffers = 1,
        .label = oversizedLabel,
    });
    REQUIRE_FALSE(oversized);
    CHECK(oversized.error().code == rachis::ErrorCode::invalidArgument);

    auto unknownFlags = device.createCommandPool(rachis::CommandPoolDesc{
        .maxCommandBuffers = 1,
        .flags = static_cast<rachis::CommandPoolFlag>(1U << 31U),
    });
    REQUIRE_FALSE(unknownFlags);
    CHECK(unknownFlags.error().code == rachis::ErrorCode::invalidArgument);

    auto pool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Metal bounded pool",
    });
    REQUIRE(pool);

    auto recording = device.acquireCommandBuffer(*pool);
    REQUIRE(recording);
    CHECK(recording->queueType() == rachis::QueueType::graphics);
    auto exhausted = device.acquireCommandBuffer(*pool);
    REQUIRE_FALSE(exhausted);
    CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);

    recording = std::unexpected(rachis::Error{});
    auto recycled = device.acquireCommandBuffer(*pool);
    REQUIRE(recycled);
    recycled = std::unexpected(rachis::Error{});

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    auto staleAcquire = device.acquireCommandBuffer(*pool);
    REQUIRE_FALSE(staleAcquire);
    CHECK(staleAcquire.error().code == rachis::ErrorCode::invalidArgument);
    auto staleDestroy = device.destroyCommandPool(*pool);
    REQUIRE_FALSE(staleDestroy);
    CHECK(staleDestroy.error().code == rachis::ErrorCode::invalidArgument);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal command-pool reset rejects open scopes and in-flight leases",
          "[size:large][rhi][metal][command-pool][reset]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto noIndividualReset = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
    });
    REQUIRE(noIndividualReset);
    auto noResetCommand = device.acquireCommandBuffer(*noIndividualReset);
    REQUIRE(noResetCommand);
    auto rejectedIndividual = device.resetCommandBuffer(*noResetCommand);
    REQUIRE_FALSE(rejectedIndividual);
    CHECK(rejectedIndividual.error().code == rachis::ErrorCode::invalidState);
    noResetCommand = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyCommandPool(*noIndividualReset).has_value());

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 2,
        .label = "Metal resettable pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(pool);
    auto first = device.acquireCommandBuffer(*pool);
    auto second = device.acquireCommandBuffer(*pool);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(device.resetCommandBuffer(*first).has_value());
    REQUIRE(first->isValid());

    REQUIRE(device.beginDebugGroup(*first, {.label = "Reset validation"}).has_value());
    auto openScopeReset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(openScopeReset);
    CHECK(openScopeReset.error().code == rachis::ErrorCode::invalidState);
    auto openBufferReset = device.resetCommandBuffer(*first);
    REQUIRE_FALSE(openBufferReset);
    CHECK(openBufferReset.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.endDebugGroup(*first).has_value());
    REQUIRE(device.insertDebugMarker(*first, {.label = "After group"}).has_value());

    REQUIRE(device.resetCommandPool(*pool).has_value());
    REQUIRE(first->isValid());
    REQUIRE(second->isValid());

    REQUIRE(rachis::metal::markCommandBufferInFlight(*first, true).has_value());
    auto inFlightReset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(inFlightReset);
    CHECK(inFlightReset.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(rachis::metal::markCommandBufferInFlight(*first, false).has_value());
    REQUIRE(device.resetCommandPool(*pool).has_value());

    first = std::unexpected(rachis::Error{});
    second = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyCommandPool(*pool).has_value());
    auto staleReset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(staleReset);
    CHECK(staleReset.error().code == rachis::ErrorCode::invalidArgument);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal command-pool reset waits while a live recording's state lock is held",
          "[size:large][rhi][metal][command-pool][reset]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Metal locked recording-state pool",
    });
    REQUIRE(pool);
    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command);
    const auto* native = rachis::detail::CommandBufferAccess::get(*command);
    REQUIRE(native != nullptr);

    rachis::Status reset;
    std::atomic<bool> resetReturned{false};
    bool resetWaited = false;
    {
        std::jthread resetter;
        const auto held = native->lockRecordingState();
        REQUIRE(held.ownsLock());
        resetter = std::jthread([&] {
            reset = device.resetCommandPool(*pool);
            resetReturned.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        resetWaited = !resetReturned.load(std::memory_order_acquire);
    }

    CHECK(resetWaited);
    INFO("resetCommandPool: " << errorMessage(reset));
    REQUIRE(reset.has_value());
    REQUIRE(command->isValid());
    command = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyCommandPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal command-pool reset returns an entry quarantined by a failed acquire",
          "[size:large][rhi][metal][command-pool][reset][regression]")
{
    constexpr std::uint64_t transferSize = 64;

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Metal quarantine recovery pool",
    });
    auto upload = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal quarantine recovery upload",
    });
    auto readback = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal quarantine recovery readback",
    });
    REQUIRE(pool);
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> pattern(static_cast<std::size_t>(transferSize));
    for (std::size_t index = 0; index < pattern.size(); ++index)
    {
        pattern[index] = static_cast<std::byte>((index * 7U + 3U) & 0xFFU);
    }
    REQUIRE(device.writeBuffer(*upload, 0, pattern).has_value());

    {
        const rachis::metal::ScopedFailNextResidencySetCreate injectFailure;
        auto failed = device.acquireCommandBuffer(*pool);
        REQUIRE_FALSE(failed);
        CHECK(failed.error().code == rachis::ErrorCode::nativeFailure);
    }
    auto quarantined = device.acquireCommandBuffer(*pool);
    REQUIRE_FALSE(quarantined);
    CHECK(quarantined.error().code == rachis::ErrorCode::resourceExhausted);

    auto reset = device.resetCommandPool(*pool);
    INFO("resetCommandPool: " << errorMessage(reset));
    REQUIRE(reset.has_value());
    auto command = device.acquireCommandBuffer(*pool);
    INFO("acquireCommandBuffer: " << errorMessage(command));
    REQUIRE(command);
    REQUIRE(
        device.copyBuffer(*command, {.source = *upload, .destination = *readback, .size = transferSize}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    std::vector<std::byte> copied(pattern.size());
    REQUIRE(device.readBuffer(*readback, 0, copied).has_value());
    CHECK(copied == pattern);

    auto reacquired = device.acquireCommandBuffer(*pool);
    INFO("acquireCommandBuffer after wait: " << errorMessage(reacquired));
    REQUIRE(reacquired);
    reacquired = std::unexpected(rachis::Error{});

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyCommandPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal copy and timestamp recording rejects an in-flight lease",
          "[size:large][rhi][metal][command-pool][query][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .label = "Metal in-flight recording pool",
    });
    auto queryPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 4,
        .label = "Metal in-flight timestamp pool",
    });
    auto upload = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal in-flight upload",
    });
    auto readback = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal in-flight readback",
    });
    auto timestamps = device.createBuffer({
        .size = 4 * rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal in-flight timestamps",
    });
    REQUIRE(pool);
    REQUIRE(queryPool);
    REQUIRE(upload);
    REQUIRE(readback);
    REQUIRE(timestamps);
    const rachis::GpuProfileScopeDesc scope{
        .queryPool = *queryPool,
        .beginningQueryIndex = 1,
        .endQueryIndex = 2,
        .label = "In-flight scope",
    };

    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command);
    REQUIRE(device.writeTimestamp(*command, *queryPool, 0).has_value());
    REQUIRE(rachis::metal::markCommandBufferInFlight(*command, true).has_value());
    auto copied = device.copyBuffer(*command, {.source = *upload, .destination = *readback, .size = 64});
    auto reset = device.resetQueryPool(*command, *queryPool, 3, 1);
    auto stamped = device.writeTimestamp(*command, *queryPool, 3);
    auto resolved =
        device.resolveTimestamps(*command, {.queryPool = *queryPool, .queryCount = 1, .destination = *timestamps});
    auto begun = device.beginGpuProfileScope(*command, scope);
    REQUIRE(rachis::metal::markCommandBufferInFlight(*command, false).has_value());

    REQUIRE_FALSE(copied);
    CHECK(copied.error().code == rachis::ErrorCode::invalidState);
    REQUIRE_FALSE(reset);
    CHECK(reset.error().code == rachis::ErrorCode::invalidState);
    REQUIRE_FALSE(stamped);
    CHECK(stamped.error().code == rachis::ErrorCode::invalidState);
    REQUIRE_FALSE(resolved);
    CHECK(resolved.error().code == rachis::ErrorCode::invalidState);
    REQUIRE_FALSE(begun);
    CHECK(begun.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.beginGpuProfileScope(*command, scope).has_value());
    REQUIRE(rachis::metal::markCommandBufferInFlight(*command, true).has_value());
    auto ended = device.endGpuProfileScope(*command);
    REQUIRE(rachis::metal::markCommandBufferInFlight(*command, false).has_value());
    REQUIRE_FALSE(ended);
    CHECK(ended.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.endGpuProfileScope(*command).has_value());

    REQUIRE(device.copyBuffer(*command, {.source = *upload, .destination = *readback, .size = 64}).has_value());
    REQUIRE(device.resolveTimestamps(*command, {.queryPool = *queryPool, .queryCount = 3, .destination = *timestamps})
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal command-pool destruction keeps live leases until they are abandoned",
          "[size:large][rhi][metal][command-pool][lifetime]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto firstPool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "First Metal pool",
    });
    auto secondPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Second Metal pool",
    });
    REQUIRE(firstPool);
    REQUIRE(secondPool);

    auto first = device.acquireCommandBuffer(*firstPool);
    auto second = device.acquireCommandBuffer(*secondPool);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(device.destroyCommandPool(*firstPool).has_value());
    auto staleAcquire = device.acquireCommandBuffer(*firstPool);
    REQUIRE_FALSE(staleAcquire);
    CHECK(staleAcquire.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(first->isValid());
    first = std::unexpected(rachis::Error{});
    second = std::unexpected(rachis::Error{});

    auto recycled = device.acquireCommandBuffer(*secondPool);
    REQUIRE(recycled);
    recycled = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyCommandPool(*secondPool).has_value());

    auto replacement = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Replacement Metal pool",
    });
    REQUIRE(replacement);
    CHECK(*replacement != *firstPool);
    REQUIRE(device.destroyCommandPool(*replacement).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal command pools record independently without a device-wide recording lock",
          "[size:large][rhi][metal][command-pool][concurrency]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto graphicsPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Concurrent graphics",
    });
    auto copyPool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Concurrent copy",
    });
    REQUIRE(graphicsPool);
    REQUIRE(copyPool);

    std::atomic<int> successes{0};
    std::thread graphicsThread([&] {
        auto command = device.acquireCommandBuffer(*graphicsPool);
        if (command)
        {
            successes.fetch_add(1, std::memory_order_relaxed);
            REQUIRE(device.beginDebugGroup(*command, {.label = "Graphics thread"}).has_value());
            REQUIRE(device.endDebugGroup(*command).has_value());
        }
    });
    std::thread copyThread([&] {
        auto command = device.acquireCommandBuffer(*copyPool);
        if (command)
        {
            successes.fetch_add(1, std::memory_order_relaxed);
            REQUIRE(device.insertDebugMarker(*command, {.label = "Copy thread"}).has_value());
        }
    });
    graphicsThread.join();
    copyThread.join();
    CHECK(successes.load(std::memory_order_relaxed) == 2);

    REQUIRE(device.destroyCommandPool(*graphicsPool).has_value());
    REQUIRE(device.destroyCommandPool(*copyPool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal copy and timestamp recording waits for a concurrent command-pool reset",
          "[size:large][rhi][metal][command-pool][concurrency][query][regression]")
{
    using namespace std::chrono_literals;
    constexpr std::uint64_t transferSize = 64;

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 6,
        .label = "Metal reset-race pool",
    });
    auto queryPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 8,
        .label = "Metal reset-race timestamps",
    });
    auto upload = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal reset-race upload",
    });
    auto readback = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal reset-race readback",
    });
    auto timestamps = device.createBuffer({
        .size = 8 * rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal reset-race timestamp readback",
    });
    REQUIRE(pool);
    REQUIRE(queryPool);
    REQUIRE(upload);
    REQUIRE(readback);
    REQUIRE(timestamps);

    std::vector<std::byte> pattern(static_cast<std::size_t>(transferSize));
    for (std::size_t index = 0; index < pattern.size(); ++index)
    {
        pattern[index] = static_cast<std::byte>((index * 13U + 5U) & 0xFFU);
    }
    REQUIRE(device.writeBuffer(*upload, 0, pattern).has_value());
    {
        auto primer = device.acquireCommandBuffer(*pool);
        REQUIRE(primer);
        REQUIRE(device.writeTimestamp(*primer, *queryPool, 3).has_value());
        auto primed = device.submit(std::move(*primer));
        REQUIRE(primed);
        REQUIRE(device.wait(*primed).has_value());
    }

    auto copyCommand = device.acquireCommandBuffer(*pool);
    auto resetCommand = device.acquireCommandBuffer(*pool);
    auto stampCommand = device.acquireCommandBuffer(*pool);
    auto scopeCommand = device.acquireCommandBuffer(*pool);
    auto resolveCommand = device.acquireCommandBuffer(*pool);
    REQUIRE(copyCommand);
    REQUIRE(resetCommand);
    REQUIRE(stampCommand);
    REQUIRE(scopeCommand);
    REQUIRE(resolveCommand);
    const auto uploadObserver = rachis::metal::boundBufferOwnerObserver(*upload);
    REQUIRE_FALSE(uploadObserver.expired());

    struct ResetGateRelease final
    {
        std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate;

        ~ResetGateRelease()
        {
            gate->release();
            rachis::metal::holdNextCommandPoolReset(nullptr);
        }
    };

    rachis::Status reset;
    rachis::Status copied;
    rachis::Status queriesReset;
    rachis::Status stamped;
    rachis::Status scopeBegun;
    rachis::Status scopeEnded;
    rachis::Status resolved;
    std::atomic<int> finished{0};
    bool parked = false;
    int finishedWhileParked = 0;
    {
        std::jthread resetter;
        std::array<std::jthread, 5> recorders;
        const ResetGateRelease release{std::make_shared<rachis::metal::MetalCommitFeedbackGate>()};
        rachis::metal::holdNextCommandPoolReset(release.gate);
        resetter = std::jthread([&] { reset = device.resetCommandPool(*pool); });
        parked = release.gate->waitForEntry(10s);
        recorders[0] = std::jthread([&] {
            copied =
                device.copyBuffer(*copyCommand, {.source = *upload, .destination = *readback, .size = transferSize});
            finished.fetch_add(1, std::memory_order_acq_rel);
        });
        recorders[1] = std::jthread([&] {
            queriesReset = device.resetQueryPool(*resetCommand, *queryPool, 6, 2);
            finished.fetch_add(1, std::memory_order_acq_rel);
        });
        recorders[2] = std::jthread([&] {
            stamped = device.writeTimestamp(*stampCommand, *queryPool, 0);
            finished.fetch_add(1, std::memory_order_acq_rel);
        });
        recorders[3] = std::jthread([&] {
            scopeBegun = device.beginGpuProfileScope(*scopeCommand,
                                                     {
                                                         .queryPool = *queryPool,
                                                         .beginningQueryIndex = 1,
                                                         .endQueryIndex = 2,
                                                         .label = "Reset-race scope",
                                                     });
            finished.fetch_add(1, std::memory_order_acq_rel);
            scopeEnded = device.endGpuProfileScope(*scopeCommand);
        });
        recorders[4] = std::jthread([&] {
            resolved = device.resolveTimestamps(*resolveCommand,
                                                {
                                                    .queryPool = *queryPool,
                                                    .firstQuery = 3,
                                                    .queryCount = 1,
                                                    .destination = *timestamps,
                                                    .destinationOffset = 3 * rachis::timestampQueryResultSize,
                                                });
            finished.fetch_add(1, std::memory_order_acq_rel);
        });
        std::this_thread::sleep_for(200ms);
        finishedWhileParked = finished.load(std::memory_order_acquire);
    }

    REQUIRE(parked);
    CHECK(finishedWhileParked == 0);
    INFO("resetCommandPool: " << errorMessage(reset));
    REQUIRE(reset.has_value());
    REQUIRE(copied.has_value());
    REQUIRE(queriesReset.has_value());
    REQUIRE(stamped.has_value());
    REQUIRE(scopeBegun.has_value());
    REQUIRE(scopeEnded.has_value());
    REQUIRE(resolved.has_value());

    REQUIRE(device.destroyBuffer(*upload).has_value());
    CHECK_FALSE(uploadObserver.expired());

    std::array batch{
        std::move(*copyCommand),
        std::move(*resetCommand),
        std::move(*stampCommand),
        std::move(*scopeCommand),
        std::move(*resolveCommand),
    };
    auto submitted = device.submit(batch);
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    std::vector<std::byte> copiedBytes(pattern.size());
    REQUIRE(device.readBuffer(*readback, 0, copiedBytes).has_value());
    CHECK(copiedBytes == pattern);
    std::uint64_t resolvedTick = 0;
    REQUIRE(device
                .readBuffer(*timestamps,
                            3 * rachis::timestampQueryResultSize,
                            std::as_writable_bytes(std::span{&resolvedTick, 1}))
                .has_value());
    CHECK(resolvedTick != 0);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal waitIdle parked after retiring a submission leaves its command-pool entry reusable",
          "[size:large][rhi][metal][command-pool][concurrency]")
{
    using rachis::detail::test::CommandPoolRetirementParking;
    constexpr std::uint64_t transferSize = 256;

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto commandPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Metal parked retirement test pool",
    });
    REQUIRE(commandPool.has_value());
    auto upload = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal parked retirement upload",
    });
    auto readback = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal parked retirement readback",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());

    auto command = device.acquireCommandBuffer(*commandPool);
    REQUIRE(command.has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *upload,
                                .destination = *readback,
                                .size = transferSize,
                            })
                .has_value());
    auto point = device.submit(std::move(*command));
    REQUIRE(point.has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    rachis::Status collectorIdle;
    rachis::Status ownerWait;
    rachis::Status ownerReset;
    rachis::Result<rachis::CommandBuffer> reacquired = std::unexpected(rachis::Error{});
    bool parked = false;
    {
        const CommandPoolRetirementParking parking;
        std::jthread collector([&device, &collectorIdle] {
            CommandPoolRetirementParking::armCurrentThread();
            collectorIdle = device.waitIdle(rachis::QueueType::graphics);
        });
        parked = CommandPoolRetirementParking::waitUntilParked(std::chrono::seconds(10));
        ownerWait = device.wait(*point);
        ownerReset = device.resetCommandPool(*commandPool);
        reacquired = device.acquireCommandBuffer(*commandPool);
        CommandPoolRetirementParking::release();
    }

    REQUIRE(parked);
    REQUIRE(collectorIdle.has_value());
    REQUIRE(ownerWait.has_value());
    INFO("resetCommandPool: " << errorMessage(ownerReset));
    INFO("acquireCommandBuffer: " << errorMessage(reacquired));
    CHECK(ownerReset.has_value());
    REQUIRE(reacquired.has_value());
    auto resubmitted = device.submit(std::move(*reacquired));
    REQUIRE(resubmitted.has_value());
    REQUIRE(device.wait(*resubmitted).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
    REQUIRE(device.destroyCommandPool(*commandPool).has_value());
}

TEST_CASE("Metal one-entry command pools reacquire after their own wait while another thread drains the queue",
          "[size:large][rhi][metal][command-pool][concurrency]")
{
    constexpr std::size_t workerCount = 8;
    constexpr std::size_t iterationCount = 200;
    constexpr std::uint64_t transferSize = 256;

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    std::array<rachis::CommandPoolHandle, workerCount> commandPools{};
    std::array<rachis::BufferHandle, workerCount> uploads{};
    std::array<rachis::BufferHandle, workerCount> readbacks{};
    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        auto commandPool = device.createCommandPool({
            .queueType = rachis::QueueType::graphics,
            .maxCommandBuffers = 1,
            .label = "Metal one-entry reacquisition test pool",
        });
        auto upload = device.createBuffer({
            .size = transferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Metal one-entry reacquisition upload",
        });
        auto readback = device.createBuffer({
            .size = transferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Metal one-entry reacquisition readback",
        });
        REQUIRE(commandPool.has_value());
        REQUIRE(upload.has_value());
        REQUIRE(readback.has_value());
        commandPools[workerIndex] = *commandPool;
        uploads[workerIndex] = *upload;
        readbacks[workerIndex] = *readback;
    }

    std::atomic<std::size_t> exhaustedAfterOwnWait{0};
    std::atomic<std::size_t> otherFailures{0};
    std::atomic<bool> stopCollector{false};
    std::barrier startBarrier(static_cast<std::ptrdiff_t>(workerCount + 1));
    {
        std::jthread collector([&device, &stopCollector, &startBarrier] {
            startBarrier.arrive_and_wait();
            while (!stopCollector.load(std::memory_order_acquire))
            {
                static_cast<void>(device.waitIdle(rachis::QueueType::graphics));
                std::this_thread::yield();
            }
        });
        {
            std::vector<std::jthread> workers;
            workers.reserve(workerCount);
            for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
            {
                workers.emplace_back([&, workerIndex] {
                    startBarrier.arrive_and_wait();
                    for (std::size_t iteration = 0; iteration < iterationCount; ++iteration)
                    {
                        auto command = device.acquireCommandBuffer(commandPools[workerIndex]);
                        if (!command)
                        {
                            auto& counter = command.error().code == rachis::ErrorCode::resourceExhausted
                                                ? exhaustedAfterOwnWait
                                                : otherFailures;
                            counter.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        if (!device.copyBuffer(*command,
                                               rachis::BufferCopy{
                                                   .source = uploads[workerIndex],
                                                   .destination = readbacks[workerIndex],
                                                   .size = transferSize,
                                               }))
                        {
                            otherFailures.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        auto point = device.submit(std::move(*command));
                        if (!point || !device.wait(*point))
                        {
                            otherFailures.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                });
            }
        }
        stopCollector.store(true, std::memory_order_release);
    }

    CHECK(exhaustedAfterOwnWait.load() == 0);
    CHECK(otherFailures.load() == 0);
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        REQUIRE(device.destroyBuffer(readbacks[workerIndex]).has_value());
        REQUIRE(device.destroyBuffer(uploads[workerIndex]).has_value());
        REQUIRE(device.destroyCommandPool(commandPools[workerIndex]).has_value());
    }
}

TEST_CASE("Metal recording on a worker thread leaves no autoreleased objects in the caller's pool",
          "[size:large][rhi][metal][command-pool][concurrency]")
{
    constexpr std::uint64_t transferSize = 256;

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto variant7 = loadPnmsBytes("MetalSpecialization.7.cs.metallib");
    REQUIRE_FALSE(variant7.empty());
    const std::array storageBinding{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = variant7,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, storageBinding),
    });
    REQUIRE(shader.has_value());
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout.has_value());
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = std::array{*bindingLayout}});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    auto storage = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto upload = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(storage.has_value());
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    auto set = rachis::test::createBindingSet(
        device,
        {.layout = *bindingLayout,
         .writes = std::array{
             rachis::BindingWrite{0, 0, rachis::BufferBinding{*storage, 0, transferSize, sizeof(std::uint32_t)}}}});
    REQUIRE(set.has_value());
    auto target = device.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE(targetView.has_value());
    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Metal worker autorelease pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(pool.has_value());

    std::vector<AutoreleaseMeasurement> measurements;
    std::thread worker([&] {
        const auto measure = [&](std::string operation, const std::function<bool()>& body) {
            bool succeeded = false;
            const auto count = rachis::metal::autoreleasedObjectCount([&] { succeeded = body(); });
            measurements.push_back(AutoreleaseMeasurement{
                .operation = std::move(operation),
                .autoreleasedObjects = count,
                .succeeded = succeeded,
            });
            return succeeded;
        };
        rachis::Result<rachis::CommandBuffer> command = std::unexpected(rachis::Error{});
        if (!measure("acquireCommandBuffer", [&] {
                command = device.acquireCommandBuffer(*pool);
                return command.has_value();
            }))
        {
            return;
        }
        const std::array colorAttachments{
            rachis::ColorAttachment{.view = *targetView, .loadOperation = rachis::LoadOperation::clear},
        };
        static_cast<void>(measure("beginRendering and endRendering", [&] {
            return device.beginRendering(*command, {.colorAttachments = colorAttachments}).has_value() &&
                   device.endRendering(*command).has_value();
        }));
        static_cast<void>(measure("copyBuffer", [&] {
            return device.copyBuffer(*command, {.source = *upload, .destination = *readback, .size = transferSize})
                .has_value();
        }));
        static_cast<void>(measure("bindComputePipeline and bindComputeBindingSet", [&] {
            return device.bindComputePipeline(*command, *pipeline).has_value() &&
                   device.bindComputeBindingSet(*command, 0, *set).has_value();
        }));
        static_cast<void>(measure("dispatch", [&] { return device.dispatch(*command, 1).has_value(); }));
        static_cast<void>(measure("debug groups and markers", [&] {
            return device.beginDebugGroup(*command, {.label = "Worker group"}).has_value() &&
                   device.insertDebugMarker(*command, {.label = "Worker marker"}).has_value() &&
                   device.endDebugGroup(*command).has_value();
        }));
        static_cast<void>(
            measure("resetCommandBuffer", [&] { return device.resetCommandBuffer(*command).has_value(); }));
        static_cast<void>(measure("deferred barrier and submit", [&] {
            const std::array barriers{
                rachis::GlobalBarrier{
                    .stagesBefore = rachis::PipelineStage::copy,
                    .stagesAfter = rachis::PipelineStage::copy,
                    .accessBefore = rachis::Access::copyDestination,
                    .accessAfter = rachis::Access::copyDestination,
                },
            };
            if (!device.resourceBarriers(*command, {.globalBarriers = barriers}).has_value())
            {
                return false;
            }
            auto submitted = device.submit(std::move(*command));
            return submitted.has_value() && device.wait(*submitted).has_value();
        }));
    });
    worker.join();

    REQUIRE(measurements.size() == 8);
    for (const auto& measurement : measurements)
    {
        CAPTURE(measurement.operation, measurement.autoreleasedObjects);
        CHECK(measurement.succeeded);
        CHECK(measurement.autoreleasedObjects == 0);
    }

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

#endif
