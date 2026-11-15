#include "D3D12/D3D12AccelerationStructureLease.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    using namespace rachis::d3d12;

}

TEST_CASE("D3D12 acceleration-structure execution leases decouple GPU execution from passive host ownership",
          "[size:small][rhi][d3d12][acceleration-structure]")
{
    AccelerationStructureRecordingSequence recordings;
    auto blasTracker = std::make_shared<AccelerationStructureExecutionTracker>();
    auto tlasTracker = std::make_shared<AccelerationStructureExecutionTracker>();

    const auto passiveTlasDependency = blasTracker;
    const auto passiveBindingSetReference = tlasTracker;
    static_cast<void>(passiveTlasDependency);
    static_cast<void>(passiveBindingSetReference);

    SECTION("passive TLAS and BindingSet references allow animated BLAS and TLAS in-place refits once idle")
    {
        const auto frame0 = recordings.next();
        REQUIRE(frame0 != 0);

        {
            auto blasBuildLease = AccelerationStructureExecutionLease::tryAcquireProducer(blasTracker, frame0);
            REQUIRE(blasBuildLease.has_value());
            CHECK(blasBuildLease->active());
            CHECK(blasTracker->activeProducerCount() == 1);
            CHECK(blasTracker->activeConsumerCount() == 0);

            const std::array tlasBuildConsumers{blasTracker};
            auto tlasBuildLease =
                AccelerationStructureExecutionLease::tryAcquireProducer(tlasTracker, frame0, tlasBuildConsumers);
            REQUIRE(tlasBuildLease.has_value());
            CHECK(tlasTracker->activeProducerCount() == 1);
            CHECK(blasTracker->activeConsumerCount() == 1);

            const std::array boundSetTrackers{tlasTracker, blasTracker};
            auto boundSetLease = AccelerationStructureExecutionLease::tryAcquireConsumers(boundSetTrackers, frame0);
            REQUIRE(boundSetLease.has_value());
            CHECK(boundSetLease->active());
            CHECK(tlasTracker->activeConsumerCount() == 1);
            CHECK(blasTracker->activeConsumerCount() == 2);

            const auto concurrentRecording = recordings.next();
            CHECK_FALSE(blasTracker->canUpdateInPlace(concurrentRecording));
            CHECK_FALSE(tlasTracker->canUpdateInPlace(concurrentRecording));
        }

        CHECK_FALSE(blasTracker->hasActiveExecutions());
        CHECK_FALSE(tlasTracker->hasActiveExecutions());

        const auto frame1 = recordings.next();
        REQUIRE(blasTracker->canUpdateInPlace(frame1));
        REQUIRE(tlasTracker->canUpdateInPlace(frame1));

        auto blasRefitLease = AccelerationStructureExecutionLease::tryAcquireProducer(
            blasTracker, frame1, {}, AccelerationStructureWrite::updateInPlace);
        REQUIRE(blasRefitLease.has_value());

        const std::array tlasRefitConsumers{blasTracker};
        auto tlasRefitLease = AccelerationStructureExecutionLease::tryAcquireProducer(
            tlasTracker, frame1, tlasRefitConsumers, AccelerationStructureWrite::updateInPlace);
        REQUIRE(tlasRefitLease.has_value());
        CHECK(blasTracker->activeProducerCount() == 1);
        CHECK(blasTracker->activeConsumerCount() == 1);
        CHECK(tlasTracker->activeProducerCount() == 1);
    }

    SECTION("sequential producer on the same open command buffer succeeds while foreign or post-submit recordings fail")
    {
        const auto recordingA = recordings.next();
        const auto recordingB = recordings.next();
        REQUIRE(recordingA != recordingB);

        CHECK_FALSE(blasTracker->canUpdateInPlace(0));
        CHECK_FALSE(AccelerationStructureExecutionLease::tryAcquireProducer(blasTracker, 0).has_value());
        CHECK_FALSE(AccelerationStructureExecutionLease::tryAcquireProducer(nullptr, recordingA).has_value());

        auto initialBuild = AccelerationStructureExecutionLease::tryAcquireProducer(blasTracker, recordingA);
        REQUIRE(initialBuild.has_value());
        CHECK(blasTracker->activeProducerCount() == 1);
        CHECK(blasTracker->activeProducerRecordingId() == recordingA);

        CHECK(blasTracker->canUpdateInPlace(recordingA));
        auto sameBufferRefit = AccelerationStructureExecutionLease::tryAcquireProducer(blasTracker, recordingA);
        REQUIRE(sameBufferRefit.has_value());
        CHECK(blasTracker->activeProducerCount() == 2);
        CHECK(blasTracker->activeProducerRecordingId() == recordingA);

        CHECK_FALSE(blasTracker->canUpdateInPlace(recordingB));
        const std::array dependentConsumer{tlasTracker};
        auto rejectedForeignRefit =
            AccelerationStructureExecutionLease::tryAcquireProducer(blasTracker, recordingB, dependentConsumer);
        CHECK_FALSE(rejectedForeignRefit.has_value());
        CHECK(tlasTracker->activeConsumerCount() == 0);

        initialBuild->reset();
        CHECK_FALSE(initialBuild->active());
        CHECK(blasTracker->activeProducerCount() == 1);
        CHECK(blasTracker->activeProducerRecordingId() == recordingA);
        CHECK_FALSE(blasTracker->canUpdateInPlace(recordingB));

        sameBufferRefit->reset();
        CHECK(blasTracker->activeProducerCount() == 0);
        CHECK(blasTracker->activeProducerRecordingId() == 0);
        CHECK(blasTracker->canUpdateInPlace(recordingB));
    }

    SECTION("consumer leases block even same-command-buffer in-place updates until reset or abandoned")
    {
        const auto recordingA = recordings.next();
        auto buildLease = AccelerationStructureExecutionLease::tryAcquireProducer(blasTracker, recordingA);
        REQUIRE(buildLease.has_value());
        CHECK(blasTracker->canUpdateInPlace(recordingA));

        auto acquiredConsumer = AccelerationStructureExecutionLease::tryAcquireConsumer(blasTracker, recordingA);
        REQUIRE(acquiredConsumer.has_value());
        auto consumerLease = std::move(*acquiredConsumer);
        CHECK(consumerLease.active());
        CHECK(blasTracker->activeConsumerCount() == 1);
        CHECK_FALSE(blasTracker->canUpdateInPlace(recordingA));
        CHECK_FALSE(AccelerationStructureExecutionLease::tryAcquireProducer(blasTracker, recordingA).has_value());

        consumerLease.reset();
        CHECK_FALSE(consumerLease.active());
        CHECK(blasTracker->activeConsumerCount() == 0);
        CHECK(blasTracker->canUpdateInPlace(recordingA));
    }

    SECTION("in-place producers exclude consumers in other recordings until they are released")
    {
        const auto refitRecording = recordings.next();
        const auto foreignRecording = recordings.next();

        auto publication = AccelerationStructureExecutionLease::tryAcquireProducer(blasTracker, refitRecording);
        REQUIRE(publication.has_value());
        auto consumerOfPublication =
            AccelerationStructureExecutionLease::tryAcquireConsumer(blasTracker, foreignRecording);
        REQUIRE(consumerOfPublication.has_value());
        CHECK(blasTracker->activeConsumerCount() == 1);
        consumerOfPublication.reset();
        publication->reset();
        CHECK_FALSE(blasTracker->hasActiveExecutions());

        auto refit = AccelerationStructureExecutionLease::tryAcquireProducer(
            blasTracker, refitRecording, {}, AccelerationStructureWrite::updateInPlace);
        REQUIRE(refit.has_value());
        CHECK(blasTracker->activeInPlaceProducerCount() == 1);
        auto foreignRefit = AccelerationStructureExecutionLease::tryAcquireProducer(
            blasTracker, foreignRecording, {}, AccelerationStructureWrite::updateInPlace);
        REQUIRE_FALSE(foreignRefit.has_value());
        CHECK(foreignRefit.error() == AccelerationStructureLeaseConflict::producer);

        CHECK_FALSE(AccelerationStructureExecutionLease::tryAcquireConsumer(blasTracker, foreignRecording).has_value());
        CHECK_FALSE(AccelerationStructureExecutionLease::tryAcquireConsumer(blasTracker, 0).has_value());
        const std::array bindingTrackers{tlasTracker, blasTracker};
        CHECK_FALSE(
            AccelerationStructureExecutionLease::tryAcquireConsumers(bindingTrackers, foreignRecording).has_value());
        CHECK(tlasTracker->activeConsumerCount() == 0);
        CHECK(blasTracker->activeConsumerCount() == 0);

        const std::array refitConsumers{blasTracker};
        auto foreignTlasBuild =
            AccelerationStructureExecutionLease::tryAcquireProducer(tlasTracker, foreignRecording, refitConsumers);
        REQUIRE_FALSE(foreignTlasBuild.has_value());
        CHECK(foreignTlasBuild.error() == AccelerationStructureLeaseConflict::consumer);
        CHECK_FALSE(tlasTracker->hasActiveExecutions());
        CHECK(blasTracker->activeConsumerCount() == 0);

        auto sameRecordingTlasRefit = AccelerationStructureExecutionLease::tryAcquireProducer(
            tlasTracker, refitRecording, refitConsumers, AccelerationStructureWrite::updateInPlace);
        REQUIRE(sameRecordingTlasRefit.has_value());
        auto sameRecordingBinding =
            AccelerationStructureExecutionLease::tryAcquireConsumers(bindingTrackers, refitRecording);
        REQUIRE(sameRecordingBinding.has_value());
        CHECK(blasTracker->activeConsumerCount() == 2);
        CHECK(tlasTracker->activeConsumerCount() == 1);
        CHECK_FALSE(
            AccelerationStructureExecutionLease::tryAcquireConsumers(bindingTrackers, foreignRecording).has_value());

        sameRecordingBinding.reset();
        sameRecordingTlasRefit->reset();
        refit->reset();
        CHECK(blasTracker->activeInPlaceProducerCount() == 0);
        CHECK(tlasTracker->activeInPlaceProducerCount() == 0);
        CHECK_FALSE(blasTracker->hasActiveExecutions());
        CHECK_FALSE(tlasTracker->hasActiveExecutions());

        auto foreignBinding =
            AccelerationStructureExecutionLease::tryAcquireConsumers(bindingTrackers, foreignRecording);
        REQUIRE(foreignBinding.has_value());
        CHECK(foreignBinding->active());
        auto nullConsumer = AccelerationStructureExecutionLease::tryAcquireConsumer(nullptr, foreignRecording);
        REQUIRE(nullConsumer.has_value());
        CHECK_FALSE(nullConsumer->active());
    }

    SECTION("duplicate consumer instances and move operations preserve exact balanced counts")
    {
        auto blasB = std::make_shared<AccelerationStructureExecutionTracker>();
        const auto recording = recordings.next();
        const std::array duplicateConsumers{
            blasTracker,
            std::shared_ptr<AccelerationStructureExecutionTracker>{},
            blasTracker,
            blasB,
            blasTracker,
            blasB,
        };

        auto lease =
            AccelerationStructureExecutionLease::tryAcquireProducer(tlasTracker, recording, duplicateConsumers);
        REQUIRE(lease.has_value());
        CHECK(tlasTracker->activeProducerCount() == 1);
        CHECK(blasTracker->activeConsumerCount() == 1);
        CHECK(blasB->activeConsumerCount() == 1);

        AccelerationStructureExecutionLease movedConstructed(std::move(*lease));
        CHECK_FALSE(lease->active());
        CHECK(movedConstructed.active());
        CHECK(tlasTracker->activeProducerCount() == 1);
        CHECK(blasTracker->activeConsumerCount() == 1);
        CHECK(blasB->activeConsumerCount() == 1);

        auto acquiredOther = AccelerationStructureExecutionLease::tryAcquireConsumer(blasB, recording);
        REQUIRE(acquiredOther.has_value());
        auto otherLease = std::move(*acquiredOther);
        CHECK(blasB->activeConsumerCount() == 2);
        otherLease = std::move(movedConstructed);
        CHECK_FALSE(movedConstructed.active());
        CHECK(otherLease.active());
        CHECK(tlasTracker->activeProducerCount() == 1);
        CHECK(blasTracker->activeConsumerCount() == 1);
        CHECK(blasB->activeConsumerCount() == 1);

        otherLease.reset();
        CHECK_FALSE(tlasTracker->hasActiveExecutions());
        CHECK_FALSE(blasTracker->hasActiveExecutions());
        CHECK_FALSE(blasB->hasActiveExecutions());
    }

    SECTION("concurrent consumer and producer lease transitions remain linearizable")
    {
        std::atomic<bool> start{false};
        std::atomic<std::uint64_t> observedViolations{0};

        constexpr std::size_t consumerThreadCount = 3;
        constexpr std::size_t producerThreadCount = 2;
        constexpr std::size_t iterationsPerThread = 500;

        std::vector<std::jthread> threads;
        threads.reserve(consumerThreadCount + producerThreadCount);

        for (std::size_t worker = 0; worker < consumerThreadCount; ++worker)
        {
            threads.emplace_back([&]() {
                while (!start.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                const std::array trackers{tlasTracker, blasTracker};
                for (std::size_t iteration = 0; iteration < iterationsPerThread; ++iteration)
                {
                    auto lease = AccelerationStructureExecutionLease::tryAcquireConsumers(trackers, recordings.next());
                    if (!lease.has_value() || blasTracker->activeConsumerCount() == 0 ||
                        tlasTracker->activeConsumerCount() == 0)
                    {
                        observedViolations.fetch_add(1, std::memory_order_relaxed);
                    }
                    std::this_thread::yield();
                }
            });
        }

        for (std::size_t worker = 0; worker < producerThreadCount; ++worker)
        {
            threads.emplace_back([&]() {
                while (!start.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                for (std::size_t iteration = 0; iteration < iterationsPerThread; ++iteration)
                {
                    const auto recording = recordings.next();
                    auto lease = AccelerationStructureExecutionLease::tryAcquireProducer(blasTracker, recording);
                    if (lease.has_value())
                    {
                        if (blasTracker->activeProducerRecordingId() != recording)
                        {
                            observedViolations.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    std::this_thread::yield();
                }
            });
        }

        start.store(true, std::memory_order_release);
        threads.clear();

        CHECK(observedViolations.load(std::memory_order_relaxed) == 0);
        CHECK_FALSE(blasTracker->hasActiveExecutions());
        CHECK_FALSE(tlasTracker->hasActiveExecutions());

        const auto finalRecording = recordings.next();
        auto finalRefit = AccelerationStructureExecutionLease::tryAcquireProducer(blasTracker, finalRecording);
        REQUIRE(finalRefit.has_value());
    }

    SECTION("large tracker sets deduplicate without quadratic blowup and quarantine clearing releases leases")
    {
        constexpr std::size_t uniqueBlasCount = 2048;
        std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> uniqueTrackers;
        uniqueTrackers.reserve(uniqueBlasCount);
        for (std::size_t index = 0; index < uniqueBlasCount; ++index)
        {
            uniqueTrackers.push_back(std::make_shared<AccelerationStructureExecutionTracker>());
        }

        std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> unsortedWithDuplicates;
        unsortedWithDuplicates.reserve(uniqueBlasCount * 2U + 4U);
        unsortedWithDuplicates.push_back(nullptr);
        for (std::size_t index = uniqueBlasCount; index > 0; --index)
        {
            unsortedWithDuplicates.push_back(uniqueTrackers[index - 1U]);
            unsortedWithDuplicates.push_back(uniqueTrackers[(index * 7U) % uniqueBlasCount]);
        }
        unsortedWithDuplicates.push_back(nullptr);
        deduplicateSharedPointers(unsortedWithDuplicates);
        REQUIRE(unsortedWithDuplicates.size() == uniqueBlasCount);

        std::vector<AccelerationStructureExecutionLease> entryLeases;
        const auto recording = recordings.next();
        auto tlasBuildLease =
            AccelerationStructureExecutionLease::tryAcquireProducer(tlasTracker, recording, unsortedWithDuplicates);
        REQUIRE(tlasBuildLease.has_value());
        entryLeases.push_back(std::move(*tlasBuildLease));

        std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> presortedBindTrackers;
        presortedBindTrackers.reserve(unsortedWithDuplicates.size() + 1U);
        const auto lessByAddress = [](const auto& lhs, const auto& rhs) noexcept {
            return std::less<const AccelerationStructureExecutionTracker*>{}(lhs.get(), rhs.get());
        };
        const auto insertion = std::ranges::lower_bound(unsortedWithDuplicates, tlasTracker, lessByAddress);
        presortedBindTrackers.insert(presortedBindTrackers.end(), unsortedWithDuplicates.begin(), insertion);
        presortedBindTrackers.push_back(tlasTracker);
        presortedBindTrackers.insert(presortedBindTrackers.end(), insertion, unsortedWithDuplicates.end());

        auto bindLease = AccelerationStructureExecutionLease::tryAcquireConsumers(presortedBindTrackers, recording);
        REQUIRE(bindLease.has_value());
        entryLeases.push_back(std::move(*bindLease));
        CHECK(tlasTracker->activeProducerCount() == 1);
        CHECK(tlasTracker->activeConsumerCount() == 1);
        CHECK(uniqueTrackers.front()->activeConsumerCount() == 2);
        CHECK(uniqueTrackers.back()->activeConsumerCount() == 2);

        entryLeases.clear();
        CHECK_FALSE(tlasTracker->hasActiveExecutions());
        for (const auto& tracker : uniqueTrackers)
        {
            CHECK_FALSE(tracker->hasActiveExecutions());
        }
        CHECK(uniqueTrackers.front()->canUpdateInPlace(recordings.next()));
    }
}
