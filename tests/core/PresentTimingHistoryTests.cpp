#include "Common/PresentTimingHistory.h"
#include "D3D12/D3D12PresentationProgress.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>

using rachis::PresentTiming;
using rachis::PresentTimingFlags;
using rachis::detail::PresentTimingHistory;

TEST_CASE("Present timing history returns settled presents in order and stops at the first pending one",
          "[size:small][rhi][presentation]")
{
    PresentTimingHistory history;
    std::array<PresentTiming, 8> records{};
    auto snapshot = history.read(1, records);
    CHECK(snapshot.submittedSequence == 0);
    CHECK(snapshot.settledSequence == 0);
    CHECK(snapshot.firstAvailableSequence == 0);
    CHECK(snapshot.recordCount == 0);

    for (std::uint64_t sequence = 1; sequence <= 4; ++sequence)
    {
        history.submitted(sequence);
    }
    history.settle(3, 300, PresentTimingFlags::displayed);
    history.settle(1, 100, PresentTimingFlags::displayed);
    snapshot = history.read(1, records);
    CHECK(snapshot.submittedSequence == 4);
    CHECK(snapshot.settledSequence == 1);
    CHECK(snapshot.firstAvailableSequence == 1);
    REQUIRE(snapshot.recordCount == 1);
    CHECK(records[0].presentSequence == 1);
    CHECK(records[0].presentTime == 100);

    history.settle(2, 999, PresentTimingFlags::dropped);
    history.settle(2, 200, PresentTimingFlags::displayed);
    snapshot = history.read(1, records);
    CHECK(snapshot.settledSequence == 3);
    REQUIRE(snapshot.recordCount == 3);
    CHECK(records[1].presentSequence == 2);
    CHECK(records[1].flags == PresentTimingFlags::dropped);
    CHECK(records[1].presentTime == 0);
    CHECK(records[2].presentTime == 300);

    snapshot = history.read(3, std::span(records).first(1));
    REQUIRE(snapshot.recordCount == 1);
    CHECK(records[0].presentSequence == 3);
    CHECK(history.read(1, {}).recordCount == 0);

    history.submitted(5);
    history.settle(5, 500, PresentTimingFlags::displayed | PresentTimingFlags::estimated);
    history.settleBefore(5);
    snapshot = history.read(4, records);
    CHECK(snapshot.settledSequence == 5);
    REQUIRE(snapshot.recordCount == 2);
    CHECK(records[0].flags == PresentTimingFlags::none);
    CHECK(records[1].flags == (PresentTimingFlags::displayed | PresentTimingFlags::estimated));

    history.reset();
    snapshot = history.read(1, records);
    CHECK(snapshot.submittedSequence == 0);
    CHECK(snapshot.firstAvailableSequence == 0);
    CHECK(snapshot.recordCount == 0);
}

TEST_CASE("Present timing history keeps at least the minimum history and reports what it lost",
          "[size:small][rhi][presentation]")
{
    PresentTimingHistory history;
    constexpr std::uint64_t total = PresentTimingHistory::capacity + 10;
    for (std::uint64_t sequence = 1; sequence <= total; ++sequence)
    {
        history.submitted(sequence);
        history.settle(sequence, sequence * 10, PresentTimingFlags::displayed);
    }
    std::array<PresentTiming, PresentTimingHistory::capacity + 4> records{};
    auto snapshot = history.read(1, records);
    CHECK(snapshot.firstAvailableSequence == total - PresentTimingHistory::capacity + 1);
    CHECK(snapshot.settledSequence == total);
    REQUIRE(snapshot.recordCount == PresentTimingHistory::capacity);
    CHECK(records[0].presentSequence == snapshot.firstAvailableSequence);
    CHECK(records[snapshot.recordCount - 1].presentSequence == total);
    CHECK(records[snapshot.recordCount - 1].presentTime == total * 10);

    history.submitted(total + 2 * PresentTimingHistory::capacity);
    snapshot = history.read(1, records);
    CHECK(snapshot.firstAvailableSequence == total + PresentTimingHistory::capacity + 1);
    CHECK(snapshot.settledSequence == snapshot.firstAvailableSequence - 1);
    CHECK(snapshot.recordCount == 0);
    history.settle(total, 1, PresentTimingFlags::displayed);
    CHECK(history.read(1, records).recordCount == 0);
}

TEST_CASE("D3D12 frame statistics settle present timings by present count and back-date late samples",
          "[size:small][rhi][presentation]")
{
    rachis::d3d12::PresentationProgressTracker tracker;
    tracker.start(1, 100);
    for (std::uint32_t count = 101; count <= 104; ++count)
    {
        tracker.presented(count);
    }
    std::array<PresentTiming, 8> records{};
    CHECK(tracker.readTimings(1, records).settledSequence == 0);

    tracker.observedFrame(102, 50, 50, 5'000);
    CHECK(tracker.refreshTicks() == 0);
    auto snapshot = tracker.readTimings(1, records);
    CHECK(snapshot.settledSequence == 2);
    REQUIRE(snapshot.recordCount == 2);
    CHECK(records[0].flags == PresentTimingFlags::none);
    CHECK(records[1].flags == PresentTimingFlags::displayed);
    CHECK(records[1].presentTime == 5'000);
    CHECK(tracker.displayedSequence() == 2);

    tracker.observedFrame(104, 52, 54, 5'400);
    CHECK(tracker.refreshTicks() == 100);
    snapshot = tracker.readTimings(3, records);
    CHECK(snapshot.settledSequence == 4);
    REQUIRE(snapshot.recordCount == 2);
    CHECK(records[0].presentSequence == 3);
    CHECK(records[0].flags == PresentTimingFlags::none);
    CHECK(records[1].flags == (PresentTimingFlags::displayed | PresentTimingFlags::estimated));
    CHECK(records[1].presentTime == 5'200);

    tracker.beginGeneration(2, 104);
    snapshot = tracker.readTimings(1, records);
    CHECK(snapshot.submittedSequence == 0);
    CHECK(snapshot.recordCount == 0);
}
