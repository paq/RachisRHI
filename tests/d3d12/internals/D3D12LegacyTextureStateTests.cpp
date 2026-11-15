#include "D3D12/D3D12LegacyTextureStates.h"
#include "HeapAllocationCounter.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace
{

    using namespace rachis;
    using namespace rachis::d3d12;

} // namespace

TEST_CASE("D3D12 legacy texture state submissions resolve discards against the submitted state",
          "[size:small][rhi][d3d12][barrier]")
{
    int nativeStorage = 0;
    auto* const resource = reinterpret_cast<ID3D12Resource*>(&nativeStorage);
    const auto graphicsStates = legacyQueueResourceStates(QueueType::graphics);
    const auto computeStates = legacyQueueResourceStates(QueueType::compute);
    auto tracker = std::make_shared<LegacyTextureStateTracker>(resource, 2);

    const auto submit = [&](std::span<const LegacyTextureStateRecording* const> recordings,
                            D3D12_RESOURCE_STATES queueStates) {
        LegacyTextureStateSubmission submission;
        std::vector<std::vector<D3D12_RESOURCE_BARRIER>> prologues(recordings.size());
        for (std::size_t index = 0; index < recordings.size(); ++index)
        {
            submission.resolve(*recordings[index], queueStates, prologues[index]);
        }
        submission.commit();
        return prologues;
    };
    const auto submittedState = [&](std::uint32_t subresource) {
        const std::scoped_lock lock(tracker->mutex());
        return tracker->stateLocked(subresource);
    };

    SECTION("a discard recorded before an earlier submission gets a prologue from that submission's state")
    {
        LegacyTextureStateRecording first;
        LegacyTextureStateRecording second;
        const auto firstUse = first.use(tracker);
        const auto secondUse = second.use(tracker);
        REQUIRE(first[firstUse].stateBefore(0, graphicsStates) == D3D12_RESOURCE_STATE_COMMON);
        REQUIRE(second[secondUse].stateBefore(0, graphicsStates) == D3D12_RESOURCE_STATE_COMMON);
        first[firstUse].transition(0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE, true);
        second[secondUse].transition(0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, true);
        REQUIRE(second[secondUse].stateBefore(0, graphicsStates) == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

        const std::array recordings{&std::as_const(first), &std::as_const(second)};
        const auto prologues = submit(recordings, graphicsStates);
        REQUIRE(prologues[0].empty());
        REQUIRE(prologues[1].size() == 1);
        CHECK(prologues[1][0].Transition.pResource == resource);
        CHECK(prologues[1][0].Transition.Subresource == 0);
        CHECK(prologues[1][0].Transition.StateBefore == D3D12_RESOURCE_STATE_COPY_SOURCE);
        CHECK(prologues[1][0].Transition.StateAfter == D3D12_RESOURCE_STATE_COMMON);
        CHECK(submittedState(0) == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        CHECK(submittedState(1) == D3D12_RESOURCE_STATE_COMMON);
    }

    SECTION("explicit first transitions and matching discards need no prologue")
    {
        LegacyTextureStateRecording seed;
        const auto seedUse = seed.use(tracker);
        seed[seedUse].transition(0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET, false);
        seed[seedUse].transition(1, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, true);
        const std::array seedRecordings{&std::as_const(seed)};
        REQUIRE(submit(seedRecordings, graphicsStates)[0].empty());

        LegacyTextureStateRecording next;
        const auto nextUse = next.use(tracker);
        const auto guess = next[nextUse].stateBefore(1, graphicsStates);
        REQUIRE(guess == D3D12_RESOURCE_STATE_COPY_DEST);
        next[nextUse].transition(0, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST, false);
        next[nextUse].transition(1, guess, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
        const std::array nextRecordings{&std::as_const(next)};
        CHECK(submit(nextRecordings, graphicsStates)[0].empty());
        CHECK(submittedState(0) == D3D12_RESOURCE_STATE_COPY_DEST);
        CHECK(submittedState(1) == D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    SECTION("a uniform prologue over every subresource collapses to one whole-resource transition")
    {
        LegacyTextureStateRecording seed;
        const auto seedUse = seed.use(tracker);
        seed[seedUse].transition(0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, false);
        seed[seedUse].transition(1, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, false);

        LegacyTextureStateRecording stale;
        const auto staleUse = stale.use(tracker);
        stale[staleUse].transition(0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET, true);
        stale[staleUse].transition(1, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET, true);

        const std::array recordings{&std::as_const(seed), &std::as_const(stale)};
        const auto prologues = submit(recordings, graphicsStates);
        REQUIRE(prologues[0].empty());
        REQUIRE(prologues[1].size() == 1);
        CHECK(prologues[1][0].Transition.Subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
        CHECK(prologues[1][0].Transition.StateBefore == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        CHECK(prologues[1][0].Transition.StateAfter == D3D12_RESOURCE_STATE_COMMON);
        CHECK(submittedState(0) == D3D12_RESOURCE_STATE_RENDER_TARGET);
        CHECK(submittedState(1) == D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

    SECTION("compute queues never start from or transition out of graphics-only states")
    {
        LegacyTextureStateRecording graphics;
        const auto graphicsUse = graphics.use(tracker);
        graphics[graphicsUse].transition(0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET, false);
        const std::array graphicsRecordings{&std::as_const(graphics)};
        REQUIRE(submit(graphicsRecordings, graphicsStates)[0].empty());

        LegacyTextureStateRecording compute;
        const auto computeUse = compute.use(tracker);
        const auto guess = compute[computeUse].stateBefore(0, computeStates);
        CHECK(guess == D3D12_RESOURCE_STATE_COMMON);
        compute[computeUse].transition(0, guess, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
        const std::array computeRecordings{&std::as_const(compute)};
        CHECK(submit(computeRecordings, computeStates)[0].empty());
        CHECK(submittedState(0) == D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
}

TEST_CASE("D3D12 legacy texture state indices grow geometrically as a command buffer meets new textures",
          "[size:small][rhi][d3d12][barrier]")
{
    if (!rachis::test::ThreadHeapAllocationCounter::available())
    {
        SKIP("Heap allocations are counted through the MSVC debug heap.");
    }
    constexpr std::size_t textureCount = 256;
    std::array<int, textureCount> nativeStorage{};
    std::vector<std::unique_ptr<LegacyTextureStateTracker>> trackers;
    trackers.reserve(textureCount);
    for (auto& storage : nativeStorage)
    {
        trackers.push_back(
            std::make_unique<LegacyTextureStateTracker>(reinterpret_cast<ID3D12Resource*>(&storage), 1U));
    }
    std::vector<std::size_t> values;
    values.reserve(textureCount);
    LegacyTextureStateIndex index;

    std::array<std::size_t, textureCount> appended{};
    std::array<std::size_t, textureCount> found{};
    rachis::test::ThreadHeapAllocationCounter counter;
    const auto allocations = counter.count([&] {
        for (std::size_t position = 0; position < textureCount; ++position)
        {
            appended[position] = index.findOrAppend(values, trackers[position].get(), [&] { return position; });
        }
        for (std::size_t position = 0; position < textureCount; ++position)
        {
            found[position] = index.findOrAppend(values, trackers[position].get(), [] { return std::size_t{0}; });
        }
    });
    REQUIRE(values.size() == textureCount);
    for (std::size_t position = 0; position < textureCount; ++position)
    {
        CHECK(appended[position] == position);
        CHECK(found[position] == position);
        CHECK(values[position] == position);
    }
    CHECK(allocations <= 10U);
}
