#include "D3D12/D3D12RootSignaturePolicy.h"
#include "D3D12/D3D12RuntimePolicies.h"
#include "D3D12/D3D12UploadArena.h"
#include "HeapAllocationCounter.h"
#include "Internal/TimelinePointAccess.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WINDOWS_
#error Portable D3D12 runtime policy unit tests must not include Windows.h.
#endif

namespace
{

    using namespace rachis::d3d12;

    [[nodiscard]] BindingDescriptorCounts makeDescriptorCounts(std::uint32_t uniformCount,
                                                               std::uint32_t storageCount = 0) noexcept
    {
        BindingDescriptorCounts counts{};
        counts[static_cast<std::size_t>(rachis::BindingType::uniformBuffer)] = uniformCount;
        counts[static_cast<std::size_t>(rachis::BindingType::storageBuffer)] = storageCount;
        return counts;
    }

} // namespace

TEST_CASE(
    "D3D12 pooled binding-set accounting supports 1-set 1-version idle updates while enforcing bound copy-on-write",
    "[size:small][rhi][d3d12][binding]")
{
    SECTION("1-set 1-version pool succeeds across repeated idle updateBindingSet calls")
    {
        auto poolState = std::make_shared<BindingPoolState>();
        poolState->maxBindingSets = 1;
        poolState->descriptorCapacity = makeDescriptorCounts(1, 1);

        auto setResult = acquireBindingPoolSet(poolState);
        REQUIRE(setResult.has_value());
        auto setLease = std::move(*setResult);
        CHECK(poolState->liveBindingSets == 1);

        auto secondSet = acquireBindingPoolSet(poolState);
        REQUIRE_FALSE(secondSet.has_value());
        CHECK(secondSet.error().code == rachis::ErrorCode::resourceExhausted);

        const auto oneVersionCounts = makeDescriptorCounts(1, 1);
        auto initialVersion = acquireBindingPoolVersion(setLease, oneVersionCounts);
        REQUIRE(initialVersion.has_value());
        auto currentVersion = std::move(*initialVersion);
        CHECK(poolState->liveDescriptors == oneVersionCounts);

        for (std::uint32_t iteration = 0; iteration < 5; ++iteration)
        {
            auto nextVersion = std::make_shared<BindingPoolVersionLease>(poolState, setLease, oneVersionCounts);
            REQUIRE(commitBindingPoolVersionUpdate(*setLease, currentVersion, *nextVersion).has_value());
            CHECK_FALSE(currentVersion->armed);
            CHECK(nextVersion->armed);
            currentVersion = std::move(nextVersion);
            CHECK(poolState->liveDescriptors == oneVersionCounts);
        }

        {
            auto oversizedVersion =
                std::make_shared<BindingPoolVersionLease>(poolState, setLease, makeDescriptorCounts(2, 1));
            auto oversizedCommit = commitBindingPoolVersionUpdate(*setLease, currentVersion, *oversizedVersion);
            REQUIRE_FALSE(oversizedCommit.has_value());
            CHECK(oversizedCommit.error().code == rachis::ErrorCode::resourceExhausted);
            CHECK(currentVersion->armed);
            CHECK_FALSE(oversizedVersion->armed);
            CHECK(poolState->liveDescriptors == oneVersionCounts);
        }

        currentVersion.reset();
        setLease.reset();
        CHECK(poolState->liveBindingSets == 0);
        CHECK(poolState->liveDescriptors == BindingDescriptorCounts{});
    }

    SECTION("active command retention enforces copy-on-write on 1-version and 2-version pools")
    {
        const auto singleVersionCounts = makeDescriptorCounts(1, 1);

        {
            auto oneVersionPool = std::make_shared<BindingPoolState>();
            oneVersionPool->maxBindingSets = 1;
            oneVersionPool->descriptorCapacity = singleVersionCounts;

            auto setLease = *acquireBindingPoolSet(oneVersionPool);
            auto currentVersion = *acquireBindingPoolVersion(setLease, singleVersionCounts);

            {
                BindingPoolCommandRetention retention;
                retention.arm(setLease, currentVersion);
                CHECK(setLease->activeCommandRetentions == 1);

                auto rejectedVersion =
                    std::make_shared<BindingPoolVersionLease>(oneVersionPool, setLease, singleVersionCounts);
                auto rejected = commitBindingPoolVersionUpdate(*setLease, currentVersion, *rejectedVersion);
                REQUIRE_FALSE(rejected.has_value());
                CHECK(rejected.error().code == rachis::ErrorCode::resourceExhausted);
                CHECK(currentVersion->armed);
                CHECK_FALSE(rejectedVersion->armed);
            }

            CHECK(setLease->activeCommandRetentions == 0);
            auto postRetireVersion =
                std::make_shared<BindingPoolVersionLease>(oneVersionPool, setLease, singleVersionCounts);
            REQUIRE(commitBindingPoolVersionUpdate(*setLease, currentVersion, *postRetireVersion).has_value());
            currentVersion = std::move(postRetireVersion);
            CHECK(oneVersionPool->liveDescriptors == singleVersionCounts);
        }

        {
            auto twoVersionPool = std::make_shared<BindingPoolState>();
            twoVersionPool->maxBindingSets = 1;
            twoVersionPool->descriptorCapacity = makeDescriptorCounts(2, 2);

            auto setLease = *acquireBindingPoolSet(twoVersionPool);
            auto currentVersion = *acquireBindingPoolVersion(setLease, singleVersionCounts);

            auto commandRetention = std::make_unique<BindingPoolCommandRetention>();
            commandRetention->arm(setLease, currentVersion);

            auto secondVersion =
                std::make_shared<BindingPoolVersionLease>(twoVersionPool, setLease, singleVersionCounts);
            REQUIRE(commitBindingPoolVersionUpdate(*setLease, currentVersion, *secondVersion).has_value());
            CHECK(currentVersion->armed);
            CHECK(secondVersion->armed);
            currentVersion = std::move(secondVersion);
            CHECK(twoVersionPool->liveDescriptors == makeDescriptorCounts(2, 2));

            auto thirdVersion =
                std::make_shared<BindingPoolVersionLease>(twoVersionPool, setLease, singleVersionCounts);
            auto exhausted = commitBindingPoolVersionUpdate(*setLease, currentVersion, *thirdVersion);
            REQUIRE_FALSE(exhausted.has_value());
            CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);
            CHECK(currentVersion->armed);
            CHECK_FALSE(thirdVersion->armed);

            commandRetention.reset();
            CHECK(setLease->activeCommandRetentions == 0);
            CHECK(twoVersionPool->liveDescriptors == singleVersionCounts);

            REQUIRE(commitBindingPoolVersionUpdate(*setLease, currentVersion, *thirdVersion).has_value());
            currentVersion = std::move(thirdVersion);
            CHECK(twoVersionPool->liveDescriptors == singleVersionCounts);
        }
    }

    SECTION("destroyed pool rejects updates while command retention keeps physical leases balanced")
    {
        auto poolState = std::make_shared<BindingPoolState>();
        poolState->maxBindingSets = 1;
        poolState->descriptorCapacity = makeDescriptorCounts(1, 0);

        const auto oneUniform = makeDescriptorCounts(1, 0);
        auto setLease = *acquireBindingPoolSet(poolState);
        auto currentVersion = *acquireBindingPoolVersion(setLease, oneUniform);

        auto commandRetention = std::make_unique<BindingPoolCommandRetention>();
        commandRetention->arm(setLease, currentVersion);

        {
            const std::scoped_lock lock(poolState->mutex);
            poolState->logicalAlive = false;
        }

        auto rejectedSet = acquireBindingPoolSet(poolState);
        REQUIRE_FALSE(rejectedSet.has_value());
        CHECK(rejectedSet.error().code == rachis::ErrorCode::invalidState);

        auto rejectedVersion = acquireBindingPoolVersion(setLease, oneUniform);
        REQUIRE_FALSE(rejectedVersion.has_value());
        CHECK(rejectedVersion.error().code == rachis::ErrorCode::invalidState);

        {
            auto updateCandidate = std::make_shared<BindingPoolVersionLease>(poolState, setLease, oneUniform);
            auto rejectedUpdate = commitBindingPoolVersionUpdate(*setLease, currentVersion, *updateCandidate);
            REQUIRE_FALSE(rejectedUpdate.has_value());
            CHECK(rejectedUpdate.error().code == rachis::ErrorCode::invalidState);
        }

        currentVersion.reset();
        setLease.reset();
        CHECK(poolState->liveBindingSets == 1);
        CHECK(poolState->liveDescriptors == oneUniform);

        commandRetention.reset();
        CHECK(poolState->liveBindingSets == 0);
        CHECK(poolState->liveDescriptors == BindingDescriptorCounts{});
    }

    SECTION("concurrent command retentions and version updates maintain exact descriptor balance")
    {
        auto poolState = std::make_shared<BindingPoolState>();
        poolState->maxBindingSets = 1;
        poolState->descriptorCapacity = makeDescriptorCounts(2, 0);

        const auto oneUniform = makeDescriptorCounts(1, 0);
        auto setLease = *acquireBindingPoolSet(poolState);
        auto currentVersion = *acquireBindingPoolVersion(setLease, oneUniform);
        std::mutex tableMutex;

        std::thread commandThread([&]() {
            for (std::size_t index = 0; index < 300; ++index)
            {
                BindingPoolCommandRetention retention;
                {
                    const std::scoped_lock tableLock(tableMutex);
                    retention.arm(setLease, currentVersion);
                }
            }
        });

        std::thread updateThread([&]() {
            for (std::size_t index = 0; index < 300; ++index)
            {
                auto candidate = std::make_shared<BindingPoolVersionLease>(poolState, setLease, oneUniform);
                std::shared_ptr<BindingPoolVersionLease> retired;
                const std::scoped_lock tableLock(tableMutex);
                if (commitBindingPoolVersionUpdate(*setLease, currentVersion, *candidate).has_value())
                {
                    retired = std::exchange(currentVersion, std::move(candidate));
                }
            }
        });

        commandThread.join();
        updateThread.join();

        CHECK(setLease->activeCommandRetentions == 0);
        CHECK(poolState->liveDescriptors == oneUniform);
        currentVersion.reset();
        setLease.reset();
        CHECK(poolState->liveBindingSets == 0);
        CHECK(poolState->liveDescriptors == BindingDescriptorCounts{});
    }

    SECTION("reverting a committed update restores the published version's accounting")
    {
        const auto oneVersionCounts = makeDescriptorCounts(1, 1);
        auto poolState = std::make_shared<BindingPoolState>();
        poolState->maxBindingSets = 1;
        poolState->descriptorCapacity = makeDescriptorCounts(2, 2);

        auto setLease = *acquireBindingPoolSet(poolState);
        auto currentVersion = *acquireBindingPoolVersion(setLease, oneVersionCounts);

        auto idleVersion = std::make_shared<BindingPoolVersionLease>(poolState, setLease, oneVersionCounts);
        REQUIRE(commitBindingPoolVersionUpdate(*setLease, currentVersion, *idleVersion).has_value());
        CHECK_FALSE(currentVersion->armed);
        revertBindingPoolVersionUpdate(*setLease, currentVersion, *idleVersion);
        CHECK(currentVersion->armed);
        CHECK_FALSE(idleVersion->armed);
        CHECK(poolState->liveDescriptors == oneVersionCounts);

        BindingPoolCommandRetention retention;
        retention.arm(setLease, currentVersion);
        auto retainedVersion = std::make_shared<BindingPoolVersionLease>(poolState, setLease, oneVersionCounts);
        REQUIRE(commitBindingPoolVersionUpdate(*setLease, currentVersion, *retainedVersion).has_value());
        CHECK(currentVersion->armed);
        CHECK(poolState->liveDescriptors == makeDescriptorCounts(2, 2));
        revertBindingPoolVersionUpdate(*setLease, currentVersion, *retainedVersion);
        CHECK(currentVersion->armed);
        CHECK_FALSE(retainedVersion->armed);
        CHECK(poolState->liveDescriptors == oneVersionCounts);

        revertBindingPoolVersionUpdate(*setLease, currentVersion, *retainedVersion);
        CHECK(poolState->liveDescriptors == oneVersionCounts);
    }
}

TEST_CASE("D3D12 waitable swapchain flags and frame-latency token state machine preserve semaphore balance",
          "[size:small][rhi][d3d12][swapchain]")
{
    constexpr int maxFrameLatency = 2;
    int semaphoreCount = maxFrameLatency;
    int inFlightPresents = 0;
    SwapchainLatencySlotState latencySlot;

    const auto checkTokenInvariant = [&]() {
        CHECK(semaphoreCount + inFlightPresents + (latencySlot.hasAcquiredSlot() ? 1 : 0) == maxFrameLatency);
    };

    CHECK_FALSE(latencySlot.needsWait(false));
    CHECK(latencySlot.needsWait(true));
    checkTokenInvariant();

    REQUIRE(latencySlot.needsWait(true));
    --semaphoreCount;
    latencySlot.onWaitSucceeded();
    checkTokenInvariant();

    latencySlot.onCancel();
    CHECK(latencySlot.hasAcquiredSlot());
    CHECK_FALSE(latencySlot.needsWait(true));
    checkTokenInvariant();

    latencySlot.onPresent(false);
    CHECK(latencySlot.hasAcquiredSlot());
    CHECK_FALSE(latencySlot.needsWait(true));
    checkTokenInvariant();

    latencySlot.onResize();
    CHECK(latencySlot.hasAcquiredSlot());
    checkTokenInvariant();

    ++inFlightPresents;
    latencySlot.onPresent(true);
    CHECK_FALSE(latencySlot.hasAcquiredSlot());
    CHECK(latencySlot.needsWait(true));
    checkTokenInvariant();

    REQUIRE(latencySlot.needsWait(true));
    --semaphoreCount;
    latencySlot.onWaitSucceeded();
    CHECK(semaphoreCount == 0);
    checkTokenInvariant();

    --inFlightPresents;
    ++semaphoreCount;
    checkTokenInvariant();

    latencySlot.reset();
    CHECK_FALSE(latencySlot.hasAcquiredSlot());
}

TEST_CASE("D3D12 root-signature planner promotes scalar dynamic uniform buffers within 64-DWORD budget and keeps "
          "storage buffers descriptor-backed",
          "[size:small][rhi][d3d12][root-signature]")
{
    SECTION("dynamic-CBV-only sets omit empty resource descriptor tables while dynamic storage buffers stay in tables")
    {
        const std::array set0Entries{
            RootSignatureSlotEntryInput{
                .entry = {0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
                .descriptorOffset = 0,
            },
            RootSignatureSlotEntryInput{
                .entry = {1, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
                .descriptorOffset = 1,
            },
        };
        const std::array set1Entries{
            RootSignatureSlotEntryInput{
                .entry = {0, rachis::BindingType::dynamicReadOnlyStorageBuffer, rachis::ShaderStage::compute, 1},
                .descriptorOffset = 0,
            },
            RootSignatureSlotEntryInput{
                .entry = {1, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute, 1},
                .descriptorOffset = 1,
            },
        };
        const std::array slots{
            RootSignatureSlotInput{.entries = set0Entries},
            RootSignatureSlotInput{.entries = set1Entries},
        };

        const auto plan = planRootSignature(slots, 0);
        CHECK(plan.pushConstantRootParameter == unpromotedRootParameter);

        CHECK(plan.totalRootDwords == 5U);
        CHECK(plan.totalRootParameters == 3U);
        REQUIRE(plan.promotedRootDescriptors.size() == 2U);

        CHECK(plan.slots[0].allDynamicBuffersPromoted());
        CHECK_FALSE(plan.slots[0].hasResourceDescriptorTable);
        CHECK(plan.slots[0].resourceRootParameter == unpromotedRootParameter);
        CHECK(plan.slots[0].dynamicRootParameters[0] == 0U);
        CHECK(plan.slots[0].dynamicRootParameters[1] == 1U);
        CHECK(plan.slots[0].dynamicRootKinds[0] == RootDescriptorKind::cbv);
        CHECK(plan.slots[0].dynamicRootKinds[1] == RootDescriptorKind::cbv);

        CHECK_FALSE(plan.slots[1].allDynamicBuffersPromoted());
        CHECK(plan.slots[1].promotedDynamicBufferCount == 0U);
        CHECK(plan.slots[1].hasResourceDescriptorTable);
        CHECK(plan.slots[1].resourceRootParameter == 2U);
        CHECK(plan.slots[1].dynamicRootParameters[0] == unpromotedRootParameter);
        CHECK(plan.slots[1].dynamicRootParameters[1] == unpromotedRootParameter);
        CHECK(plan.slots[1].dynamicRootKinds[0] == RootDescriptorKind::srv);
        CHECK(plan.slots[1].dynamicRootKinds[1] == RootDescriptorKind::uav);

        CHECK(plan.slots[0].dynamicTableDescriptorCount == 0U);
        CHECK(plan.slots[0].dynamicTableRootParameter == unpromotedRootParameter);
        CHECK(plan.slots[1].resourceTableIsPerBind());
        CHECK_FALSE(plan.slots[1].hasDynamicDescriptorTable);
        CHECK(plan.slots[1].dynamicTableDescriptorCount == 2U);
        CHECK(plan.slots[1].dynamicTableRootParameter == 2U);
        CHECK(plan.slots[1].dynamicTableOffsets[0] == 0U);
        CHECK(plan.slots[1].dynamicTableOffsets[1] == 1U);
    }

    SECTION(
        "mixed static resources, samplers, dynamic storage buffers, and dynamic CBV arrays preserve descriptor tables")
    {
        const std::array entries{
            RootSignatureSlotEntryInput{
                .entry = {0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::vertex, 2},
                .descriptorOffset = 0,
            },
            RootSignatureSlotEntryInput{
                .entry = {2, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::pixel, 1},
                .descriptorOffset = 2,
            },
            RootSignatureSlotEntryInput{
                .entry = {3, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::pixel, 1},
                .descriptorOffset = 3,
            },
            RootSignatureSlotEntryInput{
                .entry = {4, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 1},
                .descriptorOffset = 4,
            },
            RootSignatureSlotEntryInput{
                .entry = {0, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 1},
                .descriptorOffset = 0,
            },
        };
        const std::array slots{RootSignatureSlotInput{.entries = entries}};

        const auto plan = planRootSignature(slots, 16);
        CHECK(plan.pushConstantRootParameter == 0U);
        CHECK(plan.pushConstantDwords == 4U);

        CHECK(plan.totalRootDwords == 9U);
        CHECK(plan.totalRootParameters == 5U);
        REQUIRE(plan.promotedRootDescriptors.size() == 1U);
        CHECK(plan.promotedRootDescriptors[0].dynamicIndex == 2U);
        CHECK(plan.promotedRootDescriptors[0].shaderRegister == 2U);
        CHECK(plan.promotedRootDescriptors[0].rootParameter == 1U);

        CHECK_FALSE(plan.slots[0].allDynamicBuffersPromoted());
        CHECK(plan.slots[0].dynamicBufferCount == 4U);
        CHECK(plan.slots[0].promotedDynamicBufferCount == 1U);
        CHECK(plan.slots[0].dynamicRootParameters[0] == unpromotedRootParameter);
        CHECK(plan.slots[0].dynamicRootParameters[1] == unpromotedRootParameter);
        CHECK(plan.slots[0].dynamicRootParameters[2] == 1U);
        CHECK(plan.slots[0].dynamicRootParameters[3] == unpromotedRootParameter);
        CHECK(plan.slots[0].entryPromoted == std::vector<std::uint8_t>{0U, 1U, 0U, 0U, 0U});
        CHECK(plan.slots[0].resourceRootParameter == 2U);
        CHECK(plan.slots[0].samplerRootParameter == 3U);
        CHECK_FALSE(plan.slots[0].resourceTableIsPerBind());
        CHECK(plan.slots[0].hasDynamicDescriptorTable);
        CHECK(plan.slots[0].dynamicTableRootParameter == 4U);
        CHECK(plan.slots[0].dynamicTableDescriptorCount == 3U);
        CHECK(plan.slots[0].dynamicTableOffsets[0] == 0U);
        CHECK(plan.slots[0].dynamicTableOffsets[1] == 1U);
        CHECK(plan.slots[0].dynamicTableOffsets[2] == unpromotedRootParameter);
        CHECK(plan.slots[0].dynamicTableOffsets[3] == 2U);
    }

    SECTION("exact 64-DWORD budget boundary promotes maximum prefix and keeps overflow in descriptor table")
    {
        std::array<RootSignatureSlotEntryInput, 8> set0Dynamic{};
        std::array<RootSignatureSlotEntryInput, 8> set1Dynamic{};
        for (std::uint32_t index = 0; index < 8; ++index)
        {
            set0Dynamic[index] = {
                .entry = {index, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
                .descriptorOffset = index,
            };
            set1Dynamic[index] = {
                .entry = {index, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
                .descriptorOffset = index,
            };
        }

        const std::array exactFitSlots{
            RootSignatureSlotInput{.entries = set0Dynamic},
            RootSignatureSlotInput{.entries = set1Dynamic},
        };

        const auto exactPlan = planRootSignature(exactFitSlots, 128);
        CHECK(exactPlan.totalRootDwords == maxRootSignatureDwords);
        CHECK(exactPlan.promotedRootDescriptors.size() == 16U);
        CHECK(exactPlan.slots[0].allDynamicBuffersPromoted());
        CHECK(exactPlan.slots[1].allDynamicBuffersPromoted());
        CHECK_FALSE(exactPlan.slots[0].hasResourceDescriptorTable);
        CHECK_FALSE(exactPlan.slots[1].hasResourceDescriptorTable);

        std::vector<RootSignatureSlotEntryInput> set0WithTexture(set0Dynamic.begin(), set0Dynamic.end());
        set0WithTexture.push_back(RootSignatureSlotEntryInput{
            .entry = {8, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
            .descriptorOffset = 8,
        });
        const std::array overflowSlots{
            RootSignatureSlotInput{.entries = set0WithTexture},
            RootSignatureSlotInput{.entries = set1Dynamic},
        };
        const auto boundedPlan = planRootSignature(overflowSlots, 128);
        CHECK(boundedPlan.totalRootDwords == maxRootSignatureDwords);
        CHECK(boundedPlan.promotedRootDescriptors.size() == 15U);
        CHECK(boundedPlan.slots[0].allDynamicBuffersPromoted());
        CHECK(boundedPlan.slots[0].hasResourceDescriptorTable);
        CHECK_FALSE(boundedPlan.slots[1].allDynamicBuffersPromoted());
        CHECK(boundedPlan.slots[1].promotedDynamicBufferCount == 7U);
        CHECK(boundedPlan.slots[1].hasResourceDescriptorTable);
        CHECK(boundedPlan.slots[1].dynamicRootParameters[6] != unpromotedRootParameter);
        CHECK(boundedPlan.slots[1].dynamicRootParameters[7] == unpromotedRootParameter);
        CHECK_FALSE(boundedPlan.slots[0].hasDynamicDescriptorTable);
        CHECK(boundedPlan.slots[0].dynamicTableDescriptorCount == 0U);
        CHECK(boundedPlan.slots[1].resourceTableIsPerBind());
        CHECK(boundedPlan.slots[1].dynamicTableDescriptorCount == 1U);
        CHECK(boundedPlan.slots[1].dynamicTableOffsets[7] == 0U);
        CHECK(boundedPlan.slots[1].dynamicTableRootParameter == boundedPlan.slots[1].resourceRootParameter);
    }

    SECTION("static resources keep their table while unpromoted dynamic buffers get a separate per-bind table")
    {
        const std::array entries{
            RootSignatureSlotEntryInput{
                .entry = {0, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute, 1},
                .descriptorOffset = 0,
            },
            RootSignatureSlotEntryInput{
                .entry = {1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 15},
                .descriptorOffset = 1,
            },
        };
        const std::array slots{RootSignatureSlotInput{.entries = entries}};

        const auto plan = planRootSignature(slots, 0);
        CHECK(plan.totalRootDwords == 2U);
        CHECK(plan.totalRootParameters == 2U);
        CHECK(plan.promotedRootDescriptors.empty());
        CHECK(plan.slots[0].hasResourceDescriptorTable);
        CHECK(plan.slots[0].hasStaticResourceDescriptors);
        CHECK_FALSE(plan.slots[0].resourceTableIsPerBind());
        CHECK(plan.slots[0].resourceRootParameter == 0U);
        CHECK(plan.slots[0].hasDynamicDescriptorTable);
        CHECK(plan.slots[0].dynamicTableRootParameter == 1U);
        CHECK(plan.slots[0].dynamicTableDescriptorCount == 1U);
        CHECK(plan.slots[0].dynamicTableOffsets[0] == 0U);

        const std::array promotedEntries{
            RootSignatureSlotEntryInput{
                .entry = {0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
                .descriptorOffset = 0,
            },
            RootSignatureSlotEntryInput{
                .entry = {1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 15},
                .descriptorOffset = 1,
            },
        };
        const std::array promotedSlots{RootSignatureSlotInput{.entries = promotedEntries}};
        const auto promotedPlan = planRootSignature(promotedSlots, 0);
        CHECK(promotedPlan.totalRootDwords == 3U);
        CHECK(promotedPlan.totalRootParameters == 2U);
        CHECK(promotedPlan.slots[0].allDynamicBuffersPromoted());
        CHECK_FALSE(promotedPlan.slots[0].hasDynamicDescriptorTable);
        CHECK(promotedPlan.slots[0].dynamicTableDescriptorCount == 0U);
        CHECK(promotedPlan.slots[0].dynamicTableRootParameter == unpromotedRootParameter);
        CHECK(promotedPlan.slots[0].resourceRootParameter == 1U);
    }
}

TEST_CASE("D3D12 upload slab arena suballocates aligned slices, rewinds idle slabs, and recycles detached slabs",
          "[size:small][rhi][d3d12][upload-arena]")
{
    struct MockSlab final
    {
        std::uint64_t id = 0;
        std::shared_ptr<std::atomic_size_t> liveCount;

        MockSlab(std::uint64_t slabId, std::shared_ptr<std::atomic_size_t> counter) noexcept
            : id(slabId), liveCount(std::move(counter))
        {
            liveCount->fetch_add(1, std::memory_order_relaxed);
        }

        ~MockSlab()
        {
            if (liveCount != nullptr)
            {
                liveCount->fetch_sub(1, std::memory_order_relaxed);
            }
        }

        MockSlab(const MockSlab&) = delete;
        MockSlab& operator=(const MockSlab&) = delete;

        MockSlab(MockSlab&& other) noexcept : id(std::exchange(other.id, 0)), liveCount(std::move(other.liveCount))
        {
        }

        MockSlab& operator=(MockSlab&& other) noexcept
        {
            if (this != &other)
            {
                if (liveCount != nullptr)
                {
                    liveCount->fetch_sub(1, std::memory_order_relaxed);
                }
                id = std::exchange(other.id, 0);
                liveCount = std::move(other.liveCount);
            }
            return *this;
        }
    };

    auto liveSlabs = std::make_shared<std::atomic_size_t>(0);
    std::uint64_t nextSlabId = 1;
    const auto createSlab = [&](std::uint64_t capacity) -> rachis::Result<MockSlab> {
        CHECK(capacity == 1024U);
        return MockSlab{nextSlabId++, liveSlabs};
    };

    auto arena = UploadArena<MockSlab>::create(1024U, 2U);
    CHECK_FALSE(arena->allocate(0, 16, createSlab).has_value());
    CHECK_FALSE(arena->allocate(64, 24, createSlab).has_value());

    auto oversized = arena->allocate(2048U, 16U, createSlab);
    REQUIRE(oversized.has_value());
    CHECK(oversized->kind == UploadArenaAllocationKind::oversized);
    CHECK_FALSE(oversized->lease.valid());
    CHECK(liveSlabs->load() == 0);

    {
        auto first = arena->allocate(100U, 16U, createSlab);
        REQUIRE(first.has_value());
        REQUIRE(first->kind == UploadArenaAllocationKind::suballocated);
        CHECK(first->lease.payload().id == 1U);
        CHECK(first->lease.offset() == 0U);

        auto second = arena->allocate(64U, 16U, createSlab);
        REQUIRE(second.has_value());
        CHECK(second->lease.payload().id == 1U);
        CHECK(second->lease.offset() == 112U);

        auto third = arena->allocate(128U, 256U, createSlab);
        REQUIRE(third.has_value());
        CHECK(third->lease.payload().id == 1U);
        CHECK(third->lease.offset() == 256U);
        CHECK(arena->statistics().currentSlabActiveLeases == 3U);
    }
    CHECK(liveSlabs->load() == 1U);
    CHECK(arena->statistics().currentSlabActiveLeases == 0U);
    CHECK(arena->statistics().currentSlabCursor == 0U);

    auto inFlightA = arena->allocate(768U, 16U, createSlab);
    REQUIRE(inFlightA.has_value());
    CHECK(inFlightA->lease.payload().id == 1U);
    CHECK(inFlightA->lease.offset() == 0U);

    auto inFlightB = arena->allocate(512U, 16U, createSlab);
    REQUIRE(inFlightB.has_value());
    CHECK(inFlightB->lease.payload().id == 2U);

    auto inFlightC = arena->allocate(600U, 16U, createSlab);
    REQUIRE(inFlightC.has_value());
    CHECK(inFlightC->lease.payload().id == 3U);

    auto inFlightD = arena->allocate(600U, 16U, createSlab);
    REQUIRE(inFlightD.has_value());
    CHECK(inFlightD->lease.payload().id == 4U);
    CHECK(liveSlabs->load() == 4U);

    inFlightB->lease.release();
    inFlightA->lease.release();
    inFlightC->lease.release();
    CHECK(arena->statistics().freeSlabCount == 2U);
    CHECK(liveSlabs->load() == 3U);

    auto reused = arena->allocate(600U, 16U, createSlab);
    REQUIRE(reused.has_value());
    CHECK(reused->lease.payload().id <= 3U);
    CHECK(nextSlabId == 5U);
    CHECK(arena->statistics().freeSlabCount == 1U);

    arena->shutdown();
    CHECK(liveSlabs->load() == 2U);
    inFlightD->lease.release();
    reused->lease.release();
    CHECK(liveSlabs->load() == 0U);
}

TEST_CASE("D3D12 upload slab arena rejects alignments larger than one slab", "[size:small][rhi][d3d12][upload-arena]")
{
    std::size_t createdSlabs = 0;
    const auto createSlab = [&](std::uint64_t capacity) -> rachis::Result<std::uint64_t> {
        ++createdSlabs;
        return capacity;
    };

    auto arena = UploadArena<std::uint64_t>::create(1024U, 2U);
    auto overAligned = arena->allocate(64U, 2048U, createSlab);
    REQUIRE_FALSE(overAligned.has_value());
    CHECK(overAligned.error().code == rachis::ErrorCode::invalidArgument);
    auto overAlignedOversized = arena->allocate(4096U, 2048U, createSlab);
    REQUIRE_FALSE(overAlignedOversized.has_value());
    CHECK(overAlignedOversized.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(createdSlabs == 0U);
    CHECK_FALSE(arena->statistics().hasCurrentSlab);

    auto slabAligned = arena->allocate(64U, 1024U, createSlab);
    REQUIRE(slabAligned.has_value());
    REQUIRE(slabAligned->kind == UploadArenaAllocationKind::suballocated);
    CHECK(slabAligned->lease.offset() == 0U);
    auto stillOverAligned = arena->allocate(64U, 2048U, createSlab);
    REQUIRE_FALSE(stillOverAligned.has_value());
    CHECK(stillOverAligned.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(createdSlabs == 1U);
    CHECK(arena->statistics().currentSlabActiveLeases == 1U);

    slabAligned->lease.release();
    arena->shutdown();
}

TEST_CASE("D3D12 SmallVector and movable BindingPoolCommandRetention preserve RAII invariants",
          "[size:small][rhi][d3d12][small-vector]")
{
    struct CountedItem final
    {
        int value = 0;
        std::atomic_size_t* live = nullptr;

        CountedItem(int v, std::atomic_size_t* counter) noexcept : value(v), live(counter)
        {
            live->fetch_add(1, std::memory_order_relaxed);
        }

        ~CountedItem()
        {
            if (live != nullptr)
            {
                live->fetch_sub(1, std::memory_order_relaxed);
            }
        }

        CountedItem(const CountedItem&) = delete;
        CountedItem& operator=(const CountedItem&) = delete;

        CountedItem(CountedItem&& other) noexcept : value(other.value), live(std::exchange(other.live, nullptr))
        {
        }

        CountedItem& operator=(CountedItem&& other) noexcept
        {
            if (this != &other)
            {
                if (live != nullptr)
                {
                    live->fetch_sub(1, std::memory_order_relaxed);
                }
                value = other.value;
                live = std::exchange(other.live, nullptr);
            }
            return *this;
        }
    };

    std::atomic_size_t liveItems = 0;
    {
        SmallVector<CountedItem, 4> items;
        CHECK(items.usingInlineStorage());
        items.emplace_back(10, &liveItems);
        items.emplace_back(20, &liveItems);
        items.emplace_back(30, &liveItems);
        items.emplace_back(40, &liveItems);
        CHECK(items.usingInlineStorage());
        CHECK(items.size() == 4U);
        CHECK(liveItems.load() == 4U);

        SmallVector<CountedItem, 4> moved(std::move(items));
        CHECK(moved.usingInlineStorage());
        CHECK(moved.size() == 4U);
        CHECK(liveItems.load() == 4U);

        moved.emplace_back(50, &liveItems);
        CHECK_FALSE(moved.usingInlineStorage());
        CHECK(moved.size() == 5U);
        CHECK(moved[4].value == 50);
        CHECK(liveItems.load() == 5U);

        moved.truncate(2U);
        CHECK(moved.size() == 2U);
        CHECK(liveItems.load() == 2U);
    }
    CHECK(liveItems.load() == 0U);

    auto poolState = std::make_shared<BindingPoolState>();
    poolState->maxBindingSets = 1;
    poolState->descriptorCapacity = {1, 0, 0, 0};
    auto setLease = acquireBindingPoolSet(poolState);
    REQUIRE(setLease.has_value());
    auto versionLease = acquireBindingPoolVersion(*setLease, {1, 0, 0, 0});
    REQUIRE(versionLease.has_value());

    {
        BindingPoolCommandRetention first;
        first.arm(*setLease, *versionLease);
        CHECK(first.armed());
        CHECK((*setLease)->activeCommandRetentions == 1U);

        BindingPoolCommandRetention second(std::move(first));
        CHECK_FALSE(first.armed());
        CHECK(second.armed());
        CHECK((*setLease)->activeCommandRetentions == 1U);

        BindingPoolCommandRetention third;
        third = std::move(second);
        CHECK_FALSE(second.armed());
        CHECK(third.armed());
        CHECK((*setLease)->activeCommandRetentions == 1U);
    }
    CHECK((*setLease)->activeCommandRetentions == 0U);
}

namespace
{

    struct MarkedItem final
    {
        int value = 0;

        explicit MarkedItem(int v) noexcept : value(v)
        {
        }

        MarkedItem(const MarkedItem&) noexcept = default;
        MarkedItem& operator=(const MarkedItem&) noexcept = default;

        MarkedItem(MarkedItem&& other) noexcept : value(std::exchange(other.value, -1))
        {
        }

        MarkedItem& operator=(MarkedItem&& other) noexcept
        {
            value = std::exchange(other.value, -1);
            return *this;
        }

        ~MarkedItem()
        {
            value = -2;
        }
    };

    struct ThrowBudget final
    {
        std::size_t live = 0;
        std::size_t copiesBeforeThrow = std::numeric_limits<std::size_t>::max();
    };

    struct ThrowingCopyItem final
    {
        int value = 0;
        ThrowBudget* budget = nullptr;

        ThrowingCopyItem(int v, ThrowBudget* owner) noexcept : value(v), budget(owner)
        {
            ++budget->live;
        }

        ThrowingCopyItem(const ThrowingCopyItem& other) : value(other.value), budget(other.budget)
        {
            if (budget->copiesBeforeThrow == 0)
            {
                throw std::runtime_error("SmallVector test copy budget exhausted.");
            }
            --budget->copiesBeforeThrow;
            ++budget->live;
        }

        ThrowingCopyItem(ThrowingCopyItem&& other) : ThrowingCopyItem(std::as_const(other))
        {
        }

        ThrowingCopyItem& operator=(const ThrowingCopyItem&) = default;
        ThrowingCopyItem& operator=(ThrowingCopyItem&&) = default;

        ~ThrowingCopyItem()
        {
            --budget->live;
        }
    };

} // namespace

TEST_CASE("D3D12 SmallVector exposes inline storage through data on empty and non-empty vectors",
          "[size:small][rhi][d3d12][small-vector]")
{
    SmallVector<MarkedItem, 2> items;
    const auto& constItems = items;
    MarkedItem* const emptyData = items.data();
    CHECK(emptyData != nullptr);
    CHECK(constItems.data() == emptyData);
    CHECK(items.begin() == items.end());
    CHECK(constItems.begin() == constItems.end());
    CHECK(items.span().empty());
    CHECK(constItems.span().data() == emptyData);

    items.emplace_back(5);
    CHECK(items.data() == emptyData);
    CHECK(&items[0] == emptyData);
    CHECK(items.end() == emptyData + 1);
    CHECK(constItems.back().value == 5);

    items.clear();
    CHECK(items.empty());
    CHECK(items.data() == emptyData);
    CHECK(items.begin() == items.end());
}

TEST_CASE("D3D12 SmallVector reads self-referential append arguments before spilling inline elements",
          "[size:small][rhi][d3d12][small-vector]")
{
    SECTION("copying push_back of an inline element at full capacity")
    {
        SmallVector<MarkedItem, 2> items;
        items.emplace_back(10);
        items.emplace_back(20);
        REQUIRE(items.usingInlineStorage());

        items.push_back(items[0]);
        CHECK_FALSE(items.usingInlineStorage());
        REQUIRE(items.size() == 3U);
        CHECK(items[0].value == 10);
        CHECK(items[1].value == 20);
        CHECK(items[2].value == 10);
    }

    SECTION("moving push_back of an inline element at full capacity")
    {
        SmallVector<MarkedItem, 2> items;
        items.emplace_back(30);
        items.emplace_back(40);
        REQUIRE(items.usingInlineStorage());

        items.push_back(std::move(items[1]));
        CHECK_FALSE(items.usingInlineStorage());
        REQUIRE(items.size() == 3U);
        CHECK(items[0].value == 30);
        CHECK(items[1].value == -1);
        CHECK(items[2].value == 40);
    }

    SECTION("emplace_back from a member of an inline element at full capacity")
    {
        SmallVector<MarkedItem, 2> items;
        items.emplace_back(50);
        items.emplace_back(60);
        REQUIRE(items.usingInlineStorage());

        const MarkedItem& emplaced = items.emplace_back(items[0].value);
        CHECK_FALSE(items.usingInlineStorage());
        REQUIRE(items.size() == 3U);
        CHECK(&emplaced == &items.back());
        CHECK(items[0].value == 50);
        CHECK(items[1].value == 60);
        CHECK(items[2].value == 50);
    }
}

TEST_CASE("D3D12 SmallVector destroys partially constructed inline copies when an element constructor throws",
          "[size:small][rhi][d3d12][small-vector]")
{
    using ThrowingVector = SmallVector<ThrowingCopyItem, 4>;

    ThrowBudget budget;
    {
        ThrowingVector source;
        source.emplace_back(1, &budget);
        source.emplace_back(2, &budget);
        source.emplace_back(3, &budget);
        REQUIRE(source.usingInlineStorage());
        REQUIRE(budget.live == 3U);

        budget.copiesBeforeThrow = 2U;
        CHECK_THROWS_AS(ThrowingVector(source), std::runtime_error);
        CHECK(budget.live == 3U);

        budget.copiesBeforeThrow = 1U;
        CHECK_THROWS_AS(ThrowingVector(std::move(source)), std::runtime_error);
        CHECK(budget.live == 3U);
        REQUIRE(source.size() == 3U);
        CHECK(source[2].value == 3);

        budget.copiesBeforeThrow = std::numeric_limits<std::size_t>::max();
        ThrowingVector target;
        target.emplace_back(7, &budget);
        REQUIRE(budget.live == 4U);

        budget.copiesBeforeThrow = 2U;
        CHECK_THROWS_AS(target = source, std::runtime_error);
        CHECK(budget.live == 4U);
        REQUIRE(target.size() == 1U);
        CHECK(target[0].value == 7);

        budget.copiesBeforeThrow = 1U;
        CHECK_THROWS_AS(target = std::move(source), std::runtime_error);
        CHECK(target.empty());
        CHECK(budget.live == 3U);
        REQUIRE(source.size() == 3U);

        budget.copiesBeforeThrow = std::numeric_limits<std::size_t>::max();
        source.emplace_back(4, &budget);
        REQUIRE(source.usingInlineStorage());
        REQUIRE(budget.live == 4U);

        budget.copiesBeforeThrow = 2U;
        CHECK_THROWS_AS(source.push_back(source[0]), std::runtime_error);
        CHECK(source.usingInlineStorage());
        REQUIRE(source.size() == 4U);
        CHECK(source[0].value == 1);
        CHECK(source[3].value == 4);
        CHECK(budget.live == 4U);

        budget.copiesBeforeThrow = std::numeric_limits<std::size_t>::max();
        ThrowingVector copied(source);
        CHECK(copied.size() == 4U);
        CHECK(budget.live == 8U);
    }
    CHECK(budget.live == 0U);
}

TEST_CASE("D3D12 SmallVector allocates only when it spills past its inline capacity",
          "[size:small][rhi][d3d12][small-vector]")
{
    if (!rachis::test::ThreadHeapAllocationCounter::available())
    {
        SKIP("Heap allocations are counted through the MSVC debug heap.");
    }
    rachis::test::ThreadHeapAllocationCounter counter;
    const auto inlineAllocations = counter.count([] {
        SmallVector<std::shared_ptr<int>, 2> values;
        values.push_back(nullptr);
        values.push_back(nullptr);
        SmallVector<std::shared_ptr<int>, 2> moved(std::move(values));
        moved.truncate(1U);
        moved.clear();
    });
    CHECK(inlineAllocations == 0U);

    const auto spilledAllocations = counter.count([] {
        SmallVector<std::shared_ptr<int>, 2> values;
        values.reserve(3U);
        values.push_back(nullptr);
        values.push_back(nullptr);
        values.push_back(nullptr);
        CHECK_FALSE(values.usingInlineStorage());
        CHECK(values.size() == 3U);
    });
    CHECK(spilledAllocations != 0U);
}

TEST_CASE("D3D12 batch conflict scans agree with a pairwise scan for batches of any size",
          "[size:small][rhi][d3d12][batch]")
{
    struct Box final
    {
        std::uint64_t group = 0;
        std::uint64_t x = 0;
        std::uint64_t width = 0;
        std::uint64_t y = 0;
        std::uint64_t height = 0;
        std::uint64_t layer = 0;
    };
    const auto spansMeet =
        [](std::uint64_t leftBegin, std::uint64_t leftSize, std::uint64_t rightBegin, std::uint64_t rightSize) {
            return leftBegin < rightBegin + rightSize && rightBegin < leftBegin + leftSize;
        };
    const auto boxesOverlap = [&](const Box& left, const Box& right) {
        return left.group == right.group && left.layer == right.layer &&
               spansMeet(left.x, left.width, right.x, right.width) &&
               spansMeet(left.y, left.height, right.y, right.height);
    };

    std::uint64_t state = 0x2545F4914F6CDD1DULL;
    const auto next = [&](std::uint64_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (state >> 33U) % bound;
    };
    const auto randomSize = [&] { return next(4) == 0 ? next(33) : next(9); };
    const auto randomBox = [&] {
        return Box{
            .group = next(3),
            .x = next(32),
            .width = randomSize(),
            .y = next(32),
            .height = randomSize(),
            .layer = next(2),
        };
    };
    const auto boxMeeting = [&](const Box& box) {
        if (box.width == 0 || box.height == 0)
        {
            return box;
        }
        return Box{
            .group = box.group,
            .x = box.x + next(box.width),
            .width = 1 + next(8),
            .y = box.y + next(box.height),
            .height = 1 + next(8),
            .layer = box.layer,
        };
    };

    for (std::size_t count = 0; count <= 40; ++count)
    {
        for (std::size_t round = 0; round < 32; ++round)
        {
            const bool oneAxis = round % 3U == 0U;
            const bool packed = round % 2U == 1U;
            const bool injected = round % 4U == 1U;
            CAPTURE(count, round, oneAxis, packed, injected);
            std::vector<std::uint64_t> keys(count);
            std::vector<Box> sources(count);
            std::vector<Box> destinations(count);
            const auto overlapsAny = [&](const Box& box, std::span<const Box> others) {
                return std::ranges::any_of(others, [&](const Box& other) { return boxesOverlap(box, other); });
            };
            for (std::size_t index = 0; index < count; ++index)
            {
                keys[index] = next(count * 2U + 1U);
                destinations[index] = randomBox();
                for (std::size_t attempt = 0;
                     packed && attempt < 32 && overlapsAny(destinations[index], std::span(destinations).first(index));
                     ++attempt)
                {
                    destinations[index] = randomBox();
                }
            }
            for (auto& source : sources)
            {
                source = randomBox();
                for (std::size_t attempt = 0; packed && attempt < 32 && overlapsAny(source, destinations); ++attempt)
                {
                    source = randomBox();
                }
            }
            if (injected && count >= 2U)
            {
                const auto later = 1U + next(count - 1U);
                destinations[later] = boxMeeting(destinations[next(later)]);
                sources[next(count)] = boxMeeting(destinations[next(count)]);
            }

            std::size_t expectedRepeat = noBatchConflict;
            bool expectedOverlap = false;
            bool expectedCross = false;
            for (std::size_t later = 0; later < count; ++later)
            {
                for (std::size_t earlier = 0; earlier < later; ++earlier)
                {
                    if (expectedRepeat == noBatchConflict && keys[earlier] == keys[later])
                    {
                        expectedRepeat = later;
                    }
                    expectedOverlap = expectedOverlap || boxesOverlap(destinations[earlier], destinations[later]);
                }
                for (std::size_t other = 0; other < count; ++other)
                {
                    expectedCross = expectedCross || boxesOverlap(sources[later], destinations[other]);
                }
            }

            const auto sweepKey = [&](const Box& box) {
                BatchSweepKey key{.group = box.group, .primary = {.begin = box.x, .end = box.x + box.width}};
                if (!oneAxis)
                {
                    key.secondary = {.begin = box.y, .end = box.y + box.height};
                }
                return key;
            };
            bool orderedPairs = true;
            CHECK(firstRepeatedKeyIndex(count, [&](std::size_t index) { return keys[index]; }) == expectedRepeat);
            CHECK(anyOverlappingPair(
                      count,
                      [&](std::size_t index) { return sweepKey(destinations[index]); },
                      [&](std::size_t earlier, std::size_t later) {
                          orderedPairs = orderedPairs && earlier < later && later < count;
                          return boxesOverlap(destinations[earlier], destinations[later]);
                      }) == expectedOverlap);
            CHECK(anyCrossOverlap(
                      count,
                      [&](std::size_t index) { return sweepKey(sources[index]); },
                      [&](std::size_t index) { return sweepKey(destinations[index]); },
                      [&](std::size_t source, std::size_t destination) {
                          orderedPairs = orderedPairs && source < count && destination < count;
                          return boxesOverlap(sources[source], destinations[destination]);
                      }) == expectedCross);
            CHECK(orderedPairs);
        }
    }
}

TEST_CASE("D3D12 batch conflict scans test only entries whose keys meet in large batches",
          "[size:small][rhi][d3d12][batch]")
{
    constexpr std::size_t count = 4096;
    constexpr std::uint32_t width = 4096;
    std::size_t comparisons = 0;
    const auto countedOverlap = [&](std::size_t, std::size_t) {
        ++comparisons;
        return false;
    };
    const auto overlapTests = [&](auto&& keyOf) {
        comparisons = 0;
        CHECK_FALSE(anyOverlappingPair(count, keyOf, countedOverlap));
        return comparisons;
    };
    constexpr std::size_t sweepStepLimit = count * 2U;
    const auto sweepSteps = [&](auto&& keyOf) {
        BatchSweepSteps steps;
        CHECK_FALSE(anyOverlappingPair(count, keyOf, countedOverlap, &steps));
        return steps.blocks + steps.entries;
    };
    const auto crossSweepSteps = [&](auto&& leftKeyOf, auto&& rightKeyOf) {
        BatchSweepSteps steps;
        CHECK_FALSE(anyCrossOverlap(count, leftKeyOf, rightKeyOf, countedOverlap, &steps));
        return steps.blocks + steps.entries;
    };

    CHECK(firstRepeatedKeyIndex(count, [](std::size_t index) { return std::uint64_t{index}; }) == noBatchConflict);
    CHECK(firstRepeatedKeyIndex(count, [](std::size_t index) { return std::uint64_t{index % (count - 1U)}; }) ==
          count - 1U);

    const auto disjoint = [](std::size_t index) {
        return BatchSweepKey{.group = index % 4U, .primary = {.begin = index * 2U, .end = index * 2U + 1U}};
    };
    CHECK(overlapTests(disjoint) == 0U);
    CHECK(sweepSteps(disjoint) <= sweepStepLimit);
    const auto stacked = [](std::size_t index) {
        return BatchSweepKey{.primary = {.begin = index, .end = index + 2U}};
    };
    CHECK(overlapTests(stacked) == count - 1U);
    CHECK(sweepSteps(stacked) <= sweepStepLimit);

    const auto rowStrips = [&](std::size_t index) {
        return textureBoxSweepKey(
            0, {.y = static_cast<std::uint32_t>(index)}, {.width = width, .height = 1, .depth = 1});
    };
    const auto columnStrips = [&](std::size_t index) {
        return textureBoxSweepKey(
            0, {.x = static_cast<std::uint32_t>(index)}, {.width = 1, .height = width, .depth = 1});
    };
    const auto tileColumn = [](std::size_t index) {
        return textureBoxSweepKey(
            0, {.y = static_cast<std::uint32_t>(index * 16U)}, {.width = 16, .height = 16, .depth = 1});
    };
    const auto atlasTiles = [](std::size_t index) {
        return textureBoxSweepKey(
            0,
            {.x = static_cast<std::uint32_t>(index % 64U * 16U), .y = static_cast<std::uint32_t>(index / 64U * 16U)},
            {.width = 16, .height = 16, .depth = 1});
    };
    const auto volumeSlices = [&](std::size_t index) {
        return textureBoxSweepKey(0, {.z = static_cast<std::uint32_t>(index)}, {.width = 64, .height = 64, .depth = 1});
    };
    const auto rowRunsFrom = [&](std::uint32_t firstRow) {
        return [=](std::size_t index) {
            const auto row = static_cast<std::uint32_t>(index);
            return textureBoxSweepKey(
                0, {.x = row, .y = firstRow + row}, {.width = width - row, .height = 1, .depth = 1});
        };
    };
    const auto columnRuns = [&](std::size_t index) {
        const auto column = static_cast<std::uint32_t>(index);
        return textureBoxSweepKey(0, {.x = column, .y = column}, {.width = 1, .height = width - column, .depth = 1});
    };
    CHECK(overlapTests(rowStrips) == 0U);
    CHECK(overlapTests(columnStrips) == 0U);
    CHECK(overlapTests(tileColumn) == 0U);
    CHECK(overlapTests(atlasTiles) == 0U);
    CHECK(overlapTests(volumeSlices) == 0U);
    CHECK(overlapTests(rowRunsFrom(0)) == 0U);
    CHECK(overlapTests(columnRuns) == 0U);
    CHECK(sweepSteps(rowStrips) <= sweepStepLimit);
    CHECK(sweepSteps(columnStrips) <= sweepStepLimit);
    CHECK(sweepSteps(tileColumn) <= sweepStepLimit);
    CHECK(sweepSteps(atlasTiles) <= sweepStepLimit);
    CHECK(sweepSteps(volumeSlices) <= sweepStepLimit);
    CHECK(sweepSteps(rowRunsFrom(0)) <= sweepStepLimit);
    CHECK(sweepSteps(columnRuns) <= sweepStepLimit);

    constexpr std::uint64_t rowPitch = 64U * 512U;
    const auto rowReadbacks = [&](std::size_t index) {
        const std::uint64_t offset = index * rowPitch;
        return linearFootprintSweepKey(offset,
                                       {
                                           .bytesPerRow = static_cast<std::uint32_t>(rowPitch),
                                           .rowsPerImage = 1,
                                           .rowCount = 1,
                                           .rowDataSize = rowPitch,
                                           .bytesPerImage = rowPitch,
                                           .requiredBufferSize = offset + rowPitch,
                                       },
                                       rowPitch);
    };
    const auto tileReadbacks = [&](std::size_t index) {
        constexpr std::uint64_t tileRows = 128;
        const std::uint64_t offset = index / 64U * tileRows * rowPitch + index % 64U * 512U;
        return linearFootprintSweepKey(offset,
                                       {
                                           .bytesPerRow = static_cast<std::uint32_t>(rowPitch),
                                           .rowsPerImage = static_cast<std::uint32_t>(tileRows),
                                           .rowCount = static_cast<std::uint32_t>(tileRows),
                                           .rowDataSize = 512,
                                           .bytesPerImage = tileRows * rowPitch,
                                           .requiredBufferSize = offset + (tileRows - 1U) * rowPitch + 512U,
                                       },
                                       rowPitch);
    };
    CHECK(overlapTests(rowReadbacks) == 0U);
    CHECK(overlapTests(tileReadbacks) == 0U);
    CHECK(sweepSteps(rowReadbacks) <= sweepStepLimit);
    CHECK(sweepSteps(tileReadbacks) <= sweepStepLimit);

    const auto subresourceBarriers = [](std::size_t index) {
        const auto layer = static_cast<std::uint32_t>(index / 8U);
        const auto mip = static_cast<std::uint32_t>(index % 8U);
        return BatchSweepKey{
            .group = 7,
            .primary = {.begin = layer, .end = layer + 1U},
            .secondary = layeredBatchSweepInterval(0, 1, mip, 1),
        };
    };
    const auto mipBarriers = [](std::size_t index) {
        const auto plane = static_cast<std::uint32_t>(index % 2U);
        const auto mip = static_cast<std::uint32_t>(index / 2U);
        return BatchSweepKey{
            .group = 7,
            .primary = {.begin = 0, .end = 2048},
            .secondary = layeredBatchSweepInterval(plane, 1, mip, 1),
        };
    };
    CHECK(overlapTests(subresourceBarriers) == 0U);
    CHECK(overlapTests(mipBarriers) == 0U);
    CHECK(sweepSteps(subresourceBarriers) <= sweepStepLimit);
    CHECK(sweepSteps(mipBarriers) <= sweepStepLimit);

    comparisons = 0;
    CHECK(anyOverlappingPair(
        count,
        [&](std::size_t index) { return rowStrips(index == count - 1U ? 17U : index); },
        [&](std::size_t, std::size_t) {
            ++comparisons;
            return true;
        }));
    CHECK(comparisons == 1U);

    const auto shiftedRowStrips = [&](std::size_t index) { return rowStrips(index + count); };
    const auto sharedSource = [&](std::size_t) { return rowStrips(count * 2U); };
    comparisons = 0;
    CHECK_FALSE(anyCrossOverlap(count, rowStrips, shiftedRowStrips, countedOverlap));
    CHECK(comparisons == 0U);
    comparisons = 0;
    CHECK_FALSE(anyCrossOverlap(count, sharedSource, rowStrips, countedOverlap));
    CHECK(comparisons == 0U);
    comparisons = 0;
    CHECK_FALSE(anyCrossOverlap(count, disjoint, disjoint, countedOverlap));
    CHECK(comparisons == count);
    comparisons = 0;
    CHECK(crossSweepSteps(rowStrips, shiftedRowStrips) <= sweepStepLimit * 2U);
    CHECK(crossSweepSteps(rowRunsFrom(0), rowRunsFrom(static_cast<std::uint32_t>(count))) <= sweepStepLimit * 2U);
    CHECK(comparisons == 0U);
}

TEST_CASE("D3D12 manual residency readiness preserves earliest threshold across same-queue and transitive cross-queue "
          "submissions",
          "[size:small][rhi][d3d12][residency]")
{
    constexpr std::uint64_t kDeviceToken = 0xD3D12ULL;
    const auto makePoint = [&](rachis::QueueType queue, std::uint64_t value) constexpr noexcept {
        return rachis::detail::TimelinePointAccess::make(queue, value, kDeviceToken);
    };

    std::array<std::uint64_t, kD3D12ResidencyQueueCount> completed{0, 0, 0};
    const auto queryCompleted = [&](std::size_t index) noexcept { return completed[index]; };

    auto state = rachis::ResidencyState::pending;
    D3D12ResidencyReadyThresholds readyAfter{3, 0, 0};

    REQUIRE(
        validateResidencySubmission(state, readyAfter, rachis::QueueType::graphics, {}, queryCompleted).has_value());
    propagateResidencySubmission(state, readyAfter, rachis::QueueType::graphics, 4, queryCompleted);
    CHECK(readyAfter == D3D12ResidencyReadyThresholds{3, 0, 0});

    REQUIRE(
        validateResidencySubmission(state, readyAfter, rachis::QueueType::graphics, {}, queryCompleted).has_value());
    propagateResidencySubmission(state, readyAfter, rachis::QueueType::graphics, 7, queryCompleted);
    CHECK(readyAfter == D3D12ResidencyReadyThresholds{3, 0, 0});

    auto missingWait = validateResidencySubmission(state, readyAfter, rachis::QueueType::compute, {}, queryCompleted);
    REQUIRE_FALSE(missingWait.has_value());
    CHECK(missingWait.error().code == rachis::ErrorCode::invalidState);

    const std::array insufficientWaits{makePoint(rachis::QueueType::graphics, 2)};
    auto insufficientWait =
        validateResidencySubmission(state, readyAfter, rachis::QueueType::compute, insufficientWaits, queryCompleted);
    REQUIRE_FALSE(insufficientWait.has_value());
    CHECK(insufficientWait.error().code == rachis::ErrorCode::invalidState);

    const std::array exactMakeResidentWait{makePoint(rachis::QueueType::graphics, 3)};
    REQUIRE(validateResidencySubmission(
                state, readyAfter, rachis::QueueType::compute, exactMakeResidentWait, queryCompleted)
                .has_value());
    propagateResidencySubmission(state, readyAfter, rachis::QueueType::compute, 5, queryCompleted);
    CHECK(readyAfter == D3D12ResidencyReadyThresholds{3, 5, 0});

    completed[d3d12ResidencyQueueIndex(rachis::QueueType::graphics)] = 2;
    CHECK_FALSE(promoteCompletedResidency(state, readyAfter, queryCompleted));
    CHECK(state == rachis::ResidencyState::pending);

    completed[d3d12ResidencyQueueIndex(rachis::QueueType::graphics)] = 3;
    CHECK(promoteCompletedResidency(state, readyAfter, queryCompleted));
    CHECK(state == rachis::ResidencyState::resident);
    CHECK(readyAfter == D3D12ResidencyReadyThresholds{0, 0, 0});

    state = rachis::ResidencyState::evicted;
    readyAfter = {};
    auto evictedUse = validateResidencySubmission(state, readyAfter, rachis::QueueType::graphics, {}, queryCompleted);
    REQUIRE_FALSE(evictedUse.has_value());
    CHECK(evictedUse.error().code == rachis::ErrorCode::invalidState);

    completed = {8, 20, 30};
    state = rachis::ResidencyState::pending;
    readyAfter = {10, 0, 0};

    const std::array graphicsTenWait{makePoint(rachis::QueueType::graphics, 10)};
    REQUIRE(validateResidencySubmission(state, readyAfter, rachis::QueueType::compute, graphicsTenWait, queryCompleted)
                .has_value());
    propagateResidencySubmission(state, readyAfter, rachis::QueueType::compute, 21, queryCompleted);
    CHECK(readyAfter == D3D12ResidencyReadyThresholds{10, 21, 0});

    REQUIRE(validateResidencySubmission(state, readyAfter, rachis::QueueType::compute, {}, queryCompleted).has_value());
    propagateResidencySubmission(state, readyAfter, rachis::QueueType::compute, 25, queryCompleted);
    CHECK(readyAfter == D3D12ResidencyReadyThresholds{10, 21, 0});

    const std::array computeTooEarly{makePoint(rachis::QueueType::compute, 20)};
    auto copyRejected =
        validateResidencySubmission(state, readyAfter, rachis::QueueType::copy, computeTooEarly, queryCompleted);
    REQUIRE_FALSE(copyRejected.has_value());
    CHECK(copyRejected.error().code == rachis::ErrorCode::invalidState);

    const std::array computePropagatedWait{makePoint(rachis::QueueType::compute, 21)};
    REQUIRE(
        validateResidencySubmission(state, readyAfter, rachis::QueueType::copy, computePropagatedWait, queryCompleted)
            .has_value());
    propagateResidencySubmission(state, readyAfter, rachis::QueueType::copy, 31, queryCompleted);
    CHECK(readyAfter == D3D12ResidencyReadyThresholds{10, 21, 31});

    completed[d3d12ResidencyQueueIndex(rachis::QueueType::graphics)] = kD3D12DeviceRemovedFenceValue;
    CHECK_FALSE(promoteCompletedResidency(state, readyAfter, queryCompleted));
    CHECK(state == rachis::ResidencyState::pending);
    CHECK(readyAfter == D3D12ResidencyReadyThresholds{10, 21, 31});

    completed[d3d12ResidencyQueueIndex(rachis::QueueType::graphics)] = 10;
    completed[d3d12ResidencyQueueIndex(rachis::QueueType::compute)] = 21;
    CHECK(promoteCompletedResidency(state, readyAfter, queryCompleted));
    CHECK(state == rachis::ResidencyState::resident);
    CHECK(readyAfter == D3D12ResidencyReadyThresholds{0, 0, 0});
}

namespace
{

    struct SimulatedResidencyLease final
    {
        explicit SimulatedResidencyLease(std::atomic_size_t* activeUseCount = nullptr) noexcept
            : useCount(activeUseCount)
        {
            if (useCount != nullptr)
            {
                useCount->fetch_add(1, std::memory_order_relaxed);
            }
        }

        ~SimulatedResidencyLease()
        {
            if (useCount != nullptr)
            {
                useCount->fetch_sub(1, std::memory_order_relaxed);
            }
        }

        SimulatedResidencyLease(const SimulatedResidencyLease&) = delete;
        SimulatedResidencyLease& operator=(const SimulatedResidencyLease&) = delete;

        SimulatedResidencyLease(SimulatedResidencyLease&& other) noexcept
            : useCount(std::exchange(other.useCount, nullptr))
        {
        }

        SimulatedResidencyLease& operator=(SimulatedResidencyLease&& other) noexcept
        {
            if (this != &other)
            {
                if (useCount != nullptr)
                {
                    useCount->fetch_sub(1, std::memory_order_relaxed);
                }
                useCount = std::exchange(other.useCount, nullptr);
            }
            return *this;
        }

        std::atomic_size_t* useCount = nullptr;
    };

    struct SimulatedCommandLease;

    struct SimulatedPoolEntry final
    {
        bool commandListOpen = false;
        bool allocatorClean = false;
        std::uint64_t recordingId = 0;
        SimulatedCommandLease* lease = nullptr;
        std::vector<std::shared_ptr<int>> retainedResources;
        std::vector<SimulatedResidencyLease> residencyLeases;
        std::size_t closeCalls = 0;
        std::size_t allocatorResetCalls = 0;
        std::size_t commandListResetCalls = 0;
        std::int32_t nextCloseHr = 0;
        std::int32_t nextAllocatorResetHr = 0;
        std::int32_t nextCommandListResetHr = 0;

        void clearRetentions() noexcept
        {
            retainedResources.clear();
            residencyLeases.clear();
            recordingId = 0;
        }

        [[nodiscard]] std::int32_t closeCommandList() noexcept
        {
            ++closeCalls;
            const auto hr = std::exchange(nextCloseHr, 0);
            if (hr < 0)
            {
                commandListOpen = false;
                return hr;
            }
            if (!commandListOpen)
            {
                return static_cast<std::int32_t>(0x80004005U);
            }
            commandListOpen = false;
            return 0;
        }

        [[nodiscard]] std::int32_t resetAllocator() noexcept
        {
            ++allocatorResetCalls;
            return std::exchange(nextAllocatorResetHr, 0);
        }

        [[nodiscard]] std::int32_t resetCommandList() noexcept
        {
            ++commandListResetCalls;
            const auto hr = std::exchange(nextCommandListResetHr, 0);
            if (hr < 0)
            {
                return hr;
            }
            commandListOpen = true;
            return 0;
        }
    };

    struct SimulatedCommandPool final
    {
        explicit SimulatedCommandPool(std::size_t capacity)
        {
            entries.reserve(capacity);
            available.reserve(capacity);
            quarantined.reserve(capacity);
            for (std::size_t index = 0; index < capacity; ++index)
            {
                auto entry = std::make_unique<SimulatedPoolEntry>();
                available.push_back(entry.get());
                entries.push_back(std::move(entry));
            }
        }

        [[nodiscard]] std::size_t recordingCount() const noexcept
        {
            std::size_t count = 0;
            for (const auto& entry : entries)
            {
                if (entry->lease != nullptr)
                {
                    ++count;
                }
            }
            return count;
        }

        [[nodiscard]] bool checkIdlePartition() const noexcept
        {
            return isD3D12CommandPoolIdleFromInFlight(
                entries.size(), available.size(), quarantined.size(), recordingCount());
        }

        std::vector<std::unique_ptr<SimulatedPoolEntry>> entries;
        BoundedFifoQueue<SimulatedPoolEntry*> available;
        std::vector<SimulatedPoolEntry*> quarantined;
        std::uint64_t nextRecordingId = 1;
    };

    struct SimulatedCommandLease final
    {
        SimulatedCommandLease() noexcept = default;

        SimulatedCommandLease(std::shared_ptr<SimulatedCommandPool> pool, SimulatedPoolEntry& target) noexcept
            : owner(std::move(pool)), entry(&target)
        {
            entry->recordingId = owner->nextRecordingId++;
            entry->lease = this;
        }

        ~SimulatedCommandLease()
        {
            abandon();
        }

        SimulatedCommandLease(const SimulatedCommandLease&) = delete;
        SimulatedCommandLease& operator=(const SimulatedCommandLease&) = delete;

        void recordDummyCommand(const std::shared_ptr<int>& resource, std::atomic_size_t* residencyUseCount)
        {
            hasFrontendBoundPipeline = true;
            entry->retainedResources.push_back(resource);
            entry->residencyLeases.emplace_back(residencyUseCount);
        }

        [[nodiscard]] rachis::Status resetRecording()
        {
            if (owner == nullptr || entry == nullptr || entry->lease != this)
            {
                return std::unexpected(rachis::Error{
                    rachis::ErrorCode::invalidState,
                    0,
                    "The command buffer does not hold a live command-pool recording lease.",
                });
            }

            auto* const target = entry;
            auto keepPoolAlive = owner;
            return executeD3D12RecordingReset(
                [&]() noexcept { return target->closeCommandList(); },
                [&]() noexcept { return target->resetAllocator(); },
                [&]() noexcept { return target->resetCommandList(); },
                [&]() noexcept { target->clearRetentions(); },
                [&](bool clean) noexcept { target->allocatorClean = clean; },
                [&]() noexcept { target->recordingId = keepPoolAlive->nextRecordingId++; },
                [&]() noexcept {
                    target->lease = nullptr;
                    owner->quarantined.push_back(target);
                    entry = nullptr;
                    owner.reset();
                },
                [&]() noexcept {
                    hasFrontendBoundPipeline = false;
                    ++frontendResetCount;
                });
        }

        void abandon() noexcept
        {
            if (owner == nullptr || entry == nullptr)
            {
                return;
            }
            auto pool = std::move(owner);
            auto* target = std::exchange(entry, nullptr);
            const auto closeResult = target->closeCommandList();
            target->clearRetentions();
            target->lease = nullptr;
            target->allocatorClean = false;
            if (closeResult >= 0)
            {
                pool->available.push_back(target);
            }
            else
            {
                pool->quarantined.push_back(target);
            }
        }

        std::shared_ptr<SimulatedCommandPool> owner;
        SimulatedPoolEntry* entry = nullptr;
        bool hasFrontendBoundPipeline = false;
        std::size_t frontendResetCount = 0;
    };

    [[nodiscard]] rachis::Result<std::unique_ptr<SimulatedCommandLease>> acquireSimulatedLease(
        const std::shared_ptr<SimulatedCommandPool>& pool)
    {
        auto reopened = acquireAndReopenD3D12PoolEntry(
            pool->available,
            pool->quarantined,
            [](SimulatedPoolEntry& candidate) noexcept { candidate.clearRetentions(); },
            [](const SimulatedPoolEntry& candidate) noexcept { return candidate.allocatorClean; },
            [](SimulatedPoolEntry& candidate, bool clean) noexcept { candidate.allocatorClean = clean; },
            [](SimulatedPoolEntry& candidate) noexcept { return candidate.resetAllocator(); },
            [](SimulatedPoolEntry& candidate) noexcept { return candidate.resetCommandList(); });
        if (!reopened)
        {
            return std::unexpected(std::move(reopened.error()));
        }
        return std::make_unique<SimulatedCommandLease>(pool, **reopened);
    }

    [[nodiscard]] rachis::Status resetSimulatedCommandPool(SimulatedCommandPool& pool)
    {
        if (!pool.checkIdlePartition())
        {
            return std::unexpected(rachis::Error{
                rachis::ErrorCode::invalidState,
                0,
                "A command pool cannot be reset while one of its entries is in flight.",
            });
        }
        for (const auto& entry : pool.entries)
        {
            if (entry->lease != nullptr)
            {
                auto reset = entry->lease->resetRecording();
                if (!reset)
                {
                    return reset;
                }
            }
        }
        return resetD3D12IdleCommandPoolEntries(
            pool.available,
            pool.quarantined,
            [](const SimulatedPoolEntry& idleEntry) noexcept { return idleEntry.allocatorClean; },
            [](SimulatedPoolEntry& idleEntry, bool clean) noexcept { idleEntry.allocatorClean = clean; },
            [](SimulatedPoolEntry& idleEntry) noexcept { return idleEntry.resetAllocator(); });
    }

} // namespace

TEST_CASE(
    "D3D12 command-buffer recording reset quarantines failed entries and detaches leases across all failure points",
    "[size:small][rhi][d3d12][command-pool][reset]")
{
    enum class FailureStage : std::uint8_t
    {
        closeCommandList,
        resetAllocator,
        resetCommandList,
    };

    struct FailureScenario final
    {
        FailureStage stage;
        std::int32_t hresult;
        rachis::ErrorCode expectedCode;
    };

    constexpr auto kGenericFail = static_cast<std::int32_t>(0x80004005U);
    constexpr std::array scenarios{
        FailureScenario{FailureStage::closeCommandList, kGenericFail, rachis::ErrorCode::validationError},
        FailureScenario{FailureStage::closeCommandList, kD3D12HresultOutOfMemory, rachis::ErrorCode::outOfMemory},
        FailureScenario{FailureStage::closeCommandList, kDxgiErrorDeviceRemoved, rachis::ErrorCode::deviceLost},
        FailureScenario{FailureStage::closeCommandList, kDxgiErrorDeviceReset, rachis::ErrorCode::deviceLost},
        FailureScenario{FailureStage::closeCommandList, kDxgiErrorDriverInternalError, rachis::ErrorCode::deviceLost},
        FailureScenario{FailureStage::resetAllocator, kGenericFail, rachis::ErrorCode::nativeFailure},
        FailureScenario{FailureStage::resetAllocator, kD3D12HresultOutOfMemory, rachis::ErrorCode::outOfMemory},
        FailureScenario{FailureStage::resetAllocator, kDxgiErrorDeviceRemoved, rachis::ErrorCode::deviceLost},
        FailureScenario{FailureStage::resetCommandList, kGenericFail, rachis::ErrorCode::nativeFailure},
        FailureScenario{FailureStage::resetCommandList, kD3D12HresultOutOfMemory, rachis::ErrorCode::outOfMemory},
        FailureScenario{FailureStage::resetCommandList, kDxgiErrorDeviceReset, rachis::ErrorCode::deviceLost},
    };

    for (const auto& scenario : scenarios)
    {
        auto pool = std::make_shared<SimulatedCommandPool>(2);
        auto resource = std::make_shared<int>(42);
        std::atomic_size_t residencyUseCount{0};

        auto firstLease = acquireSimulatedLease(pool);
        REQUIRE(firstLease.has_value());
        auto* failedEntry = (*firstLease)->entry;
        REQUIRE(failedEntry != nullptr);
        CHECK(failedEntry->commandListOpen);
        CHECK(failedEntry->recordingId != 0);

        (*firstLease)->recordDummyCommand(resource, &residencyUseCount);
        CHECK(resource.use_count() == 2);
        CHECK(residencyUseCount.load() == 1U);
        CHECK((*firstLease)->hasFrontendBoundPipeline);

        switch (scenario.stage)
        {
        case FailureStage::closeCommandList:
            failedEntry->nextCloseHr = scenario.hresult;
            break;
        case FailureStage::resetAllocator:
            failedEntry->nextAllocatorResetHr = scenario.hresult;
            break;
        case FailureStage::resetCommandList:
            failedEntry->nextCommandListResetHr = scenario.hresult;
            break;
        }

        const auto closeCallsBefore = failedEntry->closeCalls;
        auto resetStatus = (*firstLease)->resetRecording();
        REQUIRE_FALSE(resetStatus.has_value());
        CHECK(resetStatus.error().code == scenario.expectedCode);
        CHECK(resetStatus.error().nativeCode == static_cast<std::int64_t>(scenario.hresult));

        CHECK(resource.use_count() == 1);
        CHECK(residencyUseCount.load() == 0U);
        CHECK_FALSE((*firstLease)->hasFrontendBoundPipeline);
        CHECK((*firstLease)->frontendResetCount == 1U);
        CHECK((*firstLease)->entry == nullptr);
        CHECK((*firstLease)->owner == nullptr);

        CHECK(failedEntry->lease == nullptr);
        CHECK(failedEntry->recordingId == 0U);
        CHECK_FALSE(failedEntry->allocatorClean);
        CHECK_FALSE(failedEntry->commandListOpen);
        CHECK(pool->quarantined.size() == 1U);
        CHECK(pool->quarantined.front() == failedEntry);
        CHECK(pool->available.size() == 1U);
        CHECK(pool->checkIdlePartition());

        const auto closeCallsAfterFailure = failedEntry->closeCalls;
        CHECK(closeCallsAfterFailure == closeCallsBefore + 1U);
        auto secondResetAttempt = (*firstLease)->resetRecording();
        REQUIRE_FALSE(secondResetAttempt.has_value());
        CHECK(secondResetAttempt.error().code == rachis::ErrorCode::invalidState);
        CHECK(failedEntry->closeCalls == closeCallsAfterFailure);

        firstLease->reset();
        CHECK(failedEntry->closeCalls == closeCallsAfterFailure);
        CHECK(pool->quarantined.size() == 1U);
        CHECK(pool->available.size() == 1U);

        auto healthyLease = acquireSimulatedLease(pool);
        REQUIRE(healthyLease.has_value());
        CHECK((*healthyLease)->entry != failedEntry);
        CHECK((*healthyLease)->entry->commandListOpen);
        CHECK((*healthyLease)->entry->recordingId != 0U);

        auto exhausted = acquireSimulatedLease(pool);
        REQUIRE_FALSE(exhausted.has_value());
        CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);

        REQUIRE((*healthyLease)->resetRecording().has_value());
        CHECK((*healthyLease)->entry->commandListOpen);
        CHECK((*healthyLease)->entry->recordingId != 0U);
        REQUIRE(resetSimulatedCommandPool(*pool).has_value());
        CHECK(pool->checkIdlePartition());
    }
}

TEST_CASE("D3D12 pool-wide reset and acquisition preserve pool partition invariants across mid-batch native failures",
          "[size:small][rhi][d3d12][command-pool][reset]")
{
    auto pool = std::make_shared<SimulatedCommandPool>(4);
    auto resourceA = std::make_shared<int>(1);
    auto resourceB = std::make_shared<int>(2);
    std::atomic_size_t residencyUses{0};

    auto lease0 = acquireSimulatedLease(pool);
    auto lease1 = acquireSimulatedLease(pool);
    REQUIRE(lease0.has_value());
    REQUIRE(lease1.has_value());
    (*lease0)->recordDummyCommand(resourceA, &residencyUses);
    (*lease1)->recordDummyCommand(resourceB, &residencyUses);
    CHECK(residencyUses.load() == 2U);
    CHECK(pool->available.size() == 2U);
    CHECK(pool->recordingCount() == 2U);
    CHECK(pool->checkIdlePartition());

    auto* entry0 = (*lease0)->entry;
    auto* entry1 = (*lease1)->entry;
    const auto previousRecordingId0 = entry0->recordingId;
    entry1->nextCommandListResetHr = kD3D12HresultOutOfMemory;

    auto poolReset1 = resetSimulatedCommandPool(*pool);
    REQUIRE_FALSE(poolReset1.has_value());
    CHECK(poolReset1.error().code == rachis::ErrorCode::outOfMemory);

    CHECK((*lease0)->entry == entry0);
    CHECK(entry0->lease == lease0->get());
    CHECK(entry0->commandListOpen);
    CHECK(entry0->recordingId > previousRecordingId0);
    CHECK(resourceA.use_count() == 1);

    CHECK((*lease1)->entry == nullptr);
    CHECK(entry1->lease == nullptr);
    CHECK(entry1->recordingId == 0U);
    CHECK(resourceB.use_count() == 1);
    CHECK(residencyUses.load() == 0U);

    CHECK(pool->recordingCount() == 1U);
    CHECK(pool->quarantined.size() == 1U);
    CHECK(pool->available.size() == 2U);
    CHECK(pool->checkIdlePartition());

    auto* idleFirst = pool->available.front();
    pool->available.pop_front();
    auto* idleSecond = pool->available.front();
    pool->available.pop_front();
    idleFirst->allocatorClean = false;
    idleSecond->allocatorClean = false;
    idleSecond->nextAllocatorResetHr = kDxgiErrorDriverInternalError;
    pool->available.push_back(idleFirst);
    pool->available.push_back(idleSecond);

    auto poolReset2 = resetSimulatedCommandPool(*pool);
    REQUIRE_FALSE(poolReset2.has_value());
    CHECK(poolReset2.error().code == rachis::ErrorCode::deviceLost);

    CHECK(idleFirst->allocatorClean);
    CHECK_FALSE(idleSecond->allocatorClean);
    CHECK(pool->recordingCount() == 1U);
    CHECK(pool->quarantined.size() == 2U);
    CHECK(pool->available.size() == 1U);
    CHECK(pool->checkIdlePartition());

    idleFirst->allocatorClean = false;
    idleFirst->nextAllocatorResetHr = kD3D12HresultOutOfMemory;
    auto failedAcquire = acquireSimulatedLease(pool);
    REQUIRE_FALSE(failedAcquire.has_value());
    CHECK(failedAcquire.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(pool->available.empty());
    CHECK(pool->quarantined.size() == 3U);
    CHECK(pool->recordingCount() == 1U);
    CHECK(pool->checkIdlePartition());
}
