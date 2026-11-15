#include "Internal/VulkanQueues.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
#ifdef DeviceCapabilities
#undef DeviceCapabilities
#endif

#include "Common/VectorReserve.h"
#include "Internal/TimelinePointAccess.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanResources.h"
#include "Rachis/Interop.h"

#include <bit>
#ifdef RACHIS_TESTING
#include "Internal/CommandPoolRetirementTestHook.h"
#include "Vulkan/VulkanTestHooks.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rachis::vulkan
{
    namespace
    {

        constexpr std::size_t queueCount = 3;

        [[nodiscard]] bool presentWaitsEnqueued(VkResult result) noexcept
        {
            return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR ||
                   result == VK_ERROR_SURFACE_LOST_KHR || result == VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT;
        }
#ifdef RACHIS_TESTING
        constexpr std::int32_t dxgiErrorDeviceRemoved = static_cast<std::int32_t>(0x887A0005);
        constexpr std::int32_t dxgiErrorDeviceReset = static_cast<std::int32_t>(0x887A0007);

        [[nodiscard]] std::optional<VkResult> consumeInjectedPresentResult() noexcept
        {
            const auto injected = test::nextPresentResult().exchange(0, std::memory_order_relaxed);
            if (injected == 0)
            {
                return std::nullopt;
            }
            if (injected == dxgiErrorDeviceRemoved || injected == dxgiErrorDeviceReset)
            {
                return VK_ERROR_DEVICE_LOST;
            }
            return static_cast<VkResult>(injected);
        }
#endif

        [[nodiscard]] constexpr bool isValidQueueType(QueueType type) noexcept
        {
            switch (type)
            {
            case QueueType::graphics:
            case QueueType::compute:
            case QueueType::copy:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr std::size_t queueIndex(QueueType type) noexcept
        {
            return static_cast<std::size_t>(type);
        }

        [[nodiscard]] constexpr const char* timelineName(QueueType type) noexcept
        {
            switch (type)
            {
            case QueueType::graphics:
                return "RHI.graphics.timeline";
            case QueueType::compute:
                return "RHI.compute.timeline";
            case QueueType::copy:
                return "RHI.copy.timeline";
            }
            return "RHI.queue.timeline";
        }

#ifdef _WIN32
        constexpr std::uint64_t exportSignalPollNanoseconds = 100'000'000;

        struct PendingTimelineExport final
        {
            VkSemaphore semaphore = VK_NULL_HANDLE;
            std::size_t queue = 0;
            std::uint64_t value = 0;
        };
#endif

        struct ImportedSemaphoreKeepAlive final
        {
            ImportedSemaphoreKeepAlive() = default;
            ImportedSemaphoreKeepAlive(const ImportedSemaphoreKeepAlive&) = delete;
            ImportedSemaphoreKeepAlive& operator=(const ImportedSemaphoreKeepAlive&) = delete;
            ImportedSemaphoreKeepAlive(ImportedSemaphoreKeepAlive&&) = delete;
            ImportedSemaphoreKeepAlive& operator=(ImportedSemaphoreKeepAlive&&) = delete;
            VkDevice device = VK_NULL_HANDLE;
            const VolkDeviceTable* table = nullptr;
            VkSemaphore semaphore = VK_NULL_HANDLE;

            ~ImportedSemaphoreKeepAlive()
            {
                if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroySemaphore != nullptr &&
                    semaphore != VK_NULL_HANDLE)
                {
                    table->vkDestroySemaphore(device, semaphore, nullptr);
                }
            }
        };

        struct VulkanNativeQueueGate final
        {
            std::mutex mutex;
            VkQueue queue = VK_NULL_HANDLE;
        };

        struct PendingQueryPoolReset final
        {
            VkQueryPool pool = VK_NULL_HANDLE;
            VkQueryPool primitivesPool = VK_NULL_HANDLE;
            std::uint32_t firstQuery = 0;
            std::uint32_t queryCount = 1;
            std::shared_ptr<void> keepAlive;
        };

        struct VulkanPendingEntry final
        {
            std::uint64_t timelineValue = std::numeric_limits<std::uint64_t>::max();
            VkFence fence = VK_NULL_HANDLE;
            std::shared_ptr<VulkanCommandPoolState> pool = {};
            VulkanCommandEntry* entry = nullptr;
            std::vector<VkCommandBuffer> bridgeBuffers = {};
            std::vector<std::shared_ptr<void>> retained = {};
            std::vector<VulkanAccelerationStructurePublication> accelerationStructurePublications = {};
        };

        struct VulkanQueueState final
        {
            VulkanNativeQueueGate* gate = nullptr;
            VkSemaphore timeline = VK_NULL_HANDLE;
            QueueType type = QueueType::graphics;
            std::uint32_t family = VK_QUEUE_FAMILY_IGNORED;
            std::mutex bridgePoolMutex;
            VkCommandPool bridgePool = VK_NULL_HANDLE;
            std::vector<VkCommandBuffer> availableBridgeBuffers;
            std::size_t totalBridgeBuffers = 0;
            std::atomic<std::uint64_t> lastSubmittedValue{0};
            std::uint64_t nextValue = 1;
            std::uint64_t unchainedSharedWaitValue = 0;
            mutable std::mutex retirementMutex;
            std::deque<VulkanPendingEntry> pending;
            std::vector<VkFence> availableFences;
            std::size_t totalFences = 0;
        };

        [[nodiscard]] bool hasUnreservedTimelineSlot(const VulkanQueueState& state) noexcept
        {
            return state.nextValue != 0 && state.nextValue != std::numeric_limits<std::uint64_t>::max();
        }

        [[nodiscard]] std::size_t sharedWaitChainCount(const VulkanQueueState& state) noexcept
        {
            return state.unchainedSharedWaitValue != 0 ? 1U : 0U;
        }

        [[nodiscard]] std::chrono::nanoseconds remainingTimeout(std::chrono::steady_clock::time_point started,
                                                                std::chrono::nanoseconds timeout) noexcept
        {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started);
            return elapsed >= timeout ? std::chrono::nanoseconds::zero() : timeout - elapsed;
        }

        [[nodiscard]] Status recordingReady(const VulkanCommandBuffer& native)
        {
            if (native.graphicsState().renderingOwner != nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A Vulkan command buffer cannot be submitted while a rendering scope is active.",
                });
            }
            if (native.queryState().activeOcclusion.has_value() ||
                native.queryState().activePipelineStatistics.has_value() ||
                std::ranges::any_of(native.queryState().activeStreamOutputStatistics,
                                    [](const auto& active) { return active.has_value(); }) ||
                native.queryState().activeGpuProfileScopeDepth != 0 ||
                native.queryState().activeConditionalExecution.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A Vulkan command buffer cannot be submitted while a query, profile, or conditional scope is "
                    "active.",
                });
            }
            if (native.debugState().groupDepth != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A Vulkan command buffer cannot be submitted while a debug group is open.",
                });
            }
            return {};
        }

        [[nodiscard]] VkSemaphoreSubmitInfo semaphoreSubmitInfo(VkSemaphore semaphore,
                                                                std::uint64_t timelineValue = {}) noexcept
        {
            return VkSemaphoreSubmitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = {},
                .semaphore = semaphore,
                .value = timelineValue,
                .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .deviceIndex = {},
            };
        }

        [[nodiscard]] VkCommandBufferSubmitInfo commandBufferSubmitInfo(VkCommandBuffer commandBuffer) noexcept
        {
            return VkCommandBufferSubmitInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                .pNext = {},
                .commandBuffer = commandBuffer,
                .deviceMask = {},
            };
        }

        struct ValidatedBatchCommands final
        {
            VulkanQueueState* queueState = nullptr;
            std::vector<VulkanCommandBuffer*> nativeCommands;
        };

        struct SubmissionBatch final
        {
            std::vector<VulkanCommandBuffer*> nativeCommands;
            std::vector<std::shared_ptr<VulkanCommandPoolState>> pools;
            std::vector<VulkanCommandEntry*> entries;
            std::vector<VkCommandBuffer> rawCommands;
            std::vector<VulkanResourceAllocator::CommandLayoutBridgePlan> layoutPlans;
        };

        [[nodiscard]] VulkanResourceAllocator* firstResourceAllocator(
            std::span<const std::shared_ptr<VulkanCommandPoolState>> pools) noexcept
        {
            for (const auto& pool : pools)
            {
                if (pool != nullptr && pool->resources != nullptr)
                {
                    return pool->resources;
                }
            }
            return nullptr;
        }

        [[nodiscard]] Result<SubmissionBatch> gatherSubmissionBatch(std::vector<VulkanCommandBuffer*> nativeCommands)
        {
            SubmissionBatch batch;
            batch.nativeCommands = std::move(nativeCommands);
            try
            {
                batch.pools.reserve(batch.nativeCommands.size());
                batch.entries.reserve(batch.nativeCommands.size());
                batch.rawCommands.reserve(batch.nativeCommands.size());
                for (auto* native : batch.nativeCommands)
                {
                    batch.pools.push_back(native->ownerShared());
                    batch.entries.push_back(native->entry());
                    batch.rawCommands.push_back(native->native());
                }

                auto* batchResources = firstResourceAllocator(batch.pools);
                if (batchResources != nullptr)
                {
                    auto planned = batchResources->planBatchLayoutTransitions(batch.rawCommands);
                    if (!planned)
                    {
                        return std::unexpected(std::move(planned.error()));
                    }
                    batch.layoutPlans = std::move(*planned);
                }
                else
                {
                    batch.layoutPlans.resize(batch.nativeCommands.size());
                }
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate Vulkan submission batch storage.",
                });
            }
            return batch;
        }

        struct SimulatedQueryPoolDirty final
        {
            std::shared_ptr<VulkanSubmittedQueryPoolState> poolState;
            std::vector<std::uint8_t> dirty;
            std::vector<std::uint8_t> touched;
            std::uint32_t minTouchedSlot = 0;
            std::uint32_t maxTouchedSlot = 0;
            bool hasTouchedSlots = false;
        };

        struct QueryPoolResetPlan final
        {
            std::vector<std::vector<PendingQueryPoolReset>> preResets;
            std::vector<SimulatedQueryPoolDirty> simulatedPools;
        };

        [[nodiscard]] SimulatedQueryPoolDirty& findOrSnapshotQueryPool(
            std::vector<SimulatedQueryPoolDirty>& simulatedPools, const VulkanCommandQueryPoolUse& poolUse)
        {
            for (auto& existing : simulatedPools)
            {
                if (existing.poolState.get() == poolUse.poolState.get())
                {
                    return existing;
                }
            }
            std::vector<std::uint8_t> dirtySnapshot;
            {
                const std::scoped_lock poolStateLock(poolUse.poolState->mutex);
                dirtySnapshot = poolUse.poolState->dirtySlots;
            }
            std::vector<std::uint8_t> touchedMask(dirtySnapshot.size(), 0U);
            simulatedPools.push_back(SimulatedQueryPoolDirty{
                .poolState = poolUse.poolState,
                .dirty = std::move(dirtySnapshot),
                .touched = std::move(touchedMask),
            });
            return simulatedPools.back();
        }

        void appendQueryPoolReset(std::vector<PendingQueryPoolReset>& resets,
                                  const VulkanCommandQueryPoolUse& poolUse,
                                  std::uint32_t queryIndex)
        {
            const bool extendsLastReset = !resets.empty() && resets.back().pool == poolUse.poolState->pool &&
                                          resets.back().primitivesPool == poolUse.poolState->primitivesPool &&
                                          resets.back().firstQuery + resets.back().queryCount == queryIndex;
            if (extendsLastReset)
            {
                resets.back().queryCount += 1U;
                return;
            }
            resets.push_back(PendingQueryPoolReset{
                .pool = poolUse.poolState->pool,
                .primitivesPool = poolUse.poolState->primitivesPool,
                .firstQuery = queryIndex,
                .queryCount = 1U,
                .keepAlive = poolUse.keepAlive,
            });
        }

        void markSimulatedSlotTouched(SimulatedQueryPoolDirty& simulated, std::uint32_t queryIndex) noexcept
        {
            if (!simulated.hasTouchedSlots)
            {
                simulated.minTouchedSlot = queryIndex;
                simulated.maxTouchedSlot = queryIndex;
                simulated.hasTouchedSlots = true;
                return;
            }
            simulated.minTouchedSlot = std::min(simulated.minTouchedSlot, queryIndex);
            simulated.maxTouchedSlot = std::max(simulated.maxTouchedSlot, queryIndex);
        }

        void simulateQueryPoolUse(SimulatedQueryPoolDirty& simulated,
                                  const VulkanCommandQueryPoolUse& poolUse,
                                  std::vector<PendingQueryPoolReset>& resets)
        {
            const std::size_t upper = std::min<std::size_t>(static_cast<std::size_t>(poolUse.maxTouchedSlot) + 1U,
                                                            std::min(simulated.dirty.size(), poolUse.slotFlags.size()));
            for (std::size_t slot = poolUse.minTouchedSlot; slot < upper; ++slot)
            {
                const std::uint8_t flags = poolUse.slotFlags[slot];
                if ((flags & kVulkanQuerySlotTouched) == 0U)
                {
                    continue;
                }
                const auto queryIndex = static_cast<std::uint32_t>(slot);
                if ((flags & kVulkanQuerySlotNeedsExternalReset) != 0U && simulated.dirty[slot] != 0U)
                {
                    appendQueryPoolReset(resets, poolUse, queryIndex);
                }
                simulated.dirty[slot] = (flags & kVulkanQuerySlotDirtyAtEnd) != 0U ? 1U : 0U;
                simulated.touched[slot] = 1U;
                markSimulatedSlotTouched(simulated, queryIndex);
            }
        }

        void planCommandQueryPoolResets(const VulkanCommandBuffer& command,
                                        std::vector<SimulatedQueryPoolDirty>& simulatedPools,
                                        std::vector<PendingQueryPoolReset>& resets)
        {
            for (const auto& poolUse : command.queryPoolUses())
            {
                if (!poolUse.hasTouchedSlots || poolUse.poolState == nullptr ||
                    poolUse.poolState->pool == VK_NULL_HANDLE)
                {
                    continue;
                }
                auto& simulated = findOrSnapshotQueryPool(simulatedPools, poolUse);
                simulateQueryPoolUse(simulated, poolUse, resets);
            }
        }

        [[nodiscard]] Result<QueryPoolResetPlan> planQueryPoolResets(
            std::span<VulkanCommandBuffer* const> nativeCommands)
        {
            QueryPoolResetPlan plan;
            try
            {
                plan.preResets.resize(nativeCommands.size());
                for (std::size_t cmdIndex = 0; cmdIndex < nativeCommands.size(); ++cmdIndex)
                {
                    planCommandQueryPoolResets(
                        *nativeCommands[cmdIndex], plan.simulatedPools, plan.preResets[cmdIndex]);
                }
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to plan Vulkan submission query-pool resets.",
                });
            }
            return plan;
        }

        struct CommandBridgeWork final
        {
            VkCommandBuffer command = VK_NULL_HANDLE;
            std::span<const PendingQueryPoolReset> queryResets;
            std::span<const VulkanResourceAllocator::PendingImageBarrierTransition> preTransitions;
            std::span<const VulkanResourceAllocator::PendingImageBarrierTransition> postTransitions;
        };

        [[nodiscard]] CommandBridgeWork bridgeWorkFor(
            const SubmissionBatch& batch,
            std::span<const std::vector<PendingQueryPoolReset>> preQueryResets,
            std::size_t index) noexcept
        {
            return CommandBridgeWork{
                .command = batch.rawCommands[index],
                .queryResets = preQueryResets[index],
                .preTransitions = batch.layoutPlans[index].preTransitions,
                .postTransitions = batch.layoutPlans[index].postTransitions,
            };
        }

        [[nodiscard]] bool needsPreBridge(const CommandBridgeWork& work) noexcept
        {
            return !work.queryResets.empty() || !work.preTransitions.empty();
        }

        [[nodiscard]] bool needsPostBridge(const CommandBridgeWork& work) noexcept
        {
            return !work.postTransitions.empty();
        }

        struct SubmissionBridges final
        {
            std::vector<std::vector<VkCommandBuffer>> perCommandBuffers;
            std::vector<std::vector<std::shared_ptr<void>>> perCommandRetained;
            std::vector<VkCommandBufferSubmitInfo> orderedSubmitInfos;
        };

        [[nodiscard]] Result<SubmissionBridges> reserveBridgeBookkeeping(
            const SubmissionBatch& batch, std::span<const std::vector<PendingQueryPoolReset>> preQueryResets)
        {
            SubmissionBridges bridges;
            try
            {
                bridges.perCommandBuffers.resize(batch.nativeCommands.size());
                bridges.perCommandRetained.resize(batch.nativeCommands.size());
                bridges.orderedSubmitInfos.reserve(batch.nativeCommands.size() * 3U);
                for (std::size_t index = 0; index < batch.nativeCommands.size(); ++index)
                {
                    const auto work = bridgeWorkFor(batch, preQueryResets, index);
                    const std::size_t bridgeCount =
                        (needsPreBridge(work) ? 1U : 0U) + (needsPostBridge(work) ? 1U : 0U);
                    if (bridgeCount != 0U)
                    {
                        bridges.perCommandBuffers[index].reserve(bridgeCount);
                    }
                    const std::size_t retainedCount =
                        work.queryResets.size() + work.preTransitions.size() + work.postTransitions.size();
                    if (retainedCount != 0U)
                    {
                        bridges.perCommandRetained[index].reserve(retainedCount);
                    }
                }
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to reserve Vulkan submission bridge bookkeeping.",
                });
            }
            return bridges;
        }

        void reserveBridgeRetention(std::span<VulkanCommandBuffer* const> nativeCommands,
                                    std::span<const std::vector<std::shared_ptr<void>>> perCommandBridgeRetained)
        {
            for (std::size_t index = 0; index < nativeCommands.size(); ++index)
            {
                if (!perCommandBridgeRetained[index].empty())
                {
                    auto& retained = nativeCommands[index]->retained();
                    detail::reserveAdditional(retained, perCommandBridgeRetained[index].size());
                }
            }
        }

        [[nodiscard]] bool poolAppearsEarlier(std::span<const std::shared_ptr<VulkanCommandPoolState>> pools,
                                              std::size_t index) noexcept
        {
            for (std::size_t prior = 0; prior < index; ++prior)
            {
                if (pools[prior].get() == pools[index].get())
                {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] std::size_t countPoolUsesFrom(std::span<const std::shared_ptr<VulkanCommandPoolState>> pools,
                                                    std::size_t index) noexcept
        {
            std::size_t countForPool = 0;
            for (std::size_t other = index; other < pools.size(); ++other)
            {
                if (pools[other].get() == pools[index].get())
                {
                    ++countForPool;
                }
            }
            return countForPool;
        }

        void reserveQuarantineCapacity(std::span<const std::shared_ptr<VulkanCommandPoolState>> pools)
        {
            for (std::size_t index = 0; index < pools.size(); ++index)
            {
                if (pools[index] == nullptr || poolAppearsEarlier(pools, index))
                {
                    continue;
                }
                const auto countForPool = countPoolUsesFrom(pools, index);
                const std::scoped_lock poolLock(pools[index]->mutex);
                pools[index]->quarantined.reserve(
                    std::max(pools[index]->entries.size(), pools[index]->quarantined.size() + countForPool));
            }
        }

        [[nodiscard]] VulkanPendingEntry& batchPendingEntry(VulkanQueueState& state,
                                                            const SubmissionBatch& batch,
                                                            std::size_t index) noexcept
        {
            return state.pending[state.pending.size() - batch.entries.size() + index];
        }

        void attachBridgesToPendingEntries(VulkanQueueState& state,
                                           const SubmissionBatch& batch,
                                           SubmissionBridges& bridges)
        {
            for (std::size_t index = 0; index < batch.entries.size(); ++index)
            {
                if (!bridges.perCommandRetained[index].empty())
                {
                    auto& retained = batch.nativeCommands[index]->retained();
                    retained.insert(retained.end(),
                                    std::make_move_iterator(bridges.perCommandRetained[index].begin()),
                                    std::make_move_iterator(bridges.perCommandRetained[index].end()));
                }
                batchPendingEntry(state, batch, index).bridgeBuffers = std::move(bridges.perCommandBuffers[index]);
            }
        }

        void publishSubmittedBatch(VulkanQueueState& state, const SubmissionBatch& batch, std::uint64_t submittedValue)
        {
            ++state.nextValue;
            state.lastSubmittedValue.store(submittedValue, std::memory_order_release);
            for (std::size_t index = 0; index < batch.entries.size(); ++index)
            {
                if (batch.pools[index]->resources != nullptr)
                {
                    batch.pools[index]->resources->commitCommandLayouts(batch.entries[index]->command);
                }
                auto& pending = batchPendingEntry(state, batch, index);
                pending.accelerationStructurePublications.clear();
                pending.timelineValue = submittedValue;
            }
        }

        void commitSimulatedQueryPools(std::span<const SimulatedQueryPoolDirty> simulatedPools)
        {
            for (const auto& simulated : simulatedPools)
            {
                if (!simulated.hasTouchedSlots || simulated.poolState == nullptr)
                {
                    continue;
                }
                const std::scoped_lock poolStateLock(simulated.poolState->mutex);
                const std::size_t upper = std::min<std::size_t>(static_cast<std::size_t>(simulated.maxTouchedSlot) + 1U,
                                                                simulated.poolState->dirtySlots.size());
                for (std::size_t slot = simulated.minTouchedSlot; slot < upper; ++slot)
                {
                    if (simulated.touched[slot] != 0U)
                    {
                        simulated.poolState->dirtySlots[slot] = simulated.dirty[slot];
                    }
                }
            }
        }

        [[nodiscard]] VkImageMemoryBarrier2 layoutTransitionBarrier(
            const VulkanResourceAllocator::PendingImageBarrierTransition& transition) noexcept
        {
            return VkImageMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = transition.oldLayout,
                .newLayout = transition.newLayout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = transition.image,
                .subresourceRange =
                    {
                        .aspectMask = transition.aspectMask,
                        .baseMipLevel = transition.baseMipLevel,
                        .levelCount = 1,
                        .baseArrayLayer = transition.baseArrayLayer,
                        .layerCount = 1,
                    },
            };
        }
    } // namespace

    class VulkanQueueManager::Impl final
    {
    public:
        Impl(const Impl&) = delete;
        Impl& operator=(const Impl&) = delete;
        Impl(Impl&&) = delete;
        Impl& operator=(Impl&&) = delete;
        Impl(VkDevice device,
             const VolkDeviceTable* table,
             const VulkanInstanceTable* instanceTable,
             std::uint64_t deviceToken) noexcept
            : device_(device), table_(table), instanceTable_(instanceTable), deviceToken_(deviceToken)
        {
        }

        ~Impl()
        {
            shutdown();
        }

        [[nodiscard]] Status initialize(const VulkanQueueHandles& queues, bool exportableTimelines)
        {
            const std::array native{
                std::pair{QueueType::graphics, queues.graphics},
                std::pair{QueueType::compute, queues.compute},
                std::pair{QueueType::copy, queues.copy},
            };
#ifdef RACHIS_TESTING
            const auto initialTimelineValue = test::initialQueueTimelineValue().load(std::memory_order_acquire);
#else
            constexpr std::uint64_t initialTimelineValue = 0;
#endif
            for (const auto& [type, handle] : native)
            {
                if (handle == VK_NULL_HANDLE)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "Vulkan queue submission requires a native queue for every public queue type.",
                    });
                }
                auto& state = queue(type);
                state.type = type;
                state.family = type == QueueType::graphics
                                   ? queues.families.graphics
                                   : (type == QueueType::compute ? queues.families.compute : queues.families.transfer);
                state.gate = gateFor(handle);
                if (state.gate == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::outOfMemory,
                        0,
                        "Failed to allocate a Vulkan native-queue gate.",
                    });
                }

                exportableTimelines_ = exportableTimelines;
                auto timeline = createQueueTimeline(false, timelineName(type), initialTimelineValue);
                if (!timeline)
                {
                    return std::unexpected(std::move(timeline.error()));
                }
                state.timeline = *timeline;
                state.nextValue = initialTimelineValue + 1;
            }
            retireOnSubmissionFence_ = queues.retireOnSubmissionFence;
#ifdef RACHIS_TESTING
            if (test::forceSubmissionRetirementFences().load(std::memory_order_acquire))
            {
                retireOnSubmissionFence_ = true;
            }
#endif
            return initializePresent(queues);
        }

        void shutdown() noexcept
        {
            if (device_ == VK_NULL_HANDLE || table_ == nullptr)
            {
                return;
            }
#ifdef _WIN32
            stopTimelineExportSignaller();
#endif
            for (auto& state : queues_)
            {
                if (state.timeline != VK_NULL_HANDLE && table_->vkDestroySemaphore != nullptr)
                {
                    table_->vkDestroySemaphore(device_, state.timeline, nullptr);
                }
                if (state.bridgePool != VK_NULL_HANDLE && table_->vkDestroyCommandPool != nullptr)
                {
                    table_->vkDestroyCommandPool(device_, state.bridgePool, nullptr);
                }
                state.bridgePool = VK_NULL_HANDLE;
                state.availableBridgeBuffers.clear();
                state.totalBridgeBuffers = 0;
                state.family = VK_QUEUE_FAMILY_IGNORED;
                state.timeline = VK_NULL_HANDLE;
                state.gate = nullptr;
                destroyRetirementFences(state);
                state.pending.clear();
                state.lastSubmittedValue.store(0, std::memory_order_relaxed);
                state.nextValue = 1;
                state.unchainedSharedWaitValue = 0;
            }
            gates_.clear();
            presentGate_ = nullptr;
            presentQueue_ = VK_NULL_HANDLE;
            vkQueuePresentKHR_ = nullptr;
            device_ = VK_NULL_HANDLE;
            table_ = nullptr;
            instanceTable_ = nullptr;
        }

    private:
        [[nodiscard]] Result<VkSemaphore> createQueueTimeline(bool exportable,
                                                              const char* name,
                                                              std::uint64_t initialValue)
        {
            VkSemaphoreTypeCreateInfo typeInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .pNext = {},
                .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                .initialValue = initialValue,
            };
#ifdef _WIN32
            VkExportSemaphoreCreateInfo exportInfo{
                .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
                .pNext = {},
                .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };
            if (exportable)
            {
                typeInfo.pNext = &exportInfo;
            }
#endif
            const VkSemaphoreCreateInfo create{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = &typeInfo,
                .flags = {},
            };
            VkSemaphore semaphore = VK_NULL_HANDLE;
            const auto created = table_->vkCreateSemaphore(device_, &create, nullptr, &semaphore);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(created == VK_ERROR_OUT_OF_HOST_MEMORY || created == VK_ERROR_OUT_OF_DEVICE_MEMORY
                                        ? ErrorCode::outOfMemory
                                        : vulkanResultErrorCode(created),
                                    created,
                                    exportable ? "vkCreateSemaphore(shared timeline)" : "vkCreateSemaphore(timeline)"));
            }
            setVulkanObjectName(
                instanceTable_, device_, VK_OBJECT_TYPE_SEMAPHORE, reinterpret_cast<std::uint64_t>(semaphore), name);
            return semaphore;
        }

        [[nodiscard]] VulkanCommandBuffer* submittableCommandBuffer(
            const std::unique_ptr<detail::BackendCommandBuffer>& commandBuffer) const noexcept
        {
            const bool ownedByThisDevice = commandBuffer != nullptr &&
                                           commandBuffer->kind() == detail::BackendCommandBufferKind::vulkan &&
                                           deviceToken_ != 0 && commandBuffer->deviceToken() == deviceToken_;
            if (!ownedByThisDevice)
            {
                return nullptr;
            }
            auto* native = static_cast<VulkanCommandBuffer*>(commandBuffer.get());
            const bool leaseIsIntact =
                native->owner() != nullptr && native->entry() != nullptr && native->native() != VK_NULL_HANDLE;
            return leaseIsIntact ? native : nullptr;
        }

        [[nodiscard]] Result<VulkanQueueState*> resolveBatchQueue(VulkanQueueState* batchQueue,
                                                                  const VulkanCommandBuffer& native)
        {
            if (batchQueue == nullptr)
            {
                if (!isValidQueueType(native.queueType()))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A Vulkan submission batch targeted an invalid queue type.",
                    });
                }
                return &queue(native.queueType());
            }
            if (batchQueue != &queue(native.queueType()))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Every Vulkan command buffer in a submission batch must target the same queue.",
                });
            }
            return batchQueue;
        }

        [[nodiscard]] Result<ValidatedBatchCommands> collectBatchCommands(
            std::span<const std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers)
        {
            std::vector<VulkanCommandBuffer*> nativeCommands;
            try
            {
                nativeCommands.reserve(commandBuffers.size());
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate Vulkan submission command buffer batch.",
                });
            }
            VulkanQueueState* queueState = nullptr;
            for (const auto& commandBuffer : commandBuffers)
            {
                auto* native = submittableCommandBuffer(commandBuffer);
                if (native == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A command buffer in the submission batch belongs to another device or backend.",
                    });
                }
                auto batchQueue = resolveBatchQueue(queueState, *native);
                if (!batchQueue)
                {
                    return std::unexpected(std::move(batchQueue.error()));
                }
                queueState = *batchQueue;
                if (std::ranges::any_of(nativeCommands,
                                        [&](const auto* other) { return other->entry() == native->entry(); }))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A Vulkan submission batch cannot contain the same command-buffer lease more than once.",
                    });
                }
                auto ready = recordingReady(*native);
                if (!ready)
                {
                    return std::unexpected(std::move(ready.error()));
                }
                nativeCommands.push_back(native);
            }
            if (queueState == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A Vulkan submission batch must contain at least one command buffer.",
                });
            }
            return ValidatedBatchCommands{
                .queueState = queueState,
                .nativeCommands = std::move(nativeCommands),
            };
        }

        [[nodiscard]] Status validateBatchWaits(std::span<const TimelinePoint> waits) const
        {
            for (const auto point : waits)
            {
                if (!point.isValid() || !isValidQueueType(point.queueType()) ||
                    detail::TimelinePointAccess::deviceToken(point) != deviceToken_ ||
                    point.value() > queue(point.queueType()).lastSubmittedValue.load(std::memory_order_acquire))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A queue wait references an invalid, foreign, or unsubmitted timeline point.",
                    });
                }
            }
            return {};
        }

        void rollbackRecordedBridges(VulkanQueueState& state, SubmissionBridges& bridges) noexcept
        {
            for (auto& buffers : bridges.perCommandBuffers)
            {
                recycleBridgeCommandBuffers(state, buffers, false);
                buffers.clear();
            }
        }

        [[nodiscard]] Status recordBridgesAroundCommand(VulkanQueueState& state,
                                                        std::uint32_t fallbackFamily,
                                                        const CommandBridgeWork& work,
                                                        SubmissionBridges& bridges,
                                                        std::size_t index)
        {
            if (needsPreBridge(work))
            {
                auto preBridge = recordBridgeCommandBuffer(
                    state, fallbackFamily, work.queryResets, work.preTransitions, bridges.perCommandRetained[index]);
                if (!preBridge)
                {
                    return std::unexpected(std::move(preBridge.error()));
                }
                bridges.perCommandBuffers[index].push_back(*preBridge);
                bridges.orderedSubmitInfos.push_back(commandBufferSubmitInfo(*preBridge));
            }
            bridges.orderedSubmitInfos.push_back(commandBufferSubmitInfo(work.command));
            if (needsPostBridge(work))
            {
                auto postBridge = recordBridgeCommandBuffer(
                    state, fallbackFamily, {}, work.postTransitions, bridges.perCommandRetained[index]);
                if (!postBridge)
                {
                    return std::unexpected(std::move(postBridge.error()));
                }
                bridges.perCommandBuffers[index].push_back(*postBridge);
                bridges.orderedSubmitInfos.push_back(commandBufferSubmitInfo(*postBridge));
            }
            return {};
        }

        [[nodiscard]] Result<SubmissionBridges> recordBatchBridges(
            VulkanQueueState& state,
            const SubmissionBatch& batch,
            std::span<const std::vector<PendingQueryPoolReset>> preQueryResets)
        {
            auto bridges = reserveBridgeBookkeeping(batch, preQueryResets);
            if (!bridges)
            {
                return bridges;
            }
            const std::uint32_t fallbackFamily =
                (!batch.pools.empty() && batch.pools[0] != nullptr) ? batch.pools[0]->family : VK_QUEUE_FAMILY_IGNORED;
            try
            {
                for (std::size_t index = 0; index < batch.nativeCommands.size(); ++index)
                {
                    const auto work = bridgeWorkFor(batch, preQueryResets, index);
                    auto recorded = recordBridgesAroundCommand(state, fallbackFamily, work, *bridges, index);
                    if (!recorded)
                    {
                        rollbackRecordedBridges(state, *bridges);
                        return std::unexpected(std::move(recorded.error()));
                    }
                }
            }
            catch (const std::bad_alloc&)
            {
                rollbackRecordedBridges(state, *bridges);
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate Vulkan submission bridge diagnostics.",
                });
            }
            return bridges;
        }

        [[nodiscard]] Result<std::vector<VkSemaphoreSubmitInfo>> stagePendingBatch(
            VulkanQueueState& state,
            const SubmissionBatch& batch,
            SubmissionBridges& bridges,
            std::size_t waitCount,
            std::vector<std::vector<std::shared_ptr<void>>>& deferredRetained)
        {
            std::vector<VkSemaphoreSubmitInfo> reservedWaitInfos;
            std::size_t reservedPendingCount = 0;
            try
            {
                reservedWaitInfos.reserve(waitCount);
                reserveBridgeRetention(batch.nativeCommands, bridges.perCommandRetained);
                deferredRetained.reserve(batch.entries.size());
                for (std::size_t index = 0; index < batch.entries.size(); ++index)
                {
                    state.pending.push_back(VulkanPendingEntry{
                        .timelineValue = std::numeric_limits<std::uint64_t>::max(),
                        .pool = batch.pools[index],
                        .entry = batch.entries[index],
                    });
                    ++reservedPendingCount;
                }
                reserveQuarantineCapacity(batch.pools);
                attachBridgesToPendingEntries(state, batch, bridges);
            }
            catch (const std::bad_alloc&)
            {
                while (reservedPendingCount != 0)
                {
                    recycleBridgeCommandBuffers(state, state.pending.back().bridgeBuffers, false);
                    state.pending.pop_back();
                    --reservedPendingCount;
                }
                rollbackRecordedBridges(state, bridges);
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to reserve Vulkan batch retirement records.",
                });
            }
            return reservedWaitInfos;
        }

        [[nodiscard]] VkResult endAndReleaseCommandBuffers(
            VulkanQueueState& state,
            const SubmissionBatch& batch,
            std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers)
        {
            VkResult endFailure = VK_SUCCESS;
            for (std::size_t index = 0; index < batch.nativeCommands.size(); ++index)
            {
                auto& pending = batchPendingEntry(state, batch, index);
                pending.retained = batch.nativeCommands[index]->takeRetained();
                pending.accelerationStructurePublications =
                    batch.nativeCommands[index]->takeAccelerationStructurePublications();
                {
                    const std::scoped_lock poolLock(batch.pools[index]->mutex);
                    static_cast<void>(batch.nativeCommands[index]->release());
                    batch.entries[index]->inFlight = true;
                    const auto ended = table_->vkEndCommandBuffer(batch.entries[index]->command);
                    if (ended != VK_SUCCESS && endFailure == VK_SUCCESS)
                    {
                        endFailure = ended;
                    }
                }
                commandBuffers[index].reset();
            }
#ifdef RACHIS_TESTING
            if (const auto injected = test::nextEndCommandBufferResult().exchange(0, std::memory_order_relaxed);
                injected != 0 && endFailure == VK_SUCCESS)
            {
                endFailure = static_cast<VkResult>(injected);
            }
#endif
            return endFailure;
        }

        void discardPendingBatch(VulkanQueueState& state,
                                 const SubmissionBatch& batch,
                                 std::vector<std::vector<std::shared_ptr<void>>>& deferredRetained) noexcept
        {
            for (std::size_t index = 0; index < batch.entries.size(); ++index)
            {
                rollBackVulkanAccelerationStructurePublications(state.pending.back().accelerationStructurePublications);
                recycleBridgeCommandBuffers(state, state.pending.back().bridgeBuffers, false);
                recycleRetirementFence(state, state.pending.back().fence);
                deferredRetained.push_back(std::move(state.pending.back().retained));
                state.pending.pop_back();
            }
        }

        void quarantineBatch(VulkanQueueState& state,
                             const SubmissionBatch& batch,
                             std::vector<std::vector<std::shared_ptr<void>>>& deferredRetained) noexcept
        {
            discardPendingBatch(state, batch, deferredRetained);
            for (std::size_t index = 0; index < batch.entries.size(); ++index)
            {
                if (batch.entries[index] == nullptr || batch.pools[index] == nullptr)
                {
                    continue;
                }
                const std::scoped_lock poolLock(batch.pools[index]->mutex);
                if (batch.pools[index]->resources != nullptr)
                {
                    batch.pools[index]->resources->discardCommandLayouts(batch.entries[index]->command);
                }
                batch.entries[index]->inFlight = false;
                batch.entries[index]->lease = nullptr;
                batch.pools[index]->quarantined.push_back(batch.entries[index]);
            }
        }

        void appendTimelineWaits(std::vector<VkSemaphoreSubmitInfo>& waitInfos,
                                 const VulkanQueueState& state,
                                 std::span<const TimelinePoint> waits)
        {
            for (const auto point : waits)
            {
                waitInfos.push_back(semaphoreSubmitInfo(queue(point.queueType()).timeline, point.value()));
            }
            if (state.unchainedSharedWaitValue != 0)
            {
                waitInfos.push_back(semaphoreSubmitInfo(state.timeline, state.unchainedSharedWaitValue));
            }
        }

        [[nodiscard]] VkResult submitBatchSignalingTimeline(const VulkanQueueState& state,
                                                            std::span<const VkSemaphoreSubmitInfo> waitInfos,
                                                            std::span<const VkCommandBufferSubmitInfo> commandInfos,
                                                            std::uint64_t signalValue,
                                                            VkFence fence)
        {
            const auto signalInfo = semaphoreSubmitInfo(state.timeline, signalValue);
            const VkSubmitInfo2 submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                .pNext = {},
                .flags = {},
                .waitSemaphoreInfoCount = static_cast<std::uint32_t>(waitInfos.size()),
                .pWaitSemaphoreInfos = waitInfos.empty() ? nullptr : waitInfos.data(),
                .commandBufferInfoCount = static_cast<std::uint32_t>(commandInfos.size()),
                .pCommandBufferInfos = commandInfos.data(),
                .signalSemaphoreInfoCount = 1,
                .pSignalSemaphoreInfos = &signalInfo,
            };
            return table_->vkQueueSubmit2(state.gate->queue, 1, &submitInfo, fence);
        }

        [[nodiscard]] Result<TimelinePoint> recordBridgesAndSubmitBatch(
            VulkanQueueState& state,
            std::vector<VulkanCommandBuffer*> nativeCommands,
            std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers,
            std::span<const TimelinePoint> waits,
            std::vector<std::vector<std::shared_ptr<void>>>& deferredRetained)
        {
            auto batch = gatherSubmissionBatch(std::move(nativeCommands));
            if (!batch)
            {
                return std::unexpected(std::move(batch.error()));
            }
            auto queryResets = planQueryPoolResets(batch->nativeCommands);
            if (!queryResets)
            {
                return std::unexpected(std::move(queryResets.error()));
            }
            auto bridges = recordBatchBridges(state, *batch, queryResets->preResets);
            if (!bridges)
            {
                return std::unexpected(std::move(bridges.error()));
            }

            const std::unique_lock retirementLock(state.retirementMutex);
            auto fence = acquireRetirementFence(state);
            if (!fence)
            {
                rollbackRecordedBridges(state, *bridges);
                return std::unexpected(std::move(fence.error()));
            }
            auto waitInfos = stagePendingBatch(
                state, *batch, *bridges, waits.size() + sharedWaitChainCount(state), deferredRetained);
            if (!waitInfos)
            {
                recycleRetirementFence(state, *fence);
                return std::unexpected(std::move(waitInfos.error()));
            }
            if (batch->entries.empty())
            {
                recycleRetirementFence(state, std::exchange(*fence, VK_NULL_HANDLE));
            }
            else
            {
                batchPendingEntry(state, *batch, 0).fence = *fence;
            }

            const auto endFailure = endAndReleaseCommandBuffers(state, *batch, commandBuffers);
            if (endFailure != VK_SUCCESS)
            {
                quarantineBatch(state, *batch, deferredRetained);
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(endFailure), endFailure, "vkEndCommandBuffer"));
            }

            appendTimelineWaits(*waitInfos, state, waits);
            const auto submittedValue = state.nextValue;
            const auto submitted =
                submitBatchSignalingTimeline(state, *waitInfos, bridges->orderedSubmitInfos, submittedValue, *fence);
            if (submitted != VK_SUCCESS)
            {
                quarantineBatch(state, *batch, deferredRetained);
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(submitted), submitted, "vkQueueSubmit2"));
            }
            publishSubmittedBatch(state, *batch, submittedValue);
            state.unchainedSharedWaitValue = 0;
            commitSimulatedQueryPools(queryResets->simulatedPools);
            return detail::TimelinePointAccess::make(state.type, submittedValue, deviceToken_);
        }

    public:
        [[nodiscard]] Result<TimelinePoint> submit(
            std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers,
            std::span<const TimelinePoint> waits)
        {
            auto validated = collectBatchCommands(commandBuffers);
            if (!validated)
            {
                return std::unexpected(std::move(validated.error()));
            }
            auto& state = *validated->queueState;
            auto waitsValid = validateBatchWaits(waits);
            if (!waitsValid)
            {
                return std::unexpected(std::move(waitsValid.error()));
            }
            if (state.gate == nullptr || state.timeline == VK_NULL_HANDLE || table_ == nullptr ||
                table_->vkEndCommandBuffer == nullptr || table_->vkQueueSubmit2 == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan queue submission is unavailable because the device table is incomplete.",
                });
            }

            std::vector<std::vector<std::shared_ptr<void>>> deferredRetained;
            const std::scoped_lock submissionLock(state.gate->mutex);
            if (!hasUnreservedTimelineSlot(state))
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "The queue timeline is exhausted and cannot be reused safely.",
                });
            }
            return recordBridgesAndSubmitBatch(
                state, std::move(validated->nativeCommands), commandBuffers, waits, deferredRetained);
        }

        [[nodiscard]] Result<TimelinePointStatus> queryTimelinePoint(TimelinePoint point)
        {
            auto valid = checkSubmittedPoint(point);
            if (!valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            auto& state = queue(point.queueType());
            auto status = queryValue(state, point.value());
            if (status && *status == TimelinePointStatus::complete)
            {
                status = waitRetirementFencesFor(state, point.value(), std::chrono::nanoseconds::zero());
            }
            if (status)
            {
                collectGarbage();
            }
            return status;
        }

        [[nodiscard]] Result<TimelinePointStatus> wait(TimelinePoint point, std::chrono::nanoseconds timeout)
        {
            if (timeout < std::chrono::nanoseconds::zero())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A bounded timeline wait requires a non-negative timeout.",
                });
            }
            auto valid = checkSubmittedPoint(point);
            if (!valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            auto& state = queue(point.queueType());
            const auto started = std::chrono::steady_clock::now();
            auto status = queryValue(state, point.value());
            if (!status)
            {
                return status;
            }
            if (*status != TimelinePointStatus::complete && timeout != std::chrono::nanoseconds::zero())
            {
                status = waitForValue(state, point.value(), static_cast<std::uint64_t>(timeout.count()));
                if (!status)
                {
                    return status;
                }
            }
            if (*status == TimelinePointStatus::complete)
            {
                status = waitRetirementFencesFor(state, point.value(), remainingTimeout(started, timeout));
                if (!status)
                {
                    return status;
                }
            }
            collectGarbage();
            return status;
        }

        [[nodiscard]] Status wait(TimelinePoint point)
        {
            auto valid = checkSubmittedPoint(point);
            if (!valid)
            {
                return valid;
            }
            auto& state = queue(point.queueType());
            auto waited = waitForValue(state, point.value(), std::numeric_limits<std::uint64_t>::max());
            if (!waited)
            {
                return std::unexpected(std::move(waited.error()));
            }
            if (*waited != TimelinePointStatus::complete)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "An unbounded Vulkan timeline wait returned before the requested point completed.",
                });
            }
            if (auto fenced = waitRetirementFences(state, point.value()); !fenced)
            {
                return fenced;
            }
            collectGarbage();
            return {};
        }

        [[nodiscard]] Status waitIdle(QueueType queueType)
        {
            if (!isValidQueueType(queueType))
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, "The queue type is invalid."});
            }
            auto& state = queue(queueType);
            if (state.gate == nullptr || state.timeline == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan queue waitIdle requested a queue that was not created.",
                });
            }
            const std::scoped_lock submissionLock(state.gate->mutex);
            auto waited = waitForSubmittedWork(state);
            if (!waited)
            {
                return waited;
            }
            if (table_ != nullptr && table_->vkQueueWaitIdle != nullptr)
            {
                const auto idle = table_->vkQueueWaitIdle(state.gate->queue);
                if (idle != VK_SUCCESS)
                {
                    return std::unexpected(makeVulkanError(vulkanResultErrorCode(idle), idle, "vkQueueWaitIdle"));
                }
            }
            auto fenced = waitRetirementFences(state);
            if (!fenced)
            {
                return fenced;
            }
            collectQueue(state);
            return {};
        }

        [[nodiscard]] Status waitPresentIdle()
        {
            if (presentGate_ == nullptr || presentQueue_ == VK_NULL_HANDLE)
            {
                return {};
            }
            const std::scoped_lock submissionLock(presentGate_->mutex);
            if (table_ != nullptr && table_->vkQueueWaitIdle != nullptr)
            {
                const auto idle = table_->vkQueueWaitIdle(presentQueue_);
                if (idle != VK_SUCCESS)
                {
                    return std::unexpected(makeVulkanError(vulkanResultErrorCode(idle), idle, "vkQueueWaitIdle"));
                }
            }
            return {};
        }

        [[nodiscard]] Status waitIdle()
        {
            if (device_ == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan device waitIdle requested a device that was not created.",
                });
            }

            std::array<std::mutex*, queueCount + 1> mutexes{};
            std::size_t mutexCount = 0;
            for (auto& gate : gates_)
            {
                if (gate != nullptr && mutexCount < mutexes.size())
                {
                    mutexes[mutexCount++] = &gate->mutex;
                }
            }
            if (mutexCount == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan device waitIdle requested a device that was not created.",
                });
            }
            if (mutexCount == 1)
            {
                const std::scoped_lock submissionLocks(*mutexes[0]);
                return waitIdleLocked();
            }
            if (mutexCount == 2)
            {
                const std::scoped_lock submissionLocks(*mutexes[0], *mutexes[1]);
                return waitIdleLocked();
            }
            if (mutexCount == 3)
            {
                const std::scoped_lock submissionLocks(*mutexes[0], *mutexes[1], *mutexes[2]);
                return waitIdleLocked();
            }
            const std::scoped_lock submissionLocks(*mutexes[0], *mutexes[1], *mutexes[2], *mutexes[3]);
            return waitIdleLocked();
        }

        [[nodiscard]] std::uint64_t completedValue(QueueType queueType) noexcept
        {
            if (!isValidQueueType(queueType))
            {
                return 0;
            }
            auto& state = queue(queueType);
            const auto value = counterValue(state);
            if (value && *value > 0)
            {
                collectGarbage();
            }
            if (!value || !retireOnSubmissionFence_)
            {
                return value.value_or(0);
            }
            return retiredValue(state, *value);
        }

        void collectGarbage() noexcept
        {
            for (auto& state : queues_)
            {
                collectQueue(state);
            }
#ifdef _WIN32
            signalCompletedTimelineExports();
#endif
        }

        [[nodiscard]] Status validateSubmittedPoint(TimelinePoint point) const
        {
            return checkSubmittedPoint(point);
        }

        [[nodiscard]] Result<TimelinePoint> submitGraphicsPresentationSync(
            const VulkanQueueManager::PresentationSync& sync)
        {
#ifdef RACHIS_TESTING
            const auto injected = test::nextGraphicsPresentationSyncResult().exchange(0, std::memory_order_relaxed);
            if (injected != 0)
            {
                const auto result = static_cast<VkResult>(injected);
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(result), result, "vkQueueSubmit2"));
            }
#endif
            auto& graphics = queue(QueueType::graphics);
            auto signaled = submitPresentationSync(graphics.gate, sync, &graphics);
            if (!signaled)
            {
                return std::unexpected(std::move(signaled.error()));
            }
            return detail::TimelinePointAccess::make(QueueType::graphics, *signaled, deviceToken_);
        }

        [[nodiscard]] Status submitPresentPresentationSync(const VulkanQueueManager::PresentationSync& sync)
        {
            auto submitted = submitPresentationSync(presentGate_, sync, nullptr);
            if (!submitted)
            {
                return std::unexpected(std::move(submitted.error()));
            }
            return {};
        }

        void drainGraphicsAcquire(VkSemaphore semaphore)
        {
            const auto wait = semaphoreSubmitInfo(semaphore);
            const VkSubmitInfo2 submit{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                .pNext = {},
                .flags = {},
                .waitSemaphoreInfoCount = 1,
                .pWaitSemaphoreInfos = &wait,
                .commandBufferInfoCount = {},
                .pCommandBufferInfos = {},
                .signalSemaphoreInfoCount = {},
                .pSignalSemaphoreInfos = {},
            };
            auto* gate = queue(QueueType::graphics).gate;
            const std::scoped_lock gateLock(gate->mutex);
            const bool submitted = retryTransientVulkanFailures(
                [&] { return table_->vkQueueSubmit2(gate->queue, 1, &submit, VK_NULL_HANDLE); });
            if (!submitted)
            {
                return;
            }
            static_cast<void>(retryTransientVulkanFailures([&] { return table_->vkQueueWaitIdle(gate->queue); }));
        }

        [[nodiscard]] Result<VulkanQueueManager::PresentResult> presentOnPresentQueue(const VkPresentInfoKHR& info)
        {
#ifdef RACHIS_TESTING
            const auto injected = consumeInjectedPresentResult();
            if (injected && !presentWaitsEnqueued(*injected))
            {
                return VulkanQueueManager::PresentResult{
                    .result = *injected,
                    .injectedDeviceLoss = *injected == VK_ERROR_DEVICE_LOST,
                };
            }
#endif
            if (presentGate_ == nullptr || presentQueue_ == VK_NULL_HANDLE || vkQueuePresentKHR_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::backendUnavailable,
                    0,
                    "Vulkan presentation is unavailable because the present queue is incomplete.",
                });
            }
            const std::scoped_lock gateLock(presentGate_->mutex);
            auto result = vkQueuePresentKHR_(presentQueue_, &info);
#ifdef RACHIS_TESTING
            if (injected && (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR))
            {
                result = *injected;
            }
#endif
            return VulkanQueueManager::PresentResult{.result = result, .waitsEnqueued = presentWaitsEnqueued(result)};
        }

        [[nodiscard]] Status initializePresent(const VulkanQueueHandles& queues)
        {
            presentQueue_ = queues.present != VK_NULL_HANDLE ? queues.present : queues.graphics;
            presentGate_ = gateFor(presentQueue_);
            if (presentGate_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate a Vulkan present-queue gate.",
                });
            }
            if (table_ != nullptr && table_->vkQueuePresentKHR != nullptr)
            {
                vkQueuePresentKHR_ = table_->vkQueuePresentKHR;
            }
            else if (device_ != VK_NULL_HANDLE && instanceTable_ != nullptr &&
                     instanceTable_->vkGetDeviceProcAddr != nullptr)
            {
                vkQueuePresentKHR_ = dispatch_detail::resolveVulkanProc<PFN_vkQueuePresentKHR>(
                    device_, instanceTable_->vkGetDeviceProcAddr, "vkQueuePresentKHR");
            }
            return {};
        }

    private:
        [[nodiscard]] VulkanQueueState& queue(QueueType type) noexcept
        {
            return queues_[queueIndex(type)];
        }

        [[nodiscard]] const VulkanQueueState& queue(QueueType type) const noexcept
        {
            return queues_[queueIndex(type)];
        }

        [[nodiscard]] VulkanNativeQueueGate* gateFor(VkQueue handle)
        {
            for (auto& gate : gates_)
            {
                if (gate != nullptr && gate->queue == handle)
                {
                    return gate.get();
                }
            }
            try
            {
                auto gate = std::make_unique<VulkanNativeQueueGate>();
                gate->queue = handle;
                auto* pointer = gate.get();
                gates_.push_back(std::move(gate));
                return pointer;
            }
            catch (const std::bad_alloc&)
            {
                return nullptr;
            }
        }

        [[nodiscard]] Status checkSubmittedPoint(TimelinePoint point) const
        {
            if (!isValidQueueType(point.queueType()) || !point.isValid() ||
                detail::TimelinePointAccess::deviceToken(point) != deviceToken_ ||
                point.value() > queue(point.queueType()).lastSubmittedValue.load(std::memory_order_acquire))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The timeline point is invalid, foreign, on the wrong queue, or has not been submitted.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<std::uint64_t> submitPresentationSync(VulkanNativeQueueGate* gate,
                                                                   const VulkanQueueManager::PresentationSync& sync,
                                                                   VulkanQueueState* signaledTimeline)
        {
            if (gate == nullptr || table_ == nullptr || table_->vkQueueSubmit2 == nullptr ||
                (signaledTimeline != nullptr && signaledTimeline->timeline == VK_NULL_HANDLE))
            {
                return std::unexpected(Error{
                    ErrorCode::backendUnavailable,
                    0,
                    "Vulkan presentation synchronization is unavailable because the device table is incomplete.",
                });
            }
            if (!sync.waitTimeline.has_value() && sync.waitBinary == VK_NULL_HANDLE &&
                sync.signalBinary == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A presentation sync submit requires at least one binary or timeline semaphore.",
                });
            }
            if (sync.waitTimeline.has_value())
            {
                auto valid = checkSubmittedPoint(*sync.waitTimeline);
                if (!valid)
                {
                    return std::unexpected(std::move(valid.error()));
                }
            }

            std::vector<VkSemaphoreSubmitInfo> waits;
            if (sync.waitTimeline.has_value())
            {
                const auto& point = *sync.waitTimeline;
                waits.push_back(semaphoreSubmitInfo(queue(point.queueType()).timeline, point.value()));
            }
            if (sync.waitBinary != VK_NULL_HANDLE)
            {
                waits.push_back(semaphoreSubmitInfo(sync.waitBinary));
            }
            std::vector<VkSemaphoreSubmitInfo> signals;
            signals.reserve(2);
            if (sync.signalBinary != VK_NULL_HANDLE)
            {
                signals.push_back(semaphoreSubmitInfo(sync.signalBinary));
            }

            const std::scoped_lock gateLock(gate->mutex);
            std::uint64_t signalValue = 0;
            if (signaledTimeline != nullptr)
            {
                if (!hasUnreservedTimelineSlot(*signaledTimeline))
                {
                    return std::unexpected(Error{
                        ErrorCode::resourceExhausted,
                        0,
                        "The queue timeline is exhausted and cannot be reused safely.",
                    });
                }
                signalValue = signaledTimeline->nextValue;
                signals.push_back(semaphoreSubmitInfo(signaledTimeline->timeline, signalValue));
            }
            const VkSubmitInfo2 submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                .pNext = {},
                .flags = {},
                .waitSemaphoreInfoCount = static_cast<std::uint32_t>(waits.size()),
                .pWaitSemaphoreInfos = waits.empty() ? nullptr : waits.data(),
                .commandBufferInfoCount = 0,
                .pCommandBufferInfos = {},
                .signalSemaphoreInfoCount = static_cast<std::uint32_t>(signals.size()),
                .pSignalSemaphoreInfos = signals.empty() ? nullptr : signals.data(),
            };
            const auto submitted = table_->vkQueueSubmit2(gate->queue, 1, &submitInfo, VK_NULL_HANDLE);
            if (submitted != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(submitted), submitted, "vkQueueSubmit2"));
            }
            if (signaledTimeline != nullptr)
            {
                ++signaledTimeline->nextValue;
                signaledTimeline->lastSubmittedValue.store(signalValue, std::memory_order_release);
            }
            return signalValue;
        }

        [[nodiscard]] bool counterQueryable(const VulkanQueueState& state) const noexcept
        {
            return device_ != VK_NULL_HANDLE && table_ != nullptr && table_->vkGetSemaphoreCounterValue != nullptr &&
                   state.timeline != VK_NULL_HANDLE;
        }

        [[nodiscard]] VkResult readCounterValue(const VulkanQueueState& state, std::uint64_t& value) const noexcept
        {
#ifdef RACHIS_TESTING
            if (const auto injected = test::nextSemaphoreCounterValueResult().exchange(0, std::memory_order_relaxed);
                injected != 0)
            {
                return static_cast<VkResult>(injected);
            }
#endif
            return table_->vkGetSemaphoreCounterValue(device_, state.timeline, &value);
        }

        [[nodiscard]] std::optional<std::uint64_t> counterValue(const VulkanQueueState& state) const noexcept
        {
            std::uint64_t value = 0;
            if (!counterQueryable(state) || readCounterValue(state, value) != VK_SUCCESS)
            {
                return std::nullopt;
            }
            return value;
        }

        [[nodiscard]] Result<TimelinePointStatus> queryValue(const VulkanQueueState& state, std::uint64_t value) const
        {
            if (value == 0 || value > state.lastSubmittedValue.load(std::memory_order_acquire))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The timeline point was not submitted by this queue.",
                });
            }
            if (!counterQueryable(state))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan timeline queries are unavailable because the device table is incomplete.",
                });
            }
            std::uint64_t completed = 0;
            if (const auto queried = readCounterValue(state, completed); queried != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(queried), queried, "vkGetSemaphoreCounterValue"));
            }
            return completed >= value ? TimelinePointStatus::complete : TimelinePointStatus::pending;
        }

        [[nodiscard]] Result<TimelinePointStatus> waitForValue(VulkanQueueState& state,
                                                               std::uint64_t value,
                                                               std::uint64_t timeoutNs)
        {
            auto status = queryValue(state, value);
            if (!status || *status == TimelinePointStatus::complete || timeoutNs == 0)
            {
                return status;
            }
            if (table_ == nullptr || table_->vkWaitSemaphores == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan timeline waits are unavailable because the device table is incomplete.",
                });
            }
            const VkSemaphoreWaitInfo waitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .pNext = {},
                .flags = {},
                .semaphoreCount = 1,
                .pSemaphores = &state.timeline,
                .pValues = &value,
            };
            const auto waited = table_->vkWaitSemaphores(device_, &waitInfo, timeoutNs);
            if (waited == VK_TIMEOUT)
            {
                return TimelinePointStatus::pending;
            }
            if (waited != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(waited), waited, "vkWaitSemaphores"));
            }
            return queryValue(state, value);
        }

        [[nodiscard]] Status waitForSubmittedWork(VulkanQueueState& state)
        {
            const auto lastSubmitted = state.lastSubmittedValue.load(std::memory_order_acquire);
            if (lastSubmitted == 0)
            {
                return {};
            }
            auto waited = waitForValue(state, lastSubmitted, std::numeric_limits<std::uint64_t>::max());
            if (!waited)
            {
                return std::unexpected(std::move(waited.error()));
            }
            if (*waited != TimelinePointStatus::complete)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "Vulkan queue idle returned before the published timeline frontier completed.",
                });
            }
            return {};
        }

        [[nodiscard]] Status waitIdleLocked()
        {
            for (auto& state : queues_)
            {
                auto waited = waitForSubmittedWork(state);
                if (!waited)
                {
                    return waited;
                }
            }
            if (table_ != nullptr && table_->vkDeviceWaitIdle != nullptr)
            {
                const auto idle = table_->vkDeviceWaitIdle(device_);
                if (idle != VK_SUCCESS)
                {
                    return std::unexpected(makeVulkanError(vulkanResultErrorCode(idle), idle, "vkDeviceWaitIdle"));
                }
            }
            for (auto& state : queues_)
            {
                auto fenced = waitRetirementFences(state);
                if (!fenced)
                {
                    return fenced;
                }
            }
            for (auto& state : queues_)
            {
                collectQueue(state);
            }
#ifdef _WIN32
            signalCompletedTimelineExports();
#endif
            return {};
        }

#ifdef _WIN32
        [[nodiscard]] Result<HANDLE> exportSemaphoreHandle(VkSemaphore semaphore)
        {
            const VkSemaphoreGetWin32HandleInfoKHR info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
                .pNext = {},
                .semaphore = semaphore,
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };
            HANDLE handle = nullptr;
            const auto exported = table_->vkGetSemaphoreWin32HandleKHR(device_, &info, &handle);
            if (exported != VK_SUCCESS || handle == nullptr)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(exported), exported, "vkGetSemaphoreWin32HandleKHR"));
            }
            return handle;
        }

        [[nodiscard]] Status createExportSignalWake()
        {
            if (exportSignalWake_ == VK_NULL_HANDLE)
            {
                auto wake = createQueueTimeline(false, "RHI.exportSignalWake", 0);
                if (!wake)
                {
                    return std::unexpected(std::move(wake.error()));
                }
                exportSignalWake_ = *wake;
            }
            return {};
        }

        [[nodiscard]] Status queueTimelineExportSignal(PendingTimelineExport pending)
        {
            const std::scoped_lock lock(exportSignalMutex_);
            if (table_->vkSignalSemaphore == nullptr || table_->vkWaitSemaphores == nullptr ||
                table_->vkGetSemaphoreCounterValue == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan timeline export requires the timeline semaphore entry points.",
                });
            }
            if (auto wake = createExportSignalWake(); !wake)
            {
                return wake;
            }
            try
            {
                pendingExports_.push_back(pending);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to retain an exported Vulkan timeline semaphore.",
                });
            }
            if (!exportSignaller_.joinable())
            {
                try
                {
                    exportSignaller_ = std::thread([this] { runTimelineExportSignaller(); });
                }
                catch (...)
                {
                    pendingExports_.pop_back();
                    return std::unexpected(Error{
                        ErrorCode::outOfMemory,
                        0,
                        "Failed to start the Vulkan timeline export signaller.",
                    });
                }
            }
            wakeTimelineExportSignallerLocked();
            return {};
        }

        void wakeTimelineExportSignallerLocked() noexcept
        {
            if (exportSignalWake_ != VK_NULL_HANDLE)
            {
                const VkSemaphoreSignalInfo signal{
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                    .pNext = {},
                    .semaphore = exportSignalWake_,
                    .value = exportSignalWakeValue_ + 1,
                };
                if (table_->vkSignalSemaphore(device_, &signal) == VK_SUCCESS)
                {
                    ++exportSignalWakeValue_;
                }
            }
            exportSignalChanged_.notify_one();
        }

        [[nodiscard]] VkResult signalCompletedTimelineExportsLocked() noexcept
        {
            std::array<std::optional<std::uint64_t>, queueCount> completed{};
            VkResult failure = VK_SUCCESS;
            std::size_t kept = 0;
            for (const auto& pending : pendingExports_)
            {
                auto& reached = completed[pending.queue];
                if (!reached.has_value() && failure == VK_SUCCESS)
                {
                    std::uint64_t value = 0;
                    const auto queried =
                        table_->vkGetSemaphoreCounterValue(device_, queues_[pending.queue].timeline, &value);
                    if (queried == VK_SUCCESS)
                    {
                        reached = value;
                    }
                    else
                    {
                        failure = queried;
                    }
                }
                if (reached.has_value() && *reached >= pending.value && failure == VK_SUCCESS)
                {
                    const VkSemaphoreSignalInfo signal{
                        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                        .pNext = {},
                        .semaphore = pending.semaphore,
                        .value = pending.value,
                    };
                    const auto signalled = table_->vkSignalSemaphore(device_, &signal);
                    if (signalled == VK_SUCCESS)
                    {
                        table_->vkDestroySemaphore(device_, pending.semaphore, nullptr);
                        continue;
                    }
                    failure = signalled;
                }
                pendingExports_[kept++] = pending;
            }
            pendingExports_.resize(kept);
            return failure;
        }

        void signalCompletedTimelineExports() noexcept
        {
            const std::scoped_lock lock(exportSignalMutex_);
            if (!pendingExports_.empty())
            {
                static_cast<void>(signalCompletedTimelineExportsLocked());
            }
        }

        void runTimelineExportSignaller() noexcept
        {
            std::unique_lock lock(exportSignalMutex_);
            while (!exportSignallerStopping_)
            {
                const auto signalled = signalCompletedTimelineExportsLocked();
                if (signalled == VK_ERROR_DEVICE_LOST)
                {
                    break;
                }
                if (signalled != VK_SUCCESS)
                {
                    exportSignalChanged_.wait_for(lock, std::chrono::nanoseconds{exportSignalPollNanoseconds});
                    continue;
                }
                if (pendingExports_.empty())
                {
                    exportSignalChanged_.wait(lock,
                                              [this] { return exportSignallerStopping_ || !pendingExports_.empty(); });
                    continue;
                }

                std::array<VkSemaphore, queueCount + 1> semaphores{};
                std::array<std::uint64_t, queueCount + 1> values{};
                std::uint32_t count = 0;
                for (std::size_t index = 0; index < queueCount; ++index)
                {
                    auto target = std::numeric_limits<std::uint64_t>::max();
                    for (const auto& pending : pendingExports_)
                    {
                        if (pending.queue == index)
                        {
                            target = std::min(target, pending.value);
                        }
                    }
                    if (target != std::numeric_limits<std::uint64_t>::max())
                    {
                        semaphores[count] = queues_[index].timeline;
                        values[count] = target;
                        ++count;
                    }
                }
                semaphores[count] = exportSignalWake_;
                values[count] = exportSignalWakeValue_ + 1;
                ++count;
                lock.unlock();
                const VkSemaphoreWaitInfo waitInfo{
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                    .pNext = {},
                    .flags = VK_SEMAPHORE_WAIT_ANY_BIT,
                    .semaphoreCount = count,
                    .pSemaphores = semaphores.data(),
                    .pValues = values.data(),
                };
                const auto waited = table_->vkWaitSemaphores(device_, &waitInfo, exportSignalPollNanoseconds);
                lock.lock();
                if (waited == VK_ERROR_DEVICE_LOST)
                {
                    break;
                }
                if (waited != VK_SUCCESS && waited != VK_TIMEOUT && !exportSignallerStopping_)
                {
                    exportSignalChanged_.wait_for(lock, std::chrono::nanoseconds{exportSignalPollNanoseconds});
                }
            }
        }

        void stopTimelineExportSignaller() noexcept
        {
            {
                const std::scoped_lock lock(exportSignalMutex_);
                exportSignallerStopping_ = true;
                wakeTimelineExportSignallerLocked();
            }
            if (exportSignaller_.joinable())
            {
                exportSignaller_.join();
            }
            const std::scoped_lock lock(exportSignalMutex_);
            if (!pendingExports_.empty())
            {
                static_cast<void>(signalCompletedTimelineExportsLocked());
            }
            for (const auto& pending : pendingExports_)
            {
                table_->vkDestroySemaphore(device_, pending.semaphore, nullptr);
            }
            pendingExports_.clear();
            if (exportSignalWake_ != VK_NULL_HANDLE)
            {
                table_->vkDestroySemaphore(device_, exportSignalWake_, nullptr);
                exportSignalWake_ = VK_NULL_HANDLE;
            }
        }

        [[nodiscard]] Result<VkSemaphore> importSharedTimelineSemaphore(SharedTimelinePointView point)
        {
            VkSemaphoreTypeCreateInfo typeInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .pNext = {},
                .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                .initialValue = point.value - 1,
            };
            const VkSemaphoreCreateInfo create{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = &typeInfo,
                .flags = {},
            };
            VkSemaphore imported = VK_NULL_HANDLE;
            const auto created = table_->vkCreateSemaphore(device_, &create, nullptr, &imported);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(
                    vulkanResultErrorCode(created), created, "vkCreateSemaphore(shared timeline import)"));
            }
            const VkImportSemaphoreWin32HandleInfoKHR importInfo{
                .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
                .pNext = {},
                .semaphore = imported,
                .flags = {},
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
                .handle = std::bit_cast<HANDLE>(point.handle.value),
                .name = {},
            };
            const auto importedResult = table_->vkImportSemaphoreWin32HandleKHR(device_, &importInfo);
            if (importedResult != VK_SUCCESS)
            {
                table_->vkDestroySemaphore(device_, imported, nullptr);
                return std::unexpected(makeVulkanError(importedResult == VK_ERROR_INVALID_EXTERNAL_HANDLE
                                                           ? ErrorCode::invalidArgument
                                                           : vulkanResultErrorCode(importedResult),
                                                       importedResult,
                                                       "vkImportSemaphoreWin32HandleKHR"));
            }
            return imported;
        }

        [[nodiscard]] Status submitSharedTimelineWait(VulkanQueueState& state,
                                                      std::shared_ptr<ImportedSemaphoreKeepAlive> keepAlive,
                                                      std::uint64_t sharedValue,
                                                      std::uint64_t signalValue)
        {
            const std::array waitInfos{
                semaphoreSubmitInfo(keepAlive->semaphore, sharedValue),
                semaphoreSubmitInfo(state.timeline, state.unchainedSharedWaitValue),
            };
            const auto signalInfo = semaphoreSubmitInfo(state.timeline, signalValue);
            const VkSubmitInfo2 submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                .pNext = {},
                .flags = {},
                .waitSemaphoreInfoCount = static_cast<std::uint32_t>(1U + sharedWaitChainCount(state)),
                .pWaitSemaphoreInfos = waitInfos.data(),
                .commandBufferInfoCount = {},
                .pCommandBufferInfos = {},
                .signalSemaphoreInfoCount = 1,
                .pSignalSemaphoreInfos = &signalInfo,
            };
            const std::scoped_lock retirementLock(state.retirementMutex);
            try
            {
                state.pending.push_back(VulkanPendingEntry{
                    .timelineValue = signalValue,
                    .retained = {std::move(keepAlive)},
                });
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to retain an imported Vulkan timeline semaphore.",
                });
            }
            const auto submitted = table_->vkQueueSubmit2(state.gate->queue, 1, &submitInfo, VK_NULL_HANDLE);
            if (submitted != VK_SUCCESS)
            {
                state.pending.pop_back();
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(submitted), submitted, "vkQueueSubmit2"));
            }
            return {};
        }
#endif

    public:
        [[nodiscard]] Result<SharedTimelinePoint> exportTimelinePoint(TimelinePoint point)
        {
#ifndef _WIN32
            static_cast<void>(point);
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "exportTimelinePoint uses Windows NT handles. Linux FD sharing is not implemented.",
            });
#else
            if (!exportableTimelines_ || table_ == nullptr || table_->vkGetSemaphoreWin32HandleKHR == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "exportTimelinePoint is not implemented on the Vulkan backend yet.",
                });
            }
            auto valid = checkSubmittedPoint(point);
            if (!valid)
            {
                return std::unexpected(std::move(valid.error()));
            }

            const auto& state = queue(point.queueType());
            const auto completed = counterValue(state);
            if (!completed)
            {
                return std::unexpected(Error{
                    ErrorCode::deviceLost,
                    0,
                    "The Vulkan timeline semaphore could not be queried.",
                });
            }
            const bool complete = *completed >= point.value();
            auto semaphore =
                createQueueTimeline(true, "RHI.exportedTimeline", complete ? point.value() : point.value() - 1);
            if (!semaphore)
            {
                return std::unexpected(std::move(semaphore.error()));
            }
            auto handle = exportSemaphoreHandle(*semaphore);
            if (!handle)
            {
                table_->vkDestroySemaphore(device_, *semaphore, nullptr);
                return std::unexpected(std::move(handle.error()));
            }
            if (complete)
            {
                table_->vkDestroySemaphore(device_, *semaphore, nullptr);
            }
            else if (auto queued = queueTimelineExportSignal(PendingTimelineExport{
                         .semaphore = *semaphore,
                         .queue = queueIndex(point.queueType()),
                         .value = point.value(),
                     });
                     !queued)
            {
                static_cast<void>(CloseHandle(*handle));
                table_->vkDestroySemaphore(device_, *semaphore, nullptr);
                return std::unexpected(std::move(queued.error()));
            }
            return SharedTimelinePoint{
                .handle = SharedHandle::adopt(SharedHandleView{
                    .type = SharedHandleType::windowsNt,
                    .value = reinterpret_cast<std::uintptr_t>(*handle),
                }),
                .value = point.value(),
            };
#endif
        }

        [[nodiscard]] Result<TimelinePoint> enqueueSharedTimelineWait(QueueType queueType,
                                                                      SharedTimelinePointView point)
        {
#ifndef _WIN32
            static_cast<void>(queueType);
            static_cast<void>(point);
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "enqueueSharedTimelineWait uses Windows NT handles. Linux FD sharing is not implemented.",
            });
#else
            if (!exportableTimelines_ || table_ == nullptr || table_->vkImportSemaphoreWin32HandleKHR == nullptr ||
                table_->vkQueueSubmit2 == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "enqueueSharedTimelineWait is not implemented on the Vulkan backend yet.",
                });
            }
            if (!isValidQueueType(queueType) || !point.isValid() || point.handle.type != SharedHandleType::windowsNt ||
                point.value == std::numeric_limits<std::uint64_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A shared timeline wait requires a valid queue and Windows NT fence/value pair.",
                });
            }

            auto imported = importSharedTimelineSemaphore(point);
            if (!imported)
            {
                return std::unexpected(std::move(imported.error()));
            }
            auto keepAlive = std::make_shared<ImportedSemaphoreKeepAlive>();
            keepAlive->device = device_;
            keepAlive->table = table_;
            keepAlive->semaphore = *imported;

            auto& state = queue(queueType);
            if (state.gate == nullptr || state.timeline == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan shared timeline waits require an initialized queue timeline.",
                });
            }
            const std::scoped_lock submissionLock(state.gate->mutex);
            if (!hasUnreservedTimelineSlot(state))
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "The queue timeline is exhausted and cannot be reused safely.",
                });
            }
            const auto submittedValue = state.nextValue;
            auto submitted = submitSharedTimelineWait(state, std::move(keepAlive), point.value, submittedValue);
            if (!submitted)
            {
                return std::unexpected(std::move(submitted.error()));
            }
            ++state.nextValue;
            state.lastSubmittedValue.store(submittedValue, std::memory_order_release);
            state.unchainedSharedWaitValue = submittedValue;
            return detail::TimelinePointAccess::make(state.type, submittedValue, deviceToken_);
#endif
        }

    private:
        [[nodiscard]] Result<VkFence> acquireRetirementFence(VulkanQueueState& state)
        {
            if (!retireOnSubmissionFence_)
            {
                return VkFence{VK_NULL_HANDLE};
            }
            if (!state.availableFences.empty())
            {
                const auto fence = state.availableFences.back();
                state.availableFences.pop_back();
                return fence;
            }
            if (table_->vkCreateFence == nullptr || table_->vkGetFenceStatus == nullptr ||
                table_->vkResetFences == nullptr || table_->vkWaitForFences == nullptr ||
                table_->vkDestroyFence == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan submission retirement fences require the fence entry points.",
                });
            }
            try
            {
                state.availableFences.reserve(state.totalFences + 1U);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to reserve Vulkan submission retirement fence storage.",
                });
            }
            const VkFenceCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = {},
                .flags = {},
            };
            VkFence fence = VK_NULL_HANDLE;
            const auto created = table_->vkCreateFence(device_, &info, nullptr, &fence);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateFence"));
            }
            ++state.totalFences;
            return fence;
        }

        [[nodiscard]] VkResult retirementFenceStatus(VkFence fence) const noexcept
        {
#ifdef RACHIS_TESTING
            if (const auto injected = test::retirementFenceStatusOverride().load(std::memory_order_acquire);
                injected != 0)
            {
                return static_cast<VkResult>(injected);
            }
#endif
            return table_->vkGetFenceStatus(device_, fence);
        }

        [[nodiscard]] bool retirementFenceSignaled(VkFence fence) const noexcept
        {
            return fence == VK_NULL_HANDLE || retirementFenceStatus(fence) == VK_SUCCESS;
        }

        void recycleRetirementFence(VulkanQueueState& state, VkFence fence) noexcept
        {
            if (fence == VK_NULL_HANDLE)
            {
                return;
            }
            if (table_->vkResetFences(device_, 1, &fence) != VK_SUCCESS)
            {
                table_->vkDestroyFence(device_, fence, nullptr);
                --state.totalFences;
                return;
            }
            state.availableFences.push_back(fence);
        }

        [[nodiscard]] Status waitRetirementFences(VulkanQueueState& state,
                                                  std::uint64_t value = std::numeric_limits<std::uint64_t>::max())
        {
            const std::scoped_lock lock(state.retirementMutex);
            for (const auto& entry : state.pending)
            {
                if (entry.timelineValue > value)
                {
                    break;
                }
                if (entry.fence == VK_NULL_HANDLE)
                {
                    continue;
                }
                const auto waited = table_->vkWaitForFences(
                    device_, 1, &entry.fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
                if (waited != VK_SUCCESS)
                {
                    return std::unexpected(makeVulkanError(vulkanResultErrorCode(waited), waited, "vkWaitForFences"));
                }
            }
            return {};
        }

        [[nodiscard]] VkResult waitRetirementFence(VkFence fence, std::uint64_t timeoutNs) const noexcept
        {
#ifdef RACHIS_TESTING
            if (const auto injected = test::retirementFenceStatusOverride().load(std::memory_order_acquire);
                injected != 0)
            {
                return injected == VK_NOT_READY ? VK_TIMEOUT : static_cast<VkResult>(injected);
            }
#endif
            return table_->vkWaitForFences(device_, 1, &fence, VK_TRUE, timeoutNs);
        }

        [[nodiscard]] Result<TimelinePointStatus> waitRetirementFencesFor(VulkanQueueState& state,
                                                                          std::uint64_t value,
                                                                          std::chrono::nanoseconds timeout)
        {
            if (!retireOnSubmissionFence_)
            {
                return TimelinePointStatus::complete;
            }
            const auto started = std::chrono::steady_clock::now();
            const std::scoped_lock lock(state.retirementMutex);
            for (const auto& entry : state.pending)
            {
                if (entry.timelineValue > value)
                {
                    break;
                }
                if (entry.fence == VK_NULL_HANDLE)
                {
                    continue;
                }
                const auto remaining = remainingTimeout(started, timeout);
                const auto polled = remaining == std::chrono::nanoseconds::zero();
                const auto waited =
                    polled ? retirementFenceStatus(entry.fence)
                           : waitRetirementFence(entry.fence, static_cast<std::uint64_t>(remaining.count()));
                if (waited == VK_NOT_READY || waited == VK_TIMEOUT)
                {
                    return TimelinePointStatus::pending;
                }
                if (waited != VK_SUCCESS)
                {
                    return std::unexpected(makeVulkanError(
                        vulkanResultErrorCode(waited), waited, polled ? "vkGetFenceStatus" : "vkWaitForFences"));
                }
            }
            return TimelinePointStatus::complete;
        }

        [[nodiscard]] std::uint64_t retiredValue(const VulkanQueueState& state, std::uint64_t counter) const noexcept
        {
            const std::scoped_lock lock(state.retirementMutex);
            for (const auto& entry : state.pending)
            {
                if (entry.timelineValue > counter)
                {
                    break;
                }
                if (entry.fence != VK_NULL_HANDLE && retirementFenceStatus(entry.fence) != VK_SUCCESS)
                {
                    return entry.timelineValue - 1;
                }
            }
            return counter;
        }

        void destroyRetirementFences(VulkanQueueState& state) noexcept
        {
            if (table_->vkDestroyFence == nullptr)
            {
                return;
            }
            for (const auto& entry : state.pending)
            {
                if (entry.fence != VK_NULL_HANDLE)
                {
                    table_->vkDestroyFence(device_, entry.fence, nullptr);
                }
            }
            for (const auto fence : state.availableFences)
            {
                table_->vkDestroyFence(device_, fence, nullptr);
            }
            state.availableFences.clear();
            state.totalFences = 0;
        }

        void recycleBridgeCommandBuffers(VulkanQueueState& state,
                                         std::span<const VkCommandBuffer> buffers,
                                         bool freeInsteadOfRecycle) noexcept
        {
            if (buffers.empty())
            {
                return;
            }
            const std::scoped_lock bridgeLock(state.bridgePoolMutex);
            if (state.bridgePool == VK_NULL_HANDLE)
            {
                return;
            }
            const bool insufficientCapacity =
                state.availableBridgeBuffers.capacity() - state.availableBridgeBuffers.size() < buffers.size();
            if (freeInsteadOfRecycle || insufficientCapacity)
            {
                if (table_ != nullptr && table_->vkFreeCommandBuffers != nullptr)
                {
                    table_->vkFreeCommandBuffers(
                        device_, state.bridgePool, static_cast<std::uint32_t>(buffers.size()), buffers.data());
                }
                state.totalBridgeBuffers =
                    state.totalBridgeBuffers >= buffers.size() ? (state.totalBridgeBuffers - buffers.size()) : 0U;
                return;
            }
            state.availableBridgeBuffers.insert(state.availableBridgeBuffers.end(), buffers.begin(), buffers.end());
        }

        [[nodiscard]] Status ensureBridgePool(VulkanQueueState& state, std::uint32_t fallbackFamily)
        {
            if (state.bridgePool != VK_NULL_HANDLE)
            {
                return {};
            }
            if (table_->vkCreateCommandPool == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan submission bridge pool creation requires vkCreateCommandPool.",
                });
            }
            const std::uint32_t familyIndex = state.family != VK_QUEUE_FAMILY_IGNORED
                                                  ? state.family
                                                  : (fallbackFamily != VK_QUEUE_FAMILY_IGNORED ? fallbackFamily : 0U);
            const VkCommandPoolCreateInfo createInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = {},
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = familyIndex,
            };
            const auto created = table_->vkCreateCommandPool(device_, &createInfo, nullptr, &state.bridgePool);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(created == VK_ERROR_OUT_OF_HOST_MEMORY || created == VK_ERROR_OUT_OF_DEVICE_MEMORY
                                        ? ErrorCode::outOfMemory
                                        : vulkanResultErrorCode(created),
                                    created,
                                    "vkCreateCommandPool(bridge)"));
            }
            return {};
        }

        [[nodiscard]] Result<VkCommandBuffer> allocateBridgeCommandBuffer(VulkanQueueState& state)
        {
            if (table_->vkAllocateCommandBuffers == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan submission bridge command allocation requires vkAllocateCommandBuffers.",
                });
            }
            try
            {
                state.availableBridgeBuffers.reserve(state.totalBridgeBuffers + 1U);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to reserve Vulkan submission bridge command recycle capacity.",
                });
            }
            const VkCommandBufferAllocateInfo allocInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = {},
                .commandPool = state.bridgePool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };
            VkCommandBuffer bridge = VK_NULL_HANDLE;
            const auto allocated = table_->vkAllocateCommandBuffers(device_, &allocInfo, &bridge);
            if (allocated != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(allocated == VK_ERROR_OUT_OF_HOST_MEMORY ||
                                                               allocated == VK_ERROR_OUT_OF_DEVICE_MEMORY
                                                           ? ErrorCode::outOfMemory
                                                           : vulkanResultErrorCode(allocated),
                                                       allocated,
                                                       "vkAllocateCommandBuffers(bridge)"));
            }
            ++state.totalBridgeBuffers;
            return bridge;
        }

        [[nodiscard]] Result<VkCommandBuffer> acquireBridgeCommandBuffer(VulkanQueueState& state)
        {
            if (state.availableBridgeBuffers.empty())
            {
                return allocateBridgeCommandBuffer(state);
            }
            const auto bridge = state.availableBridgeBuffers.back();
            state.availableBridgeBuffers.pop_back();
            return bridge;
        }

        void freeFailedBridgeCommandBuffer(VulkanQueueState& state, VkCommandBuffer bridge)
        {
            if (table_->vkFreeCommandBuffers != nullptr)
            {
                table_->vkFreeCommandBuffers(device_, state.bridgePool, 1, &bridge);
            }
            if (state.totalBridgeBuffers != 0U)
            {
                --state.totalBridgeBuffers;
            }
        }

        [[nodiscard]] Status beginBridgeCommandBuffer(VulkanQueueState& state, VkCommandBuffer bridge)
        {
            const VkCommandBufferBeginInfo beginInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = {},
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = {},
            };
            const auto begun = table_->vkBeginCommandBuffer(bridge, &beginInfo);
            if (begun != VK_SUCCESS)
            {
                freeFailedBridgeCommandBuffer(state, bridge);
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(begun), begun, "vkBeginCommandBuffer(bridge)"));
            }
            return {};
        }

        void recordBridgeQueryResets(VkCommandBuffer bridge,
                                     std::span<const PendingQueryPoolReset> queryResets,
                                     std::vector<std::shared_ptr<void>>& bridgeRetained)
        {
            for (const auto& reset : queryResets)
            {
                if (reset.keepAlive != nullptr)
                {
                    bridgeRetained.push_back(reset.keepAlive);
                }
                table_->vkCmdResetQueryPool(bridge, reset.pool, reset.firstQuery, reset.queryCount);
                if (reset.primitivesPool != VK_NULL_HANDLE)
                {
                    table_->vkCmdResetQueryPool(bridge, reset.primitivesPool, reset.firstQuery, reset.queryCount);
                }
            }
        }

        void recordBridgeLayoutTransitions(
            VkCommandBuffer bridge,
            std::span<const VulkanResourceAllocator::PendingImageBarrierTransition> transitions,
            std::vector<VkImageMemoryBarrier2>& reservedBarriers,
            std::vector<std::shared_ptr<void>>& bridgeRetained)
        {
            if (transitions.empty())
            {
                return;
            }
            for (const auto& transition : transitions)
            {
                if (transition.keepAlive != nullptr)
                {
                    bridgeRetained.push_back(transition.keepAlive);
                }
                reservedBarriers.push_back(layoutTransitionBarrier(transition));
            }
            const VkDependencyInfo dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = {},
                .dependencyFlags = {},
                .memoryBarrierCount = {},
                .pMemoryBarriers = {},
                .bufferMemoryBarrierCount = {},
                .pBufferMemoryBarriers = {},
                .imageMemoryBarrierCount = static_cast<std::uint32_t>(reservedBarriers.size()),
                .pImageMemoryBarriers = reservedBarriers.data(),
            };
            table_->vkCmdPipelineBarrier2(bridge, &dependency);
        }

        [[nodiscard]] Status endBridgeCommandBuffer(VulkanQueueState& state, VkCommandBuffer bridge)
        {
            const auto ended = table_->vkEndCommandBuffer(bridge);
            if (ended != VK_SUCCESS)
            {
                freeFailedBridgeCommandBuffer(state, bridge);
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(ended), ended, "vkEndCommandBuffer(bridge)"));
            }
            return {};
        }

        [[nodiscard]] Result<VkCommandBuffer> recordBridgeCommandBuffer(
            VulkanQueueState& state,
            std::uint32_t fallbackFamily,
            std::span<const PendingQueryPoolReset> queryResets,
            std::span<const VulkanResourceAllocator::PendingImageBarrierTransition> transitions,
            std::vector<std::shared_ptr<void>>& bridgeRetained)
        {
            if (table_ == nullptr || table_->vkBeginCommandBuffer == nullptr || table_->vkEndCommandBuffer == nullptr ||
                (!queryResets.empty() && table_->vkCmdResetQueryPool == nullptr) ||
                (!transitions.empty() && table_->vkCmdPipelineBarrier2 == nullptr))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan submission bridge commands are unavailable because the device table is incomplete.",
                });
            }

            std::vector<VkImageMemoryBarrier2> barriers;
            try
            {
                const std::size_t retainedNeeded = queryResets.size() + transitions.size();
                if (retainedNeeded != 0U)
                {
                    detail::reserveAdditional(bridgeRetained, retainedNeeded);
                }
                if (!transitions.empty())
                {
                    barriers.reserve(transitions.size());
                }
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to reserve Vulkan submission bridge command resources.",
                });
            }

            const std::scoped_lock bridgeLock(state.bridgePoolMutex);
            auto poolReady = ensureBridgePool(state, fallbackFamily);
            if (!poolReady)
            {
                return std::unexpected(std::move(poolReady.error()));
            }
            auto bridge = acquireBridgeCommandBuffer(state);
            if (!bridge)
            {
                return bridge;
            }
            auto begun = beginBridgeCommandBuffer(state, *bridge);
            if (!begun)
            {
                return std::unexpected(std::move(begun.error()));
            }
            recordBridgeQueryResets(*bridge, queryResets, bridgeRetained);
            recordBridgeLayoutTransitions(*bridge, transitions, barriers, bridgeRetained);
            auto ended = endBridgeCommandBuffer(state, *bridge);
            if (!ended)
            {
                return std::unexpected(std::move(ended.error()));
            }
            return bridge;
        }

        void collectQueue(VulkanQueueState& state) noexcept
        {
            const auto completed = counterValue(state);
            if (!completed)
            {
                return;
            }
            while (true)
            {
                VulkanPendingEntry retired;
                {
                    const std::scoped_lock lock(state.retirementMutex);
                    if (state.pending.empty() || state.pending.front().timelineValue > *completed ||
                        !retirementFenceSignaled(state.pending.front().fence))
                    {
                        break;
                    }
                    retired = std::move(state.pending.front());
                    state.pending.pop_front();
                    recycleRetirementFence(state, retired.fence);
                    recycleBridgeCommandBuffers(state, retired.bridgeBuffers, false);
                    if (retired.entry != nullptr && retired.pool != nullptr)
                    {
                        const std::scoped_lock poolLock(retired.pool->mutex);
                        retired.entry->inFlight = false;
                        retired.entry->lease = nullptr;
                        retired.pool->available.push_back(retired.entry);
                    }
                }
#ifdef RACHIS_TESTING
                detail::test::observeCommandPoolRetirement();
#endif
            }
        }

        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        const VulkanInstanceTable* instanceTable_ = nullptr;
        std::uint64_t deviceToken_ = 0;
        bool exportableTimelines_ = false;
        bool retireOnSubmissionFence_ = false;
#ifdef _WIN32
        std::mutex exportSignalMutex_{};
        std::condition_variable exportSignalChanged_{};
        std::vector<PendingTimelineExport> pendingExports_{};
        VkSemaphore exportSignalWake_ = VK_NULL_HANDLE;
        std::uint64_t exportSignalWakeValue_ = 0;
        bool exportSignallerStopping_ = false;
        std::thread exportSignaller_{};
#endif
        std::vector<std::unique_ptr<VulkanNativeQueueGate>> gates_;
        std::array<VulkanQueueState, queueCount> queues_{};
        VulkanNativeQueueGate* presentGate_ = nullptr;
        VkQueue presentQueue_ = VK_NULL_HANDLE;
        PFN_vkQueuePresentKHR vkQueuePresentKHR_ = nullptr;
    };

    VulkanQueueManager::VulkanQueueManager() = default;

    VulkanQueueManager::~VulkanQueueManager()
    {
        shutdown();
    }

    Status VulkanQueueManager::initialize(VkDevice device,
                                          const VolkDeviceTable* table,
                                          const VulkanQueueHandles& queues,
                                          std::uint64_t deviceToken,
                                          bool exportableTimelines,
                                          const VulkanInstanceTable* instanceTable) noexcept
    {
        if (device == VK_NULL_HANDLE || table == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan queue timelines require a created device table.",
            });
        }
        try
        {
            impl_ = std::make_unique<Impl>(device, table, instanceTable, deviceToken);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate Vulkan queue timeline state.",
            });
        }
        auto initialized = impl_->initialize(queues, exportableTimelines);
        if (!initialized)
        {
            impl_.reset();
            return initialized;
        }
        return {};
    }

    void VulkanQueueManager::shutdown() noexcept
    {
        impl_.reset();
    }

    Result<TimelinePoint> VulkanQueueManager::submit(
        std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers, std::span<const TimelinePoint> waits)
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot submit to an uninitialized Vulkan queue manager.",
            });
        }
        return impl_->submit(commandBuffers, waits);
    }

    Result<TimelinePointStatus> VulkanQueueManager::queryTimelinePoint(TimelinePoint point) const
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot query a timeline point on an uninitialized Vulkan queue manager.",
            });
        }
        return impl_->queryTimelinePoint(point);
    }

    Result<TimelinePointStatus> VulkanQueueManager::wait(TimelinePoint point, std::chrono::nanoseconds timeout)
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot wait on an uninitialized Vulkan queue manager.",
            });
        }
        return impl_->wait(point, timeout);
    }

    Status VulkanQueueManager::wait(TimelinePoint point)
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot wait on an uninitialized Vulkan queue manager.",
            });
        }
        return impl_->wait(point);
    }

    Status VulkanQueueManager::waitIdle(QueueType queueType)
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan queue waitIdle requested a queue that was not created.",
            });
        }
        return impl_->waitIdle(queueType);
    }

    Status VulkanQueueManager::waitPresentIdle()
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan present queue waitIdle requested a queue that was not created.",
            });
        }
        return impl_->waitPresentIdle();
    }

    Status VulkanQueueManager::waitIdle()
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan device waitIdle requested a device that was not created.",
            });
        }
        return impl_->waitIdle();
    }

    std::uint64_t VulkanQueueManager::completedValue(QueueType queueType) const noexcept
    {
        return impl_ != nullptr ? impl_->completedValue(queueType) : 0;
    }

    void VulkanQueueManager::collectGarbage() noexcept
    {
        if (impl_ != nullptr)
        {
            impl_->collectGarbage();
        }
    }

    Status VulkanQueueManager::validateSubmittedPoint(TimelinePoint point) const
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot validate a timeline point on an uninitialized Vulkan queue manager.",
            });
        }
        return impl_->validateSubmittedPoint(point);
    }

    Result<TimelinePoint> VulkanQueueManager::submitGraphicsPresentationSync(const PresentationSync& sync)
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot submit presentation sync on an uninitialized Vulkan queue manager.",
            });
        }
        return impl_->submitGraphicsPresentationSync(sync);
    }

    void VulkanQueueManager::drainGraphicsAcquire(VkSemaphore semaphore)
    {
        if (impl_ != nullptr)
        {
            impl_->drainGraphicsAcquire(semaphore);
        }
    }

    Status VulkanQueueManager::submitPresentPresentationSync(const PresentationSync& sync)
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot submit present-queue presentation sync on an uninitialized Vulkan queue manager.",
            });
        }
        return impl_->submitPresentPresentationSync(sync);
    }

    Result<SharedTimelinePoint> VulkanQueueManager::exportTimelinePoint(TimelinePoint point)
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot export a timeline point on an uninitialized Vulkan queue manager.",
            });
        }
        return impl_->exportTimelinePoint(point);
    }

    Result<TimelinePoint> VulkanQueueManager::enqueueSharedTimelineWait(QueueType queueType,
                                                                        SharedTimelinePointView point)
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot wait on a shared timeline on an uninitialized Vulkan queue manager.",
            });
        }
        return impl_->enqueueSharedTimelineWait(queueType, point);
    }

    Result<VulkanQueueManager::PresentResult> VulkanQueueManager::presentOnPresentQueue(const VkPresentInfoKHR& info)
    {
        if (impl_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot present on an uninitialized Vulkan queue manager.",
            });
        }
        return impl_->presentOnPresentQueue(info);
    }

} // namespace rachis::vulkan
