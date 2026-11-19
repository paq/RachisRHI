#pragma once

#include "Common/AccelerationStructureExecution.h"
#include "Common/HandleTable.h"
#include "Internal/BackendCommandBuffer.h"
#include "Internal/VulkanRuntime.h"
#include "Rachis/AccelerationStructure.h"
#include "Rachis/Pipeline.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <volk.h>

namespace rachis::vulkan
{

    class VulkanCommandBuffer;
    class VulkanResourceAllocator;

    struct VulkanAccelerationStructureVersion final
    {
        AccelerationStructureBuildFlag buildFlags = AccelerationStructureBuildFlag::none;
        std::uint32_t instanceCount = 0;
        std::vector<std::shared_ptr<void>> dependencies;
        std::vector<std::shared_ptr<detail::AccelerationStructureExecutionTracker>> dependencyTrackers;
    };

    struct VulkanAccelerationStructureVersionSlot final
    {
        std::mutex mutex;
        std::shared_ptr<VulkanAccelerationStructureVersion> current;
    };

    struct VulkanAccelerationStructurePublication final
    {
        std::weak_ptr<VulkanAccelerationStructureVersionSlot> slot;
        std::weak_ptr<VulkanAccelerationStructureVersion> published;
        std::shared_ptr<VulkanAccelerationStructureVersion> previous;
    };

    struct VulkanAccelerationStructureUse final
    {
        std::shared_ptr<VulkanAccelerationStructureVersionSlot> version;
        std::shared_ptr<detail::AccelerationStructureExecutionTracker> executionTracker;
    };

    void rollBackVulkanAccelerationStructurePublications(
        std::vector<VulkanAccelerationStructurePublication>& publications) noexcept;

    [[nodiscard]] Status retainVulkanAccelerationStructureConsumers(
        VulkanCommandBuffer& commandBuffer, std::span<const VulkanAccelerationStructureUse> uses);

    class VulkanCommandUse final
    {
    public:
        VulkanCommandUse() noexcept = default;

        VulkanCommandUse(std::shared_ptr<std::atomic<std::uint64_t>> counter, std::shared_ptr<void> owner) noexcept
            : counter_(std::move(counter)), owner_(std::move(owner))
        {
            if (counter_ != nullptr)
            {
                counter_->fetch_add(1, std::memory_order_acq_rel);
            }
        }

        ~VulkanCommandUse()
        {
            reset();
        }

        VulkanCommandUse(const VulkanCommandUse&) = delete;
        VulkanCommandUse& operator=(const VulkanCommandUse&) = delete;

        VulkanCommandUse(VulkanCommandUse&& other) noexcept
            : counter_(std::move(other.counter_)), owner_(std::move(other.owner_))
        {
        }

        VulkanCommandUse& operator=(VulkanCommandUse&& other) noexcept
        {
            if (this != &other)
            {
                reset();
                counter_ = std::move(other.counter_);
                owner_ = std::move(other.owner_);
            }
            return *this;
        }

        void reset() noexcept
        {
            owner_.reset();
            if (counter_ != nullptr)
            {
                counter_->fetch_sub(1, std::memory_order_acq_rel);
                counter_.reset();
            }
        }

        [[nodiscard]] const void* owner() const noexcept
        {
            return owner_.get();
        }

    private:
        std::shared_ptr<std::atomic<std::uint64_t>> counter_;
        std::shared_ptr<void> owner_;
    };

    struct VulkanCommandUses final
    {
        static constexpr std::size_t recentUseWindow = 8;

        void add(VulkanCommandUse use)
        {
            if (use.owner() == nullptr)
            {
                return;
            }
            const auto recent = (std::min)(entries.size(), recentUseWindow);
            for (auto index = entries.size() - recent; index < entries.size(); ++index)
            {
                if (entries[index].owner() == use.owner())
                {
                    return;
                }
            }
            entries.push_back(std::move(use));
        }

        std::vector<VulkanCommandUse> entries;
    };

    struct VulkanCommandEntry final
    {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VulkanCommandBuffer* lease = nullptr;
        bool inFlight = false;
    };

    struct VulkanCommandPoolState final
    {
        VulkanCommandPoolState() = default;
        ~VulkanCommandPoolState()
        {
            destroyNative();
        }
        VulkanCommandPoolState(const VulkanCommandPoolState&) = delete;
        VulkanCommandPoolState& operator=(const VulkanCommandPoolState&) = delete;
        VulkanCommandPoolState(VulkanCommandPoolState&&) = delete;
        VulkanCommandPoolState& operator=(VulkanCommandPoolState&&) = delete;

        mutable std::mutex mutex;
        VkDevice device = VK_NULL_HANDLE;
        const VolkDeviceTable* table = nullptr;
        PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabel = nullptr;
        PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabel = nullptr;
        PFN_vkCmdInsertDebugUtilsLabelEXT cmdInsertDebugUtilsLabel = nullptr;
        VulkanResourceAllocator* resources = nullptr;
        VkCommandPool pool = VK_NULL_HANDLE;
        QueueType queueType = QueueType::graphics;
        std::uint32_t family = VK_QUEUE_FAMILY_IGNORED;
        CommandPoolFlag flags = CommandPoolFlag::none;
        std::string label;
        std::vector<std::unique_ptr<VulkanCommandEntry>> entries;
        std::vector<VulkanCommandEntry*> available;
        std::vector<VulkanCommandEntry*> quarantined;
        std::atomic<bool> trackedForShutdown{false};

        void destroyNative() noexcept;
    };

    struct VulkanSubmittedQueryPoolState final
    {
        std::mutex mutex;
        VkQueryPool pool = VK_NULL_HANDLE;
        VkQueryPool primitivesPool = VK_NULL_HANDLE;
        std::uint32_t queryCount = 0;
        std::vector<std::uint8_t> dirtySlots;
    };

    inline constexpr std::uint8_t kVulkanQuerySlotTouched = 1U << 0U;
    inline constexpr std::uint8_t kVulkanQuerySlotNeedsExternalReset = 1U << 1U;
    inline constexpr std::uint8_t kVulkanQuerySlotDirtyAtEnd = 1U << 2U;

    struct VulkanConditionalRendering final
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        bool inverted = false;
    };

    struct VulkanCommandQueryPoolUse final
    {
        std::shared_ptr<VulkanSubmittedQueryPoolState> poolState;
        std::shared_ptr<void> keepAlive;
        std::vector<std::uint8_t> slotFlags;
        std::uint32_t minTouchedSlot = 0;
        std::uint32_t maxTouchedSlot = 0;
        bool hasTouchedSlots = false;
    };

    class VulkanCommandBuffer final : public detail::BackendCommandBuffer
    {
    public:
        VulkanCommandBuffer(std::shared_ptr<VulkanCommandPoolState> owner,
                            VulkanCommandEntry& entry,
                            std::uint64_t deviceToken) noexcept;
        ~VulkanCommandBuffer() override;

        VulkanCommandBuffer(const VulkanCommandBuffer&) = delete;
        VulkanCommandBuffer& operator=(const VulkanCommandBuffer&) = delete;
        VulkanCommandBuffer(VulkanCommandBuffer&&) = delete;
        VulkanCommandBuffer& operator=(VulkanCommandBuffer&&) = delete;

        [[nodiscard]] QueueType queueType() const noexcept override;
        [[nodiscard]] std::uint64_t deviceToken() const noexcept override;
        [[nodiscard]] detail::RecordingStateLock lockRecordingState() const override;
        [[nodiscard]] VulkanCommandPoolState* owner() const noexcept
        {
            return owner_.get();
        }
        [[nodiscard]] const std::shared_ptr<VulkanCommandPoolState>& ownerShared() const noexcept
        {
            return owner_;
        }
        [[nodiscard]] VulkanCommandEntry* entry() const noexcept
        {
            return entry_;
        }
        [[nodiscard]] VulkanCommandEntry* release() noexcept;
        [[nodiscard]] VkCommandBuffer native() const noexcept;
        [[nodiscard]] Status validateReset(bool requireIndividualOptIn) const;
        [[nodiscard]] Status resetRecording();
        void clearRecordingState() noexcept;
        [[nodiscard]] std::vector<std::shared_ptr<void>>& retained() noexcept
        {
            return retained_;
        }
        [[nodiscard]] std::vector<std::shared_ptr<void>> takeRetained() noexcept
        {
            commandUses_.reset();
            return std::move(retained_);
        }
        [[nodiscard]] std::vector<VulkanAccelerationStructurePublication>& accelerationStructurePublications() noexcept
        {
            return accelerationStructurePublications_;
        }
        [[nodiscard]] std::vector<VulkanAccelerationStructurePublication>
        takeAccelerationStructurePublications() noexcept
        {
            return std::move(accelerationStructurePublications_);
        }
        [[nodiscard]] std::uint64_t accelerationStructureRecordingId() noexcept;
        [[nodiscard]] Result<VulkanCommandUses*> reserveCommandUses(std::size_t additional);
        [[nodiscard]] const std::vector<VulkanCommandQueryPoolUse>& queryPoolUses() const noexcept
        {
            return queryPoolUses_;
        }
        [[nodiscard]] Status reserveQueryPoolUse(const std::shared_ptr<VulkanSubmittedQueryPoolState>& poolState,
                                                 const std::shared_ptr<void>& keepAlive);
        void recordQuerySlotReset(const std::shared_ptr<VulkanSubmittedQueryPoolState>& poolState,
                                  std::uint32_t firstQuery,
                                  std::uint32_t queryCount) noexcept;
        void recordQuerySlotWriteAfterLocalReset(const std::shared_ptr<VulkanSubmittedQueryPoolState>& poolState,
                                                 std::uint32_t queryIndex) noexcept;
        void recordQuerySlotWriteWithoutLocalReset(const std::shared_ptr<VulkanSubmittedQueryPoolState>& poolState,
                                                   std::uint32_t queryIndex) noexcept;
        [[nodiscard]] std::optional<VulkanConditionalRendering>& conditionalRendering() noexcept
        {
            return conditionalRendering_;
        }
#ifdef RACHIS_TESTING
        void setInFlightForTesting(bool inFlight) noexcept;
#endif

    private:
        void abandon() noexcept;

        std::shared_ptr<VulkanCommandPoolState> owner_;
        VulkanCommandEntry* entry_ = nullptr;
        std::uint64_t deviceToken_ = 0;
        std::vector<std::shared_ptr<void>> retained_;
        std::vector<VulkanAccelerationStructurePublication> accelerationStructurePublications_;
        std::uint64_t accelerationStructureRecordingId_ = 0;
        std::weak_ptr<VulkanCommandUses> commandUses_;
        std::vector<VulkanCommandQueryPoolUse> queryPoolUses_;
        std::optional<VulkanConditionalRendering> conditionalRendering_;
    };

    class VulkanCommandAllocator final
    {
    public:
        VulkanCommandAllocator() = default;
        ~VulkanCommandAllocator();

        VulkanCommandAllocator(const VulkanCommandAllocator&) = delete;
        VulkanCommandAllocator& operator=(const VulkanCommandAllocator&) = delete;
        VulkanCommandAllocator(VulkanCommandAllocator&&) = delete;
        VulkanCommandAllocator& operator=(VulkanCommandAllocator&&) = delete;

        void initialize(VkDevice device,
                        const VolkDeviceTable* table,
                        const VulkanQueueHandles& queues,
                        std::uint64_t deviceToken,
                        VulkanResourceAllocator* resources = nullptr,
                        const VulkanInstanceTable* instanceTable = nullptr) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc& desc);
        [[nodiscard]] Status destroyCommandPool(CommandPoolHandle commandPool);
        [[nodiscard]] Status resetCommandPool(CommandPoolHandle commandPool);
        [[nodiscard]] Result<std::unique_ptr<detail::BackendCommandBuffer>> acquireCommandBuffer(
            CommandPoolHandle commandPool);
        [[nodiscard]] Status resetCommandBuffer(detail::BackendCommandBuffer& commandBuffer);
        [[nodiscard]] Status beginDebugGroup(detail::BackendCommandBuffer& commandBuffer, const DebugMarkerDesc& desc);
        [[nodiscard]] Status endDebugGroup(detail::BackendCommandBuffer& commandBuffer);
        [[nodiscard]] Status insertDebugMarker(detail::BackendCommandBuffer& commandBuffer,
                                               const DebugMarkerDesc& desc);
#ifdef RACHIS_TESTING
        [[nodiscard]] Status markCommandBufferInFlight(detail::BackendCommandBuffer& commandBuffer, bool inFlight);
#endif

    private:
        [[nodiscard]] std::uint32_t familyFor(QueueType queueType) const noexcept;

        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        const VulkanInstanceTable* instanceTable_ = nullptr;
        VulkanQueueHandles queues_{};
        VulkanResourceAllocator* resources_ = nullptr;
        std::uint64_t deviceToken_ = 0;
        mutable std::mutex mutex_;
        detail::HandleTable<std::shared_ptr<VulkanCommandPoolState>, CommandPoolHandle> pools_;
    };

#ifdef RACHIS_TESTING
    void bindVulkanCommandAllocator(VulkanCommandAllocator* allocator) noexcept;
    void unbindVulkanCommandAllocator(VulkanCommandAllocator* allocator) noexcept;
    [[nodiscard]] Status markVulkanCommandBufferInFlight(CommandBuffer& commandBuffer, bool inFlight);
#endif

} // namespace rachis::vulkan
