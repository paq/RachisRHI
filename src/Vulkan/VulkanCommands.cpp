#include "Internal/VulkanCommands.h"

#include "Common/VectorReserve.h"
#include "Internal/CommandBufferAccess.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanResources.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace rachis::vulkan
{
    namespace
    {

        detail::AccelerationStructureRecordingSequence vulkanAccelerationStructureRecordings;

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

        [[nodiscard]] constexpr bool isValidCommandPoolFlags(CommandPoolFlag flags) noexcept
        {
            constexpr auto allowed = static_cast<std::uint32_t>(CommandPoolFlag::allowIndividualReset) |
                                     static_cast<std::uint32_t>(CommandPoolFlag::unretainedResources);
            return (static_cast<std::uint32_t>(flags) & ~allowed) == 0;
        }

        [[nodiscard]] bool validColor(const DebugColor& color) noexcept
        {
            const std::array components{color.red, color.green, color.blue, color.alpha};
            return std::ranges::all_of(components, [](float component) {
                return std::isfinite(component) && component >= 0.0F && component <= 1.0F;
            });
        }

        [[nodiscard]] Status validateMarkerDesc(const DebugMarkerDesc& desc)
        {
            if (desc.label.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A debug marker label cannot be empty.",
                });
            }
            if (desc.label.size() > maxDebugLabelBytes)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A debug marker label exceeds the portable byte limit.",
                });
            }
            if (desc.label.contains('\0'))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A debug marker label cannot contain an embedded null byte.",
                });
            }
            if (desc.color && !validColor(*desc.color))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Debug marker color components must be finite values in the inclusive range [0, 1].",
                });
            }
            return {};
        }

        [[nodiscard]] Status beginRecording(VulkanCommandPoolState& pool, VulkanCommandEntry& entry)
        {
            if (pool.table == nullptr || pool.table->vkBeginCommandBuffer == nullptr || entry.command == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Command-buffer recording is unavailable because the Vulkan device table is incomplete.",
                });
            }
            const VkCommandBufferBeginInfo begin{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = {},
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = {},
            };
            const auto began = pool.table->vkBeginCommandBuffer(entry.command, &begin);
            if (began != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(began), began, "vkBeginCommandBuffer"));
            }
            return {};
        }

        [[nodiscard]] Status resetNative(VulkanCommandPoolState& pool, VulkanCommandEntry& entry)
        {
            if (pool.table == nullptr || pool.table->vkResetCommandBuffer == nullptr || entry.command == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Command-buffer reset is unavailable because the Vulkan device table is incomplete.",
                });
            }
            const auto reset = pool.table->vkResetCommandBuffer(entry.command, 0);
            if (reset != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(reset), reset, "vkResetCommandBuffer"));
            }
            return {};
        }

        [[nodiscard]] bool hasLiveCommand(const VulkanCommandEntry* entry) noexcept
        {
            return entry != nullptr && entry->command != VK_NULL_HANDLE;
        }

        void returnQuarantinedEntries(VulkanCommandPoolState& pool) noexcept
        {
            const auto recovered = std::ranges::partition(
                pool.quarantined, [](const VulkanCommandEntry* entry) noexcept { return !hasLiveCommand(entry); });
            pool.available.insert(pool.available.end(), recovered.begin(), recovered.end());
            pool.quarantined.erase(recovered.begin(), recovered.end());
        }

        [[nodiscard]] VulkanCommandEntry* takeReusableEntry(VulkanCommandPoolState& pool) noexcept
        {
            if (!pool.available.empty())
            {
                auto* entry = pool.available.back();
                pool.available.pop_back();
                return entry;
            }
            const auto quarantined = std::ranges::find_if(pool.quarantined, hasLiveCommand);
            if (quarantined == pool.quarantined.end())
            {
                return nullptr;
            }
            auto* entry = *quarantined;
            pool.quarantined.erase(quarantined);
            return entry;
        }

        void emitLabel(const VulkanCommandPoolState& pool,
                       VkCommandBuffer command,
                       const DebugMarkerDesc& desc,
                       bool beginGroup) noexcept
        {
            if (command == VK_NULL_HANDLE)
            {
                return;
            }
            std::string const name(desc.label);
            VkDebugUtilsLabelEXT label{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                .pNext = {},
                .pLabelName = name.c_str(),
                .color = {0.0F, 0.0F, 0.0F, 1.0F},
            };
            if (desc.color)
            {
                label.color[0] = desc.color->red;
                label.color[1] = desc.color->green;
                label.color[2] = desc.color->blue;
                label.color[3] = desc.color->alpha;
            }
            if (beginGroup && pool.cmdBeginDebugUtilsLabel != nullptr)
            {
                pool.cmdBeginDebugUtilsLabel(command, &label);
            }
            else if (!beginGroup && pool.cmdInsertDebugUtilsLabel != nullptr)
            {
                pool.cmdInsertDebugUtilsLabel(command, &label);
            }
        }

        void closeAbandonedDebugGroups(const VulkanCommandPoolState& pool,
                                       VulkanCommandEntry& entry,
                                       detail::DebugCommandState& state) noexcept
        {
            if (entry.command == VK_NULL_HANDLE || pool.cmdEndDebugUtilsLabel == nullptr)
            {
                state.groupDepth = 0;
                return;
            }
            while (state.groupDepth != 0)
            {
                pool.cmdEndDebugUtilsLabel(entry.command);
                --state.groupDepth;
            }
        }

        [[nodiscard]] VulkanCommandBuffer* asVulkanCommand(detail::BackendCommandBuffer& commandBuffer,
                                                           std::uint64_t deviceToken) noexcept
        {
            if (commandBuffer.kind() != detail::BackendCommandBufferKind::vulkan || deviceToken == 0 ||
                commandBuffer.deviceToken() != deviceToken)
            {
                return nullptr;
            }
            auto* native = static_cast<VulkanCommandBuffer*>(&commandBuffer);
            if (native->owner() == nullptr || native->entry() == nullptr)
            {
                return nullptr;
            }
            return native;
        }

    } // namespace

    void VulkanCommandPoolState::destroyNative() noexcept
    {
        const std::scoped_lock lock(mutex);
        if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroyCommandPool != nullptr &&
            pool != VK_NULL_HANDLE)
        {
            table->vkDestroyCommandPool(device, pool, nullptr);
        }
        pool = VK_NULL_HANDLE;
        for (auto& entry : entries)
        {
            if (entry != nullptr)
            {
                if (resources != nullptr)
                {
                    resources->discardCommandLayouts(entry->command);
                }
                entry->command = VK_NULL_HANDLE;
            }
        }
        resources = nullptr;
        cmdBeginDebugUtilsLabel = nullptr;
        cmdEndDebugUtilsLabel = nullptr;
        cmdInsertDebugUtilsLabel = nullptr;
    }

    VulkanCommandBuffer::VulkanCommandBuffer(std::shared_ptr<VulkanCommandPoolState> owner,
                                             VulkanCommandEntry& entry,
                                             std::uint64_t deviceToken) noexcept
        : detail::BackendCommandBuffer(detail::BackendCommandBufferKind::vulkan), owner_(std::move(owner)),
          entry_(&entry), deviceToken_(deviceToken)
    {
    }

    VulkanCommandBuffer::~VulkanCommandBuffer()
    {
        abandon();
    }

    QueueType VulkanCommandBuffer::queueType() const noexcept
    {
        return owner_ != nullptr ? owner_->queueType : QueueType::graphics;
    }

    std::uint64_t VulkanCommandBuffer::deviceToken() const noexcept
    {
        return deviceToken_;
    }

    detail::RecordingStateLock VulkanCommandBuffer::lockRecordingState() const
    {
        return detail::RecordingStateLock::lockPool(owner_);
    }

    VulkanCommandEntry* VulkanCommandBuffer::release() noexcept
    {
        if (entry_ != nullptr)
        {
            entry_->lease = nullptr;
        }
        owner_.reset();
        return std::exchange(entry_, nullptr);
    }

    VkCommandBuffer VulkanCommandBuffer::native() const noexcept
    {
        return entry_ != nullptr ? entry_->command : VK_NULL_HANDLE;
    }

    Status VulkanCommandBuffer::validateReset(bool requireIndividualOptIn) const
    {
        if (owner_ == nullptr || entry_ == nullptr || entry_->lease != this)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The command buffer does not hold a live command-pool recording lease.",
            });
        }
        if (requireIndividualOptIn && !hasAny(owner_->flags, CommandPoolFlag::allowIndividualReset))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The command pool did not opt into individual command-buffer reset.",
            });
        }
        if (entry_->inFlight)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A command buffer cannot be reset while it is in flight.",
            });
        }
        if (graphicsState().renderingOwner != nullptr || queryState().activeOcclusion.has_value() ||
            queryState().activePipelineStatistics.has_value() ||
            std::ranges::any_of(queryState().activeStreamOutputStatistics,
                                [](const auto& active) { return active.has_value(); }) ||
            queryState().activeGpuProfileScopeDepth != 0 || queryState().activeConditionalExecution.has_value() ||
            debugState().groupDepth != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Command recording reset requires every rendering, query, conditional, profile, and debug scope to be "
                "closed.",
            });
        }
        return {};
    }

    Status VulkanCommandBuffer::resetRecording()
    {
        auto reset = resetNative(*owner_, *entry_);
        if (!reset)
        {
            return reset;
        }
        clearRecordingState();
        auto began = beginRecording(*owner_, *entry_);
        if (!began)
        {
            return began;
        }
        return {};
    }

    void rollBackVulkanAccelerationStructurePublications(
        std::vector<VulkanAccelerationStructurePublication>& publications) noexcept
    {
        for (auto publication = publications.rbegin(); publication != publications.rend(); ++publication)
        {
            const auto slot = publication->slot.lock();
            const auto published = publication->published.lock();
            if (slot == nullptr || published == nullptr)
            {
                continue;
            }
            std::shared_ptr<VulkanAccelerationStructureVersion> discarded;
            {
                const std::scoped_lock lock(slot->mutex);
                if (slot->current == published)
                {
                    discarded = std::exchange(slot->current, std::move(publication->previous));
                }
            }
        }
        publications.clear();
    }

    std::uint64_t VulkanCommandBuffer::accelerationStructureRecordingId() noexcept
    {
        if (accelerationStructureRecordingId_ == 0)
        {
            accelerationStructureRecordingId_ = vulkanAccelerationStructureRecordings.next();
        }
        return accelerationStructureRecordingId_;
    }

    void VulkanCommandBuffer::clearRecordingState() noexcept
    {
        if (owner_ != nullptr && owner_->resources != nullptr)
        {
            owner_->resources->discardCommandLayouts(native());
        }
        resetState();
        rollBackVulkanAccelerationStructurePublications(accelerationStructurePublications_);
        commandUses_.reset();
        retained_.clear();
        queryPoolUses_.clear();
        conditionalRendering_.reset();
        accelerationStructureRecordingId_ = 0;
    }

    Result<VulkanCommandUses*> VulkanCommandBuffer::reserveCommandUses(std::size_t additional)
    {
        try
        {
#ifdef RACHIS_TESTING
            throwInjectedVulkanBadAlloc(VulkanBadAllocPoint::commandUses);
#endif
            auto uses = commandUses_.lock();
            if (uses == nullptr)
            {
                detail::reserveAdditional(retained_, 1);
                uses = std::make_shared<VulkanCommandUses>();
                retained_.push_back(uses);
                commandUses_ = uses;
            }
            detail::reserveAdditional(uses->entries, additional);
            return uses.get();
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Command-buffer uses could not be retained.",
            });
        }
    }

    Status VulkanCommandBuffer::reserveQueryPoolUse(const std::shared_ptr<VulkanSubmittedQueryPoolState>& poolState,
                                                    const std::shared_ptr<void>& keepAlive)
    {
        if (poolState == nullptr)
        {
            return {};
        }
        const auto existing = std::ranges::find_if(
            queryPoolUses_, [&](const VulkanCommandQueryPoolUse& use) noexcept { return use.poolState == poolState; });
        if (existing != queryPoolUses_.end())
        {
            if (existing->keepAlive == nullptr && keepAlive != nullptr)
            {
                existing->keepAlive = keepAlive;
            }
            return {};
        }
        try
        {
            VulkanCommandQueryPoolUse use{
                .poolState = poolState,
                .keepAlive = keepAlive,
                .slotFlags = {},
            };
            use.slotFlags.assign(poolState->queryCount, 0U);
            queryPoolUses_.push_back(std::move(use));
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Query command slot bookkeeping could not be allocated.",
            });
        }
        return {};
    }

    void VulkanCommandBuffer::recordQuerySlotReset(const std::shared_ptr<VulkanSubmittedQueryPoolState>& poolState,
                                                   std::uint32_t firstQuery,
                                                   std::uint32_t queryCount) noexcept
    {
        if (poolState == nullptr || queryCount == 0)
        {
            return;
        }
        const auto existing = std::ranges::find_if(
            queryPoolUses_, [&](const VulkanCommandQueryPoolUse& use) noexcept { return use.poolState == poolState; });
        if (existing == queryPoolUses_.end() || firstQuery >= existing->slotFlags.size())
        {
            return;
        }
        const auto availableCount = static_cast<std::uint32_t>(existing->slotFlags.size()) - firstQuery;
        const auto clampedCount = std::min(queryCount, availableCount);
        if (clampedCount == 0)
        {
            return;
        }
        const auto lastQuery = firstQuery + clampedCount - 1U;
        if (!existing->hasTouchedSlots)
        {
            existing->minTouchedSlot = firstQuery;
            existing->maxTouchedSlot = lastQuery;
            existing->hasTouchedSlots = true;
        }
        else
        {
            existing->minTouchedSlot = std::min(existing->minTouchedSlot, firstQuery);
            existing->maxTouchedSlot = std::max(existing->maxTouchedSlot, lastQuery);
        }
        auto* const flags = existing->slotFlags.data();
        for (std::uint32_t slot = firstQuery; slot <= lastQuery; ++slot)
        {
            flags[slot] =
                static_cast<std::uint8_t>((flags[slot] | kVulkanQuerySlotTouched) & ~kVulkanQuerySlotDirtyAtEnd);
        }
    }

    void VulkanCommandBuffer::recordQuerySlotWriteAfterLocalReset(
        const std::shared_ptr<VulkanSubmittedQueryPoolState>& poolState, std::uint32_t queryIndex) noexcept
    {
        recordQuerySlotReset(poolState, queryIndex, 1);
        const auto existing = std::ranges::find_if(
            queryPoolUses_, [&](const VulkanCommandQueryPoolUse& use) noexcept { return use.poolState == poolState; });
        if (existing != queryPoolUses_.end() && queryIndex < existing->slotFlags.size())
        {
            existing->slotFlags[queryIndex] =
                static_cast<std::uint8_t>(existing->slotFlags[queryIndex] | kVulkanQuerySlotDirtyAtEnd);
        }
    }

    void VulkanCommandBuffer::recordQuerySlotWriteWithoutLocalReset(
        const std::shared_ptr<VulkanSubmittedQueryPoolState>& poolState, std::uint32_t queryIndex) noexcept
    {
        if (poolState == nullptr)
        {
            return;
        }
        const auto existing = std::ranges::find_if(
            queryPoolUses_, [&](const VulkanCommandQueryPoolUse& use) noexcept { return use.poolState == poolState; });
        if (existing == queryPoolUses_.end() || queryIndex >= existing->slotFlags.size())
        {
            return;
        }
        if (!existing->hasTouchedSlots)
        {
            existing->minTouchedSlot = queryIndex;
            existing->maxTouchedSlot = queryIndex;
            existing->hasTouchedSlots = true;
        }
        else
        {
            existing->minTouchedSlot = std::min(existing->minTouchedSlot, queryIndex);
            existing->maxTouchedSlot = std::max(existing->maxTouchedSlot, queryIndex);
        }
        auto& flags = existing->slotFlags[queryIndex];
        if ((flags & kVulkanQuerySlotTouched) == 0)
        {
            flags = kVulkanQuerySlotTouched | kVulkanQuerySlotNeedsExternalReset | kVulkanQuerySlotDirtyAtEnd;
        }
        else
        {
            flags = static_cast<std::uint8_t>(flags | kVulkanQuerySlotDirtyAtEnd);
        }
    }

#ifdef RACHIS_TESTING
    void VulkanCommandBuffer::setInFlightForTesting(bool inFlight) noexcept
    {
        if (entry_ != nullptr)
        {
            entry_->inFlight = inFlight;
        }
    }
#endif

    void VulkanCommandBuffer::abandon() noexcept
    {
        if (owner_ == nullptr || entry_ == nullptr)
        {
            return;
        }

        rollBackVulkanAccelerationStructurePublications(accelerationStructurePublications_);
        commandUses_.reset();
        std::vector<std::shared_ptr<void>> retiredRetained;
        retiredRetained.swap(retained_);
        std::vector<VulkanCommandQueryPoolUse> retiredQueryUses;
        retiredQueryUses.swap(queryPoolUses_);
        auto owner = std::move(owner_);
        auto* entry = std::exchange(entry_, nullptr);
        {
            const std::scoped_lock lock(owner->mutex);
            if (owner->resources != nullptr && entry->command != VK_NULL_HANDLE)
            {
                owner->resources->discardCommandLayouts(entry->command);
            }
            Status reset{};
            if (entry->command != VK_NULL_HANDLE && owner->pool != VK_NULL_HANDLE)
            {
                closeAbandonedDebugGroups(*owner, *entry, debugState());
                reset = resetNative(*owner, *entry);
            }
            entry->lease = nullptr;
            entry->inFlight = false;
            if (reset && entry->command != VK_NULL_HANDLE && owner->pool != VK_NULL_HANDLE)
            {
                owner->available.push_back(entry);
            }
            else
            {
                owner->quarantined.push_back(entry);
            }
        }
    }

    VulkanCommandAllocator::~VulkanCommandAllocator()
    {
        shutdown();
    }

    void VulkanCommandAllocator::initialize(VkDevice device,
                                            const VolkDeviceTable* table,
                                            const VulkanQueueHandles& queues,
                                            std::uint64_t deviceToken,
                                            VulkanResourceAllocator* resources,
                                            const VulkanInstanceTable* instanceTable) noexcept
    {
        device_ = device;
        table_ = table;
        instanceTable_ = instanceTable;
        queues_ = queues;
        deviceToken_ = deviceToken;
        resources_ = resources;
    }

    void VulkanCommandAllocator::shutdown() noexcept
    {
        detail::HandleTable<std::shared_ptr<VulkanCommandPoolState>, CommandPoolHandle> pools{};
        {
            const std::scoped_lock lock(mutex_);
            std::swap(pools, pools_);
            device_ = VK_NULL_HANDLE;
            table_ = nullptr;
            instanceTable_ = nullptr;
        }
        pools.forEach([&](CommandPoolHandle, std::shared_ptr<VulkanCommandPoolState>& pool) {
            if (pool != nullptr)
            {
                pool->destroyNative();
            }
        });
    }

    std::uint32_t VulkanCommandAllocator::familyFor(QueueType queueType) const noexcept
    {
        switch (queueType)
        {
        case QueueType::compute:
            return queues_.families.compute;
        case QueueType::copy:
            return queues_.families.transfer;
        case QueueType::graphics:
            break;
        }
        return queues_.families.graphics;
    }

    Result<CommandPoolHandle> VulkanCommandAllocator::createCommandPool(const CommandPoolDesc& desc)
    {
        if (!isValidQueueType(desc.queueType) || desc.maxCommandBuffers == 0 ||
            desc.label.size() > maxDebugLabelBytes || !isValidCommandPoolFlags(desc.flags))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A command pool requires a valid queue, non-zero capacity, bounded label, and known flags.",
            });
        }
        if (device_ == VK_NULL_HANDLE || table_ == nullptr || table_->vkCreateCommandPool == nullptr ||
            table_->vkAllocateCommandBuffers == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Command-pool creation requires a live Vulkan device.",
            });
        }

        const auto family = familyFor(desc.queueType);
        if (family == VK_QUEUE_FAMILY_IGNORED)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The requested command-pool queue family is unavailable.",
            });
        }

        auto pool = std::make_shared<VulkanCommandPoolState>();
        pool->device = device_;
        pool->table = table_;
        if (instanceTable_ != nullptr)
        {
            pool->cmdBeginDebugUtilsLabel = instanceTable_->vkCmdBeginDebugUtilsLabelEXT;
            pool->cmdEndDebugUtilsLabel = instanceTable_->vkCmdEndDebugUtilsLabelEXT;
            pool->cmdInsertDebugUtilsLabel = instanceTable_->vkCmdInsertDebugUtilsLabelEXT;
        }
        pool->resources = resources_;
        pool->queueType = desc.queueType;
        pool->family = family;
        pool->flags = desc.flags;
        pool->label = std::string(desc.label);

        VkCommandPoolCreateFlags const flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        const VkCommandPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = {},
            .flags = flags,
            .queueFamilyIndex = family,
        };
        const auto created = table_->vkCreateCommandPool(device_, &info, nullptr, &pool->pool);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(created == VK_ERROR_OUT_OF_HOST_MEMORY || created == VK_ERROR_OUT_OF_DEVICE_MEMORY
                                    ? ErrorCode::outOfMemory
                                    : vulkanResultErrorCode(created),
                                created,
                                "vkCreateCommandPool"));
        }
        if (!pool->label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_COMMAND_POOL,
                                reinterpret_cast<std::uint64_t>(pool->pool),
                                pool->label.c_str());
        }

        pool->entries.reserve(desc.maxCommandBuffers);
        pool->available.reserve(desc.maxCommandBuffers);
        pool->quarantined.reserve(desc.maxCommandBuffers);
        std::vector<VkCommandBuffer> commands(desc.maxCommandBuffers, VK_NULL_HANDLE);
        const VkCommandBufferAllocateInfo allocate{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = {},
            .commandPool = pool->pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = desc.maxCommandBuffers,
        };
        const auto allocated = table_->vkAllocateCommandBuffers(device_, &allocate, commands.data());
        if (allocated != VK_SUCCESS)
        {
            table_->vkDestroyCommandPool(device_, pool->pool, nullptr);
            pool->pool = VK_NULL_HANDLE;
            return std::unexpected(
                makeVulkanError(allocated == VK_ERROR_OUT_OF_HOST_MEMORY || allocated == VK_ERROR_OUT_OF_DEVICE_MEMORY
                                    ? ErrorCode::outOfMemory
                                    : vulkanResultErrorCode(allocated),
                                allocated,
                                "vkAllocateCommandBuffers"));
        }
        for (std::uint32_t index = 0; index < desc.maxCommandBuffers; ++index)
        {
            auto entry = std::make_unique<VulkanCommandEntry>();
            entry->command = commands[index];
            if (!pool->label.empty())
            {
                setVulkanObjectName(instanceTable_,
                                    device_,
                                    VK_OBJECT_TYPE_COMMAND_BUFFER,
                                    reinterpret_cast<std::uint64_t>(entry->command),
                                    pool->label.c_str());
            }
            pool->available.push_back(entry.get());
            pool->entries.push_back(std::move(entry));
        }

        const std::scoped_lock lock(mutex_);
        auto handle = pools_.emplace(std::move(pool));
        if (!handle)
        {
            return std::unexpected(std::move(handle.error()));
        }
        return handle;
    }

    Status VulkanCommandAllocator::destroyCommandPool(CommandPoolHandle commandPool)
    {
        std::shared_ptr<VulkanCommandPoolState> removed;
        {
            const std::scoped_lock lock(mutex_);
            auto* found = pools_.find(commandPool);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The command-pool handle is invalid or stale.",
                });
            }
            removed = *found;
            static_cast<void>(pools_.erase(commandPool));
        }
        return {};
    }

    Status VulkanCommandAllocator::resetCommandPool(CommandPoolHandle commandPool)
    {
        std::shared_ptr<VulkanCommandPoolState> pool;
        {
            const std::scoped_lock lock(mutex_);
            const auto* found = pools_.find(commandPool);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The command-pool handle is invalid or stale.",
                });
            }
            pool = *found;
        }

        const std::scoped_lock lock(pool->mutex);
        std::size_t recordingCount = 0;
        for (const auto& entry : pool->entries)
        {
            if (entry != nullptr && entry->lease != nullptr)
            {
                ++recordingCount;
            }
            if (entry != nullptr && entry->inFlight)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A command pool cannot be reset while one of its entries is in flight.",
                });
            }
        }
        if (pool->available.size() + pool->quarantined.size() + recordingCount != pool->entries.size())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A command pool cannot be reset while one of its entries is in flight.",
            });
        }
        for (const auto& entry : pool->entries)
        {
            if (entry != nullptr && entry->lease != nullptr)
            {
                auto valid = entry->lease->validateReset(false);
                if (!valid)
                {
                    return valid;
                }
            }
        }
        if (table_ == nullptr || table_->vkResetCommandPool == nullptr || pool->pool == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Command-pool reset is unavailable because the Vulkan device table is incomplete.",
            });
        }
        const auto reset = table_->vkResetCommandPool(device_, pool->pool, 0);
        if (reset != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(reset), reset, "vkResetCommandPool"));
        }
        returnQuarantinedEntries(*pool);
        for (const auto& entry : pool->entries)
        {
            if (entry != nullptr && entry->lease != nullptr)
            {
                entry->lease->clearRecordingState();
            }
        }
        for (const auto& entry : pool->entries)
        {
            if (entry != nullptr && entry->lease != nullptr)
            {
                auto began = beginRecording(*pool, *entry);
                if (!began)
                {
                    return began;
                }
            }
        }
        return {};
    }

    Result<std::unique_ptr<detail::BackendCommandBuffer>> VulkanCommandAllocator::acquireCommandBuffer(
        CommandPoolHandle commandPool)
    {
        std::shared_ptr<VulkanCommandPoolState> pool;
        {
            const std::scoped_lock lock(mutex_);
            const auto* found = pools_.find(commandPool);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The command-pool handle is invalid or stale.",
                });
            }
            pool = *found;
        }

        const std::scoped_lock lock(pool->mutex);
        auto* entry = takeReusableEntry(*pool);
        if (entry == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The bounded command pool is exhausted.",
            });
        }
        auto reset = resetNative(*pool, *entry);
        if (!reset)
        {
            pool->quarantined.push_back(entry);
            return std::unexpected(std::move(reset.error()));
        }
        auto began = beginRecording(*pool, *entry);
        if (!began)
        {
            pool->quarantined.push_back(entry);
            return std::unexpected(std::move(began.error()));
        }

        auto* allocation = new (std::nothrow) VulkanCommandBuffer(pool, *entry, deviceToken_);
        if (allocation == nullptr)
        {
            static_cast<void>(resetNative(*pool, *entry));
            pool->available.push_back(entry);
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate a command-buffer lease.",
            });
        }
        entry->lease = allocation;
        return std::unique_ptr<detail::BackendCommandBuffer>(allocation);
    }

    Status VulkanCommandAllocator::resetCommandBuffer(detail::BackendCommandBuffer& commandBuffer)
    {
        auto* native = asVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The command buffer belongs to another device or backend.",
            });
        }
        const auto owner = native->ownerShared();
        const std::scoped_lock lock(owner->mutex);
        auto valid = native->validateReset(true);
        if (!valid)
        {
            return valid;
        }
        return native->resetRecording();
    }

    Status VulkanCommandAllocator::beginDebugGroup(detail::BackendCommandBuffer& commandBuffer,
                                                   const DebugMarkerDesc& desc)
    {
        auto* native = asVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The command buffer belongs to another device or backend.",
            });
        }
        auto valid = validateMarkerDesc(desc);
        if (!valid)
        {
            return valid;
        }
        const auto owner = native->ownerShared();
        if (owner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Debug groups require a live Vulkan command pool.",
            });
        }
        const std::scoped_lock lock(owner->mutex);
        auto& state = native->debugState();
        const auto currentThread = std::this_thread::get_id();
        if (state.groupDepth == maxDebugGroupDepth)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The command buffer reached the portable debug-group nesting limit.",
            });
        }
        if (state.groupDepth != 0 && state.groupThread != currentThread)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Nested debug and GPU profile groups must be recorded on the thread that opened the outer group.",
            });
        }
        if (owner->table == nullptr || owner->pool == VK_NULL_HANDLE || native->native() == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Debug groups require a live Vulkan command pool.",
            });
        }
        emitLabel(*owner, native->native(), desc, true);
        if (state.groupDepth == 0)
        {
            state.groupThread = currentThread;
        }
        state.groupOwners[state.groupDepth] = detail::DebugGroupOwner::user;
        ++state.groupDepth;
        return {};
    }

    Status VulkanCommandAllocator::endDebugGroup(detail::BackendCommandBuffer& commandBuffer)
    {
        auto* native = asVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The command buffer belongs to another device or backend.",
            });
        }
        const auto owner = native->ownerShared();
        if (owner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Debug groups require a live Vulkan command pool.",
            });
        }
        const std::scoped_lock lock(owner->mutex);
        auto& state = native->debugState();
        if (state.groupDepth == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot end a debug or GPU profile group when no group is open.",
            });
        }
        if (state.groupOwners[state.groupDepth - 1] != detail::DebugGroupOwner::user)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A user debug-group end cannot close a GPU profile scope.",
            });
        }
        if (state.groupThread != std::this_thread::get_id())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A debug or GPU profile group must end on the thread that opened it.",
            });
        }
        if (owner->pool != VK_NULL_HANDLE && native->native() != VK_NULL_HANDLE &&
            owner->cmdEndDebugUtilsLabel != nullptr)
        {
            owner->cmdEndDebugUtilsLabel(native->native());
        }
        --state.groupDepth;
        return {};
    }

    Status VulkanCommandAllocator::insertDebugMarker(detail::BackendCommandBuffer& commandBuffer,
                                                     const DebugMarkerDesc& desc)
    {
        auto* native = asVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The command buffer belongs to another device or backend.",
            });
        }
        auto valid = validateMarkerDesc(desc);
        if (!valid)
        {
            return valid;
        }
        const auto owner = native->ownerShared();
        if (owner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Debug markers require a live Vulkan command pool.",
            });
        }
        const std::scoped_lock lock(owner->mutex);
        if (owner->table == nullptr || owner->pool == VK_NULL_HANDLE || native->native() == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Debug markers require a live Vulkan command pool.",
            });
        }
        emitLabel(*owner, native->native(), desc, false);
        return {};
    }

#ifdef RACHIS_TESTING
    Status VulkanCommandAllocator::markCommandBufferInFlight(detail::BackendCommandBuffer& commandBuffer, bool inFlight)
    {
        auto* native = asVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The command buffer belongs to another device or backend.",
            });
        }
        const auto owner = native->ownerShared();
        const std::scoped_lock lock(owner->mutex);
        native->setInFlightForTesting(inFlight);
        return {};
    }

    namespace
    {

        VulkanCommandAllocator* boundCommands = nullptr;

    }

    void bindVulkanCommandAllocator(VulkanCommandAllocator* allocator) noexcept
    {
        boundCommands = allocator;
    }

    void unbindVulkanCommandAllocator(VulkanCommandAllocator* allocator) noexcept
    {
        if (boundCommands == allocator)
        {
            boundCommands = nullptr;
        }
    }

    Status markVulkanCommandBufferInFlight(CommandBuffer& commandBuffer, bool inFlight)
    {
        if (boundCommands == nullptr || detail::CommandBufferAccess::get(commandBuffer) == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No live Vulkan command allocator is bound for in-flight marking.",
            });
        }
        return boundCommands->markCommandBufferInFlight(*detail::CommandBufferAccess::get(commandBuffer), inFlight);
    }
#endif

} // namespace rachis::vulkan
