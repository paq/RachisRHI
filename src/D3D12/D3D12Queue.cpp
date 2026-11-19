#include "D3D12/D3D12Queue.h"

#include "Common/HandleTable.h"
#include "Common/VectorReserve.h"
#include "D3D12/D3D12DebugMarker.h"
#include "D3D12/D3D12LegacyTextureStates.h"
#include "D3D12/D3D12NativeHelpers.h"
#include "D3D12/D3D12RuntimePolicies.h"
#include "D3D12/D3D12TestHooks.h"
#include "Internal/CommandBufferAccess.h"
#include "Internal/CommandPoolRetirementTestHook.h"
#include "Internal/TimelinePointAccess.h"

#include <wrl/client.h>

#include <Windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rachis::d3d12
{
    namespace
    {

        using Microsoft::WRL::ComPtr;

        class D3D12CommandBuffer;

        static_assert(residencyQueueCount == kD3D12ResidencyQueueCount);
        constexpr std::size_t queueCount = kD3D12ResidencyQueueCount;

        [[nodiscard]] constexpr std::size_t queueIndex(QueueType type) noexcept
        {
            return d3d12ResidencyQueueIndex(type);
        }

        [[nodiscard]] constexpr bool isValidQueueType(QueueType type) noexcept
        {
            return queueIndex(type) < queueCount;
        }

        [[nodiscard]] constexpr bool isValidCommandPoolFlags(CommandPoolFlag flags) noexcept
        {
            constexpr auto allowed = static_cast<std::uint32_t>(CommandPoolFlag::allowIndividualReset) |
                                     static_cast<std::uint32_t>(CommandPoolFlag::unretainedResources);
            return (static_cast<std::uint32_t>(flags) & ~allowed) == 0;
        }

        [[nodiscard]] constexpr D3D12_COMMAND_LIST_TYPE nativeQueueType(QueueType type) noexcept
        {
            switch (type)
            {
            case QueueType::compute:
                return D3D12_COMMAND_LIST_TYPE_COMPUTE;
            case QueueType::copy:
                return D3D12_COMMAND_LIST_TYPE_COPY;
            default:
                return D3D12_COMMAND_LIST_TYPE_DIRECT;
            }
        }

        [[nodiscard]] std::uint64_t nextRecordingId() noexcept
        {
            static AccelerationStructureRecordingSequence sequence;
            return sequence.next();
        }

        struct PoolEntry final
        {
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList> commandList;
            ComPtr<ID3D12GraphicsCommandList1> commandList1;
            ComPtr<ID3D12GraphicsCommandList4> commandList4;
            ComPtr<ID3D12GraphicsCommandList5> commandList5;
            ComPtr<ID3D12GraphicsCommandList6> commandList6;
            ComPtr<ID3D12GraphicsCommandList7> commandList7;
            ComPtr<ID3D12GraphicsCommandList8> commandList8;
            ComPtr<ID3D12GraphicsCommandList9> commandList9;
            std::vector<std::shared_ptr<void>> retainedResources;
            std::vector<std::shared_ptr<SwapchainImageCommandReference>> swapchainImageReferences;
            std::vector<ResidencyLease> residencyLeases;
            std::vector<InPlaceResidencyReference> inPlaceResidencies;
            std::vector<TimelinePoint> sparseMappingDependencies;
            std::vector<AccelerationStructureExecutionLease> accelerationStructureLeases;
            std::vector<AccelerationStructurePublication> accelerationStructurePublications;
            std::weak_ptr<void> descriptorArena;
            std::uint64_t descriptorArenaRecordingId = 0;
            LegacyTextureStateRecording legacyTextureStates;
            ComPtr<ID3D12CommandAllocator> legacyStateAllocator;
            ComPtr<ID3D12GraphicsCommandList> legacyStateCommandList;
            std::uint64_t recordingId = 0;
            D3D12CommandBuffer* lease = nullptr;
            bool allocatorClean = false;
        };

        void clearPoolEntryRetentions(PoolEntry& entry) noexcept
        {
            entry.retainedResources.clear();
            entry.swapchainImageReferences.clear();
            entry.residencyLeases.clear();
            entry.inPlaceResidencies.clear();
            entry.sparseMappingDependencies.clear();
            entry.accelerationStructureLeases.clear();
            entry.accelerationStructurePublications.clear();
            entry.descriptorArena.reset();
            entry.descriptorArenaRecordingId = 0;
            entry.legacyTextureStates.clear();
            entry.recordingId = 0;
        }

        void discardPoolEntryRecording(PoolEntry& entry) noexcept
        {
            rollBackAccelerationStructurePublications(entry.accelerationStructurePublications);
            clearPoolEntryRetentions(entry);
        }

        struct PoolEntryRetentions final
        {
            std::vector<std::shared_ptr<void>> retainedResources;
            std::vector<std::shared_ptr<SwapchainImageCommandReference>> swapchainImageReferences;
            std::vector<ResidencyLease> residencyLeases;
            std::vector<InPlaceResidencyReference> inPlaceResidencies;
            std::vector<TimelinePoint> sparseMappingDependencies;
            std::vector<AccelerationStructureExecutionLease> accelerationStructureLeases;
        };

        [[nodiscard]] PoolEntryRetentions takePoolEntryRetentions(PoolEntry& entry) noexcept
        {
            PoolEntryRetentions retentions{
                .retainedResources = std::move(entry.retainedResources),
                .swapchainImageReferences = std::move(entry.swapchainImageReferences),
                .residencyLeases = std::move(entry.residencyLeases),
                .inPlaceResidencies = std::move(entry.inPlaceResidencies),
                .sparseMappingDependencies = std::move(entry.sparseMappingDependencies),
                .accelerationStructureLeases = std::move(entry.accelerationStructureLeases),
            };
            entry.accelerationStructurePublications.clear();
            entry.descriptorArena.reset();
            entry.descriptorArenaRecordingId = 0;
            entry.legacyTextureStates.clear();
            entry.recordingId = 0;
            return retentions;
        }

        struct CommandPoolState;

        struct PendingEntry final
        {
            std::uint64_t fenceValue = 0;
            std::shared_ptr<CommandPoolState> pool;
            PoolEntry* entry = nullptr;
        };

        struct PendingRetention final
        {
            std::uint64_t fenceValue = 0;
            std::shared_ptr<void> owner;
        };

        struct SubmissionBatch final
        {
            std::vector<PoolEntry*> entries;
            std::vector<std::shared_ptr<CommandPoolState>> pools;
            std::vector<ID3D12CommandList*> lists;
            LegacyTextureStateSubmission legacyTextureStates;
        };

        struct EventCloser final
        {
            void operator()(void* event) const noexcept
            {
                if (event != nullptr)
                {
                    static_cast<void>(CloseHandle(event));
                }
            }
        };

        using UniqueEvent = std::unique_ptr<void, EventCloser>;
        using SharedEvent = std::shared_ptr<void>;
        using FenceEventPool = ReusableWaitEventPool<UniqueEvent>;

        struct MakeResidentRetention final
        {
            explicit MakeResidentRetention(std::span<const std::shared_ptr<void>> values)
                : owners(values.begin(), values.end())
            {
            }

            MakeResidentRetention(const MakeResidentRetention&) = delete;
            MakeResidentRetention& operator=(const MakeResidentRetention&) = delete;
            MakeResidentRetention(MakeResidentRetention&&) = delete;
            MakeResidentRetention& operator=(MakeResidentRetention&&) = delete;

            ~MakeResidentRetention()
            {
                if (!quarantined)
                {
                    return;
                }
                const auto completed = residencyFence != nullptr ? residencyFence->GetCompletedValue() : 0;
                if (completed != UINT64_MAX && completed < residencyValue)
                {
                    test::prematureMakeResidentQuarantineReleases().fetch_add(1, std::memory_order_relaxed);
                }
                test::activeMakeResidentQuarantines().fetch_sub(1, std::memory_order_relaxed);
            }

            void quarantine(ID3D12Fence& fence, std::uint64_t value) noexcept
            {
                if (quarantined)
                {
                    return;
                }
                residencyFence = &fence;
                residencyValue = value;
                quarantined = true;
                test::activeMakeResidentQuarantines().fetch_add(1, std::memory_order_relaxed);
            }

            std::vector<std::shared_ptr<void>> owners;
            ComPtr<ID3D12Fence> residencyFence;
            std::uint64_t residencyValue = 0;
            bool quarantined = false;
        };

        struct QueueState final
        {
            QueueState() = default;

            QueueType type = QueueType::graphics;
            std::shared_ptr<FenceEventPool> eventPool;
            mutable std::mutex pendingWaitMutex;
            mutable std::map<std::uint64_t, SharedEvent> pendingWaitEvents;
            ComPtr<ID3D12CommandQueue> queue;
            ComPtr<ID3D12Fence> fence;
            std::uint64_t nextFenceValue = 1;
            std::uint64_t reservedSignalCount = 0;
            std::atomic_uint64_t lastSubmittedValue = 0;
            mutable std::atomic_uint64_t lastCompletedValue = 0;
            bool hasUnpublishedWork = false;
            std::mutex submissionMutex;
            std::mutex retirementMutex;
            std::deque<PendingEntry> pending;
            std::deque<PendingRetention> pendingRetentions;
            std::vector<std::shared_ptr<void>> quarantinedRetentions;

            ~QueueState() = default;
            QueueState(const QueueState&) = delete;
            QueueState& operator=(const QueueState&) = delete;
            QueueState(QueueState&&) = delete;
            QueueState& operator=(QueueState&&) = delete;
        };

        [[nodiscard]] std::uint64_t recordCompletedValue(const QueueState& state, std::uint64_t completed) noexcept
        {
            auto recorded = state.lastCompletedValue.load(std::memory_order_acquire);
            while (recorded < completed &&
                   !state.lastCompletedValue.compare_exchange_weak(
                       recorded, completed, std::memory_order_acq_rel, std::memory_order_acquire))
            {
            }
            return (std::max)(recorded, completed);
        }

        void collectPendingWaitEvents(const QueueState& state, std::uint64_t completed) noexcept
        {
            const std::scoped_lock lock(state.pendingWaitMutex);
            state.pendingWaitEvents.erase(state.pendingWaitEvents.begin(),
                                          state.pendingWaitEvents.upper_bound(completed));
        }

        [[nodiscard]] bool hasUnreservedTimelineSlot(const QueueState& state) noexcept
        {
            constexpr auto sentinel = std::numeric_limits<std::uint64_t>::max();
            if (state.nextFenceValue >= sentinel)
            {
                return false;
            }
            const auto remaining = sentinel - state.nextFenceValue;
            return state.reservedSignalCount < remaining;
        }

        struct CommandPoolState final
        {
            std::shared_ptr<QueueState> queue;
            CommandPoolFlag flags = CommandPoolFlag::none;
            std::mutex mutex;
            std::vector<std::unique_ptr<PoolEntry>> entries;
            BoundedFifoQueue<PoolEntry*> available;
            std::vector<PoolEntry*> quarantined;
        };

        class D3D12CommandBuffer final : public detail::BackendCommandBuffer
        {
        public:
            D3D12CommandBuffer(std::shared_ptr<CommandPoolState> owner,
                               PoolEntry& entry,
                               std::uint64_t deviceToken) noexcept
                : detail::BackendCommandBuffer(detail::BackendCommandBufferKind::d3d12), owner_(std::move(owner)),
                  entry_(&entry), deviceToken_(deviceToken),
                  queueType_(owner_ != nullptr && owner_->queue != nullptr ? owner_->queue->type : QueueType::graphics)
            {
            }

            ~D3D12CommandBuffer() override
            {
                abandon();
            }

            D3D12CommandBuffer(const D3D12CommandBuffer&) = delete;
            D3D12CommandBuffer& operator=(const D3D12CommandBuffer&) = delete;
            D3D12CommandBuffer(D3D12CommandBuffer&&) = delete;
            D3D12CommandBuffer& operator=(D3D12CommandBuffer&&) = delete;

            [[nodiscard]] QueueType queueType() const noexcept override
            {
                return queueType_;
            }

            [[nodiscard]] std::uint64_t deviceToken() const noexcept override
            {
                return deviceToken_;
            }

            [[nodiscard]] detail::RecordingStateLock lockRecordingState() const override
            {
                return detail::RecordingStateLock::lockPool(owner_);
            }

            [[nodiscard]] CommandPoolState* owner() const noexcept
            {
                return owner_.get();
            }

            [[nodiscard]] const std::shared_ptr<CommandPoolState>& ownerShared() const noexcept
            {
                return owner_;
            }

            [[nodiscard]] QueueState* queue() const noexcept
            {
                return owner_ != nullptr ? owner_->queue.get() : nullptr;
            }

            [[nodiscard]] PoolEntry* release() noexcept
            {
                auto* entry = entry_;
                if (entry != nullptr)
                {
                    entry->recordingId = 0;
                    entry->lease = nullptr;
                }
                owner_.reset();
                entry_ = nullptr;
                return entry;
            }

            [[nodiscard]] PoolEntry* entry() const noexcept
            {
                return entry_;
            }

            [[nodiscard]] Status validateReset(bool requireIndividualOptIn) const
            {
                if (owner_ == nullptr || entry_ == nullptr || entry_->lease != this)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
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
                if (graphicsState().renderingOwner != nullptr || queryState().activeOcclusion.has_value() ||
                    queryState().activePipelineStatistics.has_value() ||
                    std::ranges::any_of(queryState().activeStreamOutputStatistics,
                                        [](const auto& active) { return active.has_value(); }) ||
                    queryState().activeGpuProfileScopeDepth != 0 ||
                    queryState().activeConditionalExecution.has_value() || debugState().groupDepth != 0)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "Command recording reset requires every rendering, query, conditional, profile, and debug "
                        "scope to be closed.",
                    });
                }
                return {};
            }

            [[nodiscard]] Status resetRecording()
            {
                auto* const entry = entry_;
                return executeD3D12RecordingReset(
                    [&]() noexcept -> std::int32_t { return static_cast<std::int32_t>(entry->commandList->Close()); },
                    [&]() noexcept -> std::int32_t { return static_cast<std::int32_t>(entry->allocator->Reset()); },
                    [&]() noexcept -> std::int32_t {
                        return static_cast<std::int32_t>(entry->commandList->Reset(entry->allocator.Get(), nullptr));
                    },
                    [&]() noexcept { discardPoolEntryRecording(*entry); },
                    [&](bool clean) noexcept { entry->allocatorClean = clean; },
                    [&]() noexcept { entry->recordingId = nextRecordingId(); },
                    [&]() noexcept {
                        entry->lease = nullptr;
                        owner_->quarantined.push_back(entry);
                        entry_ = nullptr;
                        owner_.reset();
                    },
                    [&]() noexcept { resetState(); });
            }

            void detachForShutdown() noexcept
            {
                if (entry_ != nullptr)
                {
                    if (entry_->commandList != nullptr)
                    {
                        closeAbandonedDebugGroups(*entry_->commandList.Get(), debugState());
                        static_cast<void>(entry_->commandList->Close());
                    }
                    entry_->lease = nullptr;
                    entry_ = nullptr;
                }
                resetState();
                owner_.reset();
            }

        private:
            void abandon() noexcept
            {
                if (owner_ == nullptr || entry_ == nullptr)
                {
                    return;
                }

                auto owner = std::move(owner_);
                auto* entry = std::exchange(entry_, nullptr);
                closeAbandonedDebugGroups(*entry->commandList.Get(), debugState());

                const HRESULT closeResult = entry->commandList->Close();
                discardPoolEntryRecording(*entry);
                const std::scoped_lock lock(owner->mutex);
                entry->lease = nullptr;
                entry->allocatorClean = false;
                if (SUCCEEDED(closeResult))
                {
                    owner->available.push_back(entry);
                }
                else
                {
                    owner->quarantined.push_back(entry);
                }
            }

            std::shared_ptr<CommandPoolState> owner_;
            PoolEntry* entry_ = nullptr;
            std::uint64_t deviceToken_ = 0;
            QueueType queueType_ = QueueType::graphics;
        };

        [[nodiscard]] D3D12CommandBuffer* asD3D12Command(detail::BackendCommandBuffer* commandBuffer,
                                                         std::uint64_t expectedDeviceToken) noexcept
        {
            if (commandBuffer == nullptr || commandBuffer->kind() != detail::BackendCommandBufferKind::d3d12 ||
                expectedDeviceToken == 0 || commandBuffer->deviceToken() != expectedDeviceToken)
            {
                return nullptr;
            }
            return static_cast<D3D12CommandBuffer*>(commandBuffer);
        }

        [[nodiscard]] Result<D3D12CommandBuffer*> requireLiveD3D12Command(
            detail::BackendCommandBuffer* commandBuffer,
            std::uint64_t expectedDeviceToken,
            std::string_view foreignMessage = "The command buffer belongs to another device or backend.")
        {
            auto* native = asD3D12Command(commandBuffer, expectedDeviceToken);
            if (native == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    std::string(foreignMessage),
                });
            }
            if (native->owner() == nullptr || native->queue() == nullptr || native->entry() == nullptr ||
                native->entry()->lease != native)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The command buffer does not hold a live command-pool recording lease.",
                });
            }
            return native;
        }

        [[nodiscard]] Status initializeQueue(ID3D12Device& device, QueueState& state, QueueType type)
        {
            state.type = type;

            const D3D12_COMMAND_QUEUE_DESC queueDesc{
                .Type = nativeQueueType(type),
                .Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
                .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
                .NodeMask = 0,
            };
            HRESULT result = device.CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&state.queue));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(ErrorCode::deviceCreationFailed, result, "ID3D12Device::CreateCommandQueue"));
            }

            result = device.CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&state.fence));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(ErrorCode::deviceCreationFailed, result, "ID3D12Device::CreateFence"));
            }

            constexpr std::array<const wchar_t*, queueCount> queueNames{
                L"RHI graphics queue",
                L"RHI compute queue",
                L"RHI copy queue",
            };
            constexpr std::array<const wchar_t*, queueCount> fenceNames{
                L"RHI graphics timeline",
                L"RHI compute timeline",
                L"RHI copy timeline",
            };
            const auto typeIndex = static_cast<std::size_t>(type);
            if (typeIndex < queueNames.size())
            {
                static_cast<void>(state.queue->SetName(queueNames[typeIndex]));
                static_cast<void>(state.fence->SetName(fenceNames[typeIndex]));
            }

            return {};
        }

        [[nodiscard]] Result<std::shared_ptr<CommandPoolState>> createCommandPoolState(
            ID3D12Device& device,
            std::shared_ptr<QueueState> queue,
            std::uint32_t capacity,
            std::string_view label,
            CommandPoolFlag flags)
        {
            std::shared_ptr<CommandPoolState> pool;
            std::wstring nativeLabel;
            try
            {
                pool = std::make_shared<CommandPoolState>();
                pool->queue = std::move(queue);
                pool->flags = flags;
                nativeLabel = wideLabel(label);
                pool->entries.reserve(capacity);
                pool->available.reserve(capacity);
                pool->quarantined.reserve(capacity);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate command-pool bookkeeping.",
                });
            }

            for (std::uint32_t index = 0; index < capacity; ++index)
            {
                auto entry = std::unique_ptr<PoolEntry>(new (std::nothrow) PoolEntry());
                if (entry == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::outOfMemory,
                        0,
                        "Failed to allocate a command-pool entry.",
                    });
                }
                HRESULT result =
                    device.CreateCommandAllocator(nativeQueueType(pool->queue->type), IID_PPV_ARGS(&entry->allocator));
                if (FAILED(result))
                {
                    return std::unexpected(
                        makeClassifiedD3D12HresultError(static_cast<std::int32_t>(result),
                                                        "ID3D12Device::CreateCommandAllocator(command pool)",
                                                        ErrorCode::outOfMemory));
                }
                result = device.CreateCommandList(0,
                                                  nativeQueueType(pool->queue->type),
                                                  entry->allocator.Get(),
                                                  nullptr,
                                                  IID_PPV_ARGS(&entry->commandList));
                if (FAILED(result))
                {
                    return std::unexpected(
                        makeClassifiedD3D12HresultError(static_cast<std::int32_t>(result),
                                                        "ID3D12Device::CreateCommandList(command pool)",
                                                        ErrorCode::outOfMemory));
                }
                static_cast<void>(entry->commandList.As(&entry->commandList1));
                static_cast<void>(entry->commandList.As(&entry->commandList4));
                static_cast<void>(entry->commandList.As(&entry->commandList5));
                static_cast<void>(entry->commandList.As(&entry->commandList6));
                static_cast<void>(entry->commandList.As(&entry->commandList7));
                static_cast<void>(entry->commandList.As(&entry->commandList8));
                static_cast<void>(entry->commandList.As(&entry->commandList9));
                result = entry->commandList->Close();
                if (FAILED(result))
                {
                    return std::unexpected(
                        makeClassifiedD3D12HresultError(static_cast<std::int32_t>(result),
                                                        "ID3D12GraphicsCommandList::Close(command pool)",
                                                        ErrorCode::nativeFailure));
                }
                if (!nativeLabel.empty())
                {
                    static_cast<void>(entry->allocator->SetName(nativeLabel.c_str()));
                    static_cast<void>(entry->commandList->SetName(nativeLabel.c_str()));
                }
                entry->allocatorClean = false;
                auto* entryPointer = entry.get();
                pool->entries.push_back(std::move(entry));
                pool->available.push_back(entryPointer);
            }
            return pool;
        }

        void collectQueue(QueueState& state) noexcept
        {
            const auto completed = state.fence->GetCompletedValue();
            if (completed == UINT64_MAX)
            {
                return;
            }
            static_cast<void>(recordCompletedValue(state, completed));
            collectPendingWaitEvents(state, completed);

            while (true)
            {
                PendingEntry retired;
                PoolEntryRetentions retentions;
                {
                    const std::scoped_lock lock(state.retirementMutex);
                    if (state.pending.empty() || state.pending.front().fenceValue > completed)
                    {
                        break;
                    }
                    retired = std::move(state.pending.front());
                    state.pending.pop_front();
                    retentions = takePoolEntryRetentions(*retired.entry);
                    const std::scoped_lock poolLock(retired.pool->mutex);
                    retired.entry->allocatorClean = false;
                    retired.pool->available.push_back(retired.entry);
                }
#ifdef RACHIS_TESTING
                detail::test::observeCommandPoolRetirement();
#endif
            }
            {
                const std::scoped_lock lock(state.retirementMutex);
                while (!state.pendingRetentions.empty() && state.pendingRetentions.front().fenceValue <= completed)
                {
                    state.pendingRetentions.pop_front();
                }
            }
        }

        [[nodiscard]] Result<std::unique_ptr<detail::BackendCommandBuffer>> acquireFromPool(
            const std::shared_ptr<CommandPoolState>& pool, std::uint64_t deviceToken)
        {
            if (pool == nullptr || pool->queue == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The command pool is invalid.",
                });
            }

            collectQueue(*pool->queue);
            const std::scoped_lock lock(pool->mutex);
            auto reopened = acquireAndReopenD3D12PoolEntry(
                pool->available,
                pool->quarantined,
                [](PoolEntry& candidate) noexcept { clearPoolEntryRetentions(candidate); },
                [](const PoolEntry& candidate) noexcept { return candidate.allocatorClean; },
                [](PoolEntry& candidate, bool clean) noexcept { candidate.allocatorClean = clean; },
                [](PoolEntry& candidate) noexcept -> std::int32_t {
                    return static_cast<std::int32_t>(candidate.allocator->Reset());
                },
                [](PoolEntry& candidate) noexcept -> std::int32_t {
                    return static_cast<std::int32_t>(candidate.commandList->Reset(candidate.allocator.Get(), nullptr));
                });
            if (!reopened)
            {
                return std::unexpected(std::move(reopened.error()));
            }
            auto* entry = *reopened;

            auto* allocation = new (std::nothrow) D3D12CommandBuffer(pool, *entry, deviceToken);
            if (allocation == nullptr)
            {
                const HRESULT closeResult = entry->commandList->Close();
                if (SUCCEEDED(closeResult))
                {
                    pool->available.push_back(entry);
                }
                else
                {
                    pool->quarantined.push_back(entry);
                }
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate a command-buffer lease.",
                });
            }
            entry->recordingId = nextRecordingId();
            entry->lease = allocation;
            return std::unique_ptr<detail::BackendCommandBuffer>(allocation);
        }

        [[nodiscard]] Result<TimelinePointStatus> queryValue(const QueueState& state, std::uint64_t value)
        {
            if (value == 0 || value > state.lastSubmittedValue.load(std::memory_order_acquire))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The timeline point was not submitted by this queue.",
                });
            }

            const auto completed = state.fence->GetCompletedValue();
            if (completed == UINT64_MAX)
            {
                return std::unexpected(Error{
                    ErrorCode::deviceLost,
                    0,
                    "The D3D12 fence reported device removal.",
                });
            }
            collectPendingWaitEvents(state, completed);
            if (completed >= value)
            {
                return TimelinePointStatus::complete;
            }

            return TimelinePointStatus::pending;
        }

        [[nodiscard]] UniqueEvent tryAcquireOrCreateCompletionEvent(FenceEventPool* eventPool) noexcept
        {
            if (eventPool != nullptr)
            {
                while (auto cached = eventPool->tryAcquire())
                {
                    if (ResetEvent(cached.get()) != FALSE)
                    {
                        return cached;
                    }
                }
            }
            return UniqueEvent{CreateEventW(nullptr, FALSE, FALSE, nullptr)};
        }

        [[nodiscard]] Result<SharedEvent> acquireValueCompletionEvent(QueueState& state, std::uint64_t value)
        {
            try
            {
                const std::scoped_lock lock(state.pendingWaitMutex);
                const auto position = state.pendingWaitEvents.lower_bound(value);
                if (position != state.pendingWaitEvents.end() && position->first == value)
                {
                    return position->second;
                }

                HANDLE created = CreateEventW(nullptr, TRUE, FALSE, nullptr);
                if (created == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::nativeFailure,
                        static_cast<std::int64_t>(GetLastError()),
                        "CreateEventW failed while creating a timeline-wait event.",
                    });
                }
                SharedEvent event(created, EventCloser{});
                const auto inserted = state.pendingWaitEvents.emplace_hint(position, value, event);
                const HRESULT result = state.fence->SetEventOnCompletion(value, event.get());
                if (FAILED(result))
                {
                    state.pendingWaitEvents.erase(inserted);
                    return std::unexpected(
                        nativeError(ErrorCode::nativeFailure, result, "ID3D12Fence::SetEventOnCompletion"));
                }
                return event;
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate a timeline-wait event entry.",
                });
            }
        }

        void recycleCompletionEvent(FenceEventPool* eventPool, UniqueEvent event) noexcept
        {
            if (eventPool != nullptr && event != nullptr)
            {
                eventPool->recycle(std::move(event));
            }
        }

        [[nodiscard]] Status waitForValue(QueueState& state, std::uint64_t value)
        {
            auto status = queryValue(state, value);
            if (!status)
            {
                return std::unexpected(std::move(status.error()));
            }
            if (*status == TimelinePointStatus::complete)
            {
                return {};
            }

            auto event = acquireValueCompletionEvent(state, value);
            if (!event)
            {
                return std::unexpected(std::move(event.error()));
            }
            const DWORD waitResult = WaitForSingleObject(event->get(), INFINITE);
            if (waitResult != WAIT_OBJECT_0)
            {
                const DWORD lastError = waitResult == WAIT_FAILED ? GetLastError() : 0;
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    static_cast<std::int64_t>(lastError),
                    "WaitForSingleObject failed while waiting for a queue timeline point.",
                });
            }
            const auto reached = state.fence->GetCompletedValue();
            if (reached == UINT64_MAX)
            {
                return std::unexpected(Error{
                    ErrorCode::deviceLost,
                    0,
                    "The D3D12 fence reported device removal while waiting for queue completion.",
                });
            }
            if (reached < value)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "The queue completion event was signaled before its requested timeline value completed.",
                });
            }
            return {};
        }

        [[nodiscard]] DWORD boundedWaitMilliseconds(std::chrono::steady_clock::time_point deadline) noexcept
        {
            const auto remaining = deadline - std::chrono::steady_clock::now();
            if (remaining < std::chrono::milliseconds{1})
            {
                return 0;
            }
            auto milliseconds = std::chrono::ceil<std::chrono::milliseconds>(remaining);
            constexpr auto maxFinite = static_cast<std::chrono::milliseconds::rep>(INFINITE - 1U);
            if (milliseconds.count() > maxFinite)
            {
                return INFINITE - 1U;
            }
            return static_cast<DWORD>(milliseconds.count());
        }

        [[nodiscard]] Result<TimelinePointStatus> waitForValue(QueueState& state,
                                                               std::uint64_t value,
                                                               std::chrono::nanoseconds timeout)
        {
            if (timeout < std::chrono::nanoseconds::zero())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A bounded timeline wait requires a non-negative timeout.",
                });
            }

            auto status = queryValue(state, value);
            if (!status || *status == TimelinePointStatus::complete || timeout == std::chrono::nanoseconds::zero())
            {
                return status;
            }

            const auto now = std::chrono::steady_clock::now();
            const auto clockTimeout = std::chrono::ceil<std::chrono::steady_clock::duration>(timeout);
            const auto remainingClockRange = std::chrono::steady_clock::time_point::max() - now;
            const auto deadline =
                clockTimeout >= remainingClockRange ? std::chrono::steady_clock::time_point::max() : now + clockTimeout;

            auto event = acquireValueCompletionEvent(state, value);
            if (!event)
            {
                return std::unexpected(std::move(event.error()));
            }

            for (;;)
            {
                const DWORD milliseconds = boundedWaitMilliseconds(deadline);
                const DWORD waitResult = WaitForSingleObject(event->get(), milliseconds);
                if (waitResult == WAIT_TIMEOUT)
                {
                    if (std::chrono::steady_clock::now() >= deadline)
                    {
                        return TimelinePointStatus::pending;
                    }
                    if (milliseconds == 0)
                    {
                        static_cast<void>(SwitchToThread());
                    }
                    continue;
                }
                if (waitResult != WAIT_OBJECT_0)
                {
                    const DWORD lastError = waitResult == WAIT_FAILED ? GetLastError() : 0;
                    return std::unexpected(Error{
                        ErrorCode::nativeFailure,
                        static_cast<std::int64_t>(lastError),
                        "WaitForSingleObject failed during a bounded queue timeline wait.",
                    });
                }

                status = queryValue(state, value);
                if (!status || *status == TimelinePointStatus::complete)
                {
                    return status;
                }
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "The queue completion event was signaled before its requested timeline value completed.",
                });
            }
        }

        [[nodiscard]] Status publishUntrackedWork(QueueState& state)
        {
            if (!state.hasUnpublishedWork)
            {
                return {};
            }
            if (!hasUnreservedTimelineSlot(state))
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "The queue timeline is exhausted and cannot publish its accepted native work.",
                });
            }

            const auto value = state.nextFenceValue++;
            const HRESULT result = state.queue->Signal(state.fence.Get(), value);
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(ErrorCode::deviceLost, result, "ID3D12CommandQueue::Signal(idle recovery)"));
            }
            state.hasUnpublishedWork = false;
            state.lastSubmittedValue.store(value, std::memory_order_release);
            return {};
        }

        [[nodiscard]] Status waitForSubmittedWork(QueueState& state)
        {
            auto published = publishUntrackedWork(state);
            if (!published)
            {
                return published;
            }
            const auto lastSubmitted = state.lastSubmittedValue.load(std::memory_order_acquire);
            if (lastSubmitted != 0)
            {
                return waitForValue(state, lastSubmitted);
            }
            if (state.fence->GetCompletedValue() == UINT64_MAX)
            {
                return std::unexpected(Error{
                    ErrorCode::deviceLost,
                    0,
                    "The D3D12 fence reported device removal while checking an idle queue.",
                });
            }
            return {};
        }

        void waitForFenceValueNoexcept(ID3D12Fence& fence, std::uint64_t value, FenceEventPool* eventPool) noexcept
        {
            const auto completed = fence.GetCompletedValue();
            if (completed == UINT64_MAX || completed >= value)
            {
                return;
            }

            auto event = tryAcquireOrCreateCompletionEvent(eventPool);
            if (event != nullptr)
            {
                if (SUCCEEDED(fence.SetEventOnCompletion(value, event.get())))
                {
                    if (WaitForSingleObject(event.get(), INFINITE) == WAIT_OBJECT_0)
                    {
                        const auto reached = fence.GetCompletedValue();
                        if (reached == UINT64_MAX || reached >= value)
                        {
                            recycleCompletionEvent(eventPool, std::move(event));
                            return;
                        }
                    }
                }
                else
                {
                    recycleCompletionEvent(eventPool, std::move(event));
                }
            }

            while (true)
            {
                const auto polled = fence.GetCompletedValue();
                if (polled == UINT64_MAX || polled >= value)
                {
                    return;
                }
                static_cast<void>(SwitchToThread());
            }
        }

        void waitForValueNoexcept(QueueState& state, std::uint64_t value) noexcept
        {
            waitForFenceValueNoexcept(*state.fence.Get(), value, state.eventPool.get());
        }

        void waitForSubmittedWorkNoexcept(QueueState& state) noexcept
        {
            if (state.hasUnpublishedWork && hasUnreservedTimelineSlot(state))
            {
                const auto value = state.nextFenceValue++;
                if (SUCCEEDED(state.queue->Signal(state.fence.Get(), value)))
                {
                    state.hasUnpublishedWork = false;
                    state.lastSubmittedValue.store(value, std::memory_order_release);
                }
            }

            const auto lastSubmitted = state.lastSubmittedValue.load(std::memory_order_acquire);
            if (lastSubmitted != 0)
            {
                waitForValueNoexcept(state, lastSubmitted);
            }
        }

    } // namespace

    std::optional<test::CommandDependencySnapshot> test::commandDependencySnapshot(
        const CommandBuffer& commandBuffer) noexcept
    {
        const auto* backend = detail::CommandBufferAccess::get(commandBuffer);
        const auto* native = backend != nullptr && backend->kind() == detail::BackendCommandBufferKind::d3d12
                                 ? static_cast<const D3D12CommandBuffer*>(backend)
                                 : nullptr;
        if (native == nullptr || native->entry() == nullptr)
        {
            return std::nullopt;
        }
        return CommandDependencySnapshot{
            .resourceOwners = native->entry()->retainedResources.size(),
            .residencyLeases = native->entry()->residencyLeases.size(),
            .swapchainReferences = native->entry()->swapchainImageReferences.size(),
            .sparseMappingDependencies = native->entry()->sparseMappingDependencies.size(),
        };
    }

    struct QueueSignalReservation::Impl final
    {
        Impl(QueueState& stateValue, std::uint64_t deviceTokenValue) : state(&stateValue), deviceToken(deviceTokenValue)
        {
        }

        ~Impl()
        {
            if (!active || state == nullptr)
            {
                return;
            }
            const std::scoped_lock lock(state->submissionMutex);
            if (state->reservedSignalCount == 0)
            {
                std::terminate();
            }
            --state->reservedSignalCount;
        }

        Impl(const Impl&) = delete;
        Impl& operator=(const Impl&) = delete;
        Impl(Impl&&) = delete;
        Impl& operator=(Impl&&) = delete;

        QueueState* state = nullptr;
        std::uint64_t deviceToken = 0;
        bool active = false;
    };

    QueueSignalReservation::QueueSignalReservation() noexcept = default;

    QueueSignalReservation::~QueueSignalReservation() = default;

    QueueSignalReservation::QueueSignalReservation(QueueSignalReservation&&) noexcept = default;

    QueueSignalReservation& QueueSignalReservation::operator=(QueueSignalReservation&&) noexcept = default;

    QueueSignalReservation::QueueSignalReservation(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl))
    {
    }

    Result<TimelinePoint> QueueSignalReservation::signal()
    {
        if (impl_ == nullptr || impl_->state == nullptr)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, "The queue signal reservation is not active."});
        }
        auto reservation = std::move(impl_);
        auto& state = *reservation->state;
        const std::scoped_lock lock(state.submissionMutex);
        if (!reservation->active || state.reservedSignalCount == 0)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, "The queue signal reservation was released."});
        }
        reservation->active = false;
        --state.reservedSignalCount;
        const auto value = state.nextFenceValue;
        const HRESULT result = state.queue->Signal(state.fence.Get(), value);
        if (FAILED(result))
        {
            return std::unexpected(nativeError(ErrorCode::deviceLost, result, "ID3D12CommandQueue::Signal"));
        }
        ++state.nextFenceValue;
        state.hasUnpublishedWork = false;
        state.lastSubmittedValue.store(value, std::memory_order_release);
        return detail::TimelinePointAccess::make(state.type, value, reservation->deviceToken);
    }

    struct QueueManager::Impl final
    {
        ID3D12Device* device = nullptr;
        std::shared_ptr<FenceEventPool> eventPool = std::make_shared<FenceEventPool>();
        ComPtr<ID3D12Device3> residencyDevice;
        ComPtr<ID3D12Fence> residencyFence;
        std::mutex residencyMutex;
        std::uint64_t nextResidencyFenceValue = 1;
        std::uint64_t lastAcceptedResidencyFenceValue = 0;
        std::uint64_t deviceToken = 0;
        std::array<std::shared_ptr<QueueState>, queueCount> queues;
        std::mutex commandPoolMutex;
        detail::HandleTable<std::shared_ptr<CommandPoolState>, CommandPoolHandle> commandPools;
        std::vector<std::weak_ptr<CommandPoolState>> allCommandPools;
        std::size_t nextCommandPoolPruneThreshold = 64;

        [[nodiscard]] QueueState& queue(QueueType type) noexcept
        {
            return *queues[queueIndex(type)];
        }

        [[nodiscard]] const QueueState& queue(QueueType type) const noexcept
        {
            return *queues[queueIndex(type)];
        }

        [[nodiscard]] std::uint64_t queryCompletedFenceValue(std::size_t index) const noexcept
        {
            return queues[index]->fence->GetCompletedValue();
        }

        [[nodiscard]] bool promoteCompleted(ResidencyRecord& residency) const noexcept
        {
            return promoteCompletedResidency(residency.state, residency.readyAfter, [&](std::size_t index) noexcept {
                return queryCompletedFenceValue(index);
            });
        }

        [[nodiscard]] Status validateResidencyDependencies(const PoolEntry& entry,
                                                           QueueType submissionQueue,
                                                           std::span<const TimelinePoint> waits) const
        {
            return validateResidencyLeases(entry.residencyLeases, submissionQueue, waits);
        }

        [[nodiscard]] Status validateResidencyLeases(std::span<const ResidencyLease> leases,
                                                     QueueType submissionQueue,
                                                     std::span<const TimelinePoint> waits) const
        {
            for (const auto& lease : leases)
            {
                auto* residency = lease.record();
                if (residency == nullptr)
                {
                    continue;
                }
                const std::scoped_lock lock(residency->mutex);
                auto valid = validateResidencySubmission(
                    residency->state, residency->readyAfter, submissionQueue, waits, [&](std::size_t index) noexcept {
                        return queryCompletedFenceValue(index);
                    });
                if (!valid)
                {
                    return valid;
                }
            }
            return {};
        }

        [[nodiscard]] static Status acquireInPlaceResidencies(const PoolEntry& entry,
                                                              std::vector<ResidencyLease>& staged)
        {
            if (entry.inPlaceResidencies.empty())
            {
                return {};
            }
            try
            {
                std::vector<std::shared_ptr<ResidencyRecord>> residencies;
                for (const auto& reference : entry.inPlaceResidencies)
                {
                    reference.log->visitSince(reference.sequence,
                                              [&](const std::shared_ptr<ResidencyRecord>& residency) {
                                                  if (residency != nullptr)
                                                  {
                                                      residencies.push_back(residency);
                                                  }
                                              });
                }
                return appendResidencyLeases(staged, residencies);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to retain the residency of update-after-bind binding elements.",
                });
            }
        }

        [[nodiscard]] bool isSubmittedPointOfThisDevice(TimelinePoint point) const noexcept
        {
            return point.isValid() && detail::TimelinePointAccess::deviceToken(point) == deviceToken &&
                   point.value() <= queue(point.queueType()).lastSubmittedValue.load(std::memory_order_acquire);
        }

        [[nodiscard]] Status validateSparseMappingDependencies(const PoolEntry& entry,
                                                               QueueType submissionQueue,
                                                               std::span<const TimelinePoint> waits) const
        {
            for (const auto dependency : entry.sparseMappingDependencies)
            {
                if (!isSubmittedPointOfThisDevice(dependency))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "A recorded sparse mapping dependency is invalid or no longer submitted.",
                    });
                }
                if (dependency.queueType() == submissionQueue)
                {
                    continue;
                }
                const bool ordered = std::ranges::any_of(waits, [&](const auto wait) {
                    return wait.queueType() == dependency.queueType() && wait.value() >= dependency.value();
                });
                if (!ordered)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "A cross-queue submission that uses a sparse buffer must wait on its latest mapping timeline "
                        "point.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Status validateSubmissionDependencies(const PoolEntry& entry,
                                                            QueueType submissionQueue,
                                                            std::span<const TimelinePoint> waits) const
        {
            for (const auto& reference : entry.swapchainImageReferences)
            {
                if (reference == nullptr)
                {
                    continue;
                }
                auto validSwapchain = reference->validateSubmission(submissionQueue);
                if (!validSwapchain)
                {
                    return validSwapchain;
                }
            }
            auto residencyDependencies = validateResidencyDependencies(entry, submissionQueue, waits);
            if (!residencyDependencies)
            {
                return residencyDependencies;
            }
            return validateSparseMappingDependencies(entry, submissionQueue, waits);
        }

        void propagateResidencyDependencies(const PoolEntry& entry,
                                            QueueType submissionQueue,
                                            std::uint64_t submittedValue) const noexcept
        {
            for (const auto& lease : entry.residencyLeases)
            {
                auto* residency = lease.record();
                if (residency == nullptr)
                {
                    continue;
                }
                const std::scoped_lock lock(residency->mutex);
                propagateResidencySubmission(
                    residency->state,
                    residency->readyAfter,
                    submissionQueue,
                    submittedValue,
                    [&](std::size_t index) noexcept { return queryCompletedFenceValue(index); });
            }
        }

        [[nodiscard]] Result<TimelinePoint> executeSubmissionBatch(QueueState& state,
                                                                   const SubmissionBatch& batch,
                                                                   std::span<const TimelinePoint> waits);

        [[nodiscard]] Result<std::uint64_t> acceptPendingMakeResident(QueueState& state,
                                                                      D3D12_RESIDENCY_FLAGS flags,
                                                                      std::span<ID3D12Pageable* const> pageables);

        [[nodiscard]] ResidencySubmission publishAcceptedMakeResident(QueueState& state,
                                                                      MakeResidentRetention& retention,
                                                                      std::uint64_t residencyValue);

        [[nodiscard]] HRESULT enqueueSparseMappingDependencyWaits(QueueState& mappingState,
                                                                  std::span<const TimelinePoint> dependencies);

        [[nodiscard]] SparseMappingSubmission relaySparseMappingPublication(QueueState& mappingState,
                                                                            QueueState& publicationState,
                                                                            TimelinePoint mappingPublication);

        [[nodiscard]] Result<SparseMappingSubmission> enqueueSparseMappingsLocked(
            QueueState& mappingState,
            QueueState& publicationState,
            std::span<const SparseTileUpdate> mappings,
            std::span<const TimelinePoint> dependencies,
            const std::shared_ptr<void>& owner);
    };

    QueueManager::QueueManager() : impl_(std::make_unique<Impl>())
    {
    }

    QueueManager::~QueueManager()
    {
        if (impl_ == nullptr)
        {
            return;
        }

        for (auto& queue : impl_->queues)
        {
            if (queue != nullptr)
            {
                const std::scoped_lock submissionLock(queue->submissionMutex);
                waitForSubmittedWorkNoexcept(*queue);
                collectQueue(*queue);

                std::deque<PendingEntry> uncollectedPending;
                std::deque<PendingRetention> uncollectedRetentions;
                {
                    const std::scoped_lock retirementLock(queue->retirementMutex);
                    uncollectedPending = std::move(queue->pending);
                    uncollectedRetentions = std::move(queue->pendingRetentions);
                }
                for (auto& pending : uncollectedPending)
                {
                    if (pending.entry != nullptr)
                    {
                        pending.entry->retainedResources.clear();
                        pending.entry->swapchainImageReferences.clear();
                        pending.entry->residencyLeases.clear();
                        pending.entry->inPlaceResidencies.clear();
                        pending.entry->sparseMappingDependencies.clear();
                        pending.entry->accelerationStructureLeases.clear();
                        pending.entry->accelerationStructurePublications.clear();
                    }
                }
            }
        }

        std::uint64_t residencyValue = 0;
        {
            const std::scoped_lock lock(impl_->residencyMutex);
            residencyValue = impl_->lastAcceptedResidencyFenceValue;
        }
        if (impl_->residencyFence != nullptr && residencyValue != 0)
        {
            waitForFenceValueNoexcept(*impl_->residencyFence.Get(), residencyValue, impl_->eventPool.get());
        }

        for (auto& queue : impl_->queues)
        {
            if (queue != nullptr)
            {
                const std::scoped_lock retirementLock(queue->retirementMutex);
                queue->quarantinedRetentions.clear();
            }
        }

        std::vector<std::weak_ptr<CommandPoolState>> pools;
        {
            const std::scoped_lock lock(impl_->commandPoolMutex);
            pools = std::move(impl_->allCommandPools);
            impl_->commandPools = {};
        }
        for (const auto& weakPool : pools)
        {
            auto pool = weakPool.lock();
            if (pool == nullptr)
            {
                continue;
            }
            std::vector<std::unique_ptr<PoolEntry>> poolEntries;
            std::shared_ptr<QueueState> poolQueue;
            {
                const std::scoped_lock poolLock(pool->mutex);
                for (const auto& entry : pool->entries)
                {
                    if (entry != nullptr && entry->lease != nullptr)
                    {
                        entry->lease->detachForShutdown();
                    }
                }
                pool->available.clear();
                pool->quarantined.clear();
                poolEntries = std::move(pool->entries);
                poolQueue = std::move(pool->queue);
            }
            for (const auto& entry : poolEntries)
            {
                if (entry != nullptr)
                {
                    entry->retainedResources.clear();
                    entry->swapchainImageReferences.clear();
                    entry->residencyLeases.clear();
                    entry->inPlaceResidencies.clear();
                    entry->sparseMappingDependencies.clear();
                    entry->accelerationStructureLeases.clear();
                    entry->accelerationStructurePublications.clear();
                }
            }
            poolEntries.clear();
            poolQueue.reset();
        }
    }

    Status QueueManager::initialize(ID3D12Device& device, std::uint64_t deviceToken)
    {
        if (deviceToken == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The queue manager requires a non-zero device token.",
            });
        }

        impl_->device = &device;
        impl_->deviceToken = deviceToken;
        if (SUCCEEDED(device.QueryInterface(IID_PPV_ARGS(&impl_->residencyDevice))))
        {
            const HRESULT result = device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&impl_->residencyFence));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(ErrorCode::deviceCreationFailed, result, "ID3D12Device::CreateFence(residency)"));
            }
        }
        constexpr std::array types{QueueType::graphics, QueueType::compute, QueueType::copy};
        for (const auto type : types)
        {
            auto state = std::make_shared<QueueState>();
            state->eventPool = impl_->eventPool;
            auto initialized = initializeQueue(device, *state, type);
            if (!initialized)
            {
                return std::unexpected(std::move(initialized.error()));
            }
            impl_->queues[queueIndex(type)] = std::move(state);
        }
        return {};
    }

    Result<CommandPoolHandle> QueueManager::createCommandPool(const CommandPoolDesc& desc)
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

        auto created = createCommandPoolState(
            *impl_->device, impl_->queues[queueIndex(desc.queueType)], desc.maxCommandBuffers, desc.label, desc.flags);
        if (!created)
        {
            return std::unexpected(std::move(created.error()));
        }
        try
        {
            const std::scoped_lock lock(impl_->commandPoolMutex);
            if (impl_->allCommandPools.size() >= impl_->nextCommandPoolPruneThreshold)
            {
                static_cast<void>(
                    std::erase_if(impl_->allCommandPools, [](const auto& weakPool) { return weakPool.expired(); }));
                impl_->nextCommandPoolPruneThreshold = std::max<std::size_t>(64, impl_->allCommandPools.size() * 2);
            }
            if (impl_->allCommandPools.size() == impl_->allCommandPools.capacity())
            {
                const auto currentCapacity = impl_->allCommandPools.capacity();
                const auto grownCapacity = currentCapacity == 0
                                               ? std::size_t{16}
                                               : (currentCapacity > std::numeric_limits<std::size_t>::max() / 2
                                                      ? impl_->allCommandPools.size() + 1
                                                      : currentCapacity * 2);
                impl_->allCommandPools.reserve(grownCapacity);
            }
            auto poolState = std::move(*created);
            auto handle = impl_->commandPools.emplace(poolState);
            if (handle)
            {
                impl_->allCommandPools.emplace_back(std::move(poolState));
            }
            return handle;
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to publish the command-pool handle.",
            });
        }
    }

    Status QueueManager::destroyCommandPool(CommandPoolHandle commandPool)
    {
        const std::scoped_lock lock(impl_->commandPoolMutex);
        if (!impl_->commandPools.erase(commandPool))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The command-pool handle is invalid or stale.",
            });
        }
        return {};
    }

    Status QueueManager::resetCommandPool(CommandPoolHandle commandPool)
    {
        std::shared_ptr<CommandPoolState> pool;
        {
            const std::scoped_lock lock(impl_->commandPoolMutex);
            const auto* found = impl_->commandPools.find(commandPool);
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

        collectQueue(*pool->queue);
        const std::scoped_lock lock(pool->mutex);
        std::size_t recordingCount = 0;
        for (const auto& entry : pool->entries)
        {
            if (entry->lease != nullptr)
            {
                ++recordingCount;
            }
        }
        if (!isD3D12CommandPoolIdleFromInFlight(
                pool->entries.size(), pool->available.size(), pool->quarantined.size(), recordingCount))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A command pool cannot be reset while one of its entries is in flight.",
            });
        }

        for (const auto& entry : pool->entries)
        {
            if (entry->lease != nullptr)
            {
                auto valid = entry->lease->validateReset(false);
                if (!valid)
                {
                    return valid;
                }
            }
        }
        for (const auto& entry : pool->entries)
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
            pool->available,
            pool->quarantined,
            [](const PoolEntry& idleEntry) noexcept { return idleEntry.allocatorClean; },
            [](PoolEntry& idleEntry, bool clean) noexcept { idleEntry.allocatorClean = clean; },
            [](PoolEntry& idleEntry) noexcept -> std::int32_t {
                return static_cast<std::int32_t>(idleEntry.allocator->Reset());
            });
    }

    Result<std::unique_ptr<detail::BackendCommandBuffer>> QueueManager::acquire(CommandPoolHandle commandPool)
    {
        std::shared_ptr<CommandPoolState> pool;
        {
            const std::scoped_lock lock(impl_->commandPoolMutex);
            const auto* found = impl_->commandPools.find(commandPool);
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
        return acquireFromPool(pool, impl_->deviceToken);
    }

    Status QueueManager::reset(detail::BackendCommandBuffer& commandBuffer)
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;

        const auto owner = native->ownerShared();
        const std::scoped_lock lock(owner->mutex);
        auto valid = native->validateReset(true);
        if (!valid)
        {
            return valid;
        }
        return native->resetRecording();
    }

    namespace
    {

        [[nodiscard]] Status validateNoOpenRecordingScopes(const D3D12CommandBuffer& native)
        {
            if (native.graphicsState().renderingOwner != nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A D3D12 command buffer cannot be submitted while a rendering scope is active.",
                });
            }
            if (native.queryState().activeOcclusion.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A D3D12 command buffer cannot be submitted while an occlusion query is active.",
                });
            }
            if (native.queryState().activePipelineStatistics.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A D3D12 command buffer cannot be submitted while a pipeline-statistics query is active.",
                });
            }
            if (std::ranges::any_of(native.queryState().activeStreamOutputStatistics,
                                    [](const auto& active) { return active.has_value(); }))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A D3D12 command buffer cannot be submitted while a stream-output-statistics query is active.",
                });
            }
            if (native.queryState().activeGpuProfileScopeDepth != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A D3D12 command buffer cannot be submitted while a GPU profile scope is active.",
                });
            }
            if (native.queryState().activeConditionalExecution.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A D3D12 command buffer cannot be submitted while conditional execution is active.",
                });
            }
            if (native.debugState().groupDepth != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A D3D12 command buffer cannot be submitted while a debug group is open.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<std::vector<D3D12CommandBuffer*>> resolveSubmissionCommands(
            std::span<const std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers, std::uint64_t deviceToken)
        {
            if (commandBuffers.empty() || commandBuffers.size() > std::numeric_limits<UINT>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A D3D12 submission batch must contain a representable non-empty command-list sequence.",
                });
            }

            std::vector<D3D12CommandBuffer*> nativeCommands;
            try
            {
                nativeCommands.reserve(commandBuffers.size());
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate D3D12 submission command buffer batch.",
                });
            }
            const QueueState* queueState = nullptr;
            for (const auto& commandBuffer : commandBuffers)
            {
                auto live = requireLiveD3D12Command(
                    commandBuffer.get(),
                    deviceToken,
                    "A command buffer in the submission batch belongs to another device or backend.");
                if (!live)
                {
                    return std::unexpected(std::move(live.error()));
                }
                auto* native = *live;
                if (queueState == nullptr)
                {
                    queueState = native->queue();
                }
                else if (queueState != native->queue())
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Every D3D12 command buffer in a submission batch must target the same queue.",
                    });
                }
                if (std::ranges::any_of(nativeCommands,
                                        [&](const auto* other) { return other->entry() == native->entry(); }))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A D3D12 submission batch cannot contain the same command-buffer lease more than once.",
                    });
                }
                auto scopesClosed = validateNoOpenRecordingScopes(*native);
                if (!scopesClosed)
                {
                    return std::unexpected(std::move(scopesClosed.error()));
                }
                nativeCommands.push_back(native);
            }
            return nativeCommands;
        }

        [[nodiscard]] Result<SubmissionBatch> collectSubmissionBatch(
            std::span<D3D12CommandBuffer* const> nativeCommands)
        {
            SubmissionBatch batch;
            try
            {
                batch.entries.reserve(nativeCommands.size());
                batch.pools.reserve(nativeCommands.size());
                batch.lists.reserve(nativeCommands.size());
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate D3D12 submission batch storage.",
                });
            }
            for (const auto* native : nativeCommands)
            {
                batch.entries.push_back(native->entry());
                batch.pools.push_back(native->ownerShared());
                batch.lists.push_back(native->entry()->commandList.Get());
            }
            return batch;
        }

        void reservePoolQuarantineCapacity(std::span<const std::shared_ptr<CommandPoolState>> pools)
        {
            for (std::size_t index = 0; index < pools.size(); ++index)
            {
                const auto isSamePool = [&](const std::shared_ptr<CommandPoolState>& other) {
                    return other.get() == pools[index].get();
                };
                if (std::ranges::any_of(pools.first(index), isSamePool))
                {
                    continue;
                }
                const auto countForPool =
                    static_cast<std::size_t>(std::ranges::count_if(pools.subspan(index), isSamePool));
                const std::scoped_lock poolLock(pools[index]->mutex);
                pools[index]->quarantined.reserve(
                    std::max(pools[index]->entries.size(), pools[index]->quarantined.size() + countForPool));
            }
        }

        [[nodiscard]] Status reserveBatchRetirementRecords(QueueState& state, const SubmissionBatch& batch)
        {
            std::size_t reservedPendingCount = 0;
            try
            {
                for (std::size_t index = 0; index < batch.entries.size(); ++index)
                {
                    state.pending.push_back(PendingEntry{
                        .fenceValue = std::numeric_limits<std::uint64_t>::max(),
                        .pool = batch.pools[index],
                        .entry = batch.entries[index],
                    });
                    ++reservedPendingCount;
                }
                reservePoolQuarantineCapacity(batch.pools);
            }
            catch (const std::bad_alloc&)
            {
                while (reservedPendingCount != 0)
                {
                    state.pending.pop_back();
                    --reservedPendingCount;
                }
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to reserve D3D12 batch retirement records.",
                });
            }
            return {};
        }

        void discardBatchRetirementRecords(QueueState& state, std::size_t batchSize) noexcept
        {
            for (std::size_t index = 0; index < batchSize; ++index)
            {
                state.pending.pop_back();
            }
        }

        [[nodiscard]] HRESULT closeCommandLists(std::span<PoolEntry* const> entries) noexcept
        {
            HRESULT closeFailure = S_OK;
            for (auto* entry : entries)
            {
                const HRESULT closeResult = entry->commandList->Close();
                if (FAILED(closeResult) && SUCCEEDED(closeFailure))
                {
                    closeFailure = closeResult;
                }
            }
            return closeFailure;
        }

        enum class BatchRetentions : std::uint8_t
        {
            release,
            keep,
        };

        void quarantineBatchEntries(QueueState& state,
                                    const SubmissionBatch& batch,
                                    BatchRetentions retentions) noexcept
        {
            discardBatchRetirementRecords(state, batch.entries.size());
            for (std::size_t index = 0; index < batch.entries.size(); ++index)
            {
                if (retentions == BatchRetentions::release)
                {
                    rollBackAccelerationStructurePublications(batch.entries[index]->accelerationStructurePublications);
                    batch.entries[index]->retainedResources.clear();
                    batch.entries[index]->swapchainImageReferences.clear();
                    batch.entries[index]->residencyLeases.clear();
                    batch.entries[index]->inPlaceResidencies.clear();
                    batch.entries[index]->sparseMappingDependencies.clear();
                    batch.entries[index]->accelerationStructureLeases.clear();
                    batch.entries[index]->legacyTextureStates.clear();
                }
                const std::scoped_lock poolLock(batch.pools[index]->mutex);
                batch.pools[index]->quarantined.push_back(batch.entries[index]);
            }
        }

        void commitAccelerationStructurePublications(std::span<PoolEntry* const> entries) noexcept
        {
            for (auto* entry : entries)
            {
                entry->accelerationStructurePublications.clear();
            }
        }

        void markSwapchainImagesSubmitted(std::span<PoolEntry* const> entries)
        {
            for (auto* entry : entries)
            {
                for (const auto& reference : entry->swapchainImageReferences)
                {
                    static_cast<void>(reference->markSubmitted());
                }
            }
        }

        void publishSwapchainSubmissionPoint(std::span<PoolEntry* const> entries, TimelinePoint submittedPoint)
        {
            for (auto* entry : entries)
            {
                for (const auto& reference : entry->swapchainImageReferences)
                {
                    static_cast<void>(reference->publishSubmissionPoint(submittedPoint));
                }
            }
        }

        void releaseLegacyTextureStatePrologue(PoolEntry& entry) noexcept
        {
            entry.legacyStateCommandList.Reset();
            entry.legacyStateAllocator.Reset();
        }

        [[nodiscard]] Result<ID3D12CommandList*> recordLegacyTextureStatePrologue(
            ID3D12Device& device,
            PoolEntry& entry,
            QueueType queueType,
            std::span<const D3D12_RESOURCE_BARRIER> barriers)
        {
            if (barriers.size() > std::numeric_limits<UINT>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "A legacy texture-state prologue expands beyond the native barrier-count limit.",
                });
            }
            HRESULT result = S_OK;
            if (entry.legacyStateCommandList == nullptr)
            {
                ComPtr<ID3D12CommandAllocator> allocator;
                result = device.CreateCommandAllocator(nativeQueueType(queueType), IID_PPV_ARGS(&allocator));
                if (FAILED(result))
                {
                    return std::unexpected(
                        makeClassifiedD3D12HresultError(static_cast<std::int32_t>(result),
                                                        "ID3D12Device::CreateCommandAllocator(legacy texture states)",
                                                        ErrorCode::outOfMemory));
                }
                ComPtr<ID3D12GraphicsCommandList> commandList;
                result = device.CreateCommandList(
                    0, nativeQueueType(queueType), allocator.Get(), nullptr, IID_PPV_ARGS(&commandList));
                if (FAILED(result))
                {
                    return std::unexpected(
                        makeClassifiedD3D12HresultError(static_cast<std::int32_t>(result),
                                                        "ID3D12Device::CreateCommandList(legacy texture states)",
                                                        ErrorCode::outOfMemory));
                }
                static_cast<void>(commandList->SetName(L"RHI legacy texture state prologue"));
                entry.legacyStateAllocator = std::move(allocator);
                entry.legacyStateCommandList = std::move(commandList);
            }
            else
            {
                result = entry.legacyStateAllocator->Reset();
                if (SUCCEEDED(result))
                {
                    result = entry.legacyStateCommandList->Reset(entry.legacyStateAllocator.Get(), nullptr);
                }
                if (FAILED(result))
                {
                    releaseLegacyTextureStatePrologue(entry);
                    return std::unexpected(nativeError(
                        ErrorCode::nativeFailure, result, "ID3D12GraphicsCommandList::Reset(legacy texture states)"));
                }
            }
            entry.legacyStateCommandList->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
            result = entry.legacyStateCommandList->Close();
            if (FAILED(result))
            {
                releaseLegacyTextureStatePrologue(entry);
                return std::unexpected(nativeError(
                    ErrorCode::nativeFailure, result, "ID3D12GraphicsCommandList::Close(legacy texture states)"));
            }
            return entry.legacyStateCommandList.Get();
        }

        [[nodiscard]] Status checkSubmissionTextureLayouts(const SubmissionBatch& batch)
        {
            try
            {
                std::vector<const LegacyTextureStateRecording*> recordings;
                recordings.reserve(batch.entries.size());
                for (const auto* entry : batch.entries)
                {
                    recordings.push_back(&entry->legacyTextureStates);
                }
                return checkTextureLayoutEntries(recordings);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to check the texture layouts of a D3D12 submission.",
                });
            }
        }

        [[nodiscard]] Status recordLegacyTextureStatePrologues(ID3D12Device& device,
                                                               QueueType queueType,
                                                               SubmissionBatch& batch)
        {
            if (std::ranges::all_of(batch.entries,
                                    [](const PoolEntry* entry) { return entry->legacyTextureStates.empty(); }))
            {
                return {};
            }
            try
            {
                const auto queueStates = legacyQueueResourceStates(queueType);
                std::vector<ID3D12CommandList*> lists;
                lists.reserve(batch.entries.size() * 2);
                std::vector<D3D12_RESOURCE_BARRIER> barriers;
                for (auto* entry : batch.entries)
                {
                    barriers.clear();
                    batch.legacyTextureStates.resolve(entry->legacyTextureStates, queueStates, barriers);
                    if (!barriers.empty())
                    {
                        auto prologue = recordLegacyTextureStatePrologue(device, *entry, queueType, barriers);
                        if (!prologue)
                        {
                            return std::unexpected(std::move(prologue.error()));
                        }
                        lists.push_back(*prologue);
                    }
                    lists.push_back(entry->commandList.Get());
                }
                batch.lists = std::move(lists);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to resolve legacy texture states for a D3D12 submission.",
                });
            }
            return {};
        }

        void assignBatchRetirementFenceValue(QueueState& state,
                                             std::size_t batchSize,
                                             std::uint64_t fenceValue) noexcept
        {
            for (std::size_t index = 0; index < batchSize; ++index)
            {
                state.pending[state.pending.size() - batchSize + index].fenceValue = fenceValue;
            }
        }

    } // namespace

    Result<TimelinePoint> QueueManager::Impl::executeSubmissionBatch(QueueState& state,
                                                                     const SubmissionBatch& batch,
                                                                     std::span<const TimelinePoint> waits)
    {
        const HRESULT closeFailure = closeCommandLists(batch.entries);
        if (FAILED(closeFailure))
        {
            quarantineBatchEntries(state, batch, BatchRetentions::release);
            return std::unexpected(
                nativeError(ErrorCode::validationError, closeFailure, "ID3D12GraphicsCommandList::Close(batch)"));
        }

        for (const auto point : waits)
        {
            auto& dependency = queue(point.queueType());
            const HRESULT waitResult = state.queue->Wait(dependency.fence.Get(), point.value());
            if (FAILED(waitResult))
            {
                quarantineBatchEntries(state, batch, BatchRetentions::release);
                return std::unexpected(nativeError(ErrorCode::nativeFailure, waitResult, "ID3D12CommandQueue::Wait"));
            }
            state.hasUnpublishedWork = true;
        }

        state.queue->ExecuteCommandLists(static_cast<UINT>(batch.lists.size()), batch.lists.data());
        batch.legacyTextureStates.commit();
        state.hasUnpublishedWork = true;
        commitAccelerationStructurePublications(batch.entries);
        markSwapchainImagesSubmitted(batch.entries);
        const auto submittedValue = state.nextFenceValue++;
        const HRESULT signalResult = state.queue->Signal(state.fence.Get(), submittedValue);
        if (FAILED(signalResult))
        {
            quarantineBatchEntries(state, batch, BatchRetentions::keep);
            return std::unexpected(nativeError(ErrorCode::deviceLost, signalResult, "ID3D12CommandQueue::Signal"));
        }
        const auto submittedPoint = detail::TimelinePointAccess::make(state.type, submittedValue, deviceToken);
        publishSwapchainSubmissionPoint(batch.entries, submittedPoint);
        state.hasUnpublishedWork = false;
        state.lastSubmittedValue.store(submittedValue, std::memory_order_release);
        for (auto* entry : batch.entries)
        {
            propagateResidencyDependencies(*entry, state.type, submittedValue);
        }
        assignBatchRetirementFenceValue(state, batch.entries.size(), submittedValue);
        return submittedPoint;
    }

    Result<TimelinePoint> QueueManager::submit(std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers,
                                               std::span<const TimelinePoint> waits)
    {
        auto resolved = resolveSubmissionCommands(commandBuffers, impl_->deviceToken);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        const auto& nativeCommands = *resolved;

        auto& state = *nativeCommands.front()->queue();
        for (const auto point : waits)
        {
            if (!impl_->isSubmittedPointOfThisDevice(point))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A queue wait references an invalid, foreign, or unsubmitted timeline point.",
                });
            }
        }

        const std::scoped_lock submissionLock(state.submissionMutex);

        if (!hasUnreservedTimelineSlot(state))
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The queue timeline is exhausted and cannot be reused safely.",
            });
        }

        std::vector<std::vector<ResidencyLease>> inPlaceLeases;
        try
        {
            inPlaceLeases.resize(nativeCommands.size());
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to prepare the update-after-bind elements of a D3D12 submission.",
            });
        }

        for (std::size_t index = 0; index < nativeCommands.size(); ++index)
        {
            auto acquired = Impl::acquireInPlaceResidencies(*nativeCommands[index]->entry(), inPlaceLeases[index]);
            if (!acquired)
            {
                return std::unexpected(std::move(acquired.error()));
            }
        }

        for (std::size_t index = 0; index < nativeCommands.size(); ++index)
        {
            const auto& entry = *nativeCommands[index]->entry();
            auto validDependencies = impl_->validateSubmissionDependencies(entry, state.type, waits);
            if (!validDependencies)
            {
                return std::unexpected(std::move(validDependencies.error()));
            }
            auto validInPlace = impl_->validateResidencyLeases(inPlaceLeases[index], state.type, waits);
            if (!validInPlace)
            {
                return std::unexpected(std::move(validInPlace.error()));
            }
        }

        auto collected = collectSubmissionBatch(nativeCommands);
        if (!collected)
        {
            return std::unexpected(std::move(collected.error()));
        }
        auto& batch = *collected;
        auto layouts = checkSubmissionTextureLayouts(batch);
        if (!layouts)
        {
            return std::unexpected(std::move(layouts.error()));
        }
        auto legacyStates = recordLegacyTextureStatePrologues(*impl_->device, state.type, batch);
        if (!legacyStates)
        {
            return std::unexpected(std::move(legacyStates.error()));
        }
        try
        {
            for (std::size_t index = 0; index < nativeCommands.size(); ++index)
            {
                auto& leases = nativeCommands[index]->entry()->residencyLeases;
                leases.reserve(leases.size() + inPlaceLeases[index].size());
            }
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to retain the residency of update-after-bind binding elements.",
            });
        }

        const std::unique_lock retirementLock(state.retirementMutex);
        auto reserved = reserveBatchRetirementRecords(state, batch);
        if (!reserved)
        {
            return std::unexpected(std::move(reserved.error()));
        }

        for (std::size_t index = 0; index < nativeCommands.size(); ++index)
        {
            auto& leases = nativeCommands[index]->entry()->residencyLeases;
            leases.insert(leases.end(),
                          std::make_move_iterator(inPlaceLeases[index].begin()),
                          std::make_move_iterator(inPlaceLeases[index].end()));
            static_cast<void>(nativeCommands[index]->release());
            commandBuffers[index].reset();
        }
        return impl_->executeSubmissionBatch(state, batch, waits);
    }

    Result<TimelinePointStatus> QueueManager::queryTimelinePoint(TimelinePoint point) const
    {
        if (!isValidQueueType(point.queueType()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The timeline point has an invalid queue type.",
            });
        }
        auto valid = validateSubmittedPoint(point, point.queueType());
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        auto status = queryValue(impl_->queue(point.queueType()), point.value());
        if (status)
        {
            collectGarbage();
        }
        return status;
    }

    Result<TimelinePointStatus> QueueManager::wait(TimelinePoint point, std::chrono::nanoseconds timeout)
    {
        if (!isValidQueueType(point.queueType()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The timeline point has an invalid queue type.",
            });
        }
        auto valid = validateSubmittedPoint(point, point.queueType());
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        auto status = waitForValue(impl_->queue(point.queueType()), point.value(), timeout);
        if (status)
        {
            collectGarbage();
        }
        return status;
    }

    Status QueueManager::wait(TimelinePoint point)
    {
        if (!isValidQueueType(point.queueType()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The timeline point has an invalid queue type.",
            });
        }
        auto valid = validateSubmittedPoint(point, point.queueType());
        if (!valid)
        {
            return valid;
        }
        auto waited = waitForValue(impl_->queue(point.queueType()), point.value());
        if (waited)
        {
            collectGarbage();
        }
        return waited;
    }

    Result<SharedTimelinePoint> QueueManager::exportTimelinePoint(TimelinePoint point)
    {
        if (!point.isValid() || !isValidQueueType(point.queueType()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Only a valid submitted timeline point can be exported.",
            });
        }
        auto valid = validateSubmittedPoint(point, point.queueType());
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }

        HANDLE handle = nullptr;
        const HRESULT result = impl_->device->CreateSharedHandle(
            impl_->queue(point.queueType()).fence.Get(), nullptr, GENERIC_ALL, nullptr, &handle);
        if (FAILED(result))
        {
            return std::unexpected(
                nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                            result,
                            "ID3D12Device::CreateSharedHandle(queue fence)"));
        }
        return SharedTimelinePoint{
            .handle = SharedHandle::adopt(SharedHandleView{
                .type = SharedHandleType::windowsNt,
                .value = reinterpret_cast<std::uintptr_t>(handle),
            }),
            .value = point.value(),
        };
    }

    Result<TimelinePoint> QueueManager::enqueueSharedTimelineWait(QueueType queueType, SharedTimelinePointView point)
    {
        if (!isValidQueueType(queueType) || !point.isValid() || point.handle.type != SharedHandleType::windowsNt ||
            point.value == std::numeric_limits<std::uint64_t>::max())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shared timeline wait requires a valid queue and Windows NT fence/value pair.",
            });
        }

        ComPtr<ID3D12Fence> externalFence;
        const HRESULT openResult =
            impl_->device->OpenSharedHandle(std::bit_cast<HANDLE>(point.handle.value), IID_PPV_ARGS(&externalFence));
        if (FAILED(openResult))
        {
            return std::unexpected(
                nativeError(openResult == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::invalidArgument,
                            openResult,
                            "ID3D12Device::OpenSharedHandle(queue fence)"));
        }
        if (externalFence->GetCompletedValue() == std::numeric_limits<std::uint64_t>::max())
        {
            const HRESULT removedReason = impl_->device->GetDeviceRemovedReason();
            if (FAILED(removedReason))
            {
                return std::unexpected(nativeError(
                    ErrorCode::deviceLost, removedReason, "ID3D12Device::GetDeviceRemovedReason(shared wait)"));
            }
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The imported D3D12 fence reads UINT64_MAX because its producer device was removed or signalled that "
                "value. No shared wait was enqueued.",
            });
        }

        std::shared_ptr<void> retention;
        try
        {
            retention = std::make_shared<ComPtr<ID3D12Fence>>(externalFence);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to retain the imported D3D12 fence.",
            });
        }

        auto& state = impl_->queue(queueType);
        const std::scoped_lock submissionLock(state.submissionMutex);
        if (!hasUnreservedTimelineSlot(state))
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The queue timeline is exhausted and cannot represent a shared wait.",
            });
        }

        const std::unique_lock retentionLock(state.retirementMutex);
        try
        {
            state.quarantinedRetentions.reserve(state.quarantinedRetentions.size() + 1);
            state.pendingRetentions.push_back(PendingRetention{
                .fenceValue = std::numeric_limits<std::uint64_t>::max(),
                .owner = retention,
            });
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to reserve a shared-wait retirement record.",
            });
        }
        const auto discardRetention = [&]() noexcept { state.pendingRetentions.pop_back(); };

        HRESULT result = state.queue->Wait(externalFence.Get(), point.value);
        if (FAILED(result))
        {
            discardRetention();
            return std::unexpected(
                nativeError(ErrorCode::deviceLost, result, "ID3D12CommandQueue::Wait(shared fence)"));
        }
        state.hasUnpublishedWork = true;

        const auto submittedValue = state.nextFenceValue++;
        result = state.queue->Signal(state.fence.Get(), submittedValue);
        if (FAILED(result))
        {
            discardRetention();
            state.quarantinedRetentions.push_back(std::move(retention));
            return std::unexpected(
                nativeError(ErrorCode::deviceLost, result, "ID3D12CommandQueue::Signal(shared wait)"));
        }
        state.hasUnpublishedWork = false;
        state.lastSubmittedValue.store(submittedValue, std::memory_order_release);
        state.pendingRetentions.back().fenceValue = submittedValue;
        return detail::TimelinePointAccess::make(state.type, submittedValue, impl_->deviceToken);
    }

    Status QueueManager::waitIdle(QueueType queueType)
    {
        if (!isValidQueueType(queueType))
        {
            return std::unexpected(Error{ErrorCode::invalidArgument, 0, "The queue type is invalid."});
        }

        auto& state = impl_->queue(queueType);
        const std::scoped_lock submissionLock(state.submissionMutex);
        auto waited = waitForSubmittedWork(state);
        if (!waited)
        {
            return waited;
        }
        collectQueue(state);
        return {};
    }

    Status QueueManager::waitIdle()
    {
        auto& graphics = impl_->queue(QueueType::graphics);
        auto& compute = impl_->queue(QueueType::compute);
        auto& copy = impl_->queue(QueueType::copy);
        const std::scoped_lock submissionLocks(graphics.submissionMutex, compute.submissionMutex, copy.submissionMutex);
        const std::array queues{&graphics, &compute, &copy};
        for (auto* queue : queues)
        {
            auto waited = waitForSubmittedWork(*queue);
            if (!waited)
            {
                return waited;
            }
        }
        for (auto* queue : queues)
        {
            collectQueue(*queue);
        }
        return {};
    }

    std::uint64_t QueueManager::completedValue(QueueType queueType) const noexcept
    {
        if (!isValidQueueType(queueType))
        {
            return 0;
        }
        const auto& state = impl_->queue(queueType);
        const auto value = state.fence->GetCompletedValue();
        if (value == UINT64_MAX)
        {
            return state.lastCompletedValue.load(std::memory_order_acquire);
        }
        const auto completed = recordCompletedValue(state, value);
        if (value > 0)
        {
            collectGarbage();
        }
        return completed;
    }

    void QueueManager::collectGarbage() const noexcept
    {
        for (auto& queue : impl_->queues)
        {
            collectQueue(*queue);
        }
    }

    Result<ID3D12GraphicsCommandList*> QueueManager::commandList(detail::BackendCommandBuffer& commandBuffer) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        return (*live)->entry()->commandList.Get();
    }

    Result<ID3D12GraphicsCommandList1*> QueueManager::commandList1(detail::BackendCommandBuffer& commandBuffer) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        if (native->entry()->commandList1 == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The native graphics command list does not support dynamic depth bounds.",
            });
        }
        return native->entry()->commandList1.Get();
    }

    Result<ID3D12GraphicsCommandList4*> QueueManager::commandList4(detail::BackendCommandBuffer& commandBuffer) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        if (native->entry()->commandList4 == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The native graphics command list does not support acceleration-structure builds.",
            });
        }
        return native->entry()->commandList4.Get();
    }

    Result<ID3D12GraphicsCommandList9*> QueueManager::commandList9(detail::BackendCommandBuffer& commandBuffer) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        if (native->entry()->commandList9 == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The native graphics command list does not support dynamic depth bias.",
            });
        }
        return native->entry()->commandList9.Get();
    }

    Result<ID3D12GraphicsCommandList8*> QueueManager::commandList8(detail::BackendCommandBuffer& commandBuffer) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        if (native->entry()->commandList8 == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The native graphics command list does not support independent stencil references.",
            });
        }
        return native->entry()->commandList8.Get();
    }

    Result<ID3D12GraphicsCommandList6*> QueueManager::commandList6(detail::BackendCommandBuffer& commandBuffer) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        if (native->entry()->commandList6 == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The native graphics command list does not support mesh dispatch.",
            });
        }
        return native->entry()->commandList6.Get();
    }

    Result<ID3D12GraphicsCommandList7*> QueueManager::commandList7(detail::BackendCommandBuffer& commandBuffer) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        if (native->entry()->commandList7 == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The native graphics command list does not support enhanced barriers.",
            });
        }
        return native->entry()->commandList7.Get();
    }

    Result<ID3D12GraphicsCommandList5*> QueueManager::commandList5(detail::BackendCommandBuffer& commandBuffer) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        if (native->entry()->commandList5 == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The native graphics command list does not support variable-rate shading.",
            });
        }
        return native->entry()->commandList5.Get();
    }

    Status QueueManager::retainResource(detail::BackendCommandBuffer& commandBuffer,
                                        std::shared_ptr<void> resource) const
    {
        return retainResource(commandBuffer, std::move(resource), std::span<const std::shared_ptr<ResidencyRecord>>{});
    }

    Status QueueManager::retainResource(detail::BackendCommandBuffer& commandBuffer,
                                        std::shared_ptr<void> resource,
                                        const std::shared_ptr<ResidencyRecord>& residency) const
    {
        const std::array residencies{residency};
        return retainResource(commandBuffer, std::move(resource), residencies);
    }

    Status QueueManager::retainResource(detail::BackendCommandBuffer& commandBuffer,
                                        std::shared_ptr<void> resource,
                                        std::span<const std::shared_ptr<ResidencyRecord>> residencies) const
    {
        return retainResource(commandBuffer, std::move(resource), residencies, false);
    }

    Status QueueManager::retainInternalResource(detail::BackendCommandBuffer& commandBuffer,
                                                std::shared_ptr<void> resource) const
    {
        return retainResource(
            commandBuffer, std::move(resource), std::span<const std::shared_ptr<ResidencyRecord>>{}, true);
    }

    Status QueueManager::retainInternalResource(detail::BackendCommandBuffer& commandBuffer,
                                                std::shared_ptr<void> resource,
                                                const std::shared_ptr<ResidencyRecord>& residency) const
    {
        const std::array residencies{residency};
        return retainResource(commandBuffer, std::move(resource), residencies, true);
    }

    Status QueueManager::retainInternalResource(detail::BackendCommandBuffer& commandBuffer,
                                                std::shared_ptr<void> resource,
                                                std::span<const std::shared_ptr<ResidencyRecord>> residencies) const
    {
        return retainResource(commandBuffer, std::move(resource), residencies, true);
    }

    Status QueueManager::retainSwapchainImageLease(detail::BackendCommandBuffer& commandBuffer,
                                                   const std::shared_ptr<SwapchainImageLeaseState>& lease) const
    {
        const std::array leases{lease};
        return retainSwapchainImageLeases(commandBuffer, leases);
    }

    Status QueueManager::retainSwapchainImageLeases(
        detail::BackendCommandBuffer& commandBuffer,
        std::span<const std::shared_ptr<SwapchainImageLeaseState>> leases) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer,
                                            impl_->deviceToken,
                                            "Cannot retain a swapchain image on a foreign or invalid command buffer.");
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        const auto leaseCount =
            static_cast<std::size_t>(std::ranges::count_if(leases, [](const auto& lease) { return lease != nullptr; }));
        if (leaseCount == 0)
        {
            return {};
        }

        std::vector<std::shared_ptr<SwapchainImageCommandReference>> references;
        try
        {
            references.reserve(leaseCount);
            native->entry()->swapchainImageReferences.reserve(native->entry()->swapchainImageReferences.size() +
                                                              leaseCount);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to reserve swapchain image command retention.",
            });
        }
        for (const auto& lease : leases)
        {
            if (lease == nullptr)
            {
                continue;
            }
            auto reference = lease->retainCommandReference(lease, native->queueType());
            if (!reference)
            {
                return std::unexpected(std::move(reference.error()));
            }
            references.push_back(std::move(*reference));
        }
        native->entry()->swapchainImageReferences.insert(native->entry()->swapchainImageReferences.end(),
                                                         std::make_move_iterator(references.begin()),
                                                         std::make_move_iterator(references.end()));
        return {};
    }

    Status QueueManager::retainCommandDependencies(detail::BackendCommandBuffer& commandBuffer,
                                                   std::span<const ResourceRetention> resources,
                                                   std::span<const std::shared_ptr<SwapchainImageLeaseState>> leases,
                                                   std::span<const TimelinePoint> sparseMappingDependencies,
                                                   std::span<const InPlaceResidencyReference> inPlaceResidencies) const
    {
        auto live =
            requireLiveD3D12Command(&commandBuffer,
                                    impl_->deviceToken,
                                    "Cannot retain command dependencies on a foreign or invalid command buffer.");
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;

        auto& entry = *native->entry();
        const auto retainedResourceCount = entry.retainedResources.size();
        const auto residencyLeaseCount = entry.residencyLeases.size();
        const auto inPlaceResidencyCount = entry.inPlaceResidencies.size();
        const auto swapchainReferenceCount = entry.swapchainImageReferences.size();
        const auto sparseMappingDependencyCount = entry.sparseMappingDependencies.size();
        const auto rollback = [&]() noexcept {
            entry.retainedResources.resize(retainedResourceCount);
            entry.residencyLeases.resize(residencyLeaseCount);
            entry.inPlaceResidencies.resize(inPlaceResidencyCount);
            entry.swapchainImageReferences.resize(swapchainReferenceCount);
            entry.sparseMappingDependencies.resize(sparseMappingDependencyCount);
        };

        try
        {
            for (const auto& resource : resources)
            {
                const std::array singularResidency{resource.residency};
                std::span<const std::shared_ptr<ResidencyRecord>> residencies = resource.residencySet;
                if (residencies.empty())
                {
                    residencies = singularResidency;
                }
                auto retained = retainResource(commandBuffer, resource.owner, residencies, resource.internal);
                if (!retained)
                {
                    rollback();
                    return retained;
                }
            }
            auto retainedLeases = retainSwapchainImageLeases(commandBuffer, leases);
            if (!retainedLeases)
            {
                rollback();
                return retainedLeases;
            }
            auto retainedSparseDependencies = retainSparseMappingDependencies(commandBuffer, sparseMappingDependencies);
            if (!retainedSparseDependencies)
            {
                rollback();
                return retainedSparseDependencies;
            }
            for (const auto& reference : inPlaceResidencies)
            {
                const bool repeated = !entry.inPlaceResidencies.empty() &&
                                      entry.inPlaceResidencies.back().log == reference.log &&
                                      entry.inPlaceResidencies.back().sequence <= reference.sequence;
                if (!repeated)
                {
                    entry.inPlaceResidencies.push_back(reference);
                }
            }
        }
        catch (const std::bad_alloc&)
        {
            rollback();
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate transactional command dependency retention.",
            });
        }
        return {};
    }

    Result<LegacyTextureStateRecording*> QueueManager::legacyTextureStates(
        detail::BackendCommandBuffer& commandBuffer) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        return &(*live)->entry()->legacyTextureStates;
    }

    Result<std::uint64_t> QueueManager::recordingId(detail::BackendCommandBuffer& commandBuffer) const
    {
        auto live = requireLiveD3D12Command(&commandBuffer, impl_->deviceToken);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        if (native->entry()->recordingId == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The command buffer does not hold an active command-list recording.",
            });
        }
        return native->entry()->recordingId;
    }

    Result<std::shared_ptr<void>> QueueManager::recordingDescriptorArena(
        detail::BackendCommandBuffer& commandBuffer) const
    {
        auto recording = recordingId(commandBuffer);
        if (!recording)
        {
            return std::unexpected(std::move(recording.error()));
        }
        const auto* entry = asD3D12Command(&commandBuffer, impl_->deviceToken)->entry();
        if (entry->descriptorArenaRecordingId != *recording)
        {
            return std::shared_ptr<void>{};
        }
        return entry->descriptorArena.lock();
    }

    Status QueueManager::setRecordingDescriptorArena(detail::BackendCommandBuffer& commandBuffer,
                                                     const std::shared_ptr<void>& arena) const
    {
        auto recording = recordingId(commandBuffer);
        if (!recording)
        {
            return std::unexpected(std::move(recording.error()));
        }
        auto* entry = asD3D12Command(&commandBuffer, impl_->deviceToken)->entry();
        entry->descriptorArena = arena;
        entry->descriptorArenaRecordingId = *recording;
        return {};
    }

    Status QueueManager::retainAccelerationStructureDependencies(detail::BackendCommandBuffer& commandBuffer,
                                                                 std::span<const ResourceRetention> resources,
                                                                 AccelerationStructureExecutionLease lease,
                                                                 AccelerationStructurePublication publication) const
    {
        auto live =
            requireLiveD3D12Command(&commandBuffer,
                                    impl_->deviceToken,
                                    "Cannot retain command dependencies on a foreign or invalid command buffer.");
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;

        auto& entry = *native->entry();
        const bool publishes = !publication.record.expired();
        try
        {
            if (lease.active())
            {
                reserveForAppend(entry.accelerationStructureLeases, 1U);
            }
            if (publishes)
            {
                entry.accelerationStructurePublications.reserve(entry.accelerationStructurePublications.size() + 1U);
            }
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate acceleration-structure execution lease retention.",
            });
        }

        auto retained = retainCommandDependencies(commandBuffer, resources, {});
        if (!retained)
        {
            return retained;
        }
        if (lease.active())
        {
            entry.accelerationStructureLeases.push_back(std::move(lease));
        }
        if (publishes)
        {
            entry.accelerationStructurePublications.push_back(std::move(publication));
        }
        return {};
    }

    Status QueueManager::retainResource(detail::BackendCommandBuffer& commandBuffer,
                                        std::shared_ptr<void> resource,
                                        std::span<const std::shared_ptr<ResidencyRecord>> residencies,
                                        bool internal) const
    {
        if (resource == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Cannot retain a null resource on a command buffer.",
            });
        }
        auto live = requireLiveD3D12Command(
            &commandBuffer, impl_->deviceToken, "Cannot retain a resource on a foreign or invalid command buffer.");
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        const bool retainOwner = internal || native->owner() == nullptr ||
                                 !hasAny(native->owner()->flags, CommandPoolFlag::unretainedResources);
        auto& entry = *native->entry();
        const auto retainedResourceCount = entry.retainedResources.size();
        const auto residencyLeaseCount = entry.residencyLeases.size();
        const auto rollback = [&]() noexcept {
            entry.retainedResources.resize(retainedResourceCount);
            entry.residencyLeases.resize(residencyLeaseCount);
        };
        try
        {
            if (test::consumeResourceRetentionFailure())
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Injected command resource retention failure.",
                });
            }
            if (retainOwner)
            {
                detail::reserveAdditional(entry.retainedResources, 1U);
            }
            auto retainedResidencies = appendResidencyLeases(entry.residencyLeases, residencies);
            if (!retainedResidencies)
            {
                rollback();
                return retainedResidencies;
            }
            if (retainOwner)
            {
                entry.retainedResources.push_back(std::move(resource));
            }
            return {};
        }
        catch (const std::bad_alloc&)
        {
            rollback();
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate command resource retention.",
            });
        }
    }

    Status QueueManager::retainSparseMappingDependencies(detail::BackendCommandBuffer& commandBuffer,
                                                         std::span<const TimelinePoint> dependencies) const
    {
        auto live = requireLiveD3D12Command(
            &commandBuffer,
            impl_->deviceToken,
            "Cannot retain sparse mapping dependencies on a foreign or invalid command buffer.");
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }
        auto* native = *live;
        const auto originalSize = native->entry()->sparseMappingDependencies.size();
        try
        {
            for (const auto dependency : dependencies)
            {
                if (!dependency.isValid() || detail::TimelinePointAccess::deviceToken(dependency) != impl_->deviceToken)
                {
                    native->entry()->sparseMappingDependencies.resize(originalSize);
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "A sparse mapping dependency is invalid or belongs to another device.",
                    });
                }
                if (std::ranges::find(native->entry()->sparseMappingDependencies, dependency) ==
                    native->entry()->sparseMappingDependencies.end())
                {
                    native->entry()->sparseMappingDependencies.push_back(dependency);
                }
            }
        }
        catch (const std::bad_alloc&)
        {
            native->entry()->sparseMappingDependencies.resize(originalSize);
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to retain sparse mapping dependencies.",
            });
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] Result<D3D12_RESIDENCY_FLAGS> nativeResidencyFlags(ResidencyBudgetPolicy budgetPolicy)
        {
            switch (budgetPolicy)
            {
            case ResidencyBudgetPolicy::allowOverBudget:
                return D3D12_RESIDENCY_FLAG_NONE;
            case ResidencyBudgetPolicy::denyOverBudget:
                return D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET;
            default:
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The residency budget policy is invalid.",
                });
            }
        }

        [[nodiscard]] Result<std::shared_ptr<MakeResidentRetention>> allocateMakeResidentRetention(
            std::span<const std::shared_ptr<void>> owners)
        {
            try
            {
                return std::make_shared<MakeResidentRetention>(owners);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Could not allocate make-resident retention storage.",
                });
            }
        }

        [[nodiscard]] Status reservePendingRetention(QueueState& state,
                                                     const std::shared_ptr<void>& owner,
                                                     std::string_view failureMessage)
        {
            try
            {
                state.quarantinedRetentions.reserve(state.quarantinedRetentions.size() + 1);
                state.pendingRetentions.push_back(PendingRetention{
                    .fenceValue = std::numeric_limits<std::uint64_t>::max(),
                    .owner = owner,
                });
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    std::string(failureMessage),
                });
            }
            return {};
        }

        void discardPendingRetention(QueueState& state) noexcept
        {
            state.pendingRetentions.pop_back();
        }

        void quarantinePendingRetention(QueueState& state) noexcept
        {
            auto owner = std::move(state.pendingRetentions.back().owner);
            discardPendingRetention(state);
            state.quarantinedRetentions.push_back(std::move(owner));
        }

        [[nodiscard]] Error syntheticMakeResidentFailure(std::string_view message)
        {
            return Error{
                ErrorCode::nativeFailure,
                static_cast<std::int64_t>(E_FAIL),
                std::string(message),
            };
        }

    } // namespace

    Result<std::uint64_t> QueueManager::Impl::acceptPendingMakeResident(QueueState& state,
                                                                        D3D12_RESIDENCY_FLAGS flags,
                                                                        std::span<ID3D12Pageable* const> pageables)
    {
        const std::scoped_lock residencyLock(residencyMutex);
        if (nextResidencyFenceValue == std::numeric_limits<std::uint64_t>::max())
        {
            discardPendingRetention(state);
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The internal residency timeline is exhausted.",
            });
        }
        if (test::consumeMakeResidentFailure(test::MakeResidentFailurePoint::beforeNativeAcceptance))
        {
            discardPendingRetention(state);
            return std::unexpected(
                syntheticMakeResidentFailure("Synthetic make-resident failure before native acceptance."));
        }
        const auto residencyValue = nextResidencyFenceValue++;
        const HRESULT result = residencyDevice->EnqueueMakeResident(
            flags, static_cast<UINT>(pageables.size()), pageables.data(), residencyFence.Get(), residencyValue);
        if (FAILED(result))
        {
            discardPendingRetention(state);
            return std::unexpected(
                nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                            result,
                            "ID3D12Device3::EnqueueMakeResident"));
        }
        lastAcceptedResidencyFenceValue = residencyValue;
        return residencyValue;
    }

    QueueManager::ResidencySubmission QueueManager::Impl::publishAcceptedMakeResident(QueueState& state,
                                                                                      MakeResidentRetention& retention,
                                                                                      std::uint64_t residencyValue)
    {
        const auto quarantineRetention = [&]() noexcept {
            retention.quarantine(*residencyFence.Get(), residencyValue);
            quarantinePendingRetention(state);
        };

        if (test::consumeMakeResidentFailure(test::MakeResidentFailurePoint::afterNativeAcceptance))
        {
            quarantineRetention();
            return ResidencySubmission{
                .failure = syntheticMakeResidentFailure("Synthetic make-resident failure after native acceptance."),
            };
        }

        HRESULT result = state.queue->Wait(residencyFence.Get(), residencyValue);
        if (FAILED(result))
        {
            quarantineRetention();
            return ResidencySubmission{
                .failure = nativeError(ErrorCode::deviceLost, result, "ID3D12CommandQueue::Wait(residency)"),
            };
        }
        state.hasUnpublishedWork = true;
        if (test::consumeMakeResidentFailure(test::MakeResidentFailurePoint::afterQueueWait))
        {
            quarantineRetention();
            return ResidencySubmission{
                .failure =
                    syntheticMakeResidentFailure("Synthetic make-resident publication failure after the queue wait."),
            };
        }
        const auto submittedValue = state.nextFenceValue++;
        result = state.queue->Signal(state.fence.Get(), submittedValue);
        if (FAILED(result))
        {
            quarantineRetention();
            return ResidencySubmission{
                .failure = nativeError(ErrorCode::deviceLost, result, "ID3D12CommandQueue::Signal(residency)"),
            };
        }
        if (test::consumeMakeResidentFailure(test::MakeResidentFailurePoint::afterQueueSignal))
        {
            quarantineRetention();
            return ResidencySubmission{
                .failure =
                    syntheticMakeResidentFailure("Synthetic make-resident publication failure after the queue signal."),
            };
        }
        state.hasUnpublishedWork = false;
        state.lastSubmittedValue.store(submittedValue, std::memory_order_release);
        state.pendingRetentions.back().fenceValue = submittedValue;
        return ResidencySubmission{
            .publication = detail::TimelinePointAccess::make(state.type, submittedValue, deviceToken),
        };
    }

    Result<QueueManager::ResidencySubmission> QueueManager::enqueueMakeResident(
        QueueType queueType,
        std::span<ID3D12Pageable* const> pageables,
        std::span<const std::shared_ptr<void>> owners,
        ResidencyBudgetPolicy budgetPolicy)
    {
        if (!isValidQueueType(queueType) || pageables.empty() || pageables.size() != owners.size() ||
            pageables.size() > std::numeric_limits<UINT>::max() ||
            std::ranges::any_of(pageables, [](const auto* pageable) { return pageable == nullptr; }))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Make-resident requires a valid queue and matching non-empty pageable-resource storage within D3D12 "
                "limits.",
            });
        }
        if (impl_->residencyDevice == nullptr || impl_->residencyFence == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Asynchronous manual residency is unavailable on this device.",
            });
        }
        auto flags = nativeResidencyFlags(budgetPolicy);
        if (!flags)
        {
            return std::unexpected(std::move(flags.error()));
        }

        auto retention = allocateMakeResidentRetention(owners);
        if (!retention)
        {
            return std::unexpected(std::move(retention.error()));
        }
        auto& state = impl_->queue(queueType);
        const std::scoped_lock submissionLock(state.submissionMutex);
        if (!hasUnreservedTimelineSlot(state))
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The queue timeline is exhausted and cannot represent a make-resident transition.",
            });
        }

        const std::unique_lock retentionLock(state.retirementMutex);
        auto reserved =
            reservePendingRetention(state, *retention, "Could not allocate make-resident publication storage.");
        if (!reserved)
        {
            return std::unexpected(std::move(reserved.error()));
        }

        auto residencyValue = impl_->acceptPendingMakeResident(state, *flags, pageables);
        if (!residencyValue)
        {
            return std::unexpected(std::move(residencyValue.error()));
        }
        return impl_->publishAcceptedMakeResident(state, **retention, *residencyValue);
    }

    namespace
    {

        void updateTileMappings(ID3D12CommandQueue& queue, std::span<const QueueManager::SparseTileUpdate> mappings)
        {
            for (const auto& mapping : mappings)
            {
                const D3D12_TILE_RANGE_FLAGS rangeFlag =
                    mapping.heap == nullptr ? D3D12_TILE_RANGE_FLAG_NULL : D3D12_TILE_RANGE_FLAG_NONE;
                const UINT heapTileOffset = mapping.heapTileOffset;
                const UINT tileCount = mapping.tileCount;
                queue.UpdateTileMappings(mapping.resource,
                                         1,
                                         &mapping.coordinate,
                                         &mapping.region,
                                         mapping.heap,
                                         1,
                                         &rangeFlag,
                                         &heapTileOffset,
                                         &tileCount,
                                         D3D12_TILE_MAPPING_FLAG_NONE);
            }
        }

    } // namespace

    HRESULT QueueManager::Impl::enqueueSparseMappingDependencyWaits(QueueState& mappingState,
                                                                    std::span<const TimelinePoint> dependencies)
    {
        for (const auto dependency : dependencies)
        {
            if (dependency.queueType() == mappingState.type)
            {
                continue;
            }
            auto& dependencyQueue = queue(dependency.queueType());
            const HRESULT waitResult = mappingState.queue->Wait(dependencyQueue.fence.Get(), dependency.value());
            if (FAILED(waitResult))
            {
                return waitResult;
            }
            mappingState.hasUnpublishedWork = true;
        }
        return S_OK;
    }

    QueueManager::SparseMappingSubmission QueueManager::Impl::relaySparseMappingPublication(
        QueueState& mappingState, QueueState& publicationState, TimelinePoint mappingPublication)
    {
        HRESULT publicationResult = publicationState.queue->Wait(mappingState.fence.Get(), mappingPublication.value());
        if (FAILED(publicationResult))
        {
            quarantinePendingRetention(mappingState);
            return SparseMappingSubmission{
                .publication = mappingPublication,
                .failure = nativeError(
                    ErrorCode::deviceLost, publicationResult, "ID3D12CommandQueue::Wait(sparse mapping publication)"),
            };
        }
        publicationState.hasUnpublishedWork = true;
        const auto publicationValue = publicationState.nextFenceValue++;
        publicationResult = publicationState.queue->Signal(publicationState.fence.Get(), publicationValue);
        if (FAILED(publicationResult))
        {
            quarantinePendingRetention(mappingState);
            return SparseMappingSubmission{
                .publication = mappingPublication,
                .failure = nativeError(
                    ErrorCode::deviceLost, publicationResult, "ID3D12CommandQueue::Signal(sparse mapping publication)"),
            };
        }
        publicationState.hasUnpublishedWork = false;
        publicationState.lastSubmittedValue.store(publicationValue, std::memory_order_release);
        return SparseMappingSubmission{
            .publication = detail::TimelinePointAccess::make(publicationState.type, publicationValue, deviceToken),
        };
    }

    Result<QueueManager::SparseMappingSubmission> QueueManager::Impl::enqueueSparseMappingsLocked(
        QueueState& mappingState,
        QueueState& publicationState,
        std::span<const SparseTileUpdate> mappings,
        std::span<const TimelinePoint> dependencies,
        const std::shared_ptr<void>& owner)
    {
        for (const auto dependency : dependencies)
        {
            if (!isSubmittedPointOfThisDevice(dependency))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse mapping batch depends on an invalid, foreign, or unsubmitted timeline point.",
                });
            }
        }
        if (!hasUnreservedTimelineSlot(mappingState) ||
            (&publicationState != &mappingState && !hasUnreservedTimelineSlot(publicationState)))
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The mapping or publication queue timeline is exhausted and cannot represent a sparse update.",
            });
        }

        const std::unique_lock retentionLock(mappingState.retirementMutex);
        auto reserved =
            reservePendingRetention(mappingState, owner, "Failed to reserve sparse mapping retirement storage.");
        if (!reserved)
        {
            return std::unexpected(std::move(reserved.error()));
        }

        const HRESULT waitResult = enqueueSparseMappingDependencyWaits(mappingState, dependencies);
        if (FAILED(waitResult))
        {
            discardPendingRetention(mappingState);
            return std::unexpected(
                nativeError(ErrorCode::deviceLost, waitResult, "ID3D12CommandQueue::Wait(sparse mapping)"));
        }
        updateTileMappings(*mappingState.queue.Get(), mappings);
        mappingState.hasUnpublishedWork = true;

        const auto mappingValue = mappingState.nextFenceValue++;
        const HRESULT mappingSignal = test::nextSparseMappingSignalFailure().exchange(false, std::memory_order_relaxed)
                                          ? E_FAIL
                                          : mappingState.queue->Signal(mappingState.fence.Get(), mappingValue);
        if (FAILED(mappingSignal))
        {
            quarantinePendingRetention(mappingState);
            return SparseMappingSubmission{
                .failure =
                    nativeError(ErrorCode::deviceLost, mappingSignal, "ID3D12CommandQueue::Signal(sparse mapping)"),
            };
        }
        mappingState.hasUnpublishedWork = false;
        mappingState.lastSubmittedValue.store(mappingValue, std::memory_order_release);
        mappingState.pendingRetentions.back().fenceValue = mappingValue;
        const auto mappingPublication = detail::TimelinePointAccess::make(mappingState.type, mappingValue, deviceToken);
        if (&publicationState == &mappingState)
        {
            return SparseMappingSubmission{.publication = mappingPublication};
        }
        return relaySparseMappingPublication(mappingState, publicationState, mappingPublication);
    }

    Result<QueueManager::SparseMappingSubmission> QueueManager::enqueueSparseMappings(
        QueueType queueType,
        std::span<const SparseTileUpdate> mappings,
        std::span<const TimelinePoint> dependencies,
        std::shared_ptr<void> owner)
    {
        if (!isValidQueueType(queueType) || mappings.empty() || owner == nullptr ||
            mappings.size() > std::numeric_limits<UINT>::max() ||
            std::ranges::any_of(mappings, [](const SparseTileUpdate& mapping) {
                return mapping.resource == nullptr || mapping.tileCount == 0;
            }))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Sparse mapping requires a valid queue, owner, and non-empty native tile updates.",
            });
        }

        auto& mappingState = impl_->queue(QueueType::graphics);
        auto& publicationState = impl_->queue(queueType);
        if (&publicationState == &mappingState)
        {
            const std::scoped_lock submissionLock(mappingState.submissionMutex);
            return impl_->enqueueSparseMappingsLocked(mappingState, publicationState, mappings, dependencies, owner);
        }
        const std::scoped_lock submissionLocks(mappingState.submissionMutex, publicationState.submissionMutex);
        return impl_->enqueueSparseMappingsLocked(mappingState, publicationState, mappings, dependencies, owner);
    }

    Result<ResidencyState> QueueManager::residencyState(
        const std::shared_ptr<ResidencyRecord>& residency) const noexcept
    {
        if (residency == nullptr)
        {
            return ResidencyState::automatic;
        }
        const std::scoped_lock lock(residency->mutex);
        if (residency->transitionUnpublishable)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The native make-resident transition has no publishable portable completion point.",
            });
        }
        static_cast<void>(impl_->promoteCompleted(*residency));
        return residency->state;
    }

    Result<ID3D12CommandQueue*> QueueManager::nativeQueue(QueueType queueType) const
    {
        if (!isValidQueueType(queueType))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The queue type is invalid.",
            });
        }
        auto* queue = impl_->queue(queueType).queue.Get();
        if (queue == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The requested D3D12 queue is not initialized.",
            });
        }
        return queue;
    }

    Status QueueManager::validateSubmittedPoint(TimelinePoint point, QueueType requiredQueue) const
    {
        if (!isValidQueueType(requiredQueue) || !point.isValid() || point.queueType() != requiredQueue ||
            detail::TimelinePointAccess::deviceToken(point) != impl_->deviceToken ||
            point.value() > impl_->queue(requiredQueue).lastSubmittedValue.load(std::memory_order_acquire))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The timeline point is invalid, foreign, on the wrong queue, or has not been submitted.",
            });
        }
        return {};
    }

    Result<QueueSignalReservation> QueueManager::reserveSignal(QueueType queueType)
    {
        if (!isValidQueueType(queueType))
        {
            return std::unexpected(Error{ErrorCode::invalidArgument, 0, "The queue type is invalid."});
        }
        if (test::nextQueueSignalReservationFailure().exchange(false, std::memory_order_relaxed))
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Injected queue signal reservation failure.",
            });
        }
        auto& state = impl_->queue(queueType);
        std::unique_ptr<QueueSignalReservation::Impl> reservation;
        try
        {
            reservation = std::make_unique<QueueSignalReservation::Impl>(state, impl_->deviceToken);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to reserve queue signal synchronization.",
            });
        }
        const std::scoped_lock lock(state.submissionMutex);
        if (!hasUnreservedTimelineSlot(state))
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The queue timeline is exhausted and cannot be reused safely.",
            });
        }
        ++state.reservedSignalCount;
        reservation->active = true;
        return QueueSignalReservation(std::move(reservation));
    }

} // namespace rachis::d3d12
