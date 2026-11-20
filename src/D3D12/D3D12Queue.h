#pragma once

#include "D3D12/D3D12AccelerationStructureLease.h"
#include "D3D12/D3D12Residency.h"
#include "D3D12/D3D12SwapchainLease.h"
#include "Internal/BackendCommandBuffer.h"
#include "Rachis/Error.h"
#include "Rachis/Interop.h"

#include <d3d12.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace rachis::d3d12
{

    class LegacyTextureStateRecording;

    class QueueSignalReservation final
    {
    public:
        QueueSignalReservation() noexcept;
        ~QueueSignalReservation();
        QueueSignalReservation(const QueueSignalReservation&) = delete;
        QueueSignalReservation& operator=(const QueueSignalReservation&) = delete;
        QueueSignalReservation(QueueSignalReservation&&) noexcept;
        QueueSignalReservation& operator=(QueueSignalReservation&&) noexcept;

        [[nodiscard]] Result<TimelinePoint> signal();

    private:
        friend class QueueManager;
        struct Impl;
        explicit QueueSignalReservation(std::unique_ptr<Impl> impl) noexcept;
        std::unique_ptr<Impl> impl_;
    };

    class QueueManager final
    {
    public:
        struct ResourceRetention final
        {
            std::shared_ptr<void> owner = {};
            std::shared_ptr<ResidencyRecord> residency = {};

            std::span<const std::shared_ptr<ResidencyRecord>> residencySet = {};

            bool internal = false;
        };

        QueueManager();
        ~QueueManager();

        QueueManager(const QueueManager&) = delete;
        QueueManager& operator=(const QueueManager&) = delete;
        QueueManager(QueueManager&&) = delete;
        QueueManager& operator=(QueueManager&&) = delete;

        [[nodiscard]] Status initialize(ID3D12Device& device, std::uint64_t deviceToken);
        [[nodiscard]] Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc& desc);
        [[nodiscard]] Status destroyCommandPool(CommandPoolHandle commandPool);
        [[nodiscard]] Status resetCommandPool(CommandPoolHandle commandPool);
        [[nodiscard]] Result<std::unique_ptr<detail::BackendCommandBuffer>> acquire(CommandPoolHandle commandPool);
        [[nodiscard]] Status reset(detail::BackendCommandBuffer& commandBuffer);
        [[nodiscard]] Result<TimelinePoint> submit(
            std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers,
            std::span<const TimelinePoint> waits);
        [[nodiscard]] Result<TimelinePointStatus> queryTimelinePoint(TimelinePoint point) const;
        [[nodiscard]] Result<TimelinePointStatus> wait(TimelinePoint point, std::chrono::nanoseconds timeout);
        [[nodiscard]] Status wait(TimelinePoint point);
        [[nodiscard]] Result<SharedTimelinePoint> exportTimelinePoint(TimelinePoint point);
        [[nodiscard]] Result<TimelinePoint> enqueueSharedTimelineWait(QueueType queueType,
                                                                      SharedTimelinePointView point);
        [[nodiscard]] Status waitIdle(QueueType queueType);
        [[nodiscard]] Status waitIdle();
        [[nodiscard]] std::uint64_t completedValue(QueueType queueType) const noexcept;
        void collectGarbage() const noexcept;
        [[nodiscard]] Result<ID3D12GraphicsCommandList*> commandList(detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Result<ID3D12GraphicsCommandList1*> commandList1(
            detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Result<ID3D12GraphicsCommandList4*> commandList4(
            detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Result<ID3D12GraphicsCommandList5*> commandList5(
            detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Result<ID3D12GraphicsCommandList6*> commandList6(
            detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Result<ID3D12GraphicsCommandList7*> commandList7(
            detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Result<ID3D12GraphicsCommandList8*> commandList8(
            detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Result<ID3D12GraphicsCommandList9*> commandList9(
            detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Status retainResource(detail::BackendCommandBuffer& commandBuffer,
                                            std::shared_ptr<void> resource) const;
        [[nodiscard]] Status retainResource(detail::BackendCommandBuffer& commandBuffer,
                                            std::shared_ptr<void> resource,
                                            const std::shared_ptr<ResidencyRecord>& residency) const;
        [[nodiscard]] Status retainResource(detail::BackendCommandBuffer& commandBuffer,
                                            std::shared_ptr<void> resource,
                                            std::span<const std::shared_ptr<ResidencyRecord>> residencies) const;
        [[nodiscard]] Status retainInternalResource(detail::BackendCommandBuffer& commandBuffer,
                                                    std::shared_ptr<void> resource) const;
        [[nodiscard]] Status retainInternalResource(detail::BackendCommandBuffer& commandBuffer,
                                                    std::shared_ptr<void> resource,
                                                    const std::shared_ptr<ResidencyRecord>& residency) const;
        [[nodiscard]] Status retainInternalResource(
            detail::BackendCommandBuffer& commandBuffer,
            std::shared_ptr<void> resource,
            std::span<const std::shared_ptr<ResidencyRecord>> residencies) const;
        [[nodiscard]] Status retainSparseMappingDependencies(detail::BackendCommandBuffer& commandBuffer,
                                                             std::span<const TimelinePoint> dependencies) const;
        [[nodiscard]] Status retainSwapchainImageLease(detail::BackendCommandBuffer& commandBuffer,
                                                       const std::shared_ptr<SwapchainImageLeaseState>& lease) const;
        [[nodiscard]] Status retainSwapchainImageLeases(
            detail::BackendCommandBuffer& commandBuffer,
            std::span<const std::shared_ptr<SwapchainImageLeaseState>> leases) const;

        [[nodiscard]] Status retainCommandDependencies(
            detail::BackendCommandBuffer& commandBuffer,
            std::span<const ResourceRetention> resources,
            std::span<const std::shared_ptr<SwapchainImageLeaseState>> leases,
            std::span<const TimelinePoint> sparseMappingDependencies = {},
            std::span<const InPlacePublicationReference> inPlacePublications = {}) const;
        [[nodiscard]] Result<std::uint64_t> recordingId(detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Result<std::shared_ptr<void>> recordingDescriptorArena(
            detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Status setRecordingDescriptorArena(detail::BackendCommandBuffer& commandBuffer,
                                                         const std::shared_ptr<void>& arena) const;
        [[nodiscard]] Result<LegacyTextureStateRecording*> legacyTextureStates(
            detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Status retainAccelerationStructureDependencies(
            detail::BackendCommandBuffer& commandBuffer,
            std::span<const ResourceRetention> resources,
            AccelerationStructureExecutionLease lease,
            AccelerationStructurePublication publication = {},
            std::span<const TimelinePoint> sparseMappingDependencies = {}) const;
        struct ResidencySubmission final
        {
            TimelinePoint publication = {};
            std::optional<Error> failure = {};
        };
        [[nodiscard]] Result<ResidencySubmission> enqueueMakeResident(QueueType queueType,
                                                                      std::span<ID3D12Pageable* const> pageables,
                                                                      std::span<const std::shared_ptr<void>> owners,
                                                                      ResidencyBudgetPolicy budgetPolicy);
        struct SparseTileUpdate final
        {
            ID3D12Resource* resource = nullptr;
            ID3D12Heap* heap = nullptr;
            D3D12_TILED_RESOURCE_COORDINATE coordinate{};
            D3D12_TILE_REGION_SIZE region{};
            std::uint32_t tileCount = 0;
            std::uint32_t heapTileOffset = 0;
        };
        struct SparseMappingSubmission final
        {
            TimelinePoint publication = {};
            std::optional<Error> failure = {};
        };
        [[nodiscard]] Result<SparseMappingSubmission> enqueueSparseMappings(QueueType queueType,
                                                                            std::span<const SparseTileUpdate> mappings,
                                                                            std::span<const TimelinePoint> dependencies,
                                                                            std::shared_ptr<void> owner);
        [[nodiscard]] Result<ResidencyState> residencyState(
            const std::shared_ptr<ResidencyRecord>& residency) const noexcept;
        [[nodiscard]] Result<ID3D12CommandQueue*> nativeQueue(QueueType queueType) const;
        [[nodiscard]] Status validateSubmittedPoint(TimelinePoint point, QueueType requiredQueue) const;
        [[nodiscard]] Result<QueueSignalReservation> reserveSignal(QueueType queueType);

    private:
        [[nodiscard]] Status retainResource(detail::BackendCommandBuffer& commandBuffer,
                                            std::shared_ptr<void> resource,
                                            std::span<const std::shared_ptr<ResidencyRecord>> residencies,
                                            bool internal) const;

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rachis::d3d12
