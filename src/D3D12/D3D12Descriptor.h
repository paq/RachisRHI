#pragma once

#include "Rachis/Error.h"

#include <d3d12.h>

#include <cstdint>
#include <memory>

namespace rachis::d3d12
{

    struct DescriptorAllocationStatistics final
    {
        std::uint64_t liveCount = 0;
        std::uint64_t reservedCount = 0;

        DescriptorAllocationStatistics& operator+=(const DescriptorAllocationStatistics& other) noexcept
        {
            liveCount += other.liveCount;
            reservedCount += other.reservedCount;
            return *this;
        }
    };

    struct CpuDescriptorPage;

    class CpuDescriptorLease final
    {
    public:
        CpuDescriptorLease(std::shared_ptr<CpuDescriptorPage> page, std::uint32_t index);
        ~CpuDescriptorLease();

        CpuDescriptorLease(const CpuDescriptorLease&) = delete;
        CpuDescriptorLease& operator=(const CpuDescriptorLease&) = delete;
        CpuDescriptorLease(CpuDescriptorLease&&) = delete;
        CpuDescriptorLease& operator=(CpuDescriptorLease&&) = delete;

        [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE handle() const noexcept;

    private:
        std::shared_ptr<CpuDescriptorPage> page_;
        std::uint32_t index_ = 0;
    };

    class CpuDescriptorPool final
    {
    public:
        explicit CpuDescriptorPool(D3D12_DESCRIPTOR_HEAP_TYPE type);
        ~CpuDescriptorPool();

        CpuDescriptorPool(const CpuDescriptorPool&) = delete;
        CpuDescriptorPool& operator=(const CpuDescriptorPool&) = delete;
        CpuDescriptorPool(CpuDescriptorPool&&) = delete;
        CpuDescriptorPool& operator=(CpuDescriptorPool&&) = delete;

        [[nodiscard]] Result<std::shared_ptr<CpuDescriptorLease>> allocate(ID3D12Device& device);
        [[nodiscard]] DescriptorAllocationStatistics statistics() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    class GpuDescriptorLease final
    {
    public:
        ~GpuDescriptorLease();

        GpuDescriptorLease(const GpuDescriptorLease&) = delete;
        GpuDescriptorLease& operator=(const GpuDescriptorLease&) = delete;
        GpuDescriptorLease(GpuDescriptorLease&&) = delete;
        GpuDescriptorLease& operator=(GpuDescriptorLease&&) = delete;

        [[nodiscard]] std::uint32_t count() const noexcept;
        [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle(std::uint32_t index = 0) const noexcept;
        [[nodiscard]] D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle(std::uint32_t index = 0) const noexcept;

    private:
        struct State;
        friend class GpuDescriptorHeap;

        GpuDescriptorLease(std::shared_ptr<State> state, std::uint32_t offset, std::uint32_t count);

        std::shared_ptr<State> state_;
        std::uint32_t offset_ = 0;
        std::uint32_t count_ = 0;
    };

    class GpuDescriptorHeap final
    {
    public:
        GpuDescriptorHeap();
        ~GpuDescriptorHeap();

        GpuDescriptorHeap(const GpuDescriptorHeap&) = delete;
        GpuDescriptorHeap& operator=(const GpuDescriptorHeap&) = delete;
        GpuDescriptorHeap(GpuDescriptorHeap&&) = delete;
        GpuDescriptorHeap& operator=(GpuDescriptorHeap&&) = delete;

        [[nodiscard]] Status initialize(ID3D12Device& device,
                                        D3D12_DESCRIPTOR_HEAP_TYPE type,
                                        std::uint32_t capacity,
                                        std::uint32_t transientCapacity = 0,
                                        std::uint32_t bindTableCapacity = 0);
        [[nodiscard]] Result<std::shared_ptr<GpuDescriptorLease>> allocate(std::uint32_t count);
        [[nodiscard]] Result<std::shared_ptr<GpuDescriptorLease>> allocateTransient(std::uint32_t count);
        [[nodiscard]] Result<std::shared_ptr<GpuDescriptorLease>> allocateBindTablePage(std::uint32_t count);
        [[nodiscard]] ID3D12DescriptorHeap* nativeHeap() const noexcept;
        [[nodiscard]] DescriptorAllocationStatistics statistics() const noexcept;

    private:
        [[nodiscard]] static Result<std::shared_ptr<GpuDescriptorLease>> allocateFrom(
            const std::shared_ptr<GpuDescriptorLease::State>& state, std::uint32_t count, const char* exhaustedMessage);

        std::shared_ptr<GpuDescriptorLease::State> state_;
        std::shared_ptr<GpuDescriptorLease::State> transientState_;
        std::shared_ptr<GpuDescriptorLease::State> bindTableState_;
    };

} // namespace rachis::d3d12
