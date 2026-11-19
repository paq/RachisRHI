#pragma once

#include "D3D12/D3D12Descriptor.h"
#include "D3D12/D3D12Resource.h"
#include "D3D12/D3D12RootSignaturePolicy.h"
#include "D3D12/D3D12RuntimePolicies.h"
#include "D3D12/D3D12Sampler.h"
#include "D3D12/D3D12View.h"
#include "Rachis/Binding.h"

#include <d3d12.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rachis::d3d12
{

    inline constexpr std::uint32_t utilityResourceDescriptorCapacity = 65'536;
    inline constexpr std::uint32_t dynamicTableDescriptorCapacity = 65'536;
    inline constexpr std::uint32_t maxDynamicTablePageDescriptors = 256;
    inline constexpr std::uint32_t bindingPoolResourceDescriptorBudget =
        D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1 - utilityResourceDescriptorCapacity -
        dynamicTableDescriptorCapacity;

    static_assert(maxDynamicBufferOffsetsPerBindingSet <= maxDynamicTablePageDescriptors);
    inline constexpr std::uint32_t bindingSamplerDescriptorBudget = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;

    struct NativeBindingLayoutEntry final
    {
        BindingLayoutEntry entry;
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        std::uint32_t descriptorOffset = 0;
    };

    struct NativeBindingLayout final
    {
        std::shared_ptr<void> owner = {};
        std::vector<NativeBindingLayoutEntry> entries = {};
        std::uint32_t resourceDescriptorCount = 0;
        std::uint32_t samplerDescriptorCount = 0;
        std::uint32_t dynamicBufferCount = 0;
        bool hasVariableDescriptorCount = false;
    };

    struct NativeRootBufferBinding final
    {
        std::uint32_t rootParameter = unpromotedRootParameter;
        RootDescriptorKind kind = RootDescriptorKind::cbv;
        D3D12_GPU_VIRTUAL_ADDRESS gpuVirtualAddress = 0;
    };

    struct NativeBindingSet final
    {
        std::shared_ptr<void> owner;
        std::shared_ptr<void> bindingLayoutOwner;
        BindingPoolCommandRetention bindingPoolRetention;
        std::shared_ptr<GpuDescriptorLease> resourceDescriptorOwner;
        std::shared_ptr<GpuDescriptorLease> samplerDescriptorOwner;
        std::span<const std::shared_ptr<ResidencyRecord>> staticResidencies;
        std::shared_ptr<const void> staticResidencyOwner;
        SmallVector<std::shared_ptr<ResidencyRecord>, 8> extraResidencies;
        InPlacePublicationReference inPlacePublication;
        SmallVector<std::shared_ptr<SwapchainImageLeaseState>, 4> swapchainImageLeases;
        SmallVector<std::shared_ptr<void>, 4> sparseMappingOwners;
        SmallVector<TimelinePoint, 4> sparseMappingDependencies;
        SmallVector<std::shared_ptr<void>, 4> accelerationStructureOwners;
        SmallVector<AccelerationStructureExecutionLease, 4> accelerationStructureLeases;
        std::array<NativeRootBufferBinding, maxDynamicBufferOffsetsPerBindingSet> rootBuffers{};
        std::uint32_t rootBufferCount = 0;
        D3D12_GPU_DESCRIPTOR_HANDLE resources{};
        D3D12_GPU_DESCRIPTOR_HANDLE samplers{};
        D3D12_GPU_DESCRIPTOR_HANDLE dynamicResources{};
        std::uint32_t resourceDescriptorCount = 0;
        std::uint32_t samplerDescriptorCount = 0;
        std::uint32_t dynamicDescriptorCount = 0;
    };

    class DynamicDescriptorArena final
    {
    public:
        struct Impl;

        DynamicDescriptorArena();
        ~DynamicDescriptorArena();

        DynamicDescriptorArena(const DynamicDescriptorArena&) = delete;
        DynamicDescriptorArena& operator=(const DynamicDescriptorArena&) = delete;
        DynamicDescriptorArena(DynamicDescriptorArena&&) = delete;
        DynamicDescriptorArena& operator=(DynamicDescriptorArena&&) = delete;

        [[nodiscard]] Impl& impl() noexcept;

    private:
        std::unique_ptr<Impl> impl_;
    };

    struct NativeDescriptorHeaps final
    {
        ID3D12DescriptorHeap* resources = nullptr;
        ID3D12DescriptorHeap* samplers = nullptr;
    };

    struct ClearResourceDescriptorPair final
    {
        std::shared_ptr<GpuDescriptorLease> shaderVisible;
        std::shared_ptr<CpuDescriptorLease> cpuVisible;
    };

    class BindingManager final
    {
    public:
        BindingManager();
        ~BindingManager();

        BindingManager(const BindingManager&) = delete;
        BindingManager& operator=(const BindingManager&) = delete;
        BindingManager(BindingManager&&) = delete;
        BindingManager& operator=(BindingManager&&) = delete;

        [[nodiscard]] Status initialize(ID3D12Device& device,
                                        bool variableDescriptorCountSupported,
                                        bool rayTracingSupported,
                                        bool samplerFeedbackSupported);
        [[nodiscard]] Result<BindingLayoutHandle> createBindingLayout(const BindingLayoutDesc& desc);
        [[nodiscard]] Status destroyBindingLayout(BindingLayoutHandle layout);
        [[nodiscard]] Result<BindingPoolHandle> createBindingPool(const BindingPoolDesc& desc);
        [[nodiscard]] Status destroyBindingPool(BindingPoolHandle pool);
        [[nodiscard]] Result<BindingSetHandle> createBindingSet(ResourceManager& resources,
                                                                ViewManager& views,
                                                                SamplerManager& samplers,
                                                                const BindingSetDesc& desc);
        [[nodiscard]] Status updateBindingSet(ResourceManager& resources,
                                              ViewManager& views,
                                              SamplerManager& samplers,
                                              BindingSetHandle set,
                                              std::span<const BindingWrite> writes);
        [[nodiscard]] Status destroyBindingSet(BindingSetHandle set);
        [[nodiscard]] Result<NativeBindingLayout> nativeBindingLayout(BindingLayoutHandle layout);
        [[nodiscard]] Result<NativeBindingSet> nativeBindingSet(ResourceManager& resources,
                                                                BindingSetHandle set,
                                                                std::span<const std::uint32_t> dynamicOffsets = {},
                                                                const RootSignatureSlotPlan* slotPlan = nullptr,
                                                                const void* expectedBindingLayoutOwner = nullptr,
                                                                std::uint64_t recordingId = 0,
                                                                DynamicDescriptorArena* dynamicArena = nullptr);
        [[nodiscard]] Result<std::shared_ptr<GpuDescriptorLease>> allocateResourceDescriptors(std::uint32_t count);
        [[nodiscard]] Result<ClearResourceDescriptorPair> allocateClearResourceDescriptors();
        [[nodiscard]] BindingUpdateStatistics statistics() const noexcept;
        [[nodiscard]] DescriptorAllocationStatistics descriptorAllocationStatistics() const noexcept;
        [[nodiscard]] NativeDescriptorHeaps nativeDescriptorHeaps() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rachis::d3d12
