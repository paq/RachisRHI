#pragma once

#include "D3D12/D3D12Descriptor.h"
#include "Rachis/Sampler.h"

#include <d3d12.h>

#include <memory>

namespace rachis::d3d12
{

    struct NativeSampler final
    {
        std::shared_ptr<void> owner;
        D3D12_CPU_DESCRIPTOR_HANDLE descriptor{};
    };

    [[nodiscard]] Result<D3D12_FILTER> nativeSamplerFilter(const SamplerDesc& desc, bool anisotropicPointMipSupported);

    class SamplerManager final
    {
    public:
        SamplerManager();
        ~SamplerManager();

        SamplerManager(const SamplerManager&) = delete;
        SamplerManager& operator=(const SamplerManager&) = delete;
        SamplerManager(SamplerManager&&) = delete;
        SamplerManager& operator=(SamplerManager&&) = delete;

        [[nodiscard]] Status initialize(ID3D12Device& device);
        [[nodiscard]] DescriptorAllocationStatistics descriptorAllocationStatistics() const noexcept;
        [[nodiscard]] Result<SamplerHandle> createSampler(const SamplerDesc& desc);
        [[nodiscard]] Status destroySampler(SamplerHandle sampler);
        [[nodiscard]] Result<NativeSampler> nativeSampler(SamplerHandle sampler);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rachis::d3d12
