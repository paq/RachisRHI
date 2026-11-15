#include "D3D12/D3D12Sampler.h"

#include "Common/HandleTable.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

namespace rachis::d3d12
{
    namespace
    {

        [[nodiscard]] constexpr D3D12_FILTER_TYPE nativeFilterType(FilterMode filter) noexcept
        {
            return filter == FilterMode::linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
        }

        [[nodiscard]] constexpr D3D12_FILTER_TYPE nativeMipFilterType(MipmapFilterMode filter) noexcept
        {
            return filter == MipmapFilterMode::linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
        }

        [[nodiscard]] constexpr D3D12_FILTER_REDUCTION_TYPE nativeReductionType(const SamplerDesc& desc) noexcept
        {
            if (desc.compareEnabled)
            {
                return D3D12_FILTER_REDUCTION_TYPE_COMPARISON;
            }
            switch (desc.reductionMode)
            {
            case SamplerReductionMode::minimum:
                return D3D12_FILTER_REDUCTION_TYPE_MINIMUM;
            case SamplerReductionMode::maximum:
                return D3D12_FILTER_REDUCTION_TYPE_MAXIMUM;
            default:
                return D3D12_FILTER_REDUCTION_TYPE_STANDARD;
            }
        }

        [[nodiscard]] constexpr float nativeMaxLod(const SamplerDesc& desc) noexcept
        {
            constexpr float unmipmappedMaxLod = 0.25F;
            if (desc.mipFilter != MipmapFilterMode::none)
            {
                return desc.maxLod;
            }
            if (desc.maxAnisotropy > 1)
            {
                return 0.0F;
            }
            return std::min(desc.maxLod, unmipmappedMaxLod);
        }

        [[nodiscard]] constexpr D3D12_TEXTURE_ADDRESS_MODE nativeAddressMode(AddressMode mode) noexcept
        {
            switch (mode)
            {
            case AddressMode::repeat:
                return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            case AddressMode::mirrorRepeat:
                return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            case AddressMode::clampToBorder:
                return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            default:
                return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            }
        }

        [[nodiscard]] constexpr D3D12_COMPARISON_FUNC nativeCompare(CompareOp compare) noexcept
        {
            switch (compare)
            {
            case CompareOp::never:
                return D3D12_COMPARISON_FUNC_NEVER;
            case CompareOp::less:
                return D3D12_COMPARISON_FUNC_LESS;
            case CompareOp::equal:
                return D3D12_COMPARISON_FUNC_EQUAL;
            case CompareOp::lessEqual:
                return D3D12_COMPARISON_FUNC_LESS_EQUAL;
            case CompareOp::greater:
                return D3D12_COMPARISON_FUNC_GREATER;
            case CompareOp::notEqual:
                return D3D12_COMPARISON_FUNC_NOT_EQUAL;
            case CompareOp::greaterEqual:
                return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
            default:
                return D3D12_COMPARISON_FUNC_ALWAYS;
            }
        }

        void setBorderColor(BorderColor color, float (&destination)[4]) noexcept
        {
            const float alpha = color == BorderColor::transparentBlack ? 0.0F : 1.0F;
            const float value = color == BorderColor::opaqueWhite ? 1.0F : 0.0F;
            destination[0] = value;
            destination[1] = value;
            destination[2] = value;
            destination[3] = alpha;
        }

        [[nodiscard]] Status validateSamplerDesc(const SamplerDesc& desc)
        {
            if (desc.reductionMode > SamplerReductionMode::maximum ||
                (desc.compareEnabled && desc.reductionMode != SamplerReductionMode::standard))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Sampler comparison and minimum/maximum reduction modes are mutually exclusive.",
                });
            }
            if (!std::isfinite(desc.lodBias) || !std::isfinite(desc.minLod) || !std::isfinite(desc.maxLod) ||
                std::fabs(desc.lodBias) > D3D12_MIP_LOD_BIAS_MAX || desc.minLod < 0.0F || desc.maxLod < desc.minLod ||
                (desc.mipFilter == MipmapFilterMode::none && desc.minLod != 0.0F))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Sampler LOD values must be finite, ordered, and within D3D12 limits.",
                });
            }
            if (desc.maxAnisotropy == 0 || desc.maxAnisotropy > D3D12_MAX_MAXANISOTROPY)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Sampler anisotropy must be between 1 and 16.",
                });
            }
            return {};
        }

    } // namespace

    Result<D3D12_FILTER> nativeSamplerFilter(const SamplerDesc& desc, bool anisotropicPointMipSupported)
    {
        const auto reduction = nativeReductionType(desc);
        if (desc.maxAnisotropy <= 1)
        {
            return D3D12_ENCODE_BASIC_FILTER(nativeFilterType(desc.minFilter),
                                             nativeFilterType(desc.magFilter),
                                             nativeMipFilterType(desc.mipFilter),
                                             reduction);
        }
        if (desc.mipFilter != MipmapFilterMode::nearest)
        {
            return D3D12_ENCODE_ANISOTROPIC_FILTER(reduction);
        }
        if (!anisotropicPointMipSupported)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Anisotropic samplers with mipFilter nearest require D3D12 AnisoFilterWithPointMipSupported.",
            });
        }
        return D3D12_ENCODE_MIN_MAG_ANISOTROPIC_MIP_POINT_FILTER(reduction);
    }

    struct SamplerManager::Impl final
    {
        ID3D12Device* device = nullptr;
        bool anisotropicPointMipSupported = false;
        std::mutex mutex;
        CpuDescriptorPool descriptors{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER};
        detail::HandleTable<std::shared_ptr<CpuDescriptorLease>, SamplerHandle> samplers;
    };

    SamplerManager::SamplerManager() : impl_(std::make_unique<Impl>())
    {
    }

    SamplerManager::~SamplerManager() = default;

    Status SamplerManager::initialize(ID3D12Device& device)
    {
        impl_->device = &device;
        D3D12_FEATURE_DATA_D3D12_OPTIONS19 options19{};
        impl_->anisotropicPointMipSupported =
            SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &options19, sizeof(options19))) &&
            options19.AnisoFilterWithPointMipSupported != FALSE;
        return {};
    }

    DescriptorAllocationStatistics SamplerManager::descriptorAllocationStatistics() const noexcept
    {
        return impl_->descriptors.statistics();
    }

    Result<SamplerHandle> SamplerManager::createSampler(const SamplerDesc& desc)
    {
        auto valid = validateSamplerDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        auto filter = nativeSamplerFilter(desc, impl_->anisotropicPointMipSupported);
        if (!filter)
        {
            return std::unexpected(std::move(filter.error()));
        }

        auto allocation = impl_->descriptors.allocate(*impl_->device);
        if (!allocation)
        {
            return std::unexpected(std::move(allocation.error()));
        }

        D3D12_SAMPLER_DESC native{
            .Filter = *filter,
            .AddressU = nativeAddressMode(desc.addressU),
            .AddressV = nativeAddressMode(desc.addressV),
            .AddressW = nativeAddressMode(desc.addressW),
            .MipLODBias = desc.lodBias,
            .MaxAnisotropy = desc.maxAnisotropy,
            .ComparisonFunc = desc.compareEnabled ? nativeCompare(desc.compare) : D3D12_COMPARISON_FUNC_NEVER,
            .BorderColor = {},
            .MinLOD = desc.minLod,
            .MaxLOD = nativeMaxLod(desc),
        };
        setBorderColor(desc.borderColor, native.BorderColor);
        impl_->device->CreateSampler(&native, (*allocation)->handle());

        const std::scoped_lock lock(impl_->mutex);
        return impl_->samplers.emplace(std::move(*allocation));
    }

    Status SamplerManager::destroySampler(SamplerHandle sampler)
    {
        const std::scoped_lock lock(impl_->mutex);
        if (!impl_->samplers.erase(sampler))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The sampler handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<NativeSampler> SamplerManager::nativeSampler(SamplerHandle sampler)
    {
        const std::scoped_lock lock(impl_->mutex);
        const auto* stored = impl_->samplers.find(sampler);
        if (stored == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The sampler handle is invalid or stale.",
            });
        }
        return NativeSampler{
            .owner = *stored,
            .descriptor = (*stored)->handle(),
        };
    }

} // namespace rachis::d3d12
