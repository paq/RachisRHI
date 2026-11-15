#include "D3D12/D3D12Sampler.h"
#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace
{

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

} // namespace

TEST_CASE("immutable samplers validate state and recycle descriptors", "[size:medium][rhi][sampler][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto nearest = device.createSampler(rachis::SamplerDesc{
        .label = "Nearest sampler",
    });
    auto anisotropic = device.createSampler(rachis::SamplerDesc{
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .mipFilter = rachis::MipmapFilterMode::linear,
        .addressU = rachis::AddressMode::repeat,
        .addressV = rachis::AddressMode::mirrorRepeat,
        .addressW = rachis::AddressMode::clampToBorder,
        .lodBias = 0.25F,
        .maxAnisotropy = 16,
        .minLod = 0.0F,
        .maxLod = 12.0F,
        .borderColor = rachis::BorderColor::opaqueWhite,
        .label = "Anisotropic sampler",
    });
    auto comparison = device.createSampler(rachis::SamplerDesc{
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .mipFilter = rachis::MipmapFilterMode::nearest,
        .addressU = rachis::AddressMode::clampToBorder,
        .addressV = rachis::AddressMode::clampToBorder,
        .compareEnabled = true,
        .compare = rachis::CompareOp::lessEqual,
        .maxLod = 4.0F,
        .borderColor = rachis::BorderColor::opaqueBlack,
        .label = "Comparison sampler",
    });
    REQUIRE(nearest.has_value());
    REQUIRE(anisotropic.has_value());
    REQUIRE(comparison.has_value());

    CHECK_FALSE(device
                    .createSampler(rachis::SamplerDesc{
                        .maxAnisotropy = 0,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createSampler(rachis::SamplerDesc{
                        .maxAnisotropy = 17,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createSampler(rachis::SamplerDesc{
                        .lodBias = std::numeric_limits<float>::quiet_NaN(),
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createSampler(rachis::SamplerDesc{
                        .minLod = 2.0F,
                        .maxLod = 1.0F,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createSampler(rachis::SamplerDesc{
                        .mipFilter = rachis::MipmapFilterMode::none,
                        .minLod = 1.0F,
                    })
                    .has_value());

    std::vector<rachis::SamplerHandle> pagedSamplers;
    pagedSamplers.reserve(300);
    for (std::uint32_t index = 0; index < 300; ++index)
    {
        auto sampler = device.createSampler(rachis::SamplerDesc{
            .minFilter = index % 2 == 0 ? rachis::FilterMode::nearest : rachis::FilterMode::linear,
            .magFilter = index % 3 == 0 ? rachis::FilterMode::linear : rachis::FilterMode::nearest,
            .mipFilter = rachis::MipmapFilterMode::linear,
            .addressU = rachis::AddressMode::repeat,
        });
        REQUIRE(sampler.has_value());
        pagedSamplers.push_back(*sampler);
    }
    for (const auto sampler : pagedSamplers)
    {
        REQUIRE(device.destroySampler(sampler).has_value());
    }

    const auto staleNearest = *nearest;
    REQUIRE(device.destroySampler(*nearest).has_value());
    CHECK_FALSE(device.destroySampler(staleNearest).has_value());
    REQUIRE(device.destroySampler(*anisotropic).has_value());
    REQUIRE(device.destroySampler(*comparison).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("D3D12 anisotropic samplers encode mipFilter nearest as a point mip filter",
          "[size:small][rhi][d3d12][sampler]")
{
    using rachis::FilterMode;
    using rachis::MipmapFilterMode;
    using rachis::SamplerReductionMode;
    using rachis::d3d12::nativeSamplerFilter;

    const rachis::SamplerDesc pointMip{
        .minFilter = FilterMode::linear,
        .magFilter = FilterMode::linear,
        .mipFilter = MipmapFilterMode::nearest,
        .maxAnisotropy = 16,
    };
    auto comparisonDesc = pointMip;
    comparisonDesc.compareEnabled = true;
    comparisonDesc.compare = rachis::CompareOp::lessEqual;
    auto minimumDesc = pointMip;
    minimumDesc.reductionMode = SamplerReductionMode::minimum;
    auto maximumDesc = pointMip;
    maximumDesc.reductionMode = SamplerReductionMode::maximum;
    auto nearestDesc = pointMip;
    nearestDesc.minFilter = FilterMode::nearest;
    nearestDesc.magFilter = FilterMode::nearest;

    const auto comparison = nativeSamplerFilter(comparisonDesc, true);
    REQUIRE(comparison.has_value());
    CHECK(*comparison == D3D12_FILTER_COMPARISON_MIN_MAG_ANISOTROPIC_MIP_POINT);
    const auto standard = nativeSamplerFilter(pointMip, true);
    REQUIRE(standard.has_value());
    CHECK(*standard == D3D12_FILTER_MIN_MAG_ANISOTROPIC_MIP_POINT);
    const auto minimum = nativeSamplerFilter(minimumDesc, true);
    REQUIRE(minimum.has_value());
    CHECK(*minimum == D3D12_FILTER_MINIMUM_MIN_MAG_ANISOTROPIC_MIP_POINT);
    const auto maximum = nativeSamplerFilter(maximumDesc, true);
    REQUIRE(maximum.has_value());
    CHECK(*maximum == D3D12_FILTER_MAXIMUM_MIN_MAG_ANISOTROPIC_MIP_POINT);
    const auto nearest = nativeSamplerFilter(nearestDesc, true);
    REQUIRE(nearest.has_value());
    CHECK(*nearest == D3D12_FILTER_MIN_MAG_ANISOTROPIC_MIP_POINT);

    const auto unsupported = nativeSamplerFilter(comparisonDesc, false);
    REQUIRE_FALSE(unsupported.has_value());
    CHECK(unsupported.error().code == rachis::ErrorCode::unsupported);

    auto linearMipDesc = comparisonDesc;
    linearMipDesc.mipFilter = MipmapFilterMode::linear;
    const auto linearMip = nativeSamplerFilter(linearMipDesc, false);
    REQUIRE(linearMip.has_value());
    CHECK(*linearMip == D3D12_FILTER_COMPARISON_ANISOTROPIC);
    auto noMipDesc = pointMip;
    noMipDesc.mipFilter = MipmapFilterMode::none;
    const auto noMip = nativeSamplerFilter(noMipDesc, false);
    REQUIRE(noMip.has_value());
    CHECK(*noMip == D3D12_FILTER_ANISOTROPIC);

    auto isotropicDesc = pointMip;
    isotropicDesc.magFilter = FilterMode::nearest;
    isotropicDesc.maxAnisotropy = 1;
    const auto isotropic = nativeSamplerFilter(isotropicDesc, false);
    REQUIRE(isotropic.has_value());
    CHECK(*isotropic == D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT);
}
