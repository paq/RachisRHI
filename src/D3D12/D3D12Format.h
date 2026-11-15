#pragma once

#include "Rachis/Resource.h"

#include <d3d12.h>
#include <dxgiformat.h>

#include <cstdint>
#include <optional>

namespace rachis::d3d12::format
{

    struct Info final
    {
        DXGI_FORMAT resource = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT view = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT sampled = DXGI_FORMAT_UNKNOWN;
        std::uint32_t blockWidth = 1;
        std::uint32_t blockHeight = 1;
        std::uint32_t bytesPerBlock = 0;
        std::uint32_t planeCount = 1;
        bool depth = false;
        bool stencil = false;
        bool compressed = false;
    };

    [[nodiscard]] constexpr std::optional<Info> info(Format value) noexcept
    {
        switch (value)
        {
        case Format::r8Unorm:
            return Info{DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_UNORM, 1, 1, 1};
        case Format::rg8Unorm:
            return Info{DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8_UNORM, 1, 1, 2};
        case Format::rgba8Unorm:
            return Info{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1, 4};
        case Format::rgba8UnormSrgb:
            return Info{DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                        1,
                        1,
                        4};
        case Format::bgra8Unorm:
            return Info{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, 1, 1, 4};
        case Format::bgra8UnormSrgb:
            return Info{DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
                        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
                        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
                        1,
                        1,
                        4};
        case Format::r16Float:
            return Info{DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_FLOAT, 1, 1, 2};
        case Format::rg16Float:
            return Info{DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, 1, 1, 4};
        case Format::rgba16Float:
            return Info{DXGI_FORMAT_R16G16B16A16_FLOAT,
                        DXGI_FORMAT_R16G16B16A16_FLOAT,
                        DXGI_FORMAT_R16G16B16A16_FLOAT,
                        1,
                        1,
                        8};
        case Format::r32Float:
            return Info{DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_FLOAT, 1, 1, 4};
        case Format::rg32Float:
            return Info{DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, 1, 1, 8};
        case Format::rgba32Float:
            return Info{DXGI_FORMAT_R32G32B32A32_FLOAT,
                        DXGI_FORMAT_R32G32B32A32_FLOAT,
                        DXGI_FORMAT_R32G32B32A32_FLOAT,
                        1,
                        1,
                        16};
        case Format::d32Float:
            return Info{DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, 1, 1, 4, 1, true};
        case Format::d24UnormS8Uint:
            return Info{DXGI_FORMAT_R24G8_TYPELESS,
                        DXGI_FORMAT_D24_UNORM_S8_UINT,
                        DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                        1,
                        1,
                        4,
                        2,
                        true,
                        true};
        case Format::bc1RgbaUnorm:
            return Info{
                DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM, 4, 4, 8, 1, false, false, true};
        case Format::bc1RgbaUnormSrgb:
            return Info{DXGI_FORMAT_BC1_UNORM_SRGB,
                        DXGI_FORMAT_BC1_UNORM_SRGB,
                        DXGI_FORMAT_BC1_UNORM_SRGB,
                        4,
                        4,
                        8,
                        1,
                        false,
                        false,
                        true};
        case Format::bc2RgbaUnorm:
            return Info{
                DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC2_UNORM, 4, 4, 16, 1, false, false, true};
        case Format::bc2RgbaUnormSrgb:
            return Info{DXGI_FORMAT_BC2_UNORM_SRGB,
                        DXGI_FORMAT_BC2_UNORM_SRGB,
                        DXGI_FORMAT_BC2_UNORM_SRGB,
                        4,
                        4,
                        16,
                        1,
                        false,
                        false,
                        true};
        case Format::bc3RgbaUnorm:
            return Info{
                DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM, 4, 4, 16, 1, false, false, true};
        case Format::bc3RgbaUnormSrgb:
            return Info{DXGI_FORMAT_BC3_UNORM_SRGB,
                        DXGI_FORMAT_BC3_UNORM_SRGB,
                        DXGI_FORMAT_BC3_UNORM_SRGB,
                        4,
                        4,
                        16,
                        1,
                        false,
                        false,
                        true};
        case Format::bc4RUnorm:
            return Info{
                DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_UNORM, 4, 4, 8, 1, false, false, true};
        case Format::bc4RSnorm:
            return Info{
                DXGI_FORMAT_BC4_SNORM, DXGI_FORMAT_BC4_SNORM, DXGI_FORMAT_BC4_SNORM, 4, 4, 8, 1, false, false, true};
        case Format::bc5RgUnorm:
            return Info{
                DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC5_UNORM, 4, 4, 16, 1, false, false, true};
        case Format::bc5RgSnorm:
            return Info{
                DXGI_FORMAT_BC5_SNORM, DXGI_FORMAT_BC5_SNORM, DXGI_FORMAT_BC5_SNORM, 4, 4, 16, 1, false, false, true};
        case Format::bc6hRgbUfloat:
            return Info{
                DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC6H_UF16, 4, 4, 16, 1, false, false, true};
        case Format::bc6hRgbSfloat:
            return Info{
                DXGI_FORMAT_BC6H_SF16, DXGI_FORMAT_BC6H_SF16, DXGI_FORMAT_BC6H_SF16, 4, 4, 16, 1, false, false, true};
        case Format::bc7RgbaUnorm:
            return Info{
                DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM, 4, 4, 16, 1, false, false, true};
        case Format::bc7RgbaUnormSrgb:
            return Info{DXGI_FORMAT_BC7_UNORM_SRGB,
                        DXGI_FORMAT_BC7_UNORM_SRGB,
                        DXGI_FORMAT_BC7_UNORM_SRGB,
                        4,
                        4,
                        16,
                        1,
                        false,
                        false,
                        true};
        case Format::r8Uint:
            return Info{DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8_UINT, 1, 1, 1};
        case Format::rg8Uint:
            return Info{DXGI_FORMAT_R8G8_UINT, DXGI_FORMAT_R8G8_UINT, DXGI_FORMAT_R8G8_UINT, 1, 1, 2};
        case Format::rgba8Uint:
            return Info{DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R8G8B8A8_UINT, 1, 1, 4};
        case Format::r8Sint:
            return Info{DXGI_FORMAT_R8_SINT, DXGI_FORMAT_R8_SINT, DXGI_FORMAT_R8_SINT, 1, 1, 1};
        case Format::rg8Sint:
            return Info{DXGI_FORMAT_R8G8_SINT, DXGI_FORMAT_R8G8_SINT, DXGI_FORMAT_R8G8_SINT, 1, 1, 2};
        case Format::rgba8Sint:
            return Info{DXGI_FORMAT_R8G8B8A8_SINT, DXGI_FORMAT_R8G8B8A8_SINT, DXGI_FORMAT_R8G8B8A8_SINT, 1, 1, 4};
        case Format::r16Uint:
            return Info{DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16_UINT, 1, 1, 2};
        case Format::rg16Uint:
            return Info{DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_R16G16_UINT, 1, 1, 4};
        case Format::rgba16Uint:
            return Info{
                DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_UINT, 1, 1, 8};
        case Format::r16Sint:
            return Info{DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16_SINT, 1, 1, 2};
        case Format::rg16Sint:
            return Info{DXGI_FORMAT_R16G16_SINT, DXGI_FORMAT_R16G16_SINT, DXGI_FORMAT_R16G16_SINT, 1, 1, 4};
        case Format::rgba16Sint:
            return Info{
                DXGI_FORMAT_R16G16B16A16_SINT, DXGI_FORMAT_R16G16B16A16_SINT, DXGI_FORMAT_R16G16B16A16_SINT, 1, 1, 8};
        case Format::r32Uint:
            return Info{DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_UINT, 1, 1, 4};
        case Format::rg32Uint:
            return Info{DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_UINT, 1, 1, 8};
        case Format::rgba32Uint:
            return Info{
                DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_UINT, 1, 1, 16};
        case Format::r32Sint:
            return Info{DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32_SINT, 1, 1, 4};
        case Format::rg32Sint:
            return Info{DXGI_FORMAT_R32G32_SINT, DXGI_FORMAT_R32G32_SINT, DXGI_FORMAT_R32G32_SINT, 1, 1, 8};
        case Format::rgba32Sint:
            return Info{
                DXGI_FORMAT_R32G32B32A32_SINT, DXGI_FORMAT_R32G32B32A32_SINT, DXGI_FORMAT_R32G32B32A32_SINT, 1, 1, 16};
        case Format::r64Uint:
            return Info{DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_UINT, 1, 1, 8};
        case Format::unknown:
            return std::nullopt;
        }
        return std::nullopt;
    }

    struct Capabilities final
    {
        DXGI_FORMAT resource = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT sampled = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT storage = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT attachment = DXGI_FORMAT_UNKNOWN;
        bool depthStencil = false;
    };

    [[nodiscard]] constexpr Capabilities capabilities(Format value) noexcept
    {
        const auto metadata = info(value);
        if (!metadata)
        {
            return {};
        }
        if (metadata->depth)
        {
            return {metadata->resource, metadata->sampled, DXGI_FORMAT_UNKNOWN, metadata->view, true};
        }
        return {metadata->resource, metadata->sampled, metadata->view, metadata->view};
    }

    [[nodiscard]] constexpr bool supportsSampledUsage(D3D12_FORMAT_SUPPORT1 support) noexcept
    {
        return (support & (D3D12_FORMAT_SUPPORT1_SHADER_LOAD | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE)) != 0;
    }

    [[nodiscard]] inline SampleCountMask attachmentSampleCounts(ID3D12Device& device, DXGI_FORMAT format) noexcept
    {
        auto result = SampleCountMask::x1;
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{
            .Format = format,
            .Support1 = {},
            .Support2 = {},
        };
        if (format == DXGI_FORMAT_UNKNOWN ||
            FAILED(device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
            (support.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET) == 0)
        {
            return result;
        }
        const auto probe = [&](std::uint32_t count, SampleCountMask flag) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS levels{
                .Format = format,
                .SampleCount = count,
                .Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
                .NumQualityLevels = {},
            };
            if (SUCCEEDED(
                    device.CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &levels, sizeof(levels))) &&
                levels.NumQualityLevels > 0)
            {
                result |= flag;
            }
        };
        probe(2, SampleCountMask::x2);
        probe(4, SampleCountMask::x4);
        probe(8, SampleCountMask::x8);
        return result;
    }

    [[nodiscard]] constexpr DXGI_FORMAT view(Format value) noexcept
    {
        const auto metadata = info(value);
        return metadata ? metadata->view : DXGI_FORMAT_UNKNOWN;
    }

    [[nodiscard]] constexpr bool isColorTarget(Format value) noexcept
    {
        const auto metadata = info(value);
        return metadata && !metadata->depth && !metadata->compressed;
    }

    [[nodiscard]] constexpr Format alternateView(Format value) noexcept
    {
        switch (value)
        {
        case Format::rgba8Unorm:
            return Format::rgba8UnormSrgb;
        case Format::rgba8UnormSrgb:
            return Format::rgba8Unorm;
        case Format::bgra8Unorm:
            return Format::bgra8UnormSrgb;
        case Format::bgra8UnormSrgb:
            return Format::bgra8Unorm;
        case Format::bc1RgbaUnorm:
            return Format::bc1RgbaUnormSrgb;
        case Format::bc1RgbaUnormSrgb:
            return Format::bc1RgbaUnorm;
        case Format::bc2RgbaUnorm:
            return Format::bc2RgbaUnormSrgb;
        case Format::bc2RgbaUnormSrgb:
            return Format::bc2RgbaUnorm;
        case Format::bc3RgbaUnorm:
            return Format::bc3RgbaUnormSrgb;
        case Format::bc3RgbaUnormSrgb:
            return Format::bc3RgbaUnorm;
        case Format::bc7RgbaUnorm:
            return Format::bc7RgbaUnormSrgb;
        case Format::bc7RgbaUnormSrgb:
            return Format::bc7RgbaUnorm;
        default:
            return Format::unknown;
        }
    }

    [[nodiscard]] constexpr DXGI_FORMAT mutableResource(Format value) noexcept
    {
        switch (value)
        {
        case Format::rgba8Unorm:
        case Format::rgba8UnormSrgb:
            return DXGI_FORMAT_R8G8B8A8_TYPELESS;
        case Format::bgra8Unorm:
        case Format::bgra8UnormSrgb:
            return DXGI_FORMAT_B8G8R8A8_TYPELESS;
        case Format::bc1RgbaUnorm:
        case Format::bc1RgbaUnormSrgb:
            return DXGI_FORMAT_BC1_TYPELESS;
        case Format::bc2RgbaUnorm:
        case Format::bc2RgbaUnormSrgb:
            return DXGI_FORMAT_BC2_TYPELESS;
        case Format::bc3RgbaUnorm:
        case Format::bc3RgbaUnormSrgb:
            return DXGI_FORMAT_BC3_TYPELESS;
        case Format::bc7RgbaUnorm:
        case Format::bc7RgbaUnormSrgb:
            return DXGI_FORMAT_BC7_TYPELESS;
        default:
            return DXGI_FORMAT_UNKNOWN;
        }
    }

} // namespace rachis::d3d12::format
