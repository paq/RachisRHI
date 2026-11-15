#pragma once

#include "Rachis/Resource.h"

#include <cstdint>

namespace rachis
{

    enum class MetalReadWriteTextureTier : std::uint8_t
    {
        none,
        tier1,
        tier2,
    };

    [[nodiscard]] constexpr bool metalReadWriteFormat(Format format, MetalReadWriteTextureTier tier) noexcept
    {
        switch (format)
        {
        case Format::r32Float:
        case Format::r32Uint:
        case Format::r32Sint:
            return tier == MetalReadWriteTextureTier::tier1 || tier == MetalReadWriteTextureTier::tier2;
        case Format::rgba32Float:
        case Format::rgba32Uint:
        case Format::rgba32Sint:
        case Format::rgba16Float:
        case Format::rgba16Uint:
        case Format::rgba16Sint:
        case Format::rgba8Unorm:
        case Format::rgba8Uint:
        case Format::rgba8Sint:
        case Format::r16Float:
        case Format::r16Uint:
        case Format::r16Sint:
        case Format::r8Unorm:
        case Format::r8Uint:
        case Format::r8Sint:
            return tier == MetalReadWriteTextureTier::tier2;
        default:
            return false;
        }
    }

    [[nodiscard]] constexpr Format metalLinearViewFormat(Format format) noexcept
    {
        switch (format)
        {
        case Format::rgba8UnormSrgb:
            return Format::rgba8Unorm;
        case Format::bgra8UnormSrgb:
            return Format::bgra8Unorm;
        default:
            return Format::unknown;
        }
    }

} // namespace rachis
