#pragma once

#include "Rachis/Resource.h"

#include <type_traits>

namespace rachis::detail
{

    inline constexpr FormatFeature allTextureDimensionFeatures =
        FormatFeature::texture1D | FormatFeature::texture2D | FormatFeature::texture3D | FormatFeature::cube;

    inline constexpr FormatFeature depthTextureDimensionFeatures =
        FormatFeature::texture1D | FormatFeature::texture2D | FormatFeature::cube;

    [[nodiscard]] constexpr FormatFeature textureDimensionFeature(TextureDimension dimension) noexcept
    {
        switch (dimension)
        {
        case TextureDimension::texture1D:
            return FormatFeature::texture1D;
        case TextureDimension::texture2D:
            return FormatFeature::texture2D;
        case TextureDimension::texture3D:
            return FormatFeature::texture3D;
        }
        return FormatFeature::none;
    }

    [[nodiscard]] constexpr bool depthTextureSupportsDimension(TextureDimension dimension) noexcept
    {
        return hasAny(depthTextureDimensionFeatures, textureDimensionFeature(dimension));
    }

    [[nodiscard]] constexpr FormatFeature depthTextureFormatFeatures(FormatFeature features) noexcept
    {
        using Bits = std::underlying_type_t<FormatFeature>;
        const auto otherFeatures = static_cast<Bits>(features) & ~static_cast<Bits>(allTextureDimensionFeatures);
        return static_cast<FormatFeature>(otherFeatures) | (features & depthTextureDimensionFeatures);
    }

} // namespace rachis::detail
