#pragma once

#include "Rachis/Handle.h"

#include <cstdint>
#include <limits>
#include <string_view>

namespace rachis
{

    enum class FilterMode : std::uint8_t
    {
        nearest,
        linear,
    };

    enum class MipmapFilterMode : std::uint8_t
    {
        none,
        nearest,
        linear,
    };

    enum class AddressMode : std::uint8_t
    {
        repeat,
        mirrorRepeat,
        clampToEdge,
        clampToBorder,
    };

    enum class CompareOp : std::uint8_t
    {
        never,
        less,
        equal,
        lessEqual,
        greater,
        notEqual,
        greaterEqual,
        always,
    };

    enum class BorderColor : std::uint8_t
    {
        transparentBlack,
        opaqueBlack,
        opaqueWhite,
    };

    enum class SamplerReductionMode : std::uint8_t
    {
        standard,
        minimum,
        maximum,
    };

    struct SamplerDesc final
    {
        FilterMode minFilter = FilterMode::nearest;
        FilterMode magFilter = FilterMode::nearest;
        MipmapFilterMode mipFilter = MipmapFilterMode::none;
        AddressMode addressU = AddressMode::clampToEdge;
        AddressMode addressV = AddressMode::clampToEdge;
        AddressMode addressW = AddressMode::clampToEdge;
        float lodBias = 0.0F;
        std::uint32_t maxAnisotropy = 1;
        bool compareEnabled = false;
        CompareOp compare = CompareOp::always;
        float minLod = 0.0F;
        float maxLod = std::numeric_limits<float>::max();
        BorderColor borderColor = BorderColor::transparentBlack;
        SamplerReductionMode reductionMode = SamplerReductionMode::standard;
        std::string_view label = {};
    };

} // namespace rachis
