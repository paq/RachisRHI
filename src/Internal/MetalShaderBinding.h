#pragma once

#include <cstdint>

namespace rachis
{

    inline constexpr std::uint32_t metalConstantBufferIndex = 0;
    inline constexpr std::uint32_t metalStorageBufferIndex = 1;
    inline constexpr std::uint32_t metalNestedScalarOffset = 0;
    inline constexpr std::uint32_t metalNestedFloat3Offset = 16;
    inline constexpr std::uint32_t metalNonSquareMatrixOffset = 32;
    inline constexpr std::uint32_t metalArrayOffset = 64;
    inline constexpr std::uint32_t metalMarkerOffset = 72;
    inline constexpr std::uint32_t metalConstantBufferSize = 80;

} // namespace rachis
