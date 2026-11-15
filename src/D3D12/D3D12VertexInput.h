#pragma once

#include "Rachis/Pipeline.h"

#include <dxgiformat.h>

namespace rachis::d3d12
{

    [[nodiscard]] constexpr DXGI_FORMAT nativeVertexFormat(VertexFormat format) noexcept
    {
        switch (format)
        {
        case VertexFormat::float32:
            return DXGI_FORMAT_R32_FLOAT;
        case VertexFormat::float32x2:
            return DXGI_FORMAT_R32G32_FLOAT;
        case VertexFormat::float32x3:
            return DXGI_FORMAT_R32G32B32_FLOAT;
        case VertexFormat::float32x4:
            return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case VertexFormat::uint32:
            return DXGI_FORMAT_R32_UINT;
        case VertexFormat::uint32x2:
            return DXGI_FORMAT_R32G32_UINT;
        case VertexFormat::uint32x3:
            return DXGI_FORMAT_R32G32B32_UINT;
        case VertexFormat::uint32x4:
            return DXGI_FORMAT_R32G32B32A32_UINT;
        case VertexFormat::sint32:
            return DXGI_FORMAT_R32_SINT;
        case VertexFormat::sint32x2:
            return DXGI_FORMAT_R32G32_SINT;
        case VertexFormat::sint32x3:
            return DXGI_FORMAT_R32G32B32_SINT;
        case VertexFormat::sint32x4:
            return DXGI_FORMAT_R32G32B32A32_SINT;
        case VertexFormat::uint8x2:
            return DXGI_FORMAT_R8G8_UINT;
        case VertexFormat::uint8x4:
            return DXGI_FORMAT_R8G8B8A8_UINT;
        case VertexFormat::sint8x2:
            return DXGI_FORMAT_R8G8_SINT;
        case VertexFormat::sint8x4:
            return DXGI_FORMAT_R8G8B8A8_SINT;
        case VertexFormat::unorm8x2:
            return DXGI_FORMAT_R8G8_UNORM;
        case VertexFormat::unorm8x4:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case VertexFormat::snorm8x2:
            return DXGI_FORMAT_R8G8_SNORM;
        case VertexFormat::snorm8x4:
            return DXGI_FORMAT_R8G8B8A8_SNORM;
        case VertexFormat::uint16x2:
            return DXGI_FORMAT_R16G16_UINT;
        case VertexFormat::uint16x4:
            return DXGI_FORMAT_R16G16B16A16_UINT;
        case VertexFormat::sint16x2:
            return DXGI_FORMAT_R16G16_SINT;
        case VertexFormat::sint16x4:
            return DXGI_FORMAT_R16G16B16A16_SINT;
        case VertexFormat::unorm16x2:
            return DXGI_FORMAT_R16G16_UNORM;
        case VertexFormat::unorm16x4:
            return DXGI_FORMAT_R16G16B16A16_UNORM;
        case VertexFormat::snorm16x2:
            return DXGI_FORMAT_R16G16_SNORM;
        case VertexFormat::snorm16x4:
            return DXGI_FORMAT_R16G16B16A16_SNORM;
        case VertexFormat::float16x2:
            return DXGI_FORMAT_R16G16_FLOAT;
        case VertexFormat::float16x4:
            return DXGI_FORMAT_R16G16B16A16_FLOAT;
        }
        return DXGI_FORMAT_UNKNOWN;
    }

} // namespace rachis::d3d12
