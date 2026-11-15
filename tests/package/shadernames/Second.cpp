#include "FillShader.h"
#include "Libraries.h"

std::span<const std::byte> secondFillShader(rachis::Backend backend)
{
    return rachis_shaders::FillShader(backend);
}
