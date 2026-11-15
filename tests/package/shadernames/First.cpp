#include "FillShader.h"
#include "Libraries.h"

std::span<const std::byte> firstFillShader(rachis::Backend backend)
{
    return rachis_shaders::FillShader(backend);
}
