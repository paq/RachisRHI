#include "FillShader.h"

std::span<const std::byte> fillShader(rachis::Backend backend)
{
    return rachis_shaders::FillShader(backend);
}
