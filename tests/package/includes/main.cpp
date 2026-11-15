#include "FillShader.h"

#include <cstdio>

int main()
{
    std::printf("%zu\n", rachis_shaders::FillShader(rachis::Backend::vulkan).size());
    return 0;
}
