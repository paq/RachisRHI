#include "Fill.h"

#include <cstdio>

int main()
{
    std::printf("%zu\n", rachis_shaders::Fill(rachis::Backend::vulkan).size());
    return 0;
}
