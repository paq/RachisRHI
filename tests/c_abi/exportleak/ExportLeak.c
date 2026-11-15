#include "ExportLeak.h"

int rachis_fixture_version(void)
{
    return RACHIS_ABI_VERSION;
}

RACHIS_API int fixture_leaked_function(void)
{
    return 2;
}

RACHIS_API const int rachis_fixture_leaked_constant = 3;

#if !defined(_WIN32)
__attribute__((weak, visibility("default"))) int rachis_fixture_leaked_weak(void)
{
    return 4;
}
#endif
