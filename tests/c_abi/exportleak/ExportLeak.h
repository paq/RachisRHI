#pragma once

#define RACHIS_ABI_VERSION 1

#if defined(_WIN32)
#define RACHIS_API __declspec(dllexport)
#else
#define RACHIS_API __attribute__((visibility("default")))
#endif

RACHIS_API int rachis_fixture_version(void);
