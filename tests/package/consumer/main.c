#include <Rachis/rachis.h>

int main(void)
{
    rachis_device_desc desc = {0};
    rachis_error error = {0};
    rachis_device* device = 0;
    desc.struct_size = sizeof(desc);
    desc.abi_version = RACHIS_ABI_VERSION;
    desc.backend = RACHIS_PACKAGE_BACKEND;
    desc.adapter_preference = RACHIS_PACKAGE_ADAPTER;
    error.struct_size = sizeof(error);
    if (rachis_abi_version() != RACHIS_ABI_VERSION)
    {
        return 1;
    }
    const rachis_status status = rachis_device_create(&desc, &device, &error);

    if (status == RACHIS_STATUS_BACKEND_UNAVAILABLE)
    {
        return 0;
    }
    if (status != RACHIS_STATUS_OK)
    {
        return 1;
    }
    rachis_device_destroy(device);
    return 0;
}
