#include <Rachis/Device.h>
#include <cstdio>

int main()
{
    rachis::DeviceDesc desc{};
    desc.backend = static_cast<rachis::Backend>(RACHIS_PACKAGE_BACKEND);
    desc.adapterPreference = static_cast<rachis::AdapterPreference>(RACHIS_PACKAGE_ADAPTER);
    desc.validation = rachis::ValidationLevel::disabled;

    desc.enableDred = desc.backend == rachis::Backend::d3d12;
    const auto device = rachis::createDevice(desc);
    if (device)
    {
        return 0;
    }
    const auto& error = device.error();
    std::fprintf(stderr, "create device failed: %u (%s)\n", static_cast<unsigned>(error.code), error.message.c_str());

    return error.code == rachis::ErrorCode::backendUnavailable ? 0 : 1;
}
