#pragma once

#include <Rachis/Device.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <utility>

namespace example
{
    inline constexpr int skipExitCode = 77;

    struct Options final
    {
        rachis::DeviceDesc device = {};
        const char* output = nullptr;
    };

    [[nodiscard]] inline std::optional<Options> parseOptions(int argc, char** argv, std::string_view applicationName)
    {
        Options options;
        options.device.applicationName = std::string{applicationName};
        options.device.validation = rachis::ValidationLevel::standard;
        for (int index = 1; index < argc; ++index)
        {
            const std::string_view argument = argv[index];
            if (argument == "--backend" && index + 1 < argc)
            {
                const std::string_view name = argv[++index];
                if (name == "d3d12")
                {
                    options.device.backend = rachis::Backend::d3d12;
                }
                else if (name == "vulkan")
                {
                    options.device.backend = rachis::Backend::vulkan;
                }
                else if (name == "metal")
                {
                    options.device.backend = rachis::Backend::metal;
                }
                else
                {
                    std::fprintf(stderr, "Unknown backend '%.*s'.\n", static_cast<int>(name.size()), name.data());
                    return std::nullopt;
                }
            }
            else if (argument == "--warp")
            {
                options.device.adapterPreference = rachis::AdapterPreference::warp;
            }
            else if (argument == "--validation" && index + 1 < argc)
            {
                const std::string_view level = argv[++index];
                if (level == "disabled")
                {
                    options.device.validation = rachis::ValidationLevel::disabled;
                }
                else if (level == "standard")
                {
                    options.device.validation = rachis::ValidationLevel::standard;
                }
                else if (level == "gpu")
                {
                    options.device.validation = rachis::ValidationLevel::gpu;
                }
                else
                {
                    std::fprintf(
                        stderr, "Unknown validation level '%.*s'.\n", static_cast<int>(level.size()), level.data());
                    return std::nullopt;
                }
            }
            else if (argument == "--output" && index + 1 < argc)
            {
                options.output = argv[++index];
            }
            else
            {
                std::fprintf(stderr,
                             "Usage: %s [--backend d3d12|vulkan|metal] [--warp] [--validation disabled|standard|gpu] "
                             "[--output <path>]\n",
                             argc > 0 ? argv[0] : "example");
                return std::nullopt;
            }
        }
        return options;
    }

    [[nodiscard]] inline const char* backendName(rachis::Backend backend) noexcept
    {
        switch (backend)
        {
        case rachis::Backend::d3d12:
            return "Direct3D 12";
        case rachis::Backend::vulkan:
            return "Vulkan";
        case rachis::Backend::metal:
            return "Metal";
        }
        return "unknown";
    }

    inline void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        const char* severity = "info";
        switch (message.severity)
        {
        case rachis::DiagnosticSeverity::information:
            severity = "info";
            break;
        case rachis::DiagnosticSeverity::warning:
            severity = "warning";
            break;
        case rachis::DiagnosticSeverity::error:
            severity = "error";
            break;
        case rachis::DiagnosticSeverity::corruption:
            severity = "corruption";
            break;
        }
        std::fprintf(stderr,
                     "[%s %u] %.*s\n",
                     severity,
                     static_cast<unsigned int>(message.nativeId),
                     static_cast<int>(message.description.size()),
                     message.description.data());
    }

    [[noreturn]] inline void fail(std::string_view what, const rachis::Error& error)
    {
        std::fprintf(stderr, "%.*s failed: %s\n", static_cast<int>(what.size()), what.data(), error.message.c_str());
        std::exit(EXIT_FAILURE);
    }

    template<typename T>
    [[nodiscard]] T require(rachis::Result<T> result, std::string_view what)
    {
        if (!result)
        {
            fail(what, result.error());
        }
        return std::move(*result);
    }

    inline void require(const rachis::Status& status, std::string_view what)
    {
        if (!status)
        {
            fail(what, status.error());
        }
    }

    [[nodiscard]] inline bool vulkanHostHasOnlySoftwareAdapters()
    {
        const auto adapters = rachis::enumerateAdapters(rachis::Backend::vulkan);
        return adapters.has_value() && std::ranges::none_of(*adapters, [](const rachis::AdapterInfo& adapter) {
                   return adapter.type != rachis::AdapterType::software;
               });
    }

    [[nodiscard]] inline bool backendCannotRunHere(rachis::Backend backend, rachis::ErrorCode code)
    {
        if (backend != rachis::Backend::vulkan)
        {
            return code == rachis::ErrorCode::backendUnavailable || code == rachis::ErrorCode::adapterNotFound;
        }
#ifdef _WIN32
        return false;
#else
        return code == rachis::ErrorCode::backendUnavailable ||
               (code == rachis::ErrorCode::unsupported && vulkanHostHasOnlySoftwareAdapters());
#endif
    }

    [[nodiscard]] inline rachis::Device createDevice(rachis::DeviceDesc desc)
    {
        desc.diagnosticCallback = &printDiagnostic;
        auto device = rachis::createDevice(desc);
        if (!device)
        {
            std::fprintf(stderr, "createDevice failed: %s\n", device.error().message.c_str());
            std::exit(backendCannotRunHere(desc.backend, device.error().code) ? skipExitCode : EXIT_FAILURE);
        }
        if (const auto adapter = device->adapterInfo())
        {
            std::printf("%s on %s\n", backendName(adapter->backend), adapter->name.c_str());
        }
        return std::move(*device);
    }

    [[nodiscard]] inline int finish(const rachis::Device& device)
    {
        const auto counts = device.diagnosticCounts();
        if (counts.errors != 0 || counts.corruptions != 0)
        {
            std::fprintf(stderr,
                         "Validation reported %llu errors and %llu corruptions.\n",
                         static_cast<unsigned long long>(counts.errors),
                         static_cast<unsigned long long>(counts.corruptions));
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

} // namespace example
