#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{

    enum class Invocation
    {
        spirvProbe,
        metalProbe,
        metallib,
        metalSource,
    };

    struct Request final
    {
        Invocation invocation = Invocation::metalSource;
        std::string entryPoint;
        std::optional<std::filesystem::path> output;
        std::optional<std::filesystem::path> reflectionJson;
        std::vector<std::string> defines{};
        std::vector<std::string> includeDirectories{};
    };

    [[nodiscard]] std::string asciiText(const std::filesystem::path& arg)
    {
        const std::u8string text = arg.u8string();
        return std::string(text.begin(), text.end());
    }

    [[nodiscard]] Request parseRequest(std::span<const std::filesystem::path> args)
    {
        Request request;
        std::string target;
        for (std::size_t index = 0; index < args.size(); ++index)
        {
            const std::string flag = asciiText(args[index]);
            const bool hasValue = index + 1 < args.size();
            if (flag == "-target" && hasValue)
            {
                target = asciiText(args[++index]);
            }
            else if (flag == "-entry" && hasValue)
            {
                request.entryPoint = asciiText(args[++index]);
            }
            else if (flag == "-o" && hasValue)
            {
                request.output = args[++index];
            }
            else if (flag == "-reflection-json" && hasValue)
            {
                request.reflectionJson = args[++index];
            }
            else if (flag == "-I" && hasValue)
            {
                request.includeDirectories.push_back(asciiText(args[++index]));
            }
            else if (flag.starts_with("-D"))
            {
                request.defines.push_back(flag.substr(2));
            }
        }
        if (target == "spirv")
        {
            request.invocation = Invocation::spirvProbe;
        }
        else if (target == "metallib")
        {
            request.invocation = Invocation::metallib;
        }
        else if (request.reflectionJson.has_value())
        {
            request.invocation = Invocation::metalProbe;
        }
        return request;
    }

    [[nodiscard]] std::string_view invocationName(Invocation invocation) noexcept
    {
        switch (invocation)
        {
        case Invocation::spirvProbe:
            return "spirv-probe";
        case Invocation::metalProbe:
            return "metal-probe";
        case Invocation::metallib:
            return "metallib";
        case Invocation::metalSource:
            return "metal-source";
        }
        return "unknown";
    }

    [[nodiscard]] std::string requestedBehavior(const Request& request)
    {
        std::string key;
        if (request.invocation == Invocation::spirvProbe)
        {
            key = "FAKE_SLANGC_SPIRV_PROBE=";
        }
        else if (request.invocation == Invocation::metalProbe)
        {
            key = "FAKE_SLANGC_METAL_PROBE=";
        }
        else
        {
            return {};
        }
        for (const auto& define : request.defines)
        {
            if (define.starts_with(key))
            {
                return define.substr(key.size());
            }
        }
        return {};
    }

    [[nodiscard]] bool writeFile(const std::filesystem::path& path, std::string_view contents)
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        stream.flush();
        return stream.good();
    }

    [[nodiscard]] std::string spirvReflection(std::string_view entryPoint)
    {
        return std::string(R"({"parameters": [)"
                           R"({"name": "pushBlock", "binding": {"kind": "pushConstantBuffer"}, )"
                           R"("type": {"kind": "constantBuffer"}}, )"
                           R"({"name": "output", "binding": {"kind": "descriptorTableSlot", "space": 1, "index": 3}, )"
                           R"("type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}}], )"
                           R"("entryPoints": [{"name": ")") +
               std::string(entryPoint) + R"(", "stage": "compute"}]})";
    }

    [[nodiscard]] std::string metalReflection(std::string_view entryPoint)
    {
        return std::string(R"({"parameters": [)"
                           R"({"name": "pushBlock", "binding": {"kind": "constantBuffer", "index": 0}, )"
                           R"("type": {"kind": "constantBuffer"}}, )"
                           R"({"name": "output", "binding": {"kind": "constantBuffer", "index": 1}, )"
                           R"("type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}}], )"
                           R"("entryPoints": [{"name": ")") +
               std::string(entryPoint) + R"(", "stage": "compute", "threadGroupSize": [1, 1, 1]}]})";
    }

    [[nodiscard]] int run(std::span<const std::filesystem::path> args)
    {
        const Request request = parseRequest(args);
        const std::string_view name = invocationName(request.invocation);
        std::cerr << "fake slangc: " << name << " ran\n";
        for (const auto& directory : request.includeDirectories)
        {
            std::cerr << "fake slangc: " << name << " includes " << directory << '\n';
        }
        constexpr std::string_view expectedIncludesKey = "FAKE_SLANGC_INCLUDES=";
        for (const auto& define : request.defines)
        {
            if (define.starts_with(expectedIncludesKey) &&
                define.substr(expectedIncludesKey.size()) != std::to_string(request.includeDirectories.size()))
            {
                std::cerr << "fake slangc: " << name << " got " << request.includeDirectories.size()
                          << " include directories\n";
                return 5;
            }
        }

        const std::string behavior = requestedBehavior(request);
        if (behavior == "fail")
        {
            std::cerr << "fake slangc: " << name << " failed\n";
            return 3;
        }
        if (behavior == "silent")
        {
            return 0;
        }

        std::string reflection;
        if (behavior == "malformed")
        {
            reflection = R"({"parameters": [)";
        }
        else if (request.invocation == Invocation::spirvProbe)
        {
            reflection = spirvReflection(request.entryPoint);
        }
        else
        {
            reflection = metalReflection(request.entryPoint);
        }
        if (request.reflectionJson.has_value() && !writeFile(*request.reflectionJson, reflection))
        {
            std::cerr << "fake slangc: cannot write the reflection JSON\n";
            return 4;
        }
        if (request.output.has_value())
        {
            const std::string_view contents = request.invocation == Invocation::metallib
                                                  ? std::string_view{"MTLB\x01\x02\x03\x04", 8}
                                                  : std::string_view{"// fake metal source\n"};
            if (!writeFile(*request.output, contents))
            {
                std::cerr << "fake slangc: cannot write the output\n";
                return 4;
            }
        }
        return 0;
    }

} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
    const std::vector<std::filesystem::path> args(argv + 1, argv + argc);
    return run(args);
}
