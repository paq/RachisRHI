#include "FillShader.h"
#include "Libraries.h"

#include <Rachis/Device.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    struct Owner final
    {
        const char* name;
        std::span<const std::byte> (*shader)(rachis::Backend);
        const char* directory;
    };

    struct BackendCase final
    {
        const char* name;
        rachis::Backend backend;
        const char* extension;
    };

    std::span<const std::byte> programFillShader(rachis::Backend backend)
    {
        return rachis_shaders::FillShader(backend);
    }

    std::vector<std::byte> readFile(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        const std::vector<char> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        std::vector<std::byte> result(bytes.size());
        if (!bytes.empty())
        {
            std::memcpy(result.data(), bytes.data(), bytes.size());
        }
        return result;
    }
} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: %s <program dir> <first dir> <second dir> [d3d12|vulkan|metal...]\n", argv[0]);
        return 2;
    }
    const std::array owners{
        Owner{"the program", programFillShader, argv[1]},
        Owner{"the first library", firstFillShader, argv[2]},
        Owner{"the second library", secondFillShader, argv[3]},
    };
    const std::array backends{
        BackendCase{"d3d12", rachis::Backend::d3d12, "dxil"},
        BackendCase{"vulkan", rachis::Backend::vulkan, "spv"},
        BackendCase{"metal", rachis::Backend::metal, "pnms"},
    };
    const std::span<char*> enabledNames(argv + 4, static_cast<std::size_t>(argc - 4));

    int failures = 0;
    for (const BackendCase& backend : backends)
    {
        const bool enabled =
            std::ranges::any_of(enabledNames, [&](const char* name) { return std::string_view{name} == backend.name; });
        std::array<std::span<const std::byte>, owners.size()> spans{};
        for (std::size_t index = 0; index < owners.size(); ++index)
        {
            spans[index] = owners[index].shader(backend.backend);
        }

        if (!enabled)
        {
            for (std::size_t index = 0; index < owners.size(); ++index)
            {
                if (!spans[index].empty())
                {
                    std::fprintf(stderr,
                                 "%s: FillShader in %s returned %zu bytes for a backend the package lacks.\n",
                                 backend.name,
                                 owners[index].name,
                                 spans[index].size());
                    ++failures;
                }
            }
            continue;
        }

        bool shared = false;
        for (std::size_t first = 0; first < owners.size(); ++first)
        {
            for (std::size_t second = first + 1; second < owners.size(); ++second)
            {
                if (spans[first].data() == spans[second].data())
                {
                    std::fprintf(stderr,
                                 "%s: FillShader in %s (%zu bytes) and in %s (%zu bytes) point at the same array %p.\n",
                                 backend.name,
                                 owners[first].name,
                                 spans[first].size(),
                                 owners[second].name,
                                 spans[second].size(),
                                 static_cast<const void*>(spans[first].data()));
                    shared = true;
                    ++failures;
                }
            }
        }
        if (shared)
        {
            continue;
        }

        for (std::size_t index = 0; index < owners.size(); ++index)
        {
            const std::string path = std::string{owners[index].directory} + "/FillShader." + backend.extension;
            const std::vector<std::byte> file = readFile(path);
            if (file.empty())
            {
                std::fprintf(stderr, "%s: could not read %s.\n", backend.name, path.c_str());
                ++failures;
                continue;
            }
            if (spans[index].size() != file.size() || std::memcmp(spans[index].data(), file.data(), file.size()) != 0)
            {
                std::fprintf(stderr,
                             "%s: FillShader in %s returned %zu bytes that differ from the %zu bytes of %s.\n",
                             backend.name,
                             owners[index].name,
                             spans[index].size(),
                             file.size(),
                             path.c_str());
                ++failures;
                continue;
            }
            std::printf("%s: %s got its own %zu-byte FillShader.\n", backend.name, owners[index].name, file.size());
        }
    }
    return failures == 0 ? 0 : 1;
}
