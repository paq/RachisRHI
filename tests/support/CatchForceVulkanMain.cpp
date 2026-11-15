#include "ForceVulkanRequest.h"

#include <catch2/catch_session.hpp>

#include <cstdio>
#include <exception>

int main(int argc, char* argv[])
try
{
    auto forceVulkan = false;
    Catch::Session session;
    auto cli = session.cli() | Catch::Clara::Opt(forceVulkan)["--force-vulkan"](
                                   "Require Vulkan. Missing MoltenVK or a native ICD fails instead of skipping.");
    session.cli(cli);
    const auto parsed = session.applyCommandLine(argc, argv);
    if (parsed != 0)
    {
        return parsed;
    }

    rachis::test::setForceVulkanRequested(forceVulkan);
    return session.run();
}

catch (const std::exception& error)
{
    std::fputs("Native test runner failed: ", stderr);
    std::fputs(error.what(), stderr);
    std::fputc('\n', stderr);
    return 1;
}
catch (...)
{
    std::fputs("Native test runner failed with an unknown exception.\n", stderr);
    return 1;
}
