#include "FillAndClearContract.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadBytes(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        return stream ? bytes : std::vector<std::byte>{};
    }

} // namespace

TEST_CASE("Vulkan fillBuffer resets a count buffer before a counted indirect draw",
          "[size:medium][rhi][vulkan][fill][indirect]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto vertexBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanGraphicsTriangle.input.vs.spv");
    const auto pixelBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanGraphicsTriangle.ps.spv");
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    rachis::test::fill_and_clear::verifyCountedDrawAfterFill(device,
                                                             {.inputVertex = vertexBytecode, .pixel = pixelBytecode});
}

TEST_CASE("Vulkan fillBuffer writes 32-bit patterns across a large buffer", "[size:medium][rhi][vulkan][fill]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    rachis::test::fill_and_clear::verifyLargeBufferFill(device, rachis::QueueType::compute);
}

TEST_CASE("Vulkan clearStorageTexture clears the subresources a storage view covers",
          "[size:medium][rhi][vulkan][clear]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    rachis::test::fill_and_clear::verifyStorageTextureClears(device, rachis::QueueType::graphics);
    rachis::test::fill_and_clear::verifyStorageTextureClears(device, rachis::QueueType::compute);
}

TEST_CASE("Vulkan fillBuffer and clearStorageTexture reject invalid queues, ranges, usages, values, and layouts",
          "[size:medium][rhi][vulkan][fill][clear]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    rachis::test::fill_and_clear::verifyFillAndClearRejections(device);
}
