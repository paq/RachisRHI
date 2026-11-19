#include "FillAndClearContract.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

namespace
{

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadPnms(std::string_view name)
    {
        std::ifstream stream(std::filesystem::path(RACHIS_METALLIB_DIR) / std::filesystem::path(name),
                             std::ios::binary | std::ios::ate);
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

TEST_CASE("Metal fillBuffer resets a count buffer before a counted indirect draw and writes 32-bit patterns",
          "[size:large][rhi][metal][fill][indirect]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto vertexBytecode = loadPnms("GraphicsTriangle.input.vs.pnms");
    const auto pixelBytecode = loadPnms("GraphicsTriangle.ps.pnms");
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    rachis::test::fill_and_clear::verifyCountedDrawAfterFill(device,
                                                             {.inputVertex = vertexBytecode, .pixel = pixelBytecode});
    rachis::test::fill_and_clear::verifyLargeBufferFill(device, rachis::QueueType::compute);
}

TEST_CASE("Metal clearStorageTexture clears the subresources a storage view covers", "[size:large][rhi][metal][clear]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    rachis::test::fill_and_clear::verifyStorageTextureClears(device, rachis::QueueType::graphics);
    rachis::test::fill_and_clear::verifyStorageTextureClears(device, rachis::QueueType::compute);
}

TEST_CASE("Metal fillBuffer and clearStorageTexture reject invalid queues, ranges, usages, values, and layouts",
          "[size:large][rhi][metal][fill][clear]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    rachis::test::fill_and_clear::verifyFillAndClearRejections(device);
}
