#include "D3D12/D3D12TestHooks.h"
#include "FillAndClearContract.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <vector>

namespace
{

    class ForceLegacyBarriers final
    {
    public:
        explicit ForceLegacyBarriers(bool enabled)
            : previous_(rachis::d3d12::test::forceLegacyBarriers().exchange(enabled, std::memory_order_relaxed))
        {
        }

        ~ForceLegacyBarriers()
        {
            rachis::d3d12::test::forceLegacyBarriers().store(previous_, std::memory_order_relaxed);
        }

        ForceLegacyBarriers(const ForceLegacyBarriers&) = delete;
        ForceLegacyBarriers& operator=(const ForceLegacyBarriers&) = delete;
        ForceLegacyBarriers(ForceLegacyBarriers&&) = delete;
        ForceLegacyBarriers& operator=(ForceLegacyBarriers&&) = delete;

    private:
        bool previous_ = false;
    };

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] std::vector<std::byte> loadShader(const char* path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> result(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(result.data()), end);
        return stream ? result : std::vector<std::byte>{};
    }

    [[nodiscard]] rachis::Device createWarpDevice(bool legacyBarriers)
    {
        const ForceLegacyBarriers forced(legacyBarriers);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .diagnosticCallback = &printDiagnostic,
        });
        if (!created)
        {
            INFO(created.error().message);
        }
        REQUIRE(created.has_value());
        const auto capabilities = rachis::test::requireCapabilities(*created);
        if (legacyBarriers)
        {
            REQUIRE_FALSE(capabilities.enhancedBarriers);
        }
        return std::move(*created);
    }

} // namespace

TEST_CASE("D3D12 fillBuffer resets a count buffer before a counted indirect draw",
          "[size:medium][rhi][d3d12][fill][indirect][warp]")
{
    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_INPUT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    for (const bool legacyBarriers : {false, true})
    {
        INFO("Legacy barriers: " << legacyBarriers);
        auto device = createWarpDevice(legacyBarriers);
        rachis::test::fill_and_clear::verifyCountedDrawAfterFill(
            device, {.inputVertex = vertexBytecode, .pixel = pixelBytecode});
    }
}

TEST_CASE("D3D12 fillBuffer writes 32-bit patterns across a large buffer", "[size:medium][rhi][d3d12][fill][warp]")
{
    for (const bool legacyBarriers : {false, true})
    {
        INFO("Legacy barriers: " << legacyBarriers);
        auto device = createWarpDevice(legacyBarriers);
        rachis::test::fill_and_clear::verifyLargeBufferFill(device, rachis::QueueType::compute);
    }
}

TEST_CASE("D3D12 clearStorageTexture clears the subresources a storage view covers",
          "[size:medium][rhi][d3d12][clear][warp]")
{
    for (const bool legacyBarriers : {false, true})
    {
        INFO("Legacy barriers: " << legacyBarriers);
        auto device = createWarpDevice(legacyBarriers);
        rachis::test::fill_and_clear::verifyStorageTextureClears(device, rachis::QueueType::graphics);
        rachis::test::fill_and_clear::verifyStorageTextureClears(device, rachis::QueueType::compute);
    }
}

TEST_CASE("D3D12 fillBuffer and clearStorageTexture reject invalid queues, ranges, usages, values, and layouts",
          "[size:medium][rhi][d3d12][fill][clear][warp]")
{
    for (const bool legacyBarriers : {false, true})
    {
        INFO("Legacy barriers: " << legacyBarriers);
        auto device = createWarpDevice(legacyBarriers);
        rachis::test::fill_and_clear::verifyFillAndClearRejections(device);
    }
}
