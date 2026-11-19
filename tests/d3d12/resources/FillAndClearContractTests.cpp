#include "D3D12/D3D12TestHooks.h"
#include "FillAndClearContract.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <utility>
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

    class ForceBufferFillChunkElements final
    {
    public:
        explicit ForceBufferFillChunkElements(std::uint64_t elements)
            : previous_(
                  rachis::d3d12::test::bufferFillChunkElementLimit().exchange(elements, std::memory_order_relaxed))
        {
        }

        ~ForceBufferFillChunkElements()
        {
            rachis::d3d12::test::bufferFillChunkElementLimit().store(previous_, std::memory_order_relaxed);
        }

        ForceBufferFillChunkElements(const ForceBufferFillChunkElements&) = delete;
        ForceBufferFillChunkElements& operator=(const ForceBufferFillChunkElements&) = delete;
        ForceBufferFillChunkElements(ForceBufferFillChunkElements&&) = delete;
        ForceBufferFillChunkElements& operator=(ForceBufferFillChunkElements&&) = delete;

    private:
        std::uint64_t previous_ = 0;
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

TEST_CASE("D3D12 fillBuffer orders the UAV clears of a fill that spans several typed views",
          "[size:medium][rhi][d3d12][fill][warp]")
{
    constexpr std::uint64_t chunkElements = 65536;
    constexpr std::uint64_t size = 3ULL * 1024ULL * 1024ULL + 64ULL;
    constexpr std::uint64_t offset = 4;
    const auto chunkBarriers = [](std::uint64_t bytes) { return (bytes / 4 + chunkElements - 1) / chunkElements - 1; };
    const ForceBufferFillChunkElements forcedChunks(chunkElements);
    for (const bool legacyBarriers : {false, true})
    {
        INFO("Legacy barriers: " << legacyBarriers);
        auto device = createWarpDevice(legacyBarriers);
        const auto before = rachis::d3d12::test::bufferFillChunkBarrierCount().load(std::memory_order_relaxed);
        rachis::test::fill_and_clear::verifyLargeBufferFill(device, rachis::QueueType::compute);
        const auto recorded = rachis::d3d12::test::bufferFillChunkBarrierCount().load(std::memory_order_relaxed);
        CHECK(recorded - before == chunkBarriers(size) + chunkBarriers(size - 2 * offset));
    }
}

TEST_CASE("D3D12 clearStorageTexture clamps normalized values like a shader write",
          "[size:medium][rhi][d3d12][clear][warp]")
{
    namespace contract = rachis::test::fill_and_clear;
    const std::array clamped{std::bit_cast<std::uint32_t>(1.0F),
                             std::bit_cast<std::uint32_t>(0.0F),
                             std::bit_cast<std::uint32_t>(0.0F),
                             std::bit_cast<std::uint32_t>(1.0F)};
    for (const bool legacyBarriers : {false, true})
    {
        auto device = createWarpDevice(legacyBarriers);
        for (const auto viewFormat : {rachis::Format::unknown, rachis::Format::rgba8Unorm})
        {
            INFO("Legacy barriers: " << legacyBarriers
                                     << ", explicit view format: " << (viewFormat != rachis::Format::unknown));
            const auto texture = contract::createStorageTexture(device,
                                                                rachis::TextureDesc{
                                                                    .width = 4,
                                                                    .height = 4,
                                                                    .format = rachis::Format::rgba8Unorm,
                                                                });
            auto view = device.createTextureView({
                .texture = texture.texture,
                .usage = rachis::TextureViewUsage::storage,
                .dimension = rachis::TextureViewDimension::texture2D,
                .format = viewFormat,
            });
            REQUIRE(view.has_value());
            auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
            REQUIRE(command.has_value());
            contract::requireTextureBarrier(device,
                                            *command,
                                            texture.texture,
                                            rachis::PipelineStage::all,
                                            rachis::Access::none,
                                            rachis::TextureLayout::undefined,
                                            rachis::PipelineStage::computeShader,
                                            rachis::Access::unorderedAccess,
                                            rachis::TextureLayout::unorderedAccess);
            rachis::d3d12::test::lastStorageTextureClear() = {};
            REQUIRE(
                device.clearStorageTexture(*command, *view, contract::floatClear(2.0F, -1.0F, 0.0F, 8.0F)).has_value());
            const auto native = rachis::d3d12::test::lastStorageTextureClear();
            CHECK(native.floatPath);
            CHECK(native.bits == clamped);
            contract::requireTextureBarrier(device,
                                            *command,
                                            texture.texture,
                                            rachis::PipelineStage::computeShader,
                                            rachis::Access::unorderedAccess,
                                            rachis::TextureLayout::unorderedAccess,
                                            rachis::PipelineStage::copy,
                                            rachis::Access::copySource,
                                            rachis::TextureLayout::copySource);
            const auto readback = contract::recordSubresourceReadback(device, *command, texture, 0, 0);
            contract::submitAndWait(device, std::move(*command));
            CHECK(contract::countMismatchedTexels(device, readback, std::array<std::uint8_t, 4>{255, 0, 0, 255}) == 0);
            REQUIRE(device.destroyTextureView(*view).has_value());
            REQUIRE(device.destroyTexture(texture.texture).has_value());
        }
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }
}

TEST_CASE("D3D12 clearStorageTexture writes 32-bit float values bit for bit", "[size:medium][rhi][d3d12][clear][warp]")
{
    namespace contract = rachis::test::fill_and_clear;
    const std::array<std::uint32_t, 4> bits{1U, 0x8000'0000U, 0x007F'FFFFU, std::bit_cast<std::uint32_t>(-2.5F)};
    std::vector<rachis::Device> devices;
    devices.push_back(createWarpDevice(false));
    devices.push_back(createWarpDevice(true));
    auto adapters = rachis::enumerateAdapters(rachis::Backend::d3d12);
    REQUIRE(adapters.has_value());
    const auto hardware = std::ranges::find_if(
        *adapters, [](const rachis::AdapterInfo& adapter) { return adapter.type != rachis::AdapterType::software; });
    if (hardware != adapters->end())
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::highPerformance,
            .adapterId = hardware->id,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .diagnosticCallback = &printDiagnostic,
        });
        if (!created)
        {
            UNSCOPED_INFO(hardware->name << ": " << created.error().message);
        }
        REQUIRE(created.has_value());
        devices.push_back(std::move(*created));
    }
    for (auto& device : devices)
    {
        INFO("Adapter: " << rachis::test::requireAdapterInfo(device).name
                         << ", enhanced barriers: " << rachis::test::requireCapabilities(device).enhancedBarriers);
        const auto texture = contract::createStorageTexture(device,
                                                            rachis::TextureDesc{
                                                                .width = 4,
                                                                .height = 4,
                                                                .format = rachis::Format::rgba32Float,
                                                            });
        const auto view =
            contract::requireStorageView(device, texture.texture, rachis::TextureViewDimension::texture2D, 0);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        contract::requireTextureBarrier(device,
                                        *command,
                                        texture.texture,
                                        rachis::PipelineStage::all,
                                        rachis::Access::none,
                                        rachis::TextureLayout::undefined,
                                        rachis::PipelineStage::computeShader,
                                        rachis::Access::unorderedAccess,
                                        rachis::TextureLayout::unorderedAccess);
        rachis::d3d12::test::lastStorageTextureClear() = {};
        REQUIRE(device
                    .clearStorageTexture(*command,
                                         view,
                                         contract::floatClear(std::bit_cast<float>(bits[0]),
                                                              std::bit_cast<float>(bits[1]),
                                                              std::bit_cast<float>(bits[2]),
                                                              std::bit_cast<float>(bits[3])))
                    .has_value());
        const auto native = rachis::d3d12::test::lastStorageTextureClear();
        CHECK_FALSE(native.floatPath);
        CHECK(native.bits == bits);
        contract::requireTextureBarrier(device,
                                        *command,
                                        texture.texture,
                                        rachis::PipelineStage::computeShader,
                                        rachis::Access::unorderedAccess,
                                        rachis::TextureLayout::unorderedAccess,
                                        rachis::PipelineStage::copy,
                                        rachis::Access::copySource,
                                        rachis::TextureLayout::copySource);
        const auto readback = contract::recordSubresourceReadback(device, *command, texture, 0, 0);
        contract::submitAndWait(device, std::move(*command));
        CHECK(contract::countMismatchedTexels(device, readback, bits) == 0);
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
        REQUIRE(device.destroyTextureView(view).has_value());
        REQUIRE(device.destroyTexture(texture.texture).has_value());
    }
}
