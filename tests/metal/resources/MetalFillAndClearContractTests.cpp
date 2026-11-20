#include "FillAndClearContract.h"
#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

TEST_CASE("Metal fillBuffer resets a count buffer before a counted indirect draw",
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
}

TEST_CASE("Metal fillBuffer writes 32-bit patterns across a large buffer", "[size:large][rhi][metal][fill]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
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

TEST_CASE("Metal fillBuffer encodes no chunk when it cannot prepare every chunk", "[size:large][rhi][metal][fill]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    constexpr std::uint64_t chunkBytes = std::uint64_t{65535} * 256U * sizeof(std::uint32_t);
    constexpr std::uint64_t bufferSize = chunkBytes + sizeof(std::uint32_t);
    auto buffer = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal two-chunk fill target",
    });
    REQUIRE(buffer.has_value());
    const auto fill = [&](std::uint32_t value, std::int64_t failSliceAfter) {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        rachis::metal::setFailPushConstantSliceAfter(failSliceAfter);
        const auto filled = device.fillBuffer(*command, *buffer, 0, rachis::wholeResource, value);
        rachis::metal::setFailPushConstantSliceAfter(-1);
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        return filled;
    };
    const auto words = [&] {
        auto mapped = device.mapBuffer(*buffer);
        REQUIRE(mapped.has_value());
        std::array<std::uint32_t, 2> ends{};
        std::memcpy(&ends[0], mapped->data, sizeof(std::uint32_t));
        std::memcpy(&ends[1], mapped->data + bufferSize - sizeof(std::uint32_t), sizeof(std::uint32_t));
        REQUIRE(device.unmapBuffer(*buffer).has_value());
        return ends;
    };

    REQUIRE(fill(0x11111111U, -1).has_value());
    CHECK(words() == std::array<std::uint32_t, 2>{0x11111111U, 0x11111111U});

    const auto failed = fill(0x22222222U, 1);
    REQUIRE_FALSE(failed.has_value());
    CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(words() == std::array<std::uint32_t, 2>{0x11111111U, 0x11111111U});

    REQUIRE(fill(0x33333333U, -1).has_value());
    CHECK(words() == std::array<std::uint32_t, 2>{0x33333333U, 0x33333333U});
    REQUIRE(device.destroyBuffer(*buffer).has_value());
}

TEST_CASE("Metal clearStorageTexture that fails to encode leaves no layout requirement",
          "[size:large][rhi][metal][clear]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto texture = device.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::storage | rachis::TextureUsage::copyDestination,
        .label = "Metal failed-clear texture",
    });
    REQUIRE(texture.has_value());
    auto view = device.createTextureView({.texture = *texture, .usage = rachis::TextureViewUsage::storage});
    REQUIRE(view.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    rachis::metal::setFailPushConstantSliceAfter(0);
    const auto cleared = device.clearStorageTexture(
        *command, *view, {.type = rachis::StorageTextureClearType::float32, .float32 = {1.0F, 0.0F, 0.0F, 1.0F}});
    rachis::metal::setFailPushConstantSliceAfter(-1);
    REQUIRE_FALSE(cleared.has_value());
    CHECK(cleared.error().code == rachis::ErrorCode::outOfMemory);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    REQUIRE(device.destroyTextureView(*view).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
}

TEST_CASE("Metal clearStorageTexture that fails while recording layouts can be retried",
          "[size:large][rhi][metal][clear]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto texture = device.createTexture({
        .dimension = rachis::TextureDimension::texture2D,
        .width = 4,
        .height = 4,
        .depthOrArrayLayers = 3,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::storage | rachis::TextureUsage::copyDestination,
        .label = "Metal retried-clear texture",
    });
    REQUIRE(texture.has_value());
    auto view = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::storage,
        .dimension = rachis::TextureViewDimension::texture2DArray,
    });
    REQUIRE(view.has_value());

    auto transition = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(transition.has_value());
    REQUIRE(device
                .textureBarrier(*transition,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    auto transitioned = device.submit(std::move(*transition));
    REQUIRE(transitioned.has_value());
    REQUIRE(device.wait(*transitioned).has_value());

    const rachis::StorageTextureClearValue value{
        .type = rachis::StorageTextureClearType::float32,
        .float32 = {1.0F, 0.0F, 0.0F, 1.0F},
    };
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    rachis::metal::setFailNextLayoutStaging(true);
    const auto failed = device.clearStorageTexture(*command, *view, value);
    rachis::metal::setFailNextLayoutStaging(false);
    REQUIRE_FALSE(failed.has_value());
    CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    REQUIRE(device.clearStorageTexture(*command, *view, value).has_value());
    REQUIRE(device.clearStorageTexture(*command, *view, value).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    REQUIRE(device.destroyTextureView(*view).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
}
