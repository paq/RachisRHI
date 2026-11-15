#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cstdint>

namespace
{

    void checkRequirements(const rachis::ResourceMemoryRequirements& requirements, std::uint64_t minimumSize)
    {
        CHECK(requirements.size >= minimumSize);
        CHECK(requirements.alignment != 0);
        CHECK(std::has_single_bit(requirements.alignment));
        CHECK(requirements.size % requirements.alignment == 0);
    }

    void checkSameSnapshot(const rachis::ResourceMemorySnapshot& left, const rachis::ResourceMemorySnapshot& right)
    {
        CHECK(left.entries.size() == right.entries.size());
        CHECK(left.liveAllocationBytes == right.liveAllocationBytes);
        CHECK(left.retainedAllocationBytes == right.retainedAllocationBytes);
        CHECK(left.liveAllocationCount == right.liveAllocationCount);
        CHECK(left.retainedAllocationCount == right.retainedAllocationCount);
    }

} // namespace

TEST_CASE("resource memory requirement queries are conservative and side-effect free",
          "[size:medium][rhi][memory][requirements][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto baseline = device.captureResourceMemorySnapshot();
    REQUIRE(baseline.has_value());

    const rachis::BufferDesc bufferDesc{
        .size = 4'097,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Requirements buffer",
    };
    const auto bufferRequirements = device.queryBufferMemoryRequirements(bufferDesc);
    REQUIRE(bufferRequirements.has_value());
    checkRequirements(*bufferRequirements, bufferDesc.size);
    CHECK_FALSE(bufferRequirements->dedicatedAllocation);

    const rachis::TextureDesc textureDesc{
        .dimension = rachis::TextureDimension::texture2D,
        .width = 65,
        .height = 33,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        .label = "Requirements texture",
    };
    const auto textureRequirements = device.queryTextureMemoryRequirements(textureDesc);
    REQUIRE(textureRequirements.has_value());
    checkRequirements(*textureRequirements, 1);
    CHECK_FALSE(textureRequirements->dedicatedAllocation);

    auto invalidBuffer = bufferDesc;
    invalidBuffer.memoryType = static_cast<rachis::MemoryType>(0xFF);
    CHECK_FALSE(device.queryBufferMemoryRequirements(invalidBuffer).has_value());
    CHECK_FALSE(device.createBuffer(invalidBuffer).has_value());

    auto invalidTexture = textureDesc;
    invalidTexture.usage = static_cast<rachis::TextureUsage>(1U << 31U);
    CHECK_FALSE(device.queryTextureMemoryRequirements(invalidTexture).has_value());
    CHECK_FALSE(device.createTexture(invalidTexture).has_value());
    invalidTexture = textureDesc;
    invalidTexture.dimension = static_cast<rachis::TextureDimension>(0xFF);
    CHECK_FALSE(device.queryTextureMemoryRequirements(invalidTexture).has_value());
    CHECK_FALSE(device.createTexture(invalidTexture).has_value());

    if (rachis::test::requireCapabilities(device).manualResidency)
    {
        auto manualBuffer = bufferDesc;
        manualBuffer.allocationFlags = rachis::ResourceAllocationFlag::manualResidency;
        const auto manualBufferRequirements = device.queryBufferMemoryRequirements(manualBuffer);
        REQUIRE(manualBufferRequirements.has_value());
        checkRequirements(*manualBufferRequirements, manualBuffer.size);
        CHECK(manualBufferRequirements->dedicatedAllocation);

        auto manualTexture = textureDesc;
        manualTexture.allocationFlags = rachis::ResourceAllocationFlag::manualResidency;
        const auto manualTextureRequirements = device.queryTextureMemoryRequirements(manualTexture);
        REQUIRE(manualTextureRequirements.has_value());
        checkRequirements(*manualTextureRequirements, 1);
        CHECK(manualTextureRequirements->dedicatedAllocation);
    }

    const auto afterQueries = device.captureResourceMemorySnapshot();
    REQUIRE(afterQueries.has_value());
    checkSameSnapshot(*baseline, *afterQueries);

    const auto buffer = device.createBuffer(bufferDesc);
    const auto texture = device.createTexture(textureDesc);
    REQUIRE(buffer.has_value());
    REQUIRE(texture.has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    device.collectGarbage();

    const auto released = device.captureResourceMemorySnapshot();
    REQUIRE(released.has_value());
    checkSameSnapshot(*baseline, *released);
}
