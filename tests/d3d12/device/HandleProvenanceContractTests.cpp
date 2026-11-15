#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>

TEST_CASE("same-type handles cannot cross D3D12 device boundaries", "[size:medium][rhi][handle][device][warp]")
{
    auto firstCreated = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
    });
    auto secondCreated = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
    });
    if (!firstCreated)
    {
        INFO(firstCreated.error().message);
    }
    if (!secondCreated)
    {
        INFO(secondCreated.error().message);
    }
    REQUIRE(firstCreated.has_value());
    REQUIRE(secondCreated.has_value());
    auto first = std::move(*firstCreated);
    auto second = std::move(*secondCreated);

    constexpr rachis::BufferDesc desc{
        .size = 4,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    };
    auto firstBuffer = first.createBuffer(desc);
    auto secondBuffer = second.createBuffer(desc);
    REQUIRE(firstBuffer.has_value());
    REQUIRE(secondBuffer.has_value());
    CHECK(*firstBuffer != *secondBuffer);

    constexpr std::array baseline{
        std::byte{0x11},
        std::byte{0x22},
        std::byte{0x33},
        std::byte{0x44},
    };
    constexpr std::array replacement{
        std::byte{0xAA},
        std::byte{0xBB},
        std::byte{0xCC},
        std::byte{0xDD},
    };
    REQUIRE(second.writeBuffer(*secondBuffer, 0, baseline).has_value());

    const auto foreignWrite = second.writeBuffer(*firstBuffer, 0, replacement);
    REQUIRE_FALSE(foreignWrite.has_value());
    CHECK(foreignWrite.error().code == rachis::ErrorCode::invalidArgument);

    auto mapped = second.mapBuffer(*secondBuffer);
    REQUIRE(mapped.has_value());
    REQUIRE(mapped->size == baseline.size());
    for (std::size_t index = 0; index < baseline.size(); ++index)
    {
        CHECK(mapped->data[index] == baseline[index]);
    }
    REQUIRE(second.unmapBuffer(*secondBuffer).has_value());
}
