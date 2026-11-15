#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{

#if RACHIS_ENABLE_METAL
    constexpr std::uint64_t bufferSize = 64;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::byte sourceValue(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 17U + 3U) & 0xFFU);
    }
#endif

} // namespace

#if RACHIS_ENABLE_METAL

TEST_CASE("Metal same-type handles cannot cross device boundaries", "[size:large][rhi][metal][handle][provenance]")
{
    auto firstCreated = rachis::createDevice(metalDesc());
    auto secondCreated = rachis::createDevice(metalDesc());
    REQUIRE(firstCreated);
    REQUIRE(secondCreated);
    auto first = std::move(*firstCreated);
    auto second = std::move(*secondCreated);

    const rachis::BufferDesc desc{
        .size = 4,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    };
    auto firstBuffer = first.createBuffer(desc);
    auto secondBuffer = second.createBuffer(desc);
    REQUIRE(firstBuffer);
    REQUIRE(secondBuffer);
    CHECK(*firstBuffer != *secondBuffer);

    constexpr std::array baseline{std::byte{0x11}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}};
    constexpr std::array replacement{std::byte{0xAA}, std::byte{0xBB}, std::byte{0xCC}, std::byte{0xDD}};
    REQUIRE(second.writeBuffer(*secondBuffer, 0, baseline).has_value());
    const auto foreignWrite = second.writeBuffer(*firstBuffer, 0, replacement);
    REQUIRE_FALSE(foreignWrite);
    CHECK(foreignWrite.error().code == rachis::ErrorCode::invalidArgument);
    CHECK_FALSE(second.destroyBuffer(*firstBuffer).has_value());
    CHECK_FALSE(first.destroyBuffer(*secondBuffer).has_value());

    auto mapped = second.mapBuffer(*secondBuffer, 0, 4);
    REQUIRE(mapped);
    REQUIRE(mapped->data != nullptr);
    CHECK(mapped->data[0] == baseline[0]);
    CHECK(mapped->data[1] == baseline[1]);
    CHECK(mapped->data[2] == baseline[2]);
    CHECK(mapped->data[3] == baseline[3]);
    REQUIRE(second.unmapBuffer(*secondBuffer).has_value());
}

TEST_CASE("Metal destroy after record keeps in-flight copies and rejects stale handles",
          "[size:large][rhi][metal][handle][lifetime]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal provenance source",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal provenance destination",
    });
    REQUIRE(source);
    REQUIRE(destination);

    std::array<std::byte, bufferSize> sourceBytes{};
    for (std::uint64_t index = 0; index < bufferSize; ++index)
    {
        sourceBytes[index] = sourceValue(index);
    }
    REQUIRE(device.writeBuffer(*source, 0, sourceBytes).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *source,
                                .destination = *destination,
                                .size = bufferSize,
                            })
                .has_value());

    const auto stale = *source;
    REQUIRE(device.destroyBuffer(*source).has_value());
    CHECK_FALSE(device.writeBuffer(stale, 0, sourceBytes).has_value());
    CHECK_FALSE(device.destroyBuffer(stale).has_value());

    auto reused = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal provenance reused",
    });
    REQUIRE(reused);
    CHECK(*reused != stale);
    CHECK_FALSE(device.writeBuffer(stale, 0, sourceBytes).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*destination, 0, actual).has_value());
    CHECK(actual == sourceBytes);
    REQUIRE(device.destroyBuffer(*destination).has_value());
    REQUIRE(device.destroyBuffer(*reused).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal multi-queue destroy retires only after every submitted point",
          "[size:large][rhi][metal][handle][lifetime][queue]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal multi-queue source",
    });
    auto copyDest = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal multi-queue copy dest",
    });
    auto graphicsDest = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal multi-queue graphics dest",
    });
    REQUIRE(source);
    REQUIRE(copyDest);
    REQUIRE(graphicsDest);

    std::array<std::byte, bufferSize> sourceBytes{};
    for (std::uint64_t index = 0; index < bufferSize; ++index)
    {
        sourceBytes[index] = sourceValue(index + 4);
    }
    REQUIRE(device.writeBuffer(*source, 0, sourceBytes).has_value());

    auto copy = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    auto graphics = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(copy);
    REQUIRE(graphics);
    REQUIRE(device.copyBuffer(*copy, {.source = *source, .destination = *copyDest, .size = bufferSize}).has_value());
    REQUIRE(device.copyBuffer(*graphics, {.source = *source, .destination = *graphicsDest, .size = bufferSize})
                .has_value());
    REQUIRE(device.destroyBuffer(*source).has_value());

    auto copyPoint = device.submit(std::move(*copy));
    auto graphicsPoint = device.submit(std::move(*graphics));
    REQUIRE(copyPoint);
    REQUIRE(graphicsPoint);
    REQUIRE(device.wait(*copyPoint).has_value());
    REQUIRE(device.wait(*graphicsPoint).has_value());
    device.collectGarbage();

    std::array<std::byte, bufferSize> copyActual{};
    std::array<std::byte, bufferSize> graphicsActual{};
    REQUIRE(device.readBuffer(*copyDest, 0, copyActual).has_value());
    REQUIRE(device.readBuffer(*graphicsDest, 0, graphicsActual).has_value());
    CHECK(copyActual == sourceBytes);
    CHECK(graphicsActual == sourceBytes);
    REQUIRE(device.destroyBuffer(*copyDest).has_value());
    REQUIRE(device.destroyBuffer(*graphicsDest).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
}

TEST_CASE("Metal unretained recordings require the caller to keep application resources",
          "[size:large][rhi][metal][handle][unretained]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal unretained source",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal unretained destination",
    });
    REQUIRE(source);
    REQUIRE(destination);

    std::array<std::byte, bufferSize> sourceBytes{};
    for (std::uint64_t index = 0; index < bufferSize; ++index)
    {
        sourceBytes[index] = sourceValue(index + 11);
    }
    REQUIRE(device.writeBuffer(*source, 0, sourceBytes).has_value());

    auto command = rachis::test::acquireCommandBuffer(
        device, rachis::QueueType::copy, rachis::CommandPoolFlag::unretainedResources);
    REQUIRE(command);
    REQUIRE(
        device.copyBuffer(*command, {.source = *source, .destination = *destination, .size = bufferSize}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*destination, 0, actual).has_value());
    CHECK(actual == sourceBytes);
    REQUIRE(device.destroyBuffer(*source).has_value());
    REQUIRE(device.destroyBuffer(*destination).has_value());
}

#endif
