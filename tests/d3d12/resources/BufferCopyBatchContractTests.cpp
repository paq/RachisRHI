#include "HeapAllocationCounter.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace
{

    constexpr std::uint64_t bufferSize = 64;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] constexpr std::byte sourceValue(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 29U + 7U) & 0xFFU);
    }

} // namespace

TEST_CASE("buffer-copy region batches validate and record atomically",
          "[size:medium][rhi][buffer][transfer][batch][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Buffer-copy batch source",
    });
    auto zeroSource = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Buffer-copy batch zero source",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Buffer-copy batch destination",
    });
    auto sameBuffer = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Buffer-copy batch overlap validation",
    });
    auto renderTarget = device.createTexture({
        .width = 1,
        .height = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Buffer-copy batch rendering-scope validation",
    });
    REQUIRE(source.has_value());
    REQUIRE(zeroSource.has_value());
    REQUIRE(destination.has_value());
    REQUIRE(sameBuffer.has_value());
    REQUIRE(renderTarget.has_value());
    auto renderTargetView = device.createTextureView({
        .texture = *renderTarget,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(renderTargetView.has_value());

    std::array<std::byte, bufferSize> sourceBytes{};
    std::array<std::byte, bufferSize> zeroBytes{};
    for (std::uint64_t index = 0; index < bufferSize; ++index)
    {
        sourceBytes[index] = sourceValue(index);
    }
    REQUIRE(device.writeBuffer(*source, 0, sourceBytes).has_value());
    REQUIRE(device.writeBuffer(*zeroSource, 0, zeroBytes).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *zeroSource,
                                .destination = *destination,
                                .size = bufferSize,
                            })
                .has_value());

    const std::span<const rachis::BufferCopyRegion> emptyRegions;
    CHECK_FALSE(device.copyBuffer(*command, *source, *destination, emptyRegions).has_value());

    const std::array overlappingDestinations{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 48, .size = 8},
        rachis::BufferCopyRegion{.sourceOffset = 8, .destinationOffset = 52, .size = 8},
    };
    CHECK_FALSE(device.copyBuffer(*command, *source, *destination, overlappingDestinations).has_value());

    const std::array sameBufferCrossOverlap{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 16, .size = 8},
        rachis::BufferCopyRegion{.sourceOffset = 16, .destinationOffset = 32, .size = 8},
    };
    CHECK_FALSE(device.copyBuffer(*command, *sameBuffer, *sameBuffer, sameBufferCrossOverlap).has_value());
    const std::array sameBufferSingleOverlap{
        rachis::BufferCopyRegion{.sourceOffset = 8, .destinationOffset = 12, .size = 8},
    };
    CHECK_FALSE(device.copyBuffer(*command, *sameBuffer, *sameBuffer, sameBufferSingleOverlap).has_value());

    const std::array partiallyInvalid{
        rachis::BufferCopyRegion{.sourceOffset = 48, .destinationOffset = 56, .size = 8},
        rachis::BufferCopyRegion{.sourceOffset = 60, .destinationOffset = 0, .size = 8},
    };
    CHECK_FALSE(device.copyBuffer(*command, *source, *destination, partiallyInvalid).has_value());

    const std::array validRegions{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 8, .size = 8},
        rachis::BufferCopyRegion{.sourceOffset = 16, .destinationOffset = 24, .size = 8},
        rachis::BufferCopyRegion{.sourceOffset = 32, .destinationOffset = 40, .size = 8},
    };
    REQUIRE(device.copyBuffer(*command, *source, *destination, validRegions).has_value());

    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *renderTarget,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = colorAttachments}).has_value());
    const std::array insideRendering{
        rachis::BufferCopyRegion{.sourceOffset = 56, .destinationOffset = 56, .size = 8},
    };
    const auto renderingCopy = device.copyBuffer(*command, *source, *destination, insideRendering);
    REQUIRE_FALSE(renderingCopy.has_value());
    CHECK(renderingCopy.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device.destroyBuffer(*source).has_value());
    REQUIRE(device.destroyBuffer(*zeroSource).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*destination, 0, actual).has_value());
    std::array<std::byte, bufferSize> expected{};
    for (const auto& region : validRegions)
    {
        for (std::uint64_t byteIndex = 0; byteIndex < region.size; ++byteIndex)
        {
            expected[region.destinationOffset + byteIndex] = sourceValue(region.sourceOffset + byteIndex);
        }
    }
    CHECK(actual == expected);

    REQUIRE(device.destroyTextureView(*renderTargetView).has_value());
    REQUIRE(device.destroyTexture(*renderTarget).has_value());
    REQUIRE(device.destroyBuffer(*sameBuffer).has_value());
    REQUIRE(device.destroyBuffer(*destination).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("single-region buffer copies record without per-copy scratch allocations",
          "[size:medium][rhi][buffer][transfer][batch][warp]")
{
    if (!rachis::test::ThreadHeapAllocationCounter::available())
    {
        SKIP("Heap allocations are counted through the MSVC debug heap.");
    }
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Allocation-counted copy source",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Allocation-counted copy destination",
    });
    auto scratch = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Allocation-counted same-buffer copy",
    });
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());
    REQUIRE(scratch.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *scratch,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copySource | rachis::Access::copyDestination,
                               })
                .has_value());

    const rachis::BufferCopy betweenBuffers{
        .source = *source,
        .destination = *destination,
        .size = bufferSize / 2,
    };
    const std::array withinBuffer{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = bufferSize / 2, .size = bufferSize / 2},
    };
    constexpr std::size_t iterations = 64;
    std::size_t recorded = 0;
    const auto recordBetweenBuffers = [&] {
        for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        {
            recorded += device.copyBuffer(*command, betweenBuffers).has_value() ? 1U : 0U;
        }
    };
    const auto recordWithinBuffer = [&] {
        for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        {
            recorded += device.copyBuffer(*command, *scratch, *scratch, withinBuffer).has_value() ? 1U : 0U;
        }
    };
    recordBetweenBuffers();
    recordWithinBuffer();

    rachis::test::ThreadHeapAllocationCounter counter;
    const auto betweenAllocations = counter.count(recordBetweenBuffers);
    const auto withinAllocations = counter.count(recordWithinBuffer);
    REQUIRE(recorded == 4 * iterations);

    constexpr std::size_t retentionGrowthAllowance = 8;
    CAPTURE(betweenAllocations, withinAllocations);
    CHECK(betweenAllocations <= iterations + retentionGrowthAllowance);
    CHECK(withinAllocations <= iterations + retentionGrowthAllowance);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    REQUIRE(device.destroyBuffer(*scratch).has_value());
    REQUIRE(device.destroyBuffer(*destination).has_value());
    REQUIRE(device.destroyBuffer(*source).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
