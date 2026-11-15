#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstdio>

namespace
{

    constexpr std::uint64_t bufferSize = 256;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

} // namespace

TEST_CASE("D3D12 reports and enforces its whole-buffer barrier boundary", "[size:medium][rhi][barrier][range][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);
    CHECK_FALSE(rachis::test::requireCapabilities(device).bufferBarrierRanges);

    auto first = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Buffer range capability first",
    });
    auto second = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Buffer range capability second",
    });
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command.has_value());
    const std::array unsupportedBatch{
        rachis::BufferBarrier{
            .buffer = *first,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copySource,
            .size = bufferSize / 2,
        },
        rachis::BufferBarrier{
            .buffer = *second,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    const auto unsupported = device.resourceBarriers(*command, {.bufferBarriers = unsupportedBatch});
    REQUIRE_FALSE(unsupported.has_value());
    CHECK(unsupported.error().code == rachis::ErrorCode::unsupported);

    auto remainingRange = unsupportedBatch[0];
    remainingRange.offset = bufferSize / 2;
    remainingRange.size = rachis::wholeResource;
    const auto remaining = device.bufferBarrier(*command, remainingRange);
    REQUIRE_FALSE(remaining.has_value());
    CHECK(remaining.error().code == rachis::ErrorCode::unsupported);

    const std::array wholeBatch{
        rachis::BufferBarrier{
            .buffer = *first,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copySource,
        },
        unsupportedBatch[1],
    };
    REQUIRE(device.resourceBarriers(*command, {.bufferBarriers = wholeBatch}).has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *first,
                                .destination = *second,
                                .size = bufferSize,
                            })
                .has_value());

    REQUIRE(device.destroyBuffer(*first).has_value());
    REQUIRE(device.destroyBuffer(*second).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
