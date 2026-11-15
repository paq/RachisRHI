#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>

namespace
{

    struct MemoryTotals final
    {
        std::uint64_t blocks = 0;
        std::uint64_t allocations = 0;
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

    [[nodiscard]] MemoryTotals memoryTotals(rachis::Device& device)
    {
        const auto budgets = device.queryMemoryBudgets();
        REQUIRE(budgets.has_value());

        MemoryTotals result;
        for (const auto& budget : *budgets)
        {
            result.blocks += budget.blockCount;
            result.allocations += budget.allocationCount;
        }
        return result;
    }

} // namespace

TEST_CASE("explicit residency priorities use isolated lazy allocation pools",
          "[size:medium][rhi][residency][memory][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).residencyPriorities);

    const auto baseline = memoryTotals(device);
    auto invalid = device.createBuffer(rachis::BufferDesc{
        .size = 4'096,
        .usage = rachis::BufferUsage::copySource,
        .residencyPriority = static_cast<rachis::ResidencyPriority>(255),
        .label = "Invalid residency priority",
    });
    REQUIRE_FALSE(invalid.has_value());
    CHECK(invalid.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(memoryTotals(device).blocks == baseline.blocks);

    auto normalBuffer = device.createBuffer(rachis::BufferDesc{
        .size = 4'096,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .residencyPriority = rachis::ResidencyPriority::normal,
        .label = "Normal-priority buffer A",
    });
    REQUIRE(normalBuffer.has_value());
    const auto afterNormal = memoryTotals(device);
    CHECK(afterNormal.blocks == baseline.blocks + 1);

    auto secondNormalBuffer = device.createBuffer(rachis::BufferDesc{
        .size = 4'096,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .residencyPriority = rachis::ResidencyPriority::normal,
        .label = "Normal-priority buffer B",
    });
    REQUIRE(secondNormalBuffer.has_value());
    const auto afterSecondNormal = memoryTotals(device);
    CHECK(afterSecondNormal.blocks == afterNormal.blocks);

    auto lowBuffer = device.createBuffer(rachis::BufferDesc{
        .size = 4'096,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .residencyPriority = rachis::ResidencyPriority::low,
        .label = "Low-priority buffer",
    });
    REQUIRE(lowBuffer.has_value());
    const auto afterLow = memoryTotals(device);
    CHECK(afterLow.blocks == afterSecondNormal.blocks + 1);

    auto highTexture = device.createTexture(rachis::TextureDesc{
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
        .residencyPriority = rachis::ResidencyPriority::high,
        .label = "High-priority sampled texture",
    });
    REQUIRE(highTexture.has_value());
    const auto afterTexture = memoryTotals(device);
    CHECK(afterTexture.blocks == afterLow.blocks + 1);

    auto highAttachment = device.createTexture(rachis::TextureDesc{
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .residencyPriority = rachis::ResidencyPriority::high,
        .label = "High-priority attachment",
    });
    REQUIRE(highAttachment.has_value());
    const auto live = memoryTotals(device);
    CHECK(live.blocks == afterTexture.blocks + 1);
    CHECK(live.allocations == baseline.allocations + 5);

    REQUIRE(device.destroyTexture(*highAttachment).has_value());
    REQUIRE(device.destroyTexture(*highTexture).has_value());
    REQUIRE(device.destroyBuffer(*lowBuffer).has_value());
    REQUIRE(device.destroyBuffer(*secondNormalBuffer).has_value());
    REQUIRE(device.destroyBuffer(*normalBuffer).has_value());
    CHECK(memoryTotals(device).allocations == baseline.allocations);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
