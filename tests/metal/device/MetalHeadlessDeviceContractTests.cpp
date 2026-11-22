#include "Internal/MetalMacosProfile.h"
#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>

#include <optional>
#include <string>

namespace
{

#if RACHIS_ENABLE_METAL
    [[nodiscard]] rachis::MetalHostReport completeHost() noexcept
    {
        return {
            .macosMajor = 26,
            .macosMinor = 6,
            .appleSiliconArm64 = true,
            .intelMac = false,
            .hasDevice = true,
            .translationLayer = false,
            .metal3Family = true,
            .metal4Family = true,
            .classicCommandQueue = true,
            .metal4CommandQueue = true,
            .metal4CommandAllocator = true,
            .metal4CommandBuffer = true,
            .formatRgba8Unorm = true,
            .formatBgra8Unorm = true,
            .formatD32Float = true,
            .maxBufferLength = 1ULL << 32,
            .recommendedMaxWorkingSetSize = 24ULL << 30,
            .maxTexture2D = 16384,
        };
    }

    using HostOverrideGuard = rachis::metal::ScopedHostReportOverride;
#endif

} // namespace

#if RACHIS_ENABLE_METAL

TEST_CASE("headless Metal devices report identity, limits, and core formats", "[size:large][rhi][metal][device]")
{
    HostOverrideGuard const restore;
    auto adapters = rachis::enumerateAdapters(rachis::Backend::metal);
    REQUIRE(adapters.has_value());
    REQUIRE_FALSE(adapters->empty());

    rachis::AdapterId id{};
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        REQUIRE(created.has_value());
        REQUIRE(created->isValid());
        REQUIRE(created->operationalState() == rachis::DeviceOperationalState::active);
        REQUIRE(created->adapterInfo().has_value());
        REQUIRE(created->capabilities().has_value());

        const auto adapter = *created->adapterInfo();
        CHECK(adapter.backend == rachis::Backend::metal);
        CHECK(adapter.id.value != 0);
        CHECK(adapter.vendorId == 0x106B);
        if (i == 0)
        {
            id = adapter.id;
        }
        else
        {
            CHECK(adapter.id == id);
        }

        const auto capabilities = *created->capabilities();
        CHECK(capabilities.unifiedMemory);
        CHECK(capabilities.cacheCoherentUnifiedMemory);
        CHECK(capabilities.limits.maxTextureDimension2D >= rachis::metalMacosProfileMinMaxTexture2D);
        CHECK(capabilities.limits.maxBufferSize >= rachis::metalMacosProfileMinBufferBytes);
        CHECK(capabilities.limits.maxBindingSets >= 4);
        CHECK(capabilities.limits.maxPushConstantBytes >= 128);
        CHECK(capabilities.limits.maxColorAttachments >= 8);
        CHECK(capabilities.limits.maxComputeWorkgroupInvocations >= 256);

        const auto graphics = created->queueProperties(rachis::QueueType::graphics);
        CHECK(graphics.timestampQueries);
        CHECK(graphics.inlineTimestampWrites);
        CHECK(graphics.timestampValidBits == 64);
        CHECK(graphics.timestampFrequency > 0);

        const auto color = created->queryFormatProperties(rachis::Format::rgba8Unorm);
        CHECK(rachis::hasAny(color.features, rachis::FormatFeature::colorAttachment));
        CHECK(rachis::hasAny(color.features, rachis::FormatFeature::sampled));
        CHECK(rachis::supportsSampleCount(color.sampleCounts, 1));

        const auto bgra = created->queryFormatProperties(rachis::Format::bgra8Unorm);
        CHECK(rachis::hasAny(bgra.features, rachis::FormatFeature::colorAttachment));

        const auto depth = created->queryFormatProperties(rachis::Format::d32Float);
        CHECK(rachis::hasAny(depth.features, rachis::FormatFeature::depthStencilAttachment));

        auto budgets = created->queryMemoryBudgets();
        REQUIRE(budgets.has_value());
        REQUIRE_FALSE(budgets->empty());
        CHECK((*budgets)[0].budgetBytes >= rachis::metalMacosProfileMinWorkingSetBytes);
        CHECK_FALSE(created->queryMemoryBudgets(capabilities.nodeCount).has_value());

        auto buffer = created->createBuffer({});
        REQUIRE_FALSE(buffer.has_value());
        CHECK(buffer.error().code == rachis::ErrorCode::invalidArgument);

        REQUIRE(created->waitIdle().has_value());
    }

    auto explicitDevice = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::metal,
        .adapterId = id,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = false,
    });
    REQUIRE(explicitDevice.has_value());
    const auto checkedAdapter = explicitDevice->adapterInfo();
    REQUIRE(checkedAdapter.has_value());
    CHECK(checkedAdapter->id == id);
}

TEST_CASE("Metal device create rejects injected missing features and releases partial objects",
          "[size:large][rhi][metal][device]")
{
    {
        auto missingFamily = completeHost();
        missingFamily.metal4Family = false;
        HostOverrideGuard const guard(missingFamily);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        REQUIRE_FALSE(created.has_value());
        CHECK(created.error().code == rachis::ErrorCode::unsupported);
        CHECK(created.error().message.contains("MTLGPUFamilyMetal4"));
    }

    {
        auto missingFormat = completeHost();
        missingFormat.formatD32Float = false;
        HostOverrideGuard const guard(missingFormat);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        REQUIRE_FALSE(created.has_value());
        CHECK(created.error().code == rachis::ErrorCode::unsupported);
        CHECK(created.error().message.contains("d32Float"));
    }

    {
        // A command queue that stays nil after the bounded retry is transient, not missing support.
        rachis::metal::ScopedFailCommandApiCreate const failCommandApi;
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        REQUIRE_FALSE(created.has_value());
        CHECK(created.error().code == rachis::ErrorCode::notReady);
        CHECK(created.error().message.starts_with("Retry device creation later: "));
        CHECK(created.error().message.contains("MTL4CommandQueue returned nil"));
        CHECK(created.error().message.ends_with("Metal gave no NSError."));
    }

    {
        const std::string nativeDescription(4096, 'n');
        rachis::metal::ScopedFailCommandApiCreate const failCommandApi(UINT32_MAX, nativeDescription);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        REQUIRE_FALSE(created.has_value());
        CHECK(created.error().code == rachis::ErrorCode::notReady);
        CHECK(created.error().nativeCode == 1);
        CHECK(created.error().message.starts_with("Retry device creation later: "));
        CHECK(created.error().message.contains("Native detail: TestInjectedError code 1: nnn"));
        CHECK(created.error().message.size() <= rachis::maxMetalDiagnosticDescriptionBytes);
        CHECK(created.error().message.ends_with("..."));
    }

    {
        rachis::metal::ScopedFailCommandApiCreate const failCommandApi;
        rachis::metal::ScopedFailSharedEventProbe const failSharedEvent;
        rachis::metal::ScopedSharedEventCount const fullTable(65535);
        const auto started = std::chrono::steady_clock::now();
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        const auto elapsed = std::chrono::steady_clock::now() - started;
        REQUIRE_FALSE(created.has_value());
        CHECK(created.error().code == rachis::ErrorCode::resourceExhausted);
        CHECK(created.error().message.contains("shared-event table is full"));
        CHECK(created.error().message.contains("(65535 of 65535 in use)"));
        CHECK(elapsed < std::chrono::milliseconds(200));
    }

    {
        rachis::metal::ScopedFailCommandApiCreate const failCommandApi;
        rachis::metal::ScopedFailSharedEventProbe const failSharedEvent;
        rachis::metal::ScopedSharedEventCount const roomyTable(1024);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        REQUIRE_FALSE(created.has_value());
        CHECK(created.error().code == rachis::ErrorCode::notReady);
        CHECK(created.error().message.starts_with("Retry device creation later: "));
        CHECK_FALSE(created.error().message.contains("shared-event table is full"));
    }

    {
        rachis::metal::ScopedFailCommandApiCreate const failCommandApi;
        rachis::metal::ScopedSharedEventCount const nearlyFullTable(65500);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        REQUIRE_FALSE(created.has_value());
        CHECK(created.error().code == rachis::ErrorCode::notReady);
        CHECK(created.error().message.starts_with("Retry device creation later: "));
    }

    {
        rachis::metal::ScopedFailCommandApiCreate const failCommandApi;
        rachis::metal::ScopedFailSharedEventProbe const failSharedEvent;
        rachis::metal::ScopedSharedEventCount const unreadableTable(std::nullopt);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        REQUIRE_FALSE(created.has_value());
        CHECK(created.error().code == rachis::ErrorCode::notReady);
    }

    {
        // A queue that is nil for a couple of attempts is absorbed by the retry.
        rachis::metal::ScopedFailCommandApiCreate const failTwice(2);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        CHECK(created.has_value());
    }

    {
        auto missingFamily = completeHost();
        missingFamily.metal4Family = false;
        rachis::metal::ScopedHostReportOverride const unconsumedHost(missingFamily);
        rachis::metal::ScopedFailCommandApiCreate const unconsumedCommandApi;
        rachis::metal::ScopedFailNextLogStateCreate const unconsumedLogState;
    }

    auto recovered = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::metal,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = true,
        .enableDred = false,
    });
    REQUIRE(recovered.has_value());
}

#endif
