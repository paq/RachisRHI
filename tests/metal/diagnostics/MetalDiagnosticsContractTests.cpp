#include "Internal/MetalDiagnostics.h"
#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    struct DiagnosticCapture final
    {
        std::atomic<std::uint64_t> warnings{0};
        std::atomic<std::uint64_t> errors{0};
        std::atomic<std::uint64_t> information{0};
        std::atomic<std::uint64_t> corruptions{0};
    };

#if RACHIS_ENABLE_METAL
    void captureDiagnostic(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        auto* capture = static_cast<DiagnosticCapture*>(userData);
        if (capture == nullptr)
        {
            return;
        }
        switch (message.severity)
        {
        case rachis::DiagnosticSeverity::information:
            capture->information.fetch_add(1, std::memory_order_relaxed);
            break;
        case rachis::DiagnosticSeverity::warning:
            capture->warnings.fetch_add(1, std::memory_order_relaxed);
            break;
        case rachis::DiagnosticSeverity::error:
            capture->errors.fetch_add(1, std::memory_order_relaxed);
            break;
        case rachis::DiagnosticSeverity::corruption:
            capture->corruptions.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }

    [[nodiscard]] rachis::DeviceDesc metalDesc(rachis::ValidationLevel validation,
                                               bool requireValidation,
                                               rachis::DiagnosticCallback callback = nullptr,
                                               void* userData = nullptr) noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = validation,
            .requireValidation = requireValidation,
            .enableDred = false,
            .diagnosticCallback = callback,
            .diagnosticUserData = userData,
        };
    }
#endif

} // namespace

#if RACHIS_ENABLE_METAL
TEST_CASE("Metal NSError mapping retains domain, code and encoder context", "[size:small][rhi][metal][diagnostics]")
{
    const auto error = rachis::metal::makeMetalError(rachis::ErrorCode::validationError,
                                                     2,
                                                     "MTLLogStateErrorDomain",
                                                     "newLogStateWithDescriptor",
                                                     "invalid",
                                                     "blit");
    CHECK(error.code == rachis::ErrorCode::validationError);
    CHECK(error.nativeCode == 2);
    CHECK(error.message.contains("newLogStateWithDescriptor"));
    CHECK(error.message.contains("MTLLogStateErrorDomain"));
    CHECK(error.message.contains("encoder='blit'"));
    CHECK(error.message.contains("invalid"));
}

#endif

TEST_CASE("Metal validation policy is independent of device availability", "[size:small][rhi][metal][diagnostics]")
{
    CHECK(rachis::planMetalValidation(rachis::ValidationLevel::disabled, false, false) ==
          rachis::MetalValidationPlan::disabled);
    CHECK(rachis::planMetalValidation(rachis::ValidationLevel::disabled, false, true) ==
          rachis::MetalValidationPlan::disabled);
    CHECK(rachis::planMetalValidation(rachis::ValidationLevel::standard, false, true) ==
          rachis::MetalValidationPlan::enableLogging);
    CHECK(rachis::planMetalValidation(rachis::ValidationLevel::gpu, true, true) ==
          rachis::MetalValidationPlan::enableLogging);
    CHECK(rachis::planMetalValidation(rachis::ValidationLevel::standard, true, false) ==
          rachis::MetalValidationPlan::requiredUnavailable);
    CHECK(rachis::planMetalValidation(rachis::ValidationLevel::standard, false, false) ==
          rachis::MetalValidationPlan::disabled);
}

#if RACHIS_ENABLE_METAL
TEST_CASE("Metal diagnostic sink ignores callbacks after destruction and stays thread-safe",
          "[size:small][rhi][metal][diagnostics]")
{
    rachis::metal::MetalDiagnosticSink sink;
    sink.publish(rachis::DiagnosticSeverity::warning, 1, "before");
    CHECK(sink.counts().warnings == 1);

    sink.alive.store(false, std::memory_order_release);
    sink.publish(rachis::DiagnosticSeverity::error, 2, "after death");
    CHECK(sink.counts().warnings == 1);
    CHECK(sink.counts().errors == 0);

    sink.alive.store(true, std::memory_order_release);
    std::thread first([&] {
        for (int i = 0; i < 64; ++i)
        {
            sink.publish(rachis::DiagnosticSeverity::warning, 3, "threaded warning");
        }
    });
    std::thread second([&] {
        for (int i = 0; i < 64; ++i)
        {
            sink.publish(rachis::DiagnosticSeverity::error, 4, "threaded error");
        }
    });
    first.join();
    second.join();
    CHECK(sink.counts().warnings == 65);
    CHECK(sink.counts().errors == 64);
}

TEST_CASE("Metal diagnostic callbacks preserve short messages and bound oversized messages",
          "[size:small][rhi][metal][diagnostics]")
{
    struct CapturedMessage
    {
        std::array<char, rachis::maxMetalDiagnosticDescriptionBytes> bytes{};
        std::size_t size = 0;
        std::uint32_t nativeId = 0;
    } captured;
    rachis::metal::MetalDiagnosticSink sink;
    sink.userData = &captured;
    sink.callback = [](const rachis::DiagnosticMessage& message, void* userData) noexcept {
        auto& output = *static_cast<CapturedMessage*>(userData);
        output.size = message.description.size();
        output.nativeId = message.nativeId;
        std::copy_n(message.description.begin(), std::min(output.size, output.bytes.size()), output.bytes.begin());
    };

    sink.publish(rachis::DiagnosticSeverity::warning, 17, "short diagnostic");
    REQUIRE(captured.size <= captured.bytes.size());
    CHECK(std::string_view(captured.bytes.data(), captured.size) == "short diagnostic");
    CHECK(captured.nativeId == 17);

    const std::string oversized(rachis::maxMetalDiagnosticDescriptionBytes + 32, 'x');
    sink.publish(rachis::DiagnosticSeverity::error, 18, oversized);
    REQUIRE(captured.size == captured.bytes.size());
    const std::string_view bounded(captured.bytes.data(), captured.size);
    CHECK(bounded.ends_with("..."));
    CHECK(bounded.substr(0, bounded.size() - 3) == std::string_view(oversized).substr(0, bounded.size() - 3));
    CHECK(captured.nativeId == 18);
    CHECK(sink.counts().warnings == 1);
    CHECK(sink.counts().errors == 1);
}

TEST_CASE("required Metal validation fails clearly and optional absence still creates a device",
          "[size:large][rhi][metal][diagnostics]")
{
    {
        rachis::metal::ScopedFailNextLogStateCreate const failLogState;
        auto required = rachis::createDevice(metalDesc(rachis::ValidationLevel::standard, true));
        CHECK_FALSE(required.has_value());
        CHECK(required.error().code == rachis::ErrorCode::validationError);
        CHECK(required.error().message.contains("MTLLogState"));
    }

    {
        rachis::metal::ScopedFailNextLogStateCreate const failLogState;
        auto optional = rachis::createDevice(metalDesc(rachis::ValidationLevel::standard, false));
        REQUIRE(optional.has_value());
        const auto config = rachis::metal::boundValidationConfiguration();
        CHECK(config.plan == rachis::MetalValidationPlan::disabled);
        CHECK_FALSE(config.logStateEnabled);
    }
}

TEST_CASE("validation-disabled Metal production mode does not require debug tools",
          "[size:large][rhi][metal][diagnostics]")
{
    auto created = rachis::createDevice(metalDesc(rachis::ValidationLevel::disabled, false));
    REQUIRE(created.has_value());
    REQUIRE(created->isValid());
    CHECK(created->diagnosticCounts().errors == 0);
    CHECK(created->diagnosticCounts().corruptions == 0);
    const auto checkedCapabilities = created->capabilities();
    REQUIRE(checkedCapabilities.has_value());
    CHECK(checkedCapabilities->debugMarkers);

    const auto config = rachis::metal::boundValidationConfiguration();
    CHECK(config.plan == rachis::MetalValidationPlan::disabled);
    CHECK_FALSE(config.logStateEnabled);
}

TEST_CASE("valid Metal bootstrap produces no unexplained diagnostics", "[size:large][rhi][metal][diagnostics]")
{
    DiagnosticCapture capture;
    auto created =
        rachis::createDevice(metalDesc(rachis::ValidationLevel::standard, false, &captureDiagnostic, &capture));
    REQUIRE(created.has_value());
    const auto checkedAdapter = created->adapterInfo();
    REQUIRE(checkedAdapter.has_value());
    CHECK(checkedAdapter->backend == rachis::Backend::metal);
    CHECK(created->diagnosticCounts().errors == 0);
    CHECK(created->diagnosticCounts().corruptions == 0);
    CHECK(capture.errors.load() == 0);

    const auto config = rachis::metal::boundValidationConfiguration();
    CHECK(config.plan == rachis::MetalValidationPlan::enableLogging);
    CHECK(config.logStateEnabled);

    auto report = created->captureDeviceFaultReport();
    REQUIRE(report);
    CHECK(report->status == rachis::DeviceFaultStatus::healthy);
    CHECK(report->breadcrumbs.empty());
    CHECK(report->allocations.empty());
}

TEST_CASE("intentional invalid Metal shader usage produces attributed diagnostics",
          "[size:large][rhi][metal][diagnostics]")
{
    DiagnosticCapture capture;
    auto created =
        rachis::createDevice(metalDesc(rachis::ValidationLevel::standard, false, &captureDiagnostic, &capture));
    REQUIRE(created.has_value());

    const auto before = created->diagnosticCounts();
    auto failed = rachis::metal::emitIntentionalValidationFailure();
    REQUIRE_FALSE(failed);
    CHECK(failed.error().code == rachis::ErrorCode::validationError);
    CHECK(failed.error().message.contains("newLibraryWithSource"));
    CHECK(failed.error().nativeCode != 0);

    const auto after = created->diagnosticCounts();
    const bool attributed = after.errors > before.errors || after.warnings > before.warnings ||
                            capture.errors.load() > 0 || capture.warnings.load() > 0;
    CHECK(attributed);
}

#endif
