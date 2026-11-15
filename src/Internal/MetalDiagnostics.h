#pragma once

#include "Rachis/Device.h"
#include "Rachis/Diagnostics.h"
#include "Rachis/Error.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>

namespace rachis
{

    inline constexpr std::uint32_t maxMetalDiagnosticDescriptionBytes = 2048;
    inline constexpr std::uint32_t metalLogStateBufferBytes = 64U * 1024U;

    enum class MetalValidationPlan : std::uint8_t
    {
        disabled,
        enableLogging,
        requiredUnavailable,
    };

    struct MetalValidationConfiguration final
    {
        ValidationLevel requested = ValidationLevel::disabled;
        bool requireValidation = false;
        MetalValidationPlan plan = MetalValidationPlan::disabled;
        bool logStateEnabled = false;
        bool debugLayerEnvironment = false;
        bool shaderValidationEnvironment = false;
        bool gpuTraceCaptureSupported = false;
        bool debugMarkers = false;
        std::string deviceLabel;
        std::string queueLabel;
    };

    struct MetalCaptureRequest final
    {
        std::string_view label;
        std::string_view outputDirectory{};
        std::string_view commit{};
    };

    struct MetalCaptureRecord final
    {
        std::string outputPath;
        std::string backend = "metal";
        std::string deviceName;
        std::uint64_t registryId = 0;
        std::string commit;
        std::string workloadLabel;
        std::string deviceLabel;
        std::string queueLabel;
        std::string commandLabel;
        bool captured = false;
        bool toolingUnavailable = false;
    };

    [[nodiscard]] constexpr MetalValidationPlan planMetalValidation(ValidationLevel level,
                                                                    bool requireValidation,
                                                                    bool logStateAvailable) noexcept
    {
        const bool requested = level != ValidationLevel::disabled || requireValidation;
        if (!requested)
        {
            return MetalValidationPlan::disabled;
        }
        if (logStateAvailable)
        {
            return MetalValidationPlan::enableLogging;
        }
        return requireValidation ? MetalValidationPlan::requiredUnavailable : MetalValidationPlan::disabled;
    }

} // namespace rachis

namespace rachis::metal
{

    struct MetalDiagnosticSink final
    {
        std::atomic<bool> alive{true};
        std::mutex mutex;
        DiagnosticCallback callback = nullptr;
        void* userData = nullptr;
        std::atomic<std::uint64_t> information{0};
        std::atomic<std::uint64_t> warnings{0};
        std::atomic<std::uint64_t> errors{0};
        std::atomic<std::uint64_t> corruptions{0};

        void publish(DiagnosticSeverity severity, std::uint32_t nativeId, std::string_view description) noexcept;
        [[nodiscard]] DiagnosticCounts counts() const noexcept;
    };

    [[nodiscard]] Error makeMetalError(ErrorCode code,
                                       std::int64_t nativeCode,
                                       std::string_view domain,
                                       std::string_view operation,
                                       std::string_view nativeDescription = {},
                                       std::string_view encoderContext = {});
    [[nodiscard]] bool metalDebugLayerEnvironmentEnabled() noexcept;
    [[nodiscard]] bool metalShaderValidationEnvironmentEnabled() noexcept;
    [[nodiscard]] std::string truncateMetalDiagnostic(std::string_view text);

} // namespace rachis::metal
