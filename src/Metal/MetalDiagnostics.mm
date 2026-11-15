#import <Foundation/Foundation.h>

#include "Internal/MetalDiagnostics.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>

namespace rachis::metal
{
    namespace
    {

        [[nodiscard]] bool envFlagEnabled(const char* name) noexcept
        {
            const char* const value = std::getenv(name);
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }

    } // namespace

    void MetalDiagnosticSink::publish(DiagnosticSeverity severity,
                                      std::uint32_t nativeId,
                                      std::string_view description) noexcept
    {
        if (!alive.load(std::memory_order_acquire))
        {
            return;
        }

        switch (severity)
        {
        case DiagnosticSeverity::information:
            information.fetch_add(1, std::memory_order_relaxed);
            break;
        case DiagnosticSeverity::warning:
            warnings.fetch_add(1, std::memory_order_relaxed);
            break;
        case DiagnosticSeverity::error:
            errors.fetch_add(1, std::memory_order_relaxed);
            break;
        case DiagnosticSeverity::corruption:
            corruptions.fetch_add(1, std::memory_order_relaxed);
            break;
        }

        std::array<char, maxMetalDiagnosticDescriptionBytes> storage{};
        std::string_view bounded = description;
        if (description.size() > storage.size())
        {
            std::copy_n(description.begin(), storage.size() - 3, storage.begin());
            std::fill_n(storage.end() - 3, 3, '.');
            bounded = std::string_view(storage.data(), storage.size());
        }
        const DiagnosticMessage message{
            .severity = severity,
            .nativeId = nativeId,
            .description = bounded,
        };
        const std::scoped_lock lock(mutex);
        if (!alive.load(std::memory_order_acquire) || callback == nullptr)
        {
            return;
        }
        callback(message, userData);
    }

    DiagnosticCounts MetalDiagnosticSink::counts() const noexcept
    {
        return DiagnosticCounts{
            .information = information.load(std::memory_order_relaxed),
            .warnings = warnings.load(std::memory_order_relaxed),
            .errors = errors.load(std::memory_order_relaxed),
            .corruptions = corruptions.load(std::memory_order_relaxed),
        };
    }

    Error makeMetalError(ErrorCode code,
                         std::int64_t nativeCode,
                         std::string_view domain,
                         std::string_view operation,
                         std::string_view nativeDescription,
                         std::string_view encoderContext)
    {
        std::string message;
        message.reserve(operation.size() + domain.size() + nativeDescription.size() + encoderContext.size() + 96);
        message += operation;
        message += " failed";
        if (!domain.empty() || nativeCode != 0)
        {
            message += " with ";
            if (!domain.empty())
            {
                message += domain;
                message += " code ";
            }
            message += std::to_string(nativeCode);
        }
        if (!encoderContext.empty())
        {
            message += " encoder='";
            message += encoderContext;
            message += '\'';
        }
        if (!nativeDescription.empty())
        {
            message += ": ";
            message += nativeDescription;
        }
        message += '.';
        return Error{
            code,
            nativeCode,
            truncateMetalDiagnostic(message),
        };
    }

    bool metalDebugLayerEnvironmentEnabled() noexcept
    {
        return envFlagEnabled("MTL_DEBUG_LAYER") || envFlagEnabled("METAL_DEVICE_WRAPPER_TYPE");
    }

    bool metalShaderValidationEnvironmentEnabled() noexcept
    {
        return envFlagEnabled("MTL_SHADER_VALIDATION");
    }

    std::string truncateMetalDiagnostic(std::string_view text)
    {
        if (text.size() <= maxMetalDiagnosticDescriptionBytes)
        {
            return std::string(text);
        }
        std::string bounded(text.substr(0, maxMetalDiagnosticDescriptionBytes - 3));
        bounded += "...";
        return bounded;
    }

} // namespace rachis::metal
