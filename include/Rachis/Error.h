#pragma once

#include <cstdint>
#include <expected>
#include <string>

namespace rachis
{

    enum class ErrorCode : std::uint8_t
    {
        invalidArgument,
        unsupported,
        backendUnavailable,
        adapterNotFound,
        deviceCreationFailed,
        resourceExhausted,
        invalidState,
        deviceLost,
        outOfMemory,
        nativeFailure,
        validationError,
        pipelineCacheMiss,

        notReady,
        timeout,
    };

    struct Error final
    {
        ErrorCode code = ErrorCode::nativeFailure;
        std::int64_t nativeCode = 0;
        std::string message = {};
    };

    template<typename T>
    using Result = std::expected<T, Error>;

    using Status = Result<void>;

} // namespace rachis
