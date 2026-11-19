#pragma once

#include "Common/MonotonicIdentitySource.h"
#include "Rachis/Diagnostics.h"
#include "Rachis/Error.h"
#include "Rachis/rachis.h"

#include <cstddef>
#include <cstdint>

namespace rachis::cabi::detail
{

    inline constexpr std::uint64_t maxCAbiTokenTag = (std::uint64_t{1} << 40) - 1;

    [[nodiscard]] inline std::uint64_t allocateCAbiToken(rachis::detail::MonotonicIdentitySource& source) noexcept
    {
        const auto id = source.acquire();
        return !id.has_value() || *id > maxCAbiTokenTag ? 0 : *id;
    }

#ifdef RACHIS_TESTING
    void setTokenSourceOverrideForTesting(rachis::detail::MonotonicIdentitySource* overrideSource) noexcept;

    using CommandRecordingHook = void (*)(rachis_handle command, void* userData) noexcept;
    void setCommandRecordingHookForTesting(CommandRecordingHook hook, void* userData) noexcept;

    using SwapchainPresentHook = void (*)(rachis_handle swapchain, void* userData) noexcept;
    void setSwapchainPresentHookForTesting(SwapchainPresentHook hook, void* userData) noexcept;

    using TimelineWaitHook = void (*)(rachis_handle timeline, void* userData) noexcept;
    void setTimelineWaitHookForTesting(TimelineWaitHook hook, void* userData) noexcept;

    enum class NativeCall : std::uint8_t
    {
        createObject,
        destroyRetiredTexture,
    };

    using NativeCallHook = rachis::Status (*)(NativeCall call, void* userData);
    void setNativeCallHookForTesting(NativeCallHook hook, void* userData) noexcept;

    [[nodiscard]] bool deviceLockAdmitsSharedCallersForTesting(rachis_device* device) noexcept;
    [[nodiscard]] std::size_t liveTokenCountForTesting(rachis_device* device);
    void emitDiagnosticForTesting(rachis_device* device, const rachis::DiagnosticMessage& message) noexcept;
#endif

} // namespace rachis::cabi::detail
