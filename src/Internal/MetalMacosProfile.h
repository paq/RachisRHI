#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace rachis
{

    inline constexpr std::uint16_t metalMacosProfileOsMajor = 26;
    inline constexpr std::uint16_t metalMacosProfileOsMinor = 0;
    inline constexpr std::uint64_t metalMacosProfileMinBufferBytes = 256ULL * 1024ULL * 1024ULL;
    inline constexpr std::uint64_t metalMacosProfileMinWorkingSetBytes = 512ULL * 1024ULL * 1024ULL;
    inline constexpr std::uint32_t metalMacosProfileMinMaxTexture2D = 8192;

    enum class MetalCommandApi : std::uint8_t
    {
        none,
        metal3CommandQueue,
        metal4CommandQueue,
    };

    inline constexpr MetalCommandApi metalMacosSelectedCommandApi = MetalCommandApi::metal4CommandQueue;

    enum class MetalProfileRejection : std::uint8_t
    {
        none,
        intelArchitecture,
        macosVersionTooLow,
        apiUnavailable,
        deviceUnavailable,
        missingRequiredFamily,
        missingRequiredFormat,
        limitBelowMinimum,
        warpPreferenceIsD3D12Only,
        translationLayer,
    };

    struct MetalHostReport final
    {
        std::uint16_t macosMajor = 0;
        std::uint16_t macosMinor = 0;
        bool appleSiliconArm64 = false;
        bool intelMac = false;
        bool hasDevice = false;
        bool translationLayer = false;
        bool metal3Family = false;
        bool metal4Family = false;
        bool classicCommandQueue = false;
        bool metal4CommandQueue = false;
        bool metal4CommandAllocator = false;
        bool metal4CommandBuffer = false;
        bool formatRgba8Unorm = false;
        bool formatBgra8Unorm = false;
        bool formatD32Float = false;
        std::uint64_t maxBufferLength = 0;
        std::uint64_t recommendedMaxWorkingSetSize = 0;
        std::uint32_t maxTexture2D = 0;
    };

    struct MetalCreateIntent final
    {
        bool warpPreference = false;
    };

    struct MetalProfileDecision final
    {
        bool accepted = false;
        MetalProfileRejection rejection = MetalProfileRejection::none;
        MetalCommandApi selectedCommandApi = MetalCommandApi::none;
        std::array<std::string_view, 16> missingItems{};
        std::uint32_t missingItemCount = 0;
    };

    namespace detail
    {

        inline void appendMetalMissing(MetalProfileDecision& decision, std::string_view name) noexcept
        {
            if (decision.missingItemCount < decision.missingItems.size())
            {
                decision.missingItems[decision.missingItemCount++] = name;
            }
        }

    } // namespace detail

    [[nodiscard]] inline std::string_view describeMetalMacosProfileRejection(MetalProfileRejection rejection) noexcept
    {
        switch (rejection)
        {
        case MetalProfileRejection::none:
            return {};
        case MetalProfileRejection::intelArchitecture:
            return "Intel macOS is not a supported Metal host. Apple silicon arm64 is required.";
        case MetalProfileRejection::macosVersionTooLow:
            return "macOS 26.0 or newer is required for the Metal 4 command API.";
        case MetalProfileRejection::apiUnavailable:
            return "Metal 4 command-queue APIs are unavailable. The process did not fall back to Metal 3 recording.";
        case MetalProfileRejection::deviceUnavailable:
            return "No Metal device is present. Missing hardware is not a skipped success.";
        case MetalProfileRejection::missingRequiredFamily:
            return "MTLGPUFamilyMetal4 is required. Metal 3-only GPUs are unsupported.";
        case MetalProfileRejection::missingRequiredFormat:
            return "A required pixel format is missing. The backend does not substitute another format.";
        case MetalProfileRejection::limitBelowMinimum:
            return "A required Metal device limit is below the macOS Apple silicon floor.";
        case MetalProfileRejection::warpPreferenceIsD3D12Only:
            return "AdapterPreference::warp is D3D12-only and is rejected on Metal.";
        case MetalProfileRejection::translationLayer:
            return "Translation-layer adapters are not Metal support.";
        }
        return "Unknown Metal profile rejection.";
    }

    [[nodiscard]] inline std::string_view describeMetalMacosProfileRejection(
        const MetalProfileDecision& decision) noexcept
    {
        return describeMetalMacosProfileRejection(decision.rejection);
    }

    namespace detail
    {

        [[nodiscard]] inline bool rejectUnsupportedMetalHost(const MetalHostReport& host,
                                                             const MetalCreateIntent& intent,
                                                             MetalProfileDecision& decision) noexcept
        {
            if (intent.warpPreference)
            {
                decision.rejection = MetalProfileRejection::warpPreferenceIsD3D12Only;
                detail::appendMetalMissing(decision, "AdapterPreference::warp");
                return true;
            }

            if (host.intelMac || !host.appleSiliconArm64)
            {
                decision.rejection = MetalProfileRejection::intelArchitecture;
                detail::appendMetalMissing(decision, "appleSiliconArm64");
                return true;
            }

            if (host.macosMajor < metalMacosProfileOsMajor ||
                (host.macosMajor == metalMacosProfileOsMajor && host.macosMinor < metalMacosProfileOsMinor))
            {
                decision.rejection = MetalProfileRejection::macosVersionTooLow;
                detail::appendMetalMissing(decision, "macos26");
                return true;
            }

            if (!host.hasDevice)
            {
                decision.rejection = MetalProfileRejection::deviceUnavailable;
                detail::appendMetalMissing(decision, "MTLDevice");
                return true;
            }

            if (host.translationLayer)
            {
                decision.rejection = MetalProfileRejection::translationLayer;
                detail::appendMetalMissing(decision, "translationLayer");
                return true;
            }
            return false;
        }

        [[nodiscard]] inline bool rejectMissingMetal4Support(const MetalHostReport& host,
                                                             MetalProfileDecision& decision) noexcept
        {
            if (!host.metal4Family)
            {
                decision.rejection = MetalProfileRejection::missingRequiredFamily;
                detail::appendMetalMissing(decision, "MTLGPUFamilyMetal4");
                return true;
            }

            if (!host.metal4CommandQueue || !host.metal4CommandAllocator || !host.metal4CommandBuffer)
            {
                decision.rejection = MetalProfileRejection::apiUnavailable;
                if (!host.metal4CommandQueue)
                {
                    detail::appendMetalMissing(decision, "MTL4CommandQueue");
                }
                if (!host.metal4CommandAllocator)
                {
                    detail::appendMetalMissing(decision, "MTL4CommandAllocator");
                }
                if (!host.metal4CommandBuffer)
                {
                    detail::appendMetalMissing(decision, "MTL4CommandBuffer");
                }
                return true;
            }
            return false;
        }

        [[nodiscard]] inline bool rejectMissingRequiredFormats(const MetalHostReport& host,
                                                               MetalProfileDecision& decision) noexcept
        {
            if (!host.formatRgba8Unorm)
            {
                decision.rejection = MetalProfileRejection::missingRequiredFormat;
                detail::appendMetalMissing(decision, "rgba8Unorm");
                return true;
            }
            if (!host.formatBgra8Unorm)
            {
                decision.rejection = MetalProfileRejection::missingRequiredFormat;
                detail::appendMetalMissing(decision, "bgra8Unorm");
                return true;
            }
            if (!host.formatD32Float)
            {
                decision.rejection = MetalProfileRejection::missingRequiredFormat;
                detail::appendMetalMissing(decision, "d32Float");
                return true;
            }
            return false;
        }

        [[nodiscard]] inline bool rejectLimitsBelowMinimum(const MetalHostReport& host,
                                                           MetalProfileDecision& decision) noexcept
        {
            if (host.maxBufferLength < metalMacosProfileMinBufferBytes)
            {
                decision.rejection = MetalProfileRejection::limitBelowMinimum;
                detail::appendMetalMissing(decision, "maxBufferLength");
                return true;
            }
            if (host.recommendedMaxWorkingSetSize < metalMacosProfileMinWorkingSetBytes)
            {
                decision.rejection = MetalProfileRejection::limitBelowMinimum;
                detail::appendMetalMissing(decision, "recommendedMaxWorkingSetSize");
                return true;
            }
            if (host.maxTexture2D < metalMacosProfileMinMaxTexture2D)
            {
                decision.rejection = MetalProfileRejection::limitBelowMinimum;
                detail::appendMetalMissing(decision, "maxTexture2D");
                return true;
            }
            return false;
        }

    } // namespace detail

    [[nodiscard]] inline MetalProfileDecision evaluateMacosMetalProfile(const MetalHostReport& host,
                                                                        const MetalCreateIntent& intent = {}) noexcept
    {
        MetalProfileDecision decision{};
        decision.selectedCommandApi = metalMacosSelectedCommandApi;

        if (detail::rejectUnsupportedMetalHost(host, intent, decision) ||
            detail::rejectMissingMetal4Support(host, decision) ||
            detail::rejectMissingRequiredFormats(host, decision) || detail::rejectLimitsBelowMinimum(host, decision))
        {
            return decision;
        }

        decision.accepted = true;
        return decision;
    }

} // namespace rachis
