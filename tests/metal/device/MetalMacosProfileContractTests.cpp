#include "Internal/MetalMacosProfile.h"

#include <catch2/catch_test_macros.hpp>

#include <string_view>

namespace
{

    [[nodiscard]] rachis::MetalHostReport appleM4Metal4Host() noexcept
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

} // namespace

TEST_CASE("macOS Metal profile accepts Apple silicon Metal 4 and selects the MTL4 command API",
          "[size:small][rhi][metal][macos][profile]")
{
    const auto decision = rachis::evaluateMacosMetalProfile(appleM4Metal4Host());
    REQUIRE(decision.accepted);
    CHECK(decision.rejection == rachis::MetalProfileRejection::none);
    CHECK(decision.selectedCommandApi == rachis::MetalCommandApi::metal4CommandQueue);
    CHECK(rachis::describeMetalMacosProfileRejection(decision).empty());
}

TEST_CASE("macOS Metal profile distinguishes API, device, family, format, and host failures",
          "[size:small][rhi][metal][macos][profile]")
{
    auto intel = appleM4Metal4Host();
    intel.intelMac = true;
    intel.appleSiliconArm64 = false;
    const auto intelDecision = rachis::evaluateMacosMetalProfile(intel);
    REQUIRE_FALSE(intelDecision.accepted);
    CHECK(intelDecision.rejection == rachis::MetalProfileRejection::intelArchitecture);

    auto oldOs = appleM4Metal4Host();
    oldOs.macosMajor = 15;
    const auto osDecision = rachis::evaluateMacosMetalProfile(oldOs);
    REQUIRE_FALSE(osDecision.accepted);
    CHECK(osDecision.rejection == rachis::MetalProfileRejection::macosVersionTooLow);

    auto noDevice = appleM4Metal4Host();
    noDevice.hasDevice = false;
    const auto deviceDecision = rachis::evaluateMacosMetalProfile(noDevice);
    REQUIRE_FALSE(deviceDecision.accepted);
    CHECK(deviceDecision.rejection == rachis::MetalProfileRejection::deviceUnavailable);

    auto metal3Only = appleM4Metal4Host();
    metal3Only.metal4Family = false;
    const auto familyDecision = rachis::evaluateMacosMetalProfile(metal3Only);
    REQUIRE_FALSE(familyDecision.accepted);
    CHECK(familyDecision.rejection == rachis::MetalProfileRejection::missingRequiredFamily);

    auto noMtl4Queue = appleM4Metal4Host();
    noMtl4Queue.metal4CommandQueue = false;
    const auto apiDecision = rachis::evaluateMacosMetalProfile(noMtl4Queue);
    REQUIRE_FALSE(apiDecision.accepted);
    CHECK(apiDecision.rejection == rachis::MetalProfileRejection::apiUnavailable);

    auto noFormat = appleM4Metal4Host();
    noFormat.formatBgra8Unorm = false;
    const auto formatDecision = rachis::evaluateMacosMetalProfile(noFormat);
    REQUIRE_FALSE(formatDecision.accepted);
    CHECK(formatDecision.rejection == rachis::MetalProfileRejection::missingRequiredFormat);

    auto smallBudget = appleM4Metal4Host();
    smallBudget.recommendedMaxWorkingSetSize = 64ULL << 20;
    const auto limitDecision = rachis::evaluateMacosMetalProfile(smallBudget);
    REQUIRE_FALSE(limitDecision.accepted);
    CHECK(limitDecision.rejection == rachis::MetalProfileRejection::limitBelowMinimum);

    const auto warp =
        rachis::evaluateMacosMetalProfile(appleM4Metal4Host(), rachis::MetalCreateIntent{.warpPreference = true});
    REQUIRE_FALSE(warp.accepted);
    CHECK(warp.rejection == rachis::MetalProfileRejection::warpPreferenceIsD3D12Only);

    CHECK_FALSE(rachis::describeMetalMacosProfileRejection(familyDecision).empty());
}
