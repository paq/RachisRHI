#include "Internal/VulkanValidationHelpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers and pure validation helpers must not include Vulkan headers.
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#if RACHIS_ENABLE_VULKAN && defined(RACHIS_TESTING)
#include "Internal/DeviceAccess.h"
#include "Internal/VulkanAcceleration.h"
#include "Internal/VulkanBindings.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanDevice.h"
#include "Internal/VulkanGraphics.h"
#include "Internal/VulkanImageUtilities.h"
#include "Internal/VulkanPipelines.h"
#include "Internal/VulkanQueries.h"
#include "Internal/VulkanQueues.h"
#include "Internal/VulkanResources.h"
#include "Internal/VulkanRuntime.h"
#include "Rachis/TestSupport/VulkanHostClock.h"
#include "Vulkan/VulkanTestHooks.h"
#endif

TEST_CASE("push-constant planner splits subranges by exact per-word layout stage union",
          "[size:small][rhi][vuid][push-constants]")
{
    using rachis::PushConstantRange;
    using rachis::ShaderStage;
    using rachis::vulkan::validation_detail::mergePushConstantRangesPerStage;
    using rachis::vulkan::validation_detail::planComputePushConstants;
    using rachis::vulkan::validation_detail::planGraphicsPushConstants;
    using rachis::vulkan::validation_detail::planRayTracingPushConstants;
    using rachis::vulkan::validation_detail::PushConstantSubrange;

    const std::array sharedGraphicsLayout{
        PushConstantRange{
            .visibility = ShaderStage::vertex | ShaderStage::pixel,
            .offset = 0,
            .size = 16,
        },
    };
    const auto sharedGraphicsNative = mergePushConstantRangesPerStage(sharedGraphicsLayout);
    const auto narrowed =
        planGraphicsPushConstants(sharedGraphicsLayout, sharedGraphicsNative, ShaderStage::vertex, 0, 8);
    REQUIRE(narrowed.has_value());
    REQUIRE(narrowed->size() == 1);
    CHECK((*narrowed)[0] == PushConstantSubrange{
                                .stages = ShaderStage::vertex | ShaderStage::pixel,
                                .offset = 0,
                                .size = 8,
                            });

    const std::array overlappingGraphicsLayout{
        PushConstantRange{.visibility = ShaderStage::vertex, .offset = 0, .size = 16},
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 8, .size = 16},
    };
    const auto overlappingGraphicsNative = mergePushConstantRangesPerStage(overlappingGraphicsLayout);
    const auto split =
        planGraphicsPushConstants(overlappingGraphicsLayout, overlappingGraphicsNative, ShaderStage::vertex, 0, 16);
    REQUIRE(split.has_value());
    REQUIRE(split->size() == 2);
    CHECK((*split)[0] == PushConstantSubrange{.stages = ShaderStage::vertex, .offset = 0, .size = 8});
    CHECK((*split)[1] == PushConstantSubrange{
                             .stages = ShaderStage::vertex | ShaderStage::pixel,
                             .offset = 8,
                             .size = 8,
                         });

    CHECK_FALSE(
        planGraphicsPushConstants(overlappingGraphicsLayout, overlappingGraphicsNative, ShaderStage::pixel, 0, 16)
            .has_value());
    CHECK_FALSE(
        planGraphicsPushConstants(overlappingGraphicsLayout, overlappingGraphicsNative, ShaderStage::compute, 0, 8)
            .has_value());
    CHECK_FALSE(
        planGraphicsPushConstants(overlappingGraphicsLayout, overlappingGraphicsNative, ShaderStage::vertex, 2, 8)
            .has_value());
    CHECK_FALSE(
        planGraphicsPushConstants(overlappingGraphicsLayout, overlappingGraphicsNative, ShaderStage::vertex, 0, 6)
            .has_value());
    CHECK_FALSE(
        planGraphicsPushConstants(overlappingGraphicsLayout, overlappingGraphicsNative, ShaderStage::vertex, 0, 0)
            .has_value());

    const std::array mixedComputeLayout{
        PushConstantRange{.visibility = ShaderStage::compute | ShaderStage::vertex, .offset = 0, .size = 8},
        PushConstantRange{.visibility = ShaderStage::compute, .offset = 8, .size = 8},
    };
    const auto computePlan =
        planComputePushConstants(mixedComputeLayout, mergePushConstantRangesPerStage(mixedComputeLayout), 0, 16);
    REQUIRE(computePlan.has_value());
    REQUIRE(computePlan->size() == 2);
    CHECK((*computePlan)[0] == PushConstantSubrange{
                                   .stages = ShaderStage::compute | ShaderStage::vertex,
                                   .offset = 0,
                                   .size = 8,
                               });
    CHECK((*computePlan)[1] == PushConstantSubrange{.stages = ShaderStage::compute, .offset = 8, .size = 8});
    CHECK_FALSE(planComputePushConstants(overlappingGraphicsLayout, overlappingGraphicsNative, 0, 8).has_value());

    const std::array rtLayout{
        PushConstantRange{
            .visibility = ShaderStage::rayGeneration | ShaderStage::anyHit | ShaderStage::closestHit,
            .offset = 0,
            .size = 8,
        },
        PushConstantRange{
            .visibility = ShaderStage::miss | ShaderStage::intersection | ShaderStage::callable | ShaderStage::compute,
            .offset = 8,
            .size = 8,
        },
    };
    const auto rtPlan = planRayTracingPushConstants(rtLayout, mergePushConstantRangesPerStage(rtLayout), 0, 16);
    REQUIRE(rtPlan.has_value());
    REQUIRE(rtPlan->size() == 2);
    CHECK((*rtPlan)[0] == PushConstantSubrange{
                              .stages = ShaderStage::rayGeneration | ShaderStage::anyHit | ShaderStage::closestHit,
                              .offset = 0,
                              .size = 8,
                          });
    CHECK((*rtPlan)[1] ==
          PushConstantSubrange{
              .stages = ShaderStage::miss | ShaderStage::intersection | ShaderStage::callable | ShaderStage::compute,
              .offset = 8,
              .size = 8,
          });
    CHECK_FALSE(planRayTracingPushConstants(overlappingGraphicsLayout, overlappingGraphicsNative, 0, 8).has_value());
}

TEST_CASE("push-constant ranges merge per stage so no native range repeats a stage",
          "[size:small][rhi][vuid][push-constants]")
{
    using rachis::PushConstantRange;
    using rachis::ShaderStage;
    using rachis::vulkan::validation_detail::mergePushConstantRangesPerStage;
    using rachis::vulkan::validation_detail::planComputePushConstants;
    using rachis::vulkan::validation_detail::planGraphicsPushConstants;
    using rachis::vulkan::validation_detail::PushConstantSubrange;

    const auto sameRange = [](const PushConstantRange& left, const PushConstantRange& right) {
        return left.visibility == right.visibility && left.offset == right.offset && left.size == right.size;
    };
    const auto checkMerged = [](std::span<const PushConstantRange> layout, std::span<const PushConstantRange> merged) {
        auto seen = ShaderStage::none;
        for (const auto& range : merged)
        {
            CHECK_FALSE(rachis::hasAny(seen, range.visibility));
            seen = seen | range.visibility;
        }
        for (const auto& range : layout)
        {
            for (std::uint32_t bit = 0; bit < 16; ++bit)
            {
                const auto stage = static_cast<ShaderStage>(static_cast<std::uint16_t>(1U << bit));
                if (!rachis::hasAny(range.visibility, stage))
                {
                    continue;
                }
                CHECK(std::ranges::any_of(merged, [&](const PushConstantRange& candidate) {
                    return rachis::hasAny(candidate.visibility, stage) && candidate.offset <= range.offset &&
                           range.offset + range.size <= candidate.offset + candidate.size;
                }));
            }
        }
    };

    const std::array disjointPixel{
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 0, .size = 16},
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 32, .size = 16},
    };
    const auto disjointNative = mergePushConstantRangesPerStage(disjointPixel);
    checkMerged(disjointPixel, disjointNative);
    CHECK(std::ranges::equal(disjointNative,
                             std::array{PushConstantRange{.visibility = ShaderStage::pixel, .offset = 0, .size = 48}},
                             sameRange));
    const auto disjointHigh = planGraphicsPushConstants(disjointPixel, disjointNative, ShaderStage::pixel, 32, 16);
    REQUIRE(disjointHigh.has_value());
    CHECK(*disjointHigh == std::vector{PushConstantSubrange{.stages = ShaderStage::pixel, .offset = 32, .size = 16}});
    CHECK_FALSE(planGraphicsPushConstants(disjointPixel, disjointNative, ShaderStage::pixel, 16, 16).has_value());
    CHECK_FALSE(planGraphicsPushConstants(disjointPixel, disjointNative, ShaderStage::pixel, 0, 48).has_value());

    const std::array computeExtension{
        PushConstantRange{.visibility = ShaderStage::compute | ShaderStage::vertex, .offset = 0, .size = 8},
        PushConstantRange{.visibility = ShaderStage::compute, .offset = 8, .size = 8},
    };
    const auto computeNative = mergePushConstantRangesPerStage(computeExtension);
    checkMerged(computeExtension, computeNative);
    CHECK(std::ranges::equal(computeNative,
                             std::array{
                                 PushConstantRange{.visibility = ShaderStage::vertex, .offset = 0, .size = 8},
                                 PushConstantRange{.visibility = ShaderStage::compute, .offset = 0, .size = 16},
                             },
                             sameRange));
    const auto computeHigh = planComputePushConstants(computeExtension, computeNative, 8, 8);
    REQUIRE(computeHigh.has_value());
    CHECK(*computeHigh == std::vector{PushConstantSubrange{.stages = ShaderStage::compute, .offset = 8, .size = 8}});

    const std::array vertexInPixelGap{
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 0, .size = 16},
        PushConstantRange{.visibility = ShaderStage::vertex, .offset = 16, .size = 16},
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 32, .size = 16},
    };
    const auto gapNative = mergePushConstantRangesPerStage(vertexInPixelGap);
    checkMerged(vertexInPixelGap, gapNative);
    CHECK(std::ranges::equal(gapNative,
                             std::array{
                                 PushConstantRange{.visibility = ShaderStage::vertex, .offset = 16, .size = 16},
                                 PushConstantRange{.visibility = ShaderStage::pixel, .offset = 0, .size = 48},
                             },
                             sameRange));
    const auto gapVertex = planGraphicsPushConstants(vertexInPixelGap, gapNative, ShaderStage::vertex, 16, 16);
    REQUIRE(gapVertex.has_value());
    CHECK(*gapVertex == std::vector{PushConstantSubrange{
                            .stages = ShaderStage::vertex | ShaderStage::pixel,
                            .offset = 16,
                            .size = 16,
                        }});
    const auto gapPixel = planGraphicsPushConstants(vertexInPixelGap, gapNative, ShaderStage::pixel, 32, 16);
    REQUIRE(gapPixel.has_value());
    CHECK(*gapPixel == std::vector{PushConstantSubrange{.stages = ShaderStage::pixel, .offset = 32, .size = 16}});
    CHECK_FALSE(planGraphicsPushConstants(vertexInPixelGap, gapNative, ShaderStage::pixel, 16, 16).has_value());

    const std::array sharedThenPixel{
        PushConstantRange{.visibility = ShaderStage::vertex | ShaderStage::pixel, .offset = 0, .size = 16},
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 16, .size = 16},
    };
    const auto sharedNative = mergePushConstantRangesPerStage(sharedThenPixel);
    checkMerged(sharedThenPixel, sharedNative);
    CHECK(std::ranges::equal(sharedNative,
                             std::array{
                                 PushConstantRange{.visibility = ShaderStage::vertex, .offset = 0, .size = 16},
                                 PushConstantRange{.visibility = ShaderStage::pixel, .offset = 0, .size = 32},
                             },
                             sameRange));
    const auto sharedPlan = planGraphicsPushConstants(sharedThenPixel, sharedNative, ShaderStage::pixel, 0, 32);
    REQUIRE(sharedPlan.has_value());
    CHECK(*sharedPlan == std::vector{
                             PushConstantSubrange{
                                 .stages = ShaderStage::vertex | ShaderStage::pixel,
                                 .offset = 0,
                                 .size = 16,
                             },
                             PushConstantSubrange{.stages = ShaderStage::pixel, .offset = 16, .size = 16},
                         });
    CHECK_FALSE(planGraphicsPushConstants(sharedThenPixel, sharedNative, ShaderStage::vertex, 16, 16).has_value());

    const std::array sameExtent{
        PushConstantRange{.visibility = ShaderStage::vertex, .offset = 0, .size = 16},
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 0, .size = 16},
    };
    const auto sameExtentNative = mergePushConstantRangesPerStage(sameExtent);
    checkMerged(sameExtent, sameExtentNative);
    CHECK(std::ranges::equal(sameExtentNative,
                             std::array{PushConstantRange{
                                 .visibility = ShaderStage::vertex | ShaderStage::pixel,
                                 .offset = 0,
                                 .size = 16,
                             }},
                             sameRange));
    CHECK(mergePushConstantRangesPerStage({}).empty());
}

TEST_CASE("subresource layout requirement evaluator plans bridges or rejects incompatible layouts",
          "[size:small][rhi][vuid][layouts]")
{
    using rachis::ErrorCode;
    using rachis::vulkan::validation_detail::CanonicalTransferLayout;
    using rachis::vulkan::validation_detail::evaluateSubresourceLayoutRequirement;
    using rachis::vulkan::validation_detail::SubresourceLayoutTransitionStep;
    using rachis::vulkan::validation_detail::SubresourceTransferUse;

    constexpr std::int32_t kUndefined = 0;
    constexpr std::int32_t kGeneral = 1;
    constexpr std::int32_t kTransferSrc = 6;
    constexpr std::int32_t kTransferDst = 7;
    constexpr std::int32_t kShaderReadOnly = 5;

    const auto exact = evaluateSubresourceLayoutRequirement(kTransferDst,
                                                            CanonicalTransferLayout::transferDst,
                                                            kTransferDst,
                                                            CanonicalTransferLayout::transferDst,
                                                            SubresourceTransferUse::copyDestination,
                                                            false,
                                                            kUndefined,
                                                            kUndefined);
    REQUIRE(exact.has_value());
    CHECK_FALSE(exact->requiresPreBridge);
    CHECK_FALSE(exact->requiresPostBridge);

    const auto roundTrip = evaluateSubresourceLayoutRequirement(kGeneral,
                                                                CanonicalTransferLayout::general,
                                                                kTransferDst,
                                                                CanonicalTransferLayout::transferDst,
                                                                SubresourceTransferUse::copyDestination,
                                                                false,
                                                                kUndefined,
                                                                kUndefined);
    REQUIRE(roundTrip.has_value());
    CHECK(roundTrip->requiresPreBridge);
    CHECK(roundTrip->requiresPostBridge);
    CHECK(roundTrip->preBridge == SubresourceLayoutTransitionStep{.oldLayout = kGeneral, .newLayout = kTransferDst});
    CHECK(roundTrip->postBridge == SubresourceLayoutTransitionStep{.oldLayout = kTransferDst, .newLayout = kGeneral});

    const auto preOnly = evaluateSubresourceLayoutRequirement(kGeneral,
                                                              CanonicalTransferLayout::general,
                                                              kTransferSrc,
                                                              CanonicalTransferLayout::transferSrc,
                                                              SubresourceTransferUse::resolveSource,
                                                              true,
                                                              kTransferSrc,
                                                              kUndefined);
    REQUIRE(preOnly.has_value());
    CHECK(preOnly->requiresPreBridge);
    CHECK_FALSE(preOnly->requiresPostBridge);
    CHECK(preOnly->preBridge == SubresourceLayoutTransitionStep{.oldLayout = kGeneral, .newLayout = kTransferSrc});

    const auto discardFollowUp = evaluateSubresourceLayoutRequirement(kTransferDst,
                                                                      CanonicalTransferLayout::transferDst,
                                                                      kGeneral,
                                                                      CanonicalTransferLayout::general,
                                                                      SubresourceTransferUse::resolveDestination,
                                                                      true,
                                                                      kUndefined,
                                                                      kUndefined);
    REQUIRE(discardFollowUp.has_value());
    CHECK(discardFollowUp->requiresPreBridge);
    CHECK_FALSE(discardFollowUp->requiresPostBridge);

    const auto shaderReadActual = evaluateSubresourceLayoutRequirement(kShaderReadOnly,
                                                                       CanonicalTransferLayout::other,
                                                                       kTransferDst,
                                                                       CanonicalTransferLayout::transferDst,
                                                                       SubresourceTransferUse::copyDestination,
                                                                       false,
                                                                       kUndefined,
                                                                       kUndefined);
    REQUIRE_FALSE(shaderReadActual.has_value());
    CHECK(shaderReadActual.error().code == ErrorCode::invalidState);

    const auto wrongTransferDir = evaluateSubresourceLayoutRequirement(kTransferSrc,
                                                                       CanonicalTransferLayout::transferSrc,
                                                                       kTransferDst,
                                                                       CanonicalTransferLayout::transferDst,
                                                                       SubresourceTransferUse::copyDestination,
                                                                       false,
                                                                       kUndefined,
                                                                       kUndefined);
    REQUIRE_FALSE(wrongTransferDir.has_value());
    CHECK(wrongTransferDir.error().code == ErrorCode::invalidState);

    const auto conflictingFollowUp = evaluateSubresourceLayoutRequirement(kGeneral,
                                                                          CanonicalTransferLayout::general,
                                                                          kTransferDst,
                                                                          CanonicalTransferLayout::transferDst,
                                                                          SubresourceTransferUse::copyDestination,
                                                                          true,
                                                                          kGeneral,
                                                                          kUndefined);
    REQUIRE_FALSE(conflictingFollowUp.has_value());
    CHECK(conflictingFollowUp.error().code == ErrorCode::invalidState);
}

TEST_CASE("color clear encoding selects integer union members and truncates and saturates integer channels",
          "[size:small][rhi][vuid][clear]")
{
    using rachis::Format;
    using rachis::vulkan::validation_detail::ColorClearEncoding;
    using rachis::vulkan::validation_detail::colorClearEncoding;
    using rachis::vulkan::validation_detail::ColorClearNumericType;
    using rachis::vulkan::validation_detail::signedIntegerClearChannel;
    using rachis::vulkan::validation_detail::unsignedIntegerClearChannel;

    CHECK(colorClearEncoding(Format::rgba8Unorm) == ColorClearEncoding{});
    CHECK(colorClearEncoding(Format::rgba16Float) == ColorClearEncoding{});
    CHECK(colorClearEncoding(Format::rgba32Float) == ColorClearEncoding{});
    CHECK(colorClearEncoding(Format::bgra8UnormSrgb) == ColorClearEncoding{});
    CHECK(colorClearEncoding(Format::rgba8Uint) == ColorClearEncoding{ColorClearNumericType::unsignedInteger, 8});
    CHECK(colorClearEncoding(Format::rg16Uint) == ColorClearEncoding{ColorClearNumericType::unsignedInteger, 16});
    CHECK(colorClearEncoding(Format::r32Uint) == ColorClearEncoding{ColorClearNumericType::unsignedInteger, 32});
    CHECK(colorClearEncoding(Format::r64Uint) == ColorClearEncoding{ColorClearNumericType::unsignedInteger, 32});
    CHECK(colorClearEncoding(Format::r8Sint) == ColorClearEncoding{ColorClearNumericType::signedInteger, 8});
    CHECK(colorClearEncoding(Format::rgba16Sint) == ColorClearEncoding{ColorClearNumericType::signedInteger, 16});
    CHECK(colorClearEncoding(Format::rg32Sint) == ColorClearEncoding{ColorClearNumericType::signedInteger, 32});

    CHECK(unsignedIntegerClearChannel(1.0F, 32) == 1U);
    CHECK(unsignedIntegerClearChannel(1.7F, 32) == 1U);
    CHECK(unsignedIntegerClearChannel(0.5F, 8) == 0U);
    CHECK(unsignedIntegerClearChannel(-0.5F, 8) == 0U);
    CHECK(unsignedIntegerClearChannel(-5.0F, 16) == 0U);
    CHECK(unsignedIntegerClearChannel(300.0F, 8) == 255U);
    CHECK(unsignedIntegerClearChannel(70000.0F, 16) == 65535U);
    CHECK(unsignedIntegerClearChannel(70000.0F, 32) == 70000U);
    CHECK(unsignedIntegerClearChannel(4294967040.0F, 32) == 4294967040U);
    CHECK(unsignedIntegerClearChannel(5e9F, 32) == std::numeric_limits<std::uint32_t>::max());
    CHECK(unsignedIntegerClearChannel(std::numeric_limits<float>::quiet_NaN(), 32) == 0U);

    CHECK(signedIntegerClearChannel(1.0F, 32) == 1);
    CHECK(signedIntegerClearChannel(-1.7F, 8) == -1);
    CHECK(signedIntegerClearChannel(1.7F, 16) == 1);
    CHECK(signedIntegerClearChannel(-0.5F, 32) == 0);
    CHECK(signedIntegerClearChannel(200.0F, 8) == 127);
    CHECK(signedIntegerClearChannel(-200.0F, 8) == -128);
    CHECK(signedIntegerClearChannel(40000.0F, 16) == 32767);
    CHECK(signedIntegerClearChannel(-40000.0F, 16) == -32768);
    CHECK(signedIntegerClearChannel(-70000.0F, 32) == -70000);
    CHECK(signedIntegerClearChannel(-2147483648.0F, 32) == std::numeric_limits<std::int32_t>::min());
    CHECK(signedIntegerClearChannel(3e9F, 32) == std::numeric_limits<std::int32_t>::max());
    CHECK(signedIntegerClearChannel(-3e9F, 32) == std::numeric_limits<std::int32_t>::min());
    CHECK(signedIntegerClearChannel(std::numeric_limits<float>::quiet_NaN(), 32) == 0);
}

TEST_CASE("buffer-texture copy queue-family check rejects depth and stencil aspects without graphics support",
          "[size:small][rhi][vuid][transfer]")
{
    using rachis::BufferTextureCopyRegion;
    using rachis::ErrorCode;
    using rachis::TextureAspect;
    using rachis::vulkan::validation_detail::validateBufferTextureCopyQueueFamily;

    const std::array color{BufferTextureCopyRegion{.textureSubresource = {.aspect = TextureAspect::color}}};
    const std::array depth{BufferTextureCopyRegion{.textureSubresource = {.aspect = TextureAspect::depth}}};
    const std::array stencil{BufferTextureCopyRegion{.textureSubresource = {.aspect = TextureAspect::stencil}}};
    const std::array colorThenStencil{
        BufferTextureCopyRegion{.textureSubresource = {.aspect = TextureAspect::color}},
        BufferTextureCopyRegion{.textureSubresource = {.mipLevel = 1, .aspect = TextureAspect::stencil}},
    };

    CHECK(validateBufferTextureCopyQueueFamily(true, color).has_value());
    CHECK(validateBufferTextureCopyQueueFamily(true, depth).has_value());
    CHECK(validateBufferTextureCopyQueueFamily(true, stencil).has_value());
    CHECK(validateBufferTextureCopyQueueFamily(true, colorThenStencil).has_value());
    CHECK(validateBufferTextureCopyQueueFamily(false, color).has_value());

    for (const auto rejected : {std::span<const BufferTextureCopyRegion>{depth},
                                std::span<const BufferTextureCopyRegion>{stencil},
                                std::span<const BufferTextureCopyRegion>{colorThenStencil}})
    {
        const auto result = validateBufferTextureCopyQueueFamily(false, rejected);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().code == ErrorCode::unsupported);
    }
}

#if RACHIS_ENABLE_VULKAN && defined(RACHIS_TESTING)
namespace
{

    thread_local std::uint32_t gHostAllocationsUntilFailure = 0;
    thread_local bool gHostAllocationFailureInjected = false;

    template<typename Record>
    void recordWithoutInjectedFailure(Record&& record)
    {
        const auto remaining = std::exchange(gHostAllocationsUntilFailure, 0U);
        std::forward<Record>(record)();
        gHostAllocationsUntilFailure = remaining;
    }

} // namespace

void* operator new(std::size_t size)
{
    if (gHostAllocationsUntilFailure != 0 && --gHostAllocationsUntilFailure == 0)
    {
        gHostAllocationFailureInjected = true;
        throw std::bad_alloc{};
    }
    const auto bytes = size == 0 ? std::size_t{1} : size;
    while (true)
    {
        if (void* const memory = std::malloc(bytes))
        {
            return memory;
        }
        const auto handler = std::get_new_handler();
        if (handler == nullptr)
        {
            throw std::bad_alloc{};
        }
        handler();
    }
}

#if defined(__GNUC__)
[[gnu::noinline]]
#endif
void operator delete(void* memory) noexcept
{
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept
{
    ::operator delete(memory);
}

namespace
{

    template<typename Handle>
    Handle fakeHandle(std::uintptr_t value)
    {
        return reinterpret_cast<Handle>(value);
    }

    [[nodiscard]] VkQueryPool poisonedQueryPool() noexcept
    {
        return fakeHandle<VkQueryPool>(0xBAD0'0000U);
    }

    struct RecordedPushConstantCall final
    {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkShaderStageFlags stageFlags = 0;
        std::uint32_t offset = 0;
        std::uint32_t size = 0;
    };

    struct RecordedQueryResetCall final
    {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkQueryPool pool = VK_NULL_HANDLE;
        std::uint32_t firstQuery = 0;
        std::uint32_t queryCount = 0;
    };

    struct RecordedImageBarrier final
    {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VkImageAspectFlags aspectMask = 0;
        std::uint32_t baseMipLevel = 0;
        std::uint32_t baseArrayLayer = 0;
        VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 srcStageMask = 0;
        VkPipelineStageFlags2 dstStageMask = 0;
    };

    struct RecordedImageLayoutUse final
    {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct RecordedMemoryBarrier final
    {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkPipelineStageFlags2 srcStageMask = 0;
        VkAccessFlags2 srcAccessMask = 0;
        VkPipelineStageFlags2 dstStageMask = 0;
        VkAccessFlags2 dstAccessMask = 0;
    };

    struct RecordedResolveImageCall final
    {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkImage srcImage = VK_NULL_HANDLE;
        VkImageLayout srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage dstImage = VK_NULL_HANDLE;
        VkImageLayout dstLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct RecordedBeginRenderingCall final
    {
        VkCommandBuffer command = VK_NULL_HANDLE;
        std::uint32_t colorCount = 0;
        VkAttachmentLoadOp firstColorLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        VkAttachmentStoreOp firstColorStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        bool hasDepth = false;
        bool hasStencil = false;
        VkImageLayout depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout stencilLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool hasShadingRateAttachment = false;
    };

    enum class CommandOpKind : std::uint8_t
    {
        beginRendering,
        endRendering,
        memoryBarrier,
        imageBarrier,
        resetQueryPool,
        beginQuery,
        endQuery,
        writeTimestamp,
        copyQueryResults,
        copyImage,
        resolveImage,
        pushConstants,
        setDepthBias,
        clearColorImage,
    };

    struct CommandOpEvent final
    {
        CommandOpKind kind{};
        VkCommandBuffer command = VK_NULL_HANDLE;
    };

    struct MockVulkanDispatch final
    {
        std::uintptr_t nextHandle = 0x1000;
        std::uint64_t semaphoreCounter = 0;
        std::size_t hostQueryResetCount = 0;
        std::vector<RecordedPushConstantCall> pushConstants;
        std::vector<RecordedQueryResetCall> cmdQueryResets;
        std::vector<RecordedImageBarrier> imageBarriers;
        std::vector<RecordedMemoryBarrier> memoryBarriers;
        std::vector<RecordedResolveImageCall> resolves;
        std::vector<RecordedBeginRenderingCall> beginRenderings;
        std::vector<VkImageViewType> createdViewTypes;
        std::vector<VkCommandBuffer> lastSubmittedCommands;
        std::vector<CommandOpEvent> timelineEvents;
        std::vector<VkShaderStageFlags> descriptorBindingStages;
        std::vector<VkShaderStageFlags> pushConstantRangeStages;
        std::vector<VkTimeDomainEXT> calibrateableDomains;
        std::vector<VkTimeDomainEXT> calibratedDomains;
        std::size_t submitCount = 0;
        std::size_t setDepthBiasCount = 0;
        std::size_t liveQueryPools = 0;
        std::size_t createdQueryPoolCount = 0;
        std::array<VkQueryPool, 2> createdQueryPools{};
        std::size_t poisonedQueryPoolDestroys = 0;
        VkQueryType failedQueryType = VK_QUERY_TYPE_MAX_ENUM;
        VkResult failedQueryPoolResult = VK_SUCCESS;
        std::uint32_t hostAllocationFailureAfterQueryPoolCreate = 0;
        std::size_t queueFamilyPropertyQueries = 0;
        std::uint32_t hostAllocationFailureAfterQueueFamilyCount = 0;
        std::uint64_t namedQueryPool = 0;
        std::array<char, 128> queryPoolName{};
        std::size_t queryPoolNameLength = 0;
        VkResult fenceStatus = VK_SUCCESS;
        std::size_t fenceResets = 0;
        std::size_t clearColorImages = 0;
        std::vector<RecordedImageLayoutUse> clearColorLayouts;
        std::vector<RecordedImageLayoutUse> copyImageDestinationLayouts;
    };

    thread_local MockVulkanDispatch* gMock = nullptr;

    VKAPI_ATTR VkResult VKAPI_CALL mockCreateSemaphore(VkDevice,
                                                       const VkSemaphoreCreateInfo*,
                                                       const VkAllocationCallbacks*,
                                                       VkSemaphore* pSemaphore)
    {
        *pSemaphore = fakeHandle<VkSemaphore>(gMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockDestroySemaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockGetSemaphoreCounterValue(VkDevice, VkSemaphore, std::uint64_t* pValue)
    {
        *pValue = gMock->semaphoreCounter;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockWaitSemaphores(VkDevice, const VkSemaphoreWaitInfo*, std::uint64_t)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockCreateFence(VkDevice,
                                                   const VkFenceCreateInfo*,
                                                   const VkAllocationCallbacks*,
                                                   VkFence* pFence)
    {
        *pFence = fakeHandle<VkFence>(gMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockDestroyFence(VkDevice, VkFence, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockGetFenceStatus(VkDevice, VkFence)
    {
        return gMock->fenceStatus;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockResetFences(VkDevice, std::uint32_t, const VkFence*)
    {
        ++gMock->fenceResets;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockWaitForFences(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t)
    {
        return gMock->fenceStatus == VK_NOT_READY ? VK_TIMEOUT : gMock->fenceStatus;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockCreateCommandPool(VkDevice,
                                                         const VkCommandPoolCreateInfo*,
                                                         const VkAllocationCallbacks*,
                                                         VkCommandPool* pCommandPool)
    {
        *pCommandPool = fakeHandle<VkCommandPool>(gMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockResetCommandPool(VkDevice, VkCommandPool, VkCommandPoolResetFlags)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockAllocateCommandBuffers(VkDevice,
                                                              const VkCommandBufferAllocateInfo* pAllocateInfo,
                                                              VkCommandBuffer* pCommandBuffers)
    {
        for (std::uint32_t i = 0; i < pAllocateInfo->commandBufferCount; ++i)
        {
            pCommandBuffers[i] = fakeHandle<VkCommandBuffer>(gMock->nextHandle++);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockFreeCommandBuffers(VkDevice, VkCommandPool, std::uint32_t, const VkCommandBuffer*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockEndCommandBuffer(VkCommandBuffer)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockResetCommandBuffer(VkCommandBuffer, VkCommandBufferResetFlags)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockQueueSubmit2(VkQueue,
                                                    std::uint32_t submitCount,
                                                    const VkSubmitInfo2* pSubmits,
                                                    VkFence)
    {
        gMock->lastSubmittedCommands.clear();
        for (std::uint32_t s = 0; s < submitCount; ++s)
        {
            for (std::uint32_t c = 0; c < pSubmits[s].commandBufferInfoCount; ++c)
            {
                gMock->lastSubmittedCommands.push_back(pSubmits[s].pCommandBufferInfos[c].commandBuffer);
            }
            for (std::uint32_t sig = 0; sig < pSubmits[s].signalSemaphoreInfoCount; ++sig)
            {
                gMock->semaphoreCounter =
                    std::max(gMock->semaphoreCounter, pSubmits[s].pSignalSemaphoreInfos[sig].value);
            }
        }
        ++gMock->submitCount;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockCreateQueryPool(VkDevice,
                                                       const VkQueryPoolCreateInfo* pCreateInfo,
                                                       const VkAllocationCallbacks*,
                                                       VkQueryPool* pQueryPool)
    {
        if (pCreateInfo->queryType == gMock->failedQueryType)
        {
            *pQueryPool = poisonedQueryPool();
            return gMock->failedQueryPoolResult;
        }
        *pQueryPool = fakeHandle<VkQueryPool>(gMock->nextHandle++);
        ++gMock->liveQueryPools;
        if (gMock->createdQueryPoolCount < gMock->createdQueryPools.size())
        {
            gMock->createdQueryPools[gMock->createdQueryPoolCount] = *pQueryPool;
        }
        ++gMock->createdQueryPoolCount;
        if (gMock->hostAllocationFailureAfterQueryPoolCreate != 0)
        {
            gHostAllocationsUntilFailure = std::exchange(gMock->hostAllocationFailureAfterQueryPoolCreate, 0U);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockDestroyQueryPool(VkDevice, VkQueryPool queryPool, const VkAllocationCallbacks*)
    {
        if (gMock == nullptr || queryPool == VK_NULL_HANDLE)
        {
            return;
        }
        if (queryPool == poisonedQueryPool())
        {
            ++gMock->poisonedQueryPoolDestroys;
            return;
        }
        --gMock->liveQueryPools;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockSetDebugUtilsObjectName(VkDevice, const VkDebugUtilsObjectNameInfoEXT* pNameInfo)
    {
        if (pNameInfo->objectType == VK_OBJECT_TYPE_QUERY_POOL)
        {
            const std::string_view name(pNameInfo->pObjectName);
            gMock->namedQueryPool = pNameInfo->objectHandle;
            gMock->queryPoolNameLength = std::min(name.size(), gMock->queryPoolName.size());
            std::copy_n(name.begin(), gMock->queryPoolNameLength, gMock->queryPoolName.begin());
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockResetQueryPool(VkDevice, VkQueryPool, std::uint32_t, std::uint32_t)
    {
        ++gMock->hostQueryResetCount;
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdResetQueryPool(VkCommandBuffer commandBuffer,
                                                     VkQueryPool queryPool,
                                                     std::uint32_t firstQuery,
                                                     std::uint32_t queryCount)
    {
        gMock->cmdQueryResets.push_back({
            .command = commandBuffer,
            .pool = queryPool,
            .firstQuery = firstQuery,
            .queryCount = queryCount,
        });
        gMock->timelineEvents.push_back({.kind = CommandOpKind::resetQueryPool, .command = commandBuffer});
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdBeginQuery(VkCommandBuffer commandBuffer,
                                                 VkQueryPool,
                                                 std::uint32_t,
                                                 VkQueryControlFlags)
    {
        gMock->timelineEvents.push_back({.kind = CommandOpKind::beginQuery, .command = commandBuffer});
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdEndQuery(VkCommandBuffer commandBuffer, VkQueryPool, std::uint32_t)
    {
        gMock->timelineEvents.push_back({.kind = CommandOpKind::endQuery, .command = commandBuffer});
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdWriteTimestamp2(VkCommandBuffer commandBuffer,
                                                      VkPipelineStageFlags2,
                                                      VkQueryPool,
                                                      std::uint32_t)
    {
        gMock->timelineEvents.push_back({.kind = CommandOpKind::writeTimestamp, .command = commandBuffer});
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdCopyQueryPoolResults(VkCommandBuffer commandBuffer,
                                                           VkQueryPool,
                                                           std::uint32_t,
                                                           std::uint32_t,
                                                           VkBuffer,
                                                           VkDeviceSize,
                                                           VkDeviceSize,
                                                           VkQueryResultFlags)
    {
        gMock->timelineEvents.push_back({.kind = CommandOpKind::copyQueryResults, .command = commandBuffer});
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                                                       const VkDependencyInfo* pDependencyInfo)
    {
        for (std::uint32_t i = 0; i < pDependencyInfo->memoryBarrierCount; ++i)
        {
            const auto& b = pDependencyInfo->pMemoryBarriers[i];
            gMock->memoryBarriers.push_back({
                .command = commandBuffer,
                .srcStageMask = b.srcStageMask,
                .srcAccessMask = b.srcAccessMask,
                .dstStageMask = b.dstStageMask,
                .dstAccessMask = b.dstAccessMask,
            });
            gMock->timelineEvents.push_back({.kind = CommandOpKind::memoryBarrier, .command = commandBuffer});
        }
        for (std::uint32_t i = 0; i < pDependencyInfo->imageMemoryBarrierCount; ++i)
        {
            const auto& b = pDependencyInfo->pImageMemoryBarriers[i];
            gMock->imageBarriers.push_back({
                .command = commandBuffer,
                .image = b.image,
                .aspectMask = b.subresourceRange.aspectMask,
                .baseMipLevel = b.subresourceRange.baseMipLevel,
                .baseArrayLayer = b.subresourceRange.baseArrayLayer,
                .oldLayout = b.oldLayout,
                .newLayout = b.newLayout,
                .srcStageMask = b.srcStageMask,
                .dstStageMask = b.dstStageMask,
            });
            gMock->timelineEvents.push_back({.kind = CommandOpKind::imageBarrier, .command = commandBuffer});
        }
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdCopyImage(VkCommandBuffer commandBuffer,
                                                VkImage,
                                                VkImageLayout,
                                                VkImage dstImage,
                                                VkImageLayout dstImageLayout,
                                                std::uint32_t,
                                                const VkImageCopy*)
    {
        recordWithoutInjectedFailure([&] {
            gMock->timelineEvents.push_back({.kind = CommandOpKind::copyImage, .command = commandBuffer});
            gMock->copyImageDestinationLayouts.push_back(
                {.command = commandBuffer, .image = dstImage, .layout = dstImageLayout});
        });
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdClearColorImage(VkCommandBuffer commandBuffer,
                                                      VkImage image,
                                                      VkImageLayout imageLayout,
                                                      const VkClearColorValue*,
                                                      std::uint32_t,
                                                      const VkImageSubresourceRange*)
    {
        ++gMock->clearColorImages;
        recordWithoutInjectedFailure([&] {
            gMock->clearColorLayouts.push_back({.command = commandBuffer, .image = image, .layout = imageLayout});
            gMock->timelineEvents.push_back({.kind = CommandOpKind::clearColorImage, .command = commandBuffer});
        });
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdResolveImage(VkCommandBuffer commandBuffer,
                                                   VkImage srcImage,
                                                   VkImageLayout srcImageLayout,
                                                   VkImage dstImage,
                                                   VkImageLayout dstImageLayout,
                                                   std::uint32_t,
                                                   const VkImageResolve*)
    {
        gMock->resolves.push_back({
            .command = commandBuffer,
            .srcImage = srcImage,
            .srcLayout = srcImageLayout,
            .dstImage = dstImage,
            .dstLayout = dstImageLayout,
        });
        gMock->timelineEvents.push_back({.kind = CommandOpKind::resolveImage, .command = commandBuffer});
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockCreateImageView(VkDevice,
                                                       const VkImageViewCreateInfo* pCreateInfo,
                                                       const VkAllocationCallbacks*,
                                                       VkImageView* pView)
    {
        gMock->createdViewTypes.push_back(pCreateInfo->viewType);
        *pView = fakeHandle<VkImageView>(gMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockDestroyImageView(VkDevice, VkImageView, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdBeginRendering(VkCommandBuffer commandBuffer,
                                                     const VkRenderingInfo* pRenderingInfo)
    {
        RecordedBeginRenderingCall call{
            .command = commandBuffer,
            .colorCount = pRenderingInfo->colorAttachmentCount,
            .hasDepth = pRenderingInfo->pDepthAttachment != nullptr,
            .hasStencil = pRenderingInfo->pStencilAttachment != nullptr,
            .depthLayout = pRenderingInfo->pDepthAttachment != nullptr ? pRenderingInfo->pDepthAttachment->imageLayout
                                                                       : VK_IMAGE_LAYOUT_UNDEFINED,
            .stencilLayout = pRenderingInfo->pStencilAttachment != nullptr
                                 ? pRenderingInfo->pStencilAttachment->imageLayout
                                 : VK_IMAGE_LAYOUT_UNDEFINED,
            .hasShadingRateAttachment = pRenderingInfo->pNext != nullptr,
        };
        if (pRenderingInfo->colorAttachmentCount > 0 && pRenderingInfo->pColorAttachments != nullptr)
        {
            call.firstColorLoadOp = pRenderingInfo->pColorAttachments[0].loadOp;
            call.firstColorStoreOp = pRenderingInfo->pColorAttachments[0].storeOp;
        }
        gMock->beginRenderings.push_back(call);
        gMock->timelineEvents.push_back({.kind = CommandOpKind::beginRendering, .command = commandBuffer});
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdEndRendering(VkCommandBuffer commandBuffer)
    {
        gMock->timelineEvents.push_back({.kind = CommandOpKind::endRendering, .command = commandBuffer});
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdPushConstants(VkCommandBuffer commandBuffer,
                                                    VkPipelineLayout,
                                                    VkShaderStageFlags stageFlags,
                                                    std::uint32_t offset,
                                                    std::uint32_t size,
                                                    const void*)
    {
        gMock->pushConstants.push_back({
            .command = commandBuffer,
            .stageFlags = stageFlags,
            .offset = offset,
            .size = size,
        });
        gMock->timelineEvents.push_back({.kind = CommandOpKind::pushConstants, .command = commandBuffer});
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockCreateDescriptorSetLayout(VkDevice,
                                                                 const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
                                                                 const VkAllocationCallbacks*,
                                                                 VkDescriptorSetLayout* pSetLayout)
    {
        for (std::uint32_t i = 0; i < pCreateInfo->bindingCount; ++i)
        {
            gMock->descriptorBindingStages.push_back(pCreateInfo->pBindings[i].stageFlags);
        }
        *pSetLayout = fakeHandle<VkDescriptorSetLayout>(gMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockDestroyDescriptorSetLayout(VkDevice,
                                                              VkDescriptorSetLayout,
                                                              const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockCreatePipelineLayout(VkDevice,
                                                            const VkPipelineLayoutCreateInfo* pCreateInfo,
                                                            const VkAllocationCallbacks*,
                                                            VkPipelineLayout* pPipelineLayout)
    {
        for (std::uint32_t i = 0; i < pCreateInfo->pushConstantRangeCount; ++i)
        {
            gMock->pushConstantRangeStages.push_back(pCreateInfo->pPushConstantRanges[i].stageFlags);
        }
        *pPipelineLayout = fakeHandle<VkPipelineLayout>(gMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockDestroyPipelineLayout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdSetDepthBias(VkCommandBuffer commandBuffer, float, float, float)
    {
        ++gMock->setDepthBiasCount;
        gMock->timelineEvents.push_back({.kind = CommandOpKind::setDepthBias, .command = commandBuffer});
    }

    VKAPI_ATTR void VKAPI_CALL mockCmdSetFragmentShadingRateKHR(VkCommandBuffer,
                                                                const VkExtent2D*,
                                                                const VkFragmentShadingRateCombinerOpKHR[2])
    {
    }

    VKAPI_ATTR void VKAPI_CALL mockGetPhysicalDeviceFormatProperties(VkPhysicalDevice,
                                                                     VkFormat,
                                                                     VkFormatProperties* pFormatProperties)
    {
        pFormatProperties->linearTilingFeatures = 0;
        pFormatProperties->optimalTilingFeatures =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        pFormatProperties->bufferFeatures = 0;
    }

    VKAPI_ATTR void VKAPI_CALL mockGetLinearRichFormatProperties(VkPhysicalDevice,
                                                                 VkFormat,
                                                                 VkFormatProperties* pFormatProperties)
    {
        pFormatProperties->linearTilingFeatures =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        pFormatProperties->optimalTilingFeatures =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        pFormatProperties->bufferFeatures = VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT |
                                            VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT |
                                            VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT;
    }

    VKAPI_ATTR VkResult VKAPI_CALL
    mockGetPhysicalDeviceImageFormatProperties(VkPhysicalDevice,
                                               VkFormat,
                                               VkImageType,
                                               VkImageTiling,
                                               VkImageUsageFlags,
                                               VkImageCreateFlags,
                                               VkImageFormatProperties* pImageFormatProperties)
    {
        *pImageFormatProperties = VkImageFormatProperties{
            .maxExtent = {4096, 4096, 256},
            .maxMipLevels = 12,
            .maxArrayLayers = 256,
            .sampleCounts =
                VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT | VK_SAMPLE_COUNT_8_BIT,
            .maxResourceSize = 1ULL << 30U,
        };
        return VK_SUCCESS;
    }

    [[nodiscard]] constexpr VkQueueFamilyProperties mockQueueFamily(VkQueueFlags flags,
                                                                    VkExtent3D granularity,
                                                                    std::uint32_t timestampValidBits = 64) noexcept
    {
        return VkQueueFamilyProperties{
            .queueFlags = flags,
            .queueCount = 1,
            .timestampValidBits = timestampValidBits,
            .minImageTransferGranularity = granularity,
        };
    }

    constexpr VkQueueFlags kMockGraphicsFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;

    constexpr std::array kMockQueueFamilies{
        mockQueueFamily(kMockGraphicsFlags, {1, 1, 1}),
        mockQueueFamily(VK_QUEUE_TRANSFER_BIT, {1, 1, 1}, 48),
    };

    thread_local std::span<const VkQueueFamilyProperties> gMockQueueFamilies = kMockQueueFamilies;

    VKAPI_ATTR void VKAPI_CALL mockGetPhysicalDeviceQueueFamilyProperties(
        VkPhysicalDevice, std::uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties)
    {
        if (gMock != nullptr)
        {
            ++gMock->queueFamilyPropertyQueries;
        }
        const auto available = static_cast<std::uint32_t>(gMockQueueFamilies.size());
        if (pQueueFamilyProperties == nullptr)
        {
            *pQueueFamilyPropertyCount = available;
            if (gMock != nullptr && gMock->hostAllocationFailureAfterQueueFamilyCount != 0)
            {
                gHostAllocationsUntilFailure = std::exchange(gMock->hostAllocationFailureAfterQueueFamilyCount, 0U);
            }
            return;
        }
        const auto written = std::min(*pQueueFamilyPropertyCount, available);
        std::copy_n(gMockQueueFamilies.begin(), written, pQueueFamilyProperties);
        *pQueueFamilyPropertyCount = written;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockEnumeratePhysicalDevices(VkInstance,
                                                                std::uint32_t* pPhysicalDeviceCount,
                                                                VkPhysicalDevice* pPhysicalDevices)
    {
        if (pPhysicalDevices == nullptr)
        {
            *pPhysicalDeviceCount = 1;
            return VK_SUCCESS;
        }
        if (*pPhysicalDeviceCount == 0)
        {
            return VK_INCOMPLETE;
        }
        pPhysicalDevices[0] = fakeHandle<VkPhysicalDevice>(0x12);
        *pPhysicalDeviceCount = 1;
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockGetPhysicalDeviceProperties(VkPhysicalDevice,
                                                               VkPhysicalDeviceProperties* pProperties)
    {
        constexpr std::string_view deviceName = "Rachis Mock Vulkan Adapter";
        *pProperties = VkPhysicalDeviceProperties{};
        pProperties->apiVersion = VK_API_VERSION_1_3;
        pProperties->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        std::ranges::copy(deviceName, pProperties->deviceName);
        pProperties->limits.timestampComputeAndGraphics = VK_TRUE;
        pProperties->limits.timestampPeriod = 1.0F;
    }

    VKAPI_ATTR void VKAPI_CALL mockGetPhysicalDeviceProperties2(VkPhysicalDevice physical,
                                                                VkPhysicalDeviceProperties2* pProperties)
    {
        mockGetPhysicalDeviceProperties(physical, &pProperties->properties);
    }

    VKAPI_ATTR void VKAPI_CALL
    mockGetPhysicalDeviceMemoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties)
    {
        *pMemoryProperties = VkPhysicalDeviceMemoryProperties{};
    }

    VKAPI_ATTR void VKAPI_CALL mockGetPhysicalDeviceFeatures2(VkPhysicalDevice, VkPhysicalDeviceFeatures2*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockEnumerateDeviceExtensionProperties(VkPhysicalDevice,
                                                                          const char*,
                                                                          std::uint32_t* pPropertyCount,
                                                                          VkExtensionProperties*)
    {
        *pPropertyCount = 0;
        return VK_SUCCESS;
    }

    constexpr std::uint64_t mockCalibratedDeviceTimestamp = 0x0123'4567'89AB'CDEFULL;
    constexpr std::uint64_t mockCalibratedHostTimestamp = 0x0FED'CBA9'8765'4321ULL;

    VKAPI_ATTR VkResult VKAPI_CALL mockGetPhysicalDeviceCalibrateableTimeDomains(VkPhysicalDevice,
                                                                                 std::uint32_t* pTimeDomainCount,
                                                                                 VkTimeDomainEXT* pTimeDomains)
    {
        const auto& domains = gMock->calibrateableDomains;
        if (pTimeDomains == nullptr)
        {
            *pTimeDomainCount = static_cast<std::uint32_t>(domains.size());
            return VK_SUCCESS;
        }
        const auto count = std::min<std::size_t>(*pTimeDomainCount, domains.size());
        std::copy_n(domains.begin(), count, pTimeDomains);
        *pTimeDomainCount = static_cast<std::uint32_t>(count);
        return count < domains.size() ? VK_INCOMPLETE : VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockGetCalibratedTimestamps(VkDevice,
                                                               std::uint32_t timestampCount,
                                                               const VkCalibratedTimestampInfoEXT* pTimestampInfos,
                                                               std::uint64_t* pTimestamps,
                                                               std::uint64_t* pMaxDeviation)
    {
        for (std::uint32_t i = 0; i < timestampCount; ++i)
        {
            const auto domain = pTimestampInfos[i].timeDomain;
            gMock->calibratedDomains.push_back(domain);
            pTimestamps[i] =
                domain == VK_TIME_DOMAIN_DEVICE_EXT ? mockCalibratedDeviceTimestamp : mockCalibratedHostTimestamp;
        }
        *pMaxDeviation = 1;
        return VK_SUCCESS;
    }

    [[nodiscard]] rachis::Result<std::vector<rachis::vulkan::VulkanPhysicalDeviceRecord>> discoverWithQueueFamilies(
        std::span<const VkQueueFamilyProperties> families)
    {
        gMockQueueFamilies = families;
        rachis::vulkan::VulkanInstanceTable instanceTable{};
        instanceTable.vkEnumeratePhysicalDevices = &mockEnumeratePhysicalDevices;
        instanceTable.vkGetPhysicalDeviceProperties = &mockGetPhysicalDeviceProperties;
        instanceTable.vkGetPhysicalDeviceProperties2 = &mockGetPhysicalDeviceProperties2;
        instanceTable.vkGetPhysicalDeviceMemoryProperties = &mockGetPhysicalDeviceMemoryProperties;
        instanceTable.vkGetPhysicalDeviceFeatures2 = &mockGetPhysicalDeviceFeatures2;
        instanceTable.vkEnumerateDeviceExtensionProperties = &mockEnumerateDeviceExtensionProperties;
        instanceTable.vkGetPhysicalDeviceQueueFamilyProperties = &mockGetPhysicalDeviceQueueFamilyProperties;
        auto records = rachis::vulkan::discoverVulkanPhysicalDevices(
            fakeHandle<VkInstance>(0x13), instanceTable, rachis::VulkanInstanceReport{.apiMajor = 1, .apiMinor = 3});
        gMockQueueFamilies = kMockQueueFamilies;
        return records;
    }

    [[nodiscard]] VolkDeviceTable makeMockDeviceTable() noexcept
    {
        VolkDeviceTable table{};
        table.vkCreateSemaphore = &mockCreateSemaphore;
        table.vkDestroySemaphore = &mockDestroySemaphore;
        table.vkGetSemaphoreCounterValue = &mockGetSemaphoreCounterValue;
        table.vkWaitSemaphores = &mockWaitSemaphores;
        table.vkCreateFence = &mockCreateFence;
        table.vkDestroyFence = &mockDestroyFence;
        table.vkGetFenceStatus = &mockGetFenceStatus;
        table.vkResetFences = &mockResetFences;
        table.vkWaitForFences = &mockWaitForFences;
        table.vkCreateCommandPool = &mockCreateCommandPool;
        table.vkDestroyCommandPool = &mockDestroyCommandPool;
        table.vkResetCommandPool = &mockResetCommandPool;
        table.vkAllocateCommandBuffers = &mockAllocateCommandBuffers;
        table.vkFreeCommandBuffers = &mockFreeCommandBuffers;
        table.vkBeginCommandBuffer = &mockBeginCommandBuffer;
        table.vkEndCommandBuffer = &mockEndCommandBuffer;
        table.vkResetCommandBuffer = &mockResetCommandBuffer;
        table.vkQueueSubmit2 = &mockQueueSubmit2;
        table.vkCreateQueryPool = &mockCreateQueryPool;
        table.vkDestroyQueryPool = &mockDestroyQueryPool;
        table.vkResetQueryPool = &mockResetQueryPool;
        table.vkCmdResetQueryPool = &mockCmdResetQueryPool;
        table.vkCmdBeginQuery = &mockCmdBeginQuery;
        table.vkCmdEndQuery = &mockCmdEndQuery;
        table.vkCmdWriteTimestamp2 = &mockCmdWriteTimestamp2;
        table.vkCmdCopyQueryPoolResults = &mockCmdCopyQueryPoolResults;
        table.vkCmdPipelineBarrier2 = &mockCmdPipelineBarrier2;
        table.vkCmdCopyImage = &mockCmdCopyImage;
        table.vkCmdResolveImage = &mockCmdResolveImage;
        table.vkCmdClearColorImage = &mockCmdClearColorImage;
        table.vkCreateImageView = &mockCreateImageView;
        table.vkDestroyImageView = &mockDestroyImageView;
        table.vkCmdBeginRendering = &mockCmdBeginRendering;
        table.vkCmdEndRendering = &mockCmdEndRendering;
        table.vkCmdPushConstants = &mockCmdPushConstants;
        table.vkCreateDescriptorSetLayout = &mockCreateDescriptorSetLayout;
        table.vkDestroyDescriptorSetLayout = &mockDestroyDescriptorSetLayout;
        table.vkCreatePipelineLayout = &mockCreatePipelineLayout;
        table.vkDestroyPipelineLayout = &mockDestroyPipelineLayout;
        table.vkCmdSetDepthBias = &mockCmdSetDepthBias;
        table.vkCmdSetFragmentShadingRateKHR = &mockCmdSetFragmentShadingRateKHR;
        return table;
    }

} // namespace

TEST_CASE("Vulkan query manager and queue submit enforce host resets, DMA timestamp guards, and pre-submit slot resets",
          "[size:small][rhi][vulkan][vuid][queries]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();

    rachis::vulkan::VulkanInstanceTable instanceTable{};

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = fakeHandle<VkPhysicalDevice>(0x10);
    record.properties.limits.timestampComputeAndGraphics = VK_TRUE;
    record.properties.limits.timestampPeriod = 1.0F;
    rachis::vulkan::recordVulkanQueueFamilies(record, kMockQueueFamilies);
    REQUIRE(record.queues.transfer == 1);

    const auto graphicsProps = rachis::vulkan::publishVulkanQueueProperties(record, rachis::QueueType::graphics);
    const auto copyProps = rachis::vulkan::publishVulkanQueueProperties(record, rachis::QueueType::copy);
    CHECK(graphicsProps.timestampQueries);
    CHECK_FALSE(copyProps.timestampQueries);

    const auto device = fakeHandle<VkDevice>(0x20);
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeHandle<VkQueue>(0x30),
        .compute = fakeHandle<VkQueue>(0x30),
        .copy = fakeHandle<VkQueue>(0x31),
        .families = record.queues,
    };

    rachis::DeviceCapabilities caps{};
    caps.occlusionQueries = true;
    caps.pipelineStatisticsQueries = true;

    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, &instanceTable, record.physical, record.queues);

    {
        auto tableWithoutHostReset = table;
        tableWithoutHostReset.vkResetQueryPool = nullptr;
        rachis::vulkan::VulkanQueryManager brokenQueries;
        brokenQueries.initialize(device, &tableWithoutHostReset, &caps, &resources, false, &instanceTable);
        const auto rejected = brokenQueries.createOcclusionQueryPool({.queryCount = 4});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
    }

    rachis::vulkan::VulkanQueryManager queries;
    queries.initialize(device, &table, &caps, &resources, false, &instanceTable, {true, true, false});

    const auto unsupportedCopyTimestamps =
        queries.createTimestampQueryPool({.queueType = rachis::QueueType::copy, .queryCount = 2});
    REQUIRE_FALSE(unsupportedCopyTimestamps.has_value());
    CHECK(unsupportedCopyTimestamps.error().code == rachis::ErrorCode::unsupported);

    const auto occlusionPool = queries.createOcclusionQueryPool({.queryCount = 4});
    REQUIRE(occlusionPool.has_value());
    CHECK(mock.hostQueryResetCount == 1);

    constexpr std::uint64_t kDeviceToken = 0xABC0;
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, kDeviceToken, &resources);
    rachis::vulkan::VulkanQueueManager queueManager;
    REQUIRE(queueManager.initialize(device, &table, queueHandles, kDeviceToken).has_value());

    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 8});
    REQUIRE(cmdPool.has_value());

    {
        auto cb = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cb.has_value());
        auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);
        vkCb.graphicsState().renderingOwner = std::make_shared<rachis::vulkan::VulkanRenderingScope>();
        REQUIRE(queries.beginOcclusionQuery(vkCb, *occlusionPool, 0).has_value());
        REQUIRE(queries.endOcclusionQuery(vkCb).has_value());
        vkCb.graphicsState().renderingOwner.reset();

        std::array batch{std::move(*cb)};
        const auto submitted = queueManager.submit(batch, {});
        REQUIRE(submitted.has_value());
        CHECK(mock.lastSubmittedCommands.size() == 1);
        CHECK(mock.cmdQueryResets.empty());
    }

    {
        auto cb = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cb.has_value());
        auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);
        const VkCommandBuffer rawUserCb = vkCb.native();
        vkCb.graphicsState().renderingOwner = std::make_shared<rachis::vulkan::VulkanRenderingScope>();
        REQUIRE(queries.beginOcclusionQuery(vkCb, *occlusionPool, 0).has_value());
        REQUIRE(queries.endOcclusionQuery(vkCb).has_value());
        vkCb.graphicsState().renderingOwner.reset();

        std::array batch{std::move(*cb)};
        const auto submitted = queueManager.submit(batch, {});
        REQUIRE(submitted.has_value());
        REQUIRE(mock.lastSubmittedCommands.size() == 2);
        CHECK(mock.lastSubmittedCommands[1] == rawUserCb);
        REQUIRE(mock.cmdQueryResets.size() == 1);
        CHECK(mock.cmdQueryResets[0].command == mock.lastSubmittedCommands[0]);
        CHECK(mock.cmdQueryResets[0].firstQuery == 0);
        CHECK(mock.cmdQueryResets[0].queryCount == 1);
    }

    mock.cmdQueryResets.clear();
    {
        auto cbA = commands.acquireCommandBuffer(*cmdPool);
        auto cbB = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cbA.has_value());
        REQUIRE(cbB.has_value());
        for (auto* ptr : {cbA->get(), cbB->get()})
        {
            auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(*ptr);
            vkCb.graphicsState().renderingOwner = std::make_shared<rachis::vulkan::VulkanRenderingScope>();
            REQUIRE(queries.beginOcclusionQuery(vkCb, *occlusionPool, 0).has_value());
            REQUIRE(queries.endOcclusionQuery(vkCb).has_value());
            vkCb.graphicsState().renderingOwner.reset();
        }
        std::array batch{std::move(*cbA), std::move(*cbB)};
        const auto submitted = queueManager.submit(batch, {});
        REQUIRE(submitted.has_value());
        CHECK(mock.lastSubmittedCommands.size() == 4);
        CHECK(mock.cmdQueryResets.size() == 2);

        const auto status = queueManager.queryTimelinePoint(*submitted);
        REQUIRE(status.has_value());
        CHECK(*status == rachis::TimelinePointStatus::complete);
        queueManager.collectGarbage();

        auto cbReuse = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cbReuse.has_value());
        auto& vkCbReuse = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cbReuse);
        vkCbReuse.graphicsState().renderingOwner = std::make_shared<rachis::vulkan::VulkanRenderingScope>();
        REQUIRE(queries.beginOcclusionQuery(vkCbReuse, *occlusionPool, 0).has_value());
        REQUIRE(queries.endOcclusionQuery(vkCbReuse).has_value());
        vkCbReuse.graphicsState().renderingOwner.reset();

        std::array reuseBatch{std::move(*cbReuse)};
        const auto reuseSubmitted = queueManager.submit(reuseBatch, {});
        REQUIRE(reuseSubmitted.has_value());
    }

    {
        const auto largeTimestampPool =
            queries.createTimestampQueryPool({.queueType = rachis::QueueType::graphics, .queryCount = 16384});
        REQUIRE(largeTimestampPool.has_value());

        auto cbLarge = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cbLarge.has_value());
        auto& vkCbLarge = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cbLarge);
        REQUIRE(queries.resetQueryPool(vkCbLarge, *largeTimestampPool, 0, 16384).has_value());

        std::array largeBatch{std::move(*cbLarge)};
        REQUIRE(queueManager.submit(largeBatch, {}).has_value());
    }

    {
        const auto timestampPool =
            queries.createTimestampQueryPool({.queueType = rachis::QueueType::graphics, .queryCount = 4});
        const auto statisticsPool = queries.createPipelineStatisticsQueryPool({.queryCount = 1});
        REQUIRE(timestampPool.has_value());
        REQUIRE(statisticsPool.has_value());
        auto cb = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cb.has_value());
        auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);
        const auto expectUnsupported = [](const rachis::Status& status) {
            REQUIRE_FALSE(status.has_value());
            CHECK(status.error().code == rachis::ErrorCode::unsupported);
        };

        REQUIRE(queries
                    .beginGpuProfileScope(vkCb,
                                          {
                                              .queryPool = *timestampPool,
                                              .beginningQueryIndex = 2,
                                              .endQueryIndex = 3,
                                              .label = "outer",
                                          })
                    .has_value());
        auto multiview = std::make_shared<rachis::vulkan::VulkanRenderingScope>();
        multiview->viewMask = 0b110;
        vkCb.graphicsState().renderingOwner = multiview;
        mock.timelineEvents.clear();
        mock.cmdQueryResets.clear();
        const auto pendingTimestamps = vkCb.queryState().timestampWrites.size();
        expectUnsupported(queries.writeTimestamp(vkCb, *timestampPool, 0));
        expectUnsupported(queries.beginGpuProfileScope(vkCb,
                                                       {
                                                           .queryPool = *timestampPool,
                                                           .beginningQueryIndex = 0,
                                                           .endQueryIndex = 1,
                                                           .label = "multiview",
                                                       }));
        expectUnsupported(queries.endGpuProfileScope(vkCb));
        expectUnsupported(queries.beginOcclusionQuery(vkCb, *occlusionPool, 1));
        expectUnsupported(queries.beginPipelineStatisticsQuery(vkCb, *statisticsPool, 0));
        CHECK(mock.timelineEvents.empty());
        CHECK(mock.cmdQueryResets.empty());
        CHECK(vkCb.queryState().timestampWrites.size() == pendingTimestamps);
        CHECK_FALSE(vkCb.queryState().activeOcclusion.has_value());
        CHECK_FALSE(vkCb.queryState().activePipelineStatistics.has_value());
        CHECK(vkCb.queryState().activeGpuProfileScopeDepth == 1);
        CHECK(vkCb.debugState().groupDepth == 1);

        multiview->viewMask = 0b100;
        REQUIRE(queries.beginOcclusionQuery(vkCb, *occlusionPool, 1).has_value());
        REQUIRE(queries.endOcclusionQuery(vkCb).has_value());
        REQUIRE(queries.beginPipelineStatisticsQuery(vkCb, *statisticsPool, 0).has_value());
        REQUIRE(queries.endPipelineStatisticsQuery(vkCb).has_value());
        REQUIRE(queries.writeTimestamp(vkCb, *timestampPool, 0).has_value());
        REQUIRE(queries.endGpuProfileScope(vkCb).has_value());
        vkCb.graphicsState().renderingOwner.reset();

        std::array batch{std::move(*cb)};
        REQUIRE(queueManager.submit(batch, {}).has_value());
    }

    {
        const auto statisticsPool = queries.createPipelineStatisticsQueryPool({.queryCount = 2});
        REQUIRE(statisticsPool.has_value());
        auto cb = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cb.has_value());
        auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);
        const auto expectQueryKept = [&](const rachis::Status& status, std::size_t expectedWrites) {
            REQUIRE_FALSE(status.has_value());
            CHECK(status.error().code == rachis::ErrorCode::invalidState);
            CHECK(mock.timelineEvents.empty());
            CHECK(vkCb.queryState().activePipelineStatistics.has_value());
            CHECK(vkCb.queryState().pipelineStatisticsWrites.size() == expectedWrites);
        };

        REQUIRE(queries.beginPipelineStatisticsQuery(vkCb, *statisticsPool, 0).has_value());
        vkCb.graphicsState().renderingOwner = std::make_shared<rachis::vulkan::VulkanRenderingScope>();
        mock.timelineEvents.clear();
        auto pendingWrites = vkCb.queryState().pipelineStatisticsWrites.size();
        expectQueryKept(queries.endPipelineStatisticsQuery(vkCb), pendingWrites);
        vkCb.graphicsState().renderingOwner.reset();
        REQUIRE(queries.endPipelineStatisticsQuery(vkCb).has_value());

        vkCb.graphicsState().renderingOwner = std::make_shared<rachis::vulkan::VulkanRenderingScope>();
        REQUIRE(queries.beginPipelineStatisticsQuery(vkCb, *statisticsPool, 1).has_value());
        vkCb.graphicsState().renderingOwner.reset();
        mock.timelineEvents.clear();
        pendingWrites = vkCb.queryState().pipelineStatisticsWrites.size();
        expectQueryKept(queries.endPipelineStatisticsQuery(vkCb), pendingWrites);
        vkCb.graphicsState().renderingOwner = std::make_shared<rachis::vulkan::VulkanRenderingScope>();
        REQUIRE(queries.endPipelineStatisticsQuery(vkCb).has_value());
        vkCb.graphicsState().renderingOwner.reset();
        CHECK_FALSE(vkCb.queryState().activePipelineStatistics.has_value());
    }

    queueManager.shutdown();
    commands.shutdown();
    queries.shutdown();
    resources.shutdown();
    gMock = nullptr;
}

TEST_CASE("Vulkan submission retirement fences recycle command buffers only after the fence signals",
          "[size:small][rhi][vulkan][vuid][retirement]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();

    const auto device = fakeHandle<VkDevice>(0x24);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeHandle<VkQueue>(0x34),
        .compute = fakeHandle<VkQueue>(0x34),
        .copy = fakeHandle<VkQueue>(0x34),
        .families = families,
        .retireOnSubmissionFence = true,
    };

    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, nullptr, VK_NULL_HANDLE, families);
    constexpr std::uint64_t kDeviceToken = 0xABC4;
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, kDeviceToken, &resources);
    rachis::vulkan::VulkanQueueManager queueManager;
    REQUIRE(queueManager.initialize(device, &table, queueHandles, kDeviceToken).has_value());

    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 1});
    REQUIRE(cmdPool.has_value());

    mock.fenceStatus = GENERATE(VK_NOT_READY, VK_ERROR_DEVICE_LOST);
    auto cb = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(cb.has_value());
    const VkCommandBuffer rawCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb).native();
    std::array batch{std::move(*cb)};
    const auto submitted = queueManager.submit(batch, {});
    REQUIRE(submitted.has_value());
    CHECK(mock.semaphoreCounter >= submitted->value());

    queueManager.collectGarbage();
    CHECK(mock.fenceResets == 0);
    auto blocked = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE_FALSE(blocked.has_value());
    CHECK(blocked.error().code == rachis::ErrorCode::resourceExhausted);

    mock.fenceStatus = VK_SUCCESS;
    queueManager.collectGarbage();
    CHECK(mock.fenceResets == 1);
    auto reused = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(reused.has_value());
    CHECK(static_cast<rachis::vulkan::VulkanCommandBuffer&>(**reused).native() == rawCb);
    reused->reset();

    queueManager.shutdown();
    commands.shutdown();
    resources.shutdown();
    gMock = nullptr;
}

TEST_CASE("Vulkan copy queue uses a transfer-only family only when its image transfer granularity is (1,1,1)",
          "[size:small][rhi][vulkan][vuid][queues]")
{
    const auto selectedFamilies = [](std::span<const VkQueueFamilyProperties> families) {
        auto records = discoverWithQueueFamilies(families);
        REQUIRE(records.has_value());
        REQUIRE(records->size() == 1);
        return records->front().queues;
    };

    const auto unit = selectedFamilies(kMockQueueFamilies);
    CHECK(unit.graphics == 0);
    CHECK(unit.compute == 0);
    CHECK(unit.transfer == 1);

    const std::array coarse{
        mockQueueFamily(kMockGraphicsFlags, {1, 1, 1}),
        mockQueueFamily(VK_QUEUE_TRANSFER_BIT, {16, 16, 8}),
    };
    const auto coarseFamilies = selectedFamilies(coarse);
    CHECK(coarseFamilies.graphics == 0);
    CHECK(coarseFamilies.compute == 0);
    CHECK(coarseFamilies.transfer == coarseFamilies.graphics);

    const std::array wholeLevel{
        mockQueueFamily(kMockGraphicsFlags, {1, 1, 1}),
        mockQueueFamily(VK_QUEUE_TRANSFER_BIT, {0, 0, 0}),
    };
    const auto wholeLevelFamilies = selectedFamilies(wholeLevel);
    CHECK(wholeLevelFamilies.graphics == 0);
    CHECK(wholeLevelFamilies.transfer == wholeLevelFamilies.graphics);

    const std::array partlyUnit{
        mockQueueFamily(kMockGraphicsFlags, {1, 1, 1}),
        mockQueueFamily(VK_QUEUE_TRANSFER_BIT, {1, 1, 8}),
        mockQueueFamily(VK_QUEUE_TRANSFER_BIT, {1, 16, 1}),
    };
    const auto partlyUnitFamilies = selectedFamilies(partlyUnit);
    CHECK(partlyUnitFamilies.transfer == partlyUnitFamilies.graphics);

    const std::array laterUnit{
        mockQueueFamily(kMockGraphicsFlags, {1, 1, 1}),
        mockQueueFamily(VK_QUEUE_TRANSFER_BIT, {16, 16, 8}),
        mockQueueFamily(VK_QUEUE_TRANSFER_BIT, {1, 1, 1}),
    };
    const auto laterUnitFamilies = selectedFamilies(laterUnit);
    CHECK(laterUnitFamilies.graphics == 0);
    CHECK(laterUnitFamilies.transfer == 2);

    const std::array computeOnly{
        mockQueueFamily(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT, {1, 1, 1}),
        mockQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, {1, 1, 1}),
        mockQueueFamily(VK_QUEUE_TRANSFER_BIT, {16, 16, 8}),
    };
    const auto computeOnlyFamilies = selectedFamilies(computeOnly);
    CHECK(computeOnlyFamilies.graphics == 0);
    CHECK(computeOnlyFamilies.compute == 1);
    CHECK(computeOnlyFamilies.transfer == 0);
}

TEST_CASE("Vulkan format capabilities come only from the optimal tiling that textures are created with",
          "[size:small][rhi][vulkan][vuid][formats]")
{
    using rachis::FormatFeature;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetLinearRichFormatProperties;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &mockGetPhysicalDeviceImageFormatProperties;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x14);

    for (const auto format : {rachis::Format::rgba8Unorm, rachis::Format::r32Uint, rachis::Format::d32Float})
    {
        const auto properties = rachis::vulkan::queryVulkanFormatProperties(instanceTable, physical, format);
        CHECK(rachis::hasAny(properties.features, FormatFeature::sampled));
        CHECK(rachis::hasAny(properties.features, FormatFeature::texture2D));
        CHECK(rachis::hasAny(properties.features, FormatFeature::copyDestination));
        for (const auto linearOnly : {FormatFeature::filterable,
                                      FormatFeature::storageRead,
                                      FormatFeature::storageWrite,
                                      FormatFeature::colorAttachment,
                                      FormatFeature::blendable,
                                      FormatFeature::logicOperation,
                                      FormatFeature::depthStencilAttachment,
                                      FormatFeature::copySource,
                                      FormatFeature::multisampleResolve})
        {
            CHECK_FALSE(rachis::hasAny(properties.features, linearOnly));
        }
        CHECK(properties.colorResolveModes == rachis::ResolveModeMask::none);
    }
}

TEST_CASE("Vulkan integer color formats advertise no multisample resolve", "[size:small][rhi][vulkan][vuid][formats]")
{
    using rachis::FormatFeature;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetPhysicalDeviceFormatProperties;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &mockGetPhysicalDeviceImageFormatProperties;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x15);

    for (const auto format : {rachis::Format::rgba8Unorm, rachis::Format::rgba16Float})
    {
        const auto properties = rachis::vulkan::queryVulkanFormatProperties(instanceTable, physical, format);
        CHECK(rachis::hasAny(properties.features, FormatFeature::multisampleResolve));
        CHECK(properties.colorResolveModes == rachis::ResolveModeMask::average);
    }
    for (const auto format :
         {rachis::Format::rgba8Uint, rachis::Format::r32Uint, rachis::Format::rgba16Sint, rachis::Format::rg32Uint})
    {
        const auto properties = rachis::vulkan::queryVulkanFormatProperties(instanceTable, physical, format);
        CHECK(rachis::hasAny(properties.features, FormatFeature::colorAttachment));
        CHECK(rachis::hasAny(properties.features, FormatFeature::logicOperation));
        CHECK(rachis::supportsSampleCount(properties.sampleCounts, 4));
        CHECK_FALSE(rachis::hasAny(properties.features, FormatFeature::multisampleResolve));
        CHECK(properties.colorResolveModes == rachis::ResolveModeMask::none);
    }
}

TEST_CASE("Vulkan query pool creation leaks no native pool when a host allocation or a native creation fails",
          "[size:small][rhi][vulkan][vuid][queries]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkSetDebugUtilsObjectNameEXT = &mockSetDebugUtilsObjectName;

    const auto device = fakeHandle<VkDevice>(0x28);
    rachis::DeviceCapabilities caps{};
    caps.occlusionQueries = true;
    caps.pipelineStatisticsQueries = true;
    caps.meshPipelineStatisticsQueries = true;

    constexpr std::string_view label = "query pool label that outgrows every small-string buffer";
    constexpr std::uint32_t maxInjectedFailures = 64;
    using CreatePool =
        rachis::Result<rachis::QueryPoolHandle> (*)(rachis::vulkan::VulkanQueryManager&, std::string_view);
    struct PoolKind final
    {
        const char* name = nullptr;
        std::size_t nativePools = 0;
        CreatePool create = nullptr;
    };
    const std::array kinds{
        PoolKind{
            .name = "timestamp",
            .nativePools = 1,
            .create =
                [](rachis::vulkan::VulkanQueryManager& queries, std::string_view poolLabel) {
                    return queries.createTimestampQueryPool(
                        {.queueType = rachis::QueueType::graphics, .queryCount = 4, .label = poolLabel});
                },
        },
        PoolKind{
            .name = "occlusion",
            .nativePools = 1,
            .create =
                [](rachis::vulkan::VulkanQueryManager& queries, std::string_view poolLabel) {
                    return queries.createOcclusionQueryPool(
                        {.mode = rachis::OcclusionQueryMode::precise, .queryCount = 4, .label = poolLabel});
                },
        },
        PoolKind{
            .name = "pipeline statistics",
            .nativePools = 1,
            .create =
                [](rachis::vulkan::VulkanQueryManager& queries, std::string_view poolLabel) {
                    return queries.createPipelineStatisticsQueryPool(
                        {.mode = rachis::PipelineStatisticsQueryMode::base, .queryCount = 4, .label = poolLabel});
                },
        },
        PoolKind{
            .name = "mesh pipeline statistics",
            .nativePools = 2,
            .create =
                [](rachis::vulkan::VulkanQueryManager& queries, std::string_view poolLabel) {
                    return queries.createPipelineStatisticsQueryPool({
                        .mode = rachis::PipelineStatisticsQueryMode::meshShader,
                        .queryCount = 4,
                        .label = poolLabel,
                    });
                },
        },
    };

    for (const auto& kind : kinds)
    {
        CAPTURE(kind.name);
        rachis::vulkan::VulkanQueryManager queries;
        queries.initialize(device, &table, &caps, nullptr, false, &instanceTable);
        bool injected = false;
        bool created = false;
        for (std::uint32_t failure = 1; failure <= maxInjectedFailures && !created; ++failure)
        {
            CAPTURE(failure);
            mock.createdQueryPoolCount = 0;
            mock.namedQueryPool = 0;
            mock.queryPoolNameLength = 0;
            const auto resetsBefore = mock.hostQueryResetCount;
            std::optional<rachis::Result<rachis::QueryPoolHandle>> result;
            bool threw = false;
            mock.hostAllocationFailureAfterQueryPoolCreate = failure;
            try
            {
                result.emplace(kind.create(queries, label));
            }
            catch (const std::bad_alloc&)
            {
                threw = true;
            }
            gHostAllocationsUntilFailure = 0;
            mock.hostAllocationFailureAfterQueryPoolCreate = 0;
            if (std::exchange(gHostAllocationFailureInjected, false))
            {
                injected = true;
                const bool reportedOutOfMemory = result.has_value() && !result->has_value() &&
                                                 result->error().code == rachis::ErrorCode::outOfMemory;
                CHECK((threw || reportedOutOfMemory));
                REQUIRE(mock.liveQueryPools == 0);
                continue;
            }
            REQUIRE_FALSE(threw);
            REQUIRE(result.has_value());
            REQUIRE(result->has_value());
            created = true;
            CHECK(mock.liveQueryPools == kind.nativePools);
            CHECK(mock.createdQueryPoolCount == kind.nativePools);
            CHECK(mock.hostQueryResetCount - resetsBefore == kind.nativePools);
            CHECK(mock.namedQueryPool == reinterpret_cast<std::uint64_t>(mock.createdQueryPools[0]));
            CHECK(std::string_view(mock.queryPoolName.data(), mock.queryPoolNameLength) == label);
            REQUIRE(queries.destroyQueryPool(**result).has_value());
            CHECK(mock.liveQueryPools == 0);
        }
        CHECK(injected);
        CHECK(created);
        queries.shutdown();
    }

    {
        rachis::vulkan::VulkanQueryManager queries;
        queries.initialize(device, &table, &caps, nullptr, false, &instanceTable);
        mock.failedQueryPoolResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        for (const auto failedType : {VK_QUERY_TYPE_PIPELINE_STATISTICS, VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT})
        {
            CAPTURE(failedType);
            mock.failedQueryType = failedType;
            const auto resetsBefore = mock.hostQueryResetCount;
            const auto rejected = queries.createPipelineStatisticsQueryPool({
                .mode = rachis::PipelineStatisticsQueryMode::meshShader,
                .queryCount = 4,
                .label = label,
            });
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
            CHECK(rejected.error().nativeCode == VK_ERROR_OUT_OF_DEVICE_MEMORY);
            CHECK(rejected.error().message.find("vkCreateQueryPool") != std::string::npos);
            CHECK(mock.liveQueryPools == 0);
            CHECK(mock.poisonedQueryPoolDestroys == 0);
            CHECK(mock.hostQueryResetCount - resetsBefore ==
                  (failedType == VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT ? 1U : 0U));
        }
        mock.failedQueryType = VK_QUERY_TYPE_MAX_ENUM;
        mock.failedQueryPoolResult = VK_SUCCESS;
        queries.shutdown();
    }
    CHECK(mock.liveQueryPools == 0);
    gMock = nullptr;
}

TEST_CASE("Vulkan queue clock calibration pairs the device domain with the platform host domain or falls back",
          "[size:small][rhi][vulkan][queries][calibration]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    auto table = makeMockDeviceTable();
    table.vkGetCalibratedTimestampsEXT = &mockGetCalibratedTimestamps;

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = &mockGetPhysicalDeviceCalibrateableTimeDomains;

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = fakeHandle<VkPhysicalDevice>(0x14);
    record.properties.limits.timestampComputeAndGraphics = VK_TRUE;
    record.properties.limits.timestampPeriod = 1.0F;
    rachis::vulkan::recordVulkanQueueFamilies(record, kMockQueueFamilies);
    REQUIRE(record.queues.transfer == 1);
    const auto device = fakeHandle<VkDevice>(0x24);
    rachis::DeviceCapabilities caps{};

    const auto hostDomain = rachis::test::vulkanHostTimeDomain();
    const std::array everyDomain{
        VK_TIME_DOMAIN_DEVICE_EXT,
        VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT,
        VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT,
        VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT,
    };
    std::vector<VkTimeDomainEXT> withoutHostDomain;
    for (const auto domain : everyDomain)
    {
        if (domain != hostDomain)
        {
            withoutHostDomain.push_back(domain);
        }
    }

    const auto expectHostClockFallback = [&](const rachis::vulkan::VulkanQueryManager& queries,
                                             rachis::QueueType queueType) {
        mock.calibratedDomains.clear();
        const auto hostBefore = rachis::test::readVulkanHostClock();
        const auto calibration = queries.calibrateQueueClock(record.physical, queueType, record);
        const auto hostAfter = rachis::test::readVulkanHostClock();
        REQUIRE(calibration.has_value());
        REQUIRE(hostBefore.frequency > 0);
        CAPTURE(hostBefore.timestamp, calibration->cpuTimestamp, hostAfter.timestamp);
        CHECK(mock.calibratedDomains.empty());
        CHECK(calibration->gpuTimestamp == 0);
        CHECK(calibration->cpuTimestamp >= hostBefore.timestamp);
        CHECK(calibration->cpuTimestamp <= hostAfter.timestamp);
        CHECK(calibration->cpuFrequency == hostBefore.frequency);
        CHECK(calibration->gpuFrequency == 1'000'000'000ULL);
    };

    {
        rachis::vulkan::VulkanQueryManager queries;
        queries.initialize(device, &table, &caps, nullptr, true, &instanceTable);
        mock.calibrateableDomains.assign(everyDomain.begin(), everyDomain.end());
        for (const auto queueType : std::array{rachis::QueueType::graphics, rachis::QueueType::compute})
        {
            mock.calibratedDomains.clear();
            const auto calibration = queries.calibrateQueueClock(record.physical, queueType, record);
            REQUIRE(calibration.has_value());
            REQUIRE(mock.calibratedDomains.size() == 2);
            CHECK(mock.calibratedDomains[0] == VK_TIME_DOMAIN_DEVICE_EXT);
            CHECK(mock.calibratedDomains[1] == hostDomain);
            CHECK(calibration->gpuTimestamp == mockCalibratedDeviceTimestamp);
            CHECK(calibration->cpuTimestamp == mockCalibratedHostTimestamp);
            CHECK(calibration->gpuFrequency == 1'000'000'000ULL);
            CHECK(calibration->cpuFrequency == rachis::test::readVulkanHostClock().frequency);
        }
        expectHostClockFallback(queries, rachis::QueueType::copy);

        mock.calibrateableDomains = withoutHostDomain;
        expectHostClockFallback(queries, rachis::QueueType::graphics);

        mock.calibrateableDomains = {hostDomain};
        expectHostClockFallback(queries, rachis::QueueType::graphics);
        queries.shutdown();
    }

    {
        rachis::vulkan::VulkanQueryManager queries;
        queries.initialize(device, &table, &caps, nullptr, false, &instanceTable);
        mock.calibrateableDomains.assign(everyDomain.begin(), everyDomain.end());
        expectHostClockFallback(queries, rachis::QueueType::graphics);
        queries.shutdown();
    }
    gMock = nullptr;
}

TEST_CASE("Vulkan queue properties come from queue families read once at discovery and never allocate",
          "[size:small][rhi][vulkan][queries]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    auto table = makeMockDeviceTable();
    table.vkGetCalibratedTimestampsEXT = &mockGetCalibratedTimestamps;

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkEnumeratePhysicalDevices = &mockEnumeratePhysicalDevices;
    instanceTable.vkGetPhysicalDeviceProperties = &mockGetPhysicalDeviceProperties;
    instanceTable.vkGetPhysicalDeviceMemoryProperties = &mockGetPhysicalDeviceMemoryProperties;
    instanceTable.vkGetPhysicalDeviceProperties2 = &mockGetPhysicalDeviceProperties2;
    instanceTable.vkGetPhysicalDeviceFeatures2 = &mockGetPhysicalDeviceFeatures2;
    instanceTable.vkEnumerateDeviceExtensionProperties = &mockEnumerateDeviceExtensionProperties;
    instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetPhysicalDeviceFormatProperties;
    instanceTable.vkGetPhysicalDeviceQueueFamilyProperties = &mockGetPhysicalDeviceQueueFamilyProperties;
    instanceTable.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = &mockGetPhysicalDeviceCalibrateableTimeDomains;

    const auto instance = fakeHandle<VkInstance>(0x06);
    const rachis::VulkanInstanceReport instanceReport{};
    const auto discovered = rachis::vulkan::discoverVulkanPhysicalDevices(instance, instanceTable, instanceReport);
    REQUIRE(discovered.has_value());
    REQUIRE(discovered->size() == 1);
    const auto& record = discovered->front();
    CHECK(record.queues.graphics == 0);
    CHECK(record.queues.compute == 0);
    CHECK(record.queues.transfer == 1);
    const auto discoveryQueries = mock.queueFamilyPropertyQueries;
    CHECK(discoveryQueries == 2);

    constexpr std::array queueTypes{rachis::QueueType::graphics, rachis::QueueType::compute, rachis::QueueType::copy};
    const auto expectQueueProperties = [](const rachis::QueueProperties& properties, rachis::QueueType queueType) {
        const bool timestamps = queueType != rachis::QueueType::copy;
        CHECK(properties.timestampQueries == timestamps);
        CHECK(properties.inlineTimestampWrites == timestamps);
        CHECK(properties.timestampValidBits == (timestamps ? 64U : 48U));
        CHECK(properties.timestampFrequency == 1'000'000'000ULL);
    };
    const auto makeDevice = [&] {
        const rachis::vulkan::VulkanQueueHandles queueHandles{
            .graphics = fakeHandle<VkQueue>(0x36),
            .compute = fakeHandle<VkQueue>(0x36),
            .copy = fakeHandle<VkQueue>(0x37),
            .families = record.queues,
        };
        auto backend = std::make_unique<rachis::vulkan::VulkanDevice>(
            record, fakeHandle<VkDevice>(0x26), table, queueHandles, rachis::DeviceCapabilities{}, 0xABC3);
        rachis::vulkan::VulkanInstanceInfo instanceInfo{};
        instanceInfo.table = instanceTable;
        backend->adoptInstance(std::move(instanceInfo));
        return rachis::detail::DeviceAccess::make(std::move(backend));
    };

    SECTION("a host allocation failure while discovery reads the queue families escapes to the frontend guard")
    {
        std::optional<rachis::Result<std::vector<rachis::vulkan::VulkanPhysicalDeviceRecord>>> result;
        bool threw = false;
        mock.hostAllocationFailureAfterQueueFamilyCount = 1;
        try
        {
            result.emplace(rachis::vulkan::discoverVulkanPhysicalDevices(instance, instanceTable, instanceReport));
        }
        catch (const std::bad_alloc&)
        {
            threw = true;
        }
        gHostAllocationsUntilFailure = 0;
        mock.hostAllocationFailureAfterQueueFamilyCount = 0;
        CHECK(std::exchange(gHostAllocationFailureInjected, false));
        const bool reportedOutOfMemory =
            result.has_value() && !result->has_value() && result->error().code == rachis::ErrorCode::outOfMemory;
        CHECK((threw || reportedOutOfMemory));
    }

    SECTION("Device::queueProperties reads the discovered families without allocating")
    {
        auto device = makeDevice();
        for (const auto queueType : queueTypes)
        {
            CAPTURE(static_cast<int>(queueType));
            gHostAllocationsUntilFailure = 1;
            const auto properties = device.queueProperties(queueType);
            const auto unconsumed = std::exchange(gHostAllocationsUntilFailure, 0U);
            CHECK(unconsumed == 1);
            CHECK_FALSE(std::exchange(gHostAllocationFailureInjected, false));
            expectQueueProperties(properties, queueType);
        }
        CHECK(mock.queueFamilyPropertyQueries == discoveryQueries);
    }

    SECTION("Device::createTimestampQueryPool reports outOfMemory instead of terminating")
    {
        auto device = makeDevice();
        const auto unsupported =
            device.createTimestampQueryPool({.queueType = rachis::QueueType::copy, .queryCount = 2});
        REQUIRE_FALSE(unsupported.has_value());
        CHECK(unsupported.error().code == rachis::ErrorCode::unsupported);
        for (const auto queueType : queueTypes)
        {
            CAPTURE(static_cast<int>(queueType));
            gHostAllocationsUntilFailure = 1;
            const auto pool = device.createTimestampQueryPool({.queueType = queueType, .queryCount = 2});
            gHostAllocationsUntilFailure = 0;
            CHECK(std::exchange(gHostAllocationFailureInjected, false));
            REQUIRE_FALSE(pool.has_value());
            CHECK(pool.error().code == rachis::ErrorCode::outOfMemory);
        }
        CHECK(mock.queueFamilyPropertyQueries == discoveryQueries);
    }

    SECTION("calibrateQueueClock surfaces a host allocation failure instead of terminating")
    {
        mock.calibrateableDomains = {VK_TIME_DOMAIN_DEVICE_EXT, rachis::test::vulkanHostTimeDomain()};
        mock.calibratedDomains.reserve(2);
        rachis::DeviceCapabilities caps{};
        rachis::vulkan::VulkanQueryManager queries;
        queries.initialize(fakeHandle<VkDevice>(0x26), &table, &caps, nullptr, true, &instanceTable);
        bool injected = false;
        bool calibrated = false;
        for (std::uint32_t failure = 1; failure <= 8 && !calibrated; ++failure)
        {
            CAPTURE(failure);
            mock.calibratedDomains.clear();
            std::optional<rachis::Result<rachis::QueueClockCalibration>> result;
            bool threw = false;
            gHostAllocationsUntilFailure = failure;
            try
            {
                result.emplace(queries.calibrateQueueClock(record.physical, rachis::QueueType::graphics, record));
            }
            catch (const std::bad_alloc&)
            {
                threw = true;
            }
            gHostAllocationsUntilFailure = 0;
            if (std::exchange(gHostAllocationFailureInjected, false))
            {
                injected = true;
                const bool reportedOutOfMemory = result.has_value() && !result->has_value() &&
                                                 result->error().code == rachis::ErrorCode::outOfMemory;
                CHECK((threw || reportedOutOfMemory));
                continue;
            }
            REQUIRE_FALSE(threw);
            REQUIRE(result.has_value());
            REQUIRE(result->has_value());
            CHECK((*result)->gpuTimestamp == mockCalibratedDeviceTimestamp);
            CHECK((*result)->cpuTimestamp == mockCalibratedHostTimestamp);
            CHECK((*result)->gpuFrequency == 1'000'000'000ULL);
            calibrated = true;
        }
        CHECK(injected);
        CHECK(calibrated);
        CHECK(mock.queueFamilyPropertyQueries == discoveryQueries);
        queries.shutdown();
    }
    gMock = nullptr;
}

TEST_CASE("Vulkan graphics encoder and pipeline manager guard depthBiasClamp, push constants, and VRS splits",
          "[size:small][rhi][vulkan][vuid][graphics-mock]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetPhysicalDeviceFormatProperties;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &mockGetPhysicalDeviceImageFormatProperties;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x11);
    const auto device = fakeHandle<VkDevice>(0x21);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeHandle<VkQueue>(0x31),
        .compute = fakeHandle<VkQueue>(0x31),
        .copy = fakeHandle<VkQueue>(0x31),
        .families = families,
    };

    rachis::DeviceCapabilities caps{};
    caps.variableRateShading = true;
    caps.variableRateShadingTier2 = true;
    caps.shadingRateImageTileSize = 16;
    caps.rayTracing = true;

    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, &instanceTable, physical, families);

    rachis::vulkan::VulkanPipelineManager pipelinesNoClamp;
    pipelinesNoClamp.initialize(device, &table, physical, caps, nullptr, nullptr, &instanceTable, false);
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const auto rejectedPipeline = pipelinesNoClamp.createGraphicsPipeline({
        .depthBiasClamp = 0.5F,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(rejectedPipeline.has_value());
    CHECK(rejectedPipeline.error().code == rachis::ErrorCode::unsupported);

    rachis::vulkan::VulkanGraphicsEncoder graphicsNoClamp;
    graphicsNoClamp.initialize(device, &table, &caps, &resources, &pipelinesNoClamp, nullptr, false);

    constexpr std::uint64_t kDeviceToken = 0xABC1;
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, kDeviceToken, &resources);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 8});
    REQUIRE(cmdPool.has_value());
    auto cb = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(cb.has_value());
    auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);

    const auto colorTex = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x4001),
                                                        {
                                                            .width = 64,
                                                            .height = 64,
                                                            .format = rachis::Format::rgba8Unorm,
                                                            .usage = rachis::TextureUsage::renderTarget,
                                                        });
    REQUIRE(colorTex.has_value());
    const auto colorView = resources.createTextureView({
        .texture = *colorTex,
        .usage = rachis::TextureViewUsage::renderTarget,
        .format = rachis::Format::rgba8Unorm,
    });
    REQUIRE(colorView.has_value());
    const auto vrsTex = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x4002),
                                                      {
                                                          .width = 4,
                                                          .height = 4,
                                                          .format = rachis::Format::r8Uint,
                                                          .usage = rachis::TextureUsage::shadingRateSource,
                                                      });
    REQUIRE(vrsTex.has_value());

    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *colorView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::discard,
        },
    };
    REQUIRE(graphicsNoClamp.beginRendering(vkCb, {.colorAttachments = colorAttachments}).has_value());
    REQUIRE(mock.beginRenderings.size() == 1);

    CHECK(mock.beginRenderings[0].firstColorStoreOp == VK_ATTACHMENT_STORE_OP_STORE);

    auto gfxPipe = std::make_shared<rachis::vulkan::VulkanNativePipeline>();
    gfxPipe->pipeline = fakeHandle<VkPipeline>(0x5001);
    gfxPipe->layout = fakeHandle<VkPipelineLayout>(0x5002);
    gfxPipe->dynamicDepthBias = true;
    gfxPipe->pushConstantRanges = {
        {.visibility = rachis::ShaderStage::vertex, .offset = 0, .size = 16},
        {.visibility = rachis::ShaderStage::pixel, .offset = 8, .size = 16},
    };
    gfxPipe->nativePushConstantRanges =
        rachis::vulkan::validation_detail::mergePushConstantRangesPerStage(gfxPipe->pushConstantRanges);
    vkCb.graphicsState().pipelineOwner = gfxPipe;
    vkCb.graphicsState().depthBiasRequired = true;

    CHECK(graphicsNoClamp.setDepthBias(vkCb, {.constantFactor = 1.0F, .clamp = 0.0F, .slopeScale = 1.0F}).has_value());
    CHECK(mock.setDepthBiasCount == 1);
    const auto clampedBias =
        graphicsNoClamp.setDepthBias(vkCb, {.constantFactor = 1.0F, .clamp = 0.25F, .slopeScale = 1.0F});
    REQUIRE_FALSE(clampedBias.has_value());
    CHECK(clampedBias.error().code == rachis::ErrorCode::unsupported);
    CHECK(mock.setDepthBiasCount == 1);

    std::array<std::byte, 16> pushData{};
    REQUIRE(graphicsNoClamp.setGraphicsPushConstants(vkCb, rachis::ShaderStage::vertex, 0, pushData).has_value());
    REQUIRE(mock.pushConstants.size() == 2);
    CHECK(mock.pushConstants[0].stageFlags == VK_SHADER_STAGE_VERTEX_BIT);
    CHECK(mock.pushConstants[0].offset == 0);
    CHECK(mock.pushConstants[0].size == 8);
    CHECK(mock.pushConstants[1].stageFlags == (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
    CHECK(mock.pushConstants[1].offset == 8);
    CHECK(mock.pushConstants[1].size == 8);

    gfxPipe->pushConstantRanges = {
        {.visibility = rachis::ShaderStage::pixel, .offset = 0, .size = 16},
        {.visibility = rachis::ShaderStage::vertex, .offset = 16, .size = 16},
        {.visibility = rachis::ShaderStage::pixel, .offset = 32, .size = 16},
    };
    gfxPipe->nativePushConstantRanges =
        rachis::vulkan::validation_detail::mergePushConstantRangesPerStage(gfxPipe->pushConstantRanges);
    mock.pushConstants.clear();
    REQUIRE(graphicsNoClamp.setGraphicsPushConstants(vkCb, rachis::ShaderStage::vertex, 16, pushData).has_value());
    REQUIRE(mock.pushConstants.size() == 1);
    CHECK(mock.pushConstants[0].stageFlags == (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
    CHECK(mock.pushConstants[0].offset == 16);
    CHECK(mock.pushConstants[0].size == 16);
    CHECK_FALSE(graphicsNoClamp.setGraphicsPushConstants(vkCb, rachis::ShaderStage::pixel, 16, pushData).has_value());
    CHECK(mock.pushConstants.size() == 1);

    const auto narrowVrsTex = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x4003),
                                                            {
                                                                .width = 3,
                                                                .height = 4,
                                                                .format = rachis::Format::r8Uint,
                                                                .usage = rachis::TextureUsage::shadingRateSource,
                                                            });
    const auto shortVrsTex = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x4004),
                                                           {
                                                               .width = 4,
                                                               .height = 3,
                                                               .format = rachis::Format::r8Uint,
                                                               .usage = rachis::TextureUsage::shadingRateSource,
                                                           });
    REQUIRE(narrowVrsTex.has_value());
    REQUIRE(shortVrsTex.has_value());
    mock.memoryBarriers.clear();
    for (const auto undersized : {*narrowVrsTex, *shortVrsTex})
    {
        const auto uncovered = graphicsNoClamp.setShadingRateImage(vkCb, undersized);
        REQUIRE_FALSE(uncovered.has_value());
        CHECK(uncovered.error().code == rachis::ErrorCode::invalidArgument);
    }
    CHECK(mock.beginRenderings.size() == 1);
    CHECK(mock.memoryBarriers.empty());

    vkCb.queryState().activeOcclusion =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = true};
    const auto blockedByOcclusion = graphicsNoClamp.setShadingRateImage(vkCb, *vrsTex);
    REQUIRE_FALSE(blockedByOcclusion.has_value());
    CHECK(blockedByOcclusion.error().code == rachis::ErrorCode::invalidState);
    CHECK(mock.beginRenderings.size() == 1);
    vkCb.queryState().activeOcclusion.reset();

    vkCb.queryState().activePipelineStatistics =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = true};
    const auto blockedByInPassStats = graphicsNoClamp.setShadingRateImage(vkCb, *vrsTex);
    REQUIRE_FALSE(blockedByInPassStats.has_value());
    CHECK(blockedByInPassStats.error().code == rachis::ErrorCode::invalidState);

    vkCb.queryState().activePipelineStatistics =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = false};
    mock.memoryBarriers.clear();
    REQUIRE(graphicsNoClamp.setShadingRateImage(vkCb, *vrsTex).has_value());
    REQUIRE(mock.memoryBarriers.size() == 1);
    CHECK((mock.memoryBarriers[0].srcAccessMask & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) != 0);
    CHECK((mock.memoryBarriers[0].dstAccessMask & VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR) != 0);
    REQUIRE(mock.beginRenderings.size() == 2);
    CHECK(mock.beginRenderings[1].firstColorLoadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    CHECK(mock.beginRenderings[1].hasShadingRateAttachment);
    vkCb.queryState().activePipelineStatistics.reset();

    mock.timelineEvents.clear();
    const auto expectRenderingKept = [&](const rachis::Status& status) {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == rachis::ErrorCode::invalidState);
        CHECK(vkCb.graphicsState().renderingOwner != nullptr);
        CHECK(mock.timelineEvents.empty());
    };
    vkCb.queryState().activeOcclusion =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = true};
    expectRenderingKept(graphicsNoClamp.endRendering(vkCb));
    vkCb.queryState().activeOcclusion.reset();
    vkCb.queryState().activePipelineStatistics =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = true};
    expectRenderingKept(graphicsNoClamp.endRendering(vkCb));
    vkCb.queryState().activePipelineStatistics.reset();
    vkCb.queryState().activeStreamOutputStatistics[0] =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = true};
    expectRenderingKept(graphicsNoClamp.endRendering(vkCb));
    vkCb.queryState().activeStreamOutputStatistics[0].reset();

    vkCb.queryState().activePipelineStatistics =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = false};
    REQUIRE(graphicsNoClamp.endRendering(vkCb).has_value());
    CHECK(vkCb.graphicsState().renderingOwner == nullptr);
    vkCb.queryState().activePipelineStatistics.reset();

    {
        rachis::vulkan::VulkanQueryManager passQueries;
        passQueries.initialize(device, &table, &caps, &resources, false, &instanceTable, {true, true, false});
        const auto tsPool =
            passQueries.createTimestampQueryPool({.queueType = rachis::QueueType::graphics, .queryCount = 4});
        REQUIRE(tsPool.has_value());

        const rachis::RenderingDesc passDesc{
            .colorAttachments = colorAttachments,
            .timestampWrites =
                rachis::PassTimestampWrites{
                    .queryPool = *tsPool,
                    .beginningOfPassWriteIndex = 0,
                    .endOfPassWriteIndex = 1,
                },
        };
        auto prepared = passQueries.prepareRenderingTimestamps(vkCb, passDesc.timestampWrites);
        REQUIRE(prepared.has_value());
        REQUIRE(prepared->has_value());
        mock.timelineEvents.clear();
        REQUIRE(graphicsNoClamp
                    .beginRendering(vkCb, passDesc, [&]() { passQueries.beginRenderingTimestamps(vkCb, **prepared); })
                    .has_value());
        passQueries.commitRenderingTimestamps(vkCb, std::move(**prepared));
        REQUIRE(graphicsNoClamp.endRendering(vkCb).has_value());
        REQUIRE(passQueries.endRenderingTimestamps(vkCb).has_value());
        std::vector<CommandOpKind> passOrder;
        for (const auto& event : mock.timelineEvents)
        {
            if (event.kind == CommandOpKind::writeTimestamp || event.kind == CommandOpKind::beginRendering ||
                event.kind == CommandOpKind::endRendering)
            {
                passOrder.push_back(event.kind);
            }
        }
        CHECK(passOrder == std::vector{CommandOpKind::writeTimestamp,
                                       CommandOpKind::beginRendering,
                                       CommandOpKind::endRendering,
                                       CommandOpKind::writeTimestamp});
        vkCb.clearRecordingState();
        passQueries.shutdown();
    }

    auto rtPipe = std::make_shared<rachis::vulkan::VulkanNativePipeline>();
    rtPipe->pipeline = fakeHandle<VkPipeline>(0x5011);
    rtPipe->layout = fakeHandle<VkPipelineLayout>(0x5012);
    rtPipe->rayTracing = true;
    rtPipe->pushConstantRanges = {
        {
            .visibility = rachis::ShaderStage::rayGeneration | rachis::ShaderStage::anyHit |
                          rachis::ShaderStage::closestHit | rachis::ShaderStage::miss |
                          rachis::ShaderStage::intersection | rachis::ShaderStage::callable,
            .offset = 0,
            .size = 16,
        },
    };
    rtPipe->nativePushConstantRanges =
        rachis::vulkan::validation_detail::mergePushConstantRangesPerStage(rtPipe->pushConstantRanges);
    vkCb.rayTracingState().pipelineOwner = rtPipe;
    mock.pushConstants.clear();
    REQUIRE(graphicsNoClamp.setRayTracingPushConstants(vkCb, 0, pushData).has_value());
    REQUIRE(mock.pushConstants.size() == 1);
    CHECK(mock.pushConstants[0].stageFlags ==
          (VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
           VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR));

    cb->reset();
    REQUIRE(resources.destroyTextureView(*colorView).has_value());
    commands.shutdown();
    pipelinesNoClamp.shutdown();
    resources.shutdown();
    gMock = nullptr;
}

TEST_CASE(
    "Vulkan multi-CB submission bridges compatible transfer layouts, rejects drift, and validates resolve layouts",
    "[size:small][rhi][vulkan][vuid][layouts-mock]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetPhysicalDeviceFormatProperties;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &mockGetPhysicalDeviceImageFormatProperties;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x12);
    const auto device = fakeHandle<VkDevice>(0x22);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeHandle<VkQueue>(0x32),
        .compute = fakeHandle<VkQueue>(0x32),
        .copy = fakeHandle<VkQueue>(0x32),
        .families = families,
    };

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = physical;
    record.separateDepthStencilLayouts = true;
    record.independentResolveNone = true;
    record.independentResolve = true;
    record.depthResolveModes = VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT;
    record.stencilResolveModes = VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT;

    rachis::DeviceCapabilities caps{};
    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, &instanceTable, physical, families);
    rachis::vulkan::VulkanImageUtilities imageUtils;
    imageUtils.initialize(&record, &caps, &resources, nullptr, nullptr, nullptr);

    constexpr std::uint64_t kDeviceToken = 0xABC2;
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, kDeviceToken, &resources);
    rachis::vulkan::VulkanQueueManager queueManager;
    REQUIRE(queueManager.initialize(device, &table, queueHandles, kDeviceToken).has_value());

    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 8});
    REQUIRE(cmdPool.has_value());

    const auto srcTex =
        resources.adoptSwapchainImage(fakeHandle<VkImage>(0x6001),
                                      {
                                          .width = 8,
                                          .height = 8,
                                          .format = rachis::Format::rgba8Unorm,
                                          .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::storage,
                                      });
    const auto dstImage = fakeHandle<VkImage>(0x6002);
    const auto dstTex = resources.adoptSwapchainImage(
        dstImage,
        {
            .width = 8,
            .height = 8,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copyDestination | rachis::TextureUsage::storage |
                     rachis::TextureUsage::sampled | rachis::TextureUsage::renderTarget,
        });
    REQUIRE(srcTex.has_value());
    REQUIRE(dstTex.has_value());

    {
        auto initCb = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(initCb.has_value());
        auto& vkInit = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**initCb);
        REQUIRE(resources
                    .textureBarrier(vkInit.native(),
                                    rachis::QueueType::graphics,
                                    {
                                        .texture = *srcTex,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(resources
                    .textureBarrier(vkInit.native(),
                                    rachis::QueueType::graphics,
                                    {
                                        .texture = *dstTex,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        std::array batch{std::move(*initCb)};
        REQUIRE(queueManager.submit(batch, {}).has_value());
    }

    mock.imageBarriers.clear();
    {
        auto cb1 = commands.acquireCommandBuffer(*cmdPool);
        auto cb2 = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cb1.has_value());
        REQUIRE(cb2.has_value());
        auto& vkCb1 = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb1);
        auto& vkCb2 = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb2);
        const VkCommandBuffer rawCb1 = vkCb1.native();
        const VkCommandBuffer rawCb2 = vkCb2.native();

        REQUIRE(resources
                    .textureBarrier(vkCb1.native(),
                                    rachis::QueueType::graphics,
                                    {
                                        .texture = *dstTex,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::unorderedAccess,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                    })
                    .has_value());
        const std::array copyRegions{
            rachis::TextureCopyRegion{
                .extent = {.width = 8, .height = 8, .depth = 1},
            },
        };
        REQUIRE(resources.copyTextureRegions(vkCb2.native(), *srcTex, *dstTex, copyRegions).has_value());

        std::array batch{std::move(*cb1), std::move(*cb2)};
        const auto submitted = queueManager.submit(batch, {});
        REQUIRE(submitted.has_value());
        REQUIRE(mock.lastSubmittedCommands.size() == 4);
        CHECK(mock.lastSubmittedCommands[0] == rawCb1);
        CHECK(mock.lastSubmittedCommands[2] == rawCb2);

        const auto preBridgeCmd = mock.lastSubmittedCommands[1];
        const auto postBridgeCmd = mock.lastSubmittedCommands[3];
        const auto preIt = std::ranges::find_if(
            mock.imageBarriers, [&](const RecordedImageBarrier& b) { return b.command == preBridgeCmd; });
        const auto postIt = std::ranges::find_if(
            mock.imageBarriers, [&](const RecordedImageBarrier& b) { return b.command == postBridgeCmd; });
        REQUIRE(preIt != mock.imageBarriers.end());
        REQUIRE(postIt != mock.imageBarriers.end());
        CHECK(preIt->image == dstImage);
        CHECK(preIt->oldLayout == VK_IMAGE_LAYOUT_GENERAL);
        CHECK(preIt->newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        CHECK(postIt->image == dstImage);
        CHECK(postIt->oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        CHECK(postIt->newLayout == VK_IMAGE_LAYOUT_GENERAL);
    }

    {
        auto cb3 = commands.acquireCommandBuffer(*cmdPool);
        auto cb4 = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cb3.has_value());
        REQUIRE(cb4.has_value());
        auto& vkCb3 = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb3);
        auto& vkCb4 = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb4);

        REQUIRE(resources
                    .textureBarrier(vkCb3.native(),
                                    rachis::QueueType::graphics,
                                    {
                                        .texture = *dstTex,
                                        .stagesBefore = rachis::PipelineStage::computeShader,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = rachis::Access::unorderedAccess,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array copyRegions{
            rachis::TextureCopyRegion{
                .extent = {.width = 8, .height = 8, .depth = 1},
            },
        };
        REQUIRE(resources.copyTextureRegions(vkCb4.native(), *srcTex, *dstTex, copyRegions).has_value());

        const auto submitsBefore = mock.submitCount;
        std::array batch{std::move(*cb3), std::move(*cb4)};
        const auto rejected = queueManager.submit(batch, {});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        CHECK(mock.submitCount == submitsBefore);
        CHECK(batch[0] != nullptr);
        CHECK(batch[1] != nullptr);
    }

    {
        const auto msaaColor = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x6010),
                                                             {
                                                                 .width = 8,
                                                                 .height = 8,
                                                                 .sampleCount = 4,
                                                                 .format = rachis::Format::rgba8Unorm,
                                                                 .usage = rachis::TextureUsage::renderTarget,
                                                             });
        const auto singleColor = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x6011),
                                                               {
                                                                   .width = 8,
                                                                   .height = 8,
                                                                   .sampleCount = 1,
                                                                   .format = rachis::Format::rgba8Unorm,
                                                                   .usage = rachis::TextureUsage::renderTarget,
                                                               });
        REQUIRE(msaaColor.has_value());
        REQUIRE(singleColor.has_value());

        auto cb = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cb.has_value());
        auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);

        const auto unreadyColor = imageUtils.resolveTexture(vkCb, {.source = *msaaColor, .destination = *singleColor});
        REQUIRE_FALSE(unreadyColor.has_value());
        CHECK(unreadyColor.error().code == rachis::ErrorCode::invalidState);

        REQUIRE(resources
                    .textureBarrier(vkCb.native(),
                                    rachis::QueueType::graphics,
                                    {
                                        .texture = *msaaColor,
                                        .stagesAfter = rachis::PipelineStage::all,
                                        .accessAfter = rachis::Access::common,
                                        .layoutAfter = rachis::TextureLayout::common,
                                    })
                    .has_value());
        REQUIRE(resources
                    .textureBarrier(vkCb.native(),
                                    rachis::QueueType::graphics,
                                    {
                                        .texture = *singleColor,
                                        .stagesAfter = rachis::PipelineStage::all,
                                        .accessAfter = rachis::Access::common,
                                        .layoutAfter = rachis::TextureLayout::common,
                                    })
                    .has_value());
        mock.resolves.clear();
        REQUIRE(imageUtils.resolveTexture(vkCb, {.source = *msaaColor, .destination = *singleColor}).has_value());
        REQUIRE(mock.resolves.size() == 1);
        CHECK(mock.resolves[0].srcLayout == VK_IMAGE_LAYOUT_GENERAL);
        CHECK(mock.resolves[0].dstLayout == VK_IMAGE_LAYOUT_GENERAL);

        const auto msaaDs = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x6012),
                                                          {
                                                              .width = 8,
                                                              .height = 8,
                                                              .depthOrArrayLayers = 2,
                                                              .sampleCount = 4,
                                                              .format = rachis::Format::d24UnormS8Uint,
                                                              .usage = rachis::TextureUsage::depthStencil,
                                                          });
        const auto singleDs = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x6013),
                                                            {
                                                                .width = 8,
                                                                .height = 8,
                                                                .depthOrArrayLayers = 2,
                                                                .sampleCount = 1,
                                                                .format = rachis::Format::d24UnormS8Uint,
                                                                .usage = rachis::TextureUsage::depthStencil,
                                                            });
        REQUIRE(msaaDs.has_value());
        REQUIRE(singleDs.has_value());
        REQUIRE(resources
                    .textureBarrier(vkCb.native(),
                                    rachis::QueueType::graphics,
                                    {
                                        .texture = *msaaDs,
                                        .stagesAfter = rachis::PipelineStage::all,
                                        .accessAfter = rachis::Access::common,
                                        .layoutAfter = rachis::TextureLayout::common,
                                        .subresources =
                                            {
                                                .firstArrayLayer = 1,
                                                .arrayLayerCount = 1,
                                                .aspect = rachis::TextureAspect::stencil,
                                            },
                                    })
                    .has_value());
        REQUIRE(resources
                    .textureBarrier(vkCb.native(),
                                    rachis::QueueType::graphics,
                                    {
                                        .texture = *singleDs,
                                        .stagesAfter = rachis::PipelineStage::all,
                                        .accessAfter = rachis::Access::common,
                                        .layoutAfter = rachis::TextureLayout::common,
                                        .subresources =
                                            {
                                                .firstArrayLayer = 1,
                                                .arrayLayerCount = 1,
                                                .aspect = rachis::TextureAspect::stencil,
                                            },
                                    })
                    .has_value());

        mock.createdViewTypes.clear();
        mock.imageBarriers.clear();
        mock.beginRenderings.clear();
        REQUIRE(imageUtils
                    .resolveTexture(vkCb,
                                    {
                                        .source = *msaaDs,
                                        .sourceSubresource =
                                            {
                                                .arrayLayer = 1,
                                                .aspect = rachis::TextureAspect::stencil,
                                            },
                                        .destination = *singleDs,
                                        .destinationSubresource =
                                            {
                                                .arrayLayer = 1,
                                                .aspect = rachis::TextureAspect::stencil,
                                            },
                                        .mode = rachis::ResolveMode::maximum,
                                    })
                    .has_value());
        REQUIRE(mock.createdViewTypes.size() == 2);
        CHECK(mock.createdViewTypes[0] == VK_IMAGE_VIEW_TYPE_2D_ARRAY);
        CHECK(mock.createdViewTypes[1] == VK_IMAGE_VIEW_TYPE_2D_ARRAY);
        REQUIRE(mock.beginRenderings.size() == 1);
        CHECK_FALSE(mock.beginRenderings[0].hasDepth);
        CHECK(mock.beginRenderings[0].hasStencil);
        CHECK(mock.beginRenderings[0].stencilLayout == VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL);
        REQUIRE(mock.imageBarriers.size() == 4);
        CHECK(mock.imageBarriers[0].aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT);
        CHECK(mock.imageBarriers[0].oldLayout == VK_IMAGE_LAYOUT_GENERAL);
        CHECK(mock.imageBarriers[0].newLayout == VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL);
        CHECK(mock.imageBarriers[2].aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT);
        CHECK(mock.imageBarriers[2].oldLayout == VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL);
        CHECK(mock.imageBarriers[2].newLayout == VK_IMAGE_LAYOUT_GENERAL);
    }

    queueManager.shutdown();
    commands.shutdown();
    imageUtils.shutdown();
    resources.shutdown();
    gMock = nullptr;
}

TEST_CASE("Vulkan packed depth-stencil resolves need independent resolve only when both aspects are attached",
          "[size:small][rhi][vulkan][vuid][layouts-mock]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetPhysicalDeviceFormatProperties;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &mockGetPhysicalDeviceImageFormatProperties;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x13);
    const auto device = fakeHandle<VkDevice>(0x23);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeHandle<VkQueue>(0x33),
        .compute = fakeHandle<VkQueue>(0x33),
        .copy = fakeHandle<VkQueue>(0x33),
        .families = families,
    };

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = physical;
    record.separateDepthStencilLayouts = true;
    record.independentResolveNone = false;
    record.independentResolve = false;
    record.depthResolveModes = VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT;
    record.stencilResolveModes = VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT;

    rachis::DeviceCapabilities caps{};
    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, &instanceTable, physical, families);
    rachis::vulkan::VulkanImageUtilities imageUtils;
    imageUtils.initialize(&record, &caps, &resources, nullptr, nullptr, nullptr);

    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, 0xABC3, &resources);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 2});
    REQUIRE(cmdPool.has_value());

    const auto msaaDs = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x6020),
                                                      {
                                                          .width = 8,
                                                          .height = 8,
                                                          .sampleCount = 4,
                                                          .format = rachis::Format::d24UnormS8Uint,
                                                          .usage = rachis::TextureUsage::depthStencil,
                                                      });
    const auto singleDs = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x6021),
                                                        {
                                                            .width = 8,
                                                            .height = 8,
                                                            .format = rachis::Format::d24UnormS8Uint,
                                                            .usage = rachis::TextureUsage::depthStencil,
                                                        });
    REQUIRE(msaaDs.has_value());
    REQUIRE(singleDs.has_value());

    {
        auto cb = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cb.has_value());
        auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);
        for (const auto texture : {*msaaDs, *singleDs})
        {
            for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
            {
                REQUIRE(resources
                            .textureBarrier(vkCb.native(),
                                            rachis::QueueType::graphics,
                                            {
                                                .texture = texture,
                                                .stagesAfter = rachis::PipelineStage::all,
                                                .accessAfter = rachis::Access::common,
                                                .layoutAfter = rachis::TextureLayout::common,
                                                .subresources = {.aspect = aspect},
                                            })
                            .has_value());
            }
        }

        const auto resolve = [&](rachis::TextureAspect aspect) {
            return imageUtils.resolveTexture(vkCb,
                                             {
                                                 .source = *msaaDs,
                                                 .sourceSubresource = {.aspect = aspect},
                                                 .destination = *singleDs,
                                                 .destinationSubresource = {.aspect = aspect},
                                                 .mode = rachis::ResolveMode::maximum,
                                             });
        };
        for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
        {
            const bool depth = aspect == rachis::TextureAspect::depth;
            mock.imageBarriers.clear();
            mock.beginRenderings.clear();
            const auto resolved = resolve(aspect);
            if (!resolved)
            {
                INFO(resolved.error().message);
            }
            REQUIRE(resolved.has_value());
            REQUIRE(mock.beginRenderings.size() == 1);
            CHECK(mock.beginRenderings[0].hasDepth == depth);
            CHECK(mock.beginRenderings[0].hasStencil == !depth);
            REQUIRE(mock.imageBarriers.size() == 4);
            for (const auto& barrier : mock.imageBarriers)
            {
                CHECK(barrier.aspectMask ==
                      static_cast<VkImageAspectFlags>(depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT));
            }
        }

        record.separateDepthStencilLayouts = false;
        mock.beginRenderings.clear();
        const auto bothAttached = resolve(rachis::TextureAspect::depth);
        REQUIRE_FALSE(bothAttached.has_value());
        CHECK(bothAttached.error().code == rachis::ErrorCode::unsupported);
        CHECK(mock.beginRenderings.empty());

        record.independentResolveNone = true;
        REQUIRE(resolve(rachis::TextureAspect::depth).has_value());
        REQUIRE(mock.beginRenderings.size() == 1);
        CHECK(mock.beginRenderings[0].hasDepth);
        CHECK(mock.beginRenderings[0].hasStencil);
        CHECK(mock.beginRenderings[0].depthLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    commands.shutdown();
    imageUtils.shutdown();
    resources.shutdown();
    gMock = nullptr;
}

namespace rachis::vulkan
{

    struct VulkanAccelerationTestAccess final
    {
        [[nodiscard]] static Status recordBuildBarrier(const VulkanAccelerationManager& acceleration,
                                                       VkCommandBuffer command,
                                                       QueueType queueType)
        {
            return acceleration.recordBuildBarrier(command, queueType);
        }

        static void recordInPlaceUpdateBarrier(const VulkanAccelerationManager& acceleration,
                                               VkCommandBuffer command,
                                               QueueType queueType) noexcept
        {
            acceleration.recordInPlaceUpdateBarrier(command, queueType);
        }
    };

} // namespace rachis::vulkan

namespace
{

    constexpr VkShaderStageFlags coreNativeShaderStages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    constexpr VkShaderStageFlags allGraphicsNativeShaderStages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_GEOMETRY_BIT |
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
        VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    constexpr VkShaderStageFlags everyNativeShaderStage =
        allGraphicsNativeShaderStages | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR |
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
        VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR;

    [[nodiscard]] rachis::DeviceCapabilities everyOptionalShaderStage() noexcept
    {
        rachis::DeviceCapabilities capabilities{};
        capabilities.geometryShaders = true;
        capabilities.tessellationShaders = true;
        capabilities.meshShaders = true;
        capabilities.taskShaders = true;
        capabilities.rayTracing = true;
        capabilities.rayQuery = true;
        return capabilities;
    }

    constexpr auto everyBarrierShaderStage =
        rachis::PipelineStage::vertexShader | rachis::PipelineStage::pixelShader |
        rachis::PipelineStage::computeShader | rachis::PipelineStage::geometryShader |
        rachis::PipelineStage::hullShader | rachis::PipelineStage::domainShader | rachis::PipelineStage::taskShader |
        rachis::PipelineStage::meshShader | rachis::PipelineStage::rayTracingShader;

    [[nodiscard]] rachis::TextureBarrier mixedStageTextureBarrier(rachis::TextureHandle texture) noexcept
    {
        return rachis::TextureBarrier{
            .texture = texture,
            .stagesBefore = rachis::PipelineStage::computeShader | rachis::PipelineStage::meshShader,
            .stagesAfter = everyBarrierShaderStage,
            .accessBefore = rachis::Access::unorderedAccess,
            .accessAfter = rachis::Access::shaderResource,
            .layoutBefore = rachis::TextureLayout::unorderedAccess,
            .layoutAfter = rachis::TextureLayout::shaderResource,
        };
    }

} // namespace

TEST_CASE("Vulkan binding and pipeline layouts emit only the shader stages the device enabled",
          "[size:small][rhi][vulkan][vuid][bindings-mock]")
{
    using rachis::ShaderStage;
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();
    const auto device = fakeHandle<VkDevice>(0x23);

    const rachis::DeviceCapabilities baseline{};
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, baseline);

    const std::array compositeEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, ShaderStage::allGraphics},
        rachis::BindingLayoutEntry{
            2, rachis::BindingType::sampledTexture, ShaderStage::vertex | ShaderStage::geometry | ShaderStage::mesh},
        rachis::BindingLayoutEntry{3, rachis::BindingType::sampler, ShaderStage::compute | ShaderStage::allRayTracing},
    };
    auto layout = bindings.createBindingLayout({.entries = compositeEntries});
    REQUIRE(layout.has_value());
    REQUIRE(mock.descriptorBindingStages.size() == 4);
    CHECK(mock.descriptorBindingStages[0] == coreNativeShaderStages);
    CHECK(mock.descriptorBindingStages[1] == (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
    CHECK(mock.descriptorBindingStages[2] == VK_SHADER_STAGE_VERTEX_BIT);
    CHECK(mock.descriptorBindingStages[3] == VK_SHADER_STAGE_COMPUTE_BIT);

    const std::array compositeRanges{
        rachis::PushConstantRange{ShaderStage::all, 0, 16},
        rachis::PushConstantRange{ShaderStage::allGraphics, 16, 16},
        rachis::PushConstantRange{ShaderStage::compute | ShaderStage::allRayTracing, 32, 16},
    };
    auto pipelineLayout =
        bindings.createPipelineLayout({.bindingLayouts = std::array{*layout}, .pushConstantRanges = compositeRanges});
    REQUIRE(pipelineLayout.has_value());
    REQUIRE(mock.pushConstantRangeStages.size() == 2);
    CHECK(mock.pushConstantRangeStages[0] == (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
    CHECK(mock.pushConstantRangeStages[1] == VK_SHADER_STAGE_COMPUTE_BIT);

    for (const auto stages : {ShaderStage::geometry,
                              ShaderStage::hull,
                              ShaderStage::domain,
                              ShaderStage::task,
                              ShaderStage::mesh,
                              ShaderStage::geometry | ShaderStage::mesh,
                              ShaderStage::rayGeneration,
                              ShaderStage::allRayTracing})
    {
        const std::array entry{rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, stages}};
        const auto rejectedLayout = bindings.createBindingLayout({.entries = entry});
        REQUIRE_FALSE(rejectedLayout.has_value());
        CHECK(rejectedLayout.error().code == rachis::ErrorCode::unsupported);
        const std::array range{rachis::PushConstantRange{stages, 0, 16}};
        const auto rejectedPipelineLayout = bindings.createPipelineLayout({.pushConstantRanges = range});
        REQUIRE_FALSE(rejectedPipelineLayout.has_value());
        CHECK(rejectedPipelineLayout.error().code == rachis::ErrorCode::unsupported);
    }
    CHECK(mock.descriptorBindingStages.size() == 4);
    CHECK(mock.pushConstantRangeStages.size() == 2);

    const std::array disabledStageRanges{
        rachis::PushConstantRange{ShaderStage::vertex | ShaderStage::geometry, 0, 16},
        rachis::PushConstantRange{ShaderStage::pixel | ShaderStage::geometry, 32, 16},
    };
    mock.pushConstantRangeStages.clear();
    auto disabledStageLayout = bindings.createPipelineLayout({.pushConstantRanges = disabledStageRanges});
    REQUIRE(disabledStageLayout.has_value());
    REQUIRE(mock.pushConstantRangeStages.size() == 2);
    CHECK(mock.pushConstantRangeStages[0] == VK_SHADER_STAGE_VERTEX_BIT);
    CHECK(mock.pushConstantRangeStages[1] == VK_SHADER_STAGE_FRAGMENT_BIT);
    REQUIRE(bindings.destroyPipelineLayout(*disabledStageLayout).has_value());

    const std::array noStage{rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, ShaderStage::none}};
    const auto invalidLayout = bindings.createBindingLayout({.entries = noStage});
    REQUIRE_FALSE(invalidLayout.has_value());
    CHECK(invalidLayout.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(bindings.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(bindings.destroyBindingLayout(*layout).has_value());
    bindings.shutdown();

    mock.descriptorBindingStages.clear();
    mock.pushConstantRangeStages.clear();
    const auto full = everyOptionalShaderStage();
    rachis::vulkan::VulkanBindingManager fullBindings;
    fullBindings.initialize(device, &table, full);
    const std::array fullEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, ShaderStage::allGraphics},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampledTexture, ShaderStage::geometry},
    };
    auto fullLayout = fullBindings.createBindingLayout({.entries = fullEntries});
    REQUIRE(fullLayout.has_value());
    REQUIRE(mock.descriptorBindingStages.size() == 3);
    CHECK(mock.descriptorBindingStages[0] == everyNativeShaderStage);
    CHECK(mock.descriptorBindingStages[1] == allGraphicsNativeShaderStages);
    CHECK(mock.descriptorBindingStages[2] == VK_SHADER_STAGE_GEOMETRY_BIT);
    const std::array meshRange{rachis::PushConstantRange{ShaderStage::mesh, 0, 16}};
    auto fullPipelineLayout = fullBindings.createPipelineLayout({.pushConstantRanges = meshRange});
    REQUIRE(fullPipelineLayout.has_value());
    REQUIRE(mock.pushConstantRangeStages.size() == 1);
    CHECK(mock.pushConstantRangeStages[0] == VK_SHADER_STAGE_MESH_BIT_EXT);
    REQUIRE(fullBindings.destroyPipelineLayout(*fullPipelineLayout).has_value());
    REQUIRE(fullBindings.destroyBindingLayout(*fullLayout).has_value());
    fullBindings.shutdown();
    gMock = nullptr;
}

TEST_CASE("Vulkan push constants name the enabled stages of composite layout ranges",
          "[size:small][rhi][vulkan][vuid][graphics-mock]")
{
    using rachis::ShaderStage;
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();
    const auto physical = fakeHandle<VkPhysicalDevice>(0x13);
    const auto device = fakeHandle<VkDevice>(0x24);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeHandle<VkQueue>(0x34),
        .compute = fakeHandle<VkQueue>(0x34),
        .copy = fakeHandle<VkQueue>(0x34),
        .families = families,
    };

    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, nullptr, physical, families);
    const rachis::DeviceCapabilities baseline{};
    const auto full = everyOptionalShaderStage();
    rachis::vulkan::VulkanGraphicsEncoder baselineGraphics;
    baselineGraphics.initialize(device, &table, &baseline, &resources, nullptr, nullptr);
    rachis::vulkan::VulkanGraphicsEncoder fullGraphics;
    fullGraphics.initialize(device, &table, &full, &resources, nullptr, nullptr);

    constexpr std::uint64_t kDeviceToken = 0xABC3;
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, kDeviceToken, &resources);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 2});
    REQUIRE(cmdPool.has_value());
    auto cb = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(cb.has_value());
    auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);

    auto pipeline = std::make_shared<rachis::vulkan::VulkanNativePipeline>();
    pipeline->pipeline = fakeHandle<VkPipeline>(0x5021);
    pipeline->layout = fakeHandle<VkPipelineLayout>(0x5022);
    pipeline->pushConstantRanges = {
        {.visibility = ShaderStage::all, .offset = 0, .size = 16},
        {.visibility = ShaderStage::allGraphics, .offset = 16, .size = 16},
    };
    pipeline->nativePushConstantRanges =
        rachis::vulkan::validation_detail::mergePushConstantRangesPerStage(pipeline->pushConstantRanges);
    vkCb.computeState().pipelineOwner = pipeline;
    vkCb.graphicsState().pipelineOwner = pipeline;
    vkCb.graphicsState().renderingOwner = std::make_shared<int>(1);

    std::array<std::byte, 16> pushData{};
    REQUIRE(baselineGraphics.setComputePushConstants(vkCb, 0, pushData).has_value());
    REQUIRE(baselineGraphics.setGraphicsPushConstants(vkCb, ShaderStage::allGraphics, 16, pushData).has_value());
    REQUIRE(fullGraphics.setComputePushConstants(vkCb, 0, pushData).has_value());
    REQUIRE(fullGraphics.setGraphicsPushConstants(vkCb, ShaderStage::mesh, 16, pushData).has_value());
    REQUIRE(mock.pushConstants.size() == 4);
    CHECK(mock.pushConstants[0].stageFlags == coreNativeShaderStages);
    CHECK(mock.pushConstants[1].stageFlags == (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
    CHECK(mock.pushConstants[2].stageFlags == everyNativeShaderStage);
    CHECK(mock.pushConstants[3].stageFlags == allGraphicsNativeShaderStages);

    vkCb.graphicsState().renderingOwner.reset();
    vkCb.graphicsState().pipelineOwner.reset();
    vkCb.computeState().pipelineOwner.reset();
    cb->reset();
    commands.shutdown();
    resources.shutdown();
    gMock = nullptr;
}

TEST_CASE("Vulkan barriers drop pipeline stages the device has not enabled and reject scopes left without one",
          "[size:small][rhi][vulkan][vuid][barriers-mock]")
{
    using rachis::Access;
    using rachis::PipelineStage;
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetPhysicalDeviceFormatProperties;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &mockGetPhysicalDeviceImageFormatProperties;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x14);
    const auto device = fakeHandle<VkDevice>(0x25);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const auto command = fakeHandle<VkCommandBuffer>(0x7001);
    const rachis::TextureDesc textureDesc{
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
    };

    constexpr VkPipelineStageFlags2 computeNativeStages =
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
    constexpr VkPipelineStageFlags2 coreNativeStages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    constexpr VkPipelineStageFlags2 everyNativeStage =
        coreNativeStages | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
        VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

    const rachis::GlobalBarrier mixedGlobal{
        .stagesBefore =
            PipelineStage::computeShader | PipelineStage::rayTracingShader | PipelineStage::accelerationStructureCopy,
        .stagesAfter = everyBarrierShaderStage,
        .accessBefore = Access::unorderedAccess,
        .accessAfter = Access::shaderResource,
    };
    const std::array alwaysUnenabled{
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::computeShader,
            .stagesAfter = PipelineStage::conditionalExecution,
            .accessBefore = Access::unorderedAccess,
            .accessAfter = Access::predicate,
        },
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::streamOutput,
            .stagesAfter = PipelineStage::vertexShader,
            .accessBefore = Access::streamOutputWrite,
            .accessAfter = Access::shaderResource,
        },
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::accelerationStructureCopy,
            .stagesAfter = PipelineStage::computeShader,
            .accessBefore = Access::unorderedAccess,
            .accessAfter = Access::shaderResource,
        },
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::renderTarget,
            .stagesAfter = PipelineStage::pixelShader,
            .accessBefore = Access::renderTarget,
            .accessAfter = Access::shadingRateSource,
        },
    };
    const std::array optionalOnly{
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::computeShader,
            .stagesAfter = PipelineStage::meshShader,
            .accessBefore = Access::unorderedAccess,
            .accessAfter = Access::shaderResource,
        },
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::rayTracingShader,
            .stagesAfter = PipelineStage::computeShader,
        },
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::copy,
            .stagesAfter = PipelineStage::geometryShader | PipelineStage::copy,
            .accessBefore = Access::copyDestination,
            .accessAfter = Access::shaderResource | Access::copySource,
        },
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::accelerationStructureBuild,
            .stagesAfter = PipelineStage::computeShader,
            .accessBefore = Access::accelerationStructureScratch,
            .accessAfter = Access::shaderResource,
        },
    };

    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, &instanceTable, physical, families);
    const auto texture = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x7101), textureDesc);
    REQUIRE(texture.has_value());
    REQUIRE(resources
                .resourceBarriers(command,
                                  rachis::QueueType::graphics,
                                  {.globalBarriers = std::array{mixedGlobal},
                                   .textureBarriers = std::array{mixedStageTextureBarrier(*texture)}},
                                  nullptr)
                .has_value());
    REQUIRE(mock.memoryBarriers.size() == 1);
    CHECK(mock.memoryBarriers[0].srcStageMask == computeNativeStages);
    CHECK(
        mock.memoryBarriers[0].srcAccessMask ==
        (VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT));
    CHECK(mock.memoryBarriers[0].dstStageMask == coreNativeStages);
    CHECK(mock.memoryBarriers[0].dstAccessMask == VK_ACCESS_2_SHADER_READ_BIT);
    REQUIRE(mock.imageBarriers.size() == 1);
    CHECK(mock.imageBarriers[0].srcStageMask == computeNativeStages);
    CHECK(mock.imageBarriers[0].dstStageMask == coreNativeStages);

    for (const auto& barrier : optionalOnly)
    {
        const auto rejected =
            resources.resourceBarriers(command, rachis::QueueType::graphics, {.globalBarriers = std::array{barrier}});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
    }
    for (const auto& barrier : alwaysUnenabled)
    {
        const auto rejected =
            resources.resourceBarriers(command, rachis::QueueType::graphics, {.globalBarriers = std::array{barrier}});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
    }
    auto meshOnlyTexture = mixedStageTextureBarrier(*texture);
    meshOnlyTexture.stagesAfter = PipelineStage::taskShader | PipelineStage::meshShader;
    const auto rejectedTexture = resources.textureBarrier(command, rachis::QueueType::graphics, meshOnlyTexture);
    REQUIRE_FALSE(rejectedTexture.has_value());
    CHECK(rejectedTexture.error().code == rachis::ErrorCode::unsupported);
    CHECK(mock.memoryBarriers.size() == 1);
    CHECK(mock.imageBarriers.size() == 1);
    resources.discardCommandLayouts(command);
    resources.shutdown();

    mock.memoryBarriers.clear();
    mock.imageBarriers.clear();
    rachis::vulkan::VulkanResourceAllocator fullResources;
    fullResources.initializeDispatchForTesting(
        device, &table, &instanceTable, physical, families, rachis::ShaderStage::all, true);
    const auto fullTexture = fullResources.adoptSwapchainImage(fakeHandle<VkImage>(0x7102), textureDesc);
    REQUIRE(fullTexture.has_value());
    REQUIRE(fullResources
                .resourceBarriers(command,
                                  rachis::QueueType::graphics,
                                  {.globalBarriers = std::array{mixedGlobal},
                                   .textureBarriers = std::array{mixedStageTextureBarrier(*fullTexture)}},
                                  nullptr)
                .has_value());
    REQUIRE(mock.memoryBarriers.size() == 1);
    CHECK(mock.memoryBarriers[0].srcStageMask ==
          (computeNativeStages | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR));
    CHECK(mock.memoryBarriers[0].dstStageMask == everyNativeStage);
    REQUIRE(mock.imageBarriers.size() == 1);
    CHECK(mock.imageBarriers[0].srcStageMask == (computeNativeStages | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT));
    CHECK(mock.imageBarriers[0].dstStageMask == everyNativeStage);
    REQUIRE(
        fullResources.resourceBarriers(command, rachis::QueueType::graphics, {.globalBarriers = optionalOnly}, nullptr)
            .has_value());
    REQUIRE(mock.memoryBarriers.size() == 1 + optionalOnly.size());
    CHECK(mock.memoryBarriers[1].dstStageMask == VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT);
    CHECK(mock.memoryBarriers[2].srcStageMask == VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);
    CHECK(mock.memoryBarriers[4].srcStageMask == VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
    for (const auto& barrier : alwaysUnenabled)
    {
        const auto rejected = fullResources.resourceBarriers(
            command, rachis::QueueType::graphics, {.globalBarriers = std::array{barrier}});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
    }
    fullResources.discardCommandLayouts(command);
    fullResources.shutdown();
    gMock = nullptr;
}

TEST_CASE("Vulkan barriers map vertex-buffer and acceleration-structure accesses to their native scopes",
          "[size:small][rhi][vulkan][vuid][barriers-mock]")
{
    using rachis::Access;
    using rachis::PipelineStage;
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();
    const auto physical = fakeHandle<VkPhysicalDevice>(0x15);
    const auto device = fakeHandle<VkDevice>(0x27);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const auto command = fakeHandle<VkCommandBuffer>(0x7301);
    constexpr VkPipelineStageFlags2 vertexInputStages = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                                                        VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
                                                        VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
    constexpr VkAccessFlags2 scratchAccess =
        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    const std::array acceptedBefore{
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::copy,
            .stagesAfter = PipelineStage::vertexShader,
            .accessBefore = Access::copyDestination,
            .accessAfter = Access::vertexBuffer | Access::constantBuffer,
        },
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::copy,
            .stagesAfter = PipelineStage::accelerationStructureBuild,
            .accessBefore = Access::copyDestination,
            .accessAfter = Access::accelerationStructureBuildInput,
        },
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::accelerationStructureBuild,
            .stagesAfter = PipelineStage::accelerationStructureBuild,
            .accessBefore = Access::accelerationStructureScratch,
            .accessAfter = Access::accelerationStructureScratch,
        },
        rachis::GlobalBarrier{
            .stagesBefore = PipelineStage::copy,
            .stagesAfter = PipelineStage::rayTracingShader,
            .accessBefore = Access::copyDestination,
            .accessAfter = Access::shaderResource,
        },
    };
    const rachis::GlobalBarrier vertexInput{
        .stagesBefore = PipelineStage::copy,
        .stagesAfter = PipelineStage::vertexInput,
        .accessBefore = Access::copyDestination,
        .accessAfter = Access::vertexBuffer,
    };

    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, nullptr, physical, families, rachis::ShaderStage::all, true);
    REQUIRE(resources.resourceBarriers(command, rachis::QueueType::graphics, {.globalBarriers = acceptedBefore})
                .has_value());
    REQUIRE(mock.memoryBarriers.size() == acceptedBefore.size());
    CHECK(mock.memoryBarriers[0].dstStageMask ==
          (VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT));
    CHECK(mock.memoryBarriers[0].dstAccessMask ==
          (VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT));
    CHECK(mock.memoryBarriers[1].dstStageMask == VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
    CHECK(mock.memoryBarriers[1].dstAccessMask == VK_ACCESS_2_SHADER_READ_BIT);
    CHECK(mock.memoryBarriers[2].srcAccessMask == scratchAccess);
    CHECK(mock.memoryBarriers[2].dstAccessMask == scratchAccess);
    CHECK(mock.memoryBarriers[3].dstStageMask == VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);
    CHECK(mock.memoryBarriers[3].dstAccessMask == VK_ACCESS_2_SHADER_READ_BIT);

    REQUIRE(
        resources.resourceBarriers(command, rachis::QueueType::graphics, {.globalBarriers = std::array{vertexInput}})
            .has_value());
    REQUIRE(mock.memoryBarriers.size() == acceptedBefore.size() + 1);
    CHECK(mock.memoryBarriers[4].dstStageMask == vertexInputStages);
    CHECK(mock.memoryBarriers[4].dstAccessMask == VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);

    for (const auto stages : {PipelineStage::pixelShader, PipelineStage::computeShader})
    {
        auto mismatched = vertexInput;
        mismatched.stagesAfter = stages;
        const auto rejected = resources.resourceBarriers(
            command, rachis::QueueType::graphics, {.globalBarriers = std::array{mismatched}});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    }
    CHECK(mock.memoryBarriers.size() == acceptedBefore.size() + 1);
    resources.shutdown();
    gMock = nullptr;
}

TEST_CASE("Vulkan acceleration-structure barriers cover every enabled shader stage that can trace the structure",
          "[size:small][rhi][vulkan][vuid][raytracing-mock]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();
    const auto device = fakeHandle<VkDevice>(0x26);
    const auto command = fakeHandle<VkCommandBuffer>(0x7201);
    constexpr VkPipelineStageFlags2 computeReaders =
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    constexpr VkPipelineStageFlags2 coreGraphicsReaders =
        computeReaders | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    constexpr VkPipelineStageFlags2 everyGraphicsReader =
        coreGraphicsReaders | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
        VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    constexpr VkAccessFlags2 structureAccess =
        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    rachis::DeviceCapabilities rayQueryOnly{};
    rayQueryOnly.rayQuery = true;
    rachis::vulkan::VulkanAccelerationManager rayQueryAcceleration;
    rayQueryAcceleration.initialize(device, &table, nullptr, &rayQueryOnly, 256);
    REQUIRE(rachis::vulkan::VulkanAccelerationTestAccess::recordBuildBarrier(
                rayQueryAcceleration, command, rachis::QueueType::graphics)
                .has_value());

    const auto rayTracing = everyOptionalShaderStage();
    rachis::vulkan::VulkanAccelerationManager rayTracingAcceleration;
    rayTracingAcceleration.initialize(device, &table, nullptr, &rayTracing, 256);
    REQUIRE(rachis::vulkan::VulkanAccelerationTestAccess::recordBuildBarrier(
                rayTracingAcceleration, command, rachis::QueueType::graphics)
                .has_value());
    REQUIRE(rachis::vulkan::VulkanAccelerationTestAccess::recordBuildBarrier(
                rayTracingAcceleration, command, rachis::QueueType::compute)
                .has_value());
    rachis::vulkan::VulkanAccelerationTestAccess::recordInPlaceUpdateBarrier(
        rayQueryAcceleration, command, rachis::QueueType::graphics);
    rachis::vulkan::VulkanAccelerationTestAccess::recordInPlaceUpdateBarrier(
        rayTracingAcceleration, command, rachis::QueueType::graphics);
    rachis::vulkan::VulkanAccelerationTestAccess::recordInPlaceUpdateBarrier(
        rayTracingAcceleration, command, rachis::QueueType::compute);

    REQUIRE(mock.memoryBarriers.size() == 6);
    for (std::size_t index = 0; index < 3; ++index)
    {
        CAPTURE(index);
        CHECK(mock.memoryBarriers[index].srcStageMask == VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
        CHECK(mock.memoryBarriers[index].srcAccessMask == VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
        CHECK(mock.memoryBarriers[index].dstAccessMask == structureAccess);
    }
    CHECK(mock.memoryBarriers[0].dstStageMask == coreGraphicsReaders);
    CHECK(mock.memoryBarriers[1].dstStageMask == everyGraphicsReader);
    CHECK(mock.memoryBarriers[2].dstStageMask == (computeReaders | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR));
    for (std::size_t index = 3; index < 6; ++index)
    {
        CAPTURE(index);
        CHECK(mock.memoryBarriers[index].srcAccessMask == VK_ACCESS_2_NONE);
        CHECK(mock.memoryBarriers[index].dstStageMask == VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
        CHECK(mock.memoryBarriers[index].dstAccessMask == structureAccess);
    }
    CHECK(mock.memoryBarriers[3].srcStageMask == coreGraphicsReaders);
    CHECK(mock.memoryBarriers[4].srcStageMask == everyGraphicsReader);
    CHECK(mock.memoryBarriers[5].srcStageMask == (computeReaders | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR));
    rayTracingAcceleration.shutdown();
    rayQueryAcceleration.shutdown();
    gMock = nullptr;
}

TEST_CASE("Vulkan resource allocators read graphics support from the queue family behind each queue type",
          "[size:small][rhi][vulkan][vuid][transfer-mock]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceQueueFamilyProperties = &mockGetPhysicalDeviceQueueFamilyProperties;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x15);
    const auto device = fakeHandle<VkDevice>(0x27);

    rachis::vulkan::VulkanResourceAllocator dedicatedTransfer;
    dedicatedTransfer.initializeDispatchForTesting(
        device, &table, &instanceTable, physical, {.graphics = 0, .compute = 0, .transfer = 1});
    CHECK(dedicatedTransfer.queueFamilySupportsGraphics(rachis::QueueType::graphics));
    CHECK(dedicatedTransfer.queueFamilySupportsGraphics(rachis::QueueType::compute));
    CHECK_FALSE(dedicatedTransfer.queueFamilySupportsGraphics(rachis::QueueType::copy));

    rachis::vulkan::VulkanResourceAllocator dedicatedCompute;
    dedicatedCompute.initializeDispatchForTesting(
        device, &table, &instanceTable, physical, {.graphics = 0, .compute = 1, .transfer = 0});
    CHECK(dedicatedCompute.queueFamilySupportsGraphics(rachis::QueueType::graphics));
    CHECK_FALSE(dedicatedCompute.queueFamilySupportsGraphics(rachis::QueueType::compute));
    CHECK(dedicatedCompute.queueFamilySupportsGraphics(rachis::QueueType::copy));

    rachis::vulkan::VulkanResourceAllocator singleFamily;
    singleFamily.initializeDispatchForTesting(
        device, &table, &instanceTable, physical, {.graphics = 0, .compute = 0, .transfer = 0});
    CHECK(singleFamily.queueFamilySupportsGraphics(rachis::QueueType::graphics));
    CHECK(singleFamily.queueFamilySupportsGraphics(rachis::QueueType::compute));
    CHECK(singleFamily.queueFamilySupportsGraphics(rachis::QueueType::copy));

    rachis::vulkan::VulkanResourceAllocator unqueriedFamilies;
    unqueriedFamilies.initializeDispatchForTesting(
        device, &table, nullptr, physical, {.graphics = 0, .compute = 0, .transfer = 1});
    CHECK(unqueriedFamilies.queueFamilySupportsGraphics(rachis::QueueType::graphics));
    CHECK_FALSE(unqueriedFamilies.queueFamilySupportsGraphics(rachis::QueueType::copy));

    unqueriedFamilies.shutdown();
    singleFamily.shutdown();
    dedicatedCompute.shutdown();
    dedicatedTransfer.shutdown();
    gMock = nullptr;
}

TEST_CASE("Vulkan swapchain image leases count recorded references until submit, reset, or abandonment",
          "[size:small][rhi][vulkan][vuid][swapchain-mock]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetPhysicalDeviceFormatProperties;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &mockGetPhysicalDeviceImageFormatProperties;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x13);
    const auto device = fakeHandle<VkDevice>(0x23);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeHandle<VkQueue>(0x33),
        .compute = fakeHandle<VkQueue>(0x33),
        .copy = fakeHandle<VkQueue>(0x33),
        .families = families,
    };

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = physical;
    rachis::DeviceCapabilities caps{};
    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, &instanceTable, physical, families);
    rachis::vulkan::VulkanImageUtilities imageUtils;
    imageUtils.initialize(&record, &caps, &resources, nullptr, nullptr, nullptr);
    rachis::vulkan::VulkanGraphicsEncoder graphics;
    graphics.initialize(device, &table, &caps, &resources, nullptr, nullptr);

    constexpr std::uint64_t kDeviceToken = 0xABC3;
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, kDeviceToken, &resources);
    rachis::vulkan::VulkanQueueManager queueManager;
    REQUIRE(queueManager.initialize(device, &table, queueHandles, kDeviceToken).has_value());
    const auto cmdPool = commands.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 8,
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(cmdPool.has_value());

    const auto lease = std::make_shared<rachis::vulkan::VulkanSwapchainImageLease>();
    const auto image = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x7001),
                                                     {
                                                         .width = 8,
                                                         .height = 8,
                                                         .format = rachis::Format::rgba8Unorm,
                                                         .usage = rachis::TextureUsage::renderTarget |
                                                                  rachis::TextureUsage::copySource |
                                                                  rachis::TextureUsage::copyDestination,
                                                     },
                                                     lease);
    const auto multisampled = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x7002),
                                                            {
                                                                .width = 8,
                                                                .height = 8,
                                                                .sampleCount = 4,
                                                                .format = rachis::Format::rgba8Unorm,
                                                                .usage = rachis::TextureUsage::renderTarget,
                                                            });
    const auto copySource = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x7003),
                                                          {
                                                              .width = 8,
                                                              .height = 8,
                                                              .format = rachis::Format::rgba8Unorm,
                                                              .usage = rachis::TextureUsage::copySource,
                                                          });
    REQUIRE(image.has_value());
    REQUIRE(multisampled.has_value());
    REQUIRE(copySource.has_value());
    const auto view = resources.createTextureView({
        .texture = *image,
        .usage = rachis::TextureViewUsage::renderTarget,
        .format = rachis::Format::rgba8Unorm,
    });
    REQUIRE(view.has_value());
    const auto toCommon = [](rachis::TextureHandle texture) {
        return rachis::TextureBarrier{
            .texture = texture,
            .stagesAfter = rachis::PipelineStage::all,
            .accessAfter = rachis::Access::common,
            .layoutAfter = rachis::TextureLayout::common,
        };
    };
    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *view,
            .loadOperation = rachis::LoadOperation::clear,
        },
    };
    const std::array copyRegions{
        rachis::TextureCopyRegion{
            .extent = {.width = 8, .height = 8, .depth = 1},
        },
    };
    const auto requireInvalidState = [](const rachis::Status& status) {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == rachis::ErrorCode::invalidState);
    };

    {
        auto unacquired = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(unacquired.has_value());
        auto& native = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**unacquired);
        mock.imageBarriers.clear();
        mock.beginRenderings.clear();
        requireInvalidState(resources.textureBarrier(native.native(), rachis::QueueType::graphics, toCommon(*image)));
        requireInvalidState(graphics.beginRendering(native, {.colorAttachments = colorAttachments}));
        CHECK(mock.imageBarriers.empty());
        CHECK(mock.beginRenderings.empty());
        requireInvalidState(lease->cancel());
    }

    lease->beginAcquisition();
    {
        auto setup = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(setup.has_value());
        auto& native = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**setup);
        REQUIRE(resources.textureBarrier(native.native(), rachis::QueueType::graphics, toCommon(*image)).has_value());
        REQUIRE(resources.textureBarrier(native.native(), rachis::QueueType::graphics, toCommon(*multisampled))
                    .has_value());
        REQUIRE(
            resources.textureBarrier(native.native(), rachis::QueueType::graphics, toCommon(*copySource)).has_value());
        requireInvalidState(lease->cancel());
        requireInvalidState(lease->beginPresentation());
        std::array batch{std::move(*setup)};
        REQUIRE(queueManager.submit(batch, {}).has_value());
    }
    requireInvalidState(lease->cancel());
    REQUIRE(lease->beginPresentation().has_value());
    {
        auto presented = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(presented.has_value());
        auto& native = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**presented);
        const auto copiesBefore = std::ranges::count_if(
            mock.timelineEvents, [](const CommandOpEvent& event) { return event.kind == CommandOpKind::copyImage; });
        requireInvalidState(resources.copyTextureRegions(native.native(), *copySource, *image, copyRegions));
        CHECK(std::ranges::count_if(mock.timelineEvents, [](const CommandOpEvent& event) {
                  return event.kind == CommandOpKind::copyImage;
              }) == copiesBefore);
    }

    lease->beginAcquisition();
    REQUIRE(lease->cancel().has_value());
    lease->beginAcquisition();
    {
        auto copied = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(copied.has_value());
        auto& native = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**copied);
        REQUIRE(resources.copyTextureRegions(native.native(), *copySource, *image, copyRegions).has_value());
        requireInvalidState(lease->cancel());
    }
    REQUIRE(lease->cancel().has_value());

    lease->beginAcquisition();
    {
        auto resolved = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(resolved.has_value());
        auto& native = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**resolved);
        REQUIRE(imageUtils.resolveTexture(native, {.source = *multisampled, .destination = *image}).has_value());
        requireInvalidState(lease->cancel());
        REQUIRE(commands.resetCommandBuffer(**resolved).has_value());
        REQUIRE(lease->cancel().has_value());
    }

    lease->beginAcquisition();
    {
        auto rendered = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(rendered.has_value());
        auto& native = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**rendered);
        REQUIRE(graphics.beginRendering(native, {.colorAttachments = colorAttachments}).has_value());
        REQUIRE(graphics.endRendering(native).has_value());
        requireInvalidState(lease->cancel());
        rachis::vulkan::test::nextEndCommandBufferResult().store(static_cast<std::int32_t>(VK_ERROR_OUT_OF_HOST_MEMORY),
                                                                 std::memory_order_relaxed);
        std::array batch{std::move(*rendered)};
        const auto failed = queueManager.submit(batch, {});
        rachis::vulkan::test::nextEndCommandBufferResult().store(0, std::memory_order_relaxed);
        REQUIRE_FALSE(failed.has_value());
        CHECK(batch[0] == nullptr);
    }
    REQUIRE(lease->cancel().has_value());

    lease->beginAcquisition();
    {
        auto rendered = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(rendered.has_value());
        auto& native = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**rendered);
        REQUIRE(graphics.beginRendering(native, {.colorAttachments = colorAttachments}).has_value());
        REQUIRE(graphics.endRendering(native).has_value());
        std::array batch{std::move(*rendered)};
        REQUIRE(queueManager.submit(batch, {}).has_value());
    }
    requireInvalidState(lease->cancel());
    REQUIRE(lease->beginPresentation().has_value());
    lease->abortPresentation();
    requireInvalidState(lease->cancel());

    queueManager.collectGarbage();
    REQUIRE(resources.destroyTextureView(*view).has_value());
    queueManager.shutdown();
    commands.shutdown();
    imageUtils.shutdown();
    resources.shutdown();
    gMock = nullptr;
}

namespace
{

    struct MockDescriptorPool final
    {
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::uint32_t maxSets = 0;
        std::vector<VkDescriptorPoolSize> sizes;
        std::uint32_t liveSets = 0;
        VkDescriptorPoolCreateFlags flags = 0;
    };

    struct RecordedImageDescriptorWrite final
    {
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_SAMPLER;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct MockDescriptorDispatch final
    {
        std::uintptr_t nextHandle = 0x9000;
        std::vector<MockDescriptorPool> pools;
        std::vector<std::vector<VkDescriptorSetLayoutBinding>> setLayouts;
        std::vector<VkDescriptorSetLayoutCreateFlags> setLayoutFlags;
        std::vector<std::vector<VkDescriptorBindingFlags>> setLayoutBindingFlags;
        std::vector<RecordedImageDescriptorWrite> imageWrites;
        std::size_t descriptorWriteCount = 0;
        std::size_t pipelineLayoutCount = 0;
        std::vector<std::vector<VkPushConstantRange>> pipelineLayoutPushConstants;
        VkResult nextAllocateResult = VK_SUCCESS;
    };

    thread_local MockDescriptorDispatch* gDescriptorMock = nullptr;

    [[nodiscard]] MockDescriptorPool* findMockDescriptorPool(VkDescriptorPool pool)
    {
        const auto found = std::ranges::find(gDescriptorMock->pools, pool, &MockDescriptorPool::pool);
        return found == gDescriptorMock->pools.end() ? nullptr : &*found;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockCreateDescriptorPool(VkDevice,
                                                            const VkDescriptorPoolCreateInfo* pCreateInfo,
                                                            const VkAllocationCallbacks*,
                                                            VkDescriptorPool* pDescriptorPool)
    {
        *pDescriptorPool = fakeHandle<VkDescriptorPool>(gDescriptorMock->nextHandle++);
        MockDescriptorPool pool{
            .pool = *pDescriptorPool,
            .maxSets = pCreateInfo->maxSets,
            .sizes = {},
            .liveSets = 0,
            .flags = pCreateInfo->flags,
        };
        for (std::uint32_t index = 0; index < pCreateInfo->poolSizeCount; ++index)
        {
            pool.sizes.push_back(pCreateInfo->pPoolSizes[index]);
        }
        gDescriptorMock->pools.push_back(std::move(pool));
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockDestroyDescriptorPool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL
    descriptorMockCreateDescriptorSetLayout(VkDevice,
                                            const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
                                            const VkAllocationCallbacks*,
                                            VkDescriptorSetLayout* pSetLayout)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (std::uint32_t index = 0; index < pCreateInfo->bindingCount; ++index)
        {
            bindings.push_back(pCreateInfo->pBindings[index]);
        }
        gDescriptorMock->setLayouts.push_back(std::move(bindings));
        gDescriptorMock->setLayoutFlags.push_back(pCreateInfo->flags);
        std::vector<VkDescriptorBindingFlags> bindingFlags;
        if (const auto* flagsInfo = static_cast<const VkDescriptorSetLayoutBindingFlagsCreateInfo*>(pCreateInfo->pNext);
            flagsInfo != nullptr &&
            flagsInfo->sType == VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO)
        {
            bindingFlags.assign(flagsInfo->pBindingFlags, flagsInfo->pBindingFlags + flagsInfo->bindingCount);
        }
        gDescriptorMock->setLayoutBindingFlags.push_back(std::move(bindingFlags));
        *pSetLayout = fakeHandle<VkDescriptorSetLayout>(gDescriptorMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL descriptorMockDestroyDescriptorSetLayout(VkDevice,
                                                                        VkDescriptorSetLayout,
                                                                        const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockAllocateDescriptorSets(VkDevice,
                                                              const VkDescriptorSetAllocateInfo* pAllocateInfo,
                                                              VkDescriptorSet* pDescriptorSets)
    {
        if (gDescriptorMock->nextAllocateResult != VK_SUCCESS)
        {
            return std::exchange(gDescriptorMock->nextAllocateResult, VK_SUCCESS);
        }
        auto* pool = findMockDescriptorPool(pAllocateInfo->descriptorPool);
        if (pool == nullptr || pAllocateInfo->descriptorSetCount > pool->maxSets - pool->liveSets)
        {
            return VK_ERROR_OUT_OF_POOL_MEMORY;
        }
        for (std::uint32_t index = 0; index < pAllocateInfo->descriptorSetCount; ++index)
        {
            pDescriptorSets[index] = fakeHandle<VkDescriptorSet>(gDescriptorMock->nextHandle++);
        }
        pool->liveSets += pAllocateInfo->descriptorSetCount;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockFreeDescriptorSets(VkDevice,
                                                          VkDescriptorPool descriptorPool,
                                                          std::uint32_t descriptorSetCount,
                                                          const VkDescriptorSet*)
    {
        if (auto* pool = findMockDescriptorPool(descriptorPool); pool != nullptr)
        {
            pool->liveSets -= descriptorSetCount;
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockUpdateDescriptorSets(VkDevice,
                                                        std::uint32_t descriptorWriteCount,
                                                        const VkWriteDescriptorSet* pDescriptorWrites,
                                                        std::uint32_t,
                                                        const VkCopyDescriptorSet*)
    {
        gDescriptorMock->descriptorWriteCount += descriptorWriteCount;
        for (std::uint32_t index = 0; index < descriptorWriteCount; ++index)
        {
            const auto& write = pDescriptorWrites[index];
            if (write.descriptorType != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
                write.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
            {
                continue;
            }
            for (std::uint32_t element = 0; element < write.descriptorCount; ++element)
            {
                gDescriptorMock->imageWrites.push_back({
                    .type = write.descriptorType,
                    .layout = write.pImageInfo[element].imageLayout,
                });
            }
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL descriptorMockCreatePipelineLayout(VkDevice,
                                                                      const VkPipelineLayoutCreateInfo* pCreateInfo,
                                                                      const VkAllocationCallbacks*,
                                                                      VkPipelineLayout* pPipelineLayout)
    {
        ++gDescriptorMock->pipelineLayoutCount;
        gDescriptorMock->pipelineLayoutPushConstants.emplace_back(
            pCreateInfo->pPushConstantRanges, pCreateInfo->pPushConstantRanges + pCreateInfo->pushConstantRangeCount);
        *pPipelineLayout = fakeHandle<VkPipelineLayout>(gDescriptorMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL descriptorMockDestroyPipelineLayout(VkDevice,
                                                                   VkPipelineLayout,
                                                                   const VkAllocationCallbacks*)
    {
    }

    [[nodiscard]] VolkDeviceTable makeMockDescriptorTable() noexcept
    {
        VolkDeviceTable table{};
        table.vkCreateDescriptorPool = &mockCreateDescriptorPool;
        table.vkDestroyDescriptorPool = &mockDestroyDescriptorPool;
        table.vkCreateDescriptorSetLayout = &descriptorMockCreateDescriptorSetLayout;
        table.vkDestroyDescriptorSetLayout = &descriptorMockDestroyDescriptorSetLayout;
        table.vkAllocateDescriptorSets = &mockAllocateDescriptorSets;
        table.vkFreeDescriptorSets = &mockFreeDescriptorSets;
        table.vkUpdateDescriptorSets = &mockUpdateDescriptorSets;
        table.vkCreatePipelineLayout = &descriptorMockCreatePipelineLayout;
        table.vkDestroyPipelineLayout = &descriptorMockDestroyPipelineLayout;
        return table;
    }

} // namespace

TEST_CASE("Vulkan binding pools size native pools for every admitted copy-on-write version and report exhaustion",
          "[size:small][rhi][vulkan][vuid][bindings-mock]")
{
    MockDescriptorDispatch mock;
    gDescriptorMock = &mock;
    const auto table = makeMockDescriptorTable();
    const auto device = fakeHandle<VkDevice>(0x40);
    rachis::DeviceCapabilities caps{};
    caps.partiallyBoundBindingArrays = true;
    rachis::vulkan::VulkanResourceAllocator resources;
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps);

    const std::array oneUniform{rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 8}};
    REQUIRE(bindings.createBindingPool({.maxBindingSets = 0x8000'0000U, .sizes = oneUniform}).has_value());
    REQUIRE(mock.pools.size() == 1);
    CHECK(mock.pools.back().maxSets >= 0x8000'0000U);
    REQUIRE(bindings
                .createBindingPool({
                    .maxBindingSets = std::numeric_limits<std::uint32_t>::max(),
                    .sizes = oneUniform,
                })
                .has_value());
    REQUIRE(mock.pools.size() == 2);
    CHECK(mock.pools.back().maxSets == std::numeric_limits<std::uint32_t>::max());

    const std::array feedbackSizes{
        rachis::BindingPoolSize{rachis::BindingType::sampler, 4},
        rachis::BindingPoolSize{rachis::BindingType::minMipFeedback, 16},
        rachis::BindingPoolSize{rachis::BindingType::mipRegionUsedFeedback, 16},
    };
    REQUIRE(bindings.createBindingPool({.maxBindingSets = 1, .sizes = feedbackSizes}).has_value());
    REQUIRE(mock.pools.size() == 3);
    CHECK(mock.pools.back().sizes.size() == 1);
    CHECK(std::ranges::count(mock.pools.back().sizes, VK_DESCRIPTOR_TYPE_SAMPLER, &VkDescriptorPoolSize::type) == 1);

    const std::array feedbackOnly{rachis::BindingPoolSize{rachis::BindingType::minMipFeedback, 4}};
    const auto feedbackOnlyPool = bindings.createBindingPool({.maxBindingSets = 1, .sizes = feedbackOnly});
    REQUIRE(feedbackOnlyPool.has_value());
    REQUIRE(mock.pools.size() == 4);
    CHECK(mock.pools.back().sizes.empty());
    const auto emptyLayout = bindings.createBindingLayout({});
    REQUIRE(emptyLayout.has_value());
    const auto emptySet = bindings.createBindingSet(resources, {.layout = *emptyLayout, .pool = *feedbackOnlyPool});
    REQUIRE(emptySet.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0,
                                   rachis::BindingType::uniformBuffer,
                                   rachis::ShaderStage::compute,
                                   1,
                                   rachis::BindingFlag::partiallyBound},
        rachis::BindingLayoutEntry{1,
                                   rachis::BindingType::storageBuffer,
                                   rachis::ShaderStage::compute,
                                   1,
                                   rachis::BindingFlag::partiallyBound},
    };
    const auto layout = bindings.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    constexpr std::uint32_t versionCapacity = 8;
    const std::array versionSizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, versionCapacity},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, versionCapacity},
    };
    const auto pool = bindings.createBindingPool({.maxBindingSets = 1, .sizes = versionSizes});
    REQUIRE(pool.has_value());
    const auto poolIndex = mock.pools.size() - 1;
    const auto set = bindings.createBindingSet(resources, {.layout = *layout, .pool = *pool});
    REQUIRE(set.has_value());
    const std::array clear{rachis::BindingWrite{.binding = 0, .resource = std::monostate{}}};

    const auto spare = bindings.createBindingPool({.maxBindingSets = 1, .sizes = versionSizes});
    REQUIRE(spare.has_value());
    mock.nextAllocateResult = VK_ERROR_FRAGMENTED_POOL;
    const auto fragmented = bindings.createBindingSet(resources, {.layout = *layout, .pool = *spare});
    REQUIRE_FALSE(fragmented.has_value());
    CHECK(fragmented.error().code == rachis::ErrorCode::resourceExhausted);
    mock.nextAllocateResult = VK_ERROR_OUT_OF_POOL_MEMORY;
    const auto outOfPool = bindings.updateBindingSet(resources, *set, clear);
    REQUIRE_FALSE(outOfPool.has_value());
    CHECK(outOfPool.error().code == rachis::ErrorCode::resourceExhausted);

    std::vector<rachis::vulkan::VulkanCommandUse> retained;
    for (std::uint32_t round = 1; round < versionCapacity; ++round)
    {
        CAPTURE(round);
        auto native = bindings.retainBindingSetForCommand(*set, true);
        REQUIRE(native.has_value());
        retained.push_back(std::move(native->commandUse));
        auto updated = bindings.updateBindingSet(resources, *set, clear);
        if (!updated)
        {
            FAIL(updated.error().message);
        }
    }
    auto current = bindings.retainBindingSetForCommand(*set, true);
    REQUIRE(current.has_value());
    retained.push_back(std::move(current->commandUse));
    auto exhausted = bindings.updateBindingSet(resources, *set, clear);
    REQUIRE_FALSE(exhausted.has_value());
    CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);
    CHECK(mock.pools[poolIndex].liveSets == versionCapacity);
    retained.clear();
    current->owner.reset();
    CHECK(mock.pools[poolIndex].liveSets == 1);
    REQUIRE(bindings.updateBindingSet(resources, *set, clear).has_value());
    CHECK(mock.pools[poolIndex].liveSets == 1);

    const std::array exactSizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 1},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 1},
    };
    const auto exactPool = bindings.createBindingPool({.maxBindingSets = 1, .sizes = exactSizes});
    REQUIRE(exactPool.has_value());
    const auto exactPoolIndex = mock.pools.size() - 1;
    const auto exactSet = bindings.createBindingSet(resources, {.layout = *layout, .pool = *exactPool});
    REQUIRE(exactSet.has_value());
    for (std::uint32_t round = 0; round < versionCapacity; ++round)
    {
        CAPTURE(round);
        REQUIRE(bindings.updateBindingSet(resources, *exactSet, clear).has_value());
        CHECK(mock.pools[exactPoolIndex].liveSets == 1);
    }
    {
        auto bound = bindings.retainBindingSetForCommand(*exactSet, false);
        REQUIRE(bound.has_value());
        auto boundUpdate = bindings.updateBindingSet(resources, *exactSet, clear);
        REQUIRE_FALSE(boundUpdate.has_value());
        CHECK(boundUpdate.error().code == rachis::ErrorCode::resourceExhausted);
        CHECK(mock.pools[exactPoolIndex].liveSets == 1);
    }
    REQUIRE(bindings.updateBindingSet(resources, *exactSet, clear).has_value());
    REQUIRE(bindings.destroyBindingSet(*exactSet).has_value());
    CHECK(mock.pools[exactPoolIndex].liveSets == 0);

    const std::array twoVersionSizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 2},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 2},
    };
    const auto twoVersionPool = bindings.createBindingPool({.maxBindingSets = 1, .sizes = twoVersionSizes});
    REQUIRE(twoVersionPool.has_value());
    const auto twoVersionPoolIndex = mock.pools.size() - 1;
    for (const bool retainResources : {true, false})
    {
        CAPTURE(retainResources);
        const auto twoVersionSet = bindings.createBindingSet(resources, {.layout = *layout, .pool = *twoVersionPool});
        REQUIRE(twoVersionSet.has_value());
        {
            auto bound = bindings.retainBindingSetForCommand(*twoVersionSet, retainResources);
            REQUIRE(bound.has_value());
            const auto boundDescriptorSet = bound->descriptorSet;
            auto use = std::move(bound->commandUse);
            bound->owner.reset();
            REQUIRE(bindings.updateBindingSet(resources, *twoVersionSet, clear).has_value());
            CHECK(mock.pools[twoVersionPoolIndex].liveSets == 2);
            const auto exhaustedWhileBound = bindings.updateBindingSet(resources, *twoVersionSet, clear);
            REQUIRE_FALSE(exhaustedWhileBound.has_value());
            CHECK(exhaustedWhileBound.error().code == rachis::ErrorCode::resourceExhausted);
            CHECK(mock.pools[twoVersionPoolIndex].liveSets == 2);
            const auto latest = bindings.nativeBindingSet(*twoVersionSet);
            REQUIRE(latest.has_value());
            CHECK(latest->descriptorSet != boundDescriptorSet);
            use.reset();
            CHECK(mock.pools[twoVersionPoolIndex].liveSets == 1);
        }
        REQUIRE(bindings.updateBindingSet(resources, *twoVersionSet, clear).has_value());
        CHECK(mock.pools[twoVersionPoolIndex].liveSets == 1);
        REQUIRE(bindings.destroyBindingSet(*twoVersionSet).has_value());
        CHECK(mock.pools[twoVersionPoolIndex].liveSets == 0);
    }

    REQUIRE(bindings.destroyBindingSet(*set).has_value());
    REQUIRE(bindings.destroyBindingSet(*emptySet).has_value());
    bindings.shutdown();
    gDescriptorMock = nullptr;
}

TEST_CASE("Vulkan variable-count arrays and pipeline layouts stay within per-class descriptor limits",
          "[size:small][rhi][vulkan][vuid][bindings-mock]")
{
    MockDescriptorDispatch mock;
    gDescriptorMock = &mock;
    const auto table = makeMockDescriptorTable();
    const auto device = fakeHandle<VkDevice>(0x41);
    rachis::DeviceCapabilities caps{};
    caps.partiallyBoundBindingArrays = true;
    caps.variableDescriptorCountBindingArrays = true;
    caps.limits.maxVariableResourceDescriptors = 31;
    caps.limits.maxVariableSamplerDescriptors = 16;
    const rachis::vulkan::VulkanDescriptorLimits limits{
        .maxPerStageDescriptorSamplers = 16,
        .maxPerStageDescriptorUniformBuffers = 31,
        .maxPerStageDescriptorStorageBuffers = 31,
        .maxPerStageDescriptorSampledImages = 256,
        .maxPerStageDescriptorStorageImages = 256,
        .maxPerStageResources = 287,
        .maxDescriptorSetSamplers = 80,
        .maxDescriptorSetUniformBuffers = 155,
        .maxDescriptorSetUniformBuffersDynamic = 15,
        .maxDescriptorSetStorageBuffers = 155,
        .maxDescriptorSetStorageBuffersDynamic = 16,
        .maxDescriptorSetSampledImages = 1280,
        .maxDescriptorSetStorageImages = 1280,
    };
    rachis::vulkan::VulkanResourceAllocator resources;
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps, nullptr, {}, limits);

    const auto variableEntry = [](std::uint32_t binding, rachis::BindingType type, rachis::ShaderStage visibility) {
        return rachis::BindingLayoutEntry{
            .binding = binding,
            .type = type,
            .visibility = visibility,
            .count = rachis::unboundedBindingCount,
            .flags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::variableDescriptorCount,
        };
    };
    const auto createLayout = [&](std::span<const rachis::BindingLayoutEntry> entries) {
        auto layout = bindings.createBindingLayout({.entries = entries});
        REQUIRE(layout.has_value());
        return *layout;
    };
    const auto lastDescriptorCount = [&] {
        REQUIRE_FALSE(mock.setLayouts.empty());
        REQUIRE_FALSE(mock.setLayouts.back().empty());
        return mock.setLayouts.back().back().descriptorCount;
    };

    const std::array variableSamplers{variableEntry(0, rachis::BindingType::sampler, rachis::ShaderStage::compute)};
    const auto samplerLayout = createLayout(variableSamplers);
    CHECK(lastDescriptorCount() == 12);
    const std::array variableUniforms{
        variableEntry(0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute)};
    const auto uniformLayout = createLayout(variableUniforms);
    CHECK(lastDescriptorCount() == 24);
    const std::array variableStorage{
        variableEntry(0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute)};
    const auto storageLayout = createLayout(variableStorage);
    CHECK(lastDescriptorCount() == 24);
    CHECK(bindings.createPipelineLayout({.bindingLayouts = std::array{samplerLayout}}).has_value());
    CHECK(bindings.createPipelineLayout({.bindingLayouts = std::array{uniformLayout}}).has_value());
    CHECK(bindings.createPipelineLayout({.bindingLayouts = std::array{storageLayout}}).has_value());
    CHECK(mock.pipelineLayoutCount == 3);

    const auto fixedStorageLayout = [&](std::uint32_t count) {
        const std::array fixedStorage{rachis::BindingLayoutEntry{0,
                                                                 rachis::BindingType::storageBuffer,
                                                                 rachis::ShaderStage::compute,
                                                                 count,
                                                                 rachis::BindingFlag::partiallyBound}};
        return createLayout(fixedStorage);
    };
    CHECK(bindings.createPipelineLayout({.bindingLayouts = std::array{fixedStorageLayout(1), storageLayout}})
              .has_value());
    CHECK(bindings.createPipelineLayout({.bindingLayouts = std::array{fixedStorageLayout(7), storageLayout}})
              .has_value());
    CHECK(mock.pipelineLayoutCount == 5);
    const auto acrossSets =
        bindings.createPipelineLayout({.bindingLayouts = std::array{fixedStorageLayout(8), storageLayout}});
    REQUIRE_FALSE(acrossSets.has_value());
    CHECK(acrossSets.error().code == rachis::ErrorCode::unsupported);
    const auto twoVariableArrays =
        bindings.createPipelineLayout({.bindingLayouts = std::array{storageLayout, storageLayout}});
    REQUIRE_FALSE(twoVariableArrays.has_value());
    CHECK(twoVariableArrays.error().code == rachis::ErrorCode::unsupported);
    CHECK(mock.pipelineLayoutCount == 5);

    const std::array sharedStage{
        rachis::BindingLayoutEntry{0,
                                   rachis::BindingType::storageBuffer,
                                   rachis::ShaderStage::compute,
                                   1,
                                   rachis::BindingFlag::partiallyBound},
        variableEntry(1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute),
    };
    const auto sharedStageLayout = createLayout(sharedStage);
    CHECK(lastDescriptorCount() == 23);
    CHECK(bindings.createPipelineLayout({.bindingLayouts = std::array{sharedStageLayout}}).has_value());
    const std::array storageSizes{rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 64}};
    const auto pool = bindings.createBindingPool({.maxBindingSets = 2, .sizes = storageSizes});
    REQUIRE(pool.has_value());
    const auto overCapacity = bindings.createBindingSet(
        resources, {.layout = sharedStageLayout, .variableDescriptorCount = 24, .pool = *pool});
    REQUIRE_FALSE(overCapacity.has_value());
    CHECK(overCapacity.error().code == rachis::ErrorCode::invalidArgument);
    const auto atCapacity = bindings.createBindingSet(
        resources, {.layout = sharedStageLayout, .variableDescriptorCount = 23, .pool = *pool});
    REQUIRE(atCapacity.has_value());

    const std::array otherStage{
        rachis::BindingLayoutEntry{
            0, rachis::BindingType::storageBuffer, rachis::ShaderStage::pixel, 1, rachis::BindingFlag::partiallyBound},
        variableEntry(1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute),
    };
    static_cast<void>(createLayout(otherStage));
    CHECK(lastDescriptorCount() == 24);

    const std::array dynamicUniforms{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 8}};
    const auto dynamicLayout = createLayout(dynamicUniforms);
    CHECK(bindings.createPipelineLayout({.bindingLayouts = std::array{dynamicLayout}}).has_value());
    const auto dynamicOverflow =
        bindings.createPipelineLayout({.bindingLayouts = std::array{dynamicLayout, dynamicLayout}});
    REQUIRE_FALSE(dynamicOverflow.has_value());
    CHECK(dynamicOverflow.error().code == rachis::ErrorCode::unsupported);

    const std::array fixedSamplers{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 10}};
    const auto samplerTable = createLayout(fixedSamplers);
    CHECK(bindings.createPipelineLayout({.bindingLayouts = std::array{samplerTable}}).has_value());
    const auto samplerOverflow =
        bindings.createPipelineLayout({.bindingLayouts = std::array{samplerTable, samplerTable}});
    REQUIRE_FALSE(samplerOverflow.has_value());
    CHECK(samplerOverflow.error().code == rachis::ErrorCode::unsupported);

    rachis::DeviceCapabilities wideCaps = caps;
    wideCaps.limits.maxVariableResourceDescriptors = 65'536;
    rachis::vulkan::VulkanBindingManager uniformLimited;
    uniformLimited.initialize(device,
                              &table,
                              wideCaps,
                              nullptr,
                              {},
                              {
                                  .maxPerStageDescriptorUniformBuffers = 64,
                                  .maxDescriptorSetUniformBuffers = 384,
                              });
    const auto wideUniformLayout = uniformLimited.createBindingLayout({.entries = variableUniforms});
    REQUIRE(wideUniformLayout.has_value());
    CHECK(lastDescriptorCount() == 48);
    CHECK(uniformLimited.createPipelineLayout({.bindingLayouts = std::array{*wideUniformLayout}}).has_value());
    const auto fixedUniformLayout = [&](std::uint32_t count) {
        const std::array fixedUniform{
            rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, count}};
        auto layout = uniformLimited.createBindingLayout({.entries = fixedUniform});
        REQUIRE(layout.has_value());
        return *layout;
    };
    CHECK(
        uniformLimited.createPipelineLayout({.bindingLayouts = std::array{fixedUniformLayout(16), *wideUniformLayout}})
            .has_value());
    const auto uniformOverflow =
        uniformLimited.createPipelineLayout({.bindingLayouts = std::array{fixedUniformLayout(17), *wideUniformLayout}});
    REQUIRE_FALSE(uniformOverflow.has_value());
    CHECK(uniformOverflow.error().code == rachis::ErrorCode::unsupported);
    const auto wideStorageLayout = uniformLimited.createBindingLayout({.entries = variableStorage});
    REQUIRE(wideStorageLayout.has_value());
    CHECK(lastDescriptorCount() == 65'536);

    REQUIRE(bindings.destroyBindingSet(*atCapacity).has_value());
    uniformLimited.shutdown();
    bindings.shutdown();
    gDescriptorMock = nullptr;
}

TEST_CASE("Vulkan pipeline layouts give each shader stage one native push-constant range",
          "[size:small][rhi][vulkan][vuid][bindings-mock][push-constants]")
{
    using rachis::PushConstantRange;
    using rachis::ShaderStage;

    MockDescriptorDispatch mock;
    gDescriptorMock = &mock;
    const auto table = makeMockDescriptorTable();
    const auto device = fakeHandle<VkDevice>(0x42);
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, rachis::DeviceCapabilities{});

    const std::array sharedThenPixel{
        PushConstantRange{.visibility = ShaderStage::vertex | ShaderStage::pixel, .offset = 0, .size = 16},
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 16, .size = 16},
    };
    const std::array disjointPixel{
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 0, .size = 16},
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 32, .size = 16},
    };
    const std::array vertexInPixelGap{
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 0, .size = 16},
        PushConstantRange{.visibility = ShaderStage::vertex, .offset = 16, .size = 16},
        PushConstantRange{.visibility = ShaderStage::pixel, .offset = 32, .size = 16},
    };
    const std::array computeExtension{
        PushConstantRange{.visibility = ShaderStage::compute | ShaderStage::vertex, .offset = 0, .size = 8},
        PushConstantRange{.visibility = ShaderStage::compute, .offset = 8, .size = 8},
    };
    const std::array<std::span<const PushConstantRange>, 4> layouts{
        sharedThenPixel,
        disjointPixel,
        vertexInPixelGap,
        computeExtension,
    };
    const std::array<std::vector<VkPushConstantRange>, 4> expectedNative{
        std::vector{
            VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = 16},
            VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 32},
        },
        std::vector{
            VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 48},
        },
        std::vector{
            VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 16, .size = 16},
            VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 48},
        },
        std::vector{
            VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = 8},
            VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 16},
        },
    };
    const auto nativeStage = [](ShaderStage stage) -> VkShaderStageFlags {
        VkShaderStageFlags flags = 0;
        flags |= rachis::hasAny(stage, ShaderStage::vertex)
                     ? static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_VERTEX_BIT)
                     : 0U;
        flags |= rachis::hasAny(stage, ShaderStage::pixel)
                     ? static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_FRAGMENT_BIT)
                     : 0U;
        flags |= rachis::hasAny(stage, ShaderStage::compute)
                     ? static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_COMPUTE_BIT)
                     : 0U;
        return flags;
    };
    const auto sameRange = [](const VkPushConstantRange& left, const VkPushConstantRange& right) {
        return left.stageFlags == right.stageFlags && left.offset == right.offset && left.size == right.size;
    };
    const auto sameRhiRange = [](const PushConstantRange& left, const PushConstantRange& right) {
        return left.visibility == right.visibility && left.offset == right.offset && left.size == right.size;
    };

    for (std::size_t index = 0; index < layouts.size(); ++index)
    {
        CAPTURE(index);
        const auto layout = bindings.createPipelineLayout({.pushConstantRanges = layouts[index]});
        REQUIRE(layout.has_value());
        REQUIRE(mock.pipelineLayoutPushConstants.size() == index + 1);
        const auto& native = mock.pipelineLayoutPushConstants.back();
        CHECK(std::ranges::equal(native, expectedNative[index], sameRange));

        VkShaderStageFlags seenStages = 0;
        for (const auto& range : native)
        {
            CHECK((seenStages & range.stageFlags) == 0);
            seenStages |= range.stageFlags;
        }
        for (const auto& range : layouts[index])
        {
            for (const auto stage : {ShaderStage::vertex, ShaderStage::pixel, ShaderStage::compute})
            {
                if (!rachis::hasAny(range.visibility, stage))
                {
                    continue;
                }
                const auto covers = [&](const VkPushConstantRange& candidate) {
                    return (candidate.stageFlags & nativeStage(stage)) != 0 && candidate.offset <= range.offset &&
                           range.offset + range.size <= candidate.offset + candidate.size;
                };
                CHECK(std::ranges::any_of(native, covers));
            }
        }

        const auto described = bindings.nativePipelineLayout(*layout);
        REQUIRE(described.has_value());
        CHECK(std::ranges::equal(described->pushConstantRanges, layouts[index], sameRhiRange));
        REQUIRE(described->nativePushConstantRanges.size() == native.size());
        for (std::size_t range = 0; range < native.size(); ++range)
        {
            CHECK(nativeStage(described->nativePushConstantRanges[range].visibility) == native[range].stageFlags);
            CHECK(described->nativePushConstantRanges[range].offset == native[range].offset);
            CHECK(described->nativePushConstantRanges[range].size == native[range].size);
        }
        REQUIRE(bindings.destroyPipelineLayout(*layout).has_value());
    }

    bindings.shutdown();
    gDescriptorMock = nullptr;
}

TEST_CASE("Vulkan texture descriptors name the native layout of every texture layout that admits their shader access",
          "[size:small][rhi][vulkan][vuid][bindings-mock][layouts-mock]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    MockDescriptorDispatch descriptorMock;
    gDescriptorMock = &descriptorMock;
    const auto deviceTable = makeMockDeviceTable();
    auto table = makeMockDescriptorTable();
    table.vkCreateImageView = deviceTable.vkCreateImageView;
    table.vkDestroyImageView = deviceTable.vkDestroyImageView;
    table.vkCmdPipelineBarrier2 = deviceTable.vkCmdPipelineBarrier2;

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetPhysicalDeviceFormatProperties;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &mockGetPhysicalDeviceImageFormatProperties;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x13);
    const auto device = fakeHandle<VkDevice>(0x42);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, &instanceTable, physical, families);
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, rachis::DeviceCapabilities{});

    const auto image = fakeHandle<VkImage>(0x7001);
    const auto texture =
        resources.adoptSwapchainImage(image,
                                      {
                                          .width = 4,
                                          .height = 4,
                                          .format = rachis::Format::rgba8Unorm,
                                          .usage = rachis::TextureUsage::copyDestination |
                                                   rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
                                      });
    REQUIRE(texture.has_value());
    const auto sampledView =
        resources.createTextureView({.texture = *texture, .usage = rachis::TextureViewUsage::sampled});
    const auto storageView =
        resources.createTextureView({.texture = *texture, .usage = rachis::TextureViewUsage::storage});
    REQUIRE(sampledView.has_value());
    REQUIRE(storageView.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageTexture, rachis::ShaderStage::compute, 1},
    };
    const auto layout = bindings.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array sizes{
        rachis::BindingPoolSize{rachis::BindingType::sampledTexture, 1},
        rachis::BindingPoolSize{rachis::BindingType::storageTexture, 1},
    };
    const auto pool = bindings.createBindingPool({.maxBindingSets = 1, .sizes = sizes});
    REQUIRE(pool.has_value());
    const std::array writes{
        rachis::BindingWrite{0, 0, *sampledView},
        rachis::BindingWrite{1, 0, *storageView},
    };
    const auto set = bindings.createBindingSet(resources, {.layout = *layout, .writes = writes, .pool = *pool});
    REQUIRE(set.has_value());
    REQUIRE(descriptorMock.imageWrites.size() == 2);
    const auto sampledWrite = std::ranges::find(
        descriptorMock.imageWrites, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &RecordedImageDescriptorWrite::type);
    const auto storageWrite = std::ranges::find(
        descriptorMock.imageWrites, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &RecordedImageDescriptorWrite::type);
    REQUIRE(sampledWrite != descriptorMock.imageWrites.end());
    REQUIRE(storageWrite != descriptorMock.imageWrites.end());

    const auto command = fakeHandle<VkCommandBuffer>(0x7101);
    REQUIRE(resources
                .textureBarrier(command,
                                rachis::QueueType::graphics,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutAfter = rachis::TextureLayout::common,
                                })
                .has_value());
    REQUIRE(resources
                .textureBarrier(command,
                                rachis::QueueType::graphics,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::computeShader,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::shaderResource,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::common,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                })
                .has_value());
    REQUIRE(resources
                .textureBarrier(command,
                                rachis::QueueType::graphics,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::computeShader,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::shaderResource,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutBefore = rachis::TextureLayout::shaderResource,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    REQUIRE(mock.imageBarriers.size() == 3);
    CHECK(mock.imageBarriers[0].image == image);
    CHECK(mock.imageBarriers[0].newLayout == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(mock.imageBarriers[0].newLayout == sampledWrite->layout);
    CHECK(mock.imageBarriers[1].oldLayout == sampledWrite->layout);
    CHECK(mock.imageBarriers[1].newLayout == sampledWrite->layout);
    CHECK(mock.imageBarriers[2].oldLayout == sampledWrite->layout);
    CHECK(mock.imageBarriers[2].newLayout == storageWrite->layout);
    CHECK(storageWrite->layout == VK_IMAGE_LAYOUT_GENERAL);

    resources.discardCommandLayouts(command);
    REQUIRE(bindings.destroyBindingSet(*set).has_value());
    REQUIRE(resources.destroyTextureView(*sampledView).has_value());
    REQUIRE(resources.destroyTextureView(*storageView).has_value());
    bindings.shutdown();
    resources.shutdown();
    gDescriptorMock = nullptr;
    gMock = nullptr;
}

TEST_CASE("Vulkan publishes variable-count limits that sampler, storage-buffer and texture arrays can reach",
          "[size:small][rhi][vulkan][vuid][bindings-mock]")
{
    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.runtimeDescriptorArray = true;
    record.descriptorBindingPartiallyBound = true;
    record.descriptorBindingVariableDescriptorCount = true;
    record.shaderStorageBufferArrayNonUniformIndexing = true;
    record.shaderSampledImageArrayNonUniformIndexing = true;
    record.shaderStorageImageArrayNonUniformIndexing = true;
    auto& limits = record.properties.limits;
    limits.maxPerStageDescriptorSamplers = 16;
    limits.maxPerStageDescriptorUniformBuffers = 31;
    limits.maxPerStageDescriptorStorageBuffers = 1024;
    limits.maxPerStageDescriptorSampledImages = 96;
    limits.maxPerStageDescriptorStorageImages = 256;
    limits.maxPerStageResources = 2048;
    limits.maxDescriptorSetSamplers = 80;
    limits.maxDescriptorSetUniformBuffers = 155;
    limits.maxDescriptorSetStorageBuffers = 4096;
    limits.maxDescriptorSetSampledImages = 1280;
    limits.maxDescriptorSetStorageImages = 1280;

    const rachis::vulkan::VulkanInstanceTable instanceTable{};
    const auto published = rachis::vulkan::publishVulkanCapabilities(instanceTable, record, false);
    REQUIRE(published.variableDescriptorCountBindingArrays);
    CHECK(published.limits.maxVariableResourceDescriptors == 72);
    CHECK(published.limits.maxVariableBufferDescriptors == 24);
    CHECK(published.limits.maxVariableSamplerDescriptors == 12);

    {
        MockDescriptorDispatch mock;
        gDescriptorMock = &mock;
        const auto table = makeMockDescriptorTable();
        rachis::vulkan::VulkanBindingManager bindings;
        bindings.initialize(fakeHandle<VkDevice>(0x43),
                            &table,
                            published,
                            nullptr,
                            {},
                            {
                                .maxPerStageDescriptorSamplers = limits.maxPerStageDescriptorSamplers,
                                .maxPerStageDescriptorUniformBuffers = limits.maxPerStageDescriptorUniformBuffers,
                                .maxPerStageDescriptorStorageBuffers = limits.maxPerStageDescriptorStorageBuffers,
                                .maxPerStageDescriptorSampledImages = limits.maxPerStageDescriptorSampledImages,
                                .maxPerStageDescriptorStorageImages = limits.maxPerStageDescriptorStorageImages,
                                .maxPerStageResources = limits.maxPerStageResources,
                                .maxDescriptorSetSamplers = limits.maxDescriptorSetSamplers,
                                .maxDescriptorSetUniformBuffers = limits.maxDescriptorSetUniformBuffers,
                                .maxDescriptorSetStorageBuffers = limits.maxDescriptorSetStorageBuffers,
                                .maxDescriptorSetSampledImages = limits.maxDescriptorSetSampledImages,
                                .maxDescriptorSetStorageImages = limits.maxDescriptorSetStorageImages,
                            });
        const auto declaredLength = [&](rachis::BindingType type) {
            const std::array entries{rachis::BindingLayoutEntry{
                .binding = 0,
                .type = type,
                .visibility = rachis::ShaderStage::compute,
                .count = rachis::unboundedBindingCount,
                .flags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::variableDescriptorCount,
            }};
            REQUIRE(bindings.createBindingLayout({.entries = entries}).has_value());
            REQUIRE_FALSE(mock.setLayouts.empty());
            REQUIRE_FALSE(mock.setLayouts.back().empty());
            return mock.setLayouts.back().back().descriptorCount;
        };
        CHECK(declaredLength(rachis::BindingType::storageBuffer) == published.limits.maxVariableResourceDescriptors);
        CHECK(declaredLength(rachis::BindingType::sampledTexture) == published.limits.maxVariableResourceDescriptors);
        CHECK(declaredLength(rachis::BindingType::storageTexture) == published.limits.maxVariableResourceDescriptors);
        CHECK(declaredLength(rachis::BindingType::sampler) == published.limits.maxVariableSamplerDescriptors);
        CHECK(declaredLength(rachis::BindingType::uniformBuffer) == published.limits.maxVariableBufferDescriptors);
        bindings.shutdown();
        gDescriptorMock = nullptr;
    }

    limits.maxPerStageDescriptorSamplers = 1'048'576;
    limits.maxDescriptorSetSamplers = 1'048'576;
    limits.maxPerStageDescriptorSampledImages = 1'048'576;
    limits.maxDescriptorSetSampledImages = 1'048'576;
    limits.maxPerStageDescriptorStorageImages = 1'048'576;
    limits.maxDescriptorSetStorageImages = 1'048'576;
    limits.maxPerStageResources = 100;
    const auto resourceBound = rachis::vulkan::publishVulkanCapabilities(instanceTable, record, false);
    CHECK(resourceBound.limits.maxVariableResourceDescriptors == 75);
    CHECK(resourceBound.limits.maxVariableBufferDescriptors == 24);
    CHECK(resourceBound.limits.maxVariableSamplerDescriptors == 2048);

    record.descriptorBindingVariableDescriptorCount = false;
    const auto withoutVariableCounts = rachis::vulkan::publishVulkanCapabilities(instanceTable, record, false);
    CHECK(withoutVariableCounts.limits.maxVariableResourceDescriptors == 0);
    CHECK(withoutVariableCounts.limits.maxVariableBufferDescriptors == 0);
    CHECK(withoutVariableCounts.limits.maxVariableSamplerDescriptors == 0);
}
TEST_CASE("Vulkan publishes update-after-bind binding arrays only with every required descriptor-indexing feature",
          "[size:small][rhi][vulkan][vuid][bindings-mock]")
{
    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.runtimeDescriptorArray = true;
    record.descriptorBindingPartiallyBound = true;
    record.shaderStorageBufferArrayNonUniformIndexing = true;
    record.shaderSampledImageArrayNonUniformIndexing = true;
    record.shaderStorageImageArrayNonUniformIndexing = true;
    record.descriptorBindingUpdateUnusedWhilePending = true;
    record.descriptorBindingSampledImageUpdateAfterBind = true;
    record.descriptorBindingStorageImageUpdateAfterBind = true;
    record.descriptorBindingStorageBufferUpdateAfterBind = true;
    const rachis::vulkan::VulkanInstanceTable instanceTable{};
    CHECK(rachis::vulkan::publishVulkanCapabilities(instanceTable, record, false).updateAfterBindBindingArrays);

    const std::array requirements{
        &rachis::vulkan::VulkanPhysicalDeviceRecord::descriptorBindingPartiallyBound,
        &rachis::vulkan::VulkanPhysicalDeviceRecord::descriptorBindingUpdateUnusedWhilePending,
        &rachis::vulkan::VulkanPhysicalDeviceRecord::descriptorBindingSampledImageUpdateAfterBind,
        &rachis::vulkan::VulkanPhysicalDeviceRecord::descriptorBindingStorageImageUpdateAfterBind,
        &rachis::vulkan::VulkanPhysicalDeviceRecord::descriptorBindingStorageBufferUpdateAfterBind,
    };
    for (std::size_t index = 0; index < requirements.size(); ++index)
    {
        CAPTURE(index);
        auto missing = record;
        missing.*requirements[index] = false;
        CHECK_FALSE(
            rachis::vulkan::publishVulkanCapabilities(instanceTable, missing, false).updateAfterBindBindingArrays);
    }
}

TEST_CASE("Vulkan update-after-bind sets come from an update-after-bind pool and change in place while bound",
          "[size:small][rhi][vulkan][vuid][bindings-mock]")
{
    MockDescriptorDispatch mock;
    gDescriptorMock = &mock;
    const auto table = makeMockDescriptorTable();
    rachis::DeviceCapabilities caps{};
    caps.partiallyBoundBindingArrays = true;
    caps.updateAfterBindBindingArrays = true;
    rachis::vulkan::VulkanResourceAllocator resources;
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(fakeHandle<VkDevice>(0x44), &table, caps);

    const std::array defaultEntries{
        rachis::BindingLayoutEntry{0,
                                   rachis::BindingType::sampledTexture,
                                   rachis::ShaderStage::compute,
                                   8,
                                   rachis::BindingFlag::partiallyBound},
    };
    REQUIRE(bindings.createBindingLayout({.entries = defaultEntries}).has_value());
    CHECK(mock.setLayoutFlags.back() == 0);
    REQUIRE(mock.setLayoutBindingFlags.back().size() == 1);
    CHECK(mock.setLayoutBindingFlags.back()[0] == VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

    const std::array entries{
        rachis::BindingLayoutEntry{0,
                                   rachis::BindingType::storageBuffer,
                                   rachis::ShaderStage::compute,
                                   1,
                                   rachis::BindingFlag::partiallyBound},
        rachis::BindingLayoutEntry{1,
                                   rachis::BindingType::sampledTexture,
                                   rachis::ShaderStage::compute,
                                   8,
                                   rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind},
    };
    const auto layout = bindings.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    CHECK(mock.setLayoutFlags.back() == VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
    REQUIRE(mock.setLayoutBindingFlags.back().size() == 2);
    CHECK(mock.setLayoutBindingFlags.back()[0] == VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    CHECK(mock.setLayoutBindingFlags.back()[1] ==
          (VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
           VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT));

    const std::array sizes{
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 4},
        rachis::BindingPoolSize{rachis::BindingType::sampledTexture, 32},
    };
    const auto pool = bindings.createBindingPool({.maxBindingSets = 2, .sizes = sizes});
    REQUIRE(pool.has_value());
    REQUIRE(mock.pools.size() == 1);
    CHECK(mock.pools[0].flags == VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

    const auto set = bindings.createBindingSet(resources, {.layout = *layout, .pool = *pool});
    REQUIRE(set.has_value());
    REQUIRE(mock.pools.size() == 2);
    CHECK(mock.pools[1].flags ==
          (VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT));
    CHECK(mock.pools[1].maxSets == mock.pools[0].maxSets);
    REQUIRE(mock.pools[1].sizes.size() == mock.pools[0].sizes.size());
    for (std::size_t index = 0; index < mock.pools[0].sizes.size(); ++index)
    {
        CHECK(mock.pools[1].sizes[index].type == mock.pools[0].sizes[index].type);
        CHECK(mock.pools[1].sizes[index].descriptorCount == mock.pools[0].sizes[index].descriptorCount);
    }
    CHECK(mock.pools[0].liveSets == 0);
    CHECK(mock.pools[1].liveSets == 1);

    auto bound = bindings.retainBindingSetForCommand(*set, true);
    REQUIRE(bound.has_value());
    const auto boundDescriptorSet = bound->descriptorSet;
    const auto beforeInPlace = bindings.statistics();
    const std::array clearUpdateAfterBind{
        rachis::BindingWrite{.binding = 1, .arrayElement = 5, .resource = std::monostate{}},
    };
    REQUIRE(bindings.updateBindingSet(resources, *set, clearUpdateAfterBind).has_value());
    const auto afterInPlace = bindings.statistics();
    CHECK(afterInPlace.resourceDescriptorsPrepared - beforeInPlace.resourceDescriptorsPrepared == 1);
    CHECK(afterInPlace.resourceTablePublishCount - beforeInPlace.resourceTablePublishCount == 1);
    CHECK(afterInPlace.resourceDescriptorsPublished - beforeInPlace.resourceDescriptorsPublished == 1);
    CHECK(mock.pools[1].liveSets == 1);
    auto inPlace = bindings.nativeBindingSet(*set);
    REQUIRE(inPlace.has_value());
    CHECK(inPlace->descriptorSet == boundDescriptorSet);

    const std::array clearFixed{rachis::BindingWrite{.binding = 0, .resource = std::monostate{}}};
    REQUIRE(bindings.updateBindingSet(resources, *set, clearFixed).has_value());
    const auto afterVersioned = bindings.statistics();
    CHECK(afterVersioned.resourceDescriptorsPublished - afterInPlace.resourceDescriptorsPublished == 9);
    CHECK(mock.pools[1].liveSets == 2);
    auto versioned = bindings.nativeBindingSet(*set);
    REQUIRE(versioned.has_value());
    CHECK(versioned->descriptorSet != boundDescriptorSet);
    bound->commandUse.reset();
    bound->owner.reset();
    inPlace->owner.reset();
    CHECK(mock.pools[1].liveSets == 1);

    const auto secondSet = bindings.createBindingSet(resources, {.layout = *layout, .pool = *pool});
    REQUIRE(secondSet.has_value());
    CHECK(mock.pools.size() == 2);
    CHECK(mock.pools[1].liveSets == 2);
    versioned->owner.reset();
    REQUIRE(bindings.destroyBindingSet(*secondSet).has_value());
    REQUIRE(bindings.destroyBindingSet(*set).has_value());
    CHECK(mock.pools[1].liveSets == 0);

    const std::array withDynamic{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
        entries[1],
    };
    const auto dynamicLayout = bindings.createBindingLayout({.entries = withDynamic});
    REQUIRE_FALSE(dynamicLayout.has_value());
    CHECK(dynamicLayout.error().code == rachis::ErrorCode::invalidArgument);
    const std::array samplerEntries{rachis::BindingLayoutEntry{
        0, rachis::BindingType::sampler, rachis::ShaderStage::compute, 4, rachis::BindingFlag::updateAfterBind}};
    const auto samplerLayout = bindings.createBindingLayout({.entries = samplerEntries});
    REQUIRE_FALSE(samplerLayout.has_value());
    CHECK(samplerLayout.error().code == rachis::ErrorCode::invalidArgument);

    rachis::vulkan::VulkanBindingManager withoutSupport;
    caps.updateAfterBindBindingArrays = false;
    withoutSupport.initialize(fakeHandle<VkDevice>(0x45), &table, caps);
    const auto unsupported = withoutSupport.createBindingLayout({.entries = entries});
    REQUIRE_FALSE(unsupported.has_value());
    CHECK(unsupported.error().code == rachis::ErrorCode::unsupported);

    withoutSupport.shutdown();
    bindings.shutdown();
    gDescriptorMock = nullptr;
}

TEST_CASE("Vulkan pipeline layouts hold update-after-bind sets to the update-after-bind descriptor limits",
          "[size:small][rhi][vulkan][vuid][bindings-mock]")
{
    MockDescriptorDispatch mock;
    gDescriptorMock = &mock;
    const auto table = makeMockDescriptorTable();
    rachis::DeviceCapabilities caps{};
    caps.partiallyBoundBindingArrays = true;
    caps.updateAfterBindBindingArrays = true;
    const rachis::vulkan::VulkanDescriptorLimits regular{
        .maxPerStageDescriptorSampledImages = 16,
        .maxPerStageResources = 64,
        .maxDescriptorSetSampledImages = 64,
    };
    const rachis::vulkan::VulkanDescriptorLimits updateAfterBind{
        .maxPerStageDescriptorSampledImages = 96,
        .maxPerStageResources = 1024,
        .maxDescriptorSetSampledImages = 1024,
    };
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(fakeHandle<VkDevice>(0x46), &table, caps, nullptr, {}, regular, updateAfterBind);

    const auto textures = [&](std::uint32_t count, rachis::BindingFlag flags) {
        const std::array entries{rachis::BindingLayoutEntry{
            0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, count, flags}};
        auto layout = bindings.createBindingLayout({.entries = entries});
        REQUIRE(layout.has_value());
        return *layout;
    };
    const auto pipelineLayout = [&](std::initializer_list<rachis::BindingLayoutHandle> layouts) {
        const std::vector<rachis::BindingLayoutHandle> sets(layouts);
        return bindings.createPipelineLayout({.bindingLayouts = sets});
    };
    const auto updateAfterBindFlags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind;

    CHECK(pipelineLayout({textures(64, updateAfterBindFlags)}).has_value());
    const auto overRegular = pipelineLayout({textures(32, rachis::BindingFlag::partiallyBound)});
    REQUIRE_FALSE(overRegular.has_value());
    CHECK(overRegular.error().code == rachis::ErrorCode::unsupported);
    CHECK(pipelineLayout({textures(80, updateAfterBindFlags), textures(16, rachis::BindingFlag::none)}).has_value());
    const auto overUpdateAfterBind =
        pipelineLayout({textures(90, updateAfterBindFlags), textures(16, rachis::BindingFlag::none)});
    REQUIRE_FALSE(overUpdateAfterBind.has_value());
    CHECK(overUpdateAfterBind.error().code == rachis::ErrorCode::unsupported);
    const auto overRegularBesideUpdateAfterBind =
        pipelineLayout({textures(8, updateAfterBindFlags), textures(17, rachis::BindingFlag::none)});
    REQUIRE_FALSE(overRegularBesideUpdateAfterBind.has_value());
    CHECK(overRegularBesideUpdateAfterBind.error().code == rachis::ErrorCode::unsupported);

    bindings.shutdown();
    gDescriptorMock = nullptr;
}

namespace
{

    struct LayoutScope final
    {
        rachis::PipelineStage stages = rachis::PipelineStage::none;
        rachis::Access access = rachis::Access::none;
    };

    [[nodiscard]] constexpr LayoutScope layoutScope(rachis::TextureLayout layout) noexcept
    {
        switch (layout)
        {
        case rachis::TextureLayout::copyDestination:
            return {rachis::PipelineStage::copy, rachis::Access::copyDestination};
        case rachis::TextureLayout::copySource:
            return {rachis::PipelineStage::copy, rachis::Access::copySource};
        case rachis::TextureLayout::shaderResource:
            return {rachis::PipelineStage::pixelShader, rachis::Access::shaderResource};
        case rachis::TextureLayout::unorderedAccess:
            return {rachis::PipelineStage::computeShader, rachis::Access::unorderedAccess};
        case rachis::TextureLayout::renderTarget:
            return {rachis::PipelineStage::renderTarget, rachis::Access::renderTarget};
        default:
            return {};
        }
    }

    [[nodiscard]] rachis::TextureBarrier layoutBarrier(rachis::TextureHandle texture,
                                                       rachis::TextureLayout before,
                                                       rachis::TextureLayout after,
                                                       rachis::TextureSubresourceRange range = {})
    {
        const auto beforeScope = layoutScope(before);
        const auto afterScope = layoutScope(after);
        return rachis::TextureBarrier{
            .texture = texture,
            .stagesBefore = beforeScope.stages,
            .stagesAfter = afterScope.stages,
            .accessBefore = beforeScope.access,
            .accessAfter = afterScope.access,
            .layoutBefore = before,
            .layoutAfter = after,
            .subresources = range,
        };
    }

} // namespace

TEST_CASE("Vulkan texture barriers must name the current layout of every subresource or undefined",
          "[size:small][rhi][vulkan][vuid][layouts-mock][barrier]")
{
    using rachis::TextureLayout;

    MockVulkanDispatch mock;
    gMock = &mock;
    const auto table = makeMockDeviceTable();

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetPhysicalDeviceFormatProperties;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &mockGetPhysicalDeviceImageFormatProperties;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x13);
    const auto device = fakeHandle<VkDevice>(0x23);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeHandle<VkQueue>(0x33),
        .compute = fakeHandle<VkQueue>(0x33),
        .copy = fakeHandle<VkQueue>(0x33),
        .families = families,
    };

    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, &instanceTable, physical, families);
    constexpr std::uint64_t kDeviceToken = 0xABC3;
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, kDeviceToken, &resources);
    rachis::vulkan::VulkanQueueManager queueManager;
    REQUIRE(queueManager.initialize(device, &table, queueHandles, kDeviceToken).has_value());
    const auto cmdPool =
        commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 16});
    REQUIRE(cmdPool.has_value());

    const rachis::TextureDesc desc{
        .width = 8,
        .height = 8,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
    };
    const auto texture = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x7001), desc);
    const auto other = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x7002), desc);
    const auto shared = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x7003), desc);
    REQUIRE(texture.has_value());
    REQUIRE(other.has_value());
    REQUIRE(shared.has_value());
    resources.markTextureSharedForTesting(*shared, true);

    const auto record = [&](auto& commandBuffer, const rachis::TextureBarrier& barrier) {
        auto& native = static_cast<rachis::vulkan::VulkanCommandBuffer&>(*commandBuffer);
        return resources.textureBarrier(native.native(), rachis::QueueType::graphics, barrier);
    };
    const auto accepts = [&](auto& commandBuffer, const rachis::TextureBarrier& barrier) {
        return record(commandBuffer, barrier).has_value();
    };
    const auto acquire = [&] {
        auto acquired = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(acquired.has_value());
        return std::move(*acquired);
    };

    SECTION("a barrier inside one command buffer must start where the command buffer left the subresource")
    {
        auto cb = acquire();
        REQUIRE(accepts(cb, layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::copyDestination)));

        const auto barriersBefore = mock.imageBarriers.size();
        const auto wrong =
            record(cb, layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::shaderResource));
        REQUIRE_FALSE(wrong.has_value());
        CHECK(wrong.error().code == rachis::ErrorCode::invalidState);
        CHECK(mock.imageBarriers.size() == barriersBefore);

        REQUIRE(accepts(cb, layoutBarrier(*texture, TextureLayout::copyDestination, TextureLayout::shaderResource)));
        const auto repeated =
            record(cb, layoutBarrier(*texture, TextureLayout::copyDestination, TextureLayout::shaderResource));
        REQUIRE_FALSE(repeated.has_value());
        CHECK(repeated.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(accepts(cb, layoutBarrier(*texture, TextureLayout::shaderResource, TextureLayout::copySource)));

        REQUIRE(accepts(cb, layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::unorderedAccess)));
        REQUIRE(accepts(cb, layoutBarrier(*texture, TextureLayout::unorderedAccess, TextureLayout::copySource)));

        REQUIRE(accepts(cb, layoutBarrier(*other, TextureLayout::undefined, TextureLayout::shaderResource)));
        auto& native = static_cast<rachis::vulkan::VulkanCommandBuffer&>(*cb);
        const std::array batch{
            layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::copyDestination),
            layoutBarrier(*other, TextureLayout::copySource, TextureLayout::copyDestination),
        };
        const auto batchRejected =
            resources.resourceBarriers(native.native(), rachis::QueueType::graphics, {.textureBarriers = batch});
        REQUIRE_FALSE(batchRejected.has_value());
        CHECK(batchRejected.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(accepts(cb, layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::shaderResource)));

        REQUIRE(accepts(cb,
                        layoutBarrier(*texture,
                                      TextureLayout::shaderResource,
                                      TextureLayout::copyDestination,
                                      {.firstMipLevel = 1, .mipLevelCount = 1})));
        const auto mixed =
            record(cb, layoutBarrier(*texture, TextureLayout::shaderResource, TextureLayout::copySource));
        REQUIRE_FALSE(mixed.has_value());
        CHECK(mixed.error().message.find("mip 1") != std::string::npos);
        REQUIRE(accepts(cb,
                        layoutBarrier(*texture,
                                      TextureLayout::shaderResource,
                                      TextureLayout::copySource,
                                      {.firstMipLevel = 0, .mipLevelCount = 1})));

        std::array submitted{std::move(cb)};
        REQUIRE(queueManager.submit(submitted, {}).has_value());
    }

    SECTION("a barrier's entry layout is checked against earlier submissions, with no partial effects")
    {
        {
            auto init = acquire();
            REQUIRE(accepts(init, layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::copyDestination)));
            REQUIRE(accepts(init, layoutBarrier(*other, TextureLayout::undefined, TextureLayout::shaderResource)));
            std::array batch{std::move(init)};
            REQUIRE(queueManager.submit(batch, {}).has_value());
        }

        auto wrong = acquire();
        REQUIRE(accepts(wrong, layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::shaderResource)));
        const auto submitsBefore = mock.submitCount;
        std::array wrongBatch{std::move(wrong)};
        const auto rejected = queueManager.submit(wrongBatch, {});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        CHECK(mock.submitCount == submitsBefore);
        CHECK(wrongBatch[0] != nullptr);

        auto good = acquire();
        auto bad = acquire();
        REQUIRE(accepts(good, layoutBarrier(*texture, TextureLayout::copyDestination, TextureLayout::copySource)));
        REQUIRE(accepts(bad, layoutBarrier(*other, TextureLayout::copyDestination, TextureLayout::copySource)));
        std::array mixedBatch{std::move(good), std::move(bad)};
        REQUIRE_FALSE(queueManager.submit(mixedBatch, {}).has_value());
        CHECK(mock.submitCount == submitsBefore);

        auto check = acquire();
        REQUIRE(accepts(check, layoutBarrier(*texture, TextureLayout::copyDestination, TextureLayout::shaderResource)));
        std::array checkBatch{std::move(check)};
        REQUIRE(queueManager.submit(checkBatch, {}).has_value());

        auto first = acquire();
        auto second = acquire();
        REQUIRE(accepts(first, layoutBarrier(*texture, TextureLayout::shaderResource, TextureLayout::copySource)));
        REQUIRE(accepts(second, layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::unorderedAccess)));
        std::array reversed{std::move(second), std::move(first)};
        REQUIRE_FALSE(queueManager.submit(reversed, {}).has_value());
        std::array ordered{std::move(reversed[1]), std::move(reversed[0])};
        REQUIRE(queueManager.submit(ordered, {}).has_value());

        auto discard = acquire();
        REQUIRE(accepts(discard, layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::copyDestination)));
        std::array discardBatch{std::move(discard)};
        REQUIRE(queueManager.submit(discardBatch, {}).has_value());
    }

    SECTION("a shared texture's entry layout is unknown until a submitted barrier establishes it")
    {
        auto entry = acquire();
        REQUIRE(accepts(entry, layoutBarrier(*shared, TextureLayout::copyDestination, TextureLayout::shaderResource)));
        std::array entryBatch{std::move(entry)};
        REQUIRE(queueManager.submit(entryBatch, {}).has_value());

        auto inside = acquire();
        REQUIRE(accepts(inside, layoutBarrier(*shared, TextureLayout::copySource, TextureLayout::copyDestination)));
        const auto wrong =
            record(inside, layoutBarrier(*shared, TextureLayout::shaderResource, TextureLayout::copySource));
        REQUIRE_FALSE(wrong.has_value());
        CHECK(wrong.error().code == rachis::ErrorCode::invalidState);
        std::array insideBatch{std::move(inside)};
        const auto rejected = queueManager.submit(insideBatch, {});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        CHECK(rejected.error().message.find("leaves it in shaderResource") != std::string::npos);

        auto known = acquire();
        REQUIRE(accepts(known, layoutBarrier(*shared, TextureLayout::shaderResource, TextureLayout::copySource)));
        std::array knownBatch{std::move(known)};
        REQUIRE(queueManager.submit(knownBatch, {}).has_value());
    }

    SECTION("a shared texture keeps the entry layout its first submitted barrier established")
    {
        resources.markTextureSharedForTesting(*other, false);
        auto entry = acquire();
        REQUIRE(accepts(entry, layoutBarrier(*other, TextureLayout::undefined, TextureLayout::common)));
        std::array entryBatch{std::move(entry)};
        REQUIRE(queueManager.submit(entryBatch, {}).has_value());

        auto wrongEntry = acquire();
        REQUIRE(accepts(wrongEntry, layoutBarrier(*other, TextureLayout::shaderResource, TextureLayout::copySource)));
        std::array wrongBatch{std::move(wrongEntry)};
        const auto rejected = queueManager.submit(wrongBatch, {});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        CHECK(rejected.error().message.find("leaves it in common") != std::string::npos);
    }

    SECTION("portable layouts that share a Vulkan layout are still told apart")
    {
        auto inside = acquire();
        REQUIRE(accepts(inside, layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::unorderedAccess)));
        const auto barriersBefore = mock.imageBarriers.size();
        const auto fromCommon =
            record(inside, layoutBarrier(*texture, TextureLayout::common, TextureLayout::shaderResource));
        REQUIRE_FALSE(fromCommon.has_value());
        CHECK(fromCommon.error().code == rachis::ErrorCode::invalidState);
        CHECK(fromCommon.error().message.find("layoutBefore (common)") != std::string::npos);
        CHECK(fromCommon.error().message.find("leaves it in unorderedAccess") != std::string::npos);
        CHECK(mock.imageBarriers.size() == barriersBefore);
        REQUIRE(accepts(inside, layoutBarrier(*texture, TextureLayout::unorderedAccess, TextureLayout::copySource)));
        REQUIRE(accepts(inside, layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::resolveSource)));
        const auto fromCopySource =
            record(inside, layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::shaderResource));
        REQUIRE_FALSE(fromCopySource.has_value());
        CHECK(fromCopySource.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(accepts(inside, layoutBarrier(*texture, TextureLayout::resolveSource, TextureLayout::unorderedAccess)));
        std::array insideBatch{std::move(inside)};
        REQUIRE(queueManager.submit(insideBatch, {}).has_value());

        auto wrongEntry = acquire();
        REQUIRE(accepts(wrongEntry, layoutBarrier(*texture, TextureLayout::common, TextureLayout::copySource)));
        const auto submitsBefore = mock.submitCount;
        std::array wrongBatch{std::move(wrongEntry)};
        const auto rejected = queueManager.submit(wrongBatch, {});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        CHECK(rejected.error().message.find("leaves it in unorderedAccess") != std::string::npos);
        CHECK(mock.submitCount == submitsBefore);

        auto toResolve = acquire();
        REQUIRE(
            accepts(toResolve, layoutBarrier(*texture, TextureLayout::unorderedAccess, TextureLayout::resolveSource)));
        std::array toResolveBatch{std::move(toResolve)};
        REQUIRE(queueManager.submit(toResolveBatch, {}).has_value());

        auto fromCopy = acquire();
        REQUIRE(accepts(fromCopy, layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::shaderResource)));
        std::array fromCopyBatch{std::move(fromCopy)};
        REQUIRE_FALSE(queueManager.submit(fromCopyBatch, {}).has_value());

        auto exact = acquire();
        REQUIRE(accepts(exact, layoutBarrier(*texture, TextureLayout::resolveSource, TextureLayout::shaderResource)));
        std::array exactBatch{std::move(exact)};
        REQUIRE(queueManager.submit(exactBatch, {}).has_value());
    }

    queueManager.shutdown();
    commands.shutdown();
    resources.shutdown();
    gMock = nullptr;
}

namespace
{

    struct MockStorageClearFixture final
    {
        MockVulkanDispatch mock;
        VolkDeviceTable table = makeMockDeviceTable();
        rachis::vulkan::VulkanInstanceTable instanceTable{};
        rachis::vulkan::VulkanResourceAllocator resources;
        rachis::vulkan::VulkanCommandAllocator commands;
        rachis::vulkan::VulkanQueueManager queueManager;
        rachis::CommandPoolHandle pool{};
        rachis::TextureHandle texture{};
        rachis::TextureHandle other{};
        rachis::TextureHandle copySource{};
        rachis::TextureViewHandle view{};

        MockStorageClearFixture()
        {
            gMock = &mock;
            instanceTable.vkGetPhysicalDeviceFormatProperties = &mockGetPhysicalDeviceFormatProperties;
            instanceTable.vkGetPhysicalDeviceImageFormatProperties = &mockGetPhysicalDeviceImageFormatProperties;
            const auto physical = fakeHandle<VkPhysicalDevice>(0x13);
            const auto device = fakeHandle<VkDevice>(0x24);
            const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
            const rachis::vulkan::VulkanQueueHandles queueHandles{
                .graphics = fakeHandle<VkQueue>(0x34),
                .compute = fakeHandle<VkQueue>(0x34),
                .copy = fakeHandle<VkQueue>(0x34),
                .families = families,
            };
            resources.initializeDispatchForTesting(device, &table, &instanceTable, physical, families);
            constexpr std::uint64_t kDeviceToken = 0xABC4;
            commands.initialize(device, &table, queueHandles, kDeviceToken, &resources);
            REQUIRE(queueManager.initialize(device, &table, queueHandles, kDeviceToken).has_value());
            auto created =
                commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 1024});
            REQUIRE(created.has_value());
            pool = *created;
            auto adopted = resources.adoptSwapchainImage(
                fakeHandle<VkImage>(0x7101),
                {
                    .width = 8,
                    .height = 8,
                    .depthOrArrayLayers = 2,
                    .format = rachis::Format::rgba8Unorm,
                    .usage = rachis::TextureUsage::copyDestination | rachis::TextureUsage::storage |
                             rachis::TextureUsage::sampled | rachis::TextureUsage::renderTarget,
                });
            REQUIRE(adopted.has_value());
            texture = *adopted;
            auto unrelated = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x7102),
                                                           {.width = 4,
                                                            .height = 4,
                                                            .format = rachis::Format::rgba8Unorm,
                                                            .usage = rachis::TextureUsage::storage});
            REQUIRE(unrelated.has_value());
            other = *unrelated;
            auto source = resources.adoptSwapchainImage(fakeHandle<VkImage>(0x7103),
                                                        {.width = 8,
                                                         .height = 8,
                                                         .format = rachis::Format::rgba8Unorm,
                                                         .usage = rachis::TextureUsage::copySource});
            REQUIRE(source.has_value());
            copySource = *source;
            auto storage = resources.createTextureView({
                .texture = texture,
                .usage = rachis::TextureViewUsage::storage,
                .dimension = rachis::TextureViewDimension::texture2DArray,
            });
            REQUIRE(storage.has_value());
            view = *storage;
        }

        MockStorageClearFixture(const MockStorageClearFixture&) = delete;
        MockStorageClearFixture& operator=(const MockStorageClearFixture&) = delete;
        MockStorageClearFixture(MockStorageClearFixture&&) = delete;
        MockStorageClearFixture& operator=(MockStorageClearFixture&&) = delete;

        ~MockStorageClearFixture()
        {
            queueManager.shutdown();
            commands.shutdown();
            resources.shutdown();
            gMock = nullptr;
        }

        [[nodiscard]] auto acquire()
        {
            auto acquired = commands.acquireCommandBuffer(pool);
            REQUIRE(acquired.has_value());
            return std::move(*acquired);
        }

        [[nodiscard]] static VkCommandBuffer native(const auto& commandBuffer) noexcept
        {
            return static_cast<rachis::vulkan::VulkanCommandBuffer&>(*commandBuffer).native();
        }

        [[nodiscard]] bool transition(const auto& commandBuffer,
                                      rachis::TextureLayout before,
                                      rachis::TextureLayout after,
                                      rachis::TextureHandle target = {})
        {
            return resources
                .textureBarrier(native(commandBuffer),
                                rachis::QueueType::graphics,
                                layoutBarrier(target.isValid() ? target : texture, before, after))
                .has_value();
        }

        [[nodiscard]] rachis::Status clear(const auto& commandBuffer)
        {
            return resources.clearStorageTexture(native(commandBuffer),
                                                 view,
                                                 rachis::StorageTextureClearValue{
                                                     .type = rachis::StorageTextureClearType::float32,
                                                     .float32 = {1.0F, 0.5F, 0.25F, 1.0F},
                                                 },
                                                 nullptr);
        }

        [[nodiscard]] rachis::Status copyIntoLayerZero(const auto& commandBuffer)
        {
            const std::array regions{
                rachis::TextureCopyRegion{
                    .extent = {.width = 8, .height = 8, .depth = 1},
                },
            };
            return resources.copyTextureRegions(native(commandBuffer), copySource, texture, regions);
        }

        template<typename CommandBuffer>
        [[nodiscard]] bool submit(CommandBuffer commandBuffer)
        {
            std::array batch{std::move(commandBuffer)};
            return queueManager.submit(batch, {}).has_value();
        }
    };

} // namespace

TEST_CASE("Vulkan storage clears that fail to allocate leave no layout requirements in the command buffer",
          "[size:small][rhi][vulkan][vuid][layouts-mock][fill-and-clear]")
{
    using rachis::TextureLayout;
    MockStorageClearFixture fixture;
    {
        auto initial = fixture.acquire();
        REQUIRE(fixture.transition(initial, TextureLayout::undefined, TextureLayout::unorderedAccess));
        REQUIRE(fixture.submit(std::move(initial)));
    }

    bool recorded = false;
    std::uint32_t injectedFailures = 0;
    for (std::uint32_t failAt = 1; failAt < 256 && !recorded; ++failAt)
    {
        auto failed = fixture.acquire();
        auto followed = fixture.acquire();
        REQUIRE(fixture.transition(failed, TextureLayout::undefined, TextureLayout::unorderedAccess, fixture.other));
        const auto clearsBefore = fixture.mock.clearColorImages;
        gHostAllocationsUntilFailure = failAt;
        bool threw = false;
        try
        {
            static_cast<void>(fixture.clear(failed));
        }
        catch (const std::bad_alloc&)
        {
            threw = true;
        }
        gHostAllocationsUntilFailure = 0;
        const bool injected = std::exchange(gHostAllocationFailureInjected, false);
        if (!injected)
        {
            CHECK_FALSE(threw);
            CHECK(fixture.mock.clearColorImages == clearsBefore + 1);
            REQUIRE(fixture.submit(std::move(failed)));
            recorded = true;
            continue;
        }
        REQUIRE(threw);
        ++injectedFailures;
        CHECK(fixture.mock.clearColorImages == clearsBefore);

        REQUIRE(fixture.transition(failed, TextureLayout::undefined, TextureLayout::unorderedAccess));
        REQUIRE(fixture.transition(followed, TextureLayout::unorderedAccess, TextureLayout::shaderResource));
        REQUIRE(fixture.submit(std::move(followed)));
        CHECK(fixture.submit(std::move(failed)));
    }
    CHECK(recorded);
    CHECK(injectedFailures > 1);
}

TEST_CASE("Vulkan storage clears and copies in one command buffer agree on the subresource's native layout",
          "[size:small][rhi][vulkan][vuid][layouts-mock][fill-and-clear]")
{
    using rachis::TextureLayout;
    MockStorageClearFixture fixture;
    const auto image = fakeHandle<VkImage>(0x7101);
    {
        auto initial = fixture.acquire();
        REQUIRE(fixture.transition(initial, TextureLayout::undefined, TextureLayout::copyDestination));
        REQUIRE(fixture.transition(initial, TextureLayout::undefined, TextureLayout::copySource, fixture.copySource));
        REQUIRE(fixture.submit(std::move(initial)));
    }

    SECTION("a clear after a copy moves the copied subresource to the clear's layout inside the command buffer")
    {
        auto copied = fixture.acquire();
        const auto native = MockStorageClearFixture::native(copied);
        REQUIRE(fixture.copyIntoLayerZero(copied).has_value());
        REQUIRE(fixture.mock.copyImageDestinationLayouts.size() == 1);
        CHECK(fixture.mock.copyImageDestinationLayouts[0].layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const auto barriersBefore = fixture.mock.imageBarriers.size();
        REQUIRE(fixture.clear(copied).has_value());
        REQUIRE(fixture.mock.imageBarriers.size() == barriersBefore + 1);
        const auto& moved = fixture.mock.imageBarriers.back();
        CHECK(moved.command == native);
        CHECK(moved.image == image);
        CHECK(moved.baseArrayLayer == 0);
        CHECK(moved.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        CHECK(moved.newLayout == VK_IMAGE_LAYOUT_GENERAL);
        REQUIRE(fixture.mock.clearColorLayouts.size() == 1);
        CHECK(fixture.mock.clearColorLayouts[0].layout == VK_IMAGE_LAYOUT_GENERAL);
        REQUIRE_FALSE(fixture.mock.timelineEvents.empty());
        CHECK(fixture.mock.timelineEvents.back().kind == CommandOpKind::clearColorImage);
        CHECK(fixture.mock.timelineEvents[fixture.mock.timelineEvents.size() - 2].kind == CommandOpKind::imageBarrier);

        auto toStorage = fixture.acquire();
        REQUIRE(fixture.transition(toStorage, TextureLayout::copyDestination, TextureLayout::unorderedAccess));
        fixture.mock.imageBarriers.clear();
        std::array ordered{std::move(toStorage), std::move(copied)};
        REQUIRE(fixture.queueManager.submit(ordered, {}).has_value());
        REQUIRE(fixture.mock.lastSubmittedCommands.size() == 3);
        CHECK(fixture.mock.lastSubmittedCommands[2] == native);
        const auto bridge = fixture.mock.lastSubmittedCommands[1];
        const auto bridged = std::ranges::count_if(
            fixture.mock.imageBarriers, [&](const RecordedImageBarrier& barrier) { return barrier.command == bridge; });
        CHECK(bridged == 1);
        const auto preBridge = std::ranges::find_if(
            fixture.mock.imageBarriers, [&](const RecordedImageBarrier& barrier) { return barrier.command == bridge; });
        REQUIRE(preBridge != fixture.mock.imageBarriers.end());
        CHECK(preBridge->baseArrayLayer == 0);
        CHECK(preBridge->oldLayout == VK_IMAGE_LAYOUT_GENERAL);
        CHECK(preBridge->newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        auto after = fixture.acquire();
        REQUIRE(fixture.transition(after, TextureLayout::unorderedAccess, TextureLayout::shaderResource));
        REQUIRE(fixture.submit(std::move(after)));
        CHECK(fixture.mock.lastSubmittedCommands.size() == 1);
    }

    SECTION("a copy after a clear uses the layout the clear established for the command buffer")
    {
        auto cleared = fixture.acquire();
        REQUIRE(fixture.clear(cleared).has_value());
        REQUIRE(fixture.copyIntoLayerZero(cleared).has_value());
        REQUIRE(fixture.mock.copyImageDestinationLayouts.size() == 1);
        CHECK(fixture.mock.copyImageDestinationLayouts[0].layout == VK_IMAGE_LAYOUT_GENERAL);

        auto toStorage = fixture.acquire();
        REQUIRE(fixture.transition(toStorage, TextureLayout::copyDestination, TextureLayout::unorderedAccess));
        std::array ordered{std::move(toStorage), std::move(cleared)};
        REQUIRE(fixture.queueManager.submit(ordered, {}).has_value());
        CHECK(fixture.mock.lastSubmittedCommands.size() == 2);
    }
}

TEST_CASE("Vulkan storage clears check the entry layout of shared textures whose layout is known",
          "[size:small][rhi][vulkan][vuid][layouts-mock][fill-and-clear]")
{
    using rachis::TextureLayout;
    MockStorageClearFixture fixture;
    {
        auto initial = fixture.acquire();
        REQUIRE(fixture.transition(initial, TextureLayout::undefined, TextureLayout::common));
        REQUIRE(fixture.submit(std::move(initial)));
    }

    SECTION("a shared texture that a submitted barrier left in common")
    {
        fixture.resources.markTextureSharedForTesting(fixture.texture, false);
        auto cleared = fixture.acquire();
        REQUIRE(fixture.clear(cleared).has_value());
        std::array batch{std::move(cleared)};
        const auto rejected = fixture.queueManager.submit(batch, {});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        CHECK(rejected.error().message.find("leaves it in common") != std::string::npos);
    }

    SECTION("a shared texture that no submitted barrier has transitioned yet")
    {
        fixture.resources.markTextureSharedForTesting(fixture.texture, true);
        auto cleared = fixture.acquire();
        REQUIRE(fixture.clear(cleared).has_value());
        REQUIRE(fixture.submit(std::move(cleared)));

        auto established = fixture.acquire();
        REQUIRE(fixture.transition(established, TextureLayout::unorderedAccess, TextureLayout::shaderResource));
        REQUIRE(fixture.submit(std::move(established)));
        auto wrong = fixture.acquire();
        REQUIRE(fixture.clear(wrong).has_value());
        CHECK_FALSE(fixture.submit(std::move(wrong)));
    }
}

TEST_CASE("Vulkan storage clears leave a first use to submission, where earlier command buffers of the batch count",
          "[size:small][rhi][vulkan][vuid][layouts-mock][fill-and-clear]")
{
    using rachis::TextureLayout;
    MockStorageClearFixture fixture;
    {
        auto initial = fixture.acquire();
        REQUIRE(fixture.transition(initial, TextureLayout::undefined, TextureLayout::renderTarget));
        REQUIRE(fixture.submit(std::move(initial)));
    }

    auto transition = fixture.acquire();
    auto cleared = fixture.acquire();
    REQUIRE(fixture.transition(transition, TextureLayout::renderTarget, TextureLayout::unorderedAccess));
    const auto recorded = fixture.clear(cleared);
    REQUIRE(recorded.has_value());
    std::array ordered{std::move(transition), std::move(cleared)};
    REQUIRE(fixture.queueManager.submit(ordered, {}).has_value());
    CHECK(fixture.mock.clearColorImages == 1);

    auto toRenderTarget = fixture.acquire();
    REQUIRE(fixture.transition(toRenderTarget, TextureLayout::unorderedAccess, TextureLayout::renderTarget));
    REQUIRE(fixture.submit(std::move(toRenderTarget)));
    auto alone = fixture.acquire();
    REQUIRE(fixture.clear(alone).has_value());
    std::array unordered{std::move(alone)};
    const auto rejected = fixture.queueManager.submit(unordered, {});
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::invalidState);

    auto inside = fixture.acquire();
    REQUIRE(fixture.transition(inside, TextureLayout::renderTarget, TextureLayout::shaderResource));
    const auto conflicting = fixture.clear(inside);
    REQUIRE_FALSE(conflicting.has_value());
    CHECK(conflicting.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(fixture.transition(inside, TextureLayout::shaderResource, TextureLayout::unorderedAccess));
    REQUIRE(fixture.clear(inside).has_value());
    CHECK(fixture.submit(std::move(inside)));
    CHECK(fixture.mock.clearColorImages == 3);
}

namespace
{

    VKAPI_ATTR VkResult VKAPI_CALL mockCreateAccelerationStructure(VkDevice,
                                                                   const VkAccelerationStructureCreateInfoKHR*,
                                                                   const VkAllocationCallbacks*,
                                                                   VkAccelerationStructureKHR* structure)
    {
        *structure = fakeHandle<VkAccelerationStructureKHR>(gMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL mockGetZeroScratchBuildSizes(VkDevice,
                                                            VkAccelerationStructureBuildTypeKHR,
                                                            const VkAccelerationStructureBuildGeometryInfoKHR*,
                                                            const std::uint32_t*,
                                                            VkAccelerationStructureBuildSizesInfoKHR* sizes)
    {
        sizes->accelerationStructureSize = 4096;
        sizes->buildScratchSize = 0;
        sizes->updateScratchSize = 0;
    }

    VKAPI_ATTR void VKAPI_CALL
    mockCmdBuildAccelerationStructures(VkCommandBuffer,
                                       std::uint32_t,
                                       const VkAccelerationStructureBuildGeometryInfoKHR*,
                                       const VkAccelerationStructureBuildRangeInfoKHR* const*)
    {
    }

} // namespace

TEST_CASE("Vulkan acceleration-structure size queries accept a driver that needs no build scratch",
          "[size:small][rhi][vulkan][vuid][raytracing-mock]")
{
    MockVulkanDispatch mock;
    gMock = &mock;
    auto table = makeMockDeviceTable();
    table.vkCreateAccelerationStructureKHR = &mockCreateAccelerationStructure;
    table.vkGetAccelerationStructureBuildSizesKHR = &mockGetZeroScratchBuildSizes;
    table.vkCmdBuildAccelerationStructuresKHR = &mockCmdBuildAccelerationStructures;
    const auto physical = fakeHandle<VkPhysicalDevice>(0x16);
    const auto device = fakeHandle<VkDevice>(0x28);
    rachis::vulkan::VulkanResourceAllocator resources;
    resources.initializeDispatchForTesting(device, &table, nullptr, physical, {});
    rachis::DeviceCapabilities capabilities{};
    capabilities.rayQuery = true;
    rachis::vulkan::VulkanAccelerationManager acceleration;
    acceleration.initialize(device, &table, &resources, &capabilities, 256);

    const auto sizes = acceleration.queryTopLevelBuildSizes({
        .instanceCount = 1,
        .flags = rachis::AccelerationStructureBuildFlag::allowUpdate,
    });
    REQUIRE(sizes.has_value());
    CHECK(sizes->accelerationStructureSize == 4096);
    CHECK(sizes->buildScratchSize == 0);
    CHECK(sizes->updateScratchSize == 0);

    acceleration.shutdown();
    resources.shutdown();
    gMock = nullptr;
}
#endif
