#pragma once

#include "Rachis/Error.h"
#include "Rachis/Pipeline.h"
#include "Rachis/Rendering.h"
#include "Rachis/Resource.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace rachis::vulkan::validation_detail
{

    inline constexpr std::uint32_t pushConstantWordBytes = 4;
    inline constexpr std::size_t pushConstantWordCount = maxPushConstantBytes / pushConstantWordBytes;

    struct PushConstantSubrange final
    {
        ShaderStage stages = ShaderStage::none;
        std::uint32_t offset = 0;
        std::uint32_t size = 0;

        [[nodiscard]] constexpr bool operator==(const PushConstantSubrange&) const noexcept = default;
    };

    [[nodiscard]] constexpr bool isWordAlignedPushConstantRange(const PushConstantRange& range) noexcept
    {
        return range.size != 0 && (range.offset % pushConstantWordBytes) == 0 &&
               (range.size % pushConstantWordBytes) == 0 && range.offset <= maxPushConstantBytes &&
               range.size <= maxPushConstantBytes - range.offset;
    }

    [[nodiscard]] inline std::vector<PushConstantRange> mergePushConstantRangesPerStage(
        std::span<const PushConstantRange> layoutRanges)
    {
        using StageBits = std::underlying_type_t<ShaderStage>;
        std::vector<PushConstantRange> merged;
        for (int bit = 0; bit < std::numeric_limits<StageBits>::digits; ++bit)
        {
            const auto stage = static_cast<ShaderStage>(static_cast<StageBits>(1U << bit));
            if (!hasAny(stage, ShaderStage::all))
            {
                continue;
            }
            std::uint32_t begin = maxPushConstantBytes;
            std::uint32_t end = 0;
            for (const auto& range : layoutRanges)
            {
                if (isWordAlignedPushConstantRange(range) && hasAny(range.visibility, stage))
                {
                    begin = (std::min)(begin, range.offset);
                    end = (std::max)(end, range.offset + range.size);
                }
            }
            if (begin >= end)
            {
                continue;
            }
            const auto sameExtent = [&](const PushConstantRange& candidate) {
                return candidate.offset == begin && candidate.size == end - begin;
            };
            if (const auto found = std::ranges::find_if(merged, sameExtent); found != merged.end())
            {
                found->visibility = found->visibility | stage;
            }
            else
            {
                merged.push_back(PushConstantRange{.visibility = stage, .offset = begin, .size = end - begin});
            }
        }
        return merged;
    }

    [[nodiscard]] constexpr std::array<ShaderStage, pushConstantWordCount> computeWordStageUnion(
        std::span<const PushConstantRange> layoutRanges) noexcept
    {
        std::array<ShaderStage, pushConstantWordCount> words{};
        for (const auto& range : layoutRanges)
        {
            if (!isWordAlignedPushConstantRange(range))
            {
                continue;
            }
            const auto firstWord = static_cast<std::size_t>(range.offset / pushConstantWordBytes);
            const auto endWord = static_cast<std::size_t>((range.offset + range.size) / pushConstantWordBytes);
            for (std::size_t word = firstWord; word < endWord; ++word)
            {
                words[word] = words[word] | range.visibility;
            }
        }
        return words;
    }

    [[nodiscard]] inline Status validatePushConstantSlice(std::uint32_t offset, std::size_t byteSize)
    {
        if (byteSize == 0 || byteSize > maxPushConstantBytes)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Push constant updates must be non-empty and within maxPushConstantBytes.",
            });
        }
        const auto size = static_cast<std::uint32_t>(byteSize);
        if ((offset % pushConstantWordBytes) != 0 || (size % pushConstantWordBytes) != 0 ||
            offset > maxPushConstantBytes - size)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Push constant offset and size must be 4-byte aligned and within maxPushConstantBytes.",
            });
        }
        return {};
    }

    [[nodiscard]] inline std::vector<PushConstantSubrange> emitContiguousPushConstantSubranges(
        const std::array<ShaderStage, pushConstantWordCount>& words, std::size_t firstWord, std::size_t endWord)
    {
        std::vector<PushConstantSubrange> subranges;
        for (std::size_t runStart = firstWord; runStart < endWord;)
        {
            const auto stages = words[runStart];
            std::size_t runEnd = runStart + 1;
            while (runEnd < endWord && words[runEnd] == stages)
            {
                ++runEnd;
            }
            subranges.push_back(PushConstantSubrange{
                .stages = stages,
                .offset = static_cast<std::uint32_t>(runStart * pushConstantWordBytes),
                .size = static_cast<std::uint32_t>((runEnd - runStart) * pushConstantWordBytes),
            });
            runStart = runEnd;
        }
        return subranges;
    }

    [[nodiscard]] inline Result<std::vector<PushConstantSubrange>> planGraphicsPushConstants(
        std::span<const PushConstantRange> layoutRanges,
        std::span<const PushConstantRange> nativeRanges,
        ShaderStage visibility,
        std::uint32_t offset,
        std::size_t byteSize)
    {
        if (visibility == ShaderStage::none || (visibility & ShaderStage::allGraphics) != visibility)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Graphics push constants require one or more graphics shader stages.",
            });
        }
        if (auto valid = validatePushConstantSlice(offset, byteSize); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        const auto words = computeWordStageUnion(layoutRanges);
        const auto firstWord = static_cast<std::size_t>(offset / pushConstantWordBytes);
        const auto endWord =
            static_cast<std::size_t>((offset + static_cast<std::uint32_t>(byteSize)) / pushConstantWordBytes);
        for (std::size_t word = firstWord; word < endWord; ++word)
        {
            if ((words[word] & visibility) != visibility)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Push constants are outside the bound graphics pipeline push constant ranges.",
                });
            }
        }
        return emitContiguousPushConstantSubranges(computeWordStageUnion(nativeRanges), firstWord, endWord);
    }

    [[nodiscard]] inline Result<std::vector<PushConstantSubrange>> planComputePushConstants(
        std::span<const PushConstantRange> layoutRanges,
        std::span<const PushConstantRange> nativeRanges,
        std::uint32_t offset,
        std::size_t byteSize)
    {
        if (auto valid = validatePushConstantSlice(offset, byteSize); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        const auto words = computeWordStageUnion(layoutRanges);
        const auto firstWord = static_cast<std::size_t>(offset / pushConstantWordBytes);
        const auto endWord =
            static_cast<std::size_t>((offset + static_cast<std::uint32_t>(byteSize)) / pushConstantWordBytes);
        for (std::size_t word = firstWord; word < endWord; ++word)
        {
            if (!hasAny(words[word], ShaderStage::compute))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Compute push constants are outside the bound compute pipeline push constant range.",
                });
            }
        }
        return emitContiguousPushConstantSubranges(computeWordStageUnion(nativeRanges), firstWord, endWord);
    }

    [[nodiscard]] inline Result<std::vector<PushConstantSubrange>> planRayTracingPushConstants(
        std::span<const PushConstantRange> layoutRanges,
        std::span<const PushConstantRange> nativeRanges,
        std::uint32_t offset,
        std::size_t byteSize)
    {
        if (auto valid = validatePushConstantSlice(offset, byteSize); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        const auto words = computeWordStageUnion(layoutRanges);
        const auto firstWord = static_cast<std::size_t>(offset / pushConstantWordBytes);
        const auto endWord =
            static_cast<std::size_t>((offset + static_cast<std::uint32_t>(byteSize)) / pushConstantWordBytes);
        for (std::size_t word = firstWord; word < endWord; ++word)
        {
            if (!hasAny(words[word], ShaderStage::allRayTracing))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Ray tracing push constants are outside the bound ray tracing pipeline push constant range.",
                });
            }
        }
        return emitContiguousPushConstantSubranges(computeWordStageUnion(nativeRanges), firstWord, endWord);
    }

    [[nodiscard]] inline Status validateShaderPushConstantCoverage(
        std::span<const ShaderPushConstantRequirement> shaderRanges,
        std::span<const PushConstantRange> layoutRanges,
        ShaderStage stage)
    {
        std::array<bool, pushConstantWordCount> requiredWords{};
        std::array<bool, pushConstantWordCount> layoutWords{};
        const auto markWords =
            [](std::array<bool, pushConstantWordCount>& words, std::uint32_t offset, std::uint32_t size) {
                std::ranges::fill(
                    std::span(words).subspan(offset / pushConstantWordBytes, size / pushConstantWordBytes), true);
            };
        for (const auto& range : shaderRanges)
        {
            markWords(requiredWords, range.offset, range.size);
        }
        for (const auto& range : layoutRanges)
        {
            if (hasAny(range.visibility, stage))
            {
                markWords(layoutWords, range.offset, range.size);
            }
        }
        if (requiredWords != layoutWords)
        {
            return std::unexpected(Error{
                ErrorCode::validationError,
                0,
                "The pipeline layout's stage-visible push constants do not exactly match the shader interface.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateDepthBiasClampSupport(bool depthBiasClampSupported, float clamp)
    {
        if (!depthBiasClampSupported && clamp != 0.0F)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Non-zero depthBiasClamp requires the Vulkan depthBiasClamp device feature.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateShadingRateImageSplitQueryState(bool hasActiveOcclusion,
                                                                        bool hasActivePipelineStatistics,
                                                                        bool pipelineStatisticsBegunInRendering)
    {
        if (hasActiveOcclusion || (hasActivePipelineStatistics && pipelineStatisticsBegunInRendering))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "setShadingRateImage cannot split dynamic rendering while a render-pass query is active.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateBufferTextureCopyQueueFamily(bool queueFamilySupportsGraphics,
                                                                     std::span<const BufferTextureCopyRegion> regions)
    {
        const auto depthOrStencil = std::ranges::any_of(regions, [](const BufferTextureCopyRegion& region) {
            return region.textureSubresource.aspect == TextureAspect::depth ||
                   region.textureSubresource.aspect == TextureAspect::stencil;
        });
        if (!queueFamilySupportsGraphics && depthOrStencil)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Depth and stencil buffer-texture copies require a Vulkan queue family with graphics support; record "
                "them on the graphics queue.",
            });
        }
        return {};
    }

    [[nodiscard]] constexpr StoreOperation effectiveRenderingStoreOperation(StoreOperation requested,
                                                                            bool variableRateShadingTier2) noexcept
    {
        if (variableRateShadingTier2 && requested == StoreOperation::discard)
        {
            return StoreOperation::store;
        }
        return requested;
    }

    enum class ColorClearNumericType : std::uint8_t
    {
        floatingPoint,
        unsignedInteger,
        signedInteger,
    };

    struct ColorClearEncoding final
    {
        ColorClearNumericType numericType = ColorClearNumericType::floatingPoint;
        std::uint32_t channelBits = 32;

        [[nodiscard]] constexpr bool operator==(const ColorClearEncoding&) const noexcept = default;
    };

    [[nodiscard]] constexpr ColorClearEncoding colorClearEncoding(Format format) noexcept
    {
        switch (format)
        {
        case Format::r8Uint:
        case Format::rg8Uint:
        case Format::rgba8Uint:
            return {ColorClearNumericType::unsignedInteger, 8};
        case Format::r16Uint:
        case Format::rg16Uint:
        case Format::rgba16Uint:
            return {ColorClearNumericType::unsignedInteger, 16};
        case Format::r32Uint:
        case Format::rg32Uint:
        case Format::rgba32Uint:
        case Format::r64Uint:
            return {ColorClearNumericType::unsignedInteger, 32};
        case Format::r8Sint:
        case Format::rg8Sint:
        case Format::rgba8Sint:
            return {ColorClearNumericType::signedInteger, 8};
        case Format::r16Sint:
        case Format::rg16Sint:
        case Format::rgba16Sint:
            return {ColorClearNumericType::signedInteger, 16};
        case Format::r32Sint:
        case Format::rg32Sint:
        case Format::rgba32Sint:
            return {ColorClearNumericType::signedInteger, 32};
        default:
            return {};
        }
    }

    [[nodiscard]] constexpr std::uint32_t unsignedIntegerClearChannel(float value, std::uint32_t channelBits) noexcept
    {
        const auto maximum = static_cast<std::uint32_t>((std::uint64_t{1} << channelBits) - 1U);
        const auto input = static_cast<double>(value);
        if (!(input > 0.0))
        {
            return 0;
        }
        if (input >= static_cast<double>(maximum))
        {
            return maximum;
        }
        return static_cast<std::uint32_t>(input);
    }

    [[nodiscard]] constexpr std::int32_t signedIntegerClearChannel(float value, std::uint32_t channelBits) noexcept
    {
        const auto maximum = static_cast<std::int32_t>((std::int64_t{1} << (channelBits - 1U)) - 1);
        const auto minimum = -maximum - 1;
        const auto input = static_cast<double>(value);
        if (input >= static_cast<double>(maximum))
        {
            return maximum;
        }
        if (input <= static_cast<double>(minimum))
        {
            return minimum;
        }
        if (input > static_cast<double>(minimum))
        {
            return static_cast<std::int32_t>(input);
        }
        return 0;
    }

    enum class SubresourceTransferUse : std::uint8_t
    {
        copySource,
        copyDestination,
        resolveSource,
        resolveDestination,
    };

    enum class CanonicalTransferLayout : std::uint8_t
    {
        undefined,
        general,
        transferSrc,
        transferDst,
        other,
    };

    [[nodiscard]] constexpr bool isTransferUseCompatible(CanonicalTransferLayout layout,
                                                         SubresourceTransferUse use) noexcept
    {
        switch (use)
        {
        case SubresourceTransferUse::copySource:
        case SubresourceTransferUse::resolveSource:
            return layout == CanonicalTransferLayout::general || layout == CanonicalTransferLayout::transferSrc;
        case SubresourceTransferUse::copyDestination:
        case SubresourceTransferUse::resolveDestination:
            return layout == CanonicalTransferLayout::general || layout == CanonicalTransferLayout::transferDst;
        }
        return false;
    }

    struct SubresourceLayoutTransitionStep final
    {
        std::int32_t oldLayout = 0;
        std::int32_t newLayout = 0;

        [[nodiscard]] constexpr bool operator==(const SubresourceLayoutTransitionStep&) const noexcept = default;
    };

    struct SubresourceLayoutBridgeDecision final
    {
        bool requiresPreBridge = false;
        bool requiresPostBridge = false;
        SubresourceLayoutTransitionStep preBridge{};
        SubresourceLayoutTransitionStep postBridge{};
    };

    [[nodiscard]] inline Result<SubresourceLayoutBridgeDecision> evaluateSubresourceLayoutRequirement(
        std::int32_t actualLayout,
        CanonicalTransferLayout actualCanonical,
        std::int32_t expectedLayout,
        CanonicalTransferLayout expectedCanonical,
        SubresourceTransferUse use,
        bool hasSubsequentBarrier,
        std::int32_t firstBarrierOldLayout,
        std::int32_t undefinedLayoutValue)
    {
        if (actualLayout == expectedLayout)
        {
            return SubresourceLayoutBridgeDecision{};
        }
        if (!isTransferUseCompatible(actualCanonical, use) || !isTransferUseCompatible(expectedCanonical, use))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A submitted command buffer expects a transfer or resolve subresource layout incompatible with the "
                "current image layout.",
            });
        }
        if (hasSubsequentBarrier && firstBarrierOldLayout != undefinedLayoutValue &&
            firstBarrierOldLayout != expectedLayout)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A submitted command buffer transitions a subresource from an oldLayout incompatible with its recorded "
                "transfer layout.",
            });
        }

        SubresourceLayoutBridgeDecision decision{
            .requiresPreBridge = true,
            .requiresPostBridge = !hasSubsequentBarrier,
            .preBridge =
                SubresourceLayoutTransitionStep{
                    .oldLayout = actualLayout,
                    .newLayout = expectedLayout,
                },
            .postBridge = {},
        };
        if (!hasSubsequentBarrier)
        {
            decision.postBridge = SubresourceLayoutTransitionStep{
                .oldLayout = expectedLayout,
                .newLayout = actualLayout,
            };
        }
        return decision;
    }

} // namespace rachis::vulkan::validation_detail
