#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/VulkanResources.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint32_t textureWidth = 4;
    constexpr std::uint32_t textureHeight = 4;

    void captureErrors(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        if (message.severity == rachis::DiagnosticSeverity::error ||
            message.severity == rachis::DiagnosticSeverity::corruption)
        {
            static_cast<std::atomic_uint64_t*>(userData)->fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] rachis::DeviceDesc vulkanDesc(std::atomic_uint64_t* errors = nullptr) noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
            .diagnosticCallback = errors != nullptr ? &captureErrors : nullptr,
            .diagnosticUserData = errors,
        };
    }

    template<typename Create>
    [[nodiscard]] auto createWithBadAlloc(Create&& create)
    {
        rachis::vulkan::injectVulkanBadAlloc(rachis::vulkan::VulkanBadAllocPoint::nativeKeepAlive, true);
        auto created = create();
        rachis::vulkan::injectVulkanBadAlloc(rachis::vulkan::VulkanBadAllocPoint::nativeKeepAlive, false);
        return created;
    }

    struct AllocationTotals final
    {
        std::uint64_t bytes = 0;
        std::uint64_t count = 0;
    };

    [[nodiscard]] AllocationTotals allocationTotals(rachis::Device& device)
    {
        const auto budgets = device.queryMemoryBudgets();
        REQUIRE(budgets.has_value());
        AllocationTotals totals;
        for (const auto& budget : *budgets)
        {
            REQUIRE(rachis::hasAny(budget.features, rachis::MemoryBudgetFeature::allocationStatistics));
            totals.bytes += budget.allocationBytes;
            totals.count += budget.allocationCount;
        }
        return totals;
    }

    [[nodiscard]] std::vector<rachis::MemoryBudget> requireBudgets(const rachis::Device& device)
    {
        auto budgets = device.queryMemoryBudgets();
        REQUIRE(budgets.has_value());
        return std::move(*budgets);
    }

} // namespace

TEST_CASE("Vulkan format queries expose usage-specific features", "[size:medium][rhi][vulkan][texture][format]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto unknown = device.queryFormatProperties(rachis::Format::unknown);
    CHECK(unknown.features == rachis::FormatFeature::none);

    const auto color = device.queryFormatProperties(rachis::Format::rgba8Unorm);
    CHECK(rachis::hasAll(color.features,
                         rachis::FormatFeature::texture2D | rachis::FormatFeature::sampled |
                             rachis::FormatFeature::colorAttachment | rachis::FormatFeature::copySource |
                             rachis::FormatFeature::copyDestination |
                             rachis::FormatFeature::viewFormatReinterpretation));
    CHECK(rachis::supportsSampleCount(color.sampleCounts, 1));

    const auto depth = device.queryFormatProperties(rachis::Format::d32Float);
    CHECK(rachis::hasAny(depth.features, rachis::FormatFeature::depthStencilAttachment));
    CHECK_FALSE(rachis::hasAny(depth.features, rachis::FormatFeature::colorAttachment));

    const auto compressed = device.queryFormatProperties(rachis::Format::bc1RgbaUnorm);
    CHECK(rachis::hasAny(compressed.features, rachis::FormatFeature::sampled));
    CHECK_FALSE(rachis::hasAny(compressed.features, rachis::FormatFeature::colorAttachment));
}

TEST_CASE("Vulkan rejects invalid texture descriptors before allocation", "[size:medium][rhi][vulkan][texture]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto baseline = allocationTotals(device);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities);

    auto unknown = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::unknown,
        .usage = rachis::TextureUsage::copyDestination,
        .label = "Unknown format",
    });
    REQUIRE_FALSE(unknown);
    CHECK(unknown.error().code == rachis::ErrorCode::invalidArgument);

    auto noUsage = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::none,
        .label = "No usage",
    });
    REQUIRE_FALSE(noUsage);
    CHECK(noUsage.error().code == rachis::ErrorCode::invalidArgument);

    auto zeroExtent = device.createTexture({
        .width = 0,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Zero width",
    });
    REQUIRE_FALSE(zeroExtent);
    CHECK(zeroExtent.error().code == rachis::ErrorCode::invalidArgument);

    auto tooManyMips = device.createTexture({
        .width = 2,
        .height = 2,
        .mipLevels = 8,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Too many mips",
    });
    REQUIRE_FALSE(tooManyMips);
    CHECK(tooManyMips.error().code == rachis::ErrorCode::invalidArgument);

    auto depthWithoutUsage = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::sampled,
        .label = "Depth without depth-stencil usage",
    });
    REQUIRE_FALSE(depthWithoutUsage);
    CHECK(depthWithoutUsage.error().code == rachis::ErrorCode::invalidArgument);

    auto colorDepthUsage = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Color as depth",
    });
    REQUIRE_FALSE(colorDepthUsage);
    CHECK(colorDepthUsage.error().code == rachis::ErrorCode::invalidArgument);

    auto compressedRenderTarget = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::bc1RgbaUnorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Compressed RT",
    });
    REQUIRE_FALSE(compressedRenderTarget);
    CHECK(compressedRenderTarget.error().code == rachis::ErrorCode::unsupported);

    auto residency = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
        .label = "Manual residency",
    });
    REQUIRE_FALSE(residency);
    CHECK(residency.error().code == rachis::ErrorCode::unsupported);

    auto shadingRate = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::r8Unorm,
        .usage = rachis::TextureUsage::shadingRateSource,
        .label = "VRS",
    });
    REQUIRE_FALSE(shadingRate);
    CHECK(shadingRate.error().code == (capabilities->variableRateShadingTier2 ? rachis::ErrorCode::invalidArgument
                                                                              : rachis::ErrorCode::unsupported));

    const std::array repeatedBase{rachis::Format::rgba8Unorm};
    auto repeated = device.createTexture({
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .viewFormats = repeatedBase,
        .label = "Repeated base view format",
    });
    REQUIRE_FALSE(repeated);
    CHECK(repeated.error().code == rachis::ErrorCode::invalidArgument);

    const std::array incompatible{rachis::Format::bgra8Unorm};
    auto incompatibleViews = device.createTexture({
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .viewFormats = incompatible,
        .label = "Incompatible view format",
    });
    REQUIRE_FALSE(incompatibleViews);
    CHECK(incompatibleViews.error().code == rachis::ErrorCode::invalidArgument);

    const auto afterFailures = allocationTotals(device);
    CHECK(afterFailures.count == baseline.count);
    CHECK(afterFailures.bytes == baseline.bytes);
}

TEST_CASE("Vulkan textures views and samplers restore allocation counts", "[size:medium][rhi][vulkan][texture]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const bool swizzleAdvertised = rachis::test::requireCapabilities(device).textureViewComponentSwizzle;

    const auto baseline = allocationTotals(device);
    const auto baselineSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(baselineSnapshot);
    const auto baselineDescriptors = device.captureBackendAllocationSnapshot();

    auto requirements = device.queryTextureMemoryRequirements({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
        .label = "Requirements",
    });
    REQUIRE(requirements);
    CHECK(requirements->size > 0);
    CHECK(requirements->alignment > 0);

    const std::array alternateFormats{rachis::Format::rgba8UnormSrgb};
    auto color = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .depthOrArrayLayers = 2,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::renderTarget,
        .viewFormats = alternateFormats,
        .label = "Vulkan color",
    });
    auto depth = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::depthStencil,
        .label = "Vulkan depth",
    });
    auto compressed = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::bc1RgbaUnorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::sampled,
        .label = "Vulkan BC1",
    });
    REQUIRE(color);
    REQUIRE(depth);
    REQUIRE(compressed);

    const auto live = allocationTotals(device);
    CHECK(live.count == baseline.count + 3);
    CHECK(live.bytes > baseline.bytes);
    const auto liveSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(liveSnapshot);
    CHECK(liveSnapshot->liveAllocationCount == baselineSnapshot->liveAllocationCount + 3);

    const auto sampledComponents = swizzleAdvertised
                                       ? rachis::ComponentMapping{
                                             .red = rachis::ComponentSwizzle::blue,
                                             .green = rachis::ComponentSwizzle::green,
                                             .blue = rachis::ComponentSwizzle::red,
                                             .alpha = rachis::ComponentSwizzle::one,
                                         }
                                       : rachis::ComponentMapping{};
    auto sampled = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .components = sampledComponents,
        .label = "Sampled array",
    });
    auto srgb = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .format = rachis::Format::rgba8UnormSrgb,
        .arrayLayerCount = 1,
        .label = "sRGB RTV",
    });
    auto storage = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::storage,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .firstMipLevel = 1,
        .arrayLayerCount = 1,
        .label = "Storage",
    });
    auto depthView = device.createTextureView({
        .texture = *depth,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
        .label = "Depth view",
    });
    auto sampler = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .mipFilter = rachis::MipmapFilterMode::linear,
        .maxAnisotropy = 4,
        .label = "Anisotropic",
    });
    REQUIRE(sampled);
    REQUIRE(srgb);
    REQUIRE(storage);
    REQUIRE(depthView);
    REQUIRE(sampler);

    const auto descriptors = device.captureBackendAllocationSnapshot();
    CHECK(descriptors.liveDescriptorCount == baselineDescriptors.liveDescriptorCount + 5);

    CHECK_FALSE(device.destroyTexture(*color).has_value());
    const auto staleColor = *color;
    REQUIRE(device.destroyTextureView(*sampled).has_value());
    REQUIRE(device.destroyTextureView(*srgb).has_value());
    REQUIRE(device.destroyTextureView(*storage).has_value());
    REQUIRE(device.destroyTexture(*color).has_value());
    CHECK_FALSE(device.destroyTexture(staleColor).has_value());
    REQUIRE(device.destroyTextureView(*depthView).has_value());
    REQUIRE(device.destroyTexture(*depth).has_value());
    REQUIRE(device.destroyTexture(*compressed).has_value());
    REQUIRE(device.destroySampler(*sampler).has_value());

    const auto released = allocationTotals(device);
    CHECK(released.count == baseline.count);
    CHECK(released.bytes == baseline.bytes);
    const auto releasedSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(releasedSnapshot);
    CHECK(releasedSnapshot->liveAllocationCount == baselineSnapshot->liveAllocationCount);
    const auto releasedDescriptors = device.captureBackendAllocationSnapshot();
    CHECK(releasedDescriptors.liveDescriptorCount == baselineDescriptors.liveDescriptorCount);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan textures views and samplers release native objects when allocation fails",
          "[size:medium][rhi][vulkan][texture][sampler]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(vulkanDesc(&errors));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto baseline = allocationTotals(device);
    const auto baselineDescriptors = device.captureBackendAllocationSnapshot();

    const rachis::TextureDesc desc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        .label = "Vulkan texture with a failing keep-alive allocation",
    };
    const auto failedTexture = createWithBadAlloc([&] { return device.createTexture(desc); });
    REQUIRE_FALSE(failedTexture.has_value());
    CHECK(failedTexture.error().code == rachis::ErrorCode::outOfMemory);
    const auto afterFailure = allocationTotals(device);
    CHECK(afterFailure.count == baseline.count);
    CHECK(afterFailure.bytes == baseline.bytes);

    auto texture = device.createTexture(desc);
    REQUIRE(texture.has_value());
    const rachis::TextureViewDesc viewDesc{
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "Vulkan view with a failing keep-alive allocation",
    };
    const auto failedView = createWithBadAlloc([&] { return device.createTextureView(viewDesc); });
    REQUIRE_FALSE(failedView.has_value());
    CHECK(failedView.error().code == rachis::ErrorCode::outOfMemory);
    const rachis::SamplerDesc samplerDesc{
        .label = "Vulkan sampler with a failing keep-alive allocation",
    };
    const auto failedSampler = createWithBadAlloc([&] { return device.createSampler(samplerDesc); });
    REQUIRE_FALSE(failedSampler.has_value());
    CHECK(failedSampler.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(device.captureBackendAllocationSnapshot().liveDescriptorCount == baselineDescriptors.liveDescriptorCount);

    auto view = device.createTextureView(viewDesc);
    auto sampler = device.createSampler(samplerDesc);
    REQUIRE(view.has_value());
    REQUIRE(sampler.has_value());
    REQUIRE(device.destroySampler(*sampler).has_value());
    REQUIRE(device.destroyTextureView(*view).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    device.collectGarbage();

    CHECK(device.diagnosticCounts().errors == 0);
    device = {};
    CHECK(errors.load() == 0);
}

TEST_CASE("Vulkan shared textures count in memory budgets until their memory is freed",
          "[size:medium][rhi][vulkan][texture]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipWithoutCapability(capabilities.sharedTextureHandles && capabilities.memoryBudgets,
                                            "sharedTextureHandles and memoryBudgets"))
    {
        return;
    }

    const auto baseline = requireBudgets(device);
    constexpr std::string_view label = "Vulkan budgeted shared texture";
    auto shared = device.createSharedTexture({
        .width = 2048,
        .height = 2048,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = label,
    });
    if (!shared)
    {
        INFO(shared.error().message);
    }
    REQUIRE(shared.has_value());
    const auto snapshot = device.captureResourceMemorySnapshot();
    REQUIRE(snapshot.has_value());
    const auto entry = std::ranges::find(snapshot->entries, label, &rachis::ResourceMemoryEntry::label);
    REQUIRE(entry != snapshot->entries.end());
    CHECK(entry->shared);
    const auto allocationBytes = entry->allocationBytes;
    CHECK(allocationBytes >= std::uint64_t{2048} * 2048 * 4);

    const auto live = requireBudgets(device);
    REQUIRE(live.size() == baseline.size());
    std::size_t grownHeaps = 0;
    for (std::size_t index = 0; index < live.size(); ++index)
    {
        CAPTURE(index);
        if (live[index].allocationCount == baseline[index].allocationCount)
        {
            CHECK(live[index].allocationBytes == baseline[index].allocationBytes);
            continue;
        }
        ++grownHeaps;
        CHECK(rachis::hasAny(live[index].properties, rachis::MemoryHeapProperty::deviceLocal));
        CHECK(live[index].allocationCount == baseline[index].allocationCount + 1);
        CHECK(live[index].blockCount == baseline[index].blockCount + 1);
        CHECK(live[index].allocationBytes >= baseline[index].allocationBytes + allocationBytes);
        CHECK(live[index].blockBytes >= baseline[index].blockBytes + allocationBytes);
        CHECK(live[index].usageBytes >= baseline[index].usageBytes + allocationBytes);
    }
    CHECK(grownHeaps == 1);

    REQUIRE(device.destroyTexture(*shared).has_value());
    device.collectGarbage();
    const auto released = requireBudgets(device);
    REQUIRE(released.size() == baseline.size());
    for (std::size_t index = 0; index < released.size(); ++index)
    {
        CAPTURE(index);
        CHECK(released[index].blockBytes == baseline[index].blockBytes);
        CHECK(released[index].allocationBytes == baseline[index].allocationBytes);
        CHECK(released[index].blockCount == baseline[index].blockCount);
        CHECK(released[index].allocationCount == baseline[index].allocationCount);
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan views and samplers reject illegal ranges filters and aspects",
          "[size:medium][rhi][vulkan][texture][sampler]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto color = device.createTexture({
        .width = 16,
        .height = 16,
        .depthOrArrayLayers = 12,
        .mipLevels = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::renderTarget,
        .label = "View source",
    });
    REQUIRE(color);

    auto cube = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::cube,
        .label = "Cube",
    });
    auto cubeArray = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::cubeArray,
        .firstArrayLayer = 6,
        .arrayLayerCount = 6,
        .label = "Cube array",
    });
    REQUIRE(cube);
    REQUIRE(cubeArray);

    auto staleTextureView = device.createTextureView({
        .texture = {},
        .usage = rachis::TextureViewUsage::sampled,
    });
    REQUIRE_FALSE(staleTextureView);
    CHECK(staleTextureView.error().code == rachis::ErrorCode::invalidArgument);

    auto twoDOnArray = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE_FALSE(twoDOnArray);
    CHECK(twoDOnArray.error().code == rachis::ErrorCode::invalidArgument);

    auto multiMipStorage = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::storage,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .mipLevelCount = 2,
        .arrayLayerCount = 1,
    });
    REQUIRE_FALSE(multiMipStorage);
    CHECK(multiMipStorage.error().code == rachis::ErrorCode::invalidArgument);

    auto cubeFromSecondFace = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::cube,
        .firstArrayLayer = 6,
        .arrayLayerCount = 6,
    });
    REQUIRE_FALSE(cubeFromSecondFace);
    CHECK(cubeFromSecondFace.error().code == rachis::ErrorCode::invalidArgument);

    auto misalignedCubeArray = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::cubeArray,
        .firstArrayLayer = 1,
        .arrayLayerCount = 6,
    });
    REQUIRE_FALSE(misalignedCubeArray);
    CHECK(misalignedCubeArray.error().code == rachis::ErrorCode::invalidArgument);

    auto colorAsDepth = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .aspect = rachis::TextureAspect::depth,
        .arrayLayerCount = 1,
    });
    REQUIRE_FALSE(colorAsDepth);
    CHECK(colorAsDepth.error().code == rachis::ErrorCode::invalidArgument);

    rachis::ComponentMapping invalidSwizzle{};
    invalidSwizzle.alpha = static_cast<rachis::ComponentSwizzle>(255);
    auto invalidMapping = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .components = invalidSwizzle,
    });
    REQUIRE_FALSE(invalidMapping);
    CHECK(invalidMapping.error().code == rachis::ErrorCode::invalidArgument);

    auto swizzledRenderTarget = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .arrayLayerCount = 1,
        .components =
            {
                .red = rachis::ComponentSwizzle::blue,
            },
    });
    REQUIRE_FALSE(swizzledRenderTarget);
    CHECK(swizzledRenderTarget.error().code == rachis::ErrorCode::invalidArgument);

    const bool swizzleAdvertised = rachis::test::requireCapabilities(device).textureViewComponentSwizzle;
    constexpr rachis::ComponentMapping sampledSwizzle{
        .red = rachis::ComponentSwizzle::blue,
        .green = rachis::ComponentSwizzle::red,
        .blue = rachis::ComponentSwizzle::one,
        .alpha = rachis::ComponentSwizzle::zero,
    };
    auto swizzledSampled = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .arrayLayerCount = 1,
        .components = sampledSwizzle,
        .label = "Swizzled sampled",
    });
    if (swizzleAdvertised)
    {
        REQUIRE(swizzledSampled);
        REQUIRE(device.destroyTextureView(*swizzledSampled).has_value());
    }
    else
    {
        REQUIRE_FALSE(swizzledSampled);
        CHECK(swizzledSampled.error().code == rachis::ErrorCode::unsupported);
    }

    auto explicitRgba = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .arrayLayerCount = 1,
        .components =
            {
                .red = rachis::ComponentSwizzle::red,
                .green = rachis::ComponentSwizzle::green,
                .blue = rachis::ComponentSwizzle::blue,
                .alpha = rachis::ComponentSwizzle::alpha,
            },
        .label = "Explicit RGBA identity",
    });
    REQUIRE(explicitRgba);
    REQUIRE(device.destroyTextureView(*explicitRgba).has_value());

    auto zeroAnisotropy = device.createSampler({
        .maxAnisotropy = 0,
        .label = "Zero anisotropy",
    });
    REQUIRE_FALSE(zeroAnisotropy);
    CHECK(zeroAnisotropy.error().code == rachis::ErrorCode::invalidArgument);

    auto lodOrder = device.createSampler({
        .minLod = 4.0F,
        .maxLod = 1.0F,
        .label = "LOD order",
    });
    REQUIRE_FALSE(lodOrder);
    CHECK(lodOrder.error().code == rachis::ErrorCode::invalidArgument);

    auto nanLod = device.createSampler({
        .minLod = std::numeric_limits<float>::quiet_NaN(),
        .label = "NaN LOD",
    });
    REQUIRE_FALSE(nanLod);
    CHECK(nanLod.error().code == rachis::ErrorCode::invalidArgument);

    auto mipNoneMinLod = device.createSampler({
        .mipFilter = rachis::MipmapFilterMode::none,
        .minLod = 1.0F,
        .label = "Mip none minLod",
    });
    REQUIRE_FALSE(mipNoneMinLod);
    CHECK(mipNoneMinLod.error().code == rachis::ErrorCode::invalidArgument);

    auto reduction = device.createSampler({
        .reductionMode = rachis::SamplerReductionMode::minimum,
        .label = "Min reduction",
    });
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities);
    REQUIRE(reduction.has_value() == capabilities->samplerMinMaxReduction);
    if (reduction)
    {
        REQUIRE(device.destroySampler(*reduction));
    }
    else
    {
        CHECK(reduction.error().code == rachis::ErrorCode::unsupported);
    }

    REQUIRE(device.destroyTextureView(*cube).has_value());
    REQUIRE(device.destroyTextureView(*cubeArray).has_value());
    REQUIRE(device.destroyTexture(*color).has_value());
}

TEST_CASE("Vulkan samplers keep lodBias within maxSamplerLodBias without validation errors",
          "[size:medium][rhi][vulkan][texture][sampler]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto maxLodBias = rachis::test::requireCapabilities(device).limits.maxSamplerLodBias;
    CAPTURE(maxLodBias);
    CHECK(maxLodBias >= 2.0F);

    for (const auto mipFilter : {rachis::MipmapFilterMode::none, rachis::MipmapFilterMode::linear})
    {
        for (const float bias : {0.0F, maxLodBias, -maxLodBias})
        {
            CAPTURE(bias);
            auto accepted = device.createSampler({
                .minFilter = rachis::FilterMode::linear,
                .magFilter = rachis::FilterMode::linear,
                .mipFilter = mipFilter,
                .lodBias = bias,
                .label = "LOD bias within the limit",
            });
            REQUIRE(accepted.has_value());
            REQUIRE(device.destroySampler(*accepted).has_value());
        }

        const auto aboveMaxLodBias = std::nextafter(maxLodBias, std::numeric_limits<float>::infinity());
        for (const float bias : {aboveMaxLodBias, -aboveMaxLodBias, 20.0F, -20.0F, 100.0F, -100.0F})
        {
            CAPTURE(bias);
            auto rejected = device.createSampler({
                .mipFilter = mipFilter,
                .lodBias = bias,
                .label = "LOD bias beyond the limit",
            });
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
        }

        for (const float bias : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
        {
            auto rejected = device.createSampler({.mipFilter = mipFilter, .lodBias = bias});
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
        }
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan block-compressed copies accept the queried explicit rows per image",
          "[size:medium][rhi][vulkan][texture][transfer][compressed]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto texture = device.createTexture({
        .width = 7,
        .height = 5,
        .format = rachis::Format::bc1RgbaUnorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Odd-height BC1 texture",
    });
    REQUIRE(texture.has_value());
    const rachis::BufferTextureCopyRegion request{
        .textureSubresource = {.aspect = rachis::TextureAspect::color},
        .extent = {.width = 7, .height = 5},
    };
    auto layout = device.queryTextureTransferLayout(*texture, request);
    REQUIRE(layout.has_value());
    REQUIRE(layout->rowsPerImage == 5);

    auto upload = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    std::vector<std::byte> payload(static_cast<std::size_t>(layout->requiredBufferSize), std::byte{0});
    for (std::size_t index = 0; index < payload.size(); ++index)
    {
        payload[index] = static_cast<std::byte>((index * 13U + 3U) & 0xFFU);
    }
    REQUIRE(device.writeBuffer(*upload, 0, payload).has_value());

    const rachis::BufferTextureCopyRegion explicitRegion{
        .bytesPerRow = layout->bytesPerRow,
        .rowsPerImage = layout->rowsPerImage,
        .textureSubresource = request.textureSubresource,
        .extent = request.extent,
    };
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::span{&explicitRegion, 1}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::span{&explicitRegion, 1}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    std::vector<std::byte> recovered(payload.size(), std::byte{0});
    REQUIRE(device.readBuffer(*readback, 0, recovered).has_value());
    bool exact = true;
    for (std::uint32_t row = 0; row < layout->rowCount; ++row)
    {
        const auto rowOffset = static_cast<std::size_t>(row) * layout->bytesPerRow;
        for (std::uint64_t byte = 0; byte < layout->rowDataSize; ++byte)
        {
            exact = exact && recovered[rowOffset + byte] == payload[rowOffset + byte];
        }
    }
    CHECK(exact);

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
