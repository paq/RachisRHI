#include "Internal/MetalFormatSupport.h"
#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

namespace
{

    constexpr std::size_t formatCount = static_cast<std::size_t>(rachis::Format::bc6hRgbSfloat) + 1;

    constexpr std::array tier1ReadWriteFormats{
        rachis::Format::r32Float,
        rachis::Format::r32Uint,
        rachis::Format::r32Sint,
    };

    constexpr std::array tier2ReadWriteFormats{
        rachis::Format::rgba32Float,
        rachis::Format::rgba32Uint,
        rachis::Format::rgba32Sint,
        rachis::Format::rgba16Float,
        rachis::Format::rgba16Uint,
        rachis::Format::rgba16Sint,
        rachis::Format::rgba8Unorm,
        rachis::Format::rgba8Uint,
        rachis::Format::rgba8Sint,
        rachis::Format::r16Float,
        rachis::Format::r16Uint,
        rachis::Format::r16Sint,
        rachis::Format::r8Unorm,
        rachis::Format::r8Uint,
        rachis::Format::r8Sint,
    };

    constexpr std::array undefinedReadWriteFormats{
        rachis::Format::rg8Unorm,
        rachis::Format::rg8Uint,
        rachis::Format::rg8Sint,
        rachis::Format::rg16Float,
        rachis::Format::rg16Uint,
        rachis::Format::rg16Sint,
        rachis::Format::rg32Float,
        rachis::Format::rg32Uint,
        rachis::Format::rg32Sint,
        rachis::Format::bgra8Unorm,
        rachis::Format::rgba8UnormSrgb,
        rachis::Format::bgra8UnormSrgb,
    };

    [[nodiscard]] bool listsFormat(std::span<const rachis::Format> formats, rachis::Format format) noexcept
    {
        return std::ranges::find(formats, format) != formats.end();
    }

#if RACHIS_ENABLE_METAL
    constexpr std::uint32_t textureWidth = 4;
    constexpr std::uint32_t textureHeight = 4;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
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
#endif

} // namespace

TEST_CASE("Metal read-write texture tiers accept only the formats Apple defines for read_write access",
          "[size:small][rhi][metal][texture][format][regression]")
{
    for (std::size_t index = 0; index < formatCount; ++index)
    {
        const auto format = static_cast<rachis::Format>(index);
        CAPTURE(index);
        const bool tier1 = listsFormat(tier1ReadWriteFormats, format);
        const bool tier2 = tier1 || listsFormat(tier2ReadWriteFormats, format);
        CHECK_FALSE(rachis::metalReadWriteFormat(format, rachis::MetalReadWriteTextureTier::none));
        CHECK(rachis::metalReadWriteFormat(format, rachis::MetalReadWriteTextureTier::tier1) == tier1);
        CHECK(rachis::metalReadWriteFormat(format, rachis::MetalReadWriteTextureTier::tier2) == tier2);
    }
    for (const auto format : undefinedReadWriteFormats)
    {
        CAPTURE(static_cast<int>(format));
        CHECK_FALSE(rachis::metalReadWriteFormat(format, rachis::MetalReadWriteTextureTier::tier2));
    }
}

#if RACHIS_ENABLE_METAL

TEST_CASE("Metal format queries expose usage-specific features", "[size:large][rhi][metal][texture][format]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
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

TEST_CASE("Metal static format capability table covers compressed, depth, integer, float, and sRGB contracts",
          "[size:large][rhi][metal][texture][format]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto d24s8 = device.queryFormatProperties(rachis::Format::d24UnormS8Uint);
    if (d24s8.features == rachis::FormatFeature::none)
    {
        CHECK(d24s8.sampleCounts == rachis::SampleCountMask::none);
        auto unsupportedDepth = device.createTexture({
            .width = textureWidth,
            .height = textureHeight,
            .format = rachis::Format::d24UnormS8Uint,
            .usage = rachis::TextureUsage::depthStencil,
            .label = "Unsupported D24S8",
        });
        REQUIRE_FALSE(unsupportedDepth);
        CHECK(unsupportedDepth.error().code == rachis::ErrorCode::unsupported);
    }
    else
    {
        CHECK(rachis::hasAny(d24s8.features, rachis::FormatFeature::depthStencilAttachment));
        CHECK_FALSE(rachis::hasAny(d24s8.features, rachis::FormatFeature::colorAttachment));
        auto packed = device.createTexture({
            .width = textureWidth,
            .height = textureHeight,
            .format = rachis::Format::d24UnormS8Uint,
            .usage = rachis::TextureUsage::depthStencil | rachis::TextureUsage::sampled,
            .label = "D24S8 aspect views",
        });
        REQUIRE(packed);
        auto stencilView = device.createTextureView({.texture = *packed, .aspect = rachis::TextureAspect::stencil});
        if (!stencilView)
        {
            UNSCOPED_INFO(stencilView.error().message);
        }
        CHECK(stencilView);
        auto depthView = device.createTextureView({.texture = *packed, .aspect = rachis::TextureAspect::depth});
        CHECK(depthView);
    }

    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal advertises storage only for read-write tier formats and rejects storage usage on the others",
          "[size:large][rhi][metal][texture][format][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    for (std::size_t index = 0; index < formatCount; ++index)
    {
        const auto format = static_cast<rachis::Format>(index);
        CAPTURE(index);
        const auto properties = device.queryFormatProperties(format);
        const bool readWrite = rachis::metalReadWriteFormat(format, rachis::MetalReadWriteTextureTier::tier2);
        CHECK(rachis::hasAny(properties.features, rachis::FormatFeature::storageRead) == readWrite);
        CHECK(rachis::hasAny(properties.features, rachis::FormatFeature::storageWrite) == readWrite);
    }
    for (const auto format : undefinedReadWriteFormats)
    {
        CAPTURE(static_cast<int>(format));
        const auto properties = device.queryFormatProperties(format);
        CHECK(rachis::hasAll(properties.features,
                             rachis::FormatFeature::texture2D | rachis::FormatFeature::sampled |
                                 rachis::FormatFeature::colorAttachment | rachis::FormatFeature::copySource |
                                 rachis::FormatFeature::copyDestination));
    }

    const rachis::TextureDesc rgStorage{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rg16Float,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "RG16Float storage texture",
    };
    auto rgRejected = device.createTexture(rgStorage);
    REQUIRE_FALSE(rgRejected);
    CHECK(rgRejected.error().code == rachis::ErrorCode::unsupported);
    auto rgRequirements = device.queryTextureMemoryRequirements(rgStorage);
    REQUIRE_FALSE(rgRequirements);
    CHECK(rgRequirements.error().code == rachis::ErrorCode::unsupported);
    auto rgSampledDesc = rgStorage;
    rgSampledDesc.usage = rachis::TextureUsage::sampled;
    rgSampledDesc.label = "RG16Float sampled texture";
    auto rgSampled = device.createTexture(rgSampledDesc);
    REQUIRE(rgSampled);
    REQUIRE(device.destroyTexture(*rgSampled).has_value());

    rachis::TextureDesc srgbStorage{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8UnormSrgb,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "sRGB storage texture without a linear view",
    };
    auto missingLinearView = device.createTexture(srgbStorage);
    REQUIRE_FALSE(missingLinearView);
    CHECK(missingLinearView.error().code == rachis::ErrorCode::unsupported);
    const std::array rgbaLinearView{rachis::Format::rgba8Unorm};
    srgbStorage.viewFormats = rgbaLinearView;
    srgbStorage.label = "sRGB storage texture with a linear view";
    auto srgb = device.createTexture(srgbStorage);
    REQUIRE(srgb);
    auto srgbStorageView = device.createTextureView({
        .texture = *srgb,
        .usage = rachis::TextureViewUsage::storage,
        .label = "sRGB storage view",
    });
    REQUIRE_FALSE(srgbStorageView);
    CHECK(srgbStorageView.error().code == rachis::ErrorCode::unsupported);
    auto linearStorageView = device.createTextureView({
        .texture = *srgb,
        .usage = rachis::TextureViewUsage::storage,
        .format = rachis::Format::rgba8Unorm,
        .label = "Linear storage view of an sRGB texture",
    });
    REQUIRE(linearStorageView);
    REQUIRE(device.destroyTextureView(*linearStorageView).has_value());
    REQUIRE(device.destroyTexture(*srgb).has_value());

    const std::array bgraLinearView{rachis::Format::bgra8Unorm};
    auto bgraRejected = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::bgra8UnormSrgb,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .viewFormats = bgraLinearView,
        .label = "BGRA sRGB storage texture with a linear view",
    });
    REQUIRE_FALSE(bgraRejected);
    CHECK(bgraRejected.error().code == rachis::ErrorCode::unsupported);

    const std::array srgbAlternateView{rachis::Format::rgba8UnormSrgb};
    auto linear = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .viewFormats = srgbAlternateView,
        .label = "Linear storage texture with an sRGB view",
    });
    REQUIRE(linear);
    auto srgbAliasStorageView = device.createTextureView({
        .texture = *linear,
        .usage = rachis::TextureViewUsage::storage,
        .format = rachis::Format::rgba8UnormSrgb,
        .label = "sRGB alias storage view",
    });
    REQUIRE_FALSE(srgbAliasStorageView);
    CHECK(srgbAliasStorageView.error().code == rachis::ErrorCode::unsupported);
    REQUIRE(device.destroyTexture(*linear).has_value());

    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal rejects invalid texture descriptors before allocation", "[size:large][rhi][metal][texture]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
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

    const rachis::TextureDesc mipmapped1D{
        .dimension = rachis::TextureDimension::texture1D,
        .width = 64,
        .height = 1,
        .mipLevels = 7,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::sampled | rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Mipmapped 1D",
    };
    auto mipmapped1DTexture = device.createTexture(mipmapped1D);
    REQUIRE_FALSE(mipmapped1DTexture);
    CHECK(mipmapped1DTexture.error().code == rachis::ErrorCode::unsupported);
    auto mipmapped1DRequirements = device.queryTextureMemoryRequirements(mipmapped1D);
    REQUIRE_FALSE(mipmapped1DRequirements);
    CHECK(mipmapped1DRequirements.error().code == rachis::ErrorCode::unsupported);

    auto mipmapped1DArrayDesc = mipmapped1D;
    mipmapped1DArrayDesc.depthOrArrayLayers = 4;
    auto mipmapped1DArray = device.createTexture(mipmapped1DArrayDesc);
    REQUIRE_FALSE(mipmapped1DArray);
    CHECK(mipmapped1DArray.error().code == rachis::ErrorCode::unsupported);
    auto twoMip1DArrayDesc = mipmapped1DArrayDesc;
    twoMip1DArrayDesc.mipLevels = 2;
    auto twoMip1DArray = device.queryTextureMemoryRequirements(twoMip1DArrayDesc);
    REQUIRE_FALSE(twoMip1DArray);
    CHECK(twoMip1DArray.error().code == rachis::ErrorCode::unsupported);

    auto zeroMip1DDesc = mipmapped1D;
    zeroMip1DDesc.mipLevels = 0;
    auto zeroMip1D = device.createTexture(zeroMip1DDesc);
    REQUIRE_FALSE(zeroMip1D);
    CHECK(zeroMip1D.error().code == rachis::ErrorCode::invalidArgument);
    auto overlongChain1DDesc = mipmapped1D;
    overlongChain1DDesc.mipLevels = 8;
    auto overlongChain1D = device.createTexture(overlongChain1DDesc);
    REQUIRE_FALSE(overlongChain1D);
    CHECK(overlongChain1D.error().code == rachis::ErrorCode::invalidArgument);
    auto tall1DDesc = mipmapped1D;
    tall1DDesc.height = 2;
    tall1DDesc.mipLevels = 2;
    auto tall1D = device.createTexture(tall1DDesc);
    REQUIRE_FALSE(tall1D);
    CHECK(tall1D.error().code == rachis::ErrorCode::invalidArgument);

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

TEST_CASE("Metal reports 1D render-target and depth-stencil textures as unsupported and keeps other 1D usages",
          "[size:large][rhi][metal][texture][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto baseline = allocationTotals(device);

    const rachis::TextureDesc lineTarget{
        .width = 64,
        .height = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .label = "2D height-1 render target",
    };
    auto lineRequirements = device.queryTextureMemoryRequirements(lineTarget);
    REQUIRE(lineRequirements);
    auto heap = device.createResourceHeap({
        .size = lineRequirements->size,
        .alignment = lineRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::attachmentTexture,
        .label = "1D attachment heap",
    });
    REQUIRE(heap);

    for (const std::uint32_t layers : {1U, 4U})
    {
        CAPTURE(layers);
        auto colorDesc = lineTarget;
        colorDesc.dimension = rachis::TextureDimension::texture1D;
        colorDesc.depthOrArrayLayers = layers;
        colorDesc.label = "1D render target";
        auto depthDesc = colorDesc;
        depthDesc.format = rachis::Format::d32Float;
        depthDesc.usage = rachis::TextureUsage::depthStencil;
        depthDesc.label = "1D depth-stencil target";
        for (const auto& desc : {colorDesc, depthDesc})
        {
            CAPTURE(desc.label);
            auto texture = device.createTexture(desc);
            REQUIRE_FALSE(texture);
            CHECK(texture.error().code == rachis::ErrorCode::unsupported);
            auto requirements = device.queryTextureMemoryRequirements(desc);
            REQUIRE_FALSE(requirements);
            CHECK(requirements.error().code == rachis::ErrorCode::unsupported);
            auto placed = device.createPlacedTexture({.heap = *heap, .offset = 0}, desc);
            REQUIRE_FALSE(placed);
            CHECK(placed.error().code == rachis::ErrorCode::unsupported);
        }
    }

    auto colorAsDepthDesc = lineTarget;
    colorAsDepthDesc.dimension = rachis::TextureDimension::texture1D;
    colorAsDepthDesc.usage = rachis::TextureUsage::depthStencil;
    auto colorAsDepth = device.createTexture(colorAsDepthDesc);
    REQUIRE_FALSE(colorAsDepth);
    CHECK(colorAsDepth.error().code == rachis::ErrorCode::invalidArgument);
    auto tallTargetDesc = lineTarget;
    tallTargetDesc.dimension = rachis::TextureDimension::texture1D;
    tallTargetDesc.height = 2;
    auto tallTarget = device.createTexture(tallTargetDesc);
    REQUIRE_FALSE(tallTarget);
    CHECK(tallTarget.error().code == rachis::ErrorCode::invalidArgument);

    for (const std::uint32_t layers : {1U, 4U})
    {
        CAPTURE(layers);
        auto texture = device.createTexture({
            .dimension = rachis::TextureDimension::texture1D,
            .width = 64,
            .height = 1,
            .depthOrArrayLayers = layers,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::copySource |
                     rachis::TextureUsage::copyDestination,
            .label = "1D sampled storage texture",
        });
        REQUIRE(texture);
        const auto viewDimension =
            layers == 1 ? rachis::TextureViewDimension::texture1D : rachis::TextureViewDimension::texture1DArray;
        auto sampled = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::sampled,
            .dimension = viewDimension,
            .label = "1D sampled view",
        });
        REQUIRE(sampled);
        auto storage = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::storage,
            .dimension = viewDimension,
            .label = "1D storage view",
        });
        REQUIRE(storage);
        auto renderTargetView = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension = viewDimension,
            .label = "1D render-target view",
        });
        REQUIRE_FALSE(renderTargetView);
        CHECK(renderTargetView.error().code == rachis::ErrorCode::invalidArgument);
        REQUIRE(device.destroyTextureView(*storage).has_value());
        REQUIRE(device.destroyTextureView(*sampled).has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
    }

    auto line = device.createPlacedTexture({.heap = *heap, .offset = 0}, lineTarget);
    REQUIRE(line);
    auto lineView = device.createTextureView({
        .texture = *line,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2D,
        .label = "2D height-1 render-target view",
    });
    REQUIRE(lineView);
    REQUIRE(device.destroyTextureView(*lineView).has_value());
    REQUIRE(device.destroyTexture(*line).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
    device.collectGarbage();

    const auto released = allocationTotals(device);
    CHECK(released.count == baseline.count);
    CHECK(released.bytes == baseline.bytes);
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal textures views and samplers restore allocation counts", "[size:large][rhi][metal][texture]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto checkedCapabilities = device.capabilities();
    REQUIRE(checkedCapabilities.has_value());
    REQUIRE(checkedCapabilities->textureViewComponentSwizzle);

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
        .label = "Metal color",
    });
    auto depth = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::depthStencil,
        .label = "Metal depth",
    });
    auto compressed = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::bc1RgbaUnorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::sampled,
        .label = "Metal BC1",
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

    auto sampled = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .components =
            {
                .red = rachis::ComponentSwizzle::blue,
                .green = rachis::ComponentSwizzle::green,
                .blue = rachis::ComponentSwizzle::red,
                .alpha = rachis::ComponentSwizzle::one,
            },
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

TEST_CASE("Metal views and samplers reject illegal ranges filters and aspects",
          "[size:large][rhi][metal][texture][sampler]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
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
    REQUIRE_FALSE(reduction);
    CHECK(reduction.error().code == rachis::ErrorCode::unsupported);

    REQUIRE(device.destroyTextureView(*cube).has_value());
    REQUIRE(device.destroyTextureView(*cubeArray).has_value());
    REQUIRE(device.destroyTexture(*color).has_value());
}

TEST_CASE("Metal samplers that differ only in label share one native sampler state",
          "[size:large][rhi][metal][texture][sampler][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const rachis::SamplerDesc linear{
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .mipFilter = rachis::MipmapFilterMode::linear,
        .addressU = rachis::AddressMode::repeat,
    };
    auto firstDesc = linear;
    firstDesc.label = "Metal shared sampler A";
    auto secondDesc = linear;
    secondDesc.label = "Metal shared sampler B";
    auto nearestDesc = firstDesc;
    nearestDesc.minFilter = rachis::FilterMode::nearest;
    nearestDesc.magFilter = rachis::FilterMode::nearest;

    auto first = device.createSampler(firstDesc);
    auto second = device.createSampler(secondDesc);
    auto nearest = device.createSampler(nearestDesc);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(nearest);

    const auto firstId = rachis::metal::boundSamplerGpuResourceId(*first);
    const auto secondId = rachis::metal::boundSamplerGpuResourceId(*second);
    const auto nearestId = rachis::metal::boundSamplerGpuResourceId(*nearest);
    CHECK(firstId != 0);
    CHECK(nearestId != 0);
    CHECK(firstId == secondId);
    CHECK(nearestId != firstId);

    REQUIRE(device.destroySampler(*first).has_value());
    CHECK(rachis::metal::boundSamplerGpuResourceId(*second) == secondId);
    REQUIRE(device.destroySampler(*second).has_value());
    REQUIRE(device.destroySampler(*nearest).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal resolveTexture rejects formats without multisampleResolve",
          "[size:large][rhi][metal][texture][resolve]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto msaaSource = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .sampleCount = 4,
        .format = rachis::Format::rgba8Uint,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Integer MSAA source",
    });
    auto singleDestination = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .sampleCount = 1,
        .format = rachis::Format::rgba8Uint,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Integer resolve destination",
    });
    REQUIRE(msaaSource);
    REQUIRE(singleDestination);

    auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(cmd);
    auto resolved = device.resolveTexture(*cmd,
                                          {
                                              .source = *msaaSource,
                                              .destination = *singleDestination,
                                          });
    REQUIRE_FALSE(resolved);
    CHECK(resolved.error().code == rachis::ErrorCode::unsupported);
}

#endif
