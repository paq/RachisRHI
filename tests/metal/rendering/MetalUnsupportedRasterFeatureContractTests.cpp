#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

namespace
{

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadBytes(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        return stream ? bytes : std::vector<std::byte>{};
    }

    [[nodiscard]] std::filesystem::path pnmsPath(std::string_view metallibName)
    {
        return std::filesystem::path(RACHIS_METALLIB_DIR) /
               std::filesystem::path(metallibName).replace_extension(".pnms");
    }

} // namespace

TEST_CASE("Metal keeps raster-order, VRS, sampler feedback, dual-source, and HDR contracts unsupported",
          "[size:large][rhi][metal][color]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());
    auto vertex = device.createShaderModule(
        {.stage = rachis::ShaderStage::vertex, .bytecode = triangleVs, .entryPoint = "vertexMain"});
    auto pixel = device.createShaderModule(
        {.stage = rachis::ShaderStage::pixel, .bytecode = trianglePs, .entryPoint = "pixelMain"});
    REQUIRE(vertex);
    REQUIRE(pixel);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);

    const rachis::BlendState dualSourceBlend{
        .enabled = true,
        .color =
            {
                .sourceFactor = rachis::BlendFactor::source1,
                .destinationFactor = rachis::BlendFactor::oneMinusSource1,
            },
        .alpha =
            {
                .sourceFactor = rachis::BlendFactor::source1Alpha,
                .destinationFactor = rachis::BlendFactor::oneMinusSource1Alpha,
            },
    };
    const std::array dualSourceTargets{
        rachis::ColorTargetState{
            .format = rachis::Format::rgba8Unorm,
            .blend = dualSourceBlend,
        },
    };
    auto dualSource = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = dualSourceTargets,
    });
    REQUIRE_FALSE(dualSource.has_value());
    CHECK(dualSource.error().code == rachis::ErrorCode::unsupported);

    auto shadingRateTexture = device.createTexture({
        .width = 8,
        .height = 8,
        .format = rachis::Format::r8Uint,
        .usage = rachis::TextureUsage::shadingRateSource,
        .label = "Metal shading-rate image",
    });
    REQUIRE_FALSE(shadingRateTexture.has_value());
    CHECK(shadingRateTexture.error().code == rachis::ErrorCode::unsupported);
}

TEST_CASE("Metal rejects sample masks that leave active samples uncovered",
          "[size:large][rhi][metal][pipeline][multisample][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::supportsSampleCount(device.queryFormatProperties(rachis::Format::rgba8Unorm).sampleCounts, 4));

    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());
    auto vertex = device.createShaderModule(
        {.stage = rachis::ShaderStage::vertex, .bytecode = triangleVs, .entryPoint = "vertexMain"});
    auto pixel = device.createShaderModule(
        {.stage = rachis::ShaderStage::pixel, .bytecode = trianglePs, .entryPoint = "pixelMain"});
    REQUIRE(vertex);
    REQUIRE(pixel);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const auto createWithMask = [&](std::uint32_t sampleCount, std::uint32_t sampleMask) {
        return device.createGraphicsPipeline({
            .layout = *layout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .colorTargets = colorTargets,
            .sampleCount = sampleCount,
            .sampleMask = sampleMask,
        });
    };

    for (const std::uint32_t mask : {0x1U, 0x3U, 0x7U, 0x11U, 0xFFFFFFF7U})
    {
        CAPTURE(mask);
        auto partial = createWithMask(4, mask);
        REQUIRE_FALSE(partial.has_value());
        CHECK(partial.error().code == rachis::ErrorCode::unsupported);
    }
    for (const std::uint32_t mask : {0xFU, 0x1FU, 0xFFFFFFFFU})
    {
        CAPTURE(mask);
        auto covering = createWithMask(4, mask);
        REQUIRE(covering.has_value());
        REQUIRE(device.destroyPipeline(*covering).has_value());
    }
    auto empty = createWithMask(4, 0x10U);
    REQUIRE_FALSE(empty.has_value());
    CHECK(empty.error().code == rachis::ErrorCode::invalidArgument);

    auto single = createWithMask(1, 0x1U);
    REQUIRE(single.has_value());
    REQUIRE(device.destroyPipeline(*single).has_value());
    auto singleEmpty = createWithMask(1, 0xFFFFFFFEU);
    REQUIRE_FALSE(singleEmpty.has_value());
    CHECK(singleEmpty.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(device.diagnosticCounts().errors == 0);
}
