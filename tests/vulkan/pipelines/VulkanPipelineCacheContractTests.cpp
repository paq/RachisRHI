#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{

    constexpr rachis::PipelineCacheKey cacheKey{0x7265637472697821ULL, 0x706970656c696e65ULL};
    constexpr rachis::PipelineCacheKey graphicsCacheKey{0x6772617068696373ULL, 0x706970656c696e65ULL};
    constexpr rachis::PipelineCacheKey secondCacheKey{0x7365636f6e64216bULL, 0x706970656c696e65ULL};
    constexpr rachis::PipelineCacheKey secondGraphicsCacheKey{0x7365636f6e646772ULL, 0x706970656c696e65ULL};

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
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

    [[nodiscard]] rachis::ShaderInterfaceDesc cubeComputeInterface(
        std::array<rachis::ShaderBindingRequirement, 2>& bindings,
        std::array<rachis::ShaderPushConstantRequirement, 1>& push) noexcept
    {
        bindings = {
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 0,
                .type = rachis::ShaderResourceType::readOnlyStorageBuffer,
                .count = 1,
            },
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 1,
                .type = rachis::ShaderResourceType::storageBuffer,
                .count = 1,
            },
        };
        push = {rachis::ShaderPushConstantRequirement{.offset = 0, .size = 16}};
        return rachis::ShaderInterfaceDesc{
            .bindings = bindings,
            .pushConstants = push,
            .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
        };
    }

} // namespace

TEST_CASE("Vulkan pipeline caches persist keyed blobs and reject corrupt identity",
          "[size:medium][rhi][vulkan][pipeline][cache]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).pipelineCaches);

    const std::array malformedBlob{std::byte{0x52}, std::byte{0x58}, std::byte{0x50}, std::byte{0x43}};
    const auto malformed = device.createPipelineCache({.initialData = malformedBlob});
    REQUIRE_FALSE(malformed.has_value());
    CHECK(malformed.error().code == rachis::ErrorCode::invalidArgument);

    std::vector<std::byte> truncated(32, std::byte{0});
    const auto truncatedCache = device.createPipelineCache({.initialData = truncated});
    REQUIRE_FALSE(truncatedCache.has_value());
    CHECK(truncatedCache.error().code == rachis::ErrorCode::invalidArgument);

    const auto cube = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapCube.cs.spv");
    const auto triangleVs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.vs.spv");
    const auto trianglePs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.ps.spv");
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());

    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .label = "Cached Vulkan compute",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    REQUIRE(shader);
    const std::array computeEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = computeEntries});
    REQUIRE(bindingLayout);
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstants{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstants,
    });
    REQUIRE(pipelineLayout);

    const auto keyWithoutCache = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cacheKey = cacheKey,
    });
    REQUIRE_FALSE(keyWithoutCache.has_value());
    CHECK(keyWithoutCache.error().code == rachis::ErrorCode::invalidArgument);

    auto cache = device.createPipelineCache({.label = "Writable Vulkan pipeline cache"});
    REQUIRE(cache);
    const auto cacheWithoutKey = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cache = *cache,
    });
    REQUIRE_FALSE(cacheWithoutKey.has_value());
    CHECK(cacheWithoutKey.error().code == rachis::ErrorCode::invalidArgument);

    const auto cacheOnlyWithoutCache = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(cacheOnlyWithoutCache.has_value());
    CHECK(cacheOnlyWithoutCache.error().code == rachis::ErrorCode::invalidArgument);
    const auto initialMiss = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cache = *cache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(initialMiss.has_value());
    CHECK(initialMiss.error().code == rachis::ErrorCode::pipelineCacheMiss);

    auto firstPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "First cached Vulkan pipeline",
        .cache = *cache,
        .cacheKey = cacheKey,
    });
    REQUIRE(firstPipeline);
    auto cacheHitPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Cache-only Vulkan compute hit",
        .cache = *cache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(cacheHitPipeline);
    auto cacheData = device.pipelineCacheData(*cache);
    REQUIRE(cacheData);
    REQUIRE_FALSE(cacheData->empty());

    auto foreign = *cacheData;
    REQUIRE(foreign.size() >= 72);
    foreign[24] ^= std::byte{0xFF};
    const auto foreignCache = device.createPipelineCache({.initialData = foreign});
    REQUIRE_FALSE(foreignCache.has_value());
    CHECK(foreignCache.error().code == rachis::ErrorCode::invalidArgument);

    const auto staleCache = *cache;
    REQUIRE(device.destroyPipelineCache(*cache));
    CHECK_FALSE(device.pipelineCacheData(staleCache).has_value());
    CHECK_FALSE(device.destroyPipelineCache(staleCache).has_value());

    auto restoredCache = device.createPipelineCache({
        .initialData = *cacheData,
        .label = "Restored Vulkan pipeline cache",
    });
    REQUIRE(restoredCache);
    auto restoredPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Restored cached Vulkan pipeline",
        .cache = *restoredCache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(restoredPipeline);

    const std::array incompatiblePushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 3 * sizeof(std::uint32_t)},
    };
    auto incompatibleLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = incompatiblePushConstants,
    });
    REQUIRE(incompatibleLayout);
    const auto incompatiblePipeline = device.createComputePipeline({
        .layout = *incompatibleLayout,
        .computeShader = *shader,
        .cache = *restoredCache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(incompatiblePipeline.has_value());
    CHECK(incompatiblePipeline.error().code == rachis::ErrorCode::validationError);
    REQUIRE(device.destroyPipelineLayout(*incompatibleLayout));

    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(vertex);
    REQUIRE(pixel);
    auto graphicsLayout = device.createPipelineLayout({});
    REQUIRE(graphicsLayout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const auto graphicsMiss = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .cache = *restoredCache,
        .cacheKey = graphicsCacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(graphicsMiss.has_value());
    CHECK(graphicsMiss.error().code == rachis::ErrorCode::pipelineCacheMiss);
    auto graphicsPipeline = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .cache = *restoredCache,
        .cacheKey = graphicsCacheKey,
    });
    REQUIRE(graphicsPipeline);
    auto graphicsCacheHitPipeline = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .cache = *restoredCache,
        .cacheKey = graphicsCacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(graphicsCacheHitPipeline);

    REQUIRE(device.destroyPipelineCache(*restoredCache));
    REQUIRE(device.destroyPipeline(*firstPipeline));
    REQUIRE(device.destroyPipeline(*cacheHitPipeline));
    REQUIRE(device.destroyPipeline(*restoredPipeline));
    REQUIRE(device.destroyPipeline(*graphicsPipeline));
    REQUIRE(device.destroyPipeline(*graphicsCacheHitPipeline));
    REQUIRE(device.destroyPipelineLayout(*graphicsLayout));
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
    REQUIRE(device.destroyShaderModule(*shader));
    REQUIRE(device.destroyShaderModule(*vertex));
    REQUIRE(device.destroyShaderModule(*pixel));
    REQUIRE(device.destroyBindingLayout(*bindingLayout));
}

TEST_CASE("Vulkan pipeline caches keep restored entries when a restored cache saves new entries",
          "[size:medium][rhi][vulkan][pipeline][cache]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).pipelineCaches);

    const auto cube = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapCube.cs.spv");
    const auto write = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.spv");
    const auto triangleVs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.vs.spv");
    const auto trianglePs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.ps.spv");
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(write.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());

    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto cubeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    auto writeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = write,
        .entryPoint = "main",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(cubeShader);
    REQUIRE(writeShader);
    REQUIRE(vertex);
    REQUIRE(pixel);

    const std::array cubeEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    const std::array writeEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto cubeBindingLayout = device.createBindingLayout({.entries = cubeEntries});
    auto writeBindingLayout = device.createBindingLayout({.entries = writeEntries});
    REQUIRE(cubeBindingLayout);
    REQUIRE(writeBindingLayout);
    const std::array cubeBindingLayouts{*cubeBindingLayout};
    const std::array writeBindingLayouts{*writeBindingLayout};
    const std::array cubePushConstants{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    const std::array writePushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto cubeLayout =
        device.createPipelineLayout({.bindingLayouts = cubeBindingLayouts, .pushConstantRanges = cubePushConstants});
    auto writeLayout =
        device.createPipelineLayout({.bindingLayouts = writeBindingLayouts, .pushConstantRanges = writePushConstants});
    auto graphicsLayout = device.createPipelineLayout({});
    REQUIRE(cubeLayout);
    REQUIRE(writeLayout);
    REQUIRE(graphicsLayout);

    const auto requirePipeline =
        [&](std::size_t entry, rachis::PipelineCacheHandle cache, rachis::PipelineCachePolicy policy) {
            CAPTURE(entry);
            const std::array colorTargets{rachis::ColorTargetState{
                .format = entry == 2 ? rachis::Format::rgba8Unorm : rachis::Format::bgra8Unorm,
            }};
            auto pipeline = entry < 2 ? device.createComputePipeline({
                                            .layout = entry == 0 ? *cubeLayout : *writeLayout,
                                            .computeShader = entry == 0 ? *cubeShader : *writeShader,
                                            .cache = cache,
                                            .cacheKey = entry == 0 ? cacheKey : secondCacheKey,
                                            .cachePolicy = policy,
                                        })
                                      : device.createGraphicsPipeline({
                                            .layout = *graphicsLayout,
                                            .vertexShader = *vertex,
                                            .pixelShader = *pixel,
                                            .colorTargets = colorTargets,
                                            .cache = cache,
                                            .cacheKey = entry == 2 ? graphicsCacheKey : secondGraphicsCacheKey,
                                            .cachePolicy = policy,
                                        });
            if (!pipeline)
            {
                UNSCOPED_INFO(pipeline.error().message);
            }
            REQUIRE(pipeline);
            REQUIRE(device.destroyPipeline(*pipeline));
        };
    const auto requireData = [&](rachis::PipelineCacheHandle cache) {
        auto data = device.pipelineCacheData(cache);
        REQUIRE(data);
        REQUIRE_FALSE(data->empty());
        return std::move(*data);
    };
    constexpr auto populate = rachis::PipelineCachePolicy::populate;
    constexpr auto requireHit = rachis::PipelineCachePolicy::requireHit;

    auto first = device.createPipelineCache({.label = "First-session Vulkan pipeline cache"});
    REQUIRE(first);
    requirePipeline(0, *first, populate);
    requirePipeline(2, *first, populate);
    const auto firstData = requireData(*first);
    REQUIRE(device.destroyPipelineCache(*first));

    auto second =
        device.createPipelineCache({.initialData = firstData, .label = "Second-session Vulkan pipeline cache"});
    REQUIRE(second);
    requirePipeline(1, *second, populate);
    requirePipeline(3, *second, populate);
    const auto secondData = requireData(*second);
    requirePipeline(0, *second, requireHit);
    requirePipeline(2, *second, requireHit);
    REQUIRE(device.destroyPipelineCache(*second));

    auto third =
        device.createPipelineCache({.initialData = secondData, .label = "Third-session Vulkan pipeline cache"});
    REQUIRE(third);
    for (std::size_t entry = 0; entry < 4; ++entry)
    {
        requirePipeline(entry, *third, requireHit);
    }
    REQUIRE(device.destroyPipelineCache(*third));

    REQUIRE(device.destroyPipelineLayout(*graphicsLayout));
    REQUIRE(device.destroyPipelineLayout(*writeLayout));
    REQUIRE(device.destroyPipelineLayout(*cubeLayout));
    REQUIRE(device.destroyBindingLayout(*writeBindingLayout));
    REQUIRE(device.destroyBindingLayout(*cubeBindingLayout));
    REQUIRE(device.destroyShaderModule(*cubeShader));
    REQUIRE(device.destroyShaderModule(*writeShader));
    REQUIRE(device.destroyShaderModule(*vertex));
    REQUIRE(device.destroyShaderModule(*pixel));
}
