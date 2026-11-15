#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{

    constexpr rachis::PipelineCacheKey cacheKey{0x7265637472697821ULL, 0x706970656c696e65ULL};
    constexpr rachis::PipelineCacheKey graphicsCacheKey{0x6772617068696373ULL, 0x706970656c696e65ULL};
    constexpr rachis::PipelineCacheKey secondCacheKey{0x7365636f6e64216bULL, 0x706970656c696e65ULL};
    constexpr rachis::PipelineCacheKey secondGraphicsCacheKey{0x7365636f6e646772ULL, 0x706970656c696e65ULL};

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

    [[nodiscard]] rachis::ShaderInterfaceDesc cubeComputeInterface(
        std::array<rachis::ShaderBindingRequirement, 2>& bindings,
        std::array<rachis::ShaderPushConstantRequirement, 1>& push) noexcept
    {
        bindings = {
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 1,
                .type = rachis::ShaderResourceType::readOnlyStorageBuffer,
                .count = 1,
            },
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 2,
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

    [[nodiscard]] std::uint32_t readU32(rachis::Device& device, rachis::BufferHandle buffer)
    {
        std::uint32_t value = 0;
        REQUIRE(device.readBuffer(buffer, 0, std::as_writable_bytes(std::span{&value, 1})).has_value());
        return value;
    }

    [[nodiscard]] std::filesystem::path makeEmptyTemporaryDirectory(std::string_view prefix)
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        auto directory = std::filesystem::temp_directory_path() / (std::string(prefix) + std::to_string(stamp));
        std::filesystem::remove_all(directory);
        REQUIRE(std::filesystem::create_directories(directory));
        return directory;
    }

    [[nodiscard]] std::ptrdiff_t countDirectoryEntries(const std::filesystem::path& directory)
    {
        return std::distance(std::filesystem::directory_iterator(directory), std::filesystem::directory_iterator{});
    }

    struct SpecializedComputeOutput final
    {
        rachis::BufferHandle storage{};
        rachis::BufferHandle readback{};
        rachis::BindingSetHandle set{};
    };

    [[nodiscard]] SpecializedComputeOutput createSpecializedComputeOutput(rachis::Device& device,
                                                                          rachis::BindingLayoutHandle layout)
    {
        constexpr std::uint64_t outputSize = sizeof(std::uint32_t);
        auto storage = device.createBuffer({
            .size = outputSize,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::deviceLocal,
        });
        auto readback = device.createBuffer({
            .size = outputSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(storage);
        REQUIRE(readback);
        auto set = rachis::test::createBindingSet(
            device,
            {.layout = layout,
             .writes = std::array{
                 rachis::BindingWrite{0, 0, rachis::BufferBinding{*storage, 0, outputSize, sizeof(std::uint32_t)}}}});
        REQUIRE(set);
        return SpecializedComputeOutput{.storage = *storage, .readback = *readback, .set = *set};
    }

    void recordSpecializedDispatch(rachis::Device& device,
                                   rachis::CommandBuffer& command,
                                   rachis::PipelineHandle pipeline,
                                   const SpecializedComputeOutput& output)
    {
        REQUIRE(device.bindComputePipeline(command, pipeline));
        REQUIRE(device.bindComputeBindingSet(command, 0, output.set));
        REQUIRE(device.dispatch(command, 1));
        REQUIRE(device.bufferBarrier(command,
                                     {
                                         .buffer = output.storage,
                                         .stagesBefore = rachis::PipelineStage::computeShader,
                                         .stagesAfter = rachis::PipelineStage::copy,
                                         .accessBefore = rachis::Access::unorderedAccess,
                                         .accessAfter = rachis::Access::copySource,
                                     }));
        REQUIRE(device.copyBuffer(
            command, {.source = output.storage, .destination = output.readback, .size = sizeof(std::uint32_t)}));
    }

    enum class CacheEntry : std::uint8_t
    {
        cubeCompute,
        variantCompute,
        rgbaGraphics,
        bgraGraphics,
    };

    constexpr std::array allCacheEntries{
        CacheEntry::cubeCompute,
        CacheEntry::variantCompute,
        CacheEntry::rgbaGraphics,
        CacheEntry::bgraGraphics,
    };

    [[nodiscard]] constexpr const char* cacheEntryName(CacheEntry entry) noexcept
    {
        switch (entry)
        {
        case CacheEntry::cubeCompute:
            return "cube compute";
        case CacheEntry::variantCompute:
            return "variant compute";
        case CacheEntry::rgbaGraphics:
            return "rgba graphics";
        case CacheEntry::bgraGraphics:
            return "bgra graphics";
        }
        return "unknown";
    }

    struct CacheEntryPipelines final
    {
        std::vector<std::byte> cubeBytes;
        std::vector<std::byte> variantBytes;
        std::vector<std::byte> vertexBytes;
        std::vector<std::byte> pixelBytes;
        std::array<rachis::ShaderBindingRequirement, 2> cubeBindings{};
        std::array<rachis::ShaderPushConstantRequirement, 1> cubePush{};
        rachis::ShaderModuleHandle cube{};
        rachis::ShaderModuleHandle variant{};
        rachis::ShaderModuleHandle vertex{};
        rachis::ShaderModuleHandle pixel{};
        rachis::BindingLayoutHandle cubeBindingLayout{};
        rachis::BindingLayoutHandle variantBindingLayout{};
        rachis::PipelineLayoutHandle cubeLayout{};
        rachis::PipelineLayoutHandle variantLayout{};
        rachis::PipelineLayoutHandle graphicsLayout{};
    };

    void createCacheEntryPipelines(rachis::Device& device, CacheEntryPipelines& pipelines)
    {
        pipelines.cubeBytes = loadBytes(pnmsPath("BootstrapCube.cs.metallib"));
        pipelines.variantBytes = loadBytes(pnmsPath("MetalSpecialization.7.cs.metallib"));
        pipelines.vertexBytes = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
        pipelines.pixelBytes = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
        REQUIRE_FALSE(pipelines.cubeBytes.empty());
        REQUIRE_FALSE(pipelines.variantBytes.empty());
        REQUIRE_FALSE(pipelines.vertexBytes.empty());
        REQUIRE_FALSE(pipelines.pixelBytes.empty());

        auto cube = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = pipelines.cubeBytes,
            .entryPoint = "computeMain",
            .shaderInterface = cubeComputeInterface(pipelines.cubeBindings, pipelines.cubePush),
        });
        const std::array storageBinding{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
        };
        auto variant = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = pipelines.variantBytes,
            .entryPoint = "computeMain",
            .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, storageBinding),
        });
        auto vertex = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = pipelines.vertexBytes,
            .entryPoint = "vertexMain",
        });
        auto pixel = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = pipelines.pixelBytes,
            .entryPoint = "pixelMain",
        });
        REQUIRE(cube);
        REQUIRE(variant);
        REQUIRE(vertex);
        REQUIRE(pixel);
        pipelines.cube = *cube;
        pipelines.variant = *variant;
        pipelines.vertex = *vertex;
        pipelines.pixel = *pixel;

        const std::array cubeEntries{
            rachis::BindingLayoutEntry{1, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
            rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        };
        const std::array variantEntries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        };
        auto cubeBindingLayout = device.createBindingLayout({.entries = cubeEntries});
        auto variantBindingLayout = device.createBindingLayout({.entries = variantEntries});
        REQUIRE(cubeBindingLayout);
        REQUIRE(variantBindingLayout);
        pipelines.cubeBindingLayout = *cubeBindingLayout;
        pipelines.variantBindingLayout = *variantBindingLayout;

        auto cubeLayout = device.createPipelineLayout({
            .bindingLayouts = std::array{pipelines.cubeBindingLayout},
            .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
        });
        auto variantLayout =
            device.createPipelineLayout({.bindingLayouts = std::array{pipelines.variantBindingLayout}});
        auto graphicsLayout = device.createPipelineLayout({});
        REQUIRE(cubeLayout);
        REQUIRE(variantLayout);
        REQUIRE(graphicsLayout);
        pipelines.cubeLayout = *cubeLayout;
        pipelines.variantLayout = *variantLayout;
        pipelines.graphicsLayout = *graphicsLayout;
    }

    void destroyCacheEntryPipelines(rachis::Device& device, const CacheEntryPipelines& pipelines)
    {
        for (const auto layout : {pipelines.cubeLayout, pipelines.variantLayout, pipelines.graphicsLayout})
        {
            REQUIRE(device.destroyPipelineLayout(layout));
        }
        for (const auto layout : {pipelines.cubeBindingLayout, pipelines.variantBindingLayout})
        {
            REQUIRE(device.destroyBindingLayout(layout));
        }
        for (const auto shader : {pipelines.cube, pipelines.variant, pipelines.vertex, pipelines.pixel})
        {
            REQUIRE(device.destroyShaderModule(shader));
        }
    }

    [[nodiscard]] rachis::Result<rachis::PipelineHandle> createCacheEntryPipeline(rachis::Device& device,
                                                                                  const CacheEntryPipelines& pipelines,
                                                                                  CacheEntry entry,
                                                                                  rachis::PipelineCacheHandle cache,
                                                                                  rachis::PipelineCachePolicy policy)
    {
        if (entry == CacheEntry::cubeCompute || entry == CacheEntry::variantCompute)
        {
            const bool cubeEntry = entry == CacheEntry::cubeCompute;
            return device.createComputePipeline({
                .layout = cubeEntry ? pipelines.cubeLayout : pipelines.variantLayout,
                .computeShader = cubeEntry ? pipelines.cube : pipelines.variant,
                .cache = cache,
                .cacheKey = cubeEntry ? cacheKey : secondCacheKey,
                .cachePolicy = policy,
            });
        }
        const bool rgbaEntry = entry == CacheEntry::rgbaGraphics;
        const std::array colorTargets{rachis::ColorTargetState{
            .format = rgbaEntry ? rachis::Format::rgba8Unorm : rachis::Format::bgra8Unorm,
        }};
        return device.createGraphicsPipeline({
            .layout = pipelines.graphicsLayout,
            .vertexShader = pipelines.vertex,
            .pixelShader = pipelines.pixel,
            .colorTargets = colorTargets,
            .cache = cache,
            .cacheKey = rgbaEntry ? graphicsCacheKey : secondGraphicsCacheKey,
            .cachePolicy = policy,
        });
    }

    void requireCacheEntryPipeline(rachis::Device& device,
                                   const CacheEntryPipelines& pipelines,
                                   CacheEntry entry,
                                   rachis::PipelineCacheHandle cache,
                                   rachis::PipelineCachePolicy policy)
    {
        INFO(cacheEntryName(entry));
        auto pipeline = createCacheEntryPipeline(device, pipelines, entry, cache, policy);
        if (!pipeline)
        {
            UNSCOPED_INFO(pipeline.error().message);
        }
        REQUIRE(pipeline);
        REQUIRE(device.destroyPipeline(*pipeline));
    }

    [[nodiscard]] std::vector<std::byte> requirePipelineCacheData(rachis::Device& device,
                                                                  rachis::PipelineCacheHandle cache)
    {
        auto data = device.pipelineCacheData(cache);
        if (!data)
        {
            UNSCOPED_INFO(data.error().message);
        }
        REQUIRE(data);
        REQUIRE_FALSE(data->empty());
        return std::move(*data);
    }

    [[nodiscard]] rachis::PipelineCacheHandle requireRestoredPipelineCache(rachis::Device& device,
                                                                           std::span<const std::byte> data)
    {
        auto cache = device.createPipelineCache({.initialData = data, .label = "Metal restored mixed cache"});
        if (!cache)
        {
            UNSCOPED_INFO(cache.error().message);
        }
        REQUIRE(cache);
        return *cache;
    }

} // namespace

TEST_CASE("Metal pipeline caches persist keyed entries without owning live pipelines",
          "[size:large][rhi][metal][pipeline][cache]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto checkedCapabilities = device.capabilities();
    REQUIRE(checkedCapabilities.has_value());
    REQUIRE(checkedCapabilities->pipelineCaches);

    const std::array malformedBlob{std::byte{0x52}, std::byte{0x58}, std::byte{0x50}, std::byte{0x43}};
    const auto malformed = device.createPipelineCache({.initialData = malformedBlob});
    REQUIRE_FALSE(malformed.has_value());
    CHECK(malformed.error().code == rachis::ErrorCode::invalidArgument);

    std::vector<std::byte> truncated(32, std::byte{0});
    const auto truncatedCache = device.createPipelineCache({.initialData = truncated});
    REQUIRE_FALSE(truncatedCache.has_value());
    CHECK(truncatedCache.error().code == rachis::ErrorCode::invalidArgument);

    const auto cube = loadBytes(pnmsPath("BootstrapCube.cs.metallib"));
    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());

    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .label = "Cached Metal compute",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    REQUIRE(shader);
    const std::array computeEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
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
    REQUIRE_FALSE(keyWithoutCache);
    CHECK(keyWithoutCache.error().code == rachis::ErrorCode::invalidArgument);

    auto cache = device.createPipelineCache({.label = "Metal writable pipeline cache"});
    REQUIRE(cache);
    const auto cacheWithoutKey = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cache = *cache,
    });
    REQUIRE_FALSE(cacheWithoutKey);
    CHECK(cacheWithoutKey.error().code == rachis::ErrorCode::invalidArgument);

    const auto cacheOnlyWithoutCache = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(cacheOnlyWithoutCache);
    CHECK(cacheOnlyWithoutCache.error().code == rachis::ErrorCode::invalidArgument);

    const auto initialMiss = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cache = *cache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(initialMiss);
    CHECK(initialMiss.error().code == rachis::ErrorCode::pipelineCacheMiss);

    auto firstPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "First cached Metal pipeline",
        .cache = *cache,
        .cacheKey = cacheKey,
    });
    REQUIRE(firstPipeline);
    auto cacheHitPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Cache-only Metal compute hit",
        .cache = *cache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(cacheHitPipeline);
    auto cacheData = device.pipelineCacheData(*cache);
    if (!cacheData)
    {
        UNSCOPED_INFO(cacheData.error().message);
        UNSCOPED_INFO("native=" << cacheData.error().nativeCode);
    }
    REQUIRE(cacheData);
    REQUIRE_FALSE(cacheData->empty());

    auto foreign = *cacheData;
    REQUIRE(foreign.size() >= 40);
    foreign.back() ^= std::byte{0xFF};
    const auto foreignCache = device.createPipelineCache({.initialData = foreign});
    REQUIRE_FALSE(foreignCache.has_value());
    CHECK(foreignCache.error().code == rachis::ErrorCode::invalidArgument);

    auto interrupted = *cacheData;
    interrupted.pop_back();
    const auto interruptedCache = device.createPipelineCache({.initialData = interrupted});
    REQUIRE_FALSE(interruptedCache.has_value());
    CHECK(interruptedCache.error().code == rachis::ErrorCode::invalidArgument);

    const auto staleCache = *cache;
    REQUIRE(device.destroyPipelineCache(*cache));
    CHECK_FALSE(device.pipelineCacheData(staleCache).has_value());
    CHECK_FALSE(device.destroyPipelineCache(staleCache).has_value());
    REQUIRE(device.destroyPipeline(*firstPipeline));
    REQUIRE(device.destroyPipeline(*cacheHitPipeline));

    auto restoredCache = device.createPipelineCache({
        .initialData = *cacheData,
        .label = "Restored Metal pipeline cache",
    });
    if (!restoredCache)
    {
        UNSCOPED_INFO(restoredCache.error().message);
    }
    REQUIRE(restoredCache);
    auto restoredPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Restored cached Metal pipeline",
        .cache = *restoredCache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    if (!restoredPipeline)
    {
        UNSCOPED_INFO(restoredPipeline.error().message);
    }
    REQUIRE(restoredPipeline);

    auto incompatibleLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 24}},
    });
    REQUIRE(incompatibleLayout);
    const auto incompatiblePipeline = device.createComputePipeline({
        .layout = *incompatibleLayout,
        .computeShader = *shader,
        .cache = *restoredCache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(incompatiblePipeline);
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
    REQUIRE_FALSE(graphicsMiss);
    CHECK(graphicsMiss.error().code == rachis::ErrorCode::pipelineCacheMiss);
    auto graphicsPipeline = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .cache = *restoredCache,
        .cacheKey = graphicsCacheKey,
    });
    if (!graphicsPipeline)
    {
        UNSCOPED_INFO(graphicsPipeline.error().message);
    }
    REQUIRE(graphicsPipeline);
    auto graphicsHit = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .cache = *restoredCache,
        .cacheKey = graphicsCacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(graphicsHit);
    REQUIRE(device.destroyPipeline(*graphicsPipeline));
    REQUIRE(device.destroyPipeline(*graphicsHit));
    REQUIRE(device.destroyPipeline(*restoredPipeline));
    REQUIRE(device.destroyPipelineCache(*restoredCache));
    REQUIRE(device.destroyPipelineLayout(*graphicsLayout));
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
    REQUIRE(device.destroyShaderModule(*shader));
    REQUIRE(device.destroyShaderModule(*vertex));
    REQUIRE(device.destroyShaderModule(*pixel));
    REQUIRE(device.destroyBindingLayout(*bindingLayout));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal pipeline-cache restore keeps GPU specialization output identical",
          "[size:large][rhi][metal][pipeline][cache][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto variant13 = loadBytes(pnmsPath("MetalSpecialization.13.cs.metallib"));
    REQUIRE_FALSE(variant13.empty());
    std::array spec13{
        rachis::SpecializationConstant::makeUnsignedInt32(0, 13),
        rachis::SpecializationConstant::makeBoolean(5, true),
    };
    auto library = device.createShaderLibrary({
        .entries =
            std::array{
                rachis::ShaderLibraryEntryDesc{
                    .stage = rachis::ShaderStage::compute,
                    .bytecode = variant13,
                    .entryPoint = "computeMain",
                    .specializationConstants = spec13,
                },
            },
        .label = "Metal cached specialization",
    });
    REQUIRE(library);
    const std::array storageBinding{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .entryPoint = "computeMain",
        .library = *library,
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, storageBinding),
        .specializationConstants = spec13,
    });
    REQUIRE(shader);
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout);
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = std::array{*bindingLayout}});
    REQUIRE(pipelineLayout);

    auto cache = device.createPipelineCache({.label = "Metal specialization cache"});
    REQUIRE(cache);
    const auto coldStart = std::chrono::steady_clock::now();
    auto coldPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cache = *cache,
        .cacheKey = cacheKey,
    });
    const auto coldElapsed = std::chrono::steady_clock::now() - coldStart;
    REQUIRE(coldPipeline);
    auto blob = device.pipelineCacheData(*cache);
    if (!blob)
    {
        UNSCOPED_INFO(blob.error().message);
        UNSCOPED_INFO("native=" << blob.error().nativeCode);
    }
    REQUIRE(blob);
    REQUIRE_FALSE(blob->empty());
    REQUIRE(device.destroyPipeline(*coldPipeline));
    REQUIRE(device.destroyPipelineCache(*cache));

    auto restored = device.createPipelineCache({.initialData = *blob, .label = "Metal restored specialization cache"});
    if (!restored)
    {
        UNSCOPED_INFO(restored.error().message);
    }
    REQUIRE(restored);
    const auto warmStart = std::chrono::steady_clock::now();
    auto warmPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cache = *restored,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    const auto warmElapsed = std::chrono::steady_clock::now() - warmStart;
    if (!warmPipeline)
    {
        UNSCOPED_INFO(warmPipeline.error().message);
    }
    REQUIRE(warmPipeline);
    UNSCOPED_INFO("cold_ns=" << std::chrono::duration_cast<std::chrono::nanoseconds>(coldElapsed).count());
    UNSCOPED_INFO("warm_ns=" << std::chrono::duration_cast<std::chrono::nanoseconds>(warmElapsed).count());

    constexpr std::uint64_t outputSize = sizeof(std::uint32_t);
    auto storage = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(storage);
    REQUIRE(readback);
    auto set = rachis::test::createBindingSet(
        device,
        {.layout = *bindingLayout,
         .writes = std::array{
             rachis::BindingWrite{0, 0, rachis::BufferBinding{*storage, 0, outputSize, sizeof(std::uint32_t)}}}});
    REQUIRE(set);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.bindComputePipeline(*command, *warmPipeline));
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set));
    REQUIRE(device.dispatch(*command, 1));
    REQUIRE(device.copyBuffer(*command, {.source = *storage, .destination = *readback, .size = outputSize}));
    REQUIRE(device.destroyPipeline(*warmPipeline));
    REQUIRE(device.destroyPipelineCache(*restored));
    REQUIRE(device.destroyBindingSet(*set));
    REQUIRE(device.destroyBuffer(*storage));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    CHECK(readU32(device, *readback) == 13);
    REQUIRE(device.destroyBuffer(*readback));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal pipeline-cache serialization removes its temporary archive file when it fails",
          "[size:large][rhi][metal][pipeline][cache]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto cube = loadBytes(pnmsPath("BootstrapCube.cs.metallib"));
    REQUIRE_FALSE(cube.empty());
    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    REQUIRE(shader);
    const std::array computeEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = computeEntries});
    REQUIRE(bindingLayout);
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*bindingLayout},
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
    });
    REQUIRE(pipelineLayout);

    const auto directory = makeEmptyTemporaryDirectory("rachis-metal-pipeline-cache-");
    {
        rachis::metal::ScopedPipelineCacheTemporaryDirectory const temporaryDirectory(directory.string());
        auto cache = device.createPipelineCache({.label = "Metal failing serialization cache"});
        REQUIRE(cache);
        auto pipeline = device.createComputePipeline({
            .layout = *pipelineLayout,
            .computeShader = *shader,
            .cache = *cache,
            .cacheKey = cacheKey,
        });
        REQUIRE(pipeline);

        {
            rachis::metal::ScopedFailNextPipelineCacheSerialize const failSerialize;
            const auto failed = device.pipelineCacheData(*cache);
            REQUIRE_FALSE(failed);
            CHECK(failed.error().code == rachis::ErrorCode::nativeFailure);
        }
        CHECK(countDirectoryEntries(directory) == 0);

        auto cacheData = device.pipelineCacheData(*cache);
        if (!cacheData)
        {
            UNSCOPED_INFO(cacheData.error().message);
        }
        REQUIRE(cacheData);
        REQUIRE_FALSE(cacheData->empty());
        REQUIRE(device.destroyPipeline(*pipeline));
        REQUIRE(device.destroyPipelineCache(*cache));
        CHECK(countDirectoryEntries(directory) == 0);

        auto restored = device.createPipelineCache({.initialData = *cacheData});
        REQUIRE(restored);
        REQUIRE(device.destroyPipelineCache(*restored));
        CHECK(countDirectoryEntries(directory) == 0);
    }
    std::filesystem::remove_all(directory);

    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
    REQUIRE(device.destroyBindingLayout(*bindingLayout));
    REQUIRE(device.destroyShaderModule(*shader));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal pipeline caches reject a cache key reused for a different pipeline",
          "[size:large][rhi][metal][pipeline][cache][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto variant7 = loadBytes(pnmsPath("MetalSpecialization.7.cs.metallib"));
    const auto variant13 = loadBytes(pnmsPath("MetalSpecialization.13.cs.metallib"));
    REQUIRE_FALSE(variant7.empty());
    REQUIRE_FALSE(variant13.empty());
    const std::array storageBinding{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const auto createVariant = [&](std::span<const std::byte> bytecode) {
        return device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "computeMain",
            .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, storageBinding),
        });
    };
    auto shader7 = createVariant(variant7);
    auto shader13 = createVariant(variant13);
    REQUIRE(shader7);
    REQUIRE(shader13);
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout);
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = std::array{*bindingLayout}});
    REQUIRE(pipelineLayout);

    auto cache = device.createPipelineCache({.label = "Metal reused-key cache"});
    REQUIRE(cache);
    auto pipeline7 = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader7,
        .cache = *cache,
        .cacheKey = cacheKey,
    });
    REQUIRE(pipeline7);

    const auto reusedHit = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader13,
        .cache = *cache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(reusedHit);
    CHECK(reusedHit.error().code == rachis::ErrorCode::invalidArgument);
    const auto reusedPopulate = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader13,
        .cache = *cache,
        .cacheKey = cacheKey,
    });
    REQUIRE_FALSE(reusedPopulate);
    CHECK(reusedPopulate.error().code == rachis::ErrorCode::invalidArgument);

    auto reloaded7 = createVariant(variant7);
    REQUIRE(reloaded7);
    auto hit7 = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *reloaded7,
        .cache = *cache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(hit7);
    auto pipeline13 = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader13,
        .cache = *cache,
        .cacheKey = secondCacheKey,
    });
    REQUIRE(pipeline13);
    auto hit13 = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader13,
        .cache = *cache,
        .cacheKey = secondCacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(hit13);

    const auto output7 = createSpecializedComputeOutput(device, *bindingLayout);
    const auto output13 = createSpecializedComputeOutput(device, *bindingLayout);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    recordSpecializedDispatch(device, *command, *hit7, output7);
    recordSpecializedDispatch(device, *command, *hit13, output13);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    CHECK(readU32(device, output7.readback) == 7);
    CHECK(readU32(device, output13.readback) == 13);

    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());
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
    const std::array rgbaTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const std::array bgraTargets{rachis::ColorTargetState{.format = rachis::Format::bgra8Unorm}};
    auto graphicsPipeline = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = rgbaTargets,
        .cache = *cache,
        .cacheKey = graphicsCacheKey,
    });
    REQUIRE(graphicsPipeline);
    const auto reusedGraphicsHit = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = bgraTargets,
        .cache = *cache,
        .cacheKey = graphicsCacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(reusedGraphicsHit);
    CHECK(reusedGraphicsHit.error().code == rachis::ErrorCode::invalidArgument);
    auto graphicsHit = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = rgbaTargets,
        .cache = *cache,
        .cacheKey = graphicsCacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(graphicsHit);

    for (const auto pipeline : {*pipeline7, *hit7, *pipeline13, *hit13, *graphicsPipeline, *graphicsHit})
    {
        REQUIRE(device.destroyPipeline(pipeline));
    }
    for (const auto& output : {output7, output13})
    {
        REQUIRE(device.destroyBindingSet(output.set));
        REQUIRE(device.destroyBuffer(output.storage));
        REQUIRE(device.destroyBuffer(output.readback));
    }
    REQUIRE(device.destroyPipelineCache(*cache));
    REQUIRE(device.destroyPipelineLayout(*graphicsLayout));
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
    REQUIRE(device.destroyBindingLayout(*bindingLayout));
    for (const auto shaderModule : {*shader7, *shader13, *reloaded7, *vertex, *pixel})
    {
        REQUIRE(device.destroyShaderModule(shaderModule));
    }
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal pipeline caches keep restored entries when a restored cache saves new entries",
          "[size:large][rhi][metal][pipeline][cache]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    CacheEntryPipelines pipelines;
    createCacheEntryPipelines(device, pipelines);
    constexpr auto populate = rachis::PipelineCachePolicy::populate;
    constexpr auto requireHit = rachis::PipelineCachePolicy::requireHit;

    auto first = device.createPipelineCache({.label = "Metal first-session cache"});
    REQUIRE(first);
    requireCacheEntryPipeline(device, pipelines, CacheEntry::cubeCompute, *first, populate);
    requireCacheEntryPipeline(device, pipelines, CacheEntry::rgbaGraphics, *first, populate);
    const auto firstData = requirePipelineCacheData(device, *first);
    REQUIRE(device.destroyPipelineCache(*first));

    const auto second = requireRestoredPipelineCache(device, firstData);
    requireCacheEntryPipeline(device, pipelines, CacheEntry::variantCompute, second, populate);
    requireCacheEntryPipeline(device, pipelines, CacheEntry::bgraGraphics, second, populate);
    const auto secondData = requirePipelineCacheData(device, second);
    requireCacheEntryPipeline(device, pipelines, CacheEntry::cubeCompute, second, requireHit);
    requireCacheEntryPipeline(device, pipelines, CacheEntry::rgbaGraphics, second, requireHit);
    CHECK(requirePipelineCacheData(device, second) == secondData);
    REQUIRE(device.destroyPipelineCache(second));

    const auto third = requireRestoredPipelineCache(device, secondData);
    for (const auto entry : allCacheEntries)
    {
        requireCacheEntryPipeline(device, pipelines, entry, third, requireHit);
    }
    CHECK(requirePipelineCacheData(device, third) == secondData);
    REQUIRE(device.destroyPipelineCache(third));

    destroyCacheEntryPipelines(device, pipelines);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal pipeline caches keep every entry across repeated saves of one cache",
          "[size:large][rhi][metal][pipeline][cache]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    CacheEntryPipelines pipelines;
    createCacheEntryPipelines(device, pipelines);
    constexpr auto populate = rachis::PipelineCachePolicy::populate;
    constexpr auto requireHit = rachis::PipelineCachePolicy::requireHit;

    const auto directory = makeEmptyTemporaryDirectory("rachis-metal-pipeline-cache-segments-");
    {
        rachis::metal::ScopedPipelineCacheTemporaryDirectory const temporaryDirectory(directory.string());
        auto cache = device.createPipelineCache({.label = "Metal repeatedly saved cache"});
        REQUIRE(cache);
        std::vector<std::byte> data;
        for (const auto entry : allCacheEntries)
        {
            requireCacheEntryPipeline(device, pipelines, entry, *cache, populate);
            data = requirePipelineCacheData(device, *cache);
        }
        REQUIRE(device.destroyPipelineCache(*cache));
        CHECK(countDirectoryEntries(directory) == 0);

        const auto restored = requireRestoredPipelineCache(device, data);
        for (const auto entry : allCacheEntries)
        {
            requireCacheEntryPipeline(device, pipelines, entry, restored, requireHit);
        }
        requireCacheEntryPipeline(device, pipelines, CacheEntry::cubeCompute, restored, populate);
        const auto repopulated = requirePipelineCacheData(device, restored);
        REQUIRE(device.destroyPipelineCache(restored));
        CHECK(countDirectoryEntries(directory) == 0);

        const auto reloaded = requireRestoredPipelineCache(device, repopulated);
        for (const auto entry : allCacheEntries)
        {
            requireCacheEntryPipeline(device, pipelines, entry, reloaded, requireHit);
        }
        CHECK(requirePipelineCacheData(device, reloaded) == repopulated);
        REQUIRE(device.destroyPipelineCache(reloaded));
        CHECK(countDirectoryEntries(directory) == 0);
    }
    std::filesystem::remove_all(directory);

    destroyCacheEntryPipelines(device, pipelines);
    CHECK(device.diagnosticCounts().errors == 0);
}
