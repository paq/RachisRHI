#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <Windows.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <vector>

namespace
{
    void skipWithoutHardwareAdapter(const rachis::Error& error)
    {
        const auto classification = rachis::test::classifyDeviceCreateError(error.code);
        if (classification == rachis::test::DeviceCreateClassification::adapterNotFound ||
            classification == rachis::test::DeviceCreateClassification::backendUnavailable)
        {
            SKIP("No D3D12 hardware adapter is available: " << error.message);
        }
        FAIL("Creating the D3D12 hardware device failed: " << error.message);
    }

    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);
    constexpr rachis::PipelineCacheKey cacheKey{0x7265637472697821ULL, 0x706970656c696e65ULL};
    constexpr rachis::PipelineCacheKey graphicsCacheKey{0x6772617068696373ULL, 0x706970656c696e65ULL};
    constexpr rachis::PipelineCacheKey secondCacheKey{0x7365636f6e64216bULL, 0x706970656c696e65ULL};
    constexpr rachis::PipelineCacheKey secondGraphicsCacheKey{0x7365636f6e646772ULL, 0x706970656c696e65ULL};
    constexpr rachis::PipelineCacheKey depthBoundsCacheKey{0x4445505448424F55ULL, 0x706970656c696e65ULL};

    class ArmTestHook final
    {
    public:
        explicit ArmTestHook(std::atomic_bool& hook)
            : hook_(hook), previous_(hook.exchange(true, std::memory_order_relaxed))
        {
        }

        ~ArmTestHook()
        {
            hook_.store(previous_, std::memory_order_relaxed);
        }

        ArmTestHook(const ArmTestHook&) = delete;
        ArmTestHook& operator=(const ArmTestHook&) = delete;
        ArmTestHook(ArmTestHook&&) = delete;
        ArmTestHook& operator=(ArmTestHook&&) = delete;

        [[nodiscard]] bool consumed() const noexcept
        {
            return !hook_.load(std::memory_order_relaxed);
        }

    private:
        std::atomic_bool& hook_;
        bool previous_ = false;
    };

    [[nodiscard]] std::vector<std::byte> loadShader(const char* path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> result(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(result.data()), end);
        return stream ? result : std::vector<std::byte>{};
    }

    struct VirtualPagesDeleter final
    {
        void operator()(void* pages) const noexcept
        {
            static_cast<void>(VirtualFree(pages, 0, MEM_RELEASE));
        }
    };

} // namespace

TEST_CASE("pipeline caches persist D3D12 pipelines without owning them", "[size:medium][rhi][pipeline][cache][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).pipelineCaches);

    const std::array malformedBlob{std::byte{0x52}, std::byte{0x58}, std::byte{0x50}, std::byte{0x43}};
    const auto malformed = device.createPipelineCache({.initialData = malformedBlob});
    REQUIRE_FALSE(malformed.has_value());
    CHECK(malformed.error().code == rachis::ErrorCode::invalidArgument);

    const auto bytecode = loadShader(RACHIS_TEST_COMPUTE_SHADER_PATH);
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Cached compute shader",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(shader.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries, .label = "Cached output layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstants,
        .label = "Cached compute layout",
    });
    REQUIRE(pipelineLayout.has_value());

    const auto keyWithoutCache = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cacheKey = cacheKey,
    });
    REQUIRE_FALSE(keyWithoutCache.has_value());
    CHECK(keyWithoutCache.error().code == rachis::ErrorCode::invalidArgument);

    auto cache = device.createPipelineCache({.label = "Writable pipeline cache"});
    REQUIRE(cache.has_value());
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
        .label = "First cached pipeline",
        .cache = *cache,
        .cacheKey = cacheKey,
    });
    REQUIRE(firstPipeline.has_value());
    auto cacheHitPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Cache-only compute hit",
        .cache = *cache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(cacheHitPipeline.has_value());
    auto cacheData = device.pipelineCacheData(*cache);
    REQUIRE(cacheData.has_value());
    REQUIRE_FALSE(cacheData->empty());

    const auto staleCache = *cache;
    REQUIRE(device.destroyPipelineCache(*cache).has_value());
    CHECK_FALSE(device.pipelineCacheData(staleCache).has_value());
    CHECK_FALSE(device.destroyPipelineCache(staleCache).has_value());

    auto restoredCache = device.createPipelineCache({
        .initialData = *cacheData,
        .label = "Restored pipeline cache",
    });
    REQUIRE(restoredCache.has_value());
    auto restoredPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Restored cached pipeline",
        .cache = *restoredCache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(restoredPipeline.has_value());

    const std::array incompatiblePushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 3 * sizeof(std::uint32_t)},
    };
    auto incompatibleLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = incompatiblePushConstants,
    });
    REQUIRE(incompatibleLayout.has_value());
    const auto incompatiblePipeline = device.createComputePipeline({
        .layout = *incompatibleLayout,
        .computeShader = *shader,
        .cache = *restoredCache,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(incompatiblePipeline.has_value());
    CHECK(incompatiblePipeline.error().code == rachis::ErrorCode::validationError);
    REQUIRE(device.destroyPipelineLayout(*incompatibleLayout).has_value());

    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const std::array graphicsPushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto graphicsLayout = device.createPipelineLayout({.pushConstantRanges = graphicsPushConstants});
    REQUIRE(graphicsLayout.has_value());
    const auto graphicsMiss = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .colorTargets = colorTargets,
        .cache = *restoredCache,
        .cacheKey = graphicsCacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(graphicsMiss.has_value());
    CHECK(graphicsMiss.error().code == rachis::ErrorCode::pipelineCacheMiss);
    auto graphicsPipeline = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .colorTargets = colorTargets,
        .cache = *restoredCache,
        .cacheKey = graphicsCacheKey,
    });
    REQUIRE(graphicsPipeline.has_value());
    auto graphicsCacheHitPipeline = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .colorTargets = colorTargets,
        .cache = *restoredCache,
        .cacheKey = graphicsCacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE(graphicsCacheHitPipeline.has_value());

    auto storage = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(storage.has_value());
    REQUIRE(readback.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *storage,
                    .size = outputSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(bindingSet.has_value());

    REQUIRE(device.destroyPipelineCache(*restoredCache).has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *restoredPipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    const std::array constants{5U, 11U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(constants))).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(
        device.copyBuffer(*command, {.source = *storage, .destination = *readback, .size = outputSize}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::uint32_t firstMismatch = valueCount;
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        std::uint32_t value = 0;
        std::memcpy(&value, mapped->data + index * sizeof(value), sizeof(value));
        if (value != index * constants[0] + constants[1])
        {
            firstMismatch = index;
            break;
        }
    }
    CAPTURE(firstMismatch);
    CHECK(firstMismatch == valueCount);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*firstPipeline).has_value());
    REQUIRE(device.destroyPipeline(*cacheHitPipeline).has_value());
    REQUIRE(device.destroyPipeline(*restoredPipeline).has_value());
    REQUIRE(device.destroyPipeline(*graphicsPipeline).has_value());
    REQUIRE(device.destroyPipeline(*graphicsCacheHitPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*graphicsLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
}

TEST_CASE("pipeline caches keep restored D3D12 entries when a restored cache saves new entries",
          "[size:medium][rhi][pipeline][cache][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).pipelineCaches);

    const auto writeBytecode = loadShader(RACHIS_TEST_COMPUTE_SHADER_PATH);
    const auto variantBytecode = loadShader(RACHIS_TEST_SPECIALIZATION_VARIANT_7_PATH);
    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(writeBytecode.empty());
    REQUIRE_FALSE(variantBytecode.empty());
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    const std::array storageBinding{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto writeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = writeBytecode,
        .entryPoint = "main",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    auto variantShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = variantBytecode,
        .entryPoint = "main",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, storageBinding),
    });
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(writeShader.has_value());
    REQUIRE(variantShader.has_value());
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array writePushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    const std::array graphicsPushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto writeLayout =
        device.createPipelineLayout({.bindingLayouts = bindingLayouts, .pushConstantRanges = writePushConstants});
    auto variantLayout = device.createPipelineLayout({.bindingLayouts = bindingLayouts});
    auto graphicsLayout = device.createPipelineLayout({.pushConstantRanges = graphicsPushConstants});
    REQUIRE(writeLayout.has_value());
    REQUIRE(variantLayout.has_value());
    REQUIRE(graphicsLayout.has_value());

    const auto requirePipeline =
        [&](std::size_t entry, rachis::PipelineCacheHandle cache, rachis::PipelineCachePolicy policy) {
            CAPTURE(entry);
            const std::array colorTargets{rachis::ColorTargetState{
                .format = entry == 2 ? rachis::Format::rgba8Unorm : rachis::Format::bgra8Unorm,
            }};
            auto pipeline = entry < 2 ? device.createComputePipeline({
                                            .layout = entry == 0 ? *writeLayout : *variantLayout,
                                            .computeShader = entry == 0 ? *writeShader : *variantShader,
                                            .cache = cache,
                                            .cacheKey = entry == 0 ? cacheKey : secondCacheKey,
                                            .cachePolicy = policy,
                                        })
                                      : device.createGraphicsPipeline({
                                            .layout = *graphicsLayout,
                                            .vertexShader = *vertexShader,
                                            .pixelShader = *pixelShader,
                                            .colorTargets = colorTargets,
                                            .cache = cache,
                                            .cacheKey = entry == 2 ? graphicsCacheKey : secondGraphicsCacheKey,
                                            .cachePolicy = policy,
                                        });
            if (!pipeline)
            {
                UNSCOPED_INFO(pipeline.error().message);
            }
            REQUIRE(pipeline.has_value());
            REQUIRE(device.destroyPipeline(*pipeline).has_value());
        };
    const auto requireData = [&](rachis::PipelineCacheHandle cache) {
        auto data = device.pipelineCacheData(cache);
        REQUIRE(data.has_value());
        REQUIRE_FALSE(data->empty());
        return std::move(*data);
    };
    constexpr auto populate = rachis::PipelineCachePolicy::populate;
    constexpr auto requireHit = rachis::PipelineCachePolicy::requireHit;

    auto first = device.createPipelineCache({.label = "First-session pipeline cache"});
    REQUIRE(first.has_value());
    requirePipeline(0, *first, populate);
    requirePipeline(2, *first, populate);
    const auto firstData = requireData(*first);
    REQUIRE(device.destroyPipelineCache(*first).has_value());

    auto second = device.createPipelineCache({.initialData = firstData, .label = "Second-session pipeline cache"});
    REQUIRE(second.has_value());
    requirePipeline(1, *second, populate);
    requirePipeline(3, *second, populate);
    const auto secondData = requireData(*second);
    requirePipeline(0, *second, requireHit);
    requirePipeline(2, *second, requireHit);
    REQUIRE(device.destroyPipelineCache(*second).has_value());

    auto third = device.createPipelineCache({.initialData = secondData, .label = "Third-session pipeline cache"});
    REQUIRE(third.has_value());
    for (std::size_t entry = 0; entry < 4; ++entry)
    {
        requirePipeline(entry, *third, requireHit);
    }
    REQUIRE(device.destroyPipelineCache(*third).has_value());

    REQUIRE(device.destroyPipelineLayout(*graphicsLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*variantLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*writeLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*writeShader).has_value());
    REQUIRE(device.destroyShaderModule(*variantShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
}

TEST_CASE("restored D3D12 pipeline caches stop reading the caller's initial data when creation returns",
          "[size:medium][rhi][pipeline][cache][hardware]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    if (!created)
    {
        skipWithoutHardwareAdapter(created.error());
    }
    auto device = std::move(*created);
    const auto adapter = rachis::test::requireAdapterInfo(device);
    CAPTURE(adapter.name);
    if (adapter.type == rachis::AdapterType::software)
    {
        SKIP("The selected adapter is WARP, whose pipeline library never reads the blob after creation.");
    }
    if (!rachis::test::requireCapabilities(device).pipelineCaches)
    {
        SKIP("The selected hardware adapter does not expose pipeline caches.");
    }

    const auto bytecode = loadShader(RACHIS_TEST_COMPUTE_SHADER_PATH);
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(shader.has_value());
    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstants,
    });
    REQUIRE(pipelineLayout.has_value());
    const rachis::ComputePipelineDesc hitDesc{
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cacheKey = cacheKey,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    };

    auto sourceCache = device.createPipelineCache({.label = "Source pipeline cache"});
    REQUIRE(sourceCache.has_value());
    auto sourcePipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cache = *sourceCache,
        .cacheKey = cacheKey,
    });
    REQUIRE(sourcePipeline.has_value());
    const auto sourceData = device.pipelineCacheData(*sourceCache);
    REQUIRE(sourceData.has_value());
    REQUIRE_FALSE(sourceData->empty());
    REQUIRE(device.destroyPipelineCache(*sourceCache).has_value());

    const std::unique_ptr<void, VirtualPagesDeleter> pages{
        VirtualAlloc(nullptr, sourceData->size(), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE),
    };
    REQUIRE(pages != nullptr);
    std::memcpy(pages.get(), sourceData->data(), sourceData->size());
    auto restoredCache = device.createPipelineCache({
        .initialData = std::span(static_cast<const std::byte*>(pages.get()), sourceData->size()),
        .label = "Pipeline cache restored from guarded pages",
    });
    REQUIRE(restoredCache.has_value());
    DWORD previousProtection = 0;
    REQUIRE(VirtualProtect(pages.get(), sourceData->size(), PAGE_NOACCESS, &previousProtection) != 0);

    auto restoredDesc = hitDesc;
    restoredDesc.cache = *restoredCache;
    auto restoredPipeline = device.createComputePipeline(restoredDesc);
    REQUIRE(restoredPipeline.has_value());
    const auto restoredData = device.pipelineCacheData(*restoredCache);
    REQUIRE(restoredData.has_value());
    REQUIRE_FALSE(restoredData->empty());
    REQUIRE(device.destroyPipelineCache(*restoredCache).has_value());

    auto reloadedCache = device.createPipelineCache({.initialData = *restoredData});
    REQUIRE(reloadedCache.has_value());
    auto reloadedDesc = hitDesc;
    reloadedDesc.cache = *reloadedCache;
    auto reloadedPipeline = device.createComputePipeline(reloadedDesc);
    REQUIRE(reloadedPipeline.has_value());
    REQUIRE(device.destroyPipelineCache(*reloadedCache).has_value());
    REQUIRE(device.waitIdle().has_value());

    REQUIRE(device.destroyPipeline(*sourcePipeline).has_value());
    REQUIRE(device.destroyPipeline(*restoredPipeline).has_value());
    REQUIRE(device.destroyPipeline(*reloadedPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
}

TEST_CASE("D3D12 pipeline caches stay reusable after recording a cache entry fails",
          "[size:medium][rhi][pipeline][cache][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).pipelineCaches);

    const auto computeBytecode = loadShader(RACHIS_TEST_COMPUTE_SHADER_PATH);
    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(computeBytecode.empty());
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto computeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = computeBytecode,
        .entryPoint = "main",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(computeShader.has_value());
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array computePushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto computeLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = computePushConstants,
    });
    const std::array graphicsPushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto graphicsLayout = device.createPipelineLayout({.pushConstantRanges = graphicsPushConstants});
    REQUIRE(computeLayout.has_value());
    REQUIRE(graphicsLayout.has_value());

    auto cache = device.createPipelineCache({.label = "Entry-record failure pipeline cache"});
    REQUIRE(cache.has_value());
    const rachis::ComputePipelineDesc computeDesc{
        .layout = *computeLayout,
        .computeShader = *computeShader,
        .cache = *cache,
        .cacheKey = cacheKey,
    };
    auto computeHitDesc = computeDesc;
    computeHitDesc.cachePolicy = rachis::PipelineCachePolicy::requireHit;
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::GraphicsPipelineDesc graphicsDesc{
        .layout = *graphicsLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .colorTargets = colorTargets,
        .cache = *cache,
        .cacheKey = graphicsCacheKey,
    };
    auto graphicsHitDesc = graphicsDesc;
    graphicsHitDesc.cachePolicy = rachis::PipelineCachePolicy::requireHit;

    {
        const ArmTestHook failure(rachis::d3d12::test::nextPipelineCacheEntryRecordFailure());
        const auto failed = device.createComputePipeline(computeDesc);
        CHECK(failure.consumed());
        REQUIRE_FALSE(failed.has_value());
        CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    }
    {
        const ArmTestHook failure(rachis::d3d12::test::nextPipelineCacheEntryRecordFailure());
        const auto failed = device.createGraphicsPipeline(graphicsDesc);
        CHECK(failure.consumed());
        REQUIRE_FALSE(failed.has_value());
        CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    }

    const auto computeMiss = device.createComputePipeline(computeHitDesc);
    REQUIRE_FALSE(computeMiss.has_value());
    CHECK(computeMiss.error().code == rachis::ErrorCode::pipelineCacheMiss);
    const auto graphicsMiss = device.createGraphicsPipeline(graphicsHitDesc);
    REQUIRE_FALSE(graphicsMiss.has_value());
    CHECK(graphicsMiss.error().code == rachis::ErrorCode::pipelineCacheMiss);

    auto computePipeline = device.createComputePipeline(computeDesc);
    if (!computePipeline)
    {
        UNSCOPED_INFO(computePipeline.error().message);
    }
    REQUIRE(computePipeline.has_value());
    auto graphicsPipeline = device.createGraphicsPipeline(graphicsDesc);
    if (!graphicsPipeline)
    {
        UNSCOPED_INFO(graphicsPipeline.error().message);
    }
    REQUIRE(graphicsPipeline.has_value());
    auto computeHit = device.createComputePipeline(computeHitDesc);
    REQUIRE(computeHit.has_value());
    auto graphicsHit = device.createGraphicsPipeline(graphicsHitDesc);
    REQUIRE(graphicsHit.has_value());

    const auto cacheData = device.pipelineCacheData(*cache);
    REQUIRE(cacheData.has_value());
    REQUIRE(device.destroyPipelineCache(*cache).has_value());
    auto restoredCache = device.createPipelineCache({.initialData = *cacheData});
    REQUIRE(restoredCache.has_value());
    computeHitDesc.cache = *restoredCache;
    graphicsHitDesc.cache = *restoredCache;
    auto restoredCompute = device.createComputePipeline(computeHitDesc);
    REQUIRE(restoredCompute.has_value());
    auto restoredGraphics = device.createGraphicsPipeline(graphicsHitDesc);
    REQUIRE(restoredGraphics.has_value());
    REQUIRE(device.destroyPipelineCache(*restoredCache).has_value());

    REQUIRE(device.destroyPipeline(*computePipeline).has_value());
    REQUIRE(device.destroyPipeline(*graphicsPipeline).has_value());
    REQUIRE(device.destroyPipeline(*computeHit).has_value());
    REQUIRE(device.destroyPipeline(*graphicsHit).has_value());
    REQUIRE(device.destroyPipeline(*restoredCompute).has_value());
    REQUIRE(device.destroyPipeline(*restoredGraphics).has_value());
    REQUIRE(device.destroyPipelineLayout(*computeLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*graphicsLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*computeShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
}

TEST_CASE("D3D12 pipeline caches that cannot load state streams reject stream pipelines without storing them",
          "[size:medium][rhi][pipeline][cache][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    REQUIRE(capabilities.pipelineCaches);
    REQUIRE(capabilities.depthBoundsTest);

    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    const std::array pushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto layout = device.createPipelineLayout({.pushConstantRanges = pushConstants});
    REQUIRE(layout.has_value());

    auto cache = [&] {
        const ArmTestHook withoutLibrary1(rachis::d3d12::test::nextPipelineCacheWithoutLibrary1());
        auto result = device.createPipelineCache({.label = "Pipeline cache without stream loading"});
        CHECK(withoutLibrary1.consumed());
        return result;
    }();
    REQUIRE(cache.has_value());

    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::GraphicsPipelineDesc plainDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .depthStencil =
            {
                .format = rachis::Format::d32Float,
                .depthTestEnabled = true,
                .depthCompare = rachis::CompareOperation::always,
            },
        .cache = *cache,
        .cacheKey = graphicsCacheKey,
    };
    auto plainHitDesc = plainDesc;
    plainHitDesc.cachePolicy = rachis::PipelineCachePolicy::requireHit;
    auto depthBoundsDesc = plainDesc;
    depthBoundsDesc.depthStencil.depthBoundsTestEnabled = true;
    depthBoundsDesc.cacheKey = depthBoundsCacheKey;
    auto depthBoundsHitDesc = depthBoundsDesc;
    depthBoundsHitDesc.cachePolicy = rachis::PipelineCachePolicy::requireHit;
    auto uncachedDepthBoundsDesc = depthBoundsDesc;
    uncachedDepthBoundsDesc.cache = {};
    uncachedDepthBoundsDesc.cacheKey = {};

    for (std::uint32_t attempt = 0; attempt < 2; ++attempt)
    {
        CAPTURE(attempt);
        const auto populated = device.createGraphicsPipeline(depthBoundsDesc);
        REQUIRE_FALSE(populated.has_value());
        CHECK(populated.error().code == rachis::ErrorCode::unsupported);
        const auto hit = device.createGraphicsPipeline(depthBoundsHitDesc);
        REQUIRE_FALSE(hit.has_value());
        CHECK(hit.error().code == rachis::ErrorCode::unsupported);
    }
    auto uncachedDepthBounds = device.createGraphicsPipeline(uncachedDepthBoundsDesc);
    REQUIRE(uncachedDepthBounds.has_value());

    auto plainPipeline = device.createGraphicsPipeline(plainDesc);
    REQUIRE(plainPipeline.has_value());
    auto plainHit = device.createGraphicsPipeline(plainHitDesc);
    REQUIRE(plainHit.has_value());

    const auto cacheData = device.pipelineCacheData(*cache);
    REQUIRE(cacheData.has_value());
    REQUIRE(device.destroyPipelineCache(*cache).has_value());
    auto restoredCache = device.createPipelineCache({.initialData = *cacheData});
    REQUIRE(restoredCache.has_value());
    depthBoundsHitDesc.cache = *restoredCache;
    plainHitDesc.cache = *restoredCache;
    const auto restoredDepthBounds = device.createGraphicsPipeline(depthBoundsHitDesc);
    REQUIRE_FALSE(restoredDepthBounds.has_value());
    CHECK(restoredDepthBounds.error().code == rachis::ErrorCode::pipelineCacheMiss);
    auto restoredPlain = device.createGraphicsPipeline(plainHitDesc);
    REQUIRE(restoredPlain.has_value());
    REQUIRE(device.destroyPipelineCache(*restoredCache).has_value());

    REQUIRE(device.destroyPipeline(*uncachedDepthBounds).has_value());
    REQUIRE(device.destroyPipeline(*plainPipeline).has_value());
    REQUIRE(device.destroyPipeline(*plainHit).has_value());
    REQUIRE(device.destroyPipeline(*restoredPlain).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
}
