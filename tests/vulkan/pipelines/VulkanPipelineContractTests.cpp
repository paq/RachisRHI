#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/VulkanPipelines.h"
#include "Vulkan/VulkanTestHooks.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <volk.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{

    constexpr rachis::PipelineCacheKey computeCacheKey{0x636f6d7075746521ULL, 0x6f6f6d2d63726561ULL};
    constexpr rachis::PipelineCacheKey graphicsCacheKey{0x6772617068696321ULL, 0x6f6f6d2d63726561ULL};
    constexpr rachis::PipelineCacheKey staticUniformCacheKey{0x7374617469637562ULL, 0x69736f6c61746564ULL};
    constexpr rachis::PipelineCacheKey dynamicUniformCacheKey{0x64796e616d696375ULL, 0x69736f6c61746564ULL};
    constexpr rachis::PipelineCacheKey staticGraphicsCacheKey{0x7374617469636772ULL, 0x69736f6c61746564ULL};
    constexpr rachis::PipelineCacheKey dynamicGraphicsCacheKey{0x64796e616d696367ULL, 0x69736f6c61746564ULL};
    constexpr std::uint64_t uniformSlotSize = 256;

    using Float4 = std::array<float, 4>;

    class ForcedIsolatedDynamicBufferShaders final
    {
    public:
        explicit ForcedIsolatedDynamicBufferShaders(bool enabled) noexcept : enabled_(enabled)
        {
            if (enabled_)
            {
                rachis::vulkan::test::forceIsolatedDynamicBufferShaders().store(true, std::memory_order_relaxed);
            }
        }

        ~ForcedIsolatedDynamicBufferShaders()
        {
            if (enabled_)
            {
                rachis::vulkan::test::forceIsolatedDynamicBufferShaders().store(false, std::memory_order_relaxed);
            }
        }

        ForcedIsolatedDynamicBufferShaders(const ForcedIsolatedDynamicBufferShaders&) = delete;
        ForcedIsolatedDynamicBufferShaders& operator=(const ForcedIsolatedDynamicBufferShaders&) = delete;
        ForcedIsolatedDynamicBufferShaders(ForcedIsolatedDynamicBufferShaders&&) = delete;
        ForcedIsolatedDynamicBufferShaders& operator=(ForcedIsolatedDynamicBufferShaders&&) = delete;

    private:
        bool enabled_ = false;
    };

    class PipelineCreationCapture final
    {
    public:
        PipelineCreationCapture() noexcept
        {
            rachis::vulkan::captureVulkanPipelineCreations(&creations_);
        }

        ~PipelineCreationCapture()
        {
            rachis::vulkan::captureVulkanPipelineCreations(nullptr);
        }

        PipelineCreationCapture(const PipelineCreationCapture&) = delete;
        PipelineCreationCapture& operator=(const PipelineCreationCapture&) = delete;
        PipelineCreationCapture(PipelineCreationCapture&&) = delete;
        PipelineCreationCapture& operator=(PipelineCreationCapture&&) = delete;

        [[nodiscard]] std::vector<rachis::vulkan::VulkanCapturedPipelineCreation> take()
        {
            return std::exchange(creations_, {});
        }

    private:
        std::vector<rachis::vulkan::VulkanCapturedPipelineCreation> creations_;
    };

    [[nodiscard]] constexpr bool isolatesDynamicBufferShaders(bool forced) noexcept
    {
#ifdef __APPLE__
        static_cast<void>(forced);
        return true;
#else
        return forced;
#endif
    }

    constexpr std::array uniformRangeRequirements{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };

    [[nodiscard]] rachis::Result<rachis::ShaderModuleHandle> createUniformRangeShader(
        rachis::Device& device, std::span<const std::byte> bytecode)
    {
        return device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .label = "Vulkan uniform range shader",
            .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, uniformRangeRequirements),
        });
    }

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] rachis::Result<rachis::Device> createVulkanDevice(bool forceIsolation)
    {
        const ForcedIsolatedDynamicBufferShaders force(forceIsolation);
        return rachis::createDevice(vulkanDesc());
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

TEST_CASE("Vulkan pipeline creation rejects malformed state and unsupported stages",
          "[size:medium][rhi][vulkan][pipeline]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities);
    const auto cube = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapCube.cs.spv");
    const auto triangleVs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.vs.spv");
    const auto trianglePs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.ps.spv");
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());

    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto computeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = cubeComputeInterface(bindings, push),
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
    REQUIRE(computeShader);
    REQUIRE(vertex);
    REQUIRE(pixel);

    auto emptyLayout = device.createPipelineLayout({});
    REQUIRE(emptyLayout);
    CHECK_FALSE(device
                    .createComputePipeline({
                        .layout = *emptyLayout,
                        .computeShader = *vertex,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createComputePipeline({
                        .layout = *emptyLayout,
                        .computeShader = *computeShader,
                    })
                    .has_value());

    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    CHECK_FALSE(device
                    .createGraphicsPipeline({
                        .layout = *emptyLayout,
                        .vertexShader = *vertex,
                        .pixelShader = *pixel,
                    })
                    .has_value());

    auto geometryRejected = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .geometryShader = *vertex,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(geometryRejected);
    CHECK(geometryRejected.error().code ==
          (capabilities->geometryShaders ? rachis::ErrorCode::invalidArgument : rachis::ErrorCode::unsupported));

    auto meshRejected = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .meshShader = *computeShader,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(meshRejected);
    CHECK(meshRejected.error().code ==
          (capabilities->meshShaders ? rachis::ErrorCode::invalidArgument : rachis::ErrorCode::unsupported));
}

TEST_CASE("Vulkan pipeline compilation is synchronous sugar over create",
          "[size:medium][rhi][vulkan][pipeline][compilation]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto cube = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapCube.cs.spv");
    const auto triangleVs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.vs.spv");
    const auto trianglePs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.ps.spv");
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());

    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto computeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    REQUIRE(computeShader);
    const std::array computeEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto computeBindingLayout = device.createBindingLayout({.entries = computeEntries});
    REQUIRE(computeBindingLayout);
    const std::array computeSets{*computeBindingLayout};
    const std::array computePush{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    std::string computeLabel = "Vulkan async compute";
    auto computeLayout = device.createPipelineLayout({
        .bindingLayouts = computeSets,
        .pushConstantRanges = computePush,
        .label = "Vulkan async compute layout",
    });
    REQUIRE(computeLayout);
    const rachis::ComputePipelineDesc computeDesc{
        .layout = *computeLayout,
        .computeShader = *computeShader,
        .label = computeLabel,
    };
    auto compilation = device.beginComputePipelineCompilation(computeDesc);
    auto abandoned = device.beginComputePipelineCompilation(computeDesc);
    REQUIRE(compilation);
    REQUIRE(abandoned);
    const auto observed = device.pipelineCompilationState(*compilation);
    REQUIRE(observed);
    CHECK((*observed == rachis::PipelineCompilationState::pending ||
           *observed == rachis::PipelineCompilationState::succeeded));
    REQUIRE(device.destroyPipelineCompilation(*abandoned));
    CHECK_FALSE(device.destroyPipelineCompilation(*abandoned));

    computeLabel.assign("caller storage was changed");
    REQUIRE(device.destroyPipelineLayout(*computeLayout));
    REQUIRE(device.destroyShaderModule(*computeShader));
    REQUIRE(device.destroyBindingLayout(*computeBindingLayout));

    auto computePipeline = device.waitPipelineCompilation(*compilation);
    REQUIRE(computePipeline);
    CHECK_FALSE(device.pipelineCompilationState(*compilation).has_value());
    CHECK_FALSE(device.waitPipelineCompilation(*compilation).has_value());
    REQUIRE(device.destroyPipeline(*computePipeline));

    auto failed = device.beginComputePipelineCompilation({
        .layout = {},
        .computeShader = {},
    });
    REQUIRE(failed);
    CHECK_FALSE(device.waitPipelineCompilation(*failed).has_value());
    CHECK_FALSE(device.pipelineCompilationState(*failed).has_value());

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
    auto graphicsLayout = device.createPipelineLayout({});
    REQUIRE(vertex);
    REQUIRE(pixel);
    REQUIRE(graphicsLayout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto graphicsCompilation = device.beginGraphicsPipelineCompilation({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .label = "Vulkan async graphics",
    });
    REQUIRE(graphicsCompilation);
    auto graphics = device.waitPipelineCompilation(*graphicsCompilation);
    REQUIRE(graphics);
    REQUIRE(device.destroyPipeline(*graphics));
}

TEST_CASE("Vulkan pipeline creation classifies native create failures by their VkResult",
          "[size:medium][rhi][vulkan][pipeline]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto cube = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapCube.cs.spv");
    const auto triangleVs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.vs.spv");
    const auto trianglePs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.ps.spv");
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());

    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto computeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = cubeComputeInterface(bindings, push),
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
    REQUIRE(computeShader);
    REQUIRE(vertex);
    REQUIRE(pixel);
    const std::array computeEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto computeBindingLayout = device.createBindingLayout({.entries = computeEntries});
    REQUIRE(computeBindingLayout);
    const std::array computeSets{*computeBindingLayout};
    const std::array computePush{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    auto computeLayout = device.createPipelineLayout({
        .bindingLayouts = computeSets,
        .pushConstantRanges = computePush,
    });
    auto graphicsLayout = device.createPipelineLayout({});
    REQUIRE(computeLayout);
    REQUIRE(graphicsLayout);
    auto cache = device.createPipelineCache({.label = "Vulkan create-failure pipeline cache"});
    REQUIRE(cache);

    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const auto computeDesc = [&](rachis::PipelineCachePolicy policy) {
        return rachis::ComputePipelineDesc{
            .layout = *computeLayout,
            .computeShader = *computeShader,
            .label = "Vulkan create-failure compute",
            .cache = *cache,
            .cacheKey = computeCacheKey,
            .cachePolicy = policy,
        };
    };
    const auto graphicsDesc = [&](rachis::PipelineCachePolicy policy) {
        return rachis::GraphicsPipelineDesc{
            .layout = *graphicsLayout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .colorTargets = colorTargets,
            .label = "Vulkan create-failure graphics",
            .cache = *cache,
            .cacheKey = graphicsCacheKey,
            .cachePolicy = policy,
        };
    };

    const std::array failures{
        std::pair{VK_ERROR_OUT_OF_DEVICE_MEMORY, rachis::ErrorCode::outOfMemory},
        std::pair{VK_ERROR_OUT_OF_HOST_MEMORY, rachis::ErrorCode::outOfMemory},
        std::pair{VK_ERROR_UNKNOWN, rachis::ErrorCode::nativeFailure},
    };
    for (const auto& [result, code] : failures)
    {
        CAPTURE(result);
        REQUIRE(rachis::vulkan::test::injectCreateResult(result));
        auto compute = device.createComputePipeline(computeDesc(rachis::PipelineCachePolicy::populate));
        auto graphics = device.createGraphicsPipeline(graphicsDesc(rachis::PipelineCachePolicy::populate));
        auto failedCache = device.createPipelineCache({.label = "Vulkan create-failure rejected cache"});
        REQUIRE(rachis::vulkan::test::injectCreateResult(VK_SUCCESS));

        REQUIRE_FALSE(failedCache.has_value());
        CHECK(failedCache.error().code == code);
        CHECK(failedCache.error().nativeCode == result);
        CHECK(failedCache.error().message.contains("vkCreatePipelineCache"));
        REQUIRE_FALSE(compute.has_value());
        CHECK(compute.error().code == code);
        CHECK(compute.error().nativeCode == result);
        CHECK(compute.error().message.contains("vkCreateComputePipelines"));
        REQUIRE_FALSE(graphics.has_value());
        CHECK(graphics.error().code == code);
        CHECK(graphics.error().nativeCode == result);
        CHECK(graphics.error().message.contains("vkCreateGraphicsPipelines"));
        CHECK(device.operationalState() == rachis::DeviceOperationalState::active);
        CHECK_FALSE(device.deviceLossError().has_value());
    }

    auto computeMiss = device.createComputePipeline(computeDesc(rachis::PipelineCachePolicy::requireHit));
    REQUIRE_FALSE(computeMiss.has_value());
    CHECK(computeMiss.error().code == rachis::ErrorCode::pipelineCacheMiss);
    auto graphicsMiss = device.createGraphicsPipeline(graphicsDesc(rachis::PipelineCachePolicy::requireHit));
    REQUIRE_FALSE(graphicsMiss.has_value());
    CHECK(graphicsMiss.error().code == rachis::ErrorCode::pipelineCacheMiss);

    auto compute = device.createComputePipeline(computeDesc(rachis::PipelineCachePolicy::populate));
    REQUIRE(compute);
    auto graphics = device.createGraphicsPipeline(graphicsDesc(rachis::PipelineCachePolicy::populate));
    REQUIRE(graphics);

    REQUIRE(device.destroyPipeline(*graphics));
    REQUIRE(device.destroyPipeline(*compute));
    REQUIRE(device.destroyPipelineCache(*cache));
    REQUIRE(device.destroyPipelineLayout(*graphicsLayout));
    REQUIRE(device.destroyPipelineLayout(*computeLayout));
    REQUIRE(device.destroyBindingLayout(*computeBindingLayout));
    REQUIRE(device.destroyShaderModule(*pixel));
    REQUIRE(device.destroyShaderModule(*vertex));
    REQUIRE(device.destroyShaderModule(*computeShader));
}

TEST_CASE("Vulkan pipelines with dynamic buffers get private shader modules on portability-subset devices",
          "[size:medium][rhi][vulkan][pipeline][portability]")
{
    const bool forced = GENERATE(false, true);
    CAPTURE(forced);
    auto created = createVulkanDevice(forced);
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const bool isolated = isolatesDynamicBufferShaders(forced);

    const auto uniformRange = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "UniformBufferRange.spv");
    const auto triangleVs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.vs.spv");
    const auto trianglePs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.ps.spv");
    REQUIRE_FALSE(uniformRange.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());
    auto compute = createUniformRangeShader(device, uniformRange);
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
    REQUIRE(compute);
    REQUIRE(vertex);
    REQUIRE(pixel);

    const std::array staticEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    const std::array dynamicEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    const std::array graphicsEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::pixel, 1},
    };
    auto staticBindings = device.createBindingLayout({.entries = staticEntries});
    auto dynamicBindings = device.createBindingLayout({.entries = dynamicEntries});
    auto graphicsBindings = device.createBindingLayout({.entries = graphicsEntries});
    REQUIRE(staticBindings);
    REQUIRE(dynamicBindings);
    REQUIRE(graphicsBindings);
    const std::array staticSets{*staticBindings};
    const std::array dynamicSets{*dynamicBindings};
    const std::array graphicsSets{*graphicsBindings};
    auto staticLayout = device.createPipelineLayout({.bindingLayouts = staticSets});
    auto dynamicLayout = device.createPipelineLayout({.bindingLayouts = dynamicSets});
    auto graphicsLayout = device.createPipelineLayout({.bindingLayouts = graphicsSets});
    auto emptyLayout = device.createPipelineLayout({});
    auto cache = device.createPipelineCache({.label = "Vulkan private shader module cache"});
    REQUIRE(staticLayout);
    REQUIRE(dynamicLayout);
    REQUIRE(graphicsLayout);
    REQUIRE(emptyLayout);
    REQUIRE(cache);

    const auto computeDesc =
        [&](rachis::PipelineLayoutHandle layout, rachis::PipelineCacheKey key, rachis::PipelineCachePolicy policy) {
            return rachis::ComputePipelineDesc{
                .layout = layout,
                .computeShader = *compute,
                .label = "Vulkan private shader module compute",
                .cache = *cache,
                .cacheKey = key,
                .cachePolicy = policy,
            };
        };
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const auto graphicsDesc = [&](rachis::PipelineLayoutHandle layout, rachis::PipelineCacheKey key) {
        return rachis::GraphicsPipelineDesc{
            .layout = layout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .colorTargets = colorTargets,
            .label = "Vulkan private shader module graphics",
            .cache = *cache,
            .cacheKey = key,
        };
    };

    std::vector<rachis::vulkan::VulkanCapturedPipelineCreation> captured;
    std::vector<rachis::PipelineHandle> pipelines;
    {
        PipelineCreationCapture capture;
        auto staticCompute = device.createComputePipeline(
            computeDesc(*staticLayout, staticUniformCacheKey, rachis::PipelineCachePolicy::populate));
        auto dynamicCompute = device.createComputePipeline(
            computeDesc(*dynamicLayout, dynamicUniformCacheKey, rachis::PipelineCachePolicy::populate));
        auto cachedDynamicCompute = device.createComputePipeline(
            computeDesc(*dynamicLayout, dynamicUniformCacheKey, rachis::PipelineCachePolicy::requireHit));
        auto staticGraphics = device.createGraphicsPipeline(graphicsDesc(*emptyLayout, staticGraphicsCacheKey));
        auto dynamicGraphics = device.createGraphicsPipeline(graphicsDesc(*graphicsLayout, dynamicGraphicsCacheKey));
        captured = capture.take();
        REQUIRE(staticCompute);
        REQUIRE(dynamicCompute);
        REQUIRE(cachedDynamicCompute);
        REQUIRE(staticGraphics);
        REQUIRE(dynamicGraphics);
        pipelines = {*staticCompute, *dynamicCompute, *cachedDynamicCompute, *staticGraphics, *dynamicGraphics};
    }

    REQUIRE(captured.size() == 5);
    REQUIRE(captured[0].shaderModules.size() == 1);
    REQUIRE(captured[1].shaderModules.size() == 1);
    REQUIRE(captured[2].shaderModules.size() == 1);
    REQUIRE(captured[3].shaderModules.size() == 2);
    REQUIRE(captured[4].shaderModules.size() == 2);
    CHECK(captured[0].cache != VK_NULL_HANDLE);
    CHECK(captured[3].cache == captured[0].cache);
    for (const auto index : {std::size_t{1}, std::size_t{2}})
    {
        CAPTURE(index);
        CHECK((captured[index].shaderModules[0] != captured[0].shaderModules[0]) == isolated);
        CHECK(captured[index].cache == (isolated ? VK_NULL_HANDLE : captured[0].cache));
    }
    CHECK((captured[4].shaderModules[0] != captured[3].shaderModules[0]) == isolated);
    CHECK((captured[4].shaderModules[1] != captured[3].shaderModules[1]) == isolated);
    CHECK(captured[4].cache == (isolated ? VK_NULL_HANDLE : captured[0].cache));

    for (const auto pipeline : pipelines)
    {
        REQUIRE(device.destroyPipeline(pipeline));
    }
    REQUIRE(device.destroyPipelineCache(*cache));
    REQUIRE(device.destroyPipelineLayout(*emptyLayout));
    REQUIRE(device.destroyPipelineLayout(*graphicsLayout));
    REQUIRE(device.destroyPipelineLayout(*dynamicLayout));
    REQUIRE(device.destroyPipelineLayout(*staticLayout));
    REQUIRE(device.destroyBindingLayout(*graphicsBindings));
    REQUIRE(device.destroyBindingLayout(*dynamicBindings));
    REQUIRE(device.destroyBindingLayout(*staticBindings));
    REQUIRE(device.destroyShaderModule(*pixel));
    REQUIRE(device.destroyShaderModule(*vertex));
    REQUIRE(device.destroyShaderModule(*compute));
    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

TEST_CASE("Vulkan dynamic uniform-buffer pipelines apply their offsets after a static pipeline from the same SPIR-V",
          "[size:medium][rhi][vulkan][pipeline][binding][portability]")
{
    const bool forced = GENERATE(false, true);
    const bool throughCache = GENERATE(false, true);
    CAPTURE(forced, throughCache);
    auto created = createVulkanDevice(forced);
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto bytecode = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "UniformBufferRange.spv");
    REQUIRE_FALSE(bytecode.empty());
    auto staticShader = createUniformRangeShader(device, bytecode);
    REQUIRE(staticShader);
    auto dynamicShader = throughCache ? createUniformRangeShader(device, bytecode) : staticShader;
    REQUIRE(dynamicShader);
    rachis::PipelineCacheHandle cache = {};
    if (throughCache)
    {
        auto createdCache = device.createPipelineCache({.label = "Vulkan shared uniform range cache"});
        REQUIRE(createdCache);
        cache = *createdCache;
    }

    auto source = device.createBuffer({
        .size = 1024,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan uniform range source",
    });
    auto output = device.createBuffer({
        .size = uniformSlotSize * 2,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Vulkan uniform range output",
    });
    auto readback = device.createBuffer({
        .size = uniformSlotSize * 2,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan uniform range readback",
    });
    REQUIRE(source);
    REQUIRE(output);
    REQUIRE(readback);
    const Float4 first{9.0F, 10.0F, 11.0F, 12.0F};
    const Float4 second{13.0F, 14.0F, 15.0F, 16.0F};
    REQUIRE(device.writeBuffer(*source, 0, std::as_bytes(std::span{first})));
    REQUIRE(device.writeBuffer(*source, uniformSlotSize, std::as_bytes(std::span{second})));

    const std::array staticEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    const std::array dynamicEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto staticBindings = device.createBindingLayout({.entries = staticEntries});
    auto dynamicBindings = device.createBindingLayout({.entries = dynamicEntries});
    REQUIRE(staticBindings);
    REQUIRE(dynamicBindings);
    const std::array poolSizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 1},
        rachis::BindingPoolSize{rachis::BindingType::dynamicUniformBuffer, 1},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 2},
    };
    auto pool = device.createBindingPool({.maxBindingSets = 2, .sizes = poolSizes});
    REQUIRE(pool);
    const auto createSet = [&](rachis::BindingLayoutHandle layout, std::uint64_t slot) {
        const std::array writes{
            rachis::BindingWrite{
                .binding = 0,
                .resource = rachis::BufferBinding{.buffer = *source, .offset = 0, .size = 64},
            },
            rachis::BindingWrite{
                .binding = 1,
                .resource =
                    rachis::BufferBinding{
                        .buffer = *output,
                        .offset = uniformSlotSize * slot,
                        .size = sizeof(Float4),
                        .structureByteStride = sizeof(Float4),
                    },
            },
        };
        return device.createBindingSet({.layout = layout, .writes = writes, .pool = *pool});
    };
    auto staticSet = createSet(*staticBindings, 0);
    auto dynamicSet = createSet(*dynamicBindings, 1);
    REQUIRE(staticSet);
    REQUIRE(dynamicSet);

    const std::array staticSets{*staticBindings};
    const std::array dynamicSets{*dynamicBindings};
    auto staticLayout = device.createPipelineLayout({.bindingLayouts = staticSets});
    auto dynamicLayout = device.createPipelineLayout({.bindingLayouts = dynamicSets});
    REQUIRE(staticLayout);
    REQUIRE(dynamicLayout);
    auto staticPipeline = device.createComputePipeline({
        .layout = *staticLayout,
        .computeShader = *staticShader,
        .label = "Vulkan static uniform range pipeline",
        .cache = cache,
        .cacheKey = throughCache ? staticUniformCacheKey : rachis::PipelineCacheKey{},
    });
    REQUIRE(staticPipeline);
    auto dynamicPipeline = device.createComputePipeline({
        .layout = *dynamicLayout,
        .computeShader = *dynamicShader,
        .label = "Vulkan dynamic uniform range pipeline",
        .cache = cache,
        .cacheKey = throughCache ? dynamicUniformCacheKey : rachis::PipelineCacheKey{},
    });
    REQUIRE(dynamicPipeline);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.bufferBarrier(*command,
                                 {
                                     .buffer = *output,
                                     .stagesBefore = rachis::PipelineStage::all,
                                     .stagesAfter = rachis::PipelineStage::computeShader,
                                     .accessBefore = rachis::Access::common,
                                     .accessAfter = rachis::Access::unorderedAccess,
                                 }));
    REQUIRE(device.bindComputePipeline(*command, *staticPipeline));
    REQUIRE(device.bindComputeBindingSet(*command, 0, *staticSet));
    REQUIRE(device.dispatch(*command, 1));
    const std::array offsets{static_cast<std::uint32_t>(uniformSlotSize)};
    REQUIRE(device.bindComputePipeline(*command, *dynamicPipeline));
    REQUIRE(device.bindComputeBindingSet(*command, 0, *dynamicSet, offsets));
    REQUIRE(device.dispatch(*command, 1));
    REQUIRE(device.bufferBarrier(*command,
                                 {
                                     .buffer = *output,
                                     .stagesBefore = rachis::PipelineStage::computeShader,
                                     .stagesAfter = rachis::PipelineStage::copy,
                                     .accessBefore = rachis::Access::unorderedAccess,
                                     .accessAfter = rachis::Access::copySource,
                                 }));
    REQUIRE(device.copyBuffer(*command,
                              {
                                  .source = *output,
                                  .destination = *readback,
                                  .size = uniformSlotSize * 2,
                              }));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));

    std::vector<float> values(static_cast<std::size_t>(uniformSlotSize * 2 / sizeof(float)));
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span{values})));
    const auto slot = [&values](std::size_t index) {
        Float4 copied{};
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(index * uniformSlotSize / sizeof(float)),
                    copied.size(),
                    copied.begin());
        return copied;
    };
    CHECK(slot(0) == first);
    CHECK(slot(1) == second);

    REQUIRE(device.destroyPipeline(*dynamicPipeline));
    REQUIRE(device.destroyPipeline(*staticPipeline));
    REQUIRE(device.destroyPipelineLayout(*dynamicLayout));
    REQUIRE(device.destroyPipelineLayout(*staticLayout));
    REQUIRE(device.destroyBindingSet(*dynamicSet));
    REQUIRE(device.destroyBindingSet(*staticSet));
    REQUIRE(device.destroyBindingPool(*pool));
    REQUIRE(device.destroyBindingLayout(*dynamicBindings));
    REQUIRE(device.destroyBindingLayout(*staticBindings));
    REQUIRE(device.destroyBuffer(*readback));
    REQUIRE(device.destroyBuffer(*output));
    REQUIRE(device.destroyBuffer(*source));
    if (throughCache)
    {
        REQUIRE(device.destroyPipelineCache(cache));
        REQUIRE(device.destroyShaderModule(*dynamicShader));
    }
    REQUIRE(device.destroyShaderModule(*staticShader));
    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

TEST_CASE("Vulkan graphics pipelines accept the layout derived from their shader interfaces",
          "[size:medium][rhi][vulkan][pipeline][pipeline-layout]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto vertexBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanHeadlessParity.textured.vs.spv");
    const auto pixelBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanHeadlessParity.textured.ps.spv");
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    static constexpr std::array pixelBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampler, 1},
    };
    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "texturedVertex",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "texturedPixel",
        .shaderInterface = rachis::test::shaderInterface(pixelBindings),
    });
    REQUIRE(vertex);
    REQUIRE(pixel);

    auto vertexInterface = device.shaderModuleInterface(*vertex);
    auto pixelInterface = device.shaderModuleInterface(*pixel);
    REQUIRE(vertexInterface);
    REQUIRE(pixelInterface);
    const std::array interfaces{std::move(*vertexInterface), std::move(*pixelInterface)};
    const auto derived = rachis::derivePipelineLayout(interfaces);
    REQUIRE(derived);
    REQUIRE(derived->bindingSets.size() == 1);

    std::vector<rachis::BindingLayoutHandle> bindingLayouts;
    for (const auto& entries : derived->bindingSets)
    {
        auto bindingLayout = device.createBindingLayout({.entries = entries, .label = "Derived binding layout"});
        REQUIRE(bindingLayout);
        bindingLayouts.push_back(*bindingLayout);
    }
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = derived->pushConstantRanges,
        .label = "Derived pipeline layout",
    });
    REQUIRE(pipelineLayout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Pipeline built from a derived layout",
    });
    REQUIRE(pipeline);
    CHECK(device.destroyPipeline(*pipeline));
    CHECK(device.destroyPipelineLayout(*pipelineLayout));
    for (const auto bindingLayout : bindingLayouts)
    {
        CHECK(device.destroyBindingLayout(bindingLayout));
    }
}
