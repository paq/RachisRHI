#include "Rachis/Device.h"

#if defined(_WINDOWS_) || defined(VK_VERSION_1_0) || defined(__OBJC__)
#error Public Rachis headers must remain free of native platform and graphics SDK headers.
#endif

#include "PortableDrawIdPixel.h"
#include "PortableDrawIdPixelSlang.h"
#include "PortableDrawIdVertex.h"
#include "PortableDrawIdVertexSlang.h"
#include "PortableMatrixLayoutShader.h"
#include "PortableReadOnlyStorageShader.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(RACHIS_ENABLE_METAL) && RACHIS_ENABLE_METAL && !(defined(RACHIS_ENABLE_D3D12) && RACHIS_ENABLE_D3D12) &&   \
    !(defined(RACHIS_ENABLE_VULKAN) && RACHIS_ENABLE_VULKAN)
#define RACHIS_PORTABLE_CONTRACT_TAGS "[size:large][rhi][portable]"
#else
#define RACHIS_PORTABLE_CONTRACT_TAGS "[size:medium][rhi][portable]"
#endif

#if defined(RACHIS_TEST_DYNAMIC_BINDING_COUNTER_SHADER_PATH) || defined(RACHIS_RENDERER_SPIRV_DIR)
#define RACHIS_PORTABLE_DYNAMIC_COUNTER_SHADERS 1
#endif

#if defined(RACHIS_TEST_UNIFORM_BUFFER_RANGE_SHADER_PATH) || defined(RACHIS_RENDERER_SPIRV_DIR)
#define RACHIS_PORTABLE_UNIFORM_RANGE_SHADERS 1
#endif

#if (defined(RACHIS_TEST_SAMPLER_FILTER_VERTEX_SHADER_PATH) &&                                                         \
     defined(RACHIS_TEST_SAMPLER_FILTER_PIXEL_SHADER_PATH)) ||                                                         \
    defined(RACHIS_RENDERER_SPIRV_DIR)
#define RACHIS_PORTABLE_SAMPLER_FILTER_SHADERS 1
#endif

namespace
{

    struct PortableBackendSpec final
    {
        std::string_view name = {};
        rachis::Backend backend = rachis::Backend::d3d12;
        rachis::DeviceDesc desc = {};
        bool allowHeadlessSkipWhenUnavailable = false;
        std::string_view graphicsVertexShaderPath = {};
        std::string_view graphicsPixelShaderPath = {};
        std::string_view computeShaderPath = {};
        std::string_view computeEntryPoint = {};
        std::uint32_t computeStorageBinding = 0;
        std::uint32_t computePushConstantBytes = 0;
    };

    [[nodiscard]] const std::vector<PortableBackendSpec>& enabledBackends()
    {
        static const std::vector<PortableBackendSpec> specs = [] {
            std::vector<PortableBackendSpec> list;
#if defined(RACHIS_ENABLE_D3D12) && RACHIS_ENABLE_D3D12
            list.push_back(PortableBackendSpec{
                .name = "D3D12-WARP",
                .backend = rachis::Backend::d3d12,
                .desc = rachis::test::d3d12WarpDeviceDesc(),
                .allowHeadlessSkipWhenUnavailable = false,
                .graphicsVertexShaderPath = RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH,
                .graphicsPixelShaderPath = RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH,
                .computeShaderPath = RACHIS_TEST_COMPUTE_SHADER_PATH,
                .computeEntryPoint = "main",
                .computeStorageBinding = 0,
                .computePushConstantBytes = 2 * sizeof(std::uint32_t),
            });
#endif
#if defined(RACHIS_ENABLE_VULKAN) && RACHIS_ENABLE_VULKAN
            list.push_back(PortableBackendSpec{
                .name = "Vulkan",
                .backend = rachis::Backend::vulkan,
                .desc = rachis::test::vulkanDeviceDesc(),
                .allowHeadlessSkipWhenUnavailable = true,
                .graphicsVertexShaderPath = RACHIS_RENDERER_SPIRV_DIR "/VulkanGraphicsTriangle.direct.vs.spv",
                .graphicsPixelShaderPath = RACHIS_RENDERER_SPIRV_DIR "/VulkanGraphicsTriangle.ps.spv",
                .computeShaderPath = RACHIS_RENDERER_SPIRV_DIR "/VulkanComputeWrite.spv",
                .computeEntryPoint = "main",
                .computeStorageBinding = 0,
                .computePushConstantBytes = 2 * sizeof(std::uint32_t),
            });
#endif
#if defined(RACHIS_ENABLE_METAL) && RACHIS_ENABLE_METAL
            list.push_back(PortableBackendSpec{
                .name = "Metal",
                .backend = rachis::Backend::metal,
                .desc =
                    rachis::DeviceDesc{
                        .backend = rachis::Backend::metal,
                        .validation = rachis::ValidationLevel::standard,
                        .requireValidation = false,
                        .enableDred = false,
                    },
                .allowHeadlessSkipWhenUnavailable = false,
                .graphicsVertexShaderPath = RACHIS_METALLIB_DIR "/MetalHeadlessParity.direct.vs.pnms",
                .graphicsPixelShaderPath = RACHIS_METALLIB_DIR "/MetalHeadlessParity.ps.pnms",
                .computeShaderPath = RACHIS_METALLIB_DIR "/MetalHeadlessParity.cs.pnms",
                .computeEntryPoint = "computeMain",
                .computeStorageBinding = 1,
                .computePushConstantBytes = 4 * sizeof(std::uint32_t),
            });
#endif
            return list;
        }();
        return specs;
    }

    [[nodiscard]] std::optional<rachis::Device> createPortableDeviceOrSkip(const PortableBackendSpec& spec)
    {
        auto created = rachis::createDevice(spec.desc);
        if (!created)
        {
            if (spec.allowHeadlessSkipWhenUnavailable)
            {
                if (rachis::test::skipWithoutCreatedDevice(created))
                {
                    return std::nullopt;
                }
            }
            INFO("Backend: " << spec.name << " error: " << created.error().message);
            REQUIRE(created.has_value());
            return std::nullopt;
        }
        REQUIRE(created->isValid());
        REQUIRE(created->operationalState() == rachis::DeviceOperationalState::active);
        const auto adapter = rachis::test::requireAdapterInfo(*created);
        CHECK(adapter.backend == spec.backend);
        return std::move(*created);
    }

    [[nodiscard]] std::byte patternByte(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 37U + 19U) & 0xFFU);
    }

    [[nodiscard]] std::byte texelByte(std::uint32_t x, std::uint32_t y, std::uint32_t channel) noexcept
    {
        return static_cast<std::byte>((x * 29U + y * 53U + channel * 71U + 13U) & 0xFFU);
    }

    void checkNoDiagnosticsErrors(const rachis::Device& device)
    {
        const auto counts = device.diagnosticCounts();
        CHECK(counts.errors == 0);
        CHECK(counts.corruptions == 0);
    }

    struct IntegerClearFormat final
    {
        std::string_view name = {};
        rachis::Format format = rachis::Format::unknown;
        std::uint32_t channelCount = 0;
        std::uint32_t channelBytes = 0;
        bool isSigned = false;
    };

    [[nodiscard]] std::int64_t integerChannel(std::span<const std::byte> texel,
                                              const IntegerClearFormat& format,
                                              std::uint32_t channel) noexcept
    {
        std::uint64_t raw = 0;
        for (std::uint32_t index = 0; index < format.channelBytes; ++index)
        {
            raw |= static_cast<std::uint64_t>(
                       std::to_integer<std::uint8_t>(texel[(channel * format.channelBytes) + index]))
                   << (index * 8U);
        }
        if (!format.isSigned)
        {
            return static_cast<std::int64_t>(raw);
        }
        const auto signBit = std::uint64_t{1} << ((format.channelBytes * 8U) - 1U);
        return static_cast<std::int64_t>(raw ^ signBit) - static_cast<std::int64_t>(signBit);
    }

    [[nodiscard]] std::vector<std::int64_t> clearIntegerTargetAndReadBack(
        rachis::Device& device,
        const IntegerClearFormat& format,
        const rachis::ClearColor& loadClear,
        const std::optional<rachis::ClearColor>& attachmentClear)
    {
        constexpr std::uint32_t width = 4;
        constexpr std::uint32_t height = 2;
        const auto texelBytes = format.channelCount * format.channelBytes;

        auto texture = device.createTexture({
            .width = width,
            .height = height,
            .format = format.format,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
            .label = "Portable integer clear target",
        });
        REQUIRE(texture.has_value());
        auto view = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::renderTarget,
        });
        REQUIRE(view.has_value());

        const rachis::BufferTextureCopyRegion region{
            .bufferOffset = 0,
            .extent = {.width = width, .height = height, .depth = 1},
        };
        auto layout = device.queryTextureTransferLayout(*texture, region);
        REQUIRE(layout.has_value());
        auto readback = device.createBuffer({
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Portable integer clear readback",
        });
        REQUIRE(readback.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::none,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array colorAttachments{
            rachis::ColorAttachment{
                .view = *view,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
                .clearValue = loadClear,
            },
        };
        REQUIRE(
            device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = colorAttachments}).has_value());
        if (attachmentClear)
        {
            REQUIRE(device.clearColorAttachment(*command, 0, *attachmentClear).has_value());
        }
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         rachis::BufferTextureCopy{
                                             .buffer = *readback,
                                             .bufferOffset = 0,
                                             .bytesPerRow = layout->bytesPerRow,
                                             .rowsPerImage = layout->rowsPerImage,
                                             .texture = *texture,
                                             .extent = {.width = width, .height = height, .depth = 1},
                                         })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        std::vector<std::byte> bytes(static_cast<std::size_t>(layout->requiredBufferSize), std::byte{0});
        REQUIRE(device.readBuffer(*readback, 0, bytes).has_value());
        const std::span<const std::byte> first(bytes.data(), texelBytes);
        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                const auto offset =
                    (static_cast<std::size_t>(y) * layout->bytesPerRow) + (static_cast<std::size_t>(x) * texelBytes);
                CAPTURE(x, y);
                CHECK(std::equal(first.begin(), first.end(), bytes.begin() + static_cast<std::ptrdiff_t>(offset)));
            }
        }
        std::vector<std::int64_t> channels;
        channels.reserve(format.channelCount);
        for (std::uint32_t channel = 0; channel < format.channelCount; ++channel)
        {
            channels.push_back(integerChannel(first, format, channel));
        }

        REQUIRE(device.destroyBuffer(*readback).has_value());
        REQUIRE(device.destroyTextureView(*view).has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
        return channels;
    }

    [[nodiscard]] std::vector<std::byte> loadBytes(std::string_view path)
    {
        std::ifstream stream(std::filesystem::path(path), std::ios::binary | std::ios::ate);
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

#if defined(RACHIS_PORTABLE_DYNAMIC_COUNTER_SHADERS) || defined(RACHIS_PORTABLE_UNIFORM_RANGE_SHADERS) ||              \
    defined(RACHIS_PORTABLE_SAMPLER_FILTER_SHADERS)
    [[nodiscard]] std::vector<std::byte> loadBinaryFile(const std::string& path)
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
#endif

#if defined(RACHIS_PORTABLE_DYNAMIC_COUNTER_SHADERS)
    constexpr std::uint64_t dynamicCounterSlotSize = 256;
    constexpr std::uint32_t dynamicCounterSlotCount = 4096;
    constexpr std::uint32_t dynamicCounterTextureCount = 15;

    [[nodiscard]] std::vector<std::byte> dynamicCounterShader(rachis::Backend backend)
    {
#if defined(RACHIS_TEST_DYNAMIC_BINDING_COUNTER_SHADER_PATH)
        if (backend == rachis::Backend::d3d12)
        {
            return loadBinaryFile(RACHIS_TEST_DYNAMIC_BINDING_COUNTER_SHADER_PATH);
        }
#endif
#if defined(RACHIS_RENDERER_SPIRV_DIR)
        if (backend == rachis::Backend::vulkan)
        {
            return loadBinaryFile(std::string(RACHIS_RENDERER_SPIRV_DIR) + "/DynamicBindingCounter.spv");
        }
#endif
        return {};
    }

    [[nodiscard]] const std::vector<PortableBackendSpec>& dynamicCounterBackends()
    {
        static const std::vector<PortableBackendSpec> specs = [] {
            std::vector<PortableBackendSpec> list;
            for (const auto& spec : enabledBackends())
            {
                if (!dynamicCounterShader(spec.backend).empty())
                {
                    list.push_back(spec);
                }
            }
            return list;
        }();
        return specs;
    }

    struct DynamicCounterFixture final
    {
        rachis::ShaderModuleHandle shader = {};
        rachis::BindingLayoutHandle layout = {};
        rachis::PipelineLayoutHandle pipelineLayout = {};
        rachis::PipelineHandle pipeline = {};
        rachis::BufferHandle counters = {};
        rachis::BufferHandle zeros = {};
        rachis::BufferHandle readback = {};
        rachis::TextureHandle texture = {};
        rachis::TextureViewHandle view = {};
        rachis::BindingPoolHandle pool = {};
        rachis::BindingSetHandle set = {};
        rachis::BindingLayoutHandle probeLayout = {};
        rachis::BindingPoolHandle probePool = {};
        rachis::TextureHandle mipTexture = {};
        bool mipmapGeneration = false;
    };

    void createDynamicCounterBuffers(rachis::Device& device, DynamicCounterFixture& fixture)
    {
        constexpr auto bufferSize = dynamicCounterSlotSize * dynamicCounterSlotCount;
        auto counters = device.createBuffer({
            .size = bufferSize,
            .usage =
                rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
            .label = "Portable dynamic counters",
        });
        auto zeros = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Portable dynamic counter zeros",
        });
        auto readback = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Portable dynamic counter readback",
        });
        REQUIRE(counters.has_value());
        REQUIRE(zeros.has_value());
        REQUIRE(readback.has_value());
        fixture.counters = *counters;
        fixture.zeros = *zeros;
        fixture.readback = *readback;
        const std::vector<std::byte> zeroBytes(static_cast<std::size_t>(bufferSize), std::byte{0});
        REQUIRE(device.writeBuffer(fixture.zeros, 0, zeroBytes).has_value());
    }

    void createDynamicCounterBindings(rachis::Device& device, DynamicCounterFixture& fixture)
    {
        auto texture = device.createTexture({
            .width = 4,
            .height = 4,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled,
            .label = "Portable dynamic counter texture",
        });
        REQUIRE(texture.has_value());
        fixture.texture = *texture;
        auto view = device.createTextureView({
            .texture = fixture.texture,
            .usage = rachis::TextureViewUsage::sampled,
            .dimension = rachis::TextureViewDimension::texture2D,
            .label = "Portable dynamic counter view",
        });
        REQUIRE(view.has_value());
        fixture.view = *view;

        const std::array entries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute, 1},
            rachis::BindingLayoutEntry{
                1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, dynamicCounterTextureCount},
        };
        auto layout = device.createBindingLayout({.entries = entries, .label = "Portable dynamic counter layout"});
        REQUIRE(layout.has_value());
        fixture.layout = *layout;
        const std::array sizes{
            rachis::BindingPoolSize{rachis::BindingType::dynamicStorageBuffer, 1},
            rachis::BindingPoolSize{rachis::BindingType::sampledTexture, dynamicCounterTextureCount},
        };
        auto pool =
            device.createBindingPool({.maxBindingSets = 1, .sizes = sizes, .label = "Portable dynamic counter pool"});
        REQUIRE(pool.has_value());
        fixture.pool = *pool;
        std::vector<rachis::BindingWrite> writes{
            rachis::BindingWrite{
                .binding = 0,
                .resource = rachis::BufferBinding{.buffer = fixture.counters, .size = dynamicCounterSlotSize},
            },
        };
        for (std::uint32_t element = 0; element < dynamicCounterTextureCount; ++element)
        {
            writes.push_back(rachis::BindingWrite{.binding = 1, .arrayElement = element, .resource = fixture.view});
        }
        auto set = device.createBindingSet({
            .layout = fixture.layout,
            .writes = writes,
            .label = "Portable dynamic counter set",
            .pool = fixture.pool,
        });
        REQUIRE(set.has_value());
        fixture.set = *set;
    }

    void createDynamicCounterPipeline(rachis::Device& device,
                                      const PortableBackendSpec& spec,
                                      DynamicCounterFixture& fixture)
    {
        const auto bytecode = dynamicCounterShader(spec.backend);
        REQUIRE_FALSE(bytecode.empty());
        const std::array requirements{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
        };
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .label = "Portable dynamic counter shader",
            .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements),
        });
        REQUIRE(shader.has_value());
        fixture.shader = *shader;
        const std::array layouts{fixture.layout};
        auto pipelineLayout =
            device.createPipelineLayout({.bindingLayouts = layouts, .label = "Portable dynamic counter layout"});
        REQUIRE(pipelineLayout.has_value());
        fixture.pipelineLayout = *pipelineLayout;
        auto pipeline = device.createComputePipeline({
            .layout = fixture.pipelineLayout,
            .computeShader = fixture.shader,
            .label = "Portable dynamic counter pipeline",
        });
        REQUIRE(pipeline.has_value());
        fixture.pipeline = *pipeline;
    }

    void createDynamicCounterProbes(rachis::Device& device, DynamicCounterFixture& fixture)
    {
        const std::array probeEntries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
        };
        auto probeLayout =
            device.createBindingLayout({.entries = probeEntries, .label = "Portable unrelated probe layout"});
        REQUIRE(probeLayout.has_value());
        fixture.probeLayout = *probeLayout;
        const std::array probeSizes{rachis::BindingPoolSize{rachis::BindingType::sampledTexture, 1}};
        auto probePool = device.createBindingPool(
            {.maxBindingSets = 1, .sizes = probeSizes, .label = "Portable unrelated probe pool"});
        REQUIRE(probePool.has_value());
        fixture.probePool = *probePool;
        fixture.mipmapGeneration = rachis::test::requireCapabilities(device).mipmapGeneration;
        if (!fixture.mipmapGeneration)
        {
            return;
        }
        auto mipTexture = device.createTexture({
            .dimension = rachis::TextureDimension::texture2D,
            .width = 32,
            .height = 32,
            .mipLevels = 6,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::copySource |
                     rachis::TextureUsage::copyDestination,
            .label = "Portable unrelated mipmap texture",
        });
        REQUIRE(mipTexture.has_value());
        fixture.mipTexture = *mipTexture;
    }

    [[nodiscard]] DynamicCounterFixture createDynamicCounterFixture(rachis::Device& device,
                                                                    const PortableBackendSpec& spec)
    {
        DynamicCounterFixture fixture;
        createDynamicCounterBuffers(device, fixture);
        createDynamicCounterBindings(device, fixture);
        createDynamicCounterPipeline(device, spec, fixture);
        createDynamicCounterProbes(device, fixture);
        return fixture;
    }

    void destroyDynamicCounterFixture(rachis::Device& device, const DynamicCounterFixture& fixture)
    {
        REQUIRE(device.destroyPipeline(fixture.pipeline).has_value());
        REQUIRE(device.destroyPipelineLayout(fixture.pipelineLayout).has_value());
        REQUIRE(device.destroyShaderModule(fixture.shader).has_value());
        REQUIRE(device.destroyBindingSet(fixture.set).has_value());
        REQUIRE(device.destroyBindingPool(fixture.pool).has_value());
        REQUIRE(device.destroyBindingLayout(fixture.layout).has_value());
        REQUIRE(device.destroyBindingPool(fixture.probePool).has_value());
        REQUIRE(device.destroyBindingLayout(fixture.probeLayout).has_value());
        if (fixture.mipmapGeneration)
        {
            REQUIRE(device.destroyTexture(fixture.mipTexture).has_value());
        }
        REQUIRE(device.destroyTextureView(fixture.view).has_value());
        REQUIRE(device.destroyTexture(fixture.texture).has_value());
        REQUIRE(device.destroyBuffer(fixture.counters).has_value());
        REQUIRE(device.destroyBuffer(fixture.zeros).has_value());
        REQUIRE(device.destroyBuffer(fixture.readback).has_value());
    }

    void recordDynamicCounterPrologue(rachis::Device& device,
                                      rachis::CommandBuffer& command,
                                      const DynamicCounterFixture& fixture)
    {
        REQUIRE(device
                    .copyBuffer(command,
                                {
                                    .source = fixture.zeros,
                                    .destination = fixture.counters,
                                    .size = dynamicCounterSlotSize * dynamicCounterSlotCount,
                                })
                    .has_value());
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = fixture.counters,
                                       .stagesBefore = rachis::PipelineStage::copy,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessBefore = rachis::Access::copyDestination,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
        REQUIRE(device.bindComputePipeline(command, fixture.pipeline).has_value());
    }

    void recordDynamicCounterEpilogue(rachis::Device& device,
                                      rachis::CommandBuffer& command,
                                      const DynamicCounterFixture& fixture)
    {
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = fixture.counters,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(command,
                                {
                                    .source = fixture.counters,
                                    .destination = fixture.readback,
                                    .size = dynamicCounterSlotSize * dynamicCounterSlotCount,
                                })
                    .has_value());
    }

    void recordDynamicCounterBinds(rachis::Device& device,
                                   rachis::CommandBuffer& command,
                                   const DynamicCounterFixture& fixture,
                                   std::uint32_t firstIndex,
                                   std::uint32_t endIndex)
    {
        std::optional<std::uint32_t> failedIndex;
        std::string failure;
        for (std::uint32_t index = firstIndex; index < endIndex && !failedIndex.has_value(); ++index)
        {
            const std::array offsets{
                static_cast<std::uint32_t>((index % dynamicCounterSlotCount) * dynamicCounterSlotSize),
            };
            auto bound = device.bindComputeBindingSet(command, 0, fixture.set, offsets);
            auto dispatched = bound ? device.dispatch(command, 1) : rachis::Status{};
            if (!bound || !dispatched)
            {
                failedIndex = index;
                failure = !bound ? bound.error().message : dispatched.error().message;
            }
        }
        const auto failedAt = failedIndex.value_or(endIndex);
        CAPTURE(failedAt, failure);
        REQUIRE_FALSE(failedIndex.has_value());
    }

    void checkDynamicCounters(rachis::Device& device, const DynamicCounterFixture& fixture, std::uint32_t bindCount)
    {
        constexpr auto wordsPerSlot = static_cast<std::size_t>(dynamicCounterSlotSize / sizeof(std::uint32_t));
        std::vector<std::uint32_t> words(wordsPerSlot * dynamicCounterSlotCount);
        REQUIRE(device.readBuffer(fixture.readback, 0, std::as_writable_bytes(std::span{words})).has_value());
        std::uint64_t total = 0;
        std::uint32_t firstMismatch = dynamicCounterSlotCount;
        for (std::uint32_t slot = 0; slot < dynamicCounterSlotCount; ++slot)
        {
            const auto count = words[slot * wordsPerSlot];
            total += count;
            const auto expected =
                bindCount / dynamicCounterSlotCount + (slot < bindCount % dynamicCounterSlotCount ? 1U : 0U);
            if (firstMismatch == dynamicCounterSlotCount && count != expected)
            {
                firstMismatch = slot;
            }
        }
        CAPTURE(firstMismatch);
        CHECK(total == bindCount);
        CHECK(firstMismatch == dynamicCounterSlotCount);
    }

    void checkUnrelatedDescriptorUsers(rachis::Device& device, const DynamicCounterFixture& fixture)
    {
        const std::array probeWrites{rachis::BindingWrite{.binding = 0, .resource = fixture.view}};
        auto probeSet = device.createBindingSet({
            .layout = fixture.probeLayout,
            .writes = probeWrites,
            .label = "Portable unrelated probe set",
            .pool = fixture.probePool,
        });
        const std::string setFailure = probeSet ? std::string{} : probeSet.error().message;
        CAPTURE(setFailure);
        REQUIRE(probeSet.has_value());
        REQUIRE(device.destroyBindingSet(*probeSet).has_value());
        if (!fixture.mipmapGeneration)
        {
            return;
        }

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = fixture.mipTexture,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                        .subresources = {.mipLevelCount = 1, .arrayLayerCount = 1},
                                    })
                    .has_value());
        auto generated = device.generateMipmaps(*command,
                                                {
                                                    .texture = fixture.mipTexture,
                                                    .subresources = {.mipLevelCount = 6, .arrayLayerCount = 1},
                                                    .baseLayout = rachis::TextureLayout::copyDestination,
                                                    .finalLayout = rachis::TextureLayout::shaderResource,
                                                });
        const std::string mipmapFailure = generated ? std::string{} : generated.error().message;
        CAPTURE(mipmapFailure);
        REQUIRE(generated.has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
    }
#endif

    using Float4 = std::array<float, 4>;

    constexpr std::uint64_t uniformRangeOutputSlotSize = 256;

    struct UniformRangeCase final
    {
        std::string_view name = {};
        rachis::BufferBinding binding = {};
        bool dynamic = false;
        std::uint32_t dynamicOffset = 0;
        Float4 expected = {};
    };

    struct UniformRangeFixture final
    {
        rachis::BindingLayoutHandle staticLayout = {};
        rachis::BindingLayoutHandle dynamicLayout = {};
        rachis::BufferHandle output = {};
        rachis::BufferHandle readback = {};
        std::vector<rachis::BindingSetHandle> sets;
    };

    [[nodiscard]] std::vector<std::byte> uniformBufferRangeShader(rachis::Backend backend)
    {
        static_cast<void>(backend);
#if defined(RACHIS_TEST_UNIFORM_BUFFER_RANGE_SHADER_PATH)
        if (backend == rachis::Backend::d3d12)
        {
            return loadBinaryFile(RACHIS_TEST_UNIFORM_BUFFER_RANGE_SHADER_PATH);
        }
#endif
#if defined(RACHIS_RENDERER_SPIRV_DIR)
        if (backend == rachis::Backend::vulkan)
        {
            return loadBinaryFile(std::string(RACHIS_RENDERER_SPIRV_DIR) + "/UniformBufferRange.spv");
        }
#endif
        return {};
    }

    [[nodiscard]] std::vector<std::byte> dynamicUniformBufferArrayShader(rachis::Backend backend)
    {
        static_cast<void>(backend);
#if defined(RACHIS_TEST_DYNAMIC_UNIFORM_BUFFER_ARRAY_SHADER_PATH)
        if (backend == rachis::Backend::d3d12)
        {
            return loadBinaryFile(RACHIS_TEST_DYNAMIC_UNIFORM_BUFFER_ARRAY_SHADER_PATH);
        }
#endif
#if defined(RACHIS_RENDERER_SPIRV_DIR)
        if (backend == rachis::Backend::vulkan)
        {
            return loadBinaryFile(std::string(RACHIS_RENDERER_SPIRV_DIR) + "/DynamicUniformBufferArray.spv");
        }
#endif
        return {};
    }

    [[nodiscard]] rachis::BufferHandle createUniformRangeSource(rachis::Device& device,
                                                                std::uint64_t size,
                                                                std::string_view label)
    {
        auto buffer = device.createBuffer({
            .size = size,
            .usage = rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
            .label = label,
        });
        const std::string failure = buffer ? std::string{} : buffer.error().message;
        CAPTURE(size, failure);
        REQUIRE(buffer.has_value());
        return *buffer;
    }

    void writeFloat4(rachis::Device& device, rachis::BufferHandle buffer, std::uint64_t offset, const Float4& value)
    {
        REQUIRE(device.writeBuffer(buffer, offset, std::as_bytes(std::span{value})).has_value());
    }

    void recordUniformRangeCopies(rachis::Device& device,
                                  rachis::CommandBuffer& command,
                                  const UniformRangeFixture& fixture,
                                  std::span<const UniformRangeCase> cases,
                                  rachis::PipelineHandle staticPipeline,
                                  rachis::PipelineHandle dynamicPipeline)
    {
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = fixture.output,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
        for (std::size_t index = 0; index < cases.size(); ++index)
        {
            const auto& testCase = cases[index];
            CAPTURE(testCase.name);
            REQUIRE(
                device.bindComputePipeline(command, testCase.dynamic ? dynamicPipeline : staticPipeline).has_value());
            const std::array offsets{testCase.dynamicOffset};
            auto bound = testCase.dynamic ? device.bindComputeBindingSet(command, 0, fixture.sets[index], offsets)
                                          : device.bindComputeBindingSet(command, 0, fixture.sets[index]);
            const std::string failure = bound ? std::string{} : bound.error().message;
            CAPTURE(failure);
            REQUIRE(bound.has_value());
            REQUIRE(device.dispatch(command, 1).has_value());
        }
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = fixture.output,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(command,
                                {
                                    .source = fixture.output,
                                    .destination = fixture.readback,
                                    .size = uniformRangeOutputSlotSize * cases.size(),
                                })
                    .has_value());
    }

    void copyUniformRanges(rachis::Device& device,
                           std::span<const std::byte> bytecode,
                           const UniformRangeFixture& fixture,
                           std::span<const UniformRangeCase> cases)
    {
        const std::array requirements{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
        };
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .label = "Portable uniform range shader",
            .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements),
        });
        REQUIRE(shader.has_value());
        const std::array staticLayouts{fixture.staticLayout};
        const std::array dynamicLayouts{fixture.dynamicLayout};
        auto staticPipelineLayout =
            device.createPipelineLayout({.bindingLayouts = staticLayouts, .label = "Portable uniform range layout"});
        auto dynamicPipelineLayout = device.createPipelineLayout(
            {.bindingLayouts = dynamicLayouts, .label = "Portable dynamic uniform range layout"});
        REQUIRE(staticPipelineLayout.has_value());
        REQUIRE(dynamicPipelineLayout.has_value());
        auto staticPipeline = device.createComputePipeline({
            .layout = *staticPipelineLayout,
            .computeShader = *shader,
            .label = "Portable uniform range pipeline",
        });
        auto dynamicPipeline = device.createComputePipeline({
            .layout = *dynamicPipelineLayout,
            .computeShader = *shader,
            .label = "Portable dynamic uniform range pipeline",
        });
        REQUIRE(staticPipeline.has_value());
        REQUIRE(dynamicPipeline.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        recordUniformRangeCopies(device, *command, fixture, cases, *staticPipeline, *dynamicPipeline);
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        constexpr auto floatsPerSlot = static_cast<std::size_t>(uniformRangeOutputSlotSize / sizeof(float));
        std::vector<float> values(floatsPerSlot * cases.size());
        REQUIRE(device.readBuffer(fixture.readback, 0, std::as_writable_bytes(std::span{values})).has_value());
        for (std::size_t index = 0; index < cases.size(); ++index)
        {
            CAPTURE(cases[index].name);
            Float4 copied{};
            std::copy_n(
                values.begin() + static_cast<std::ptrdiff_t>(index * floatsPerSlot), copied.size(), copied.begin());
            CHECK(copied == cases[index].expected);
        }

        REQUIRE(device.destroyPipeline(*dynamicPipeline).has_value());
        REQUIRE(device.destroyPipeline(*staticPipeline).has_value());
        REQUIRE(device.destroyPipelineLayout(*dynamicPipelineLayout).has_value());
        REQUIRE(device.destroyPipelineLayout(*staticPipelineLayout).has_value());
        REQUIRE(device.destroyShaderModule(*shader).has_value());
    }

    struct DynamicUniformArrayCase final
    {
        std::array<std::uint32_t, 2> offsets = {};
        std::array<Float4, 2> expected = {};
    };

    struct DynamicUniformArrayFixture final
    {
        rachis::BindingLayoutHandle layout = {};
        rachis::BufferHandle output = {};
        rachis::BufferHandle readback = {};
        std::vector<rachis::BindingSetHandle> sets;
    };

    void recordDynamicUniformArrayCopies(rachis::Device& device,
                                         rachis::CommandBuffer& command,
                                         const DynamicUniformArrayFixture& fixture,
                                         std::span<const DynamicUniformArrayCase> cases,
                                         rachis::PipelineHandle pipeline)
    {
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = fixture.output,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
        REQUIRE(device.bindComputePipeline(command, pipeline).has_value());
        for (std::size_t index = 0; index < cases.size(); ++index)
        {
            CAPTURE(index);
            auto bound = device.bindComputeBindingSet(command, 0, fixture.sets[index], cases[index].offsets);
            const std::string failure = bound ? std::string{} : bound.error().message;
            CAPTURE(failure);
            REQUIRE(bound.has_value());
            REQUIRE(device.dispatch(command, 1).has_value());
        }
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = fixture.output,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(command,
                                {
                                    .source = fixture.output,
                                    .destination = fixture.readback,
                                    .size = uniformRangeOutputSlotSize * cases.size(),
                                })
                    .has_value());
    }

    void copyDynamicUniformArrays(rachis::Device& device,
                                  std::span<const std::byte> bytecode,
                                  const DynamicUniformArrayFixture& fixture,
                                  std::span<const DynamicUniformArrayCase> cases)
    {
        const std::array requirements{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 2},
            rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::storageBuffer, 1},
        };
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .label = "Portable dynamic uniform array shader",
            .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements),
        });
        REQUIRE(shader.has_value());
        const std::array layouts{fixture.layout};
        auto pipelineLayout = device.createPipelineLayout(
            {.bindingLayouts = layouts, .label = "Portable dynamic uniform array pipeline layout"});
        REQUIRE(pipelineLayout.has_value());
        auto pipeline = device.createComputePipeline({
            .layout = *pipelineLayout,
            .computeShader = *shader,
            .label = "Portable dynamic uniform array pipeline",
        });
        REQUIRE(pipeline.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        recordDynamicUniformArrayCopies(device, *command, fixture, cases, *pipeline);
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        constexpr auto floatsPerSlot = static_cast<std::size_t>(uniformRangeOutputSlotSize / sizeof(float));
        std::vector<float> values(floatsPerSlot * cases.size());
        REQUIRE(device.readBuffer(fixture.readback, 0, std::as_writable_bytes(std::span{values})).has_value());
        for (std::size_t index = 0; index < cases.size(); ++index)
        {
            for (std::size_t element = 0; element < cases[index].expected.size(); ++element)
            {
                CAPTURE(index, element);
                Float4 copied{};
                std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(index * floatsPerSlot + element * 4),
                            copied.size(),
                            copied.begin());
                CHECK(copied == cases[index].expected[element]);
            }
        }

        REQUIRE(device.destroyPipeline(*pipeline).has_value());
        REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
        REQUIRE(device.destroyShaderModule(*shader).has_value());
    }

    constexpr std::uint32_t samplerFilterTargetSize = 16;
    constexpr std::uint32_t samplerFilterSlightMinifyTextureSize = 17;
    constexpr std::uint8_t samplerFilterUpperLevelValue = 200;

    enum class SamplerFilterTexture : std::uint8_t
    {
        checkerboard,
        checkerboardWithUpperLevel,
        blackWithUpperLevel,
        smallCheckerboard,
        slightlyMinifiedBlackWithUpperLevel,
    };

    enum class SamplerFilterExpectation : std::uint8_t
    {
        filtered,
        unfiltered,
        black,
        partlyUpperLevel,
        magnifiedFiltered,
    };

    struct SamplerFilterCase final
    {
        std::string_view name = {};
        rachis::SamplerDesc sampler = {};
        SamplerFilterTexture texture = SamplerFilterTexture::checkerboard;
        SamplerFilterExpectation expected = SamplerFilterExpectation::filtered;
        bool mayBeUnsupportedOnD3D12 = false;
    };

    struct SamplerFilterShaders final
    {
        std::vector<std::byte> vertex;
        std::vector<std::byte> pixel;
    };

    struct SamplerFilterTarget final
    {
        rachis::TextureHandle texture = {};
        rachis::TextureViewHandle view = {};
        rachis::BufferHandle readback = {};
        rachis::TextureTransferLayout layout = {};
        bool written = false;
    };

    [[nodiscard]] SamplerFilterShaders samplerFilterShaders(rachis::Backend backend)
    {
        static_cast<void>(backend);
#if defined(RACHIS_TEST_SAMPLER_FILTER_VERTEX_SHADER_PATH) && defined(RACHIS_TEST_SAMPLER_FILTER_PIXEL_SHADER_PATH)
        if (backend == rachis::Backend::d3d12)
        {
            return {
                .vertex = loadBinaryFile(RACHIS_TEST_SAMPLER_FILTER_VERTEX_SHADER_PATH),
                .pixel = loadBinaryFile(RACHIS_TEST_SAMPLER_FILTER_PIXEL_SHADER_PATH),
            };
        }
#endif
#if defined(RACHIS_RENDERER_SPIRV_DIR)
        if (backend == rachis::Backend::vulkan)
        {
            return {
                .vertex = loadBinaryFile(std::string(RACHIS_RENDERER_SPIRV_DIR) + "/SamplerFilter.vs.spv"),
                .pixel = loadBinaryFile(std::string(RACHIS_RENDERER_SPIRV_DIR) + "/SamplerFilter.ps.spv"),
            };
        }
#endif
        return {};
    }

    [[nodiscard]] std::uint32_t samplerFilterTextureSize(SamplerFilterTexture texture) noexcept
    {
        switch (texture)
        {
        case SamplerFilterTexture::smallCheckerboard:
            return 4U;
        case SamplerFilterTexture::slightlyMinifiedBlackWithUpperLevel:
            return samplerFilterSlightMinifyTextureSize;
        default:
            return 64U;
        }
    }

    [[nodiscard]] std::uint32_t samplerFilterTextureMipLevels(SamplerFilterTexture texture) noexcept
    {
        return texture == SamplerFilterTexture::checkerboardWithUpperLevel ||
                       texture == SamplerFilterTexture::blackWithUpperLevel ||
                       texture == SamplerFilterTexture::slightlyMinifiedBlackWithUpperLevel
                   ? 2U
                   : 1U;
    }

    [[nodiscard]] std::uint8_t samplerFilterTexel(SamplerFilterTexture texture,
                                                  std::uint32_t level,
                                                  std::uint32_t x,
                                                  std::uint32_t y) noexcept
    {
        if (level > 0)
        {
            return samplerFilterUpperLevelValue;
        }
        if (texture == SamplerFilterTexture::blackWithUpperLevel ||
            texture == SamplerFilterTexture::slightlyMinifiedBlackWithUpperLevel)
        {
            return 0;
        }
        return ((x + y) & 1U) != 0 ? 255 : 0;
    }

    [[nodiscard]] rachis::TextureHandle createSamplerFilterTexture(rachis::Device& device, SamplerFilterTexture content)
    {
        const auto size = samplerFilterTextureSize(content);
        const auto mipLevels = samplerFilterTextureMipLevels(content);
        auto texture = device.createTexture({
            .width = size,
            .height = size,
            .mipLevels = mipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
            .label = "Portable sampler filter texture",
        });
        REQUIRE(texture.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::none,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        std::vector<rachis::BufferHandle> uploads;
        for (std::uint32_t level = 0; level < mipLevels; ++level)
        {
            const auto extent = std::max(1U, size >> level);
            const rachis::BufferTextureCopyRegion region{
                .textureSubresource = {.mipLevel = level},
                .extent = {.width = extent, .height = extent, .depth = 1},
            };
            auto layout = device.queryTextureTransferLayout(*texture, region);
            REQUIRE(layout.has_value());
            std::vector<std::byte> bytes(static_cast<std::size_t>(layout->requiredBufferSize), std::byte{0});
            for (std::uint32_t y = 0; y < extent; ++y)
            {
                for (std::uint32_t x = 0; x < extent; ++x)
                {
                    const auto value = std::byte{samplerFilterTexel(content, level, x, y)};
                    const auto offset =
                        static_cast<std::size_t>(y) * layout->bytesPerRow + static_cast<std::size_t>(x) * 4U;
                    bytes[offset] = value;
                    bytes[offset + 1] = value;
                    bytes[offset + 2] = value;
                    bytes[offset + 3] = std::byte{0xFF};
                }
            }
            auto upload = device.createBuffer({
                .size = layout->requiredBufferSize,
                .usage = rachis::BufferUsage::copySource,
                .memoryType = rachis::MemoryType::upload,
                .label = "Portable sampler filter upload",
            });
            REQUIRE(upload.has_value());
            uploads.push_back(*upload);
            REQUIRE(device.writeBuffer(*upload, 0, bytes).has_value());
            REQUIRE(device
                        .copyBufferToTexture(*command,
                                             {
                                                 .buffer = *upload,
                                                 .bytesPerRow = layout->bytesPerRow,
                                                 .rowsPerImage = layout->rowsPerImage,
                                                 .texture = *texture,
                                                 .textureSubresource = {.mipLevel = level},
                                                 .extent = region.extent,
                                             })
                        .has_value());
        }
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::pixelShader,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                    })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        for (const auto upload : uploads)
        {
            REQUIRE(device.destroyBuffer(upload).has_value());
        }
        return *texture;
    }

    [[nodiscard]] SamplerFilterTarget createSamplerFilterTarget(rachis::Device& device)
    {
        SamplerFilterTarget target;
        auto texture = device.createTexture({
            .width = samplerFilterTargetSize,
            .height = samplerFilterTargetSize,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
            .label = "Portable sampler filter target",
        });
        REQUIRE(texture.has_value());
        target.texture = *texture;
        auto view = device.createTextureView({
            .texture = target.texture,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension = rachis::TextureViewDimension::texture2D,
            .label = "Portable sampler filter target view",
        });
        REQUIRE(view.has_value());
        target.view = *view;
        const rachis::BufferTextureCopyRegion region{
            .extent = {.width = samplerFilterTargetSize, .height = samplerFilterTargetSize, .depth = 1},
        };
        auto layout = device.queryTextureTransferLayout(target.texture, region);
        REQUIRE(layout.has_value());
        target.layout = *layout;
        auto readback = device.createBuffer({
            .size = target.layout.requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Portable sampler filter readback",
        });
        REQUIRE(readback.has_value());
        target.readback = *readback;
        return target;
    }

    void recordSamplerFilterDraw(rachis::Device& device,
                                 rachis::CommandBuffer& command,
                                 const SamplerFilterTarget& target,
                                 rachis::PipelineHandle pipeline,
                                 rachis::BindingSetHandle set)
    {
        REQUIRE(device
                    .textureBarrier(
                        command,
                        {
                            .texture = target.texture,
                            .stagesBefore = target.written ? rachis::PipelineStage::copy : rachis::PipelineStage::none,
                            .stagesAfter = rachis::PipelineStage::renderTarget,
                            .accessBefore = target.written ? rachis::Access::copySource : rachis::Access::none,
                            .accessAfter = rachis::Access::renderTarget,
                            .layoutBefore =
                                target.written ? rachis::TextureLayout::copySource : rachis::TextureLayout::undefined,
                            .layoutAfter = rachis::TextureLayout::renderTarget,
                        })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = target.view,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
            },
        };
        REQUIRE(device.beginRendering(command, {.colorAttachments = attachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(command, pipeline).has_value());
        REQUIRE(device.bindGraphicsBindingSet(command, 0, set).has_value());
        REQUIRE(device
                    .setViewport(command,
                                 {
                                     .width = static_cast<float>(samplerFilterTargetSize),
                                     .height = static_cast<float>(samplerFilterTargetSize),
                                 })
                    .has_value());
        REQUIRE(device.setScissor(command, {.width = samplerFilterTargetSize, .height = samplerFilterTargetSize})
                    .has_value());
        REQUIRE(device.draw(command, 3).has_value());
        REQUIRE(device.endRendering(command).has_value());
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = target.texture,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyTextureToBuffer(
                        command,
                        {
                            .buffer = target.readback,
                            .bytesPerRow = target.layout.bytesPerRow,
                            .rowsPerImage = target.layout.rowsPerImage,
                            .texture = target.texture,
                            .extent = {.width = samplerFilterTargetSize, .height = samplerFilterTargetSize, .depth = 1},
                        })
                    .has_value());
    }

    [[nodiscard]] std::vector<std::uint8_t> drawSamplerFilterCase(rachis::Device& device,
                                                                  SamplerFilterTarget& target,
                                                                  rachis::PipelineHandle pipeline,
                                                                  rachis::BindingSetHandle set)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        recordSamplerFilterDraw(device, *command, target, pipeline, set);
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        target.written = true;

        std::vector<std::byte> bytes(static_cast<std::size_t>(target.layout.requiredBufferSize), std::byte{0});
        REQUIRE(device.readBuffer(target.readback, 0, bytes).has_value());
        std::vector<std::uint8_t> reds;
        reds.reserve(static_cast<std::size_t>(samplerFilterTargetSize) * samplerFilterTargetSize);
        for (std::uint32_t y = 0; y < samplerFilterTargetSize; ++y)
        {
            for (std::uint32_t x = 0; x < samplerFilterTargetSize; ++x)
            {
                const auto offset =
                    static_cast<std::size_t>(y) * target.layout.bytesPerRow + static_cast<std::size_t>(x) * 4U;
                reds.push_back(std::to_integer<std::uint8_t>(bytes[offset]));
            }
        }
        return reds;
    }

    [[nodiscard]] bool samplerFilterValueMatches(SamplerFilterExpectation expected, std::uint8_t red) noexcept
    {
        switch (expected)
        {
        case SamplerFilterExpectation::filtered:
            return red >= 100 && red <= 155;
        case SamplerFilterExpectation::unfiltered:
            return red == 0 || red == 255;
        case SamplerFilterExpectation::partlyUpperLevel:
            return red >= 8 && red <= 100;
        case SamplerFilterExpectation::magnifiedFiltered:
            return red >= 40 && red <= 215;
        default:
            return red == 0;
        }
    }

    void checkSamplerFilterCases(rachis::Device& device,
                                 rachis::Backend backend,
                                 std::span<const SamplerFilterCase> cases)
    {
        std::vector<std::optional<rachis::SamplerHandle>> samplers;
        for (const auto& testCase : cases)
        {
            CAPTURE(testCase.name);
            auto sampler = device.createSampler(testCase.sampler);
            if (!sampler && testCase.mayBeUnsupportedOnD3D12 && backend == rachis::Backend::d3d12 &&
                sampler.error().code == rachis::ErrorCode::unsupported)
            {
                samplers.emplace_back();
                continue;
            }
            const std::string failure = sampler ? std::string{} : sampler.error().message;
            CAPTURE(failure);
            REQUIRE(sampler.has_value());
            samplers.emplace_back(*sampler);
        }

        const auto shaders = samplerFilterShaders(backend);
        if (!shaders.vertex.empty() && !shaders.pixel.empty())
        {
            const std::array pixelBindings{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampler, 1},
            };
            auto vertexShader = device.createShaderModule({
                .stage = rachis::ShaderStage::vertex,
                .bytecode = shaders.vertex,
                .entryPoint = "vertexMain",
                .label = "Portable sampler filter vertex shader",
            });
            auto pixelShader = device.createShaderModule({
                .stage = rachis::ShaderStage::pixel,
                .bytecode = shaders.pixel,
                .entryPoint = "pixelMain",
                .label = "Portable sampler filter pixel shader",
                .shaderInterface = rachis::test::shaderInterface(pixelBindings),
            });
            REQUIRE(vertexShader.has_value());
            REQUIRE(pixelShader.has_value());
            const std::array entries{
                rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 1},
                rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 1},
            };
            auto bindingLayout =
                device.createBindingLayout({.entries = entries, .label = "Portable sampler filter binding layout"});
            REQUIRE(bindingLayout.has_value());
            const std::array bindingLayouts{*bindingLayout};
            auto pipelineLayout = device.createPipelineLayout(
                {.bindingLayouts = bindingLayouts, .label = "Portable sampler filter pipeline layout"});
            REQUIRE(pipelineLayout.has_value());
            const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
            auto pipeline = device.createGraphicsPipeline({
                .layout = *pipelineLayout,
                .vertexShader = *vertexShader,
                .pixelShader = *pixelShader,
                .cullMode = rachis::CullMode::none,
                .colorTargets = colorTargets,
                .label = "Portable sampler filter pipeline",
            });
            const std::string pipelineFailure = pipeline ? std::string{} : pipeline.error().message;
            CAPTURE(pipelineFailure);
            REQUIRE(pipeline.has_value());

            const auto caseCount = static_cast<std::uint32_t>(cases.size());
            const std::array poolSizes{
                rachis::BindingPoolSize{rachis::BindingType::sampledTexture, caseCount},
                rachis::BindingPoolSize{rachis::BindingType::sampler, caseCount},
            };
            auto pool = device.createBindingPool({
                .maxBindingSets = caseCount,
                .sizes = poolSizes,
                .label = "Portable sampler filter pool",
            });
            REQUIRE(pool.has_value());

            constexpr std::array textureKinds{
                SamplerFilterTexture::checkerboard,
                SamplerFilterTexture::checkerboardWithUpperLevel,
                SamplerFilterTexture::blackWithUpperLevel,
                SamplerFilterTexture::smallCheckerboard,
                SamplerFilterTexture::slightlyMinifiedBlackWithUpperLevel,
            };
            std::array<rachis::TextureHandle, textureKinds.size()> textures{};
            std::array<rachis::TextureViewHandle, textureKinds.size()> views{};
            for (std::size_t index = 0; index < textureKinds.size(); ++index)
            {
                textures[index] = createSamplerFilterTexture(device, textureKinds[index]);
                auto view = device.createTextureView({
                    .texture = textures[index],
                    .usage = rachis::TextureViewUsage::sampled,
                    .dimension = rachis::TextureViewDimension::texture2D,
                    .label = "Portable sampler filter view",
                });
                REQUIRE(view.has_value());
                views[index] = *view;
            }

            auto target = createSamplerFilterTarget(device);
            for (std::size_t index = 0; index < cases.size(); ++index)
            {
                const auto& testCase = cases[index];
                CAPTURE(testCase.name);
                if (!samplers[index])
                {
                    continue;
                }
                const std::array writes{
                    rachis::BindingWrite{.binding = 0, .resource = views[static_cast<std::size_t>(testCase.texture)]},
                    rachis::BindingWrite{.binding = 1, .resource = *samplers[index]},
                };
                auto set = device.createBindingSet({
                    .layout = *bindingLayout,
                    .writes = writes,
                    .label = "Portable sampler filter set",
                    .pool = *pool,
                });
                REQUIRE(set.has_value());
                const auto reds = drawSamplerFilterCase(device, target, *pipeline, *set);
                const auto mismatch = std::ranges::find_if(
                    reds, [&](std::uint8_t red) { return !samplerFilterValueMatches(testCase.expected, red); });
                const int firstMismatch = mismatch == reds.end() ? -1 : static_cast<int>(*mismatch);
                const auto mismatches = std::ranges::count_if(
                    reds, [&](std::uint8_t red) { return !samplerFilterValueMatches(testCase.expected, red); });
                CAPTURE(firstMismatch);
                CHECK(mismatches == 0);
                REQUIRE(device.destroyBindingSet(*set).has_value());
            }

            REQUIRE(device.destroyBuffer(target.readback).has_value());
            REQUIRE(device.destroyTextureView(target.view).has_value());
            REQUIRE(device.destroyTexture(target.texture).has_value());
            for (std::size_t index = 0; index < textureKinds.size(); ++index)
            {
                REQUIRE(device.destroyTextureView(views[index]).has_value());
                REQUIRE(device.destroyTexture(textures[index]).has_value());
            }
            REQUIRE(device.destroyBindingPool(*pool).has_value());
            REQUIRE(device.destroyPipeline(*pipeline).has_value());
            REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
            REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
            REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
            REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
        }

        for (const auto& sampler : samplers)
        {
            if (sampler)
            {
                REQUIRE(device.destroySampler(*sampler).has_value());
            }
        }
        checkNoDiagnosticsErrors(device);
    }

#ifdef RACHIS_TEST_PORTABLE_COMPUTE_SHADER_DIR
    [[nodiscard]] std::vector<std::byte> loadPortableComputeShader(rachis::Backend backend)
    {
        std::string extension = "spv";
        if (backend == rachis::Backend::d3d12)
        {
            extension = "dxil";
        }
        else if (backend == rachis::Backend::metal)
        {
            extension = "pnms";
        }
        const auto path =
            std::filesystem::path(RACHIS_TEST_PORTABLE_COMPUTE_SHADER_DIR) / ("CAbiComputeWrite." + extension);
        INFO("shader=" << path.string());
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        REQUIRE(stream);
        const auto end = stream.tellg();
        REQUIRE(end > 0);
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        REQUIRE(stream);
        return bytes;
    }
#endif

    constexpr std::uint32_t matrixLayoutOutputCount = 116;
    constexpr std::uint64_t matrixLayoutConstantBytes = 256;
    constexpr std::uint32_t matrixLayoutPushBytes = 128;
    constexpr std::uint32_t matrixLayoutStructuredElements = 2;

    struct MatrixLayoutPlacement final
    {
        std::uint32_t matrix = 0;
        std::uint32_t rows = 0;
        std::size_t offset = 0;
        std::uint32_t columnStride = 0;
        std::uint32_t firstOutput = 0;
    };

    void placeColumnMajorMatrix(std::span<std::byte> bytes,
                                const MatrixLayoutPlacement& placement,
                                std::span<std::uint32_t> expected)
    {
        constexpr std::uint32_t columns = 4;
        for (std::uint32_t row = 0; row < placement.rows; ++row)
        {
            for (std::uint32_t column = 0; column < columns; ++column)
            {
                const auto value = static_cast<float>((placement.matrix * 100U) + (row * 10U) + column + 1U);
                const auto byteOffset = placement.offset + (column * placement.columnStride) + (row * sizeof(float));
                REQUIRE(byteOffset + sizeof(float) <= bytes.size());
                std::memcpy(bytes.data() + byteOffset, &value, sizeof(value));
                expected[placement.firstOutput + (row * columns) + column] = std::bit_cast<std::uint32_t>(value);
            }
        }
    }

} // namespace

TEST_CASE("Portable handle provenance rejects cross-device and post-destroy handles", RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDeviceA = createPortableDeviceOrSkip(spec);
    if (!maybeDeviceA)
    {
        return;
    }
    auto maybeDeviceB = createPortableDeviceOrSkip(spec);
    REQUIRE(maybeDeviceB.has_value());

    auto& deviceA = *maybeDeviceA;
    auto& deviceB = *maybeDeviceB;

    CHECK_FALSE(deviceA.destroyBuffer({}).has_value());
    CHECK_FALSE(deviceA.destroyTexture({}).has_value());
    CHECK_FALSE(deviceA.destroyTextureView({}).has_value());
    CHECK_FALSE(deviceA.destroySampler({}).has_value());
    CHECK_FALSE(deviceA.destroyCommandPool({}).has_value());
    CHECK_FALSE(deviceA.destroyBindingLayout({}).has_value());
    CHECK_FALSE(deviceA.destroyBindingPool({}).has_value());
    CHECK_FALSE(deviceA.destroyPipelineLayout({}).has_value());
    CHECK_FALSE(deviceA.destroyQueryPool({}).has_value());

    auto bufferA = deviceA.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Portable provenance buffer",
    });
    REQUIRE(bufferA.has_value());

    auto textureA = deviceA.createTexture({
        .dimension = rachis::TextureDimension::texture2D,
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        .label = "Portable provenance texture",
    });
    REQUIRE(textureA.has_value());

    auto viewA = deviceA.createTextureView({
        .texture = *textureA,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
        .label = "Portable provenance view",
    });
    REQUIRE(viewA.has_value());

    auto samplerA = deviceA.createSampler({.label = "Portable provenance sampler"});
    REQUIRE(samplerA.has_value());

    auto poolA = deviceA.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Portable provenance command pool",
    });
    REQUIRE(poolA.has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayoutA = deviceA.createBindingLayout({
        .entries = layoutEntries,
        .label = "Portable provenance binding layout",
    });
    REQUIRE(bindingLayoutA.has_value());

    const std::array poolSizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::uniformBuffer, .descriptorCount = 2},
    };
    auto bindingPoolA = deviceA.createBindingPool({
        .maxBindingSets = 1,
        .sizes = poolSizes,
        .label = "Portable provenance binding pool",
    });
    REQUIRE(bindingPoolA.has_value());

    const std::array setLayouts{*bindingLayoutA};
    auto pipelineLayoutA = deviceA.createPipelineLayout({
        .bindingLayouts = setLayouts,
        .label = "Portable provenance pipeline layout",
    });
    REQUIRE(pipelineLayoutA.has_value());

    REQUIRE(deviceA.queueProperties(rachis::QueueType::graphics).timestampQueries);
    auto queryPoolA = deviceA.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 2,
        .label = "Portable provenance timestamp pool",
    });
    REQUIRE(queryPoolA.has_value());

    const std::array<std::byte, 16> payload{std::byte{0x42}};
    auto foreignWrite = deviceB.writeBuffer(*bufferA, 0, payload);
    REQUIRE_FALSE(foreignWrite.has_value());
    CHECK(foreignWrite.error().code == rachis::ErrorCode::invalidArgument);

    auto foreignDestroyBuffer = deviceB.destroyBuffer(*bufferA);
    REQUIRE_FALSE(foreignDestroyBuffer.has_value());
    CHECK(foreignDestroyBuffer.error().code == rachis::ErrorCode::invalidArgument);

    auto foreignView = deviceB.createTextureView({
        .texture = *textureA,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE_FALSE(foreignView.has_value());
    CHECK(foreignView.error().code == rachis::ErrorCode::invalidArgument);

    CHECK(deviceB.destroyTextureView(*viewA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceB.destroyTexture(*textureA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceB.destroySampler(*samplerA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceB.acquireCommandBuffer(*poolA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceB.resetCommandPool(*poolA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceB.destroyCommandPool(*poolA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceB.destroyBindingPool(*bindingPoolA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceB.destroyBindingLayout(*bindingLayoutA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceB.destroyPipelineLayout(*pipelineLayoutA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceB.destroyQueryPool(*queryPoolA).error().code == rachis::ErrorCode::invalidArgument);

    auto bufferB = deviceB.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(bufferB.has_value());
    auto cmdB = rachis::test::acquireCommandBuffer(deviceB, rachis::QueueType::graphics);
    REQUIRE(cmdB.has_value());
    auto foreignCopy = deviceB.copyBuffer(*cmdB,
                                          rachis::BufferCopy{
                                              .source = *bufferA,
                                              .sourceOffset = 0,
                                              .destination = *bufferB,
                                              .destinationOffset = 0,
                                              .size = 16,
                                          });
    REQUIRE_FALSE(foreignCopy.has_value());
    CHECK(foreignCopy.error().code == rachis::ErrorCode::invalidArgument);
    auto foreignResetQuery = deviceB.resetQueryPool(*cmdB, *queryPoolA, 0, 1);
    REQUIRE_FALSE(foreignResetQuery.has_value());
    CHECK(foreignResetQuery.error().code == rachis::ErrorCode::invalidArgument);
    cmdB = std::unexpected(rachis::Error{});
    REQUIRE(deviceB.destroyBuffer(*bufferB).has_value());

    REQUIRE(deviceA.writeBuffer(*bufferA, 0, payload).has_value());
    auto acquiredA = deviceA.acquireCommandBuffer(*poolA);
    REQUIRE(acquiredA.has_value());
    acquiredA = std::unexpected(rachis::Error{});

    REQUIRE(deviceA.destroyQueryPool(*queryPoolA).has_value());
    REQUIRE(deviceA.destroyPipelineLayout(*pipelineLayoutA).has_value());
    REQUIRE(deviceA.destroyBindingPool(*bindingPoolA).has_value());
    REQUIRE(deviceA.destroyBindingLayout(*bindingLayoutA).has_value());
    REQUIRE(deviceA.destroyCommandPool(*poolA).has_value());
    REQUIRE(deviceA.destroySampler(*samplerA).has_value());
    REQUIRE(deviceA.destroyTextureView(*viewA).has_value());
    REQUIRE(deviceA.destroyTexture(*textureA).has_value());
    REQUIRE(deviceA.destroyBuffer(*bufferA).has_value());

    CHECK(deviceA.writeBuffer(*bufferA, 0, payload).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceA.destroyBuffer(*bufferA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceA.destroyTextureView(*viewA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceA.destroyTexture(*textureA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceA.destroySampler(*samplerA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceA.acquireCommandBuffer(*poolA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceA.destroyCommandPool(*poolA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceA.destroyBindingPool(*bindingPoolA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceA.destroyBindingLayout(*bindingLayoutA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceA.destroyPipelineLayout(*pipelineLayoutA).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(deviceA.destroyQueryPool(*queryPoolA).error().code == rachis::ErrorCode::invalidArgument);

    checkNoDiagnosticsErrors(deviceA);
    checkNoDiagnosticsErrors(deviceB);
}

TEST_CASE("Portable buffers validate host ranges, mappings, and GPU copy round-trips", RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    auto zeroSize = device.createBuffer({
        .size = 0,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE_FALSE(zeroSize.has_value());
    CHECK(zeroSize.error().code == rachis::ErrorCode::invalidArgument);

    auto noUsage = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::none,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    REQUIRE_FALSE(noUsage.has_value());
    CHECK(noUsage.error().code == rachis::ErrorCode::invalidArgument);

    constexpr std::uint64_t totalBytes = 128;
    constexpr std::uint64_t halfBytes = totalBytes / 2;

    auto upload = device.createBuffer({
        .size = totalBytes,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Portable upload buffer",
    });
    auto scratch = device.createBuffer({
        .size = totalBytes,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Portable device-local buffer",
    });
    auto readback = device.createBuffer({
        .size = totalBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Portable readback buffer",
    });
    REQUIRE(upload.has_value());
    REQUIRE(scratch.has_value());
    REQUIRE(readback.has_value());

    CHECK_FALSE(device.mapBuffer(*scratch).has_value());
    CHECK(device.unmapBuffer(*upload).error().code == rachis::ErrorCode::invalidState);

    std::array<std::byte, totalBytes> expected{};
    for (std::uint64_t index = 0; index < totalBytes; ++index)
    {
        expected[index] = patternByte(index);
    }

    const std::array<std::byte, 16> overflowData{};
    CHECK(device.writeBuffer(*upload, totalBytes - 8, overflowData).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(device.mapBuffer(*upload, 0, 0).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(device.mapBuffer(*upload, totalBytes + 1, 8).error().code == rachis::ErrorCode::invalidArgument);

    auto mappedUpload = device.mapBuffer(*upload, 0, rachis::wholeResource);
    REQUIRE(mappedUpload.has_value());
    REQUIRE(mappedUpload->data != nullptr);
    CHECK(mappedUpload->size == totalBytes);
    CHECK(device.mapBuffer(*upload).error().code == rachis::ErrorCode::invalidState);
    std::memcpy(mappedUpload->data, expected.data(), halfBytes);
    const std::array wholeRange{rachis::MappedMemoryRange{.offset = 0, .size = rachis::wholeResource}};
    REQUIRE(device.flushMappedMemoryRanges(*upload, wholeRange).has_value());
    REQUIRE(device.unmapBuffer(*upload).has_value());

    REQUIRE(
        device.writeBuffer(*upload, halfBytes, std::span<const std::byte>(expected).subspan(halfBytes)).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *upload,
                                .sourceOffset = 0,
                                .destination = *scratch,
                                .destinationOffset = 0,
                                .size = totalBytes,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *scratch,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *scratch,
                                .sourceOffset = 0,
                                .destination = *readback,
                                .destinationOffset = 0,
                                .size = totalBytes,
                            })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    CHECK(submitted->isValid());
    CHECK(submitted->queueType() == rachis::QueueType::graphics);
    REQUIRE(device.wait(*submitted).has_value());

    auto timelineStatus = device.queryTimelinePoint(*submitted);
    REQUIRE(timelineStatus.has_value());
    CHECK(*timelineStatus == rachis::TimelinePointStatus::complete);

    std::array<std::byte, 16> overflowRead{};
    CHECK(device.readBuffer(*readback, totalBytes - 8, overflowRead).error().code ==
          rachis::ErrorCode::invalidArgument);

    std::array<std::byte, totalBytes> actualFromRead{};
    REQUIRE(device.readBuffer(*readback, 0, actualFromRead).has_value());
    CHECK(actualFromRead == expected);

    auto mappedReadback = device.mapBuffer(*readback, 0, rachis::wholeResource);
    REQUIRE(mappedReadback.has_value());
    REQUIRE(device.invalidateMappedMemoryRanges(*readback, wholeRange).has_value());
    std::array<std::byte, totalBytes> actualFromMap{};
    std::memcpy(actualFromMap.data(), mappedReadback->data, totalBytes);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    CHECK(actualFromMap == expected);

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*scratch).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable textures validate views, transfer layouts, and GPU copy round-trips", RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    CHECK(device
              .createTexture({
                  .width = 0,
                  .height = 4,
                  .format = rachis::Format::rgba8Unorm,
                  .usage = rachis::TextureUsage::sampled,
              })
              .error()
              .code == rachis::ErrorCode::invalidArgument);
    CHECK(device
              .createTexture({
                  .width = 4,
                  .height = 4,
                  .format = rachis::Format::unknown,
                  .usage = rachis::TextureUsage::sampled,
              })
              .error()
              .code == rachis::ErrorCode::invalidArgument);
    CHECK(device
              .createTexture({
                  .width = 4,
                  .height = 4,
                  .format = rachis::Format::rgba8Unorm,
                  .usage = rachis::TextureUsage::none,
              })
              .error()
              .code == rachis::ErrorCode::invalidArgument);

    constexpr std::uint32_t width = 5;
    constexpr std::uint32_t height = 3;
    constexpr std::uint32_t pixelBytes = 4;

    auto texture = device.createTexture({
        .dimension = rachis::TextureDimension::texture2D,
        .width = width,
        .height = height,
        .depthOrArrayLayers = 1,
        .mipLevels = 1,
        .sampleCount = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
        .label = "Portable transfer texture",
    });
    REQUIRE(texture.has_value());

    auto sampledView = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
        .label = "Portable sampled view",
    });
    REQUIRE(sampledView.has_value());

    auto undeclaredFormatView = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
        .format = rachis::Format::rgba8UnormSrgb,
    });
    REQUIRE_FALSE(undeclaredFormatView.has_value());
    CHECK(undeclaredFormatView.error().code == rachis::ErrorCode::invalidArgument);

    auto outOfRangeMipView = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
        .firstMipLevel = 1,
        .mipLevelCount = 1,
    });
    REQUIRE_FALSE(outOfRangeMipView.has_value());
    CHECK(outOfRangeMipView.error().code == rachis::ErrorCode::invalidArgument);

    auto missingStorageUsageView = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::storage,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE_FALSE(missingStorageUsageView.has_value());
    CHECK(missingStorageUsageView.error().code == rachis::ErrorCode::invalidArgument);

    const rachis::BufferTextureCopyRegion region{
        .bufferOffset = 0,
        .extent = {.width = width, .height = height, .depth = 1},
    };
    auto layout = device.queryTextureTransferLayout(*texture, region);
    REQUIRE(layout.has_value());
    CHECK(layout->rowCount == height);
    CHECK(layout->rowDataSize == static_cast<std::uint64_t>(width) * pixelBytes);
    CHECK(layout->bytesPerRow >= layout->rowDataSize);
    CHECK(layout->rowPitchAlignment >= 1);
    CHECK(layout->bytesPerRow % layout->rowPitchAlignment == 0);
    CHECK(layout->rowsPerImage == height);
    CHECK(layout->requiredBufferSize >= layout->rowDataSize);

    auto outOfBoundsRegion = region;
    outOfBoundsRegion.textureOrigin = {.x = width, .y = 0, .z = 0};
    auto invalidLayout = device.queryTextureTransferLayout(*texture, outOfBoundsRegion);
    REQUIRE_FALSE(invalidLayout.has_value());
    CHECK(invalidLayout.error().code == rachis::ErrorCode::invalidArgument);

    std::vector<std::byte> uploadBytes(static_cast<std::size_t>(layout->requiredBufferSize), std::byte{0xA5});
    for (std::uint32_t y = 0; y < height; ++y)
    {
        for (std::uint32_t x = 0; x < width; ++x)
        {
            for (std::uint32_t channel = 0; channel < pixelBytes; ++channel)
            {
                uploadBytes[static_cast<std::size_t>(y) * layout->bytesPerRow +
                            static_cast<std::size_t>(x) * pixelBytes + channel] = texelByte(x, y, channel);
            }
        }
    }

    auto upload = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Portable texture upload",
    });
    auto readback = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Portable texture readback",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(device.writeBuffer(*upload, 0, uploadBytes).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const std::array undefinedAfterBarriers{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::none,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::none,
            .accessAfter = rachis::Access::copyDestination,
            .layoutBefore = rachis::TextureLayout::undefined,
            .layoutAfter = rachis::TextureLayout::undefined,
        },
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::none,
            .stagesAfter = rachis::PipelineStage::none,
            .accessBefore = rachis::Access::none,
            .accessAfter = rachis::Access::none,
            .layoutBefore = rachis::TextureLayout::undefined,
            .layoutAfter = rachis::TextureLayout::undefined,
        },
    };
    for (const auto& barrier : undefinedAfterBarriers)
    {
        const auto rejected = device.textureBarrier(*command, barrier);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    }
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::none,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());

    const rachis::BufferTextureCopy copyIn{
        .buffer = *upload,
        .bufferOffset = 0,
        .bytesPerRow = layout->bytesPerRow,
        .rowsPerImage = layout->rowsPerImage,
        .texture = *texture,
        .extent = {.width = width, .height = height, .depth = 1},
    };
    REQUIRE(device.copyBufferToTexture(*command, copyIn).has_value());
    const auto toUndefined = device.textureBarrier(*command,
                                                   rachis::TextureBarrier{
                                                       .texture = *texture,
                                                       .stagesBefore = rachis::PipelineStage::copy,
                                                       .stagesAfter = rachis::PipelineStage::none,
                                                       .accessBefore = rachis::Access::copyDestination,
                                                       .accessAfter = rachis::Access::none,
                                                       .layoutBefore = rachis::TextureLayout::copyDestination,
                                                       .layoutAfter = rachis::TextureLayout::undefined,
                                                   });
    REQUIRE_FALSE(toUndefined.has_value());
    CHECK(toUndefined.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());

    const rachis::BufferTextureCopy copyOut{
        .buffer = *readback,
        .bufferOffset = 0,
        .bytesPerRow = layout->bytesPerRow,
        .rowsPerImage = layout->rowsPerImage,
        .texture = *texture,
        .extent = {.width = width, .height = height, .depth = 1},
    };
    REQUIRE(device.copyTextureToBuffer(*command, copyOut).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    std::vector<std::byte> actualBytes(static_cast<std::size_t>(layout->requiredBufferSize), std::byte{0});
    REQUIRE(device.readBuffer(*readback, 0, actualBytes).has_value());
    for (std::uint32_t y = 0; y < height; ++y)
    {
        for (std::uint32_t x = 0; x < width; ++x)
        {
            for (std::uint32_t channel = 0; channel < pixelBytes; ++channel)
            {
                const auto index = static_cast<std::size_t>(y) * layout->bytesPerRow +
                                   static_cast<std::size_t>(x) * pixelBytes + channel;
                CHECK(actualBytes[index] == texelByte(x, y, channel));
            }
        }
    }

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyTextureView(*sampledView).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable 1D textures accept one mip everywhere and report Metal's single-mip limit as unsupported",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;
    REQUIRE(rachis::hasAny(device.queryFormatProperties(rachis::Format::rgba8Unorm).features,
                           rachis::FormatFeature::texture1D));

    const auto textureDesc = [](std::uint32_t layers, std::uint32_t mipLevels) {
        return rachis::TextureDesc{
            .dimension = rachis::TextureDimension::texture1D,
            .width = 64,
            .height = 1,
            .depthOrArrayLayers = layers,
            .mipLevels = mipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copySource |
                     rachis::TextureUsage::copyDestination,
            .label = "Portable 1D texture",
        };
    };

    for (const std::uint32_t layers : {1U, 4U})
    {
        for (const std::uint32_t mipLevels : {1U, 7U})
        {
            CAPTURE(layers, mipLevels);
            const auto desc = textureDesc(layers, mipLevels);
            const bool unsupported = spec.backend == rachis::Backend::metal && mipLevels != 1;

            auto requirements = device.queryTextureMemoryRequirements(desc);
            auto texture = device.createTexture(desc);
            if (unsupported)
            {
                REQUIRE_FALSE(requirements.has_value());
                CHECK(requirements.error().code == rachis::ErrorCode::unsupported);
                REQUIRE_FALSE(texture.has_value());
                CHECK(texture.error().code == rachis::ErrorCode::unsupported);
                requirements = device.queryTextureMemoryRequirements(textureDesc(layers, 1));
            }
            else
            {
                REQUIRE(texture.has_value());
            }
            REQUIRE(requirements.has_value());

            auto heap = device.createResourceHeap({
                .size = requirements->size,
                .alignment = requirements->alignment,
                .memoryType = rachis::MemoryType::deviceLocal,
                .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
                .label = "Portable 1D texture heap",
            });
            REQUIRE(heap.has_value());
            auto placed = device.createPlacedTexture({.heap = *heap, .offset = 0}, desc);
            if (unsupported)
            {
                REQUIRE_FALSE(placed.has_value());
                CHECK(placed.error().code == rachis::ErrorCode::unsupported);
            }
            else
            {
                REQUIRE(placed.has_value());
                REQUIRE(device.destroyTexture(*placed).has_value());
                REQUIRE(device.destroyTexture(*texture).has_value());
            }
            REQUIRE(device.destroyResourceHeap(*heap).has_value());
        }
    }
    device.collectGarbage();
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable 1D render targets clear on D3D12 and Vulkan and are unsupported on Metal",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    constexpr std::uint32_t width = 64;
    constexpr std::uint32_t pixelBytes = 4;
    constexpr std::array expected{std::byte{0x00}, std::byte{0x00}, std::byte{0xFF}, std::byte{0xFF}};

    for (const bool oneDimensional : {false, true})
    {
        CAPTURE(oneDimensional);
        const rachis::TextureDesc desc{
            .dimension = oneDimensional ? rachis::TextureDimension::texture1D : rachis::TextureDimension::texture2D,
            .width = width,
            .height = 1,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
            .label = "Portable line render target",
        };
        if (oneDimensional && spec.backend == rachis::Backend::metal)
        {
            auto requirements = device.queryTextureMemoryRequirements(desc);
            REQUIRE_FALSE(requirements.has_value());
            CHECK(requirements.error().code == rachis::ErrorCode::unsupported);
            auto rejected = device.createTexture(desc);
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
            continue;
        }

        auto texture = device.createTexture(desc);
        REQUIRE(texture.has_value());
        auto view = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension =
                oneDimensional ? rachis::TextureViewDimension::texture1D : rachis::TextureViewDimension::texture2D,
            .label = "Portable line render-target view",
        });
        REQUIRE(view.has_value());

        const rachis::BufferTextureCopyRegion region{.extent = {.width = width, .height = 1, .depth = 1}};
        auto layout = device.queryTextureTransferLayout(*texture, region);
        REQUIRE(layout.has_value());
        auto readback = device.createBuffer({
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Portable line render target readback",
        });
        REQUIRE(readback.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::none,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *view,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
                .clearValue = {.red = 0.0F, .green = 0.0F, .blue = 1.0F, .alpha = 1.0F},
            },
        };
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         rachis::BufferTextureCopy{
                                             .buffer = *readback,
                                             .bytesPerRow = layout->bytesPerRow,
                                             .rowsPerImage = layout->rowsPerImage,
                                             .texture = *texture,
                                             .extent = region.extent,
                                         })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        std::vector<std::byte> bytes(static_cast<std::size_t>(layout->requiredBufferSize), std::byte{0});
        REQUIRE(device.readBuffer(*readback, 0, bytes).has_value());
        std::uint32_t cleared = 0;
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const auto texel = std::span{bytes}.subspan(static_cast<std::size_t>(x) * pixelBytes, pixelBytes);
            if (std::ranges::equal(texel, expected))
            {
                ++cleared;
            }
        }
        CHECK(cleared == width);

        REQUIRE(device.destroyBuffer(*readback).has_value());
        REQUIRE(device.destroyTextureView(*view).has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
    }
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable ownership barriers require shared resources and the common layout", RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    const rachis::BufferDesc bufferDesc{
        .size = 256,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Portable ownership buffer",
    };
    const rachis::TextureDesc textureDesc{
        .dimension = rachis::TextureDimension::texture2D,
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Portable ownership texture",
    };
    const auto bufferBarrier = [](rachis::BufferHandle buffer, rachis::OwnershipTransfer ownership) {
        const bool release = ownership == rachis::OwnershipTransfer::releaseExternal;
        return rachis::BufferBarrier{
            .buffer = buffer,
            .stagesBefore = release ? rachis::PipelineStage::all : rachis::PipelineStage::none,
            .stagesAfter = release ? rachis::PipelineStage::none : rachis::PipelineStage::all,
            .accessBefore = release ? rachis::Access::common : rachis::Access::none,
            .accessAfter = release ? rachis::Access::none : rachis::Access::common,
            .ownership = ownership,
        };
    };
    const auto textureBarrier = [](rachis::TextureHandle texture, rachis::OwnershipTransfer ownership) {
        const bool release = ownership == rachis::OwnershipTransfer::releaseExternal;
        return rachis::TextureBarrier{
            .texture = texture,
            .stagesBefore = release ? rachis::PipelineStage::all : rachis::PipelineStage::none,
            .stagesAfter = release ? rachis::PipelineStage::none : rachis::PipelineStage::all,
            .accessBefore = release ? rachis::Access::common : rachis::Access::none,
            .accessAfter = release ? rachis::Access::none : rachis::Access::common,
            .layoutBefore = rachis::TextureLayout::common,
            .layoutAfter = rachis::TextureLayout::common,
            .ownership = ownership,
        };
    };
    const auto requireInvalidArgument = [](const rachis::Status& status) {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == rachis::ErrorCode::invalidArgument);
    };

    auto buffer = device.createBuffer(bufferDesc);
    auto texture = device.createTexture(textureDesc);
    REQUIRE(buffer.has_value());
    REQUIRE(texture.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());

    const auto unknownOwnership = static_cast<rachis::OwnershipTransfer>(3);
    for (const auto ownership :
         {rachis::OwnershipTransfer::releaseExternal, rachis::OwnershipTransfer::acquireExternal, unknownOwnership})
    {
        CAPTURE(static_cast<int>(ownership));
        requireInvalidArgument(device.bufferBarrier(*command, bufferBarrier(*buffer, ownership)));
        requireInvalidArgument(device.textureBarrier(*command, textureBarrier(*texture, ownership)));
    }

    const auto capabilities = rachis::test::requireCapabilities(device);
    if (capabilities.sharedBufferHandles && capabilities.sharedTextureHandles)
    {
        auto sharedBuffer = device.createSharedBuffer(bufferDesc);
        auto sharedTexture = device.createSharedTexture(textureDesc);
        REQUIRE(sharedBuffer.has_value());
        REQUIRE(sharedTexture.has_value());

        requireInvalidArgument(device.bufferBarrier(*command, bufferBarrier(*sharedBuffer, unknownOwnership)));
        requireInvalidArgument(device.textureBarrier(*command, textureBarrier(*sharedTexture, unknownOwnership)));
        auto fromCopySource = textureBarrier(*sharedTexture, rachis::OwnershipTransfer::releaseExternal);
        fromCopySource.layoutBefore = rachis::TextureLayout::copySource;
        requireInvalidArgument(device.textureBarrier(*command, fromCopySource));
        auto toCopyDestination = textureBarrier(*sharedTexture, rachis::OwnershipTransfer::acquireExternal);
        toCopyDestination.layoutAfter = rachis::TextureLayout::copyDestination;
        requireInvalidArgument(device.textureBarrier(*command, toCopyDestination));

        for (const auto ownership :
             {rachis::OwnershipTransfer::releaseExternal, rachis::OwnershipTransfer::acquireExternal})
        {
            CAPTURE(static_cast<int>(ownership));
            if (!command)
            {
                command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
                REQUIRE(command.has_value());
            }
            REQUIRE(device.bufferBarrier(*command, bufferBarrier(*sharedBuffer, ownership)).has_value());
            REQUIRE(device.textureBarrier(*command, textureBarrier(*sharedTexture, ownership)).has_value());
            auto submitted = device.submit(std::move(*command));
            REQUIRE(submitted.has_value());
            REQUIRE(device.wait(*submitted).has_value());
            command = std::unexpected(rachis::Error{});
        }
        REQUIRE(device.destroyTexture(*sharedTexture).has_value());
        REQUIRE(device.destroyBuffer(*sharedBuffer).has_value());
    }
    command = std::unexpected(rachis::Error{});

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable formats that advertise sampled 2D support create sampled textures and views",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    std::uint32_t sampledFormatCount = 0;
    for (auto raw = static_cast<std::uint8_t>(rachis::Format::r8Unorm);
         raw <= static_cast<std::uint8_t>(rachis::Format::bc6hRgbSfloat);
         ++raw)
    {
        const auto format = static_cast<rachis::Format>(raw);
        CAPTURE(static_cast<unsigned int>(raw));
        const auto properties = device.queryFormatProperties(format);
        if (!rachis::hasAll(properties.features, rachis::FormatFeature::sampled | rachis::FormatFeature::texture2D))
        {
            continue;
        }
        const bool depth = format == rachis::Format::d32Float || format == rachis::Format::d24UnormS8Uint;
        if (depth && !rachis::hasAny(properties.features, rachis::FormatFeature::depthStencilAttachment))
        {
            continue;
        }
        ++sampledFormatCount;
        auto texture = device.createTexture({
            .width = 4,
            .height = 4,
            .format = format,
            .usage = depth ? rachis::TextureUsage::sampled | rachis::TextureUsage::depthStencil
                           : rachis::TextureUsage::sampled,
            .label = "Portable advertised sampled texture",
        });
        INFO("createTexture: " << (texture ? std::string{} : texture.error().message));
        REQUIRE(texture.has_value());
        auto view = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::sampled,
            .aspect = depth ? rachis::TextureAspect::depth : rachis::TextureAspect::color,
            .label = "Portable advertised sampled view",
        });
        INFO("createTextureView: " << (view ? std::string{} : view.error().message));
        REQUIRE(view.has_value());
        REQUIRE(device.destroyTextureView(*view).has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
    }
    CHECK(sampledFormatCount > 0);
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable formats that advertise 3D support create 3D textures", RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    std::uint32_t volumeFormatCount = 0;
    for (auto raw = static_cast<std::uint8_t>(rachis::Format::r8Unorm);
         raw <= static_cast<std::uint8_t>(rachis::Format::bc6hRgbSfloat);
         ++raw)
    {
        const auto format = static_cast<rachis::Format>(raw);
        CAPTURE(static_cast<unsigned int>(raw));
        const auto properties = device.queryFormatProperties(format);
        if (!rachis::hasAny(properties.features, rachis::FormatFeature::texture3D))
        {
            continue;
        }
        const bool depth = format == rachis::Format::d32Float || format == rachis::Format::d24UnormS8Uint;
        auto usage = rachis::TextureUsage::copyDestination;
        if (depth)
        {
            usage = rachis::TextureUsage::depthStencil;
        }
        else if (rachis::hasAny(properties.features, rachis::FormatFeature::sampled))
        {
            usage = rachis::TextureUsage::sampled;
        }
        ++volumeFormatCount;
        auto texture = device.createTexture({
            .dimension = rachis::TextureDimension::texture3D,
            .width = 4,
            .height = 4,
            .depthOrArrayLayers = 4,
            .format = format,
            .usage = usage,
            .label = "Portable advertised 3D texture",
        });
        INFO("createTexture: " << (texture ? std::string{} : texture.error().message));
        REQUIRE(texture.has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
    }
    CHECK(volumeFormatCount > 0);
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable formats that advertise multisampled color attachments create multisampled render targets",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    std::uint32_t multisampledTargetCount = 0;
    for (auto raw = static_cast<std::uint8_t>(rachis::Format::r8Unorm);
         raw <= static_cast<std::uint8_t>(rachis::Format::bc6hRgbSfloat);
         ++raw)
    {
        const auto format = static_cast<rachis::Format>(raw);
        CAPTURE(static_cast<unsigned int>(raw));
        const auto properties = device.queryFormatProperties(format);
        if (!rachis::hasAll(properties.features,
                            rachis::FormatFeature::colorAttachment | rachis::FormatFeature::texture2D))
        {
            continue;
        }
        for (const auto sampleCount : {2U, 4U, 8U})
        {
            if (!rachis::supportsSampleCount(properties.sampleCounts, sampleCount))
            {
                continue;
            }
            CAPTURE(sampleCount);
            auto texture = device.createTexture({
                .width = 8,
                .height = 8,
                .sampleCount = sampleCount,
                .format = format,
                .usage = rachis::TextureUsage::renderTarget,
                .label = "Portable advertised multisampled render target",
            });
            INFO("createTexture: " << (texture ? std::string{} : texture.error().message));
            REQUIRE(texture.has_value());
            ++multisampledTargetCount;
            REQUIRE(device.destroyTexture(*texture).has_value());
        }
    }
    CHECK(multisampledTargetCount > 0);
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable integer color attachments clear to truncated and saturated integers on load and explicit clears",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    struct IntegerClearCase final
    {
        IntegerClearFormat format = {};
        rachis::ClearColor loadClear = {};
        std::vector<std::int64_t> loadExpected = {};
        rachis::ClearColor attachmentClear = {};
        std::vector<std::int64_t> attachmentExpected = {};
    };
    const std::array cases{
        IntegerClearCase{
            .format = {"r32Uint", rachis::Format::r32Uint, 1, 4, false},
            .loadClear = {1.0F, 0.0F, 0.0F, 0.0F},
            .loadExpected = {1},
            .attachmentClear = {4294967040.0F, 0.0F, 0.0F, 0.0F},
            .attachmentExpected = {4294967040},
        },
        IntegerClearCase{
            .format = {"rgba8Uint", rachis::Format::rgba8Uint, 4, 1, false},
            .loadClear = {1.0F, 2.0F, 254.0F, 255.0F},
            .loadExpected = {1, 2, 254, 255},
            .attachmentClear = {300.0F, -5.0F, 7.9F, 0.5F},
            .attachmentExpected = {255, 0, 7, 0},
        },
        IntegerClearCase{
            .format = {"rgba8Sint", rachis::Format::rgba8Sint, 4, 1, true},
            .loadClear = {-1.0F, 2.0F, -128.0F, 127.0F},
            .loadExpected = {-1, 2, -128, 127},
            .attachmentClear = {200.0F, -200.0F, -7.9F, 7.9F},
            .attachmentExpected = {127, -128, -7, 7},
        },
        IntegerClearCase{
            .format = {"rgba16Uint", rachis::Format::rgba16Uint, 4, 2, false},
            .loadClear = {1.0F, 300.0F, 65535.0F, 0.0F},
            .loadExpected = {1, 300, 65535, 0},
            .attachmentClear = {70000.0F, -1.0F, 1234.5F, 2.0F},
            .attachmentExpected = {65535, 0, 1234, 2},
        },
        IntegerClearCase{
            .format = {"rgba16Sint", rachis::Format::rgba16Sint, 4, 2, true},
            .loadClear = {-1.0F, 300.0F, -32768.0F, 32767.0F},
            .loadExpected = {-1, 300, -32768, 32767},
            .attachmentClear = {40000.0F, -40000.0F, -1234.5F, 3.0F},
            .attachmentExpected = {32767, -32768, -1234, 3},
        },
        IntegerClearCase{
            .format = {"rgba32Uint", rachis::Format::rgba32Uint, 4, 4, false},
            .loadClear = {1.0F, 2.0F, 70000.0F, 4294967040.0F},
            .loadExpected = {1, 2, 70000, 4294967040},
            .attachmentClear = {-1.0F, 16777216.0F, 2.5F, 3.0F},
            .attachmentExpected = {0, 16777216, 2, 3},
        },
        IntegerClearCase{
            .format = {"rgba32Sint", rachis::Format::rgba32Sint, 4, 4, true},
            .loadClear = {-1.0F, 2.0F, -2147483648.0F, 2147483520.0F},
            .loadExpected = {-1, 2, -2147483648, 2147483520},
            .attachmentClear = {-2.5F, 2.5F, 70000.0F, -70000.0F},
            .attachmentExpected = {-2, 2, 70000, -70000},
        },
    };

    for (const auto& clearCase : cases)
    {
        CAPTURE(clearCase.format.name);
        CHECK(clearIntegerTargetAndReadBack(device, clearCase.format, clearCase.loadClear, std::nullopt) ==
              clearCase.loadExpected);
        CHECK(clearIntegerTargetAndReadBack(device, clearCase.format, clearCase.loadClear, clearCase.attachmentClear) ==
              clearCase.attachmentExpected);
    }
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable command pools enforce capacity, resets, abandon, and ordered batch submission",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    CHECK(device.createCommandPool({}).error().code == rachis::ErrorCode::invalidArgument);
    const std::string oversizedLabel(rachis::maxDebugLabelBytes + 1, 'z');
    CHECK(device
              .createCommandPool({
                  .queueType = rachis::QueueType::graphics,
                  .maxCommandBuffers = 1,
                  .label = oversizedLabel,
              })
              .error()
              .code == rachis::ErrorCode::invalidArgument);

    auto boundedPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Portable bounded command pool",
    });
    REQUIRE(boundedPool.has_value());
    auto firstLease = device.acquireCommandBuffer(*boundedPool);
    REQUIRE(firstLease.has_value());
    CHECK(device.acquireCommandBuffer(*boundedPool).error().code == rachis::ErrorCode::resourceExhausted);
    CHECK(device.resetCommandBuffer(*firstLease).error().code == rachis::ErrorCode::invalidState);

    *firstLease = rachis::CommandBuffer{};
    CHECK_FALSE(firstLease->isValid());
    CHECK(device.insertDebugMarker(*firstLease, {.label = "Moved-from command"}).error().code ==
          rachis::ErrorCode::invalidState);

    auto recycledLease = device.acquireCommandBuffer(*boundedPool);
    REQUIRE(recycledLease.has_value());
    recycledLease = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyCommandPool(*boundedPool).has_value());

    constexpr std::uint64_t byteCount = 32;
    std::array<std::byte, byteCount> initialPattern{};
    std::array<std::byte, byteCount> discardedPattern{};
    std::array<std::byte, byteCount> firstPattern{};
    std::array<std::byte, byteCount> secondPattern{};
    initialPattern.fill(std::byte{0x5A});
    discardedPattern.fill(std::byte{0x11});
    firstPattern.fill(std::byte{0x33});
    secondPattern.fill(std::byte{0x7F});

    auto srcInitial = device.createBuffer({
        .size = byteCount,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto srcDiscarded = device.createBuffer({
        .size = byteCount,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto srcFirst = device.createBuffer({
        .size = byteCount,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto srcSecond = device.createBuffer({
        .size = byteCount,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto gpuScratch = device.createBuffer({
        .size = byteCount,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto dst = device.createBuffer({
        .size = byteCount,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(srcInitial.has_value());
    REQUIRE(srcDiscarded.has_value());
    REQUIRE(srcFirst.has_value());
    REQUIRE(srcSecond.has_value());
    REQUIRE(gpuScratch.has_value());
    REQUIRE(dst.has_value());
    REQUIRE(device.writeBuffer(*srcInitial, 0, initialPattern).has_value());
    REQUIRE(device.writeBuffer(*srcDiscarded, 0, discardedPattern).has_value());
    REQUIRE(device.writeBuffer(*srcFirst, 0, firstPattern).has_value());
    REQUIRE(device.writeBuffer(*srcSecond, 0, secondPattern).has_value());

    {
        auto initCmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(initCmd.has_value());
        REQUIRE(device
                    .copyBuffer(*initCmd,
                                rachis::BufferCopy{
                                    .source = *srcInitial,
                                    .sourceOffset = 0,
                                    .destination = *dst,
                                    .destinationOffset = 0,
                                    .size = byteCount,
                                })
                    .has_value());
        auto initSubmitted = device.submit(std::move(*initCmd));
        REQUIRE(initSubmitted.has_value());
        REQUIRE(device.wait(*initSubmitted).has_value());
    }

    auto resetPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 2,
        .label = "Portable resettable command pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(resetPool.has_value());
    auto cmd0 = device.acquireCommandBuffer(*resetPool);
    auto cmd1 = device.acquireCommandBuffer(*resetPool);
    REQUIRE(cmd0.has_value());
    REQUIRE(cmd1.has_value());

    REQUIRE(device.beginDebugGroup(*cmd0, {.label = "Open debug group"}).has_value());
    CHECK(device.resetCommandBuffer(*cmd0).error().code == rachis::ErrorCode::invalidState);
    CHECK(device.resetCommandPool(*resetPool).error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.endDebugGroup(*cmd0).has_value());

    REQUIRE(device
                .copyBuffer(*cmd0,
                            rachis::BufferCopy{
                                .source = *srcDiscarded,
                                .sourceOffset = 0,
                                .destination = *dst,
                                .destinationOffset = 0,
                                .size = byteCount,
                            })
                .has_value());

    REQUIRE(device.resetCommandPool(*resetPool).has_value());
    REQUIRE(cmd0->isValid());
    REQUIRE(cmd1->isValid());

    REQUIRE(device
                .copyBuffer(*cmd0,
                            rachis::BufferCopy{
                                .source = *srcFirst,
                                .sourceOffset = 0,
                                .destination = *gpuScratch,
                                .destinationOffset = 0,
                                .size = 16,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*cmd0,
                               rachis::BufferBarrier{
                                   .buffer = *gpuScratch,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*cmd1,
                            rachis::BufferCopy{
                                .source = *gpuScratch,
                                .sourceOffset = 0,
                                .destination = *dst,
                                .destinationOffset = 0,
                                .size = 16,
                            })
                .has_value());
    REQUIRE(device
                .copyBuffer(*cmd1,
                            rachis::BufferCopy{
                                .source = *srcSecond,
                                .sourceOffset = 16,
                                .destination = *dst,
                                .destinationOffset = 16,
                                .size = 8,
                            })
                .has_value());

    std::array batch{std::move(*cmd0), std::move(*cmd1)};
    auto submitted = device.submit(std::span<rachis::CommandBuffer>(batch));
    REQUIRE(submitted.has_value());
    CHECK_FALSE(batch[0].isValid());
    CHECK_FALSE(batch[1].isValid());
    CHECK_FALSE(device.submit(std::move(batch[0])).has_value());
    REQUIRE(device.wait(*submitted).has_value());

    std::array<std::byte, byteCount> actual{};
    REQUIRE(device.readBuffer(*dst, 0, actual).has_value());
    for (std::size_t index = 0; index < 16; ++index)
    {
        CHECK(actual[index] == std::byte{0x33});
    }
    for (std::size_t index = 16; index < 24; ++index)
    {
        CHECK(actual[index] == std::byte{0x7F});
    }
    for (std::size_t index = 24; index < byteCount; ++index)
    {
        CHECK(actual[index] == std::byte{0x5A});
    }

    REQUIRE(device.destroyCommandPool(*resetPool).has_value());
    REQUIRE(device.destroyBuffer(*dst).has_value());
    REQUIRE(device.destroyBuffer(*gpuScratch).has_value());
    REQUIRE(device.destroyBuffer(*srcSecond).has_value());
    REQUIRE(device.destroyBuffer(*srcFirst).has_value());
    REQUIRE(device.destroyBuffer(*srcDiscarded).has_value());
    REQUIRE(device.destroyBuffer(*srcInitial).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable samplers, binding layouts, pools, and sets enforce capacity and validation contracts",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    CHECK(device.createSampler({.maxAnisotropy = 0}).error().code == rachis::ErrorCode::invalidArgument);
    CHECK(device.createSampler({.minLod = 4.0F, .maxLod = 1.0F}).error().code == rachis::ErrorCode::invalidArgument);

    const auto maxLodBias = rachis::test::requireCapabilities(device).limits.maxSamplerLodBias;
    CAPTURE(maxLodBias);
    CHECK(maxLodBias >= 2.0F);
    const auto aboveMaxLodBias = std::nextafter(maxLodBias, std::numeric_limits<float>::infinity());
    for (const float bias : {20.0F, -20.0F, 100.0F, -100.0F, aboveMaxLodBias, -aboveMaxLodBias})
    {
        CAPTURE(bias);
        for (const auto mipFilter : {rachis::MipmapFilterMode::none, rachis::MipmapFilterMode::linear})
        {
            auto rejected = device.createSampler({.mipFilter = mipFilter, .lodBias = bias});
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
        }
    }
    for (const float bias : {maxLodBias, -maxLodBias})
    {
        CAPTURE(bias);
        auto accepted = device.createSampler({.mipFilter = rachis::MipmapFilterMode::linear, .lodBias = bias});
        REQUIRE(accepted.has_value());
        REQUIRE(device.destroySampler(*accepted).has_value());
    }

    const std::array duplicateBindings{
        rachis::BindingLayoutEntry{1, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
    };
    CHECK(device.createBindingLayout({.entries = duplicateBindings}).error().code ==
          rachis::ErrorCode::invalidArgument);

    const std::array overlappingArrayBindings{
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 4},
        rachis::BindingLayoutEntry{3, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 1},
    };
    CHECK(device.createBindingLayout({.entries = overlappingArrayBindings}).error().code ==
          rachis::ErrorCode::invalidArgument);

    const std::array zeroCountBinding{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 0},
    };
    CHECK(device.createBindingLayout({.entries = zeroCountBinding}).error().code == rachis::ErrorCode::invalidArgument);

    const std::array noStageBinding{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::none, 1},
    };
    CHECK(device.createBindingLayout({.entries = noStageBinding}).error().code == rachis::ErrorCode::invalidArgument);

    const std::array validEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::allGraphics, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 1},
        rachis::BindingLayoutEntry{4, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 1},
    };
    auto layout = device.createBindingLayout({
        .entries = validEntries,
        .label = "Portable binding layout",
    });
    REQUIRE(layout.has_value());

    CHECK(device.createBindingPool({.maxBindingSets = 0}).error().code == rachis::ErrorCode::invalidArgument);

    const std::array poolSizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::uniformBuffer, .descriptorCount = 1},
        rachis::BindingPoolSize{.type = rachis::BindingType::sampledTexture, .descriptorCount = 1},
        rachis::BindingPoolSize{.type = rachis::BindingType::sampler, .descriptorCount = 1},
    };
    auto pool = device.createBindingPool({
        .maxBindingSets = 1,
        .sizes = poolSizes,
        .label = "Portable exact-capacity binding pool",
    });
    REQUIRE(pool.has_value());

    auto uniformBuffer = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Portable uniform buffer",
    });
    auto texture = device.createTexture({
        .dimension = rachis::TextureDimension::texture2D,
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Portable sampled texture",
    });
    REQUIRE(uniformBuffer.has_value());
    REQUIRE(texture.has_value());
    auto view = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    auto sampler = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .mipFilter = rachis::MipmapFilterMode::linear,
    });
    REQUIRE(view.has_value());
    REQUIRE(sampler.has_value());

    const std::array incompleteWrites{
        rachis::BindingWrite{.binding = 0, .resource = rachis::BufferBinding{.buffer = *uniformBuffer, .size = 256}},
    };
    auto incompleteSet = device.createBindingSet({
        .layout = *layout,
        .writes = incompleteWrites,
        .pool = *pool,
    });
    REQUIRE_FALSE(incompleteSet.has_value());
    CHECK(incompleteSet.error().code == rachis::ErrorCode::invalidArgument);

    const std::array validWrites{
        rachis::BindingWrite{.binding = 0, .resource = rachis::BufferBinding{.buffer = *uniformBuffer, .size = 256}},
        rachis::BindingWrite{.binding = 2, .resource = *view},
        rachis::BindingWrite{.binding = 4, .resource = *sampler},
    };
    auto firstSet = device.createBindingSet({
        .layout = *layout,
        .writes = validWrites,
        .label = "Portable binding set",
        .pool = *pool,
    });
    REQUIRE(firstSet.has_value());

    auto exhaustedSet = device.createBindingSet({
        .layout = *layout,
        .writes = validWrites,
        .pool = *pool,
    });
    REQUIRE_FALSE(exhaustedSet.has_value());
    CHECK(exhaustedSet.error().code == rachis::ErrorCode::resourceExhausted);

    REQUIRE(device.destroyBindingSet(*firstSet).has_value());

    auto recycledSet = device.createBindingSet({
        .layout = *layout,
        .writes = validWrites,
        .label = "Portable recycled binding set",
        .pool = *pool,
    });
    REQUIRE(recycledSet.has_value());

    REQUIRE(device.destroyBindingSet(*recycledSet).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroySampler(*sampler).has_value());
    REQUIRE(device.destroyTextureView(*view).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*uniformBuffer).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable buffer binding limits are nonzero and binding sets reject ranges past them",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;
    const auto limits = rachis::test::requireCapabilities(device).limits;
    CAPTURE(limits.maxStorageBufferBindingSize,
            limits.maxUniformBufferSize,
            limits.minUniformBufferOffsetAlignment,
            limits.minStorageBufferOffsetAlignment);

    CHECK(limits.maxStorageBufferBindingSize >= (std::uint64_t{1} << 27));
    CHECK(limits.maxUniformBufferSize >= 16384);
    for (const auto alignment : {limits.minUniformBufferOffsetAlignment, limits.minStorageBufferOffsetAlignment})
    {
        CHECK(std::has_single_bit(alignment));
        CHECK(alignment <= 256);
    }
    switch (spec.backend)
    {
    case rachis::Backend::d3d12:
        CHECK(limits.maxStorageBufferBindingSize == (std::uint64_t{1} << 29));
        CHECK(limits.maxUniformBufferSize == 65536);
        CHECK(limits.minStorageBufferOffsetAlignment == 16);
        break;
    case rachis::Backend::metal:
        CHECK(limits.maxStorageBufferBindingSize <= limits.maxBufferSize);
        CHECK(limits.maxUniformBufferSize == 65536);
        CHECK(limits.minStorageBufferOffsetAlignment == 16);
        break;
    case rachis::Backend::vulkan:
        CHECK(limits.maxStorageBufferBindingSize <= std::numeric_limits<std::uint32_t>::max());
        CHECK(limits.maxUniformBufferSize <= std::numeric_limits<std::uint32_t>::max());
        break;
    }

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Portable buffer limit layout"});
    REQUIRE(layout.has_value());
    const std::array poolSizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::uniformBuffer, .descriptorCount = 1},
        rachis::BindingPoolSize{.type = rachis::BindingType::storageBuffer, .descriptorCount = 1},
    };
    auto pool =
        device.createBindingPool({.maxBindingSets = 1, .sizes = poolSizes, .label = "Portable buffer limit pool"});
    REQUIRE(pool.has_value());

    const auto tryBind = [&](const rachis::BufferBinding& uniform,
                             const rachis::BufferBinding& storage) -> std::optional<rachis::ErrorCode> {
        const std::array writes{
            rachis::BindingWrite{.binding = 0, .resource = uniform},
            rachis::BindingWrite{.binding = 1, .resource = storage},
        };
        auto set = device.createBindingSet({.layout = *layout, .writes = writes, .pool = *pool});
        if (!set)
        {
            return set.error().code;
        }
        REQUIRE(device.destroyBindingSet(*set).has_value());
        return std::nullopt;
    };

    constexpr std::uint64_t largestTestBuffer = std::uint64_t{1} << 30;
    const auto uniformAlignment = limits.minUniformBufferOffsetAlignment;
    const auto storageAlignment = (std::max<std::uint64_t>)(limits.minStorageBufferOffsetAlignment, 4);
    const bool uniformLimitTestable = limits.maxUniformBufferSize + uniformAlignment <= largestTestBuffer;
    const auto uniformBytes = uniformLimitTestable ? limits.maxUniformBufferSize + uniformAlignment : 4096;
    auto uniformBuffer = device.createBuffer({
        .size = uniformBytes,
        .usage = rachis::BufferUsage::constant,
        .label = "Portable buffer limit uniform buffer",
    });
    REQUIRE(uniformBuffer.has_value());
    auto storageBuffer = device.createBuffer({
        .size = 4096,
        .usage = rachis::BufferUsage::storage,
        .label = "Portable buffer limit storage buffer",
    });
    REQUIRE(storageBuffer.has_value());

    const rachis::BufferBinding smallUniform{.buffer = *uniformBuffer, .size = 256};
    const rachis::BufferBinding smallStorage{.buffer = *storageBuffer, .size = 256};
    CHECK(tryBind(smallUniform, smallStorage) == std::nullopt);
    CHECK(tryBind(smallUniform, {.buffer = *storageBuffer, .offset = storageAlignment, .size = 256}) == std::nullopt);
    if (limits.minStorageBufferOffsetAlignment > 4)
    {
        CHECK(tryBind(smallUniform, {.buffer = *storageBuffer, .offset = 4, .size = 256}) ==
              rachis::ErrorCode::invalidArgument);
    }
    if (uniformAlignment > 16)
    {
        CHECK(tryBind({.buffer = *uniformBuffer, .offset = 16, .size = 256}, smallStorage) ==
              rachis::ErrorCode::invalidArgument);
    }

    if (uniformLimitTestable)
    {
        CHECK(tryBind({.buffer = *uniformBuffer, .size = limits.maxUniformBufferSize}, smallStorage) == std::nullopt);
        CHECK(tryBind({.buffer = *uniformBuffer, .size = limits.maxUniformBufferSize + uniformAlignment},
                      smallStorage) == rachis::ErrorCode::invalidArgument);
    }
    else
    {
        WARN("maxUniformBufferSize is too large to allocate a buffer past it in this test");
    }

    if (limits.maxStorageBufferBindingSize + storageAlignment <= largestTestBuffer)
    {
        auto largeStorage = device.createBuffer({
            .size = limits.maxStorageBufferBindingSize + storageAlignment,
            .usage = rachis::BufferUsage::storage,
            .label = "Portable buffer limit large storage buffer",
        });
        if (largeStorage)
        {
            CHECK(tryBind(smallUniform, {.buffer = *largeStorage, .size = limits.maxStorageBufferBindingSize}) ==
                  std::nullopt);
            CHECK(tryBind(smallUniform,
                          {.buffer = *largeStorage, .size = limits.maxStorageBufferBindingSize + storageAlignment}) ==
                  rachis::ErrorCode::invalidArgument);
            CHECK(tryBind(smallUniform, {.buffer = *largeStorage}) == rachis::ErrorCode::invalidArgument);
            REQUIRE(device.destroyBuffer(*largeStorage).has_value());
        }
        else
        {
            WARN("Could not allocate a buffer past maxStorageBufferBindingSize: " << largeStorage.error().message);
        }
    }
    else
    {
        WARN("maxStorageBufferBindingSize is too large to allocate a buffer past it in this test");
    }

    REQUIRE(device.destroyBuffer(*storageBuffer).has_value());
    REQUIRE(device.destroyBuffer(*uniformBuffer).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    checkNoDiagnosticsErrors(device);
}

#ifdef RACHIS_TEST_PORTABLE_COMPUTE_SHADER_DIR
TEST_CASE("Portable binding-set updates reuse an idle set's pool capacity and need room while a command holds the set",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    constexpr std::uint32_t workgroupWidth = 64;
    constexpr std::uint64_t outputBytes = workgroupWidth * sizeof(std::uint32_t);
    const auto bytecode = loadPortableComputeShader(spec.backend);
    const std::array requirements{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Portable binding-pool update shader",
        .shaderInterface = rachis::test::computeShaderInterface({workgroupWidth, 1, 1}, requirements),
    });
    if (!shader)
    {
        INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Portable binding-pool update layout"});
    REQUIRE(layout.has_value());
    const std::array layouts{*layout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    const std::array poolSizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::storageBuffer, .descriptorCount = 1},
    };
    auto pool = device.createBindingPool({
        .maxBindingSets = 1,
        .sizes = poolSizes,
        .label = "Portable exact-capacity update pool",
    });
    REQUIRE(pool.has_value());

    auto outputA = device.createBuffer({
        .size = outputBytes,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Portable binding-pool update output A",
    });
    auto outputB = device.createBuffer({
        .size = outputBytes,
        .usage = rachis::BufferUsage::storage,
        .label = "Portable binding-pool update output B",
    });
    REQUIRE(outputA.has_value());
    REQUIRE(outputB.has_value());

    const std::array writesA{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *outputA,
                                              .size = outputBytes,
                                              .structureByteStride = sizeof(std::uint32_t)},
        },
    };
    const std::array writesB{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *outputB,
                                              .size = outputBytes,
                                              .structureByteStride = sizeof(std::uint32_t)},
        },
    };
    auto set = device.createBindingSet({
        .layout = *layout,
        .writes = writesA,
        .label = "Portable exact-capacity update set",
        .pool = *pool,
    });
    REQUIRE(set.has_value());

    REQUIRE(device.updateBindingSet(*set, writesB).has_value());
    REQUIRE(device.updateBindingSet(*set, writesA).has_value());

    for (const auto flags : {rachis::CommandPoolFlag::none, rachis::CommandPoolFlag::unretainedResources})
    {
        CAPTURE(static_cast<std::uint32_t>(flags));
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute, flags);
        REQUIRE(command.has_value());
        REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());

        const auto boundUpdate = device.updateBindingSet(*set, writesB);
        REQUIRE_FALSE(boundUpdate.has_value());
        CHECK(boundUpdate.error().code == rachis::ErrorCode::resourceExhausted);

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        REQUIRE(device.updateBindingSet(*set, writesB).has_value());
        REQUIRE(device.updateBindingSet(*set, writesA).has_value());
    }

    const std::array twoVersionSizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::storageBuffer, .descriptorCount = 2},
    };
    auto twoVersionPool = device.createBindingPool({
        .maxBindingSets = 1,
        .sizes = twoVersionSizes,
        .label = "Portable two-version update pool",
    });
    REQUIRE(twoVersionPool.has_value());
    auto readback = device.createBuffer({
        .size = outputBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Portable binding-pool update readback",
    });
    REQUIRE(readback.has_value());

    for (const auto flags : {rachis::CommandPoolFlag::none, rachis::CommandPoolFlag::unretainedResources})
    {
        CAPTURE(static_cast<std::uint32_t>(flags));
        auto twoVersionSet = device.createBindingSet({
            .layout = *layout,
            .writes = writesA,
            .label = "Portable two-version update set",
            .pool = *twoVersionPool,
        });
        REQUIRE(twoVersionSet.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute, flags);
        REQUIRE(command.has_value());
        REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*command, 0, *twoVersionSet).has_value());

        REQUIRE(device.updateBindingSet(*twoVersionSet, writesB).has_value());
        const auto secondUpdate = device.updateBindingSet(*twoVersionSet, writesA);
        REQUIRE_FALSE(secondUpdate.has_value());
        CHECK(secondUpdate.error().code == rachis::ErrorCode::resourceExhausted);

        REQUIRE(device.dispatch(*command, 1).has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *outputA,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device.copyBuffer(*command, {.source = *outputA, .destination = *readback, .size = outputBytes})
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        std::array<std::uint32_t, workgroupWidth> written{};
        REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span{written})).has_value());
        for (std::uint32_t index = 0; index < workgroupWidth; ++index)
        {
            CAPTURE(index);
            CHECK(written[index] == index * 3U + 7U);
        }

        REQUIRE(device.updateBindingSet(*twoVersionSet, writesA).has_value());
        REQUIRE(device.updateBindingSet(*twoVersionSet, writesB).has_value());
        REQUIRE(device.destroyBindingSet(*twoVersionSet).has_value());
    }

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBindingPool(*twoVersionPool).has_value());
    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBuffer(*outputA).has_value());
    REQUIRE(device.destroyBuffer(*outputB).has_value());
    checkNoDiagnosticsErrors(device);
}
#endif

namespace
{
    void checkVariableCountArrayReachesLimit(const PortableBackendSpec& spec, rachis::BindingType type)
    {
        CAPTURE(spec.name);

        auto maybeDevice = createPortableDeviceOrSkip(spec);
        if (!maybeDevice)
        {
            return;
        }
        auto& device = *maybeDevice;
        const auto capabilities = device.capabilities();
        REQUIRE(capabilities.has_value());
        REQUIRE(capabilities->partiallyBoundBindingArrays);
        REQUIRE(capabilities->variableDescriptorCountBindingArrays);
        const auto& limits = capabilities->limits;
        REQUIRE(limits.maxVariableBufferDescriptors > 0);
        REQUIRE(limits.maxVariableResourceDescriptors > 0);

        auto limit = limits.maxVariableResourceDescriptors;
        if (type == rachis::BindingType::uniformBuffer || type == rachis::BindingType::readOnlyStorageBuffer ||
            type == rachis::BindingType::storageBuffer)
        {
            limit = limits.maxVariableBufferDescriptors;
        }
        else if (type == rachis::BindingType::sampler)
        {
            limit = limits.maxVariableSamplerDescriptors;
            if (limit == 0)
            {
                return;
            }
        }

        INFO("type=" << static_cast<int>(type) << " limit=" << limit);
        const std::array entries{
            rachis::BindingLayoutEntry{
                .binding = 0,
                .type = type,
                .visibility = rachis::ShaderStage::compute,
                .count = rachis::unboundedBindingCount,
                .flags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::variableDescriptorCount,
            },
        };
        auto layout = device.createBindingLayout({.entries = entries, .label = "Portable variable-count layout"});
        REQUIRE(layout.has_value());
        const std::array sizes{
            rachis::BindingPoolSize{.type = type, .descriptorCount = limit},
        };
        auto pool =
            device.createBindingPool({.maxBindingSets = 1, .sizes = sizes, .label = "Portable variable-count pool"});
        REQUIRE(pool.has_value());
        auto atLimit = device.createBindingSet({
            .layout = *layout,
            .variableDescriptorCount = limit,
            .label = "Portable variable-count set at the limit",
            .pool = *pool,
        });
        CHECK(atLimit.has_value());
        if (atLimit)
        {
            CHECK(device.destroyBindingSet(*atLimit).has_value());
        }
        REQUIRE(device.destroyBindingPool(*pool).has_value());
        REQUIRE(device.destroyBindingLayout(*layout).has_value());
        checkNoDiagnosticsErrors(device);
    }
} // namespace

TEST_CASE("Portable variable-count uniform buffer arrays reach the published buffer descriptor limit",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    checkVariableCountArrayReachesLimit(spec, rachis::BindingType::uniformBuffer);
}

TEST_CASE("Portable variable-count read-only storage buffer arrays reach the published buffer descriptor limit",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    checkVariableCountArrayReachesLimit(spec, rachis::BindingType::readOnlyStorageBuffer);
}

TEST_CASE("Portable variable-count storage buffer arrays reach the published buffer descriptor limit",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    checkVariableCountArrayReachesLimit(spec, rachis::BindingType::storageBuffer);
}

TEST_CASE("Portable variable-count sampled texture arrays reach the published resource descriptor limit",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    checkVariableCountArrayReachesLimit(spec, rachis::BindingType::sampledTexture);
}

TEST_CASE("Portable variable-count storage texture arrays reach the published resource descriptor limit",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    checkVariableCountArrayReachesLimit(spec, rachis::BindingType::storageTexture);
}

TEST_CASE("Portable variable-count sampler arrays reach the published sampler descriptor limit",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    checkVariableCountArrayReachesLimit(spec, rachis::BindingType::sampler);
}

TEST_CASE("Portable compute reads StructuredBuffer and ByteAddressBuffer through one read-only interface and layout",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint32_t regionCount = 2;
    constexpr std::uint64_t regionSize = valueCount * sizeof(std::uint32_t);
    const auto bytecode = rachis_shaders::PortableReadOnlyStorageShader(spec.backend);
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::readOnlyStorageBuffer, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
        rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::readOnlyStorageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "readOnlyMain",
        .label = "Portable read-only storage",
        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}, shaderBindings),
    });
    if (!shader)
    {
        UNSCOPED_INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Portable read-only layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = bindingLayouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Portable read-only pipeline",
    });
    if (!pipeline)
    {
        UNSCOPED_INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());

    std::vector<std::uint32_t> inputs(static_cast<std::size_t>(valueCount) * regionCount);
    for (std::uint32_t region = 0; region < regionCount; ++region)
    {
        for (std::uint32_t index = 0; index < valueCount; ++index)
        {
            inputs[(region * valueCount) + index] = (region * 1000U) + (index * 3U) + 1U;
        }
    }
    auto inputBuffer = device.createBuffer({
        .size = regionSize * regionCount,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Portable read-only inputs",
    });
    auto outputBuffer = device.createBuffer({
        .size = regionSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Portable read-only output",
    });
    auto upload = device.createBuffer({
        .size = regionSize * regionCount,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Portable read-only upload",
    });
    auto readback = device.createBuffer({
        .size = regionSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Portable read-only readback",
    });
    REQUIRE(inputBuffer.has_value());
    REQUIRE(outputBuffer.has_value());
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(device.writeBuffer(*upload, 0, std::as_bytes(std::span(inputs))).has_value());

    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *inputBuffer,
                    .size = regionSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *outputBuffer,
                    .size = regionSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
        rachis::BindingWrite{
            .binding = 2,
            .resource =
                rachis::BufferBinding{
                    .buffer = *inputBuffer,
                    .offset = regionSize,
                    .size = regionSize,
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     rachis::BindingSetDesc{
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Portable read-only set",
                                                     });
    if (!bindingSet)
    {
        UNSCOPED_INFO(bindingSet.error().message);
    }
    REQUIRE(bindingSet.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *upload,
                                .destination = *inputBuffer,
                                .size = regionSize * regionCount,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *inputBuffer,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::shaderResource,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *outputBuffer,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *outputBuffer,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *outputBuffer,
                                .destination = *readback,
                                .size = regionSize,
                            })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    std::vector<std::uint32_t> output(valueCount, 0U);
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(output))).has_value());
    std::uint32_t firstMismatch = valueCount;
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        if (output[index] != inputs[index] + (inputs[valueCount + index] * 5U))
        {
            firstMismatch = index;
            break;
        }
    }
    CAPTURE(firstMismatch);
    CHECK(firstMismatch == valueCount);

    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*outputBuffer).has_value());
    REQUIRE(device.destroyBuffer(*inputBuffer).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable compute reads float4x4 and float3x4 with HLSL's default packing from constant buffers, push "
          "constants, and structured buffers",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    const auto bytecode = rachis_shaders::PortableMatrixLayoutShader(spec.backend);
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::uniformBuffer, 1},
        rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::readOnlyStorageBuffer, 1},
        rachis::ShaderBindingRequirement{0, 3, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array shaderPushConstants{
        rachis::ShaderPushConstantRequirement{0, matrixLayoutPushBytes},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "matrixLayoutMain",
        .label = "Portable matrix layout",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, shaderBindings, shaderPushConstants),
    });
    if (!shader)
    {
        UNSCOPED_INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{3, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Portable matrix layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, matrixLayoutPushBytes},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Portable matrix layout pipeline",
    });
    if (!pipeline)
    {
        UNSCOPED_INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());

    const std::uint32_t structuredColumnStride = spec.backend == rachis::Backend::vulkan ? 16U : 12U;
    const std::uint32_t structuredStride = 64U + (4U * structuredColumnStride);
    const std::uint64_t structuredBytes = std::uint64_t{structuredStride} * matrixLayoutStructuredElements;
    std::vector<std::uint32_t> expected(matrixLayoutOutputCount, 0U);
    std::vector<std::byte> constantData(static_cast<std::size_t>(matrixLayoutConstantBytes), std::byte{0});
    std::vector<std::byte> pushData(matrixLayoutPushBytes, std::byte{0});
    std::vector<std::byte> structuredData(static_cast<std::size_t>(structuredBytes), std::byte{0});
    placeColumnMajorMatrix(
        constantData, {.matrix = 0, .rows = 4, .offset = 0, .columnStride = 16, .firstOutput = 0}, expected);
    placeColumnMajorMatrix(
        constantData, {.matrix = 1, .rows = 3, .offset = 64, .columnStride = 16, .firstOutput = 16}, expected);
    for (std::uint32_t component = 0; component < 4; ++component)
    {
        const auto value = static_cast<float>(900U + component);
        std::memcpy(constantData.data() + 128 + (component * sizeof(float)), &value, sizeof(value));
        expected[28 + component] = std::bit_cast<std::uint32_t>(value);
    }
    placeColumnMajorMatrix(
        pushData, {.matrix = 2, .rows = 3, .offset = 0, .columnStride = 16, .firstOutput = 32}, expected);
    placeColumnMajorMatrix(
        pushData, {.matrix = 3, .rows = 4, .offset = 64, .columnStride = 16, .firstOutput = 44}, expected);
    for (std::uint32_t element = 0; element < matrixLayoutStructuredElements; ++element)
    {
        const auto elementOffset = static_cast<std::size_t>(element) * structuredStride;
        placeColumnMajorMatrix(structuredData,
                               {
                                   .matrix = 4 + (element * 2),
                                   .rows = 4,
                                   .offset = elementOffset,
                                   .columnStride = 16,
                                   .firstOutput = 60 + (element * 28),
                               },
                               expected);
        placeColumnMajorMatrix(structuredData,
                               {
                                   .matrix = 5 + (element * 2),
                                   .rows = 3,
                                   .offset = elementOffset + 64,
                                   .columnStride = structuredColumnStride,
                                   .firstOutput = 76 + (element * 28),
                               },
                               expected);
    }

    constexpr std::uint64_t outputBytes = matrixLayoutOutputCount * sizeof(std::uint32_t);
    auto constants = device.createBuffer({
        .size = matrixLayoutConstantBytes,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Portable matrix layout constants",
    });
    auto structured = device.createBuffer({
        .size = structuredBytes,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Portable matrix layout structured",
    });
    auto upload = device.createBuffer({
        .size = structuredBytes,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Portable matrix layout upload",
    });
    auto output = device.createBuffer({
        .size = outputBytes,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Portable matrix layout output",
    });
    auto readback = device.createBuffer({
        .size = outputBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Portable matrix layout readback",
    });
    REQUIRE(constants.has_value());
    REQUIRE(structured.has_value());
    REQUIRE(upload.has_value());
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(device.writeBuffer(*constants, 0, constantData).has_value());
    REQUIRE(device.writeBuffer(*upload, 0, structuredData).has_value());

    const std::array writes{
        rachis::BindingWrite{
            .binding = 1,
            .resource = rachis::BufferBinding{.buffer = *constants, .size = matrixLayoutConstantBytes},
        },
        rachis::BindingWrite{
            .binding = 2,
            .resource =
                rachis::BufferBinding{
                    .buffer = *structured,
                    .size = structuredBytes,
                    .structureByteStride = structuredStride,
                },
        },
        rachis::BindingWrite{
            .binding = 3,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = outputBytes,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     rachis::BindingSetDesc{
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Portable matrix layout set",
                                                     });
    if (!bindingSet)
    {
        UNSCOPED_INFO(bindingSet.error().message);
    }
    REQUIRE(bindingSet.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *upload,
                                .destination = *structured,
                                .size = structuredBytes,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *structured,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::shaderResource,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.setComputePushConstants(*command, 0, pushData).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *output,
                                .destination = *readback,
                                .size = outputBytes,
                            })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    std::vector<std::uint32_t> values(matrixLayoutOutputCount, 0U);
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(values))).has_value());
    std::uint32_t mismatches = 0;
    std::uint32_t firstMismatch = matrixLayoutOutputCount;
    for (std::uint32_t index = 0; index < matrixLayoutOutputCount; ++index)
    {
        if (values[index] != expected[index])
        {
            firstMismatch = (std::min)(firstMismatch, index);
            ++mismatches;
        }
    }
    CAPTURE(firstMismatch);
    if (firstMismatch < matrixLayoutOutputCount)
    {
        UNSCOPED_INFO("output " << std::bit_cast<float>(values[firstMismatch]) << " expected "
                                << std::bit_cast<float>(expected[firstMismatch]));
    }
    CHECK(mismatches == 0);

    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*structured).has_value());
    REQUIRE(device.destroyBuffer(*constants).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable graphics pipelines reject invalid depth-stencil and blend state with the same errors",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    const auto vertexBytecode = loadBytes(spec.graphicsVertexShaderPath);
    const auto pixelBytecode = loadBytes(spec.graphicsPixelShaderPath);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .label = "Portable state validation vertex",
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .label = "Portable state validation pixel",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto layout = device.createPipelineLayout({.pushConstantRanges = pushConstantRanges});
    REQUIRE(layout.has_value());

    const auto createPipeline = [&](const rachis::DepthStencilState& depthStencil,
                                    std::span<const rachis::ColorTargetState> colorTargets) -> rachis::Status {
        auto pipeline = device.createGraphicsPipeline({
            .layout = *layout,
            .vertexShader = *vertexShader,
            .pixelShader = *pixelShader,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
            .depthStencil = depthStencil,
        });
        if (!pipeline)
        {
            return std::unexpected(std::move(pipeline.error()));
        }
        REQUIRE(device.destroyPipeline(*pipeline).has_value());
        return {};
    };
    const auto requireCreated = [&](const rachis::DepthStencilState& depthStencil,
                                    std::span<const rachis::ColorTargetState> colorTargets) {
        auto created = createPipeline(depthStencil, colorTargets);
        if (!created)
        {
            UNSCOPED_INFO(created.error().message);
        }
        REQUIRE(created.has_value());
    };
    const auto errorCode =
        [&](const rachis::DepthStencilState& depthStencil,
            std::span<const rachis::ColorTargetState> colorTargets) -> std::optional<rachis::ErrorCode> {
        auto created = createPipeline(depthStencil, colorTargets);
        if (created)
        {
            return std::nullopt;
        }
        return created.error().code;
    };

    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    requireCreated(
        {
            .format = rachis::Format::d32Float,
            .depthTestEnabled = true,
            .depthWriteEnabled = true,
            .depthCompare = rachis::CompareOperation::less,
        },
        colorTargets);
    requireCreated({.format = rachis::Format::d32Float, .depthTestEnabled = true}, colorTargets);

    CHECK(errorCode({.format = rachis::Format::rgba8Unorm, .depthTestEnabled = true}, colorTargets) ==
          rachis::ErrorCode::invalidArgument);
    CHECK(errorCode({.format = rachis::Format::d32Float, .depthWriteEnabled = true}, colorTargets) ==
          rachis::ErrorCode::invalidArgument);
    CHECK(errorCode(
              {
                  .format = rachis::Format::d32Float,
                  .depthTestEnabled = true,
                  .depthCompare = static_cast<rachis::CompareOperation>(200),
              },
              colorTargets) == rachis::ErrorCode::invalidArgument);

    const std::array invalidFaces{
        rachis::StencilFaceState{.compare = static_cast<rachis::CompareOperation>(200)},
        rachis::StencilFaceState{.failOperation = static_cast<rachis::StencilOperation>(200)},
        rachis::StencilFaceState{.depthFailOperation = static_cast<rachis::StencilOperation>(200)},
        rachis::StencilFaceState{.passOperation = static_cast<rachis::StencilOperation>(200)},
    };
    for (const auto& face : invalidFaces)
    {
        CHECK(errorCode({.format = rachis::Format::d32Float, .depthTestEnabled = true, .frontFace = face},
                        colorTargets) == rachis::ErrorCode::invalidArgument);
        CHECK(errorCode({.format = rachis::Format::d32Float, .depthTestEnabled = true, .backFace = face},
                        colorTargets) == rachis::ErrorCode::invalidArgument);
    }

    CHECK(errorCode({.format = rachis::Format::d32Float, .depthTestEnabled = true, .stencilTestEnabled = true},
                    colorTargets) == rachis::ErrorCode::unsupported);

    const rachis::StencilFaceState stencilFace{
        .compare = rachis::CompareOperation::always,
        .passOperation = rachis::StencilOperation::replace,
    };
    const rachis::DepthStencilState packedStencil{
        .format = rachis::Format::d24UnormS8Uint,
        .depthTestEnabled = true,
        .depthWriteEnabled = true,
        .stencilTestEnabled = true,
        .frontFace = stencilFace,
        .backFace = stencilFace,
    };
    if (rachis::hasAny(device.queryFormatProperties(rachis::Format::d24UnormS8Uint).features,
                       rachis::FormatFeature::depthStencilAttachment))
    {
        requireCreated(packedStencil, colorTargets);
    }
    else
    {
        CHECK(errorCode(packedStencil, colorTargets) == rachis::ErrorCode::unsupported);
    }

    const std::array constantBlendTargets{
        rachis::ColorTargetState{
            .format = rachis::Format::rgba8Unorm,
            .blend =
                {
                    .enabled = true,
                    .color =
                        {
                            .sourceFactor = rachis::BlendFactor::constant,
                            .destinationFactor = rachis::BlendFactor::oneMinusConstant,
                        },
                },
        },
    };
    requireCreated({}, constantBlendTargets);

    constexpr auto invalidFactor = static_cast<rachis::BlendFactor>(255);
    constexpr auto invalidOperation = static_cast<rachis::BlendOperation>(255);
    const std::array invalidBlends{
        rachis::BlendState{.enabled = true, .color = {.sourceFactor = invalidFactor}},
        rachis::BlendState{.enabled = true, .color = {.destinationFactor = invalidFactor}},
        rachis::BlendState{.enabled = true, .color = {.operation = invalidOperation}},
        rachis::BlendState{.enabled = true, .alpha = {.sourceFactor = invalidFactor}},
        rachis::BlendState{.enabled = true, .alpha = {.destinationFactor = invalidFactor}},
        rachis::BlendState{.enabled = true, .alpha = {.operation = invalidOperation}},
        rachis::BlendState{.enabled = false, .color = {.operation = invalidOperation}},
    };
    for (const auto& blend : invalidBlends)
    {
        const std::array invalidBlendTargets{
            rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm, .blend = blend},
        };
        CHECK(errorCode({}, invalidBlendTargets) == rachis::ErrorCode::invalidArgument);
    }

    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable compute shader modules require a workgroup size and dispatch whole declared workgroups",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    const auto computeBytecode = loadBytes(spec.computeShaderPath);
    const auto vertexBytecode = loadBytes(spec.graphicsVertexShaderPath);
    REQUIRE_FALSE(computeBytecode.empty());
    REQUIRE_FALSE(vertexBytecode.empty());

    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, spec.computeStorageBinding, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array shaderPushConstants{
        rachis::ShaderPushConstantRequirement{0, spec.computePushConstantBytes},
    };
    const auto createComputeShader = [&](std::optional<rachis::ShaderWorkgroupSize> workgroupSize) {
        return device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = computeBytecode,
            .entryPoint = spec.computeEntryPoint,
            .label = "Portable workgroup validation compute",
            .shaderInterface =
                {
                    .bindings = shaderBindings,
                    .pushConstants = shaderPushConstants,
                    .workgroupSize = workgroupSize,
                },
        });
    };
    const auto computeErrorCode =
        [&](std::optional<rachis::ShaderWorkgroupSize> workgroupSize) -> std::optional<rachis::ErrorCode> {
        auto created = createComputeShader(workgroupSize);
        if (created)
        {
            REQUIRE(device.destroyShaderModule(*created).has_value());
            return std::nullopt;
        }
        return created.error().code;
    };

    CHECK(computeErrorCode(std::nullopt) == rachis::ErrorCode::invalidArgument);
    CHECK(computeErrorCode(rachis::ShaderWorkgroupSize{0, 1, 1}) == rachis::ErrorCode::invalidArgument);
    CHECK(computeErrorCode(rachis::ShaderWorkgroupSize{64, 0, 1}) == rachis::ErrorCode::invalidArgument);
    CHECK(computeErrorCode(rachis::ShaderWorkgroupSize{64, 1, 0}) == rachis::ErrorCode::invalidArgument);

    auto vertexWithWorkgroup = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .label = "Portable workgroup validation vertex",
        .shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1}},
    });
    REQUIRE_FALSE(vertexWithWorkgroup.has_value());
    CHECK(vertexWithWorkgroup.error().code == rachis::ErrorCode::invalidArgument);

    auto shader = createComputeShader(rachis::ShaderWorkgroupSize{64, 1, 1});
    if (!shader)
    {
        UNSCOPED_INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());
    auto snapshot = device.shaderModuleInterface(*shader);
    REQUIRE(snapshot.has_value());
    REQUIRE(snapshot->workgroupSize.has_value());
    CHECK(snapshot->workgroupSize->x == 64);
    CHECK(snapshot->workgroupSize->y == 1);
    CHECK(snapshot->workgroupSize->z == 1);

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{
            spec.computeStorageBinding, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, spec.computePushConstantBytes},
    };
    auto layout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(layout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *layout,
        .computeShader = *shader,
        .label = "Portable workgroup validation pipeline",
    });
    if (!pipeline)
    {
        UNSCOPED_INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());

    constexpr std::uint32_t elementCount = 256;
    constexpr std::uint32_t workgroupCount = elementCount / 64;
    constexpr std::uint64_t outputBytes = elementCount * sizeof(std::uint32_t);
    auto storage = device.createBuffer({
        .size = outputBytes,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Portable workgroup output",
    });
    auto readback = device.createBuffer({
        .size = outputBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Portable workgroup readback",
    });
    REQUIRE(storage.has_value());
    REQUIRE(readback.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = spec.computeStorageBinding,
            .resource =
                rachis::BufferBinding{
                    .buffer = *storage,
                    .size = outputBytes,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(bindingSet.has_value());

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
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    const std::array constants{1U, 1U, 0U, 0U};
    REQUIRE(device
                .setComputePushConstants(
                    *command, 0, std::as_bytes(std::span(constants)).first(spec.computePushConstantBytes))
                .has_value());
    REQUIRE(device.dispatch(*command, workgroupCount).has_value());
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
        device.copyBuffer(*command, {.source = *storage, .destination = *readback, .size = outputBytes}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    std::array<std::uint32_t, elementCount> values{};
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(values))).has_value());
    std::uint32_t correct = 0;
    for (std::uint32_t index = 0; index < elementCount; ++index)
    {
        if (values[index] == index + 1)
        {
            ++correct;
        }
    }
    CHECK(correct == elementCount);

    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    checkNoDiagnosticsErrors(device);
}

namespace
{

    constexpr std::uint32_t drawIdTargetWidth = 64;
    constexpr std::uint32_t drawIdVertexCount = 6;
    constexpr std::uint32_t drawIdInstanceCount = 2;
    constexpr std::uint32_t drawIdVertexBase = 6;
    constexpr std::uint32_t drawIdFirstInstance = 3;
    constexpr std::uint64_t drawIdIndexedArgumentOffset = sizeof(rachis::DrawIndirectArguments);

    enum class DrawIdCall : std::uint8_t
    {
        draw,
        drawIndexed,
        drawIndirect,
        drawIndexedIndirect,
    };

    struct DrawIdCase final
    {
        std::string_view name = {};
        DrawIdCall call = DrawIdCall::draw;
        std::uint32_t vertexBase = 0;
        std::uint32_t firstInstance = 0;
    };

    struct DrawIdShaders final
    {
        std::string_view language;
        std::span<const std::byte> (*vertex)(rachis::Backend) noexcept = nullptr;
        std::span<const std::byte> (*pixel)(rachis::Backend) noexcept = nullptr;
    };

    void appendDrawIdHit(std::string& text, std::uint32_t column, std::uint32_t triangle, std::uint32_t instance)
    {
        text += std::to_string(column) + ":(" + std::to_string(triangle) + "," + std::to_string(instance) + ") ";
    }

    [[nodiscard]] std::string describeDrawIds(std::span<const std::byte> row)
    {
        std::string text;
        for (std::uint32_t column = 0; column < drawIdTargetWidth; ++column)
        {
            const auto* texel = row.data() + static_cast<std::size_t>(column) * 4U;
            if (std::to_integer<std::uint32_t>(texel[3]) == 0)
            {
                continue;
            }
            appendDrawIdHit(
                text, column, std::to_integer<std::uint32_t>(texel[0]), std::to_integer<std::uint32_t>(texel[1]));
        }
        return text;
    }

    [[nodiscard]] std::string expectedDrawIds()
    {
        std::string text;
        for (std::uint32_t instance = 0; instance < drawIdInstanceCount; ++instance)
        {
            for (std::uint32_t triangle = 0; triangle < drawIdVertexCount / 3U; ++triangle)
            {
                appendDrawIdHit(text, triangle + 8U * instance, triangle, instance);
            }
        }
        return text;
    }

    [[nodiscard]] rachis::Status recordDrawIdCall(rachis::Device& device,
                                                  rachis::CommandBuffer& command,
                                                  const DrawIdCase& drawCase,
                                                  rachis::BufferHandle indexBuffer,
                                                  rachis::BufferHandle arguments)
    {
        if (drawCase.call == DrawIdCall::drawIndexed || drawCase.call == DrawIdCall::drawIndexedIndirect)
        {
            auto bound = device.bindIndexBuffer(command, indexBuffer, rachis::IndexFormat::uint16);
            if (!bound)
            {
                return bound;
            }
        }
        switch (drawCase.call)
        {
        case DrawIdCall::draw:
            return device.draw(
                command, drawIdVertexCount, drawIdInstanceCount, drawCase.vertexBase, drawCase.firstInstance);
        case DrawIdCall::drawIndexed:
            return device.drawIndexed(command,
                                      drawIdVertexCount,
                                      drawIdInstanceCount,
                                      0,
                                      static_cast<std::int32_t>(drawCase.vertexBase),
                                      drawCase.firstInstance);
        case DrawIdCall::drawIndirect:
            return device.drawIndirect(command, {.argumentBuffer = arguments});
        case DrawIdCall::drawIndexedIndirect:
            return device.drawIndexedIndirect(
                command, {.argumentBuffer = arguments, .argumentOffset = drawIdIndexedArgumentOffset});
        }
        return std::unexpected(rachis::Error{rachis::ErrorCode::invalidArgument, 0, "Unknown draw-ID call."});
    }

} // namespace

TEST_CASE("Portable draws keep the draw bases out of SV_VertexID and SV_InstanceID", RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    const auto shaders = GENERATE(values({
        DrawIdShaders{"HLSL", &rachis_shaders::PortableDrawIdVertex, &rachis_shaders::PortableDrawIdPixel},
        DrawIdShaders{"Slang", &rachis_shaders::PortableDrawIdVertexSlang, &rachis_shaders::PortableDrawIdPixelSlang},
    }));
    CAPTURE(spec.name, shaders.language);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;
    const auto capabilities = rachis::test::requireCapabilities(device);
    const std::uint32_t indirectFirstInstance = capabilities.indirectDrawBaseInstance ? drawIdFirstInstance : 0U;

    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = shaders.vertex(spec.backend),
        .entryPoint = "drawIdVertex",
        .label = "Portable draw-ID vertex shader",
    });
    REQUIRE(vertexShader.has_value());
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = shaders.pixel(spec.backend),
        .entryPoint = "drawIdPixel",
        .label = "Portable draw-ID pixel shader",
    });
    REQUIRE(pixelShader.has_value());
    auto pipelineLayout = device.createPipelineLayout({});
    REQUIRE(pipelineLayout.has_value());
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Portable draw-ID pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto target = device.createTexture({
        .width = drawIdTargetWidth,
        .height = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .label = "Portable draw-ID target",
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView.has_value());
    const rachis::Extent3D extent{.width = drawIdTargetWidth, .height = 1, .depth = 1};
    const auto transferLayout = device.queryTextureTransferLayout(*target, {.extent = extent});
    REQUIRE(transferLayout.has_value());
    auto readback = device.createBuffer({
        .size = transferLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Portable draw-ID readback",
    });
    REQUIRE(readback.has_value());

    const std::array<std::uint16_t, drawIdVertexCount> indices{0, 1, 2, 3, 4, 5};
    auto indexBuffer = device.createBuffer({
        .size = sizeof(indices),
        .usage = rachis::BufferUsage::index,
        .memoryType = rachis::MemoryType::upload,
        .label = "Portable draw-ID indices",
    });
    REQUIRE(indexBuffer.has_value());
    REQUIRE(device.writeBuffer(*indexBuffer, 0, std::as_bytes(std::span{indices})).has_value());

    const rachis::DrawIndirectArguments drawArguments{
        .vertexCount = drawIdVertexCount,
        .instanceCount = drawIdInstanceCount,
        .firstVertex = drawIdVertexBase,
        .firstInstance = indirectFirstInstance,
    };
    const rachis::DrawIndexedIndirectArguments drawIndexedArguments{
        .indexCount = drawIdVertexCount,
        .instanceCount = drawIdInstanceCount,
        .firstIndex = 0,
        .baseVertex = static_cast<std::int32_t>(drawIdVertexBase),
        .firstInstance = indirectFirstInstance,
    };
    std::array<std::byte, drawIdIndexedArgumentOffset + sizeof(rachis::DrawIndexedIndirectArguments)> argumentBytes{};
    std::memcpy(argumentBytes.data(), &drawArguments, sizeof(drawArguments));
    std::memcpy(
        argumentBytes.data() + drawIdIndexedArgumentOffset, &drawIndexedArguments, sizeof(drawIndexedArguments));
    auto arguments = device.createBuffer({
        .size = argumentBytes.size(),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
        .label = "Portable draw-ID indirect arguments",
    });
    REQUIRE(arguments.has_value());
    REQUIRE(device.writeBuffer(*arguments, 0, argumentBytes).has_value());

    const std::array cases{
        DrawIdCase{.name = "draw(6, 2, 0, 0)", .call = DrawIdCall::draw},
        DrawIdCase{
            .name = "draw(6, 2, 6, 3)",
            .call = DrawIdCall::draw,
            .vertexBase = drawIdVertexBase,
            .firstInstance = drawIdFirstInstance,
        },
        DrawIdCase{.name = "drawIndexed(6, 2, 0, 0, 0)", .call = DrawIdCall::drawIndexed},
        DrawIdCase{
            .name = "drawIndexed(6, 2, 0, 6, 3)",
            .call = DrawIdCall::drawIndexed,
            .vertexBase = drawIdVertexBase,
            .firstInstance = drawIdFirstInstance,
        },
        DrawIdCase{
            .name = "drawIndirect(6, 2, 6, firstInstance)",
            .call = DrawIdCall::drawIndirect,
            .vertexBase = drawIdVertexBase,
            .firstInstance = indirectFirstInstance,
        },
        DrawIdCase{
            .name = "drawIndexedIndirect(6, 2, 0, 6, firstInstance)",
            .call = DrawIdCall::drawIndexedIndirect,
            .vertexBase = drawIdVertexBase,
            .firstInstance = indirectFirstInstance,
        },
    };

    bool targetWritten = false;
    for (const auto& drawCase : cases)
    {
        CAPTURE(drawCase.name, drawCase.firstInstance);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(
                        *command,
                        {
                            .texture = *target,
                            .stagesBefore = targetWritten ? rachis::PipelineStage::copy : rachis::PipelineStage::none,
                            .stagesAfter = rachis::PipelineStage::renderTarget,
                            .accessBefore = targetWritten ? rachis::Access::copySource : rachis::Access::none,
                            .accessAfter = rachis::Access::renderTarget,
                            .layoutBefore =
                                targetWritten ? rachis::TextureLayout::copySource : rachis::TextureLayout::undefined,
                            .layoutAfter = rachis::TextureLayout::renderTarget,
                        })
                    .has_value());
        const std::array colorAttachments{
            rachis::ColorAttachment{
                .view = *targetView,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
                .clearValue = {},
            },
        };
        REQUIRE(device.beginRendering(*command, {.colorAttachments = colorAttachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
        REQUIRE(
            device.setViewport(*command, {.width = static_cast<float>(drawIdTargetWidth), .height = 1.0F}).has_value());
        REQUIRE(device.setScissor(*command, {.width = drawIdTargetWidth, .height = 1}).has_value());
        const auto drawn = recordDrawIdCall(device, *command, drawCase, *indexBuffer, *arguments);
        if (!drawn)
        {
            UNSCOPED_INFO(drawn.error().message);
        }
        REQUIRE(drawn.has_value());
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *target,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         {
                                             .buffer = *readback,
                                             .bytesPerRow = transferLayout->bytesPerRow,
                                             .rowsPerImage = transferLayout->rowsPerImage,
                                             .texture = *target,
                                             .extent = extent,
                                         })
                    .has_value());
        targetWritten = true;
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        std::vector<std::byte> row(static_cast<std::size_t>(transferLayout->requiredBufferSize));
        REQUIRE(device.readBuffer(*readback, 0, row).has_value());
        CHECK(describeDrawIds(row) == expectedDrawIds());
    }

    REQUIRE(device.destroyBuffer(*arguments).has_value());
    REQUIRE(device.destroyBuffer(*indexBuffer).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable samplers without mip filtering minify with minFilter and read only level 0",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    using rachis::FilterMode;
    const std::array cases{
        SamplerFilterCase{
            .name = "minify 4x, min linear, mag nearest",
            .sampler = {.minFilter = FilterMode::linear, .magFilter = FilterMode::nearest},
            .texture = SamplerFilterTexture::checkerboard,
            .expected = SamplerFilterExpectation::filtered,
        },
        SamplerFilterCase{
            .name = "minify 4x, min nearest, mag linear",
            .sampler = {.minFilter = FilterMode::nearest, .magFilter = FilterMode::linear},
            .texture = SamplerFilterTexture::checkerboard,
            .expected = SamplerFilterExpectation::unfiltered,
        },
        SamplerFilterCase{
            .name = "minify 4x, default sampler",
            .sampler = {},
            .texture = SamplerFilterTexture::checkerboard,
            .expected = SamplerFilterExpectation::unfiltered,
        },
        SamplerFilterCase{
            .name = "minify 4x, min linear, mag nearest, maxLod 0",
            .sampler = {.minFilter = FilterMode::linear, .magFilter = FilterMode::nearest, .maxLod = 0.0F},
            .texture = SamplerFilterTexture::checkerboard,
            .expected = SamplerFilterExpectation::unfiltered,
        },
        SamplerFilterCase{
            .name = "minify 4x, min linear, mag nearest, two-level texture",
            .sampler = {.minFilter = FilterMode::linear, .magFilter = FilterMode::nearest},
            .texture = SamplerFilterTexture::checkerboardWithUpperLevel,
            .expected = SamplerFilterExpectation::filtered,
        },
        SamplerFilterCase{
            .name = "minify 4x, min linear, mag linear, black level 0",
            .sampler = {.minFilter = FilterMode::linear, .magFilter = FilterMode::linear},
            .texture = SamplerFilterTexture::blackWithUpperLevel,
            .expected = SamplerFilterExpectation::black,
        },
        SamplerFilterCase{
            .name = "minify 4x, anisotropy 4, black level 0",
            .sampler = {.minFilter = FilterMode::linear, .magFilter = FilterMode::linear, .maxAnisotropy = 4},
            .texture = SamplerFilterTexture::blackWithUpperLevel,
            .expected = SamplerFilterExpectation::black,
        },
        SamplerFilterCase{
            .name = "magnify 4x, min linear, mag nearest",
            .sampler = {.minFilter = FilterMode::linear, .magFilter = FilterMode::nearest},
            .texture = SamplerFilterTexture::smallCheckerboard,
            .expected = SamplerFilterExpectation::unfiltered,
        },
    };
    checkSamplerFilterCases(device, spec.backend, cases);
}

TEST_CASE("Portable anisotropic samplers filter linearly within a level and honor mipFilter nearest",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    using rachis::AddressMode;
    using rachis::FilterMode;
    using rachis::MipmapFilterMode;
    const std::array cases{
        SamplerFilterCase{
            .name = "minify 1.06x, anisotropy 16, mip nearest",
            .sampler = {.minFilter = FilterMode::linear,
                        .magFilter = FilterMode::linear,
                        .mipFilter = MipmapFilterMode::nearest,
                        .maxAnisotropy = 16},
            .texture = SamplerFilterTexture::slightlyMinifiedBlackWithUpperLevel,
            .expected = SamplerFilterExpectation::black,
            .mayBeUnsupportedOnD3D12 = true,
        },
        SamplerFilterCase{
            .name = "minify 1.06x, anisotropy 16, mip linear",
            .sampler = {.minFilter = FilterMode::linear,
                        .magFilter = FilterMode::linear,
                        .mipFilter = MipmapFilterMode::linear,
                        .maxAnisotropy = 16},
            .texture = SamplerFilterTexture::slightlyMinifiedBlackWithUpperLevel,
            .expected = SamplerFilterExpectation::partlyUpperLevel,
        },
        SamplerFilterCase{
            .name = "minify 1.06x, mip nearest",
            .sampler = {.minFilter = FilterMode::linear,
                        .magFilter = FilterMode::linear,
                        .mipFilter = MipmapFilterMode::nearest},
            .texture = SamplerFilterTexture::slightlyMinifiedBlackWithUpperLevel,
            .expected = SamplerFilterExpectation::black,
        },
        SamplerFilterCase{
            .name = "minify 1.06x, mip linear",
            .sampler = {.minFilter = FilterMode::linear,
                        .magFilter = FilterMode::linear,
                        .mipFilter = MipmapFilterMode::linear},
            .texture = SamplerFilterTexture::slightlyMinifiedBlackWithUpperLevel,
            .expected = SamplerFilterExpectation::partlyUpperLevel,
        },
        SamplerFilterCase{
            .name = "minify 4x, min nearest, mag nearest, anisotropy 4",
            .sampler = {.maxAnisotropy = 4},
            .texture = SamplerFilterTexture::checkerboard,
            .expected = SamplerFilterExpectation::filtered,
        },
        SamplerFilterCase{
            .name = "magnify 4x, min nearest, mag nearest, anisotropy 4",
            .sampler = {.addressU = AddressMode::repeat, .addressV = AddressMode::repeat, .maxAnisotropy = 4},
            .texture = SamplerFilterTexture::smallCheckerboard,
            .expected = SamplerFilterExpectation::magnifiedFiltered,
        },
    };
    checkSamplerFilterCases(device, spec.backend, cases);
}

TEST_CASE("Portable uniform-buffer bindings accept ranges that are not a multiple of 256 bytes",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    const auto small = createUniformRangeSource(device, 64, "Portable 64-byte uniform buffer");
    const auto odd = createUniformRangeSource(device, 80, "Portable 80-byte uniform buffer");
    const auto large = createUniformRangeSource(device, 1024, "Portable 1,024-byte uniform buffer");
    writeFloat4(device, small, 0, {1.0F, 2.0F, 3.0F, 4.0F});
    writeFloat4(device, odd, 0, {5.0F, 6.0F, 7.0F, 8.0F});
    writeFloat4(device, large, 0, {9.0F, 10.0F, 11.0F, 12.0F});
    writeFloat4(device, large, 256, {13.0F, 14.0F, 15.0F, 16.0F});
    const std::array pastEnd{std::byte{0}};
    CHECK_FALSE(device.writeBuffer(small, 64, pastEnd).has_value());

    const std::array cases{
        UniformRangeCase{
            .name = "64-byte buffer, whole resource",
            .binding = {.buffer = small},
            .expected = {1.0F, 2.0F, 3.0F, 4.0F},
        },
        UniformRangeCase{
            .name = "64-byte buffer, offset 0 size 64",
            .binding = {.buffer = small, .offset = 0, .size = 64},
            .expected = {1.0F, 2.0F, 3.0F, 4.0F},
        },
        UniformRangeCase{
            .name = "80-byte buffer, whole resource",
            .binding = {.buffer = odd},
            .expected = {5.0F, 6.0F, 7.0F, 8.0F},
        },
        UniformRangeCase{
            .name = "1,024-byte buffer, offset 256 size 64",
            .binding = {.buffer = large, .offset = 256, .size = 64},
            .expected = {13.0F, 14.0F, 15.0F, 16.0F},
        },
        UniformRangeCase{
            .name = "dynamic 1,024-byte buffer, size 64, dynamic offset 0",
            .binding = {.buffer = large, .offset = 0, .size = 64},
            .dynamic = true,
            .dynamicOffset = 0,
            .expected = {9.0F, 10.0F, 11.0F, 12.0F},
        },
        UniformRangeCase{
            .name = "dynamic 1,024-byte buffer, size 64, dynamic offset 256",
            .binding = {.buffer = large, .offset = 0, .size = 64},
            .dynamic = true,
            .dynamicOffset = 256,
            .expected = {13.0F, 14.0F, 15.0F, 16.0F},
        },
    };

    UniformRangeFixture fixture;
    const std::array staticEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    const std::array dynamicEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto staticLayout = device.createBindingLayout({.entries = staticEntries, .label = "Portable uniform range"});
    auto dynamicLayout =
        device.createBindingLayout({.entries = dynamicEntries, .label = "Portable dynamic uniform range"});
    REQUIRE(staticLayout.has_value());
    REQUIRE(dynamicLayout.has_value());
    fixture.staticLayout = *staticLayout;
    fixture.dynamicLayout = *dynamicLayout;
    const auto poolCapacity = static_cast<std::uint32_t>(cases.size() + 1);
    const std::array poolSizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, poolCapacity},
        rachis::BindingPoolSize{rachis::BindingType::dynamicUniformBuffer, poolCapacity},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, poolCapacity},
    };
    auto pool = device.createBindingPool({
        .maxBindingSets = poolCapacity,
        .sizes = poolSizes,
        .label = "Portable uniform range pool",
    });
    REQUIRE(pool.has_value());

    auto output = device.createBuffer({
        .size = uniformRangeOutputSlotSize * cases.size(),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Portable uniform range output",
    });
    auto readback = device.createBuffer({
        .size = uniformRangeOutputSlotSize * cases.size(),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Portable uniform range readback",
    });
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());
    fixture.output = *output;
    fixture.readback = *readback;

    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        const auto& testCase = cases[index];
        CAPTURE(testCase.name);
        const std::array writes{
            rachis::BindingWrite{.binding = 0, .resource = testCase.binding},
            rachis::BindingWrite{
                .binding = 1,
                .resource =
                    rachis::BufferBinding{
                        .buffer = fixture.output,
                        .offset = uniformRangeOutputSlotSize * index,
                        .size = sizeof(Float4),
                        .structureByteStride = sizeof(Float4),
                    },
            },
        };
        auto set = device.createBindingSet({
            .layout = testCase.dynamic ? fixture.dynamicLayout : fixture.staticLayout,
            .writes = writes,
            .label = "Portable uniform range set",
            .pool = *pool,
        });
        const std::string failure = set ? std::string{} : set.error().message;
        CAPTURE(failure);
        REQUIRE(set.has_value());
        fixture.sets.push_back(*set);
    }

    const std::array outOfBoundsWrites{
        rachis::BindingWrite{.binding = 0,
                             .resource = rachis::BufferBinding{.buffer = small, .offset = 0, .size = 256}},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = fixture.output,
                    .size = sizeof(Float4),
                    .structureByteStride = sizeof(Float4),
                },
        },
    };
    auto outOfBounds = device.createBindingSet({
        .layout = fixture.staticLayout,
        .writes = outOfBoundsWrites,
        .pool = *pool,
    });
    REQUIRE_FALSE(outOfBounds.has_value());
    CHECK(outOfBounds.error().code == rachis::ErrorCode::invalidArgument);

    const auto bytecode = uniformBufferRangeShader(spec.backend);
    if (!bytecode.empty())
    {
        copyUniformRanges(device, bytecode, fixture, cases);
    }

    for (const auto set : fixture.sets)
    {
        REQUIRE(device.destroyBindingSet(set).has_value());
    }
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyBindingLayout(fixture.dynamicLayout).has_value());
    REQUIRE(device.destroyBindingLayout(fixture.staticLayout).has_value());
    REQUIRE(device.destroyBuffer(fixture.readback).has_value());
    REQUIRE(device.destroyBuffer(fixture.output).has_value());
    REQUIRE(device.destroyBuffer(large).has_value());
    REQUIRE(device.destroyBuffer(odd).has_value());
    REQUIRE(device.destroyBuffer(small).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable dynamic uniform-buffer arrays bind 64-byte ranges at dynamic offsets",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    const auto source = createUniformRangeSource(device, 1024, "Portable dynamic uniform array source");
    const Float4 first{9.0F, 10.0F, 11.0F, 12.0F};
    const Float4 second{13.0F, 14.0F, 15.0F, 16.0F};
    writeFloat4(device, source, 0, first);
    writeFloat4(device, source, 256, second);
    const std::array cases{
        DynamicUniformArrayCase{.offsets = {0, 256}, .expected = {first, second}},
        DynamicUniformArrayCase{.offsets = {256, 0}, .expected = {second, first}},
    };

    DynamicUniformArrayFixture fixture;
    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 2},
        rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Portable dynamic uniform array"});
    const std::string layoutFailure = layout ? std::string{} : layout.error().message;
    CAPTURE(layoutFailure);
    REQUIRE(layout.has_value());
    fixture.layout = *layout;
    const auto setCount = static_cast<std::uint32_t>(cases.size());
    const std::array poolSizes{
        rachis::BindingPoolSize{rachis::BindingType::dynamicUniformBuffer, 2 * setCount},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, setCount},
    };
    auto pool = device.createBindingPool({
        .maxBindingSets = setCount,
        .sizes = poolSizes,
        .label = "Portable dynamic uniform array pool",
    });
    REQUIRE(pool.has_value());

    auto output = device.createBuffer({
        .size = uniformRangeOutputSlotSize * cases.size(),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Portable dynamic uniform array output",
    });
    auto readback = device.createBuffer({
        .size = uniformRangeOutputSlotSize * cases.size(),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Portable dynamic uniform array readback",
    });
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());
    fixture.output = *output;
    fixture.readback = *readback;

    const rachis::BufferBinding range{.buffer = source, .offset = 0, .size = 64};
    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        CAPTURE(index);
        const std::array writes{
            rachis::BindingWrite{.binding = 0, .arrayElement = 0, .resource = range},
            rachis::BindingWrite{.binding = 0, .arrayElement = 1, .resource = range},
            rachis::BindingWrite{
                .binding = 2,
                .resource =
                    rachis::BufferBinding{
                        .buffer = fixture.output,
                        .offset = uniformRangeOutputSlotSize * index,
                        .size = 2 * sizeof(Float4),
                        .structureByteStride = sizeof(Float4),
                    },
            },
        };
        auto set = device.createBindingSet({
            .layout = fixture.layout,
            .writes = writes,
            .label = "Portable dynamic uniform array set",
            .pool = *pool,
        });
        const std::string failure = set ? std::string{} : set.error().message;
        CAPTURE(failure);
        REQUIRE(set.has_value());
        fixture.sets.push_back(*set);
    }

    const auto bytecode = dynamicUniformBufferArrayShader(spec.backend);
    if (!bytecode.empty())
    {
        copyDynamicUniformArrays(device, bytecode, fixture, cases);
    }

    for (const auto set : fixture.sets)
    {
        REQUIRE(device.destroyBindingSet(set).has_value());
    }
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyBindingLayout(fixture.layout).has_value());
    REQUIRE(device.destroyBuffer(fixture.readback).has_value());
    REQUIRE(device.destroyBuffer(fixture.output).has_value());
    REQUIRE(device.destroyBuffer(source).has_value());
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable binding pools hold more than 2,048 sets that share one sampler", RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(enabledBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;

    auto sampler = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .label = "Portable shared sampler",
    });
    REQUIRE(sampler.has_value());
    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Portable shared sampler layout"});
    REQUIRE(layout.has_value());

    constexpr std::uint32_t setsPerPool = 1024;
    constexpr std::uint32_t poolCount = 3;
    const std::array poolSizes{rachis::BindingPoolSize{rachis::BindingType::sampler, setsPerPool}};
    std::array<rachis::BindingPoolHandle, poolCount> pools{};
    for (auto& pool : pools)
    {
        auto created = device.createBindingPool({
            .maxBindingSets = setsPerPool,
            .sizes = poolSizes,
            .label = "Portable shared sampler pool",
        });
        REQUIRE(created.has_value());
        pool = *created;
    }

    const std::array writes{rachis::BindingWrite{0, 0, *sampler}};
    std::vector<rachis::BindingSetHandle> sets;
    sets.reserve(setsPerPool * poolCount);
    for (std::uint32_t index = 0; index < setsPerPool * poolCount; ++index)
    {
        auto set = device.createBindingSet({
            .layout = *layout,
            .writes = writes,
            .label = "Portable shared sampler set",
            .pool = pools[index % poolCount],
        });
        const std::string failure = set ? std::string{} : set.error().message;
        CAPTURE(index, failure);
        REQUIRE(set.has_value());
        sets.push_back(*set);
    }

    for (const auto set : sets)
    {
        REQUIRE(device.destroyBindingSet(set).has_value());
    }
    for (const auto pool : pools)
    {
        REQUIRE(device.destroyBindingPool(pool).has_value());
    }
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroySampler(*sampler).has_value());
    checkNoDiagnosticsErrors(device);
}

#if defined(RACHIS_PORTABLE_DYNAMIC_COUNTER_SHADERS)
TEST_CASE("Portable dynamic storage buffer binds stay bounded over 70,000 binds in one command buffer",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(dynamicCounterBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;
    const auto fixture = createDynamicCounterFixture(device, spec);
    checkUnrelatedDescriptorUsers(device, fixture);
    device.collectGarbage();

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    recordDynamicCounterPrologue(device, *command, fixture);
    const auto beforeBinds = device.captureBackendAllocationSnapshot();
    constexpr std::uint32_t warmBindCount = 3 * dynamicCounterSlotCount;
    constexpr std::uint32_t bindCount = 70'000;
    recordDynamicCounterBinds(device, *command, fixture, 0, warmBindCount);
    const auto afterWarmBinds = device.captureBackendAllocationSnapshot();
    recordDynamicCounterBinds(device, *command, fixture, warmBindCount, bindCount);
    const auto afterBinds = device.captureBackendAllocationSnapshot();
    CHECK(afterBinds.liveDescriptorCount == afterWarmBinds.liveDescriptorCount);
    CHECK(afterBinds.liveDescriptorCount <= beforeBinds.liveDescriptorCount + 2ULL * dynamicCounterSlotCount);

    checkUnrelatedDescriptorUsers(device, fixture);

    recordDynamicCounterEpilogue(device, *command, fixture);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    checkDynamicCounters(device, fixture, bindCount);
    device.collectGarbage();
    CHECK(device.captureBackendAllocationSnapshot().liveDescriptorCount <= beforeBinds.liveDescriptorCount);

    destroyDynamicCounterFixture(device, fixture);
    checkNoDiagnosticsErrors(device);
}

TEST_CASE("Portable dynamic storage buffer binds succeed in command buffers recorded before any submit",
          RACHIS_PORTABLE_CONTRACT_TAGS)
{
    const auto& spec = GENERATE_REF(from_range(dynamicCounterBackends()));
    CAPTURE(spec.name);

    auto maybeDevice = createPortableDeviceOrSkip(spec);
    if (!maybeDevice)
    {
        return;
    }
    auto& device = *maybeDevice;
    const auto fixture = createDynamicCounterFixture(device, spec);

    constexpr std::uint32_t commandBufferCount = 3;
    constexpr std::uint32_t bindsPerCommandBuffer = 1'500;
    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = commandBufferCount,
        .label = "Portable deferred dynamic binding pool",
    });
    REQUIRE(pool.has_value());
    std::vector<rachis::CommandBuffer> commands;
    for (std::uint32_t commandIndex = 0; commandIndex < commandBufferCount; ++commandIndex)
    {
        auto command = device.acquireCommandBuffer(*pool);
        REQUIRE(command.has_value());
        if (commandIndex == 0)
        {
            recordDynamicCounterPrologue(device, *command, fixture);
        }
        else
        {
            REQUIRE(device.bindComputePipeline(*command, fixture.pipeline).has_value());
        }
        CAPTURE(commandIndex);
        recordDynamicCounterBinds(device,
                                  *command,
                                  fixture,
                                  commandIndex * bindsPerCommandBuffer,
                                  (commandIndex + 1) * bindsPerCommandBuffer);
        commands.push_back(std::move(*command));
    }
    checkUnrelatedDescriptorUsers(device, fixture);

    recordDynamicCounterEpilogue(device, commands.back(), fixture);
    std::vector<rachis::TimelinePoint> submittedPoints;
    for (auto& command : commands)
    {
        auto submitted = device.submit(std::move(command));
        REQUIRE(submitted.has_value());
        submittedPoints.push_back(*submitted);
    }
    for (const auto& point : submittedPoints)
    {
        REQUIRE(device.wait(point).has_value());
    }
    checkDynamicCounters(device, fixture, commandBufferCount * bindsPerCommandBuffer);

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    destroyDynamicCounterFixture(device, fixture);
    checkNoDiagnosticsErrors(device);
}
#endif
