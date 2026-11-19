#include "Rachis/Device.h"
#include "RhiTestSupport.h"
#include "ShaderInt64AtomicStorageBuffer.h"
#include "SubgroupSize128.h"
#include "SubgroupSize16.h"
#include "SubgroupSize32.h"
#include "SubgroupSize4.h"
#include "SubgroupSize64.h"
#include "SubgroupSize8.h"
#include "SubgroupSizeAny.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint64_t atomicSum = (64U * 65U) / 2U;

    constexpr std::array<std::uint32_t, 6> subgroupSizes{4, 8, 16, 32, 64, 128};
    constexpr std::uint32_t anySizeGroupWidth = 128;
    constexpr std::uint32_t laneCountGroupCount = 2;

    using LaneCounts = std::vector<std::array<std::uint32_t, 2>>;

    [[nodiscard]] std::span<const std::byte> subgroupSizeShader(std::uint32_t size)
    {
        constexpr auto backend = rachis::Backend::vulkan;
        switch (size)
        {
        case 4:
            return rachis_shaders::SubgroupSize4(backend);
        case 8:
            return rachis_shaders::SubgroupSize8(backend);
        case 16:
            return rachis_shaders::SubgroupSize16(backend);
        case 32:
            return rachis_shaders::SubgroupSize32(backend);
        case 64:
            return rachis_shaders::SubgroupSize64(backend);
        case 128:
            return rachis_shaders::SubgroupSize128(backend);
        default:
            return rachis_shaders::SubgroupSizeAny(backend);
        }
    }

    [[nodiscard]] rachis::Result<LaneCounts> runLaneCountKernel(rachis::Device& device,
                                                                std::span<const std::byte> bytecode,
                                                                std::uint32_t requiredSubgroupSize,
                                                                std::uint32_t groupWidth)
    {
        REQUIRE_FALSE(bytecode.empty());
        const std::uint32_t laneCountInvocations = groupWidth * laneCountGroupCount;
        const std::uint64_t laneCountBytes = std::uint64_t{laneCountInvocations} * 2U * sizeof(std::uint32_t);
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .label = "Vulkan subgroup size shader",
            .shaderInterface = rachis::test::computeShaderInterface(
                {groupWidth, 1, 1},
                std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
        });
        if (!shader)
        {
            UNSCOPED_INFO(shader.error().message);
        }
        REQUIRE(shader);
        const std::array entries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        };
        auto layout = device.createBindingLayout({.entries = entries});
        REQUIRE(layout);
        const std::array layouts{*layout};
        auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
        REQUIRE(pipelineLayout);
        auto pipeline = device.createComputePipeline({
            .layout = *pipelineLayout,
            .computeShader = *shader,
            .label = "Vulkan subgroup size pipeline",
            .requiredSubgroupSize = requiredSubgroupSize,
        });
        if (!pipeline)
        {
            CHECK(device.destroyPipelineLayout(*pipelineLayout));
            CHECK(device.destroyBindingLayout(*layout));
            CHECK(device.destroyShaderModule(*shader));
            return std::unexpected(std::move(pipeline.error()));
        }

        auto storage = device.createBuffer({
            .size = laneCountBytes,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
            .label = "Vulkan subgroup size output",
        });
        auto readback = device.createBuffer({
            .size = laneCountBytes,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Vulkan subgroup size readback",
        });
        REQUIRE(storage);
        REQUIRE(readback);
        const std::array writes{
            rachis::BindingWrite{0, 0, rachis::BufferBinding{*storage, 0, laneCountBytes, 2U * sizeof(std::uint32_t)}},
        };
        auto set = rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes});
        REQUIRE(set);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command);
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *storage,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
        REQUIRE(device.bindComputePipeline(*command, *pipeline));
        REQUIRE(device.bindComputeBindingSet(*command, 0, *set));
        REQUIRE(device.dispatch(*command, laneCountGroupCount));
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *storage,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device.copyBuffer(*command, {*storage, 0, *readback, 0, laneCountBytes}));
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted));
        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        LaneCounts lanes(laneCountInvocations);
        std::memcpy(lanes.data(), mapped->data, laneCountBytes);
        REQUIRE(device.unmapBuffer(*readback));

        REQUIRE(device.destroyPipeline(*pipeline));
        REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
        REQUIRE(device.destroyBindingSet(*set));
        REQUIRE(device.destroyBindingLayout(*layout));
        REQUIRE(device.destroyShaderModule(*shader));
        REQUIRE(device.destroyBuffer(*storage));
        REQUIRE(device.destroyBuffer(*readback));
        return lanes;
    }

    constexpr std::uint32_t storageAtomicGroupSize = 64;
    constexpr std::uint32_t storageAtomicGroupCount = 16;

    [[nodiscard]] constexpr std::uint64_t storageAtomicKey(std::uint32_t lane) noexcept
    {
        const std::uint32_t high = ((lane >> 1U) + 1U) * 2654435761U;
        const std::uint32_t low = lane * 2246822519U;
        return (std::uint64_t{high} << 32U) | low;
    }

    [[nodiscard]] constexpr std::array<std::uint64_t, 2> expectedStorageAtomicExtremes() noexcept
    {
        std::array<std::uint64_t, 2> extremes{0, std::numeric_limits<std::uint64_t>::max()};
        for (std::uint32_t lane = 0; lane < storageAtomicGroupSize * storageAtomicGroupCount; ++lane)
        {
            const auto key = storageAtomicKey(lane);
            extremes[0] = key > extremes[0] ? key : extremes[0];
            extremes[1] = key < extremes[1] ? key : extremes[1];
        }
        return extremes;
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

#ifdef RACHIS_SPIRV_DIR
    [[nodiscard]] std::vector<std::byte> loadSpirv(const char* name)
    {
        return loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / name);
    }

    constexpr std::uint32_t typeFeatureElementCount = 8;
    constexpr std::uint32_t stageTargetSize = 4;
    constexpr std::uint32_t stageRowPitch = 256;
    constexpr std::uint64_t stageImageBytes = static_cast<std::uint64_t>(stageRowPitch) * stageTargetSize;

    [[nodiscard]] constexpr std::uint16_t halfFromInteger(std::uint32_t value) noexcept
    {
        const auto exponent = static_cast<std::uint32_t>(std::bit_width(value)) - 1U;
        const auto mantissa = (value - (1U << exponent)) << (10U - exponent);
        return static_cast<std::uint16_t>(((exponent + 15U) << 10U) | mantissa);
    }

    struct InPlaceDispatch final
    {
        const char* file = nullptr;
        const char* entryPoint = nullptr;
        std::uint32_t binding = 0;
    };

    [[nodiscard]] std::optional<std::vector<std::byte>> dispatchInPlace(rachis::Device& device,
                                                                        const InPlaceDispatch& dispatch,
                                                                        std::span<const std::byte> initial)
    {
        const auto bytecode = loadSpirv(dispatch.file);
        REQUIRE_FALSE(bytecode.empty());
        const std::array requirements{
            rachis::ShaderBindingRequirement{0, dispatch.binding, rachis::ShaderResourceType::storageBuffer, 1},
        };
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = dispatch.entryPoint,
            .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements),
        });
        if (!shader)
        {
            UNSCOPED_INFO(shader.error().message);
            CHECK(shader.error().code == rachis::ErrorCode::unsupported);
            return std::nullopt;
        }

        const auto size = static_cast<std::uint64_t>(initial.size());
        auto data = device.createBuffer({
            .size = size,
            .usage =
                rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        });
        auto upload = device.createBuffer({
            .size = size,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        auto readback = device.createBuffer({
            .size = size,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(data);
        REQUIRE(upload);
        REQUIRE(readback);
        REQUIRE(device.writeBuffer(*upload, 0, initial));

        const std::array entries{
            rachis::BindingLayoutEntry{
                dispatch.binding, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        };
        auto layout = device.createBindingLayout({.entries = entries});
        REQUIRE(layout);
        const std::array writes{
            rachis::BindingWrite{dispatch.binding, 0, rachis::BufferBinding{*data, 0, size, sizeof(std::uint32_t)}},
        };
        auto set = rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes});
        REQUIRE(set);
        const std::array layouts{*layout};
        auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
        REQUIRE(pipelineLayout);
        auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
        REQUIRE(pipeline);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command);
        REQUIRE(device.copyBuffer(*command, {*upload, 0, *data, 0, size}));
        REQUIRE(device.bufferBarrier(*command,
                                     {
                                         .buffer = *data,
                                         .stagesBefore = rachis::PipelineStage::copy,
                                         .stagesAfter = rachis::PipelineStage::computeShader,
                                         .accessBefore = rachis::Access::copyDestination,
                                         .accessAfter = rachis::Access::unorderedAccess,
                                     }));
        REQUIRE(device.bindComputePipeline(*command, *pipeline));
        REQUIRE(device.bindComputeBindingSet(*command, 0, *set));
        REQUIRE(device.dispatch(*command, typeFeatureElementCount));
        REQUIRE(device.bufferBarrier(*command,
                                     {
                                         .buffer = *data,
                                         .stagesBefore = rachis::PipelineStage::computeShader,
                                         .stagesAfter = rachis::PipelineStage::copy,
                                         .accessBefore = rachis::Access::unorderedAccess,
                                         .accessAfter = rachis::Access::copySource,
                                     }));
        REQUIRE(device.copyBuffer(*command, {*data, 0, *readback, 0, size}));
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted));
        std::vector<std::byte> result(initial.size());
        REQUIRE(device.readBuffer(*readback, 0, result));

        REQUIRE(device.destroyPipeline(*pipeline));
        REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
        REQUIRE(device.destroyBindingSet(*set));
        REQUIRE(device.destroyBindingLayout(*layout));
        REQUIRE(device.destroyBuffer(*readback));
        REQUIRE(device.destroyBuffer(*upload));
        REQUIRE(device.destroyBuffer(*data));
        REQUIRE(device.destroyShaderModule(*shader));
        return result;
    }

    struct StagePair final
    {
        const char* vertexFile = nullptr;
        const char* vertexEntry = nullptr;
        const char* pixelFile = nullptr;
        const char* pixelEntry = nullptr;
    };

    [[nodiscard]] std::optional<std::vector<std::byte>> renderStagePair(rachis::Device& device, const StagePair& pair)
    {
        const auto vertexBytes = loadSpirv(pair.vertexFile);
        const auto pixelBytes = loadSpirv(pair.pixelFile);
        REQUIRE_FALSE(vertexBytes.empty());
        REQUIRE_FALSE(pixelBytes.empty());
        auto vertex = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vertexBytes,
            .entryPoint = pair.vertexEntry,
        });
        auto pixel = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = pixelBytes,
            .entryPoint = pair.pixelEntry,
        });
        if (!vertex || !pixel)
        {
            for (const auto* module : {&vertex, &pixel})
            {
                if (*module)
                {
                    CHECK(device.destroyShaderModule(**module).has_value());
                    continue;
                }
                UNSCOPED_INFO(module->error().message);
                CHECK(module->error().code == rachis::ErrorCode::unsupported);
            }
            return std::nullopt;
        }

        auto layout = device.createPipelineLayout({});
        REQUIRE(layout);
        const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
        auto pipeline = device.createGraphicsPipeline({
            .layout = *layout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        });
        REQUIRE(pipeline);
        auto target = device.createTexture({
            .width = stageTargetSize,
            .height = stageTargetSize,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        });
        auto readback = device.createBuffer({
            .size = stageImageBytes,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(target);
        REQUIRE(readback);
        auto view = device.createTextureView({.texture = *target, .usage = rachis::TextureViewUsage::renderTarget});
        REQUIRE(view);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(device.textureBarrier(*command,
                                      {
                                          .texture = *target,
                                          .stagesAfter = rachis::PipelineStage::renderTarget,
                                          .accessAfter = rachis::Access::renderTarget,
                                          .layoutAfter = rachis::TextureLayout::renderTarget,
                                      }));
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *view,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
            },
        };
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}));
        REQUIRE(device.bindGraphicsPipeline(*command, *pipeline));
        REQUIRE(device.setViewport(*command,
                                   {
                                       .width = static_cast<float>(stageTargetSize),
                                       .height = static_cast<float>(stageTargetSize),
                                       .maxDepth = 1.0F,
                                   }));
        REQUIRE(device.setScissor(*command, {.width = stageTargetSize, .height = stageTargetSize}));
        REQUIRE(device.draw(*command, 3));
        REQUIRE(device.endRendering(*command));
        REQUIRE(device.textureBarrier(*command,
                                      {
                                          .texture = *target,
                                          .stagesBefore = rachis::PipelineStage::renderTarget,
                                          .stagesAfter = rachis::PipelineStage::copy,
                                          .accessBefore = rachis::Access::renderTarget,
                                          .accessAfter = rachis::Access::copySource,
                                          .layoutBefore = rachis::TextureLayout::renderTarget,
                                          .layoutAfter = rachis::TextureLayout::copySource,
                                      }));
        REQUIRE(device.copyTextureToBuffer(*command,
                                           {
                                               .buffer = *readback,
                                               .bytesPerRow = stageRowPitch,
                                               .rowsPerImage = stageTargetSize,
                                               .texture = *target,
                                               .extent = {stageTargetSize, stageTargetSize, 1},
                                           }));
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted));
        std::vector<std::byte> pixels(static_cast<std::size_t>(stageImageBytes));
        REQUIRE(device.readBuffer(*readback, 0, pixels));

        REQUIRE(device.destroyTextureView(*view));
        REQUIRE(device.destroyTexture(*target));
        REQUIRE(device.destroyBuffer(*readback));
        REQUIRE(device.destroyPipeline(*pipeline));
        REQUIRE(device.destroyPipelineLayout(*layout));
        REQUIRE(device.destroyShaderModule(*pixel));
        REQUIRE(device.destroyShaderModule(*vertex));
        return pixels;
    }

    [[nodiscard]] bool everyPixelNear(std::span<const std::byte> pixels, std::array<std::uint8_t, 4> expected) noexcept
    {
        for (std::uint32_t y = 0; y < stageTargetSize; ++y)
        {
            for (std::uint32_t x = 0; x < stageTargetSize; ++x)
            {
                const auto offset = static_cast<std::size_t>(y) * stageRowPitch + static_cast<std::size_t>(x) * 4;
                for (std::size_t channel = 0; channel < expected.size(); ++channel)
                {
                    const auto actual = std::to_integer<int>(pixels[offset + channel]);
                    if (actual < expected[channel] - 1 || actual > expected[channel] + 1)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }
#endif

} // namespace

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan 16-bit and 64-bit shader arithmetic produce exact GPU results",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][shader][arithmetic][compute]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipUnlessReferenceDesktopAdvertises(rachis::test::requireCapabilities(device).shaderFloat16,
                                                           "shaderFloat16") ||
        rachis::test::skipUnlessReferenceDesktopAdvertises(rachis::test::requireCapabilities(device).shaderInt16,
                                                           "shaderInt16") ||
        rachis::test::skipUnlessReferenceDesktopAdvertises(rachis::test::requireCapabilities(device).shaderInt64,
                                                           "shaderInt64"))
    {
        return;
    }

    const auto bytecode16 = loadSpirv("Shader16Bit.spv");
    const auto bytecode64 = loadSpirv("ShaderInt64.spv");
    REQUIRE_FALSE(bytecode16.empty());
    REQUIRE_FALSE(bytecode64.empty());

    auto shader16 = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode16,
        .entryPoint = "main",
        .label = "Vulkan 16-bit arithmetic",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}},
            std::array{rachis::ShaderPushConstantRequirement{0, 8}}),
    });
    auto shader64 = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode64,
        .entryPoint = "main",
        .label = "Vulkan 64-bit arithmetic",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}},
            std::array{rachis::ShaderPushConstantRequirement{0, 16}}),
    });
    REQUIRE(shader16);
    REQUIRE(shader64);

    constexpr std::uint64_t output16 = 8;
    constexpr std::uint64_t output64 = 16;
    auto storage16 = device.createBuffer({
        .size = output16,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Vulkan 16-bit output",
    });
    auto storage64 = device.createBuffer({
        .size = output64,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Vulkan 64-bit output",
    });
    auto readback = device.createBuffer({
        .size = output64,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan arithmetic readback",
    });
    REQUIRE(storage16);
    REQUIRE(storage64);
    REQUIRE(readback);

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Vulkan arithmetic layout"});
    REQUIRE(layout);
    const std::array writes16{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*storage16, 0, output16, sizeof(std::uint32_t)}},
    };
    const std::array writes64{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*storage64, 0, output64, sizeof(std::uint32_t)}},
    };
    auto set16 = rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes16});
    auto set64 = rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes64});
    REQUIRE(set16);
    REQUIRE(set64);
    const std::array layouts{*layout};
    const std::array push16{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 8}};
    const std::array push64{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    auto pipelineLayout16 = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = push16});
    auto pipelineLayout64 = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = push64});
    REQUIRE(pipelineLayout16);
    REQUIRE(pipelineLayout64);
    auto pipeline16 = device.createComputePipeline({.layout = *pipelineLayout16, .computeShader = *shader16});
    auto pipeline64 = device.createComputePipeline({.layout = *pipelineLayout64, .computeShader = *shader64});
    REQUIRE(pipeline16);
    REQUIRE(pipeline64);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *storage16,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline16));
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set16));
    const std::array constants16{std::bit_cast<std::uint32_t>(4.0F), std::uint32_t{123}};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(constants16))));
    REQUIRE(device.dispatch(*command, 1));
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *storage16,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {*storage16, 0, *readback, 0, output16}));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));

    auto mapped16 = device.mapBuffer(*readback);
    REQUIRE(mapped16);
    std::array<std::uint32_t, 2> result16{};
    std::memcpy(result16.data(), mapped16->data, output16);
    CHECK(result16 == std::array<std::uint32_t, 2>{8, 364});
    REQUIRE(device.unmapBuffer(*readback));

    auto command64 = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command64);
    REQUIRE(device
                .bufferBarrier(*command64,
                               {
                                   .buffer = *storage64,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command64, *pipeline64));
    REQUIRE(device.bindComputeBindingSet(*command64, 0, *set64));
    const std::array constants64{
        std::uint32_t{123},
        std::uint32_t{256},
        std::bit_cast<std::uint32_t>(std::int32_t{-17}),
        std::uint32_t{0},
    };
    REQUIRE(device.setComputePushConstants(*command64, 0, std::as_bytes(std::span(constants64))));
    REQUIRE(device.dispatch(*command64, 1));
    REQUIRE(device
                .bufferBarrier(*command64,
                               {
                                   .buffer = *storage64,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command64, {*storage64, 0, *readback, 0, output64}));
    auto submitted64 = device.submit(std::move(*command64));
    REQUIRE(submitted64);
    REQUIRE(device.wait(*submitted64));

    auto mapped64 = device.mapBuffer(*readback);
    REQUIRE(mapped64);
    std::array<std::uint32_t, 4> result64{};
    std::memcpy(result64.data(), mapped64->data, output64);
    CHECK(result64 == std::array<std::uint32_t, 4>{376, 768, 106, 128});
    REQUIRE(device.unmapBuffer(*readback));

    REQUIRE(device.destroyPipeline(*pipeline16));
    REQUIRE(device.destroyPipeline(*pipeline64));
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout16));
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout64));
    REQUIRE(device.destroyBindingSet(*set16));
    REQUIRE(device.destroyBindingSet(*set64));
    REQUIRE(device.destroyBindingLayout(*layout));
    REQUIRE(device.destroyShaderModule(*shader16));
    REQUIRE(device.destroyShaderModule(*shader64));
    REQUIRE(device.destroyBuffer(*storage16));
    REQUIRE(device.destroyBuffer(*storage64));
    REQUIRE(device.destroyBuffer(*readback));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan 16-bit storage, 16-bit stage I/O, and 64-bit float shaders run cleanly or report unsupported",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][shader][arithmetic][compute]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);

    SECTION("a half2 storage buffer is read and written in place or the module is unsupported")
    {
        std::array<std::uint16_t, typeFeatureElementCount * 2> initial{};
        std::array<std::uint16_t, typeFeatureElementCount * 2> expected{};
        for (std::uint32_t index = 0; index < initial.size(); ++index)
        {
            initial[index] = halfFromInteger(index + 1);
            expected[index] = halfFromInteger(((index + 1) * 2) + 1);
        }
        const auto result = dispatchInPlace(
            device,
            {.file = "VulkanShaderTypeFeatures.half-storage.cs.spv", .entryPoint = "halfStorageCompute", .binding = 0},
            std::as_bytes(std::span(initial)));
        if (result.has_value())
        {
            CHECK(capabilities.shaderFloat16);
            std::array<std::uint16_t, typeFeatureElementCount * 2> actual{};
            REQUIRE(result->size() == sizeof(actual));
            std::memcpy(actual.data(), result->data(), sizeof(actual));
            CHECK(actual == expected);
        }
    }

    SECTION("double arithmetic computes exact results or the module is unsupported")
    {
        std::array<std::uint32_t, typeFeatureElementCount> initial{};
        std::array<std::uint32_t, typeFeatureElementCount> expected{};
        for (std::uint32_t index = 0; index < initial.size(); ++index)
        {
            initial[index] = index + 1;
            expected[index] = ((index + 1) * 2) + 1;
        }
        const auto result = dispatchInPlace(
            device,
            {.file = "VulkanShaderTypeFeatures.double.cs.spv", .entryPoint = "doubleCompute", .binding = 1},
            std::as_bytes(std::span(initial)));
        if (result.has_value())
        {
            std::array<std::uint32_t, typeFeatureElementCount> actual{};
            REQUIRE(result->size() == sizeof(actual));
            std::memcpy(actual.data(), result->data(), sizeof(actual));
            CHECK(actual == expected);
        }
    }

    SECTION("half stage inputs and outputs render the same color as float ones or the modules are unsupported")
    {
        constexpr std::array<std::uint8_t, 4> expectedColor{64, 128, 191, 255};
        const auto floatPixels = renderStagePair(device,
                                                 {
                                                     .vertexFile = "VulkanShaderTypeFeatures.float-stage.vs.spv",
                                                     .vertexEntry = "floatStageVertex",
                                                     .pixelFile = "VulkanShaderTypeFeatures.float-stage.ps.spv",
                                                     .pixelEntry = "floatStagePixel",
                                                 });
        REQUIRE(floatPixels.has_value());
        CHECK(everyPixelNear(*floatPixels, expectedColor));

        const auto halfPixels = renderStagePair(device,
                                                {
                                                    .vertexFile = "VulkanShaderTypeFeatures.half-stage.vs.spv",
                                                    .vertexEntry = "halfStageVertex",
                                                    .pixelFile = "VulkanShaderTypeFeatures.half-stage.ps.spv",
                                                    .pixelEntry = "halfStagePixel",
                                                });
        if (halfPixels.has_value())
        {
            CHECK(capabilities.shaderFloat16);
            CHECK(everyPixelNear(*halfPixels, expectedColor));
        }
    }

    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan group-shared and typed 64-bit atomics accumulate every compute lane",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][shader][atomic][compute]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(device).shaderInt64AtomicGroupShared,
                                            "shaderInt64AtomicGroupShared") ||
        rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(device).shaderInt64AtomicTypedResource,
                                            "shaderInt64AtomicTypedResource"))
    {
        return;
    }

    const auto groupBytecode = loadSpirv("ShaderInt64AtomicGroupShared.spv");
    const auto typedBytecode = loadSpirv("ShaderInt64AtomicTypedResource.spv");
    REQUIRE_FALSE(groupBytecode.empty());
    REQUIRE_FALSE(typedBytecode.empty());

    auto groupShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = groupBytecode,
        .entryPoint = "main",
        .label = "Vulkan group-shared int64 atomics",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
    });
    REQUIRE(groupShader);

    constexpr std::uint64_t groupBytes = 8;
    auto groupOutput = device.createBuffer({
        .size = groupBytes,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Vulkan group-shared atomic output",
    });
    auto groupReadback = device.createBuffer({
        .size = groupBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan group-shared atomic readback",
    });
    REQUIRE(groupOutput);
    REQUIRE(groupReadback);
    const std::array groupEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto groupLayout = device.createBindingLayout({.entries = groupEntries});
    REQUIRE(groupLayout);
    const std::array groupWrites{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*groupOutput, 0, groupBytes, 4}},
    };
    auto groupSet = rachis::test::createBindingSet(device, {.layout = *groupLayout, .writes = groupWrites});
    REQUIRE(groupSet);
    const std::array groupLayouts{*groupLayout};
    auto groupPipelineLayout = device.createPipelineLayout({.bindingLayouts = groupLayouts});
    REQUIRE(groupPipelineLayout);
    auto groupPipeline = device.createComputePipeline({.layout = *groupPipelineLayout, .computeShader = *groupShader});
    REQUIRE(groupPipeline);

    auto groupCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(groupCommand);
    REQUIRE(device
                .bufferBarrier(*groupCommand,
                               {
                                   .buffer = *groupOutput,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*groupCommand, *groupPipeline));
    REQUIRE(device.bindComputeBindingSet(*groupCommand, 0, *groupSet));
    REQUIRE(device.dispatch(*groupCommand, 1));
    REQUIRE(device
                .bufferBarrier(*groupCommand,
                               {
                                   .buffer = *groupOutput,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*groupCommand, {*groupOutput, 0, *groupReadback, 0, groupBytes}));
    auto groupSubmitted = device.submit(std::move(*groupCommand));
    REQUIRE(groupSubmitted);
    REQUIRE(device.wait(*groupSubmitted));
    auto groupMapped = device.mapBuffer(*groupReadback);
    REQUIRE(groupMapped);
    std::array<std::uint32_t, 2> groupResult{};
    std::memcpy(groupResult.data(), groupMapped->data, groupBytes);
    CHECK(groupResult == std::array<std::uint32_t, 2>{static_cast<std::uint32_t>(atomicSum), 0});
    REQUIRE(device.unmapBuffer(*groupReadback));

    auto typedShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = typedBytecode,
        .entryPoint = "main",
        .label = "Vulkan typed int64 atomics",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageTexture, 1}}),
    });
    REQUIRE(typedShader);
    auto typedTexture = device.createTexture({
        .format = rachis::Format::r64Uint,
        .usage = rachis::TextureUsage::storage | rachis::TextureUsage::copySource,
        .label = "Vulkan typed int64 atomic output",
    });
    REQUIRE(typedTexture);
    auto typedView = device.createTextureView({
        .texture = *typedTexture,
        .usage = rachis::TextureViewUsage::storage,
        .label = "Vulkan typed int64 atomic UAV",
    });
    REQUIRE(typedView);
    auto transfer = device.queryTextureTransferLayout(
        *typedTexture, {.textureSubresource = {.aspect = rachis::TextureAspect::color}, .extent = {1, 1, 1}});
    REQUIRE(transfer);
    auto typedReadback = device.createBuffer({
        .size = transfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan typed int64 atomic readback",
    });
    REQUIRE(typedReadback);
    const std::array typedEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageTexture, rachis::ShaderStage::compute},
    };
    auto typedLayout = device.createBindingLayout({.entries = typedEntries});
    REQUIRE(typedLayout);
    const std::array typedWrites{rachis::BindingWrite{.binding = 0, .resource = *typedView}};
    auto typedSet = rachis::test::createBindingSet(device, {.layout = *typedLayout, .writes = typedWrites});
    REQUIRE(typedSet);
    const std::array typedLayouts{*typedLayout};
    auto typedPipelineLayout = device.createPipelineLayout({.bindingLayouts = typedLayouts});
    REQUIRE(typedPipelineLayout);
    auto typedPipeline = device.createComputePipeline({.layout = *typedPipelineLayout, .computeShader = *typedShader});
    REQUIRE(typedPipeline);

    auto typedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(typedCommand);
    REQUIRE(device
                .textureBarrier(*typedCommand,
                                {
                                    .texture = *typedTexture,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    REQUIRE(device.bindComputePipeline(*typedCommand, *typedPipeline));
    REQUIRE(device.bindComputeBindingSet(*typedCommand, 0, *typedSet));
    REQUIRE(device.dispatch(*typedCommand, 1));
    REQUIRE(device
                .textureBarrier(*typedCommand,
                                {
                                    .texture = *typedTexture,
                                    .stagesBefore = rachis::PipelineStage::computeShader,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::unorderedAccess,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion copy{
        .bytesPerRow = transfer->bytesPerRow,
        .rowsPerImage = transfer->rowsPerImage,
        .textureSubresource = {.aspect = rachis::TextureAspect::color},
        .extent = {1, 1, 1},
    };
    REQUIRE(device.copyTextureToBuffer(*typedCommand, *typedTexture, *typedReadback, std::span{&copy, 1}));
    auto typedSubmitted = device.submit(std::move(*typedCommand));
    REQUIRE(typedSubmitted);
    REQUIRE(device.wait(*typedSubmitted));
    auto typedMapped = device.mapBuffer(*typedReadback);
    REQUIRE(typedMapped);
    std::uint64_t typedResult = 0;
    std::memcpy(&typedResult, typedMapped->data, sizeof(typedResult));
    CHECK(typedResult == atomicSum);
    REQUIRE(device.unmapBuffer(*typedReadback));

    REQUIRE(device.destroyPipeline(*typedPipeline));
    REQUIRE(device.destroyPipelineLayout(*typedPipelineLayout));
    REQUIRE(device.destroyBindingSet(*typedSet));
    REQUIRE(device.destroyBindingLayout(*typedLayout));
    REQUIRE(device.destroyTextureView(*typedView));
    REQUIRE(device.destroyTexture(*typedTexture));
    REQUIRE(device.destroyBuffer(*typedReadback));
    REQUIRE(device.destroyShaderModule(*typedShader));
    REQUIRE(device.destroyPipeline(*groupPipeline));
    REQUIRE(device.destroyPipelineLayout(*groupPipelineLayout));
    REQUIRE(device.destroyBindingSet(*groupSet));
    REQUIRE(device.destroyBindingLayout(*groupLayout));
    REQUIRE(device.destroyShaderModule(*groupShader));
    REQUIRE(device.destroyBuffer(*groupOutput));
    REQUIRE(device.destroyBuffer(*groupReadback));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan storage-buffer 64-bit atomics keep the unsigned maximum and minimum of every compute lane",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][shader][atomic][compute]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);

    const auto bytecode = rachis_shaders::ShaderInt64AtomicStorageBuffer(rachis::Backend::vulkan);
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Vulkan storage-buffer int64 atomics",
        .shaderInterface = rachis::test::computeShaderInterface(
            {storageAtomicGroupSize, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
    });
    if (!capabilities.shaderInt64AtomicStorageBuffer)
    {
        REQUIRE_FALSE(shader);
        CHECK(shader.error().code == rachis::ErrorCode::unsupported);
        SKIP("This Vulkan device does not advertise shaderInt64AtomicStorageBuffer.");
    }
    if (!shader)
    {
        UNSCOPED_INFO(shader.error().message);
    }
    REQUIRE(shader);
    CHECK(capabilities.shaderInt64);

    constexpr std::array<std::uint64_t, 2> initial{0, std::numeric_limits<std::uint64_t>::max()};
    constexpr std::uint64_t extremesBytes = sizeof(initial);
    auto extremes = device.createBuffer({
        .size = extremesBytes,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Vulkan storage-buffer atomic extremes",
    });
    auto upload = device.createBuffer({
        .size = extremesBytes,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan storage-buffer atomic upload",
    });
    auto readback = device.createBuffer({
        .size = extremesBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan storage-buffer atomic readback",
    });
    REQUIRE(extremes);
    REQUIRE(upload);
    REQUIRE(readback);
    REQUIRE(device.writeBuffer(*upload, 0, std::as_bytes(std::span(initial))));

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout);
    const std::array writes{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*extremes, 0, extremesBytes, sizeof(std::uint64_t)}},
    };
    auto set = rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes});
    REQUIRE(set);
    const std::array layouts{*layout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout);
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.copyBuffer(*command, {*upload, 0, *extremes, 0, extremesBytes}));
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *extremes,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline));
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set));
    REQUIRE(device.dispatch(*command, storageAtomicGroupCount));
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *extremes,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {*extremes, 0, *readback, 0, extremesBytes}));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    std::array<std::uint64_t, 2> result{};
    std::memcpy(result.data(), mapped->data, extremesBytes);
    CHECK(result == expectedStorageAtomicExtremes());
    REQUIRE(device.unmapBuffer(*readback));

    REQUIRE(device.destroyPipeline(*pipeline));
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
    REQUIRE(device.destroyBindingSet(*set));
    REQUIRE(device.destroyBindingLayout(*layout));
    REQUIRE(device.destroyShaderModule(*shader));
    REQUIRE(device.destroyBuffer(*extremes));
    REQUIRE(device.destroyBuffer(*upload));
    REQUIRE(device.destroyBuffer(*readback));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan compute pipelines that require a subgroup size run full subgroups of that size",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][shader][subgroup][compute]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    const auto subgroup = capabilities.subgroup;
    if (!rachis::hasAny(subgroup.requiredSizeStages, rachis::ShaderStage::compute))
    {
        auto rejected = runLaneCountKernel(device, subgroupSizeShader(32), 32, 32);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
        SKIP("This Vulkan device does not accept a required subgroup size for compute pipelines.");
    }
    CHECK(subgroup.requiredSizeStages == rachis::ShaderStage::compute);
    INFO("Subgroup sizes " << subgroup.minSize << ".." << subgroup.maxSize);

    std::size_t testedSizes = 0;
    for (const auto size : subgroupSizes)
    {
        CAPTURE(size);
        auto lanes = runLaneCountKernel(device, subgroupSizeShader(size), size, size);
        if (size < subgroup.minSize || size > subgroup.maxSize)
        {
            REQUIRE_FALSE(lanes.has_value());
            CHECK(lanes.error().code == rachis::ErrorCode::unsupported);
            continue;
        }
        if (!lanes)
        {
            UNSCOPED_INFO(lanes.error().message);
        }
        REQUIRE(lanes.has_value());
        ++testedSizes;
        for (const auto& lane : *lanes)
        {
            CHECK(lane == std::array<std::uint32_t, 2>{size, size});
        }
    }
    CHECK(testedSizes > 0);

    const auto anySize = std::min(subgroup.maxSize, anySizeGroupWidth);
    auto anyShader = runLaneCountKernel(
        device, rachis_shaders::SubgroupSizeAny(rachis::Backend::vulkan), anySize, anySizeGroupWidth);
    if (!anyShader)
    {
        UNSCOPED_INFO(anyShader.error().message);
    }
    REQUIRE(anyShader.has_value());
    for (const auto& lane : *anyShader)
    {
        CHECK(lane == std::array<std::uint32_t, 2>{anySize, anySize});
    }
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan subgroup operations and specialization constants produce exact GPU results",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][shader][subgroup][specialization][compute]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto subgroup = rachis::test::requireCapabilities(device).subgroup;
    REQUIRE(rachis::test::requireCapabilities(device).subgroupOperations);
    REQUIRE(subgroup.minSize > 0);

    const auto subgroupBytecode = loadSpirv("SubgroupOperations.spv");
    const auto specBytecode = loadSpirv("VulkanSpecialization.spv");
    REQUIRE_FALSE(subgroupBytecode.empty());
    REQUIRE_FALSE(specBytecode.empty());

    auto subgroupShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = subgroupBytecode,
        .entryPoint = "main",
        .label = "Vulkan subgroup shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
    });
    REQUIRE(subgroupShader);

    constexpr std::uint32_t invocationCount = 64;
    constexpr std::uint32_t valuesPerInvocation = 4;
    constexpr std::uint64_t subgroupBytes =
        static_cast<std::uint64_t>(invocationCount) * valuesPerInvocation * sizeof(std::uint32_t);
    auto subgroupOutput = device.createBuffer({
        .size = subgroupBytes,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Vulkan subgroup output",
    });
    auto subgroupReadback = device.createBuffer({
        .size = subgroupBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan subgroup readback",
    });
    REQUIRE(subgroupOutput);
    REQUIRE(subgroupReadback);
    const std::array subgroupEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto subgroupLayout = device.createBindingLayout({.entries = subgroupEntries});
    REQUIRE(subgroupLayout);
    const std::array subgroupWrites{
        rachis::BindingWrite{
            0,
            0,
            rachis::BufferBinding{*subgroupOutput, 0, subgroupBytes, valuesPerInvocation * sizeof(std::uint32_t)}},
    };
    auto subgroupSet = rachis::test::createBindingSet(device, {.layout = *subgroupLayout, .writes = subgroupWrites});
    REQUIRE(subgroupSet);
    const std::array subgroupLayouts{*subgroupLayout};
    auto subgroupPipelineLayout = device.createPipelineLayout({.bindingLayouts = subgroupLayouts});
    REQUIRE(subgroupPipelineLayout);
    auto subgroupPipeline =
        device.createComputePipeline({.layout = *subgroupPipelineLayout, .computeShader = *subgroupShader});
    REQUIRE(subgroupPipeline);

    auto subgroupCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(subgroupCommand);
    REQUIRE(device
                .bufferBarrier(*subgroupCommand,
                               {
                                   .buffer = *subgroupOutput,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*subgroupCommand, *subgroupPipeline));
    REQUIRE(device.bindComputeBindingSet(*subgroupCommand, 0, *subgroupSet));
    REQUIRE(device.dispatch(*subgroupCommand, 1));
    REQUIRE(device
                .bufferBarrier(*subgroupCommand,
                               {
                                   .buffer = *subgroupOutput,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*subgroupCommand, {*subgroupOutput, 0, *subgroupReadback, 0, subgroupBytes}));
    auto subgroupSubmitted = device.submit(std::move(*subgroupCommand));
    REQUIRE(subgroupSubmitted);
    REQUIRE(device.wait(*subgroupSubmitted));
    auto subgroupMapped = device.mapBuffer(*subgroupReadback);
    REQUIRE(subgroupMapped);
    for (std::uint32_t invocation = 0; invocation < invocationCount; ++invocation)
    {
        std::array<std::uint32_t, 4> lane{};
        std::memcpy(lane.data(),
                    subgroupMapped->data + std::size_t{invocation} * valuesPerInvocation * sizeof(std::uint32_t),
                    sizeof(lane));
        CAPTURE(invocation, lane[0], lane[1], lane[2], lane[3]);
        CHECK(lane[0] >= subgroup.minSize);
        CHECK(lane[0] <= subgroup.maxSize);
        CHECK(lane[1] < lane[0]);
        CHECK(lane[2] > 0);
        CHECK(lane[2] <= lane[0]);
        CHECK(lane[3] == lane[2] * (lane[2] - 1) / 2);
    }
    REQUIRE(device.unmapBuffer(*subgroupReadback));

    const auto spec13 = rachis::SpecializationConstant::makeUnsignedInt32(7, 13);
    std::array specValues{spec13};
    const std::array specEntries{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = specBytecode,
            .entryPoint = "main",
            .specializationConstants = specValues,
        },
    };
    auto library = device.createShaderLibrary({.entries = specEntries, .label = "Vulkan spec library"});
    REQUIRE(library);
    std::array specBinding{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto specShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .entryPoint = "main",
        .library = *library,
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = specBinding,
                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
            },
        .specializationConstants = specValues,
    });
    REQUIRE(specShader);
    auto specOutput = device.createBuffer({
        .size = 4,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Vulkan specialization output",
    });
    auto specReadback = device.createBuffer({
        .size = 4,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan specialization readback",
    });
    REQUIRE(specOutput);
    REQUIRE(specReadback);
    const std::array specLayoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto specLayout = device.createBindingLayout({.entries = specLayoutEntries});
    REQUIRE(specLayout);
    const std::array specWrites{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*specOutput, 0, 4, 4}},
    };
    auto specSet = rachis::test::createBindingSet(device, {.layout = *specLayout, .writes = specWrites});
    REQUIRE(specSet);
    const std::array specLayouts{*specLayout};
    auto specPipelineLayout = device.createPipelineLayout({.bindingLayouts = specLayouts});
    REQUIRE(specPipelineLayout);
    auto specPipeline = device.createComputePipeline({.layout = *specPipelineLayout, .computeShader = *specShader});
    REQUIRE(specPipeline);
    auto specCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(specCommand);
    REQUIRE(device
                .bufferBarrier(*specCommand,
                               {
                                   .buffer = *specOutput,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*specCommand, *specPipeline));
    REQUIRE(device.bindComputeBindingSet(*specCommand, 0, *specSet));
    REQUIRE(device.dispatch(*specCommand, 1));
    REQUIRE(device
                .bufferBarrier(*specCommand,
                               {
                                   .buffer = *specOutput,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*specCommand, {*specOutput, 0, *specReadback, 0, 4}));
    auto specSubmitted = device.submit(std::move(*specCommand));
    REQUIRE(specSubmitted);
    REQUIRE(device.wait(*specSubmitted));
    auto specMapped = device.mapBuffer(*specReadback);
    REQUIRE(specMapped);
    std::uint32_t specResult = 0;
    std::memcpy(&specResult, specMapped->data, sizeof(specResult));
    CHECK(specResult == 13);
    REQUIRE(device.unmapBuffer(*specReadback));

    REQUIRE(device.destroyPipeline(*specPipeline));
    REQUIRE(device.destroyPipelineLayout(*specPipelineLayout));
    REQUIRE(device.destroyBindingSet(*specSet));
    REQUIRE(device.destroyBindingLayout(*specLayout));
    REQUIRE(device.destroyShaderModule(*specShader));
    REQUIRE(device.destroyShaderLibrary(*library));
    REQUIRE(device.destroyBuffer(*specOutput));
    REQUIRE(device.destroyBuffer(*specReadback));
    REQUIRE(device.destroyPipeline(*subgroupPipeline));
    REQUIRE(device.destroyPipelineLayout(*subgroupPipelineLayout));
    REQUIRE(device.destroyBindingSet(*subgroupSet));
    REQUIRE(device.destroyBindingLayout(*subgroupLayout));
    REQUIRE(device.destroyShaderModule(*subgroupShader));
    REQUIRE(device.destroyBuffer(*subgroupOutput));
    REQUIRE(device.destroyBuffer(*subgroupReadback));
    CHECK(device.diagnosticCounts().errors == 0);
}
#endif
