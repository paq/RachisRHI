#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint64_t bindingRangeSize = 256;
    constexpr std::uint64_t bufferSize = 2 * bindingRangeSize;
    constexpr std::uint32_t valueCount = 64;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

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

    void writeConstants(rachis::Device& device, rachis::BufferHandle buffer)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        const std::array first{3U, 7U};
        const std::array second{5U, 11U};
        std::memcpy(mapped->data, first.data(), sizeof(first));
        std::memcpy(mapped->data + bindingRangeSize, second.data(), sizeof(second));
        REQUIRE(device.unmapBuffer(buffer).has_value());
    }

    void verifyRange(const std::byte* values, std::uint32_t range, std::uint32_t multiplier, std::uint32_t addend)
    {
        std::uint32_t firstMismatch = valueCount;
        for (std::uint32_t index = 0; index < valueCount; ++index)
        {
            std::uint32_t value = 0;
            const auto byteOffset = static_cast<std::uint64_t>(range) * bindingRangeSize + index * sizeof(value);
            std::memcpy(&value, values + byteOffset, sizeof(value));
            if (value != index * multiplier + addend)
            {
                firstMismatch = index;
                break;
            }
        }
        CAPTURE(range, firstMismatch);
        CHECK(firstMismatch == valueCount);
    }

    constexpr std::uint64_t counterSlotSize = 256;
    constexpr std::uint32_t counterTextureCount = 15;
    constexpr std::uint32_t repeatedOffsetCount = 4096;

    [[nodiscard]] rachis::Device createWarpDevice()
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
            .diagnosticCallback = &printDiagnostic,
        });
        if (!created)
        {
            INFO(created.error().message);
        }
        REQUIRE(created.has_value());
        return std::move(*created);
    }

    struct CounterFixture final
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
        std::uint32_t slotCount = 0;
        std::uint32_t dynamicBufferCount = 0;
    };

    [[nodiscard]] std::vector<rachis::BindingLayoutEntry> counterLayoutEntries(std::uint32_t dynamicBufferCount,
                                                                               std::uint32_t textureCount)
    {
        std::vector<rachis::BindingLayoutEntry> entries;
        for (std::uint32_t binding = 0; binding < dynamicBufferCount; ++binding)
        {
            entries.push_back({binding, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute, 1});
        }
        if (textureCount != 0)
        {
            entries.push_back(
                {dynamicBufferCount, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, textureCount});
        }
        return entries;
    }

    void createCounterBindings(rachis::Device& device, CounterFixture& fixture, std::uint32_t textureCount)
    {
        const auto entries = counterLayoutEntries(fixture.dynamicBufferCount, textureCount);
        auto layout = device.createBindingLayout({.entries = entries, .label = "Dynamic counter layout"});
        REQUIRE(layout.has_value());
        fixture.layout = *layout;

        std::vector<rachis::BindingPoolSize> sizes{
            {rachis::BindingType::dynamicStorageBuffer, fixture.dynamicBufferCount},
        };
        std::vector<rachis::BindingWrite> writes;
        for (std::uint32_t binding = 0; binding < fixture.dynamicBufferCount; ++binding)
        {
            writes.push_back(rachis::BindingWrite{
                .binding = binding,
                .resource = rachis::BufferBinding{.buffer = fixture.counters, .size = counterSlotSize},
            });
        }
        if (textureCount != 0)
        {
            sizes.push_back({rachis::BindingType::sampledTexture, textureCount});
            for (std::uint32_t element = 0; element < textureCount; ++element)
            {
                writes.push_back(rachis::BindingWrite{
                    .binding = fixture.dynamicBufferCount,
                    .arrayElement = element,
                    .resource = fixture.view,
                });
            }
        }
        auto pool = device.createBindingPool({.maxBindingSets = 1, .sizes = sizes, .label = "Dynamic counter pool"});
        REQUIRE(pool.has_value());
        fixture.pool = *pool;
        auto set = device.createBindingSet({
            .layout = fixture.layout,
            .writes = writes,
            .label = "Dynamic counter set",
            .pool = fixture.pool,
        });
        REQUIRE(set.has_value());
        fixture.set = *set;
    }

    void createCounterProbes(rachis::Device& device, CounterFixture& fixture)
    {
        const std::array probeEntries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
        };
        auto probeLayout = device.createBindingLayout({.entries = probeEntries, .label = "Unrelated probe layout"});
        REQUIRE(probeLayout.has_value());
        fixture.probeLayout = *probeLayout;
        const std::array probeSizes{rachis::BindingPoolSize{rachis::BindingType::sampledTexture, 1}};
        auto probePool =
            device.createBindingPool({.maxBindingSets = 1, .sizes = probeSizes, .label = "Unrelated probe pool"});
        REQUIRE(probePool.has_value());
        fixture.probePool = *probePool;
        auto mipTexture = device.createTexture({
            .dimension = rachis::TextureDimension::texture2D,
            .width = 32,
            .height = 32,
            .mipLevels = 6,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::copySource |
                     rachis::TextureUsage::copyDestination,
            .label = "Unrelated mipmap texture",
        });
        REQUIRE(mipTexture.has_value());
        fixture.mipTexture = *mipTexture;
    }

    [[nodiscard]] CounterFixture createCounterFixture(rachis::Device& device,
                                                      std::uint32_t slotCount,
                                                      std::uint32_t dynamicBufferCount,
                                                      std::uint32_t textureCount)
    {
        CounterFixture fixture{.slotCount = slotCount, .dynamicBufferCount = dynamicBufferCount};
        const auto bytecode = loadShader(RACHIS_TEST_DYNAMIC_BINDING_COUNTER_SHADER_PATH);
        REQUIRE_FALSE(bytecode.empty());
        const std::array requirements{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
        };
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .label = "Dynamic counter shader",
            .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements),
        });
        REQUIRE(shader.has_value());
        fixture.shader = *shader;

        const auto counterBufferSize = counterSlotSize * slotCount;
        auto counters = device.createBuffer(rachis::BufferDesc{
            .size = counterBufferSize,
            .usage =
                rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
            .label = "Dynamic counters",
        });
        auto zeros = device.createBuffer(rachis::BufferDesc{
            .size = counterBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Dynamic counter zeros",
        });
        auto readback = device.createBuffer(rachis::BufferDesc{
            .size = counterBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Dynamic counter readback",
        });
        auto texture = device.createTexture(rachis::TextureDesc{
            .width = 4,
            .height = 4,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled,
            .label = "Dynamic counter texture",
        });
        REQUIRE(counters.has_value());
        REQUIRE(zeros.has_value());
        REQUIRE(readback.has_value());
        REQUIRE(texture.has_value());
        fixture.counters = *counters;
        fixture.zeros = *zeros;
        fixture.readback = *readback;
        fixture.texture = *texture;
        const std::vector<std::byte> zeroBytes(static_cast<std::size_t>(counterBufferSize), std::byte{0});
        REQUIRE(device.writeBuffer(fixture.zeros, 0, zeroBytes).has_value());
        auto view = device.createTextureView(rachis::TextureViewDesc{
            .texture = fixture.texture,
            .usage = rachis::TextureViewUsage::sampled,
            .label = "Dynamic counter texture view",
        });
        REQUIRE(view.has_value());
        fixture.view = *view;

        createCounterBindings(device, fixture, textureCount);
        const std::array layouts{fixture.layout};
        auto pipelineLayout =
            device.createPipelineLayout({.bindingLayouts = layouts, .label = "Dynamic counter pipeline layout"});
        REQUIRE(pipelineLayout.has_value());
        fixture.pipelineLayout = *pipelineLayout;
        auto pipeline = device.createComputePipeline({
            .layout = fixture.pipelineLayout,
            .computeShader = fixture.shader,
            .label = "Dynamic counter pipeline",
        });
        REQUIRE(pipeline.has_value());
        fixture.pipeline = *pipeline;
        createCounterProbes(device, fixture);
        return fixture;
    }

    void destroyCounterFixture(rachis::Device& device, const CounterFixture& fixture)
    {
        REQUIRE(device.destroyPipeline(fixture.pipeline).has_value());
        REQUIRE(device.destroyPipelineLayout(fixture.pipelineLayout).has_value());
        REQUIRE(device.destroyShaderModule(fixture.shader).has_value());
        REQUIRE(device.destroyBindingSet(fixture.set).has_value());
        REQUIRE(device.destroyBindingPool(fixture.pool).has_value());
        REQUIRE(device.destroyBindingLayout(fixture.layout).has_value());
        REQUIRE(device.destroyBindingPool(fixture.probePool).has_value());
        REQUIRE(device.destroyBindingLayout(fixture.probeLayout).has_value());
        REQUIRE(device.destroyTexture(fixture.mipTexture).has_value());
        REQUIRE(device.destroyTextureView(fixture.view).has_value());
        REQUIRE(device.destroyTexture(fixture.texture).has_value());
        REQUIRE(device.destroyBuffer(fixture.counters).has_value());
        REQUIRE(device.destroyBuffer(fixture.zeros).has_value());
        REQUIRE(device.destroyBuffer(fixture.readback).has_value());
    }

    void recordCounterPrologue(rachis::Device& device, rachis::CommandBuffer& command, const CounterFixture& fixture)
    {
        REQUIRE(device
                    .copyBuffer(command,
                                {
                                    .source = fixture.zeros,
                                    .destination = fixture.counters,
                                    .size = counterSlotSize * fixture.slotCount,
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

    void recordCounterEpilogue(rachis::Device& device, rachis::CommandBuffer& command, const CounterFixture& fixture)
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
                                    .size = counterSlotSize * fixture.slotCount,
                                })
                    .has_value());
    }

    struct BindFailure final
    {
        std::uint32_t index = 0;
        rachis::ErrorCode code = rachis::ErrorCode::invalidArgument;
        std::string message;
    };

    [[nodiscard]] std::optional<BindFailure> recordCounterBinds(rachis::Device& device,
                                                                rachis::CommandBuffer& command,
                                                                const CounterFixture& fixture,
                                                                std::uint32_t firstIndex,
                                                                std::uint32_t endIndex,
                                                                std::uint32_t offsetPeriod)
    {
        std::vector<std::uint32_t> offsets(fixture.dynamicBufferCount);
        for (std::uint32_t index = firstIndex; index < endIndex; ++index)
        {
            std::ranges::fill(offsets, static_cast<std::uint32_t>((index % offsetPeriod) * counterSlotSize));
            auto bound = device.bindComputeBindingSet(command, 0, fixture.set, offsets);
            if (!bound)
            {
                return BindFailure{.index = index, .code = bound.error().code, .message = bound.error().message};
            }
            auto dispatched = device.dispatch(command, 1);
            if (!dispatched)
            {
                return BindFailure{
                    .index = index,
                    .code = dispatched.error().code,
                    .message = dispatched.error().message,
                };
            }
        }
        return std::nullopt;
    }

    void requireNoBindFailure(const std::optional<BindFailure>& failure)
    {
        const auto failedIndex = failure.has_value() ? failure->index : 0U;
        const auto failedMessage = failure.has_value() ? failure->message : std::string{};
        CAPTURE(failedIndex, failedMessage);
        REQUIRE_FALSE(failure.has_value());
    }

    [[nodiscard]] std::vector<std::uint32_t> readCounters(rachis::Device& device, const CounterFixture& fixture)
    {
        std::vector<std::uint32_t> words(static_cast<std::size_t>(counterSlotSize / sizeof(std::uint32_t)) *
                                         fixture.slotCount);
        REQUIRE(device.readBuffer(fixture.readback, 0, std::as_writable_bytes(std::span{words})).has_value());
        std::vector<std::uint32_t> counts(fixture.slotCount);
        for (std::uint32_t slot = 0; slot < fixture.slotCount; ++slot)
        {
            counts[slot] = words[static_cast<std::size_t>(slot) * (counterSlotSize / sizeof(std::uint32_t))];
        }
        return counts;
    }

    void checkRepeatedOffsetCounters(rachis::Device& device, const CounterFixture& fixture, std::uint32_t bindCount)
    {
        const auto counts = readCounters(device, fixture);
        std::uint64_t total = 0;
        std::uint32_t firstMismatch = fixture.slotCount;
        for (std::uint32_t slot = 0; slot < fixture.slotCount; ++slot)
        {
            total += counts[slot];
            const auto expected = bindCount / repeatedOffsetCount + (slot < bindCount % repeatedOffsetCount ? 1U : 0U);
            if (firstMismatch == fixture.slotCount && counts[slot] != expected)
            {
                firstMismatch = slot;
            }
        }
        CAPTURE(firstMismatch);
        CHECK(total == bindCount);
        CHECK(firstMismatch == fixture.slotCount);
    }

    void checkOtherDescriptorUsers(rachis::Device& device, const CounterFixture& fixture)
    {
        const std::array probeWrites{rachis::BindingWrite{.binding = 0, .resource = fixture.view}};
        auto probeSet = device.createBindingSet({
            .layout = fixture.probeLayout,
            .writes = probeWrites,
            .label = "Unrelated probe set",
            .pool = fixture.probePool,
        });
        const std::string setFailure = probeSet ? std::string{} : probeSet.error().message;
        CAPTURE(setFailure);
        REQUIRE(probeSet.has_value());
        REQUIRE(device.updateBindingSet(*probeSet, probeWrites).has_value());
        REQUIRE(device.destroyBindingSet(*probeSet).has_value());

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

    void checkCleanDiagnostics(const rachis::Device& device)
    {
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

} // namespace

TEST_CASE("dynamic buffer offsets preserve immutable ranges and submitted descriptor lifetime",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto bytecode = loadShader(RACHIS_TEST_DYNAMIC_BINDING_SHADER_PATH);
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Dynamic binding shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    auto constants = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Dynamic constants",
    });
    auto storage = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Dynamic storage",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Dynamic readback",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = 1,
        .height = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Dynamic set retained texture",
    });
    REQUIRE(shader.has_value());
    REQUIRE(constants.has_value());
    REQUIRE(storage.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());
    auto textureView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "Dynamic set retained texture view",
    });
    REQUIRE(textureView.has_value());
    writeConstants(device, *constants);

    const std::array entries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries, .label = "Dynamic binding layout"});
    REQUIRE(bindingLayout.has_value());

    std::array<rachis::BindingLayoutEntry, rachis::maxDynamicBufferOffsetsPerBindingSet + 1> tooManyEntries{};
    for (std::uint32_t index = 0; index < tooManyEntries.size(); ++index)
    {
        tooManyEntries[index] = {index, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute, 1};
    }
    CHECK_FALSE(device.createBindingLayout({.entries = tooManyEntries}).has_value());

    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *constants, .size = bindingRangeSize},
        },
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *storage,
                    .size = bindingRangeSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
        rachis::BindingWrite{
            .binding = 2,
            .resource = *textureView,
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Dynamic binding set",
                                                     });
    REQUIRE(bindingSet.has_value());

    const std::array layouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = layouts,
        .label = "Dynamic binding pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Dynamic binding pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
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
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());

    const std::array secondOffsets{256U, 256U};
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet, secondOffsets).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());

    const std::array firstOffsets{0U, 0U};
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet, firstOffsets).has_value());
    CHECK_FALSE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    CHECK_FALSE(device.bindComputeBindingSet(*command, 0, *bindingSet, std::span{firstOffsets}.first(1)).has_value());
    const std::array misalignedUniform{4U, 0U};
    CHECK_FALSE(device.bindComputeBindingSet(*command, 0, *bindingSet, misalignedUniform).has_value());
    const std::array misalignedStorage{0U, 2U};
    CHECK_FALSE(device.bindComputeBindingSet(*command, 0, *bindingSet, misalignedStorage).has_value());
    const std::array outOfBounds{512U, 0U};
    CHECK_FALSE(device.bindComputeBindingSet(*command, 0, *bindingSet, outOfBounds).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());

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
    REQUIRE(
        device.copyBuffer(*command, {.source = *storage, .destination = *readback, .size = bufferSize}).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*constants).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    REQUIRE(device.destroyTextureView(*textureView).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    verifyRange(mapped->data, 0, 3, 7);
    verifyRange(mapped->data, 1, 5, 11);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("binding an unpromoted dynamic buffer 70,000 times keeps per-bind descriptors bounded and isolated",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    const auto fixture = createCounterFixture(device, repeatedOffsetCount, 1, counterTextureCount);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    recordCounterPrologue(device, *command, fixture);
    const auto beforeBinds = device.captureBackendAllocationSnapshot();

    constexpr std::uint32_t warmBindCount = 3 * repeatedOffsetCount;
    constexpr std::uint32_t bindCount = 70'000;
    requireNoBindFailure(recordCounterBinds(device, *command, fixture, 0, warmBindCount, repeatedOffsetCount));
    const auto afterWarmBinds = device.captureBackendAllocationSnapshot();
    requireNoBindFailure(recordCounterBinds(device, *command, fixture, warmBindCount, bindCount, repeatedOffsetCount));
    const auto afterBinds = device.captureBackendAllocationSnapshot();
    CHECK(afterBinds.liveDescriptorCount == afterWarmBinds.liveDescriptorCount);
    CHECK(afterBinds.liveDescriptorCount - beforeBinds.liveDescriptorCount <= 2ULL * repeatedOffsetCount);

    checkOtherDescriptorUsers(device, fixture);

    recordCounterEpilogue(device, *command, fixture);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    checkRepeatedOffsetCounters(device, fixture, bindCount);
    device.collectGarbage();
    CHECK(device.captureBackendAllocationSnapshot().liveDescriptorCount == beforeBinds.liveDescriptorCount);

    destroyCounterFixture(device, fixture);
    checkCleanDiagnostics(device);
}

TEST_CASE("command buffers recorded before any submit keep their own dynamic binding descriptors",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    const auto fixture = createCounterFixture(device, repeatedOffsetCount, 1, counterTextureCount);

    constexpr std::uint32_t commandBufferCount = 3;
    constexpr std::uint32_t bindsPerCommandBuffer = 1'500;
    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = commandBufferCount,
        .label = "Deferred dynamic binding command pool",
    });
    REQUIRE(pool.has_value());
    const auto beforeBinds = device.captureBackendAllocationSnapshot();

    std::vector<rachis::CommandBuffer> commands;
    for (std::uint32_t commandIndex = 0; commandIndex < commandBufferCount; ++commandIndex)
    {
        auto command = device.acquireCommandBuffer(*pool);
        REQUIRE(command.has_value());
        if (commandIndex == 0)
        {
            recordCounterPrologue(device, *command, fixture);
        }
        else
        {
            REQUIRE(device.bindComputePipeline(*command, fixture.pipeline).has_value());
        }
        CAPTURE(commandIndex);
        requireNoBindFailure(recordCounterBinds(device,
                                                *command,
                                                fixture,
                                                commandIndex * bindsPerCommandBuffer,
                                                (commandIndex + 1) * bindsPerCommandBuffer,
                                                repeatedOffsetCount));
        commands.push_back(std::move(*command));
    }
    const auto afterBinds = device.captureBackendAllocationSnapshot();
    CHECK(afterBinds.liveDescriptorCount - beforeBinds.liveDescriptorCount <=
          2ULL * commandBufferCount * bindsPerCommandBuffer);

    checkOtherDescriptorUsers(device, fixture);

    recordCounterEpilogue(device, commands.back(), fixture);
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
    checkRepeatedOffsetCounters(device, fixture, commandBufferCount * bindsPerCommandBuffer);
    device.collectGarbage();
    CHECK(device.captureBackendAllocationSnapshot().liveDescriptorCount == beforeBinds.liveDescriptorCount);

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    destroyCounterFixture(device, fixture);
    checkCleanDiagnostics(device);
}

TEST_CASE("command buffers that each bind one small dynamic table hold only that table's descriptors",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    constexpr std::uint32_t commandBufferCount = 512;
    const auto fixture = createCounterFixture(device, commandBufferCount, 1, 0);

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = commandBufferCount,
        .label = "Single-bind dynamic binding command pool",
    });
    REQUIRE(pool.has_value());
    const auto beforeBinds = device.captureBackendAllocationSnapshot();

    std::vector<rachis::CommandBuffer> commands;
    commands.reserve(commandBufferCount);
    for (std::uint32_t commandIndex = 0; commandIndex < commandBufferCount; ++commandIndex)
    {
        auto command = device.acquireCommandBuffer(*pool);
        REQUIRE(command.has_value());
        if (commandIndex == 0)
        {
            recordCounterPrologue(device, *command, fixture);
        }
        else
        {
            REQUIRE(device.bindComputePipeline(*command, fixture.pipeline).has_value());
        }
        CAPTURE(commandIndex);
        requireNoBindFailure(
            recordCounterBinds(device, *command, fixture, commandIndex, commandIndex + 1, commandBufferCount));
        commands.push_back(std::move(*command));
    }
    const auto afterBinds = device.captureBackendAllocationSnapshot();
    CHECK(afterBinds.liveDescriptorCount - beforeBinds.liveDescriptorCount == commandBufferCount);

    checkOtherDescriptorUsers(device, fixture);

    recordCounterEpilogue(device, commands.back(), fixture);
    std::vector<rachis::TimelinePoint> submittedPoints;
    submittedPoints.reserve(commandBufferCount);
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
    const auto counts = readCounters(device, fixture);
    CHECK(std::ranges::count(counts, 1U) == static_cast<std::ptrdiff_t>(commandBufferCount));
    device.collectGarbage();
    CHECK(device.captureBackendAllocationSnapshot().liveDescriptorCount == beforeBinds.liveDescriptorCount);

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    destroyCounterFixture(device, fixture);
    checkCleanDiagnostics(device);
}

TEST_CASE("an exhausted dynamic binding descriptor range fails only binds that need new descriptors",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    constexpr std::uint32_t distinctOffsetCount = 16'384;
    const auto fixture =
        createCounterFixture(device, distinctOffsetCount, rachis::maxDynamicBufferOffsetsPerBindingSet, 0);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    recordCounterPrologue(device, *command, fixture);
    const auto exhausted = recordCounterBinds(device, *command, fixture, 0, distinctOffsetCount, distinctOffsetCount);
    REQUIRE(exhausted.has_value());
    const auto exhaustedIndex = exhausted->index;
    CAPTURE(exhaustedIndex, exhausted->message);
    CHECK(exhausted->code == rachis::ErrorCode::resourceExhausted);
    CHECK(exhaustedIndex >= 1'024U);

    requireNoBindFailure(recordCounterBinds(device, *command, fixture, 0, 1, distinctOffsetCount));
    const auto stillExhausted =
        recordCounterBinds(device, *command, fixture, exhaustedIndex, exhaustedIndex + 1, distinctOffsetCount);
    REQUIRE(stillExhausted.has_value());
    CHECK(stillExhausted->code == rachis::ErrorCode::resourceExhausted);
    checkOtherDescriptorUsers(device, fixture);

    recordCounterEpilogue(device, *command, fixture);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    const auto counts = readCounters(device, fixture);
    std::uint32_t firstMismatch = distinctOffsetCount;
    for (std::uint32_t slot = 0; slot < distinctOffsetCount && firstMismatch == distinctOffsetCount; ++slot)
    {
        const auto expected = slot == 0 ? 2U : (slot < exhaustedIndex ? 1U : 0U);
        if (counts[slot] != expected)
        {
            firstMismatch = slot;
        }
    }
    CHECK(firstMismatch == distinctOffsetCount);
    device.collectGarbage();

    auto retired = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(retired.has_value());
    REQUIRE(device.bindComputePipeline(*retired, fixture.pipeline).has_value());
    requireNoBindFailure(
        recordCounterBinds(device, *retired, fixture, exhaustedIndex, exhaustedIndex + 1, distinctOffsetCount));
    auto retiredSubmitted = device.submit(std::move(*retired));
    REQUIRE(retiredSubmitted.has_value());
    REQUIRE(device.wait(*retiredSubmitted).has_value());

    destroyCounterFixture(device, fixture);
    checkCleanDiagnostics(device);
}
