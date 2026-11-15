#include "Rachis/Device.h"

#include "GeneratedComputePush.h"
#include "GeneratedLayouts.h"
#include "GeneratedLayoutsSlang.h"
#include "GeneratedSlang.h"
#include "GeneratedVertexInput.h"
#include "GeneratedVertexInputPixel.h"
#include "RhiTestSupport.h"

#if defined(RACHIS_ENABLE_VULKAN) && RACHIS_ENABLE_VULKAN
#include "GeneratedGlslCompute.h"
#include "GeneratedGlslPixel.h"
#include "GeneratedGlslVertex.h"
#endif

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#if defined(RACHIS_ENABLE_METAL) && RACHIS_ENABLE_METAL && !(defined(RACHIS_ENABLE_D3D12) && RACHIS_ENABLE_D3D12) &&   \
    !(defined(RACHIS_ENABLE_VULKAN) && RACHIS_ENABLE_VULKAN)
#define RACHIS_GENERATED_INTERFACE_TAGS "[size:large][rhi][portable][shader-interface]"
#else
#define RACHIS_GENERATED_INTERFACE_TAGS "[size:medium][rhi][portable][shader-interface]"
#endif

namespace
{

    namespace push = rachis_shaders::reflection::GeneratedComputePush;
    namespace vertex = rachis_shaders::reflection::GeneratedVertexInput;
    namespace pixel = rachis_shaders::reflection::GeneratedVertexInputPixel;

    struct GeneratedBackend final
    {
        std::string_view name;
        rachis::DeviceDesc desc;
        bool skipWhenUnavailable = false;
    };

    [[nodiscard]] std::vector<GeneratedBackend> generatedBackends()
    {
        std::vector<GeneratedBackend> backends;
#if defined(RACHIS_ENABLE_D3D12) && RACHIS_ENABLE_D3D12
        backends.push_back({"D3D12-WARP", rachis::test::d3d12WarpDeviceDesc(), false});
#endif
#if defined(RACHIS_ENABLE_VULKAN) && RACHIS_ENABLE_VULKAN
        backends.push_back({"Vulkan", rachis::test::vulkanDeviceDesc(), true});
#endif
#if defined(RACHIS_ENABLE_METAL) && RACHIS_ENABLE_METAL
        backends.push_back({"Metal",
                            rachis::DeviceDesc{
                                .backend = rachis::Backend::metal,
                                .validation = rachis::ValidationLevel::standard,
                                .requireValidation = false,
                                .enableDred = false,
                            },
                            false});
#endif
        return backends;
    }

    struct GeneratedShader final
    {
        rachis::ShaderModuleDesc (*moduleDesc)(rachis::Backend, std::string_view) noexcept;
        std::uint32_t (*bindingSetCount)(rachis::Backend) noexcept;
        std::span<const rachis::BindingLayoutEntry> (*bindingLayout)(rachis::Backend, std::uint32_t) noexcept;
        std::span<const rachis::BindingPoolSize> (*bindingPoolSizes)(rachis::Backend, std::uint32_t) noexcept;
        std::span<const rachis::PushConstantRange> (*pushConstantRanges)(rachis::Backend) noexcept;
    };

#define RACHIS_GENERATED_SHADER(reflection)                                                                            \
    GeneratedShader                                                                                                    \
    {                                                                                                                  \
        &reflection::moduleDesc, &reflection::bindingSetCount, &reflection::bindingLayout,                            \
            &reflection::bindingPoolSizes, &reflection::pushConstantRanges                                             \
    }

    template<typename CreateLayout>
    [[nodiscard]] rachis::Result<rachis::PipelineLayoutHandle> createGeneratedPipelineLayout(
        rachis::Device& device,
        std::uint32_t setCount,
        CreateLayout&& entriesOf,
        std::span<const rachis::PushConstantRange> pushConstantRanges,
        std::vector<rachis::BindingLayoutHandle>& layouts)
    {
        for (std::uint32_t set = 0; set < setCount; ++set)
        {
            auto layout = device.createBindingLayout({.entries = entriesOf(set)});
            if (!layout)
            {
                return std::unexpected(std::move(layout.error()));
            }
            layouts.push_back(*layout);
        }
        return device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = pushConstantRanges});
    }

} // namespace

TEST_CASE("rachis_add_shader interfaces create modules and pipelines on every backend", RACHIS_GENERATED_INTERFACE_TAGS)
{
    for (const auto& spec : generatedBackends())
    {
        CAPTURE(spec.name);
        auto created = rachis::createDevice(spec.desc);
        if (!created && spec.skipWhenUnavailable && rachis::test::skipWithoutCreatedDevice(created))
        {
            continue;
        }
        REQUIRE(created.has_value());
        auto device = std::move(*created);
        const auto backend = spec.desc.backend;

        const auto computeInterface = push::shaderInterface(backend);
        REQUIRE(computeInterface.bindings.size() == 1);
        CHECK(computeInterface.bindings[0].type == rachis::ShaderResourceType::storageBuffer);
        REQUIRE(computeInterface.pushConstants.size() == 1);
        CHECK(computeInterface.pushConstants[0].size == 16);
        REQUIRE(computeInterface.workgroupSize.has_value());
        CHECK(computeInterface.workgroupSize->x == 64);
        REQUIRE(push::bindingPoolSizes(backend, 0).size() == 1);
        CHECK(push::bindingPoolSizes(backend, 0)[0].descriptorCount == 1);

        auto computeModule = device.createShaderModule(push::moduleDesc(backend, "Generated compute push"));
        REQUIRE(computeModule.has_value());
        std::vector<rachis::BindingLayoutHandle> computeLayouts;
        auto computeLayout = createGeneratedPipelineLayout(
            device,
            push::bindingSetCount(backend),
            [&](std::uint32_t set) { return push::bindingLayout(backend, set); },
            push::pushConstantRanges(backend),
            computeLayouts);
        REQUIRE(computeLayout.has_value());
        auto computePipeline =
            device.createComputePipeline({.layout = *computeLayout, .computeShader = *computeModule});
        REQUIRE(computePipeline.has_value());

        const auto vertexInterface = vertex::shaderInterface(backend);
        CHECK(std::vector(vertexInterface.vertexInputLocations.begin(), vertexInterface.vertexInputLocations.end()) ==
              std::vector<std::uint32_t>{0, 1, 2});
        CHECK(vertexInterface.bindings.empty());
        CHECK_FALSE(vertexInterface.workgroupSize.has_value());
        auto vertexModule = device.createShaderModule(vertex::moduleDesc(backend, "Generated vertex input"));
        REQUIRE(vertexModule.has_value());
        auto pixelModule = device.createShaderModule(pixel::moduleDesc(backend, "Generated vertex input pixel"));
        REQUIRE(pixelModule.has_value());
        auto graphicsLayout = device.createPipelineLayout({});
        REQUIRE(graphicsLayout.has_value());
        const std::array attributes{
            rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 0, .shaderLocation = 0},
            rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 8, .shaderLocation = 1},
            rachis::VertexAttribute{.format = rachis::VertexFormat::float32x4, .offset = 16, .shaderLocation = 2},
        };
        const std::array vertexBuffers{rachis::VertexBufferLayout{.stride = 32, .attributes = attributes}};
        const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
        auto graphicsPipeline = device.createGraphicsPipeline({
            .layout = *graphicsLayout,
            .vertexShader = *vertexModule,
            .pixelShader = *pixelModule,
            .vertexBuffers = vertexBuffers,
            .colorTargets = colorTargets,
            .label = "Generated vertex input",
        });
        REQUIRE(graphicsPipeline.has_value());

        CHECK(device.destroyPipeline(*graphicsPipeline).has_value());
        CHECK(device.destroyPipeline(*computePipeline).has_value());
        CHECK(device.destroyPipelineLayout(*graphicsLayout).has_value());
        CHECK(device.destroyPipelineLayout(*computeLayout).has_value());
        for (const auto layout : computeLayouts)
        {
            CHECK(device.destroyBindingLayout(layout).has_value());
        }
        CHECK(device.destroyShaderModule(*pixelModule).has_value());
        CHECK(device.destroyShaderModule(*vertexModule).has_value());
        CHECK(device.destroyShaderModule(*computeModule).has_value());
        const auto counts = device.diagnosticCounts();
        CHECK(counts.errors == 0);
        CHECK(counts.corruptions == 0);
    }
}

namespace
{

    namespace layouts = rachis_shaders::reflection::GeneratedLayouts;
    namespace layoutsSlang = rachis_shaders::reflection::GeneratedLayoutsSlang;

    constexpr std::uint32_t layoutOutputCount = 41;

    [[nodiscard]] std::uint32_t floatBits(float value) noexcept
    {
        return std::bit_cast<std::uint32_t>(value);
    }

    template<typename Constants, typename Element, typename Push>
    void checkGeneratedLayouts(rachis::Device& device, rachis::Backend backend, const GeneratedShader& shader)
    {
        Constants constants{};
        constants.a = 1.0F;
        constants.b = {2.0F, 3.0F, 4.0F};
        constants.c = {5.0F, 6.0F};
        constants.d = 7.0F;
        constants.e = {8.0F, 9.0F, 10.0F};
        for (std::uint32_t weight = 0; weight < 3; ++weight)
        {
            constants.weights[weight] = 11.0F + static_cast<float>(weight);
        }
        constants.after = 14.0F;
        constexpr bool hasMatrix = requires(Constants& value) { value.m; };
        if constexpr (hasMatrix)
        {
            for (std::uint32_t column = 0; column < 3; ++column)
            {
                for (std::uint32_t row = 0; row < 2; ++row)
                {
                    constants.m[column][row] = 15.0F + static_cast<float>(column * 2 + row);
                }
            }
        }
        constants.inner.ids = {21, 22};
        constants.inner.v = {23.0F, 24.0F, 25.0F};

        std::array<Element, 2> elements{};
        for (std::uint32_t element = 0; element < 2; ++element)
        {
            const auto base = 26 + element * 6;
            elements[element].position = {
                static_cast<float>(base), static_cast<float>(base + 1), static_cast<float>(base + 2)};
            elements[element].radius = static_cast<float>(base + 3);
            elements[element].ids = {base + 4, base + 5};
        }
        Push push{};
        push.first = 100;
        push.scale = {101.0F, 102.0F, 103.0F};

        std::vector<std::uint32_t> expected;
        for (std::uint32_t value = 1; value <= (hasMatrix ? 20U : 14U); ++value)
        {
            expected.push_back(floatBits(static_cast<float>(value)));
        }
        expected.insert(expected.end(), {21, 22, floatBits(23.0F), floatBits(24.0F), floatBits(25.0F)});
        for (std::uint32_t element = 0; element < 2; ++element)
        {
            const auto base = 26 + element * 6;
            for (std::uint32_t component = 0; component < 4; ++component)
            {
                expected.push_back(floatBits(static_cast<float>(base + component)));
            }
            expected.insert(expected.end(), {base + 4, base + 5});
        }
        expected.insert(expected.end(), {100, floatBits(101.0F), floatBits(102.0F), floatBits(103.0F)});
        const auto outputCount = static_cast<std::uint32_t>(expected.size());
        REQUIRE(outputCount == (hasMatrix ? layoutOutputCount : layoutOutputCount - 6));

        const auto constantBytes = std::max<std::uint64_t>(256, sizeof(Constants));
        auto constantBuffer = device.createBuffer({
            .size = constantBytes,
            .usage = rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
            .label = "Generated layout constants",
        });
        auto elementUpload = device.createBuffer({
            .size = sizeof(elements),
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Generated layout element upload",
        });
        auto elementBuffer = device.createBuffer({
            .size = sizeof(elements),
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copyDestination,
            .label = "Generated layout elements",
        });
        auto outputBuffer = device.createBuffer({
            .size = layoutOutputCount * sizeof(std::uint32_t),
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
            .label = "Generated layout output",
        });
        auto readback = device.createBuffer({
            .size = layoutOutputCount * sizeof(std::uint32_t),
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Generated layout readback",
        });
        REQUIRE(constantBuffer.has_value());
        REQUIRE(elementUpload.has_value());
        REQUIRE(elementBuffer.has_value());
        REQUIRE(outputBuffer.has_value());
        REQUIRE(readback.has_value());
        REQUIRE(device.writeBuffer(*constantBuffer, 0, std::as_bytes(std::span{&constants, 1})).has_value());
        REQUIRE(device.writeBuffer(*elementUpload, 0, std::as_bytes(std::span{elements})).has_value());

        auto module = device.createShaderModule(shader.moduleDesc(backend, "Generated layouts"));
        if (!module)
        {
            FAIL(module.error().message);
        }
        REQUIRE(shader.bindingSetCount(backend) == 1);
        auto bindingLayout = device.createBindingLayout({.entries = shader.bindingLayout(backend, 0)});
        REQUIRE(bindingLayout.has_value());
        const std::array bindingLayouts{*bindingLayout};
        REQUIRE(shader.pushConstantRanges(backend).size() == 1);
        CHECK(shader.pushConstantRanges(backend)[0].size == sizeof(Push));
        auto pipelineLayout = device.createPipelineLayout(
            {.bindingLayouts = bindingLayouts, .pushConstantRanges = shader.pushConstantRanges(backend)});
        REQUIRE(pipelineLayout.has_value());
        auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *module});
        REQUIRE(pipeline.has_value());
        auto pool = device.createBindingPool({.maxBindingSets = 1, .sizes = shader.bindingPoolSizes(backend, 0)});
        REQUIRE(pool.has_value());
        const std::array writes{
            rachis::BindingWrite{.binding = 0, .resource = rachis::BufferBinding{.buffer = *constantBuffer}},
            rachis::BindingWrite{
                .binding = 1,
                .resource = rachis::BufferBinding{.buffer = *elementBuffer, .structureByteStride = sizeof(Element)},
            },
            rachis::BindingWrite{
                .binding = 2,
                .resource =
                    rachis::BufferBinding{.buffer = *outputBuffer, .structureByteStride = sizeof(std::uint32_t)},
            },
        };
        auto bindingSet = device.createBindingSet({.layout = *bindingLayout, .writes = writes, .pool = *pool});
        REQUIRE(bindingSet.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                {.source = *elementUpload, .destination = *elementBuffer, .size = sizeof(elements)})
                    .has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *elementBuffer,
                                       .stagesBefore = rachis::PipelineStage::copy,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessBefore = rachis::Access::copyDestination,
                                       .accessAfter = rachis::Access::shaderResource,
                                   })
                    .has_value());
        REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
        REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span{&push, 1})).has_value());
        REQUIRE(device.dispatch(*command, 1).has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *outputBuffer,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                {.source = *outputBuffer,
                                 .destination = *readback,
                                 .size = layoutOutputCount * sizeof(std::uint32_t)})
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        std::vector<std::uint32_t> output(layoutOutputCount);
        REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span{output})).has_value());
        for (std::uint32_t index = 0; index < outputCount; ++index)
        {
            CAPTURE(index);
            CHECK(output[index] == expected[index]);
        }

        CHECK(device.destroyBindingSet(*bindingSet).has_value());
        CHECK(device.destroyBindingPool(*pool).has_value());
        CHECK(device.destroyPipeline(*pipeline).has_value());
        CHECK(device.destroyPipelineLayout(*pipelineLayout).has_value());
        CHECK(device.destroyBindingLayout(*bindingLayout).has_value());
        CHECK(device.destroyShaderModule(*module).has_value());
        for (const auto buffer : {*constantBuffer, *elementUpload, *elementBuffer, *outputBuffer, *readback})
        {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
    }

} // namespace

TEST_CASE("rachis_add_shader structs place every value where each backend's shader reads it",
          RACHIS_GENERATED_INTERFACE_TAGS)
{
    for (const auto& spec : generatedBackends())
    {
        CAPTURE(spec.name);
        auto created = rachis::createDevice(spec.desc);
        if (!created && spec.skipWhenUnavailable && rachis::test::skipWithoutCreatedDevice(created))
        {
            continue;
        }
        REQUIRE(created.has_value());
        auto device = std::move(*created);
        constexpr auto hlsl = RACHIS_GENERATED_SHADER(layouts);
        constexpr auto slang = RACHIS_GENERATED_SHADER(layoutsSlang);
        switch (spec.desc.backend)
        {
#if defined(RACHIS_ENABLE_D3D12) && RACHIS_ENABLE_D3D12
        case rachis::Backend::d3d12:
            checkGeneratedLayouts<layouts::d3d12::Constants, layouts::d3d12::Element, layouts::d3d12::Push>(
                device, spec.desc.backend, hlsl);
            checkGeneratedLayouts<layoutsSlang::d3d12::Constants,
                                  layoutsSlang::d3d12::Element,
                                  layoutsSlang::d3d12::Push>(device, spec.desc.backend, slang);
            break;
#endif
#if defined(RACHIS_ENABLE_VULKAN) && RACHIS_ENABLE_VULKAN
        case rachis::Backend::vulkan:
            checkGeneratedLayouts<layouts::vulkan::Constants, layouts::vulkan::Element, layouts::vulkan::Push>(
                device, spec.desc.backend, hlsl);
            checkGeneratedLayouts<layoutsSlang::vulkan::Constants,
                                  layoutsSlang::vulkan::Element,
                                  layoutsSlang::vulkan::Push>(device, spec.desc.backend, slang);
            break;
#endif
#if defined(RACHIS_ENABLE_METAL) && RACHIS_ENABLE_METAL
        case rachis::Backend::metal:
            checkGeneratedLayouts<layouts::metal::Constants, layouts::metal::Element, layouts::metal::Push>(
                device, spec.desc.backend, hlsl);
            checkGeneratedLayouts<layoutsSlang::metal::Constants,
                                  layoutsSlang::metal::Element,
                                  layoutsSlang::metal::Push>(device, spec.desc.backend, slang);
            break;
#endif
        default:
            FAIL("No generated layout for this backend.");
        }
        const auto counts = device.diagnosticCounts();
        CHECK(counts.errors == 0);
        CHECK(counts.corruptions == 0);
    }
}

namespace
{

    namespace slangShader = rachis_shaders::reflection::GeneratedSlang;

    constexpr std::uint32_t mappedElementCount = 4;
    constexpr std::uint32_t mappedOutputOffset = 1;
    constexpr std::uint32_t mappedOutputStride = 2;
    constexpr std::uint32_t mappedOutputCount = mappedOutputOffset + mappedElementCount * mappedOutputStride;

    [[nodiscard]] rachis::Result<rachis::BufferHandle> createFilledBuffer(rachis::Device& device,
                                                                          std::span<const std::byte> bytes,
                                                                          rachis::BufferUsage usage,
                                                                          std::string_view label)
    {
        auto buffer = device.createBuffer({
            .size = std::max<std::uint64_t>(256, bytes.size()),
            .usage = usage,
            .memoryType = rachis::MemoryType::upload,
            .label = label,
        });
        if (buffer)
        {
            if (auto written = device.writeBuffer(*buffer, 0, bytes); !written)
            {
                return std::unexpected(std::move(written.error()));
            }
        }
        return buffer;
    }

    template<typename Params, typename Element, typename Push>
    void checkMappedElements(rachis::Device& device, rachis::Backend backend, const GeneratedShader& shader)
    {
        Params params{};
        params.count = mappedElementCount;
        params.bias = {1.0F, 2.0F, 3.0F};
        params.scale = 2.0F;
        std::array<Element, mappedElementCount> elements{};
        for (std::uint32_t index = 0; index < mappedElementCount; ++index)
        {
            const auto base = static_cast<float>(index);
            elements[index].position = {base, base + 1.0F, base + 2.0F};
            elements[index].id = 100 + index;
        }
        Push push{};
        push.offset = mappedOutputOffset;
        push.stride = mappedOutputStride;

        auto paramsBuffer = createFilledBuffer(
            device, std::as_bytes(std::span{&params, 1}), rachis::BufferUsage::constant, "Mapped element params");
        REQUIRE(paramsBuffer.has_value());
        auto elementUpload = createFilledBuffer(
            device, std::as_bytes(std::span{elements}), rachis::BufferUsage::copySource, "Mapped element upload");
        REQUIRE(elementUpload.has_value());
        auto elementBuffer = device.createBuffer({
            .size = sizeof(elements),
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copyDestination,
            .label = "Mapped elements",
        });
        auto outputBuffer = device.createBuffer({
            .size = mappedOutputCount * sizeof(std::uint32_t),
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
            .label = "Mapped element output",
        });
        auto readback = device.createBuffer({
            .size = mappedOutputCount * sizeof(std::uint32_t),
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Mapped element readback",
        });
        REQUIRE(elementBuffer.has_value());
        REQUIRE(outputBuffer.has_value());
        REQUIRE(readback.has_value());

        auto module = device.createShaderModule(shader.moduleDesc(backend, "Mapped elements"));
        if (!module)
        {
            FAIL(module.error().message);
        }
        REQUIRE(shader.bindingSetCount(backend) == 2);
        std::vector<rachis::BindingLayoutHandle> bindingLayouts;
        auto pipelineLayout = createGeneratedPipelineLayout(
            device,
            2,
            [&](std::uint32_t set) { return shader.bindingLayout(backend, set); },
            shader.pushConstantRanges(backend),
            bindingLayouts);
        REQUIRE(pipelineLayout.has_value());
        REQUIRE(shader.pushConstantRanges(backend).size() == 1);
        CHECK(shader.pushConstantRanges(backend)[0].size == sizeof(Push));
        auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *module});
        REQUIRE(pipeline.has_value());

        auto resourcePool = device.createBindingPool({.maxBindingSets = 1, .sizes = shader.bindingPoolSizes(backend, 0)});
        auto paramsPool = device.createBindingPool({.maxBindingSets = 1, .sizes = shader.bindingPoolSizes(backend, 1)});
        REQUIRE(resourcePool.has_value());
        REQUIRE(paramsPool.has_value());
        const std::array resourceWrites{
            rachis::BindingWrite{
                .binding = 0,
                .resource = rachis::BufferBinding{.buffer = *elementBuffer, .structureByteStride = sizeof(Element)},
            },
            rachis::BindingWrite{
                .binding = 1,
                .resource =
                    rachis::BufferBinding{.buffer = *outputBuffer, .structureByteStride = sizeof(std::uint32_t)},
            },
        };
        const std::array paramsWrites{
            rachis::BindingWrite{.binding = 0, .resource = rachis::BufferBinding{.buffer = *paramsBuffer}},
        };
        auto resourceSet =
            device.createBindingSet({.layout = bindingLayouts[0], .writes = resourceWrites, .pool = *resourcePool});
        auto paramsSet =
            device.createBindingSet({.layout = bindingLayouts[1], .writes = paramsWrites, .pool = *paramsPool});
        REQUIRE(resourceSet.has_value());
        REQUIRE(paramsSet.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                {.source = *elementUpload, .destination = *elementBuffer, .size = sizeof(elements)})
                    .has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *elementBuffer,
                                       .stagesBefore = rachis::PipelineStage::copy,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessBefore = rachis::Access::copyDestination,
                                       .accessAfter = rachis::Access::shaderResource,
                                   })
                    .has_value());
        REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*command, 0, *resourceSet).has_value());
        REQUIRE(device.bindComputeBindingSet(*command, 1, *paramsSet).has_value());
        REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span{&push, 1})).has_value());
        REQUIRE(device.dispatch(*command, 1).has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *outputBuffer,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                {.source = *outputBuffer,
                                 .destination = *readback,
                                 .size = mappedOutputCount * sizeof(std::uint32_t)})
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        std::vector<std::uint32_t> output(mappedOutputCount);
        REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span{output})).has_value());
        for (std::uint32_t index = 0; index < mappedElementCount; ++index)
        {
            CAPTURE(index);
            CHECK(output[mappedOutputOffset + index * mappedOutputStride] == 7 * index + 118);
        }

        CHECK(device.destroyBindingSet(*paramsSet).has_value());
        CHECK(device.destroyBindingSet(*resourceSet).has_value());
        CHECK(device.destroyBindingPool(*paramsPool).has_value());
        CHECK(device.destroyBindingPool(*resourcePool).has_value());
        CHECK(device.destroyPipeline(*pipeline).has_value());
        CHECK(device.destroyPipelineLayout(*pipelineLayout).has_value());
        for (const auto layout : bindingLayouts)
        {
            CHECK(device.destroyBindingLayout(layout).has_value());
        }
        CHECK(device.destroyShaderModule(*module).has_value());
        for (const auto buffer : {*paramsBuffer, *elementUpload, *elementBuffer, *outputBuffer, *readback})
        {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
    }

} // namespace

TEST_CASE("rachis_add_shader compiles Slang for every backend", RACHIS_GENERATED_INTERFACE_TAGS)
{
    for (const auto& spec : generatedBackends())
    {
        CAPTURE(spec.name);
        auto created = rachis::createDevice(spec.desc);
        if (!created && spec.skipWhenUnavailable && rachis::test::skipWithoutCreatedDevice(created))
        {
            continue;
        }
        REQUIRE(created.has_value());
        auto device = std::move(*created);
        const auto backend = spec.desc.backend;

        const auto computeInterface = slangShader::shaderInterface(backend);
        REQUIRE(computeInterface.bindings.size() == 3);
        CHECK(computeInterface.bindings[0].type == rachis::ShaderResourceType::readOnlyStorageBuffer);
        CHECK(computeInterface.bindings[1].type == rachis::ShaderResourceType::storageBuffer);
        CHECK(computeInterface.bindings[2].type == rachis::ShaderResourceType::uniformBuffer);
        REQUIRE(computeInterface.workgroupSize.has_value());
        CHECK(computeInterface.workgroupSize->x == 4);

        constexpr auto shader = RACHIS_GENERATED_SHADER(slangShader);
        switch (backend)
        {
#if defined(RACHIS_ENABLE_D3D12) && RACHIS_ENABLE_D3D12
        case rachis::Backend::d3d12:
            checkMappedElements<slangShader::d3d12::Params, slangShader::d3d12::Element, slangShader::d3d12::Push>(
                device, backend, shader);
            break;
#endif
#if defined(RACHIS_ENABLE_VULKAN) && RACHIS_ENABLE_VULKAN
        case rachis::Backend::vulkan:
            checkMappedElements<slangShader::vulkan::Params, slangShader::vulkan::Element, slangShader::vulkan::Push>(
                device, backend, shader);
            break;
#endif
#if defined(RACHIS_ENABLE_METAL) && RACHIS_ENABLE_METAL
        case rachis::Backend::metal:
            checkMappedElements<slangShader::metal::Params, slangShader::metal::Element, slangShader::metal::Push>(
                device, backend, shader);
            break;
#endif
        default:
            FAIL("No generated layout for this backend.");
        }
        const auto counts = device.diagnosticCounts();
        CHECK(counts.errors == 0);
        CHECK(counts.corruptions == 0);
    }
}

#if defined(RACHIS_ENABLE_VULKAN) && RACHIS_ENABLE_VULKAN

namespace
{

    namespace glslCompute = rachis_shaders::reflection::GeneratedGlslCompute;
    namespace glslVertex = rachis_shaders::reflection::GeneratedGlslVertex;
    namespace glslPixel = rachis_shaders::reflection::GeneratedGlslPixel;

} // namespace

TEST_CASE("rachis_add_shader compiles GLSL for Vulkan only", RACHIS_GENERATED_INTERFACE_TAGS)
{
    for (const auto& spec : generatedBackends())
    {
        CAPTURE(spec.name);
        const auto backend = spec.desc.backend;
        if (backend != rachis::Backend::vulkan)
        {
            CHECK(rachis_shaders::GeneratedGlslCompute(backend).empty());
            CHECK(rachis_shaders::GeneratedGlslVertex(backend).empty());
            CHECK(glslCompute::shaderInterface(backend).bindings.empty());
            continue;
        }
        auto created = rachis::createDevice(spec.desc);
        if (!created && spec.skipWhenUnavailable && rachis::test::skipWithoutCreatedDevice(created))
        {
            continue;
        }
        REQUIRE(created.has_value());
        auto device = std::move(*created);

        CHECK(glslCompute::entryPoint == "glslMain");
        const auto computeInterface = glslCompute::shaderInterface(backend);
        REQUIRE(computeInterface.bindings.size() == 3);
        CHECK(computeInterface.bindings[0].type == rachis::ShaderResourceType::readOnlyStorageBuffer);
        CHECK(computeInterface.bindings[1].type == rachis::ShaderResourceType::storageBuffer);
        CHECK(computeInterface.bindings[2].set == 1);
        CHECK(computeInterface.bindings[2].type == rachis::ShaderResourceType::uniformBuffer);
        REQUIRE(computeInterface.workgroupSize.has_value());
        CHECK(computeInterface.workgroupSize->x == 4);
        CHECK(computeInterface.workgroupSize->y == 1);
        constexpr auto shader = RACHIS_GENERATED_SHADER(glslCompute);
        checkMappedElements<glslCompute::vulkan::Params, glslCompute::vulkan::Element, glslCompute::vulkan::Push>(
            device, backend, shader);

        CHECK(glslVertex::entryPoint == "main");
        const auto vertexInterface = glslVertex::shaderInterface(backend);
        CHECK(std::vector(vertexInterface.vertexInputLocations.begin(), vertexInterface.vertexInputLocations.end()) ==
              std::vector<std::uint32_t>{0, 1, 2});
        CHECK(vertexInterface.bindings.empty());
        auto vertexModule = device.createShaderModule(glslVertex::moduleDesc(backend, "GLSL vertex"));
        REQUIRE(vertexModule.has_value());
        auto pixelModule = device.createShaderModule(glslPixel::moduleDesc(backend, "GLSL pixel"));
        REQUIRE(pixelModule.has_value());
        auto graphicsLayout = device.createPipelineLayout({});
        REQUIRE(graphicsLayout.has_value());
        const std::array attributes{
            rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 0, .shaderLocation = 0},
            rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 8, .shaderLocation = 1},
            rachis::VertexAttribute{.format = rachis::VertexFormat::float32x4, .offset = 16, .shaderLocation = 2},
        };
        const std::array vertexBuffers{rachis::VertexBufferLayout{.stride = 32, .attributes = attributes}};
        const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
        auto graphicsPipeline = device.createGraphicsPipeline({
            .layout = *graphicsLayout,
            .vertexShader = *vertexModule,
            .pixelShader = *pixelModule,
            .vertexBuffers = vertexBuffers,
            .colorTargets = colorTargets,
            .label = "GLSL vertex input",
        });
        REQUIRE(graphicsPipeline.has_value());
        CHECK(device.destroyPipeline(*graphicsPipeline).has_value());
        CHECK(device.destroyPipelineLayout(*graphicsLayout).has_value());
        CHECK(device.destroyShaderModule(*pixelModule).has_value());
        CHECK(device.destroyShaderModule(*vertexModule).has_value());
        const auto counts = device.diagnosticCounts();
        CHECK(counts.errors == 0);
        CHECK(counts.corruptions == 0);
    }
}

#endif
