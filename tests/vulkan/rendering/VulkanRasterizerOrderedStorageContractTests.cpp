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

    constexpr std::uint32_t overlappingPrimitiveCount = 64;
    constexpr std::uint64_t counterSize = sizeof(std::uint32_t);

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
#endif

} // namespace

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan pixel interlock serializes overlapping fragment storage updates",
          "[size:medium][rhi][vulkan][rasterizer-ordered-storage]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipUnlessReferenceDesktopAdvertises(
            rachis::test::requireCapabilities(device).rasterizerOrderedStorageAccess, "rasterizerOrderedStorageAccess"))
    {
        return;
    }

    const auto vertexBytecode = loadSpirv("VulkanRasterizerOrderedStorage.vs.spv");
    const auto pixelBytecode = loadSpirv("RasterizerOrderedStorage.ps.spv");
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "vertexMain",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "main",
        .shaderInterface = rachis::test::shaderInterface(
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::pixel},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries});
    REQUIRE(bindingLayout.has_value());

    auto counter = device.createBuffer(rachis::BufferDesc{
        .size = counterSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = counterSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = counterSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(counter.has_value());
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());

    auto mappedUpload = device.mapBuffer(*upload);
    REQUIRE(mappedUpload.has_value());
    const std::uint32_t zero = 0;
    std::memcpy(mappedUpload->data, &zero, sizeof(zero));
    REQUIRE(device.unmapBuffer(*upload).has_value());

    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *counter, .size = counterSize},
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(bindingSet.has_value());

    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = bindingLayouts});
    REQUIRE(pipelineLayout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    auto pipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    if (!pipeline)
    {
        INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());

    auto target = device.createTexture(rachis::TextureDesc{
        .width = 1,
        .height = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *counter,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *upload, .destination = *counter, .size = counterSize}).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *counter,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::pixelShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *target,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindGraphicsBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.setViewport(*command, {.width = 1.0F, .height = 1.0F}).has_value());
    REQUIRE(device.setScissor(*command, {.width = 1, .height = 1}).has_value());
    REQUIRE(device.draw(*command, 3, overlappingPrimitiveCount).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *counter,
                                   .stagesBefore = rachis::PipelineStage::pixelShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(
        device.copyBuffer(*command, {.source = *counter, .destination = *readback, .size = counterSize}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback.has_value());
    std::uint32_t observed = 0;
    std::memcpy(&observed, mappedReadback->data, sizeof(observed));
    CHECK(observed == overlappingPrimitiveCount);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*counter).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
#endif
