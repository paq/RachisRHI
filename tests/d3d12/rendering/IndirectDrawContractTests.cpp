#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 64;
    constexpr std::uint32_t targetHeight = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetHeight;

    class FailResourceRetention final
    {
    public:
        explicit FailResourceRetention(std::int32_t countdown)
            : previous_(rachis::d3d12::test::resourceRetentionFailureCountdown().exchange(countdown,
                                                                                          std::memory_order_relaxed))
        {
        }

        ~FailResourceRetention()
        {
            rachis::d3d12::test::resourceRetentionFailureCountdown().store(previous_, std::memory_order_relaxed);
        }

        FailResourceRetention(const FailResourceRetention&) = delete;
        FailResourceRetention& operator=(const FailResourceRetention&) = delete;
        FailResourceRetention(FailResourceRetention&&) = delete;
        FailResourceRetention& operator=(FailResourceRetention&&) = delete;

    private:
        std::int32_t previous_;
    };

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

    [[nodiscard]] bool isRed(std::span<const std::byte> pixels, std::uint32_t x, std::uint32_t y) noexcept
    {
        const auto offset = static_cast<std::size_t>(y) * rowPitch + static_cast<std::size_t>(x) * 4U;
        return std::to_integer<std::uint8_t>(pixels[offset]) > 180 &&
               std::to_integer<std::uint8_t>(pixels[offset + 1]) < 100 &&
               std::to_integer<std::uint8_t>(pixels[offset + 2]) < 100;
    }

    [[nodiscard]] bool isClear(std::span<const std::byte> pixels, std::uint32_t x, std::uint32_t y) noexcept
    {
        const auto offset = static_cast<std::size_t>(y) * rowPitch + static_cast<std::size_t>(x) * 4U;
        return std::to_integer<std::uint8_t>(pixels[offset]) < 4 &&
               std::to_integer<std::uint8_t>(pixels[offset + 1]) < 4 &&
               std::to_integer<std::uint8_t>(pixels[offset + 2]) < 4;
    }

    void verifyIndirectValidation(rachis::Device& device,
                                  rachis::CommandBuffer& command,
                                  rachis::BufferHandle arguments,
                                  rachis::BufferHandle count,
                                  rachis::BufferHandle wrongUsage,
                                  rachis::BufferHandle staleArguments,
                                  std::uint64_t argumentSize)
    {
        CHECK_FALSE(device.drawIndirect(command, {.argumentBuffer = wrongUsage}).has_value());
        CHECK_FALSE(device.drawIndirect(command, {.argumentBuffer = arguments, .argumentOffset = 2, .maxDrawCount = 1})
                        .has_value());
        CHECK_FALSE(
            device
                .drawIndirect(command,
                              {.argumentBuffer = arguments, .argumentOffset = 4, .maxDrawCount = 1, .stride = 12})
                .has_value());
        CHECK_FALSE(
            device
                .drawIndirect(command, {.argumentBuffer = arguments, .argumentOffset = argumentSize, .maxDrawCount = 1})
                .has_value());
        CHECK_FALSE(device.drawIndirect(command, {.argumentBuffer = arguments, .argumentOffset = 4, .countOffset = 4})
                        .has_value());
        CHECK_FALSE(device.drawIndirect(command, {.argumentBuffer = staleArguments}).has_value());
        CHECK_FALSE(device
                        .drawIndexedIndirect(command,
                                             {.argumentBuffer = arguments,
                                              .argumentOffset = 4,
                                              .maxDrawCount = 2,
                                              .stride = 32,
                                              .countBuffer = count,
                                              .countOffset = 4})
                        .has_value());
        REQUIRE(
            device
                .drawIndirect(command, {.argumentBuffer = arguments, .argumentOffset = argumentSize, .maxDrawCount = 0})
                .has_value());
    }

} // namespace

TEST_CASE("indirect draws validate ranges, execute padded multi-draw counts, and retain buffers",
          "[size:medium][rhi][graphics][indirect][warp]")
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

    const auto computeBytecode = loadShader(RACHIS_TEST_INDIRECT_COMPUTE_SHADER_PATH);
    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_INPUT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(computeBytecode.empty());
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "inputVertex",
        .shaderInterface = rachis::test::graphicsInputVertexShaderInterface(),
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto layout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(layout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const std::array vertexAttributes{
        rachis::VertexAttribute{
            .format = rachis::VertexFormat::float32x2,
            .shaderLocation = 0,
        },
    };
    const std::array vertexLayouts{
        rachis::VertexBufferLayout{
            .stride = 2 * sizeof(float),
            .attributes = vertexAttributes,
        },
    };
    auto pipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .vertexBuffers = vertexLayouts,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE(pipeline.has_value());

    auto computeShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = computeBytecode,
        .entryPoint = "main",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
    });
    REQUIRE(computeShader.has_value());
    auto gpuArguments = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(rachis::DrawIndirectArguments),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    REQUIRE(gpuArguments.has_value());
    const std::array computeEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto computeBindingLayout = device.createBindingLayout(rachis::BindingLayoutDesc{.entries = computeEntries});
    REQUIRE(computeBindingLayout.has_value());
    const std::array computeWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *gpuArguments,
                    .size = sizeof(rachis::DrawIndirectArguments),
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto computeBindingSet = rachis::test::createBindingSet(device,
                                                            rachis::BindingSetDesc{
                                                                .layout = *computeBindingLayout,
                                                                .writes = computeWrites,
                                                            });
    REQUIRE(computeBindingSet.has_value());
    const std::array computeBindingLayouts{*computeBindingLayout};
    auto computePipelineLayout =
        device.createPipelineLayout(rachis::PipelineLayoutDesc{.bindingLayouts = computeBindingLayouts});
    REQUIRE(computePipelineLayout.has_value());
    auto computePipeline = device.createComputePipeline(rachis::ComputePipelineDesc{
        .layout = *computePipelineLayout,
        .computeShader = *computeShader,
    });
    REQUIRE(computePipeline.has_value());

    auto target = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView.has_value());
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(readback.has_value());

    const std::array vertices{
        std::array{-0.85F, 0.65F},
        std::array{-0.15F, -0.65F},
        std::array{-0.85F, -0.65F},
        std::array{0.15F, 0.65F},
        std::array{0.85F, -0.65F},
        std::array{0.15F, -0.65F},
    };
    auto vertexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(vertexBuffer.has_value());
    auto mappedVertices = device.mapBuffer(*vertexBuffer);
    REQUIRE(mappedVertices.has_value());
    std::memcpy(mappedVertices->data, vertices.data(), sizeof(vertices));
    REQUIRE(device.unmapBuffer(*vertexBuffer).has_value());

    constexpr std::uint64_t argumentOffset = 4;
    constexpr std::uint32_t paddedStride = 32;
    constexpr std::uint32_t one = 1;
    std::array<std::byte, 68> argumentBytes{};
    const rachis::DrawIndirectArguments leftDraw{.vertexCount = 3, .instanceCount = 1};
    const rachis::DrawIndirectArguments rightDraw{.vertexCount = 3, .instanceCount = 1, .firstVertex = 3};
    std::memcpy(argumentBytes.data() + argumentOffset, &leftDraw, sizeof(leftDraw));
    std::memcpy(argumentBytes.data() + argumentOffset + paddedStride, &rightDraw, sizeof(rightDraw));
    auto arguments = device.createBuffer(rachis::BufferDesc{
        .size = argumentBytes.size(),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
    });
    auto count = device.createBuffer(rachis::BufferDesc{
        .size = 8,
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(arguments.has_value());
    REQUIRE(count.has_value());
    auto mappedArguments = device.mapBuffer(*arguments);
    auto mappedCount = device.mapBuffer(*count);
    REQUIRE(mappedArguments.has_value());
    REQUIRE(mappedCount.has_value());
    std::memcpy(mappedArguments->data, argumentBytes.data(), argumentBytes.size());
    std::memcpy(mappedCount->data + sizeof(std::uint32_t), &one, sizeof(one));
    REQUIRE(device.unmapBuffer(*arguments).has_value());
    REQUIRE(device.unmapBuffer(*count).has_value());

    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    auto indexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(indices),
        .usage = rachis::BufferUsage::index,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(indexBuffer.has_value());
    auto mappedIndices = device.mapBuffer(*indexBuffer);
    REQUIRE(mappedIndices.has_value());
    std::memcpy(mappedIndices->data, indices.data(), sizeof(indices));
    REQUIRE(device.unmapBuffer(*indexBuffer).has_value());

    auto wrongUsage = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(rachis::DrawIndirectArguments),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(wrongUsage.has_value());
    auto staleArguments = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(rachis::DrawIndirectArguments),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(staleArguments.has_value());
    REQUIRE(device.destroyBuffer(*staleArguments).has_value());

    const rachis::IndirectDrawDesc countedDraws{
        .argumentBuffer = *arguments,
        .argumentOffset = argumentOffset,
        .maxDrawCount = 2,
        .stride = paddedStride,
        .countBuffer = *count,
        .countOffset = sizeof(std::uint32_t),
    };
    const rachis::IndirectDrawDesc gpuDraw{
        .argumentBuffer = *gpuArguments,
        .maxDrawCount = 1,
    };
    bool firstPass = true;
    const auto renderAndRead = [&](const rachis::IndirectDrawDesc& draws,
                                   bool writeGpuArguments,
                                   bool destroyRetainedBuffers) {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        if (writeGpuArguments)
        {
            REQUIRE(device
                        .bufferBarrier(*command,
                                       rachis::BufferBarrier{
                                           .buffer = *gpuArguments,
                                           .stagesBefore = rachis::PipelineStage::all,
                                           .stagesAfter = rachis::PipelineStage::computeShader,
                                           .accessBefore = rachis::Access::common,
                                           .accessAfter = rachis::Access::unorderedAccess,
                                       })
                        .has_value());
            REQUIRE(device.bindComputePipeline(*command, *computePipeline).has_value());
            REQUIRE(device.bindComputeBindingSet(*command, 0, *computeBindingSet).has_value());
            REQUIRE(device.dispatch(*command, 1).has_value());
            REQUIRE(device
                        .bufferBarrier(*command,
                                       rachis::BufferBarrier{
                                           .buffer = *gpuArguments,
                                           .stagesBefore = rachis::PipelineStage::computeShader,
                                           .stagesAfter = rachis::PipelineStage::indirect,
                                           .accessBefore = rachis::Access::unorderedAccess,
                                           .accessAfter = rachis::Access::indirectArgument,
                                       })
                        .has_value());
        }
        REQUIRE(device
                    .textureBarrier(
                        *command,
                        rachis::TextureBarrier{
                            .texture = *target,
                            .stagesBefore = firstPass ? rachis::PipelineStage::all : rachis::PipelineStage::copy,
                            .stagesAfter = rachis::PipelineStage::renderTarget,
                            .accessBefore = firstPass ? rachis::Access::none : rachis::Access::copySource,
                            .accessAfter = rachis::Access::renderTarget,
                            .layoutBefore =
                                firstPass ? rachis::TextureLayout::undefined : rachis::TextureLayout::copySource,
                            .layoutAfter = rachis::TextureLayout::renderTarget,
                        })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *targetView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {},
            },
        };
        REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
        REQUIRE(device.bindVertexBuffer(*command, 0, *vertexBuffer).has_value());
        REQUIRE(device
                    .setViewport(*command,
                                 rachis::Viewport{
                                     .width = static_cast<float>(targetWidth),
                                     .height = static_cast<float>(targetHeight),
                                 })
                    .has_value());
        REQUIRE(
            device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
        const std::array tint{0.9F, 0.2F, 0.1F, 1.0F};
        REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                    .has_value());

        if (firstPass)
        {
            verifyIndirectValidation(
                device, *command, *arguments, *count, *wrongUsage, *staleArguments, argumentBytes.size());

            auto aliasedCountDraws = countedDraws;
            aliasedCountDraws.countBuffer = *arguments;
            aliasedCountDraws.countOffset = 0;
            REQUIRE(device.drawIndirect(*command, aliasedCountDraws).has_value());

            const auto beforeRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
            REQUIRE(beforeRejectedPublication.has_value());
            const auto recordsBeforeRejectedPublication =
                rachis::d3d12::test::indirectExecutionRecordCount().load(std::memory_order_relaxed);
            {
                const FailResourceRetention failCountPublication(2);
                const auto rejected = device.drawIndirect(*command, countedDraws);
                REQUIRE_FALSE(rejected.has_value());
                CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
            }
            const auto afterRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
            REQUIRE(afterRejectedPublication.has_value());
            CHECK(*afterRejectedPublication == *beforeRejectedPublication);
            CHECK(rachis::d3d12::test::indirectExecutionRecordCount().load(std::memory_order_relaxed) ==
                  recordsBeforeRejectedPublication);

            REQUIRE(device.bindIndexBuffer(*command, *indexBuffer, rachis::IndexFormat::uint16).has_value());
            const auto beforeRejectedIndexedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
            REQUIRE(beforeRejectedIndexedPublication.has_value());
            const auto recordsBeforeRejectedIndexedPublication =
                rachis::d3d12::test::indirectExecutionRecordCount().load(std::memory_order_relaxed);
            {
                const FailResourceRetention failIndexedCountPublication(2);
                const auto rejected = device.drawIndexedIndirect(*command, countedDraws);
                REQUIRE_FALSE(rejected.has_value());
                CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
            }
            const auto afterRejectedIndexedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
            REQUIRE(afterRejectedIndexedPublication.has_value());
            CHECK(*afterRejectedIndexedPublication == *beforeRejectedIndexedPublication);
            CHECK(rachis::d3d12::test::indirectExecutionRecordCount().load(std::memory_order_relaxed) ==
                  recordsBeforeRejectedIndexedPublication);
        }

        REQUIRE(device.drawIndirect(*command, draws).has_value());
        if (destroyRetainedBuffers)
        {
            auto fixedDraws = countedDraws;
            fixedDraws.countBuffer = {};
            fixedDraws.countOffset = 0;
            REQUIRE(device.drawIndirect(*command, fixedDraws).has_value());
            REQUIRE(device.bindIndexBuffer(*command, *indexBuffer, rachis::IndexFormat::uint16).has_value());
            REQUIRE(device
                        .drawIndexedIndirect(*command,
                                             {.argumentBuffer = *arguments,
                                              .argumentOffset = argumentOffset,
                                              .maxDrawCount = 1,
                                              .stride = paddedStride})
                        .has_value());
        }
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
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
                                         rachis::BufferTextureCopy{
                                             .buffer = *readback,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = targetHeight,
                                             .texture = *target,
                                             .extent = {targetWidth, targetHeight, 1},
                                         })
                    .has_value());
        if (destroyRetainedBuffers)
        {
            REQUIRE(device.destroyBuffer(*arguments).has_value());
            REQUIRE(device.destroyBuffer(*count).has_value());
            REQUIRE(device.destroyBuffer(*indexBuffer).has_value());
            REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
        }
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        firstPass = false;

        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped.has_value());
        std::array<std::byte, readbackSize> pixels{};
        std::memcpy(pixels.data(), mapped->data, pixels.size());
        REQUIRE(device.unmapBuffer(*readback).has_value());
        return pixels;
    };

    const auto gpuDrawPixels = renderAndRead(gpuDraw, true, false);
    CHECK(isClear(gpuDrawPixels, 16, 36));
    CHECK(isRed(gpuDrawPixels, 48, 36));

    const auto oneDrawPixels = renderAndRead(countedDraws, false, false);
    CHECK(isRed(oneDrawPixels, 16, 36));
    CHECK(isClear(oneDrawPixels, 48, 36));

    mappedCount = device.mapBuffer(*count);
    REQUIRE(mappedCount.has_value());
    const std::uint32_t three = 3;
    std::memcpy(mappedCount->data + sizeof(std::uint32_t), &three, sizeof(three));
    REQUIRE(device.unmapBuffer(*count).has_value());
    const auto clampedPixels = renderAndRead(countedDraws, false, true);
    CHECK(isRed(clampedPixels, 16, 36));
    CHECK(isRed(clampedPixels, 48, 36));

    REQUIRE(device.destroyBuffer(*wrongUsage).has_value());
    REQUIRE(device.destroyPipeline(*computePipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*computePipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*computeBindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*computeBindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*computeShader).has_value());
    REQUIRE(device.destroyBuffer(*gpuArguments).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
}
