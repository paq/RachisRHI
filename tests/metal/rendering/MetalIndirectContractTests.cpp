#include "Metal/MetalIndirectRecordMath.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint32_t targetSize = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetSize;
    constexpr std::uint32_t quadrantCount = 4;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadPnms(std::string_view name)
    {
        std::ifstream stream(std::filesystem::path(RACHIS_METALLIB_DIR) / std::filesystem::path(name),
                             std::ios::binary | std::ios::ate);
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

    struct QuadrantVertex final
    {
        std::array<float, 2> position{};
        std::array<float, 3> color{};
    };

    static_assert(sizeof(QuadrantVertex) == 20);

    [[nodiscard]] std::array<QuadrantVertex, quadrantCount * 3> quadrantVertices() noexcept
    {
        std::array<QuadrantVertex, quadrantCount * 3> vertices{};
        for (std::uint32_t quadrant = 0; quadrant < quadrantCount; ++quadrant)
        {
            const float centerX = (quadrant % 2U) == 0 ? -0.5F : 0.5F;
            const float centerY = (quadrant / 2U) == 0 ? 0.5F : -0.5F;
            vertices[quadrant * 3 + 0] = {{centerX - 0.3F, centerY - 0.3F}, {1.0F, 1.0F, 1.0F}};
            vertices[quadrant * 3 + 1] = {{centerX + 0.3F, centerY - 0.3F}, {1.0F, 1.0F, 1.0F}};
            vertices[quadrant * 3 + 2] = {{centerX, centerY + 0.3F}, {1.0F, 1.0F, 1.0F}};
        }
        return vertices;
    }

    template<typename Value>
    [[nodiscard]] rachis::BufferHandle uploadBuffer(rachis::Device& device,
                                                    std::span<const Value> values,
                                                    rachis::BufferUsage usage,
                                                    std::string_view label)
    {
        auto buffer = device.createBuffer({
            .size = values.size_bytes(),
            .usage = usage,
            .memoryType = rachis::MemoryType::upload,
            .label = label,
        });
        REQUIRE(buffer.has_value());
        auto mapped = device.mapBuffer(*buffer);
        REQUIRE(mapped.has_value());
        std::memcpy(mapped->data, values.data(), values.size_bytes());
        REQUIRE(device.unmapBuffer(*buffer).has_value());
        return *buffer;
    }

    struct QuadrantRenderer final
    {
        rachis::Device& device;
        rachis::ShaderModuleHandle vertexShader{};
        rachis::ShaderModuleHandle pixelShader{};
        rachis::PipelineLayoutHandle layout{};
        rachis::PipelineHandle pipeline{};
        rachis::BufferHandle vertices{};
        rachis::BufferHandle indices{};
        rachis::TextureHandle target{};
        rachis::TextureViewHandle targetView{};
        rachis::BufferHandle readback{};

        explicit QuadrantRenderer(rachis::Device& owner) : device(owner)
        {
            const auto vertexBytes = loadPnms("MetalRasterVertex.vs.pnms");
            const auto pixelBytes = loadPnms("MetalRasterVertex.ps.pnms");
            REQUIRE_FALSE(vertexBytes.empty());
            REQUIRE_FALSE(pixelBytes.empty());
            const std::array<std::uint32_t, 2> locations{0, 1};
            auto vertex = device.createShaderModule({
                .stage = rachis::ShaderStage::vertex,
                .bytecode = vertexBytes,
                .entryPoint = "vertexMain",
                .shaderInterface = rachis::test::shaderInterface({}, {}, locations),
            });
            auto pixel = device.createShaderModule({
                .stage = rachis::ShaderStage::pixel,
                .bytecode = pixelBytes,
                .entryPoint = "pixelMain",
            });
            REQUIRE(vertex.has_value());
            REQUIRE(pixel.has_value());
            vertexShader = *vertex;
            pixelShader = *pixel;
            auto created = device.createPipelineLayout({});
            REQUIRE(created.has_value());
            layout = *created;
            const std::array attributes{
                rachis::VertexAttribute{rachis::VertexFormat::float32x2, 0, 0},
                rachis::VertexAttribute{rachis::VertexFormat::float32x3, 8, 1},
            };
            const std::array buffers{
                rachis::VertexBufferLayout{.stride = sizeof(QuadrantVertex), .attributes = attributes}};
            const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
            auto graphics = device.createGraphicsPipeline({
                .layout = layout,
                .vertexShader = vertexShader,
                .pixelShader = pixelShader,
                .vertexBuffers = buffers,
                .cullMode = rachis::CullMode::none,
                .colorTargets = colorTargets,
                .label = "Metal quadrant pipeline",
            });
            REQUIRE(graphics.has_value());
            pipeline = *graphics;
            const auto vertexData = quadrantVertices();
            vertices = uploadBuffer(
                device, std::span<const QuadrantVertex>(vertexData), rachis::BufferUsage::vertex, "Quadrant vertices");
            std::array<std::uint16_t, quadrantCount * 3> indexData{};
            for (std::uint16_t index = 0; index < indexData.size(); ++index)
            {
                indexData[index] = index;
            }
            indices = uploadBuffer(
                device, std::span<const std::uint16_t>(indexData), rachis::BufferUsage::index, "Quadrant indices");
            auto texture = device.createTexture({
                .width = targetSize,
                .height = targetSize,
                .format = rachis::Format::rgba8Unorm,
                .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
                .label = "Metal quadrant target",
            });
            REQUIRE(texture.has_value());
            target = *texture;
            auto view = device.createTextureView({.texture = target, .usage = rachis::TextureViewUsage::renderTarget});
            REQUIRE(view.has_value());
            targetView = *view;
            auto buffer = device.createBuffer({
                .size = readbackSize,
                .usage = rachis::BufferUsage::copyDestination,
                .memoryType = rachis::MemoryType::readback,
            });
            REQUIRE(buffer.has_value());
            readback = *buffer;
        }

        QuadrantRenderer(const QuadrantRenderer&) = delete;
        QuadrantRenderer& operator=(const QuadrantRenderer&) = delete;

        ~QuadrantRenderer()
        {
            CHECK(device.destroyBuffer(readback).has_value());
            CHECK(device.destroyTextureView(targetView).has_value());
            CHECK(device.destroyTexture(target).has_value());
            CHECK(device.destroyBuffer(indices).has_value());
            CHECK(device.destroyBuffer(vertices).has_value());
            CHECK(device.destroyPipeline(pipeline).has_value());
            CHECK(device.destroyPipelineLayout(layout).has_value());
            CHECK(device.destroyShaderModule(pixelShader).has_value());
            CHECK(device.destroyShaderModule(vertexShader).has_value());
        }

        [[nodiscard]] std::array<bool, quadrantCount> render(
            const std::function<void(rachis::CommandBuffer&)>& draws,
            const std::function<void(rachis::CommandBuffer&)>& prologue = {})
        {
            auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            REQUIRE(command.has_value());
            if (prologue)
            {
                prologue(*command);
            }
            REQUIRE(device
                        .textureBarrier(*command,
                                        {
                                            .texture = target,
                                            .stagesAfter = rachis::PipelineStage::renderTarget,
                                            .accessAfter = rachis::Access::renderTarget,
                                            .layoutAfter = rachis::TextureLayout::renderTarget,
                                        })
                        .has_value());
            const std::array attachments{rachis::ColorAttachment{
                .view = targetView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
            }};
            REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
            REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
            REQUIRE(device
                        .setViewport(*command,
                                     {.width = static_cast<float>(targetSize),
                                      .height = static_cast<float>(targetSize),
                                      .maxDepth = 1.0F})
                        .has_value());
            REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}).has_value());
            REQUIRE(
                device.bindVertexBuffers(*command, 0, std::array{rachis::VertexBufferBinding{vertices}}).has_value());
            REQUIRE(device.bindIndexBuffer(*command, indices, rachis::IndexFormat::uint16, 0).has_value());
            draws(*command);
            REQUIRE(device.endRendering(*command).has_value());
            REQUIRE(device
                        .textureBarrier(*command,
                                        {
                                            .texture = target,
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
                                                 .buffer = readback,
                                                 .bytesPerRow = rowPitch,
                                                 .rowsPerImage = targetSize,
                                                 .texture = target,
                                                 .extent = {targetSize, targetSize, 1},
                                             })
                        .has_value());
            auto submitted = device.submit(std::move(*command));
            REQUIRE(submitted.has_value());
            REQUIRE(device.wait(*submitted).has_value());
            auto mapped = device.mapBuffer(readback);
            REQUIRE(mapped.has_value());
            std::array<bool, quadrantCount> lit{};
            for (std::uint32_t quadrant = 0; quadrant < quadrantCount; ++quadrant)
            {
                const auto x = 16U + 32U * (quadrant % 2U);
                const auto y = 16U + 32U * (quadrant / 2U);
                const auto* pixel = mapped->data + static_cast<std::uint64_t>(y) * rowPitch + x * 4U;
                lit[quadrant] = pixel[0] == std::byte{255} && pixel[1] == std::byte{255};
            }
            REQUIRE(device.unmapBuffer(readback).has_value());
            return lit;
        }
    };

    template<typename Record>
    struct PaddedRecord final
    {
        Record record{};
        std::uint32_t padding = 0xDEADBEEFU;
    };

    [[nodiscard]] rachis::BufferHandle staleCountBuffer(rachis::Device& device, std::size_t countCount)
    {
        const std::vector<std::uint32_t> stale(countCount, 1U);
        return uploadBuffer(device,
                            std::span<const std::uint32_t>(stale),
                            rachis::BufferUsage::indirect | rachis::BufferUsage::storage |
                                rachis::BufferUsage::copyDestination,
                            "Metal GPU-written counts");
    }

    void writeCountsOnGpu(rachis::Device& device,
                          rachis::CommandBuffer& command,
                          rachis::BufferHandle countBuffer,
                          std::span<const std::uint32_t> counts)
    {
        for (std::size_t index = 0; index < counts.size(); ++index)
        {
            REQUIRE(device
                        .fillBuffer(
                            command, countBuffer, index * sizeof(std::uint32_t), sizeof(std::uint32_t), counts[index])
                        .has_value());
        }
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = countBuffer,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::indirect,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::indirectArgument,
                                   })
                    .has_value());
    }

} // namespace

TEST_CASE("Metal count clamps launch enough workgroups for every record count", "[size:small][rhi][metal][indirect]")
{
    STATIC_REQUIRE(rachis::metal::metalClampThreadgroups(0) == 0);
    STATIC_REQUIRE(rachis::metal::metalClampThreadgroups(1) == 1);
    STATIC_REQUIRE(rachis::metal::metalClampThreadgroups(64) == 1);
    STATIC_REQUIRE(rachis::metal::metalClampThreadgroups(65) == 2);
    STATIC_REQUIRE(rachis::metal::metalClampThreadgroups(std::numeric_limits<std::uint32_t>::max() - 62U) ==
                   67'108'864U);
    STATIC_REQUIRE(rachis::metal::metalClampThreadgroups(std::numeric_limits<std::uint32_t>::max()) == 67'108'864U);
}

TEST_CASE("Metal reports multi-record indirect calls and GPU-written counts", "[size:large][rhi][metal][indirect]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    CHECK(capabilities.multiDrawIndirect);
    CHECK(capabilities.drawIndirectCount);
    CHECK(capabilities.multiDispatchIndirect);
    CHECK(capabilities.dispatchIndirectCount);
}

TEST_CASE("Metal multi-draw indirect draws every record or only the GPU-written count",
          "[size:large][rhi][metal][indirect]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    QuadrantRenderer renderer(device);

    std::array<PaddedRecord<rachis::DrawIndirectArguments>, quadrantCount> drawRecords{};
    std::array<PaddedRecord<rachis::DrawIndexedIndirectArguments>, quadrantCount> indexedRecords{};
    for (std::uint32_t quadrant = 0; quadrant < quadrantCount; ++quadrant)
    {
        drawRecords[quadrant].record = {.vertexCount = 3, .instanceCount = 1, .firstVertex = quadrant * 3};
        indexedRecords[quadrant].record = {.indexCount = 3, .instanceCount = 1, .firstIndex = quadrant * 3};
    }
    const auto drawArguments = uploadBuffer(device,
                                            std::span<const PaddedRecord<rachis::DrawIndirectArguments>>(drawRecords),
                                            rachis::BufferUsage::indirect,
                                            "Metal draw records");
    const auto indexedArguments =
        uploadBuffer(device,
                     std::span<const PaddedRecord<rachis::DrawIndexedIndirectArguments>>(indexedRecords),
                     rachis::BufferUsage::indirect,
                     "Metal indexed draw records");
    const std::array<std::uint32_t, 4> counts{2, 3, 0, 9};
    const auto countBuffer = staleCountBuffer(device, counts.size());
    const auto writeCounts = [&](rachis::CommandBuffer& command) {
        writeCountsOnGpu(device, command, countBuffer, counts);
    };
    constexpr std::uint32_t drawStride = sizeof(PaddedRecord<rachis::DrawIndirectArguments>);
    constexpr std::uint32_t indexedStride = sizeof(PaddedRecord<rachis::DrawIndexedIndirectArguments>);

    SECTION("Without a count buffer every record draws")
    {
        const auto lit = renderer.render([&](rachis::CommandBuffer& command) {
            REQUIRE(
                device
                    .drawIndirect(
                        command, {.argumentBuffer = drawArguments, .maxDrawCount = quadrantCount, .stride = drawStride})
                    .has_value());
        });
        CHECK(lit == std::array{true, true, true, true});
    }

    SECTION("A count of two draws the first two records")
    {
        const auto lit = renderer.render(
            [&](rachis::CommandBuffer& command) {
                REQUIRE(device
                            .drawIndirect(command,
                                          {.argumentBuffer = drawArguments,
                                           .maxDrawCount = quadrantCount,
                                           .stride = drawStride,
                                           .countBuffer = countBuffer})
                            .has_value());
            },
            writeCounts);
        CHECK(lit == std::array{true, true, false, false});
    }

    SECTION("An indexed count of three draws the first three records")
    {
        const auto lit = renderer.render(
            [&](rachis::CommandBuffer& command) {
                REQUIRE(device
                            .drawIndexedIndirect(command,
                                                 {.argumentBuffer = indexedArguments,
                                                  .maxDrawCount = quadrantCount,
                                                  .stride = indexedStride,
                                                  .countBuffer = countBuffer,
                                                  .countOffset = sizeof(std::uint32_t)})
                            .has_value());
            },
            writeCounts);
        CHECK(lit == std::array{true, true, true, false});
    }

    SECTION("A zero count draws nothing and a count above the maximum draws the maximum")
    {
        const auto lit = renderer.render(
            [&](rachis::CommandBuffer& command) {
                REQUIRE(device
                            .drawIndirect(command,
                                          {.argumentBuffer = drawArguments,
                                           .maxDrawCount = quadrantCount,
                                           .stride = drawStride,
                                           .countBuffer = countBuffer,
                                           .countOffset = 2 * sizeof(std::uint32_t)})
                            .has_value());
                REQUIRE(device
                            .drawIndexedIndirect(command,
                                                 {.argumentBuffer = indexedArguments,
                                                  .argumentOffset = indexedStride,
                                                  .maxDrawCount = 2,
                                                  .stride = indexedStride,
                                                  .countBuffer = countBuffer,
                                                  .countOffset = 3 * sizeof(std::uint32_t)})
                            .has_value());
            },
            writeCounts);
        CHECK(lit == std::array{false, true, true, false});
    }

    SECTION("Records and counts are validated")
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        auto outOfBounds = device.drawIndirect(
            *command, {.argumentBuffer = drawArguments, .maxDrawCount = quadrantCount + 1, .stride = drawStride});
        REQUIRE_FALSE(outOfBounds.has_value());
        CHECK(outOfBounds.error().code == rachis::ErrorCode::invalidArgument);
        auto misalignedCount = device.drawIndirect(
            *command,
            {.argumentBuffer = drawArguments, .maxDrawCount = 1, .countBuffer = countBuffer, .countOffset = 2});
        REQUIRE_FALSE(misalignedCount.has_value());
        CHECK(misalignedCount.error().code == rachis::ErrorCode::invalidArgument);
        auto countPastEnd = device.drawIndirect(*command,
                                                {.argumentBuffer = drawArguments,
                                                 .maxDrawCount = 1,
                                                 .countBuffer = countBuffer,
                                                 .countOffset = sizeof(counts)});
        REQUIRE_FALSE(countPastEnd.has_value());
        CHECK(countPastEnd.error().code == rachis::ErrorCode::invalidArgument);
        auto shortStride =
            device.drawIndexedIndirect(*command, {.argumentBuffer = indexedArguments, .maxDrawCount = 2, .stride = 16});
        REQUIRE_FALSE(shortStride.has_value());
        CHECK(shortStride.error().code == rachis::ErrorCode::invalidArgument);
    }

    REQUIRE(device.destroyBuffer(countBuffer).has_value());
    REQUIRE(device.destroyBuffer(indexedArguments).has_value());
    REQUIRE(device.destroyBuffer(drawArguments).has_value());
}

TEST_CASE("Metal multi-dispatch indirect runs every record or only the GPU-written count",
          "[size:large][rhi][metal][indirect]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto shaderBytes = loadPnms("MetalIndirectCount.cs.pnms");
    REQUIRE_FALSE(shaderBytes.empty());
    const std::array bindings{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = shaderBytes,
        .entryPoint = "countMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, bindings),
    });
    REQUIRE(shader.has_value());
    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries});
    REQUIRE(bindingLayout.has_value());
    const std::array layouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    std::array<PaddedRecord<rachis::DispatchIndirectArguments>, 4> records{};
    const std::array<std::uint32_t, 4> groupCounts{1, 2, 4, 8};
    for (std::size_t record = 0; record < records.size(); ++record)
    {
        records[record].record = {.groupCountX = groupCounts[record], .groupCountY = 1, .groupCountZ = 1};
    }
    const auto arguments = uploadBuffer(device,
                                        std::span<const PaddedRecord<rachis::DispatchIndirectArguments>>(records),
                                        rachis::BufferUsage::indirect,
                                        "Metal dispatch records");
    const std::array<std::uint32_t, 2> counts{3, 0};
    const auto countBuffer = staleCountBuffer(device, counts.size());
    constexpr std::uint32_t stride = sizeof(PaddedRecord<rachis::DispatchIndirectArguments>);

    const auto runDispatch = [&](const rachis::IndirectDispatchDesc& desc) -> std::uint32_t {
        const std::array<std::uint32_t, 1> zero{0};
        const auto counter = uploadBuffer(device,
                                          std::span<const std::uint32_t>(zero),
                                          rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
                                          "Metal dispatch counter");
        const std::array writes{rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = counter, .structureByteStride = sizeof(std::uint32_t)},
        }};
        auto set = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
        REQUIRE(set.has_value());
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        if (desc.countBuffer.isValid())
        {
            writeCountsOnGpu(device, *command, countBuffer, counts);
        }
        REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());
        REQUIRE(device.dispatchIndirect(*command, desc).has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        auto mapped = device.mapBuffer(counter);
        REQUIRE(mapped.has_value());
        std::uint32_t value = 0;
        std::memcpy(&value, mapped->data, sizeof(value));
        REQUIRE(device.unmapBuffer(counter).has_value());
        REQUIRE(device.destroyBindingSet(*set).has_value());
        REQUIRE(device.destroyBuffer(counter).has_value());
        return value;
    };

    CHECK(runDispatch({.argumentBuffer = arguments, .maxDispatchCount = 4, .stride = stride}) == 15);
    CHECK(runDispatch(
              {.argumentBuffer = arguments, .maxDispatchCount = 4, .stride = stride, .countBuffer = countBuffer}) == 7);
    CHECK(runDispatch({.argumentBuffer = arguments,
                       .maxDispatchCount = 4,
                       .stride = stride,
                       .countBuffer = countBuffer,
                       .countOffset = sizeof(std::uint32_t)}) == 0);
    CHECK(runDispatch({.argumentBuffer = arguments,
                       .argumentOffset = stride,
                       .maxDispatchCount = 2,
                       .stride = stride,
                       .countBuffer = countBuffer}) == 6);

    REQUIRE(device.destroyBuffer(countBuffer).has_value());
    REQUIRE(device.destroyBuffer(arguments).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
}
