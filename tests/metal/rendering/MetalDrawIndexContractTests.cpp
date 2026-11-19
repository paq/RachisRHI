#include "DrawIndexContract.h"
#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

namespace
{

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

    namespace draw_index = rachis::test::draw_index;

    struct DrawIndexPipelines final
    {
        rachis::Device& device;
        std::vector<std::byte> vertexBytes = loadPnms("DrawIndex.vs.pnms");
        std::vector<std::byte> meshBytes = loadPnms("DrawIndex.ms.pnms");
        std::vector<std::byte> pixelBytes = loadPnms("DrawIndex.ps.pnms");
        rachis::ShaderModuleHandle vertexShader{};
        rachis::ShaderModuleHandle meshShader{};
        rachis::ShaderModuleHandle pixelShader{};
        rachis::PipelineLayoutHandle layout{};
        rachis::PipelineHandle pipeline{};
        rachis::PipelineHandle meshPipeline{};
        rachis::BufferHandle vertices{};
        rachis::BufferHandle indices{};

        explicit DrawIndexPipelines(rachis::Device& owner) : device(owner)
        {
            REQUIRE_FALSE(vertexBytes.empty());
            REQUIRE_FALSE(meshBytes.empty());
            REQUIRE_FALSE(pixelBytes.empty());
            auto vertex = device.createShaderModule({
                .stage = rachis::ShaderStage::vertex,
                .bytecode = vertexBytes,
                .entryPoint = "drawIndexVertex",
                .shaderInterface = draw_index::vertexInterface(true),
            });
            auto pixel = device.createShaderModule({
                .stage = rachis::ShaderStage::pixel,
                .bytecode = pixelBytes,
                .entryPoint = "recordPixel",
            });
            REQUIRE(vertex.has_value());
            REQUIRE(pixel.has_value());
            vertexShader = *vertex;
            pixelShader = *pixel;
            auto created = device.createPipelineLayout({.drawIndex = true});
            REQUIRE(created.has_value());
            layout = *created;
            const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
            const std::array attributes{
                rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .shaderLocation = 0},
            };
            const std::array vertexLayouts{
                rachis::VertexBufferLayout{.stride = 2 * sizeof(float), .attributes = attributes},
            };
            auto graphics = device.createGraphicsPipeline({
                .layout = layout,
                .vertexShader = vertexShader,
                .pixelShader = pixelShader,
                .vertexBuffers = vertexLayouts,
                .cullMode = rachis::CullMode::none,
                .colorTargets = colorTargets,
            });
            REQUIRE(graphics.has_value());
            pipeline = *graphics;
            if (rachis::test::requireCapabilities(device).meshShaders)
            {
                auto mesh = device.createShaderModule({
                    .stage = rachis::ShaderStage::mesh,
                    .bytecode = meshBytes,
                    .entryPoint = "drawIndexMesh",
                    .shaderInterface = draw_index::meshInterface(),
                });
                REQUIRE(mesh.has_value());
                meshShader = *mesh;
                auto meshGraphics = device.createGraphicsPipeline({
                    .layout = layout,
                    .pixelShader = pixelShader,
                    .meshShader = meshShader,
                    .cullMode = rachis::CullMode::none,
                    .colorTargets = colorTargets,
                });
                REQUIRE(meshGraphics.has_value());
                meshPipeline = *meshGraphics;
            }
            const auto quads = draw_index::quadVertices();
            vertices = draw_index::uploadBuffer(
                device, rachis::BufferUsage::vertex, std::span<const std::array<float, 2>>(quads));
            const std::array<std::uint16_t, draw_index::verticesPerQuad> quadIndices{0, 1, 2, 3, 4, 5};
            indices = draw_index::uploadBuffer(
                device, rachis::BufferUsage::index, std::span<const std::uint16_t>(quadIndices));
        }

        DrawIndexPipelines(const DrawIndexPipelines&) = delete;
        DrawIndexPipelines& operator=(const DrawIndexPipelines&) = delete;
        DrawIndexPipelines(DrawIndexPipelines&&) = delete;
        DrawIndexPipelines& operator=(DrawIndexPipelines&&) = delete;

        ~DrawIndexPipelines()
        {
            CHECK(device.destroyBuffer(indices).has_value());
            CHECK(device.destroyBuffer(vertices).has_value());
            if (meshPipeline.isValid())
            {
                CHECK(device.destroyPipeline(meshPipeline).has_value());
                CHECK(device.destroyShaderModule(meshShader).has_value());
            }
            CHECK(device.destroyPipeline(pipeline).has_value());
            CHECK(device.destroyPipelineLayout(layout).has_value());
            CHECK(device.destroyShaderModule(pixelShader).has_value());
            CHECK(device.destroyShaderModule(vertexShader).has_value());
        }
    };

} // namespace

TEST_CASE("Metal shaders read each multi-draw record's draw index", "[size:large][rhi][metal][indirect][draw-index]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).shaderDrawIndex);

    const auto vertex = loadPnms("DrawIndex.vs.pnms");
    const auto plainVertex = loadPnms("DrawIndex.plain.vs.pnms");
    const auto mesh = loadPnms("DrawIndex.ms.pnms");
    const auto pixel = loadPnms("DrawIndex.ps.pnms");
    REQUIRE_FALSE(vertex.empty());
    REQUIRE_FALSE(plainVertex.empty());
    REQUIRE_FALSE(mesh.empty());
    REQUIRE_FALSE(pixel.empty());

    rachis::test::draw_index::runContract(device,
                                          {
                                              .vertex = vertex,
                                              .plainVertex = plainVertex,
                                              .mesh = mesh,
                                              .pixel = pixel,
                                          });
}

TEST_CASE("Metal validates the draw-index prefix of indirect calls whose scissor is empty",
          "[size:large][rhi][metal][indirect][draw-index]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).shaderDrawIndex);
    const DrawIndexPipelines pipelines(device);

    const std::array<rachis::DrawIndirectArguments, 2> drawRecords{};
    const std::array<rachis::DrawIndexedIndirectArguments, 2> indexedRecords{};
    const std::array<rachis::DispatchIndirectArguments, 2> meshRecords{};
    const auto drawArguments = draw_index::uploadBuffer(
        device, rachis::BufferUsage::indirect, std::span<const rachis::DrawIndirectArguments>(drawRecords));
    const auto indexedArguments = draw_index::uploadBuffer(
        device, rachis::BufferUsage::indirect, std::span<const rachis::DrawIndexedIndirectArguments>(indexedRecords));
    const auto meshArguments = draw_index::uploadBuffer(
        device, rachis::BufferUsage::indirect, std::span<const rachis::DispatchIndirectArguments>(meshRecords));
    const rachis::ScissorRect outside{
        .x = static_cast<std::int32_t>(draw_index::targetWidth) + 16,
        .y = static_cast<std::int32_t>(draw_index::targetHeight) + 16,
        .width = 8,
        .height = 8,
    };

    draw_index::Target target(device);
    const auto columns = target.render([&](rachis::CommandBuffer& command) {
        REQUIRE(device.bindGraphicsPipeline(command, pipelines.pipeline).has_value());
        REQUIRE(device.bindVertexBuffer(command, 0, pipelines.vertices).has_value());
        REQUIRE(device.bindIndexBuffer(command, pipelines.indices, rachis::IndexFormat::uint16).has_value());
        target.setViewportAndScissor(command);
        REQUIRE(device.setScissor(command, outside).has_value());
        draw_index::checkRejected(device.drawIndirect(command, {.argumentBuffer = drawArguments, .maxDrawCount = 2}),
                                  rachis::ErrorCode::invalidArgument);
        draw_index::checkRejected(
            device.drawIndexedIndirect(command, {.argumentBuffer = indexedArguments, .maxDrawCount = 2}),
            rachis::ErrorCode::invalidArgument);
        CHECK(device.drawIndirect(command, {.argumentBuffer = drawArguments, .maxDrawCount = 1}).has_value());
        if (pipelines.meshPipeline.isValid())
        {
            REQUIRE(device.bindGraphicsPipeline(command, pipelines.meshPipeline).has_value());
            REQUIRE(device.setScissor(command, outside).has_value());
            draw_index::checkRejected(
                device.dispatchMeshIndirect(command, {.argumentBuffer = meshArguments, .maxDispatchCount = 2}),
                rachis::ErrorCode::invalidArgument);
        }
    });
    draw_index::checkColumns(columns, {});

    REQUIRE(device.destroyBuffer(meshArguments).has_value());
    REQUIRE(device.destroyBuffer(indexedArguments).has_value());
    REQUIRE(device.destroyBuffer(drawArguments).has_value());
}

TEST_CASE("Metal records no draw and keeps draw index zero when a multi-draw cannot grow its draw-index table",
          "[size:large][rhi][metal][indirect][draw-index]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).shaderDrawIndex);
    const DrawIndexPipelines pipelines(device);

    constexpr std::uint32_t recordCount = 1025;
    std::vector<draw_index::DrawRecord> records(recordCount);
    for (std::uint32_t record = 0; record < recordCount; ++record)
    {
        records[record] = draw_index::DrawRecord{
            .drawIndex = record,
            .arguments = {.vertexCount = draw_index::verticesPerQuad,
                          .instanceCount = 1,
                          .firstVertex = draw_index::verticesPerQuad},
        };
    }
    const auto arguments = draw_index::uploadBuffer(
        device, rachis::BufferUsage::indirect, std::span<const draw_index::DrawRecord>(records));

    draw_index::Target target(device);
    const auto failed = target.render([&](rachis::CommandBuffer& command) {
        REQUIRE(device.bindGraphicsPipeline(command, pipelines.pipeline).has_value());
        REQUIRE(device.bindVertexBuffer(command, 0, pipelines.vertices).has_value());
        target.setViewportAndScissor(command);
        rachis::metal::setFailNextDrawIndexTableGrowth(true);
        const auto drawn = device.drawIndirect(
            command, {.argumentBuffer = arguments, .maxDrawCount = recordCount, .drawIndexPrefix = true});
        rachis::metal::setFailNextDrawIndexTableGrowth(false);
        draw_index::checkRejected(drawn, rachis::ErrorCode::outOfMemory);
        REQUIRE(device.draw(command, draw_index::verticesPerQuad, 1, 0, 0).has_value());
    });
    draw_index::checkColumns(failed,
                             {draw_index::Color::red,
                              draw_index::Color::clear,
                              draw_index::Color::clear,
                              draw_index::Color::clear,
                              draw_index::Color::clear});

    const auto retried = target.render([&](rachis::CommandBuffer& command) {
        REQUIRE(device.bindGraphicsPipeline(command, pipelines.pipeline).has_value());
        REQUIRE(device.bindVertexBuffer(command, 0, pipelines.vertices).has_value());
        target.setViewportAndScissor(command);
        REQUIRE(device
                    .drawIndirect(command,
                                  {.argumentBuffer = arguments, .maxDrawCount = recordCount, .drawIndexPrefix = true})
                    .has_value());
        REQUIRE(device.draw(command, draw_index::verticesPerQuad, 1, 0, 0).has_value());
    });
    draw_index::checkColumns(retried,
                             {draw_index::Color::red,
                              draw_index::Color::magenta,
                              draw_index::Color::clear,
                              draw_index::Color::clear,
                              draw_index::Color::clear});

    REQUIRE(device.destroyBuffer(arguments).has_value());
}
