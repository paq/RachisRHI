#pragma once

#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

namespace rachis::test::draw_index
{

    struct Bytecode final
    {
        std::span<const std::byte> vertex;
        std::span<const std::byte> plainVertex;
        std::span<const std::byte> mesh;
        std::span<const std::byte> pixel;
    };

    inline constexpr std::uint32_t columnCount = 5;
    inline constexpr std::uint32_t columnWidth = 16;
    inline constexpr std::uint32_t targetWidth = columnCount * columnWidth;
    inline constexpr std::uint32_t targetHeight = 16;
    inline constexpr std::uint32_t rowPitch = 512;
    inline constexpr std::uint64_t readbackSize = std::uint64_t{rowPitch} * targetHeight;
    inline constexpr std::uint32_t recordCount = 4;
    inline constexpr std::uint32_t drawnRecordCount = 3;
    inline constexpr std::uint32_t verticesPerQuad = 6;

    enum class Color : std::uint8_t
    {
        clear,
        red,
        green,
        blue,
        yellow,
        magenta,
        other,
    };

    using Columns = std::array<Color, columnCount>;

    struct DrawRecord final
    {
        std::uint32_t drawIndex = 0;
        DrawIndirectArguments arguments = {};
    };

    struct IndexedDrawRecord final
    {
        std::uint32_t drawIndex = 0;
        DrawIndexedIndirectArguments arguments = {};
        std::array<std::uint32_t, 2> padding = {};
    };

    struct MeshRecord final
    {
        std::uint32_t drawIndex = 0;
        DispatchIndirectArguments arguments = {};
    };

    static_assert(sizeof(DrawRecord) == sizeof(DrawIndirectArguments) + drawIndexPrefixSize);
    static_assert(sizeof(IndexedDrawRecord) == 32);
    static_assert(sizeof(MeshRecord) == sizeof(DispatchIndirectArguments) + drawIndexPrefixSize);

    [[nodiscard]] inline Color classify(const std::byte* pixel) noexcept
    {
        const auto high = [](std::byte value) { return std::to_integer<std::uint8_t>(value) > 200; };
        const auto low = [](std::byte value) { return std::to_integer<std::uint8_t>(value) < 40; };
        const bool r = high(pixel[0]);
        const bool g = high(pixel[1]);
        const bool b = high(pixel[2]);
        const bool nr = low(pixel[0]);
        const bool ng = low(pixel[1]);
        const bool nb = low(pixel[2]);
        if (nr && ng && nb)
        {
            return Color::clear;
        }
        if (r && ng && nb)
        {
            return Color::red;
        }
        if (nr && g && nb)
        {
            return Color::green;
        }
        if (nr && ng && b)
        {
            return Color::blue;
        }
        if (r && g && nb)
        {
            return Color::yellow;
        }
        if (r && ng && b)
        {
            return Color::magenta;
        }
        return Color::other;
    }

    [[nodiscard]] inline std::string_view colorName(Color color) noexcept
    {
        switch (color)
        {
        case Color::clear:
            return "clear";
        case Color::red:
            return "red";
        case Color::green:
            return "green";
        case Color::blue:
            return "blue";
        case Color::yellow:
            return "yellow";
        case Color::magenta:
            return "magenta";
        case Color::other:
            break;
        }
        return "other";
    }

    inline void checkColumns(const Columns& actual, const Columns& expected)
    {
        for (std::uint32_t column = 0; column < columnCount; ++column)
        {
            INFO("column " << column << " is " << colorName(actual[column]) << ", expected "
                           << colorName(expected[column]));
            CHECK(actual[column] == expected[column]);
        }
    }

    inline void checkRejected(const Status& status, ErrorCode code)
    {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == code);
    }

    template<typename T>
    [[nodiscard]] BufferHandle uploadBuffer(Device& device, BufferUsage usage, std::span<const T> data)
    {
        auto buffer = device.createBuffer(BufferDesc{
            .size = data.size_bytes(),
            .usage = usage,
            .memoryType = MemoryType::upload,
        });
        REQUIRE(buffer.has_value());
        auto mapped = device.mapBuffer(*buffer);
        REQUIRE(mapped.has_value());
        std::memcpy(mapped->data, data.data(), data.size_bytes());
        REQUIRE(device.unmapBuffer(*buffer).has_value());
        return *buffer;
    }

    [[nodiscard]] inline std::vector<std::array<float, 2>> quadVertices()
    {
        std::vector<std::array<float, 2>> vertices;
        for (std::uint32_t column = 0; column < columnCount; ++column)
        {
            const float left = -1.0F + 0.4F * static_cast<float>(column) + 0.05F;
            const float right = left + 0.3F;
            for (const auto& vertex : {std::array{left, -0.8F},
                                       std::array{left, 0.8F},
                                       std::array{right, -0.8F},
                                       std::array{right, -0.8F},
                                       std::array{left, 0.8F},
                                       std::array{right, 0.8F}})
            {
                vertices.push_back(vertex);
            }
        }
        return vertices;
    }

    class Target final
    {
    public:
        explicit Target(Device& device) : device_(&device)
        {
            auto texture = device.createTexture(TextureDesc{
                .width = targetWidth,
                .height = targetHeight,
                .format = Format::rgba8Unorm,
                .usage = TextureUsage::copySource | TextureUsage::renderTarget,
            });
            REQUIRE(texture.has_value());
            texture_ = *texture;
            auto view = device.createTextureView(TextureViewDesc{
                .texture = texture_,
                .usage = TextureViewUsage::renderTarget,
            });
            REQUIRE(view.has_value());
            view_ = *view;
            auto readback = device.createBuffer(BufferDesc{
                .size = readbackSize,
                .usage = BufferUsage::copyDestination,
                .memoryType = MemoryType::readback,
            });
            REQUIRE(readback.has_value());
            readback_ = *readback;
        }

        ~Target()
        {
            static_cast<void>(device_->destroyBuffer(readback_));
            static_cast<void>(device_->destroyTextureView(view_));
            static_cast<void>(device_->destroyTexture(texture_));
        }

        Target(const Target&) = delete;
        Target& operator=(const Target&) = delete;
        Target(Target&&) = delete;
        Target& operator=(Target&&) = delete;

        template<typename Record>
        [[nodiscard]] Columns render(Record&& record)
        {
            auto command = acquireCommandBuffer(*device_, QueueType::graphics);
            REQUIRE(command.has_value());
            REQUIRE(device_
                        ->textureBarrier(
                            *command,
                            TextureBarrier{
                                .texture = texture_,
                                .stagesBefore = firstPass_ ? PipelineStage::all : PipelineStage::copy,
                                .stagesAfter = PipelineStage::renderTarget,
                                .accessBefore = firstPass_ ? Access::none : Access::copySource,
                                .accessAfter = Access::renderTarget,
                                .layoutBefore = firstPass_ ? TextureLayout::undefined : TextureLayout::copySource,
                                .layoutAfter = TextureLayout::renderTarget,
                            })
                        .has_value());
            const std::array attachments{
                ColorAttachment{
                    .view = view_,
                    .loadOperation = LoadOperation::clear,
                    .clearValue = {},
                },
            };
            REQUIRE(device_->beginRendering(*command, RenderingDesc{.colorAttachments = attachments}).has_value());
            record(*command);
            REQUIRE(device_->endRendering(*command).has_value());
            REQUIRE(device_
                        ->textureBarrier(*command,
                                         TextureBarrier{
                                             .texture = texture_,
                                             .stagesBefore = PipelineStage::renderTarget,
                                             .stagesAfter = PipelineStage::copy,
                                             .accessBefore = Access::renderTarget,
                                             .accessAfter = Access::copySource,
                                             .layoutBefore = TextureLayout::renderTarget,
                                             .layoutAfter = TextureLayout::copySource,
                                         })
                        .has_value());
            REQUIRE(device_
                        ->copyTextureToBuffer(*command,
                                              BufferTextureCopy{
                                                  .buffer = readback_,
                                                  .bytesPerRow = rowPitch,
                                                  .rowsPerImage = targetHeight,
                                                  .texture = texture_,
                                                  .extent = {targetWidth, targetHeight, 1},
                                              })
                        .has_value());
            auto submitted = device_->submit(std::move(*command));
            REQUIRE(submitted.has_value());
            REQUIRE(device_->wait(*submitted).has_value());
            firstPass_ = false;

            auto mapped = device_->mapBuffer(readback_);
            REQUIRE(mapped.has_value());
            Columns columns{};
            for (std::uint32_t column = 0; column < columnCount; ++column)
            {
                const auto offset =
                    std::size_t{targetHeight / 2} * rowPitch + std::size_t{column * columnWidth + columnWidth / 2} * 4U;
                columns[column] = classify(mapped->data + offset);
            }
            REQUIRE(device_->unmapBuffer(readback_).has_value());
            return columns;
        }

        void setViewportAndScissor(CommandBuffer& command) const
        {
            REQUIRE(device_
                        ->setViewport(command,
                                      Viewport{
                                          .width = static_cast<float>(targetWidth),
                                          .height = static_cast<float>(targetHeight),
                                      })
                        .has_value());
            REQUIRE(
                device_->setScissor(command, ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
        }

    private:
        Device* device_ = nullptr;
        TextureHandle texture_ = {};
        TextureViewHandle view_ = {};
        BufferHandle readback_ = {};
        bool firstPass_ = true;
    };

    [[nodiscard]] inline ShaderInterfaceDesc vertexInterface(bool drawIndex) noexcept
    {
        static constexpr std::array locations{0U};
        return ShaderInterfaceDesc{
            .vertexInputLocations = locations,
            .drawIndex = drawIndex,
        };
    }

    [[nodiscard]] inline ShaderInterfaceDesc meshInterface() noexcept
    {
        return ShaderInterfaceDesc{
            .workgroupSize = ShaderWorkgroupSize{1, 1, 1},
            .drawIndex = true,
        };
    }

    inline void checkModuleValidation(Device& device, const Bytecode& bytecode)
    {
        const auto undeclared = device.createShaderModule(ShaderModuleDesc{
            .stage = ShaderStage::vertex,
            .bytecode = bytecode.vertex,
            .entryPoint = "drawIndexVertex",
            .shaderInterface = vertexInterface(false),
        });
        REQUIRE_FALSE(undeclared.has_value());
        CHECK(undeclared.error().code == ErrorCode::invalidArgument);

        const auto unread = device.createShaderModule(ShaderModuleDesc{
            .stage = ShaderStage::vertex,
            .bytecode = bytecode.plainVertex,
            .entryPoint = "plainVertex",
            .shaderInterface = vertexInterface(true),
        });
        REQUIRE_FALSE(unread.has_value());
        CHECK(unread.error().code == ErrorCode::invalidArgument);

        const auto pixel = device.createShaderModule(ShaderModuleDesc{
            .stage = ShaderStage::pixel,
            .bytecode = bytecode.pixel,
            .entryPoint = "recordPixel",
            .shaderInterface = ShaderInterfaceDesc{.drawIndex = true},
        });
        REQUIRE_FALSE(pixel.has_value());
        CHECK(pixel.error().code == ErrorCode::invalidArgument);
    }

    inline void runContract(Device& device, const Bytecode& bytecode)
    {
        const auto capabilities = requireCapabilities(device);
        REQUIRE(capabilities.shaderDrawIndex);
        REQUIRE(capabilities.multiDrawIndirect);
        const bool counted = capabilities.drawIndirectCount;
        const std::uint32_t maxRecords = counted ? recordCount : drawnRecordCount;

        checkModuleValidation(device, bytecode);

        auto vertexShader = device.createShaderModule(ShaderModuleDesc{
            .stage = ShaderStage::vertex,
            .bytecode = bytecode.vertex,
            .entryPoint = "drawIndexVertex",
            .shaderInterface = vertexInterface(true),
        });
        auto plainVertexShader = device.createShaderModule(ShaderModuleDesc{
            .stage = ShaderStage::vertex,
            .bytecode = bytecode.plainVertex,
            .entryPoint = "plainVertex",
            .shaderInterface = vertexInterface(false),
        });
        auto pixelShader = device.createShaderModule(ShaderModuleDesc{
            .stage = ShaderStage::pixel,
            .bytecode = bytecode.pixel,
            .entryPoint = "recordPixel",
        });
        REQUIRE(vertexShader.has_value());
        REQUIRE(plainVertexShader.has_value());
        REQUIRE(pixelShader.has_value());

        const auto snapshot = device.shaderModuleInterface(*vertexShader);
        REQUIRE(snapshot.has_value());
        CHECK(snapshot->drawIndex);
        const auto pixelSnapshot = device.shaderModuleInterface(*pixelShader);
        REQUIRE(pixelSnapshot.has_value());
        const std::array snapshots{*snapshot, *pixelSnapshot};
        const auto derived = derivePipelineLayout(snapshots);
        REQUIRE(derived.has_value());
        CHECK(derived->drawIndex);

        auto drawIndexLayout = device.createPipelineLayout(PipelineLayoutDesc{.drawIndex = true});
        auto plainLayout = device.createPipelineLayout(PipelineLayoutDesc{});
        REQUIRE(drawIndexLayout.has_value());
        REQUIRE(plainLayout.has_value());

        const std::array colorTargets{ColorTargetState{.format = Format::rgba8Unorm}};
        const std::array vertexAttributes{
            VertexAttribute{.format = VertexFormat::float32x2, .shaderLocation = 0},
        };
        const std::array vertexLayouts{
            VertexBufferLayout{.stride = 2 * sizeof(float), .attributes = vertexAttributes},
        };
        const GraphicsPipelineDesc drawIndexPipelineDesc{
            .layout = *drawIndexLayout,
            .vertexShader = *vertexShader,
            .pixelShader = *pixelShader,
            .vertexBuffers = vertexLayouts,
            .cullMode = CullMode::none,
            .colorTargets = colorTargets,
        };
        auto drawIndexPipeline = device.createGraphicsPipeline(drawIndexPipelineDesc);
        REQUIRE(drawIndexPipeline.has_value());

        auto missingLayout = drawIndexPipelineDesc;
        missingLayout.layout = *plainLayout;
        const auto rejectedPipeline = device.createGraphicsPipeline(missingLayout);
        REQUIRE_FALSE(rejectedPipeline.has_value());
        CHECK(rejectedPipeline.error().code == ErrorCode::validationError);

        auto plainPipelineDesc = drawIndexPipelineDesc;
        plainPipelineDesc.layout = *plainLayout;
        plainPipelineDesc.vertexShader = *plainVertexShader;
        auto plainPipeline = device.createGraphicsPipeline(plainPipelineDesc);
        REQUIRE(plainPipeline.has_value());

        const auto vertices = quadVertices();
        const auto vertexBuffer =
            uploadBuffer(device, BufferUsage::vertex, std::span<const std::array<float, 2>>(vertices));
        const std::array<std::uint16_t, verticesPerQuad> indices{0, 1, 2, 3, 4, 5};
        const auto indexBuffer = uploadBuffer(device, BufferUsage::index, std::span<const std::uint16_t>(indices));

        std::array<DrawRecord, recordCount> drawRecords{};
        std::array<IndexedDrawRecord, recordCount> indexedRecords{};
        std::array<MeshRecord, recordCount> meshRecords{};
        for (std::uint32_t record = 0; record < recordCount; ++record)
        {
            drawRecords[record] = DrawRecord{
                .drawIndex = record,
                .arguments = {.vertexCount = verticesPerQuad,
                              .instanceCount = 1,
                              .firstVertex = record * verticesPerQuad},
            };
            indexedRecords[record] = IndexedDrawRecord{
                .drawIndex = record,
                .arguments = {.indexCount = verticesPerQuad,
                              .instanceCount = 1,
                              .baseVertex = static_cast<std::int32_t>(record * verticesPerQuad)},
            };
            meshRecords[record] = MeshRecord{.drawIndex = record, .arguments = {1, 1, 1}};
        }
        const auto drawArguments =
            uploadBuffer(device, BufferUsage::indirect, std::span<const DrawRecord>(drawRecords));
        const auto indexedArguments =
            uploadBuffer(device, BufferUsage::indirect, std::span<const IndexedDrawRecord>(indexedRecords));
        const auto meshArguments =
            uploadBuffer(device, BufferUsage::indirect, std::span<const MeshRecord>(meshRecords));
        const std::array<DrawIndirectArguments, 2> emptyRecords{};
        const auto emptyArguments =
            uploadBuffer(device, BufferUsage::indirect, std::span<const DrawIndirectArguments>(emptyRecords));
        const std::array<std::uint32_t, 1> drawnCount{drawnRecordCount};
        const auto countBuffer =
            uploadBuffer(device, BufferUsage::indirect, std::span<const std::uint32_t>(drawnCount));
        const BufferHandle recordCountBuffer = counted ? countBuffer : BufferHandle{};

        Target target(device);
        const IndirectDrawDesc prefixedDraws{
            .argumentBuffer = drawArguments,
            .maxDrawCount = maxRecords,
            .countBuffer = recordCountBuffer,
            .drawIndexPrefix = true,
        };

        const auto drawColumns = target.render([&](CommandBuffer& command) {
            REQUIRE(device.bindGraphicsPipeline(command, *drawIndexPipeline).has_value());
            REQUIRE(device.bindVertexBuffer(command, 0, vertexBuffer).has_value());
            target.setViewportAndScissor(command);

            checkRejected(
                device.drawIndirect(command, IndirectDrawDesc{.argumentBuffer = emptyArguments, .maxDrawCount = 2}),
                ErrorCode::invalidArgument);
            CHECK(device.drawIndirect(command, IndirectDrawDesc{.argumentBuffer = emptyArguments, .maxDrawCount = 1})
                      .has_value());
            auto shortStride = prefixedDraws;
            shortStride.stride = sizeof(DrawIndirectArguments);
            checkRejected(device.drawIndirect(command, shortStride), ErrorCode::invalidArgument);
            auto unalignedStride = prefixedDraws;
            unalignedStride.stride = sizeof(DrawRecord) + 2;
            checkRejected(device.drawIndirect(command, unalignedStride), ErrorCode::invalidArgument);
            auto pastEnd = prefixedDraws;
            pastEnd.maxDrawCount = recordCount + 1;
            checkRejected(device.drawIndirect(command, pastEnd), ErrorCode::invalidArgument);

            REQUIRE(device.draw(command, verticesPerQuad, 1, 3 * verticesPerQuad, 0).has_value());
            REQUIRE(device.drawIndirect(command, prefixedDraws).has_value());
            REQUIRE(device.draw(command, verticesPerQuad, 1, 4 * verticesPerQuad, 0).has_value());
        });
        checkColumns(drawColumns, {Color::red, Color::green, Color::blue, Color::red, Color::red});

        const auto indexedColumns = target.render([&](CommandBuffer& command) {
            REQUIRE(device.bindGraphicsPipeline(command, *drawIndexPipeline).has_value());
            REQUIRE(device.bindVertexBuffer(command, 0, vertexBuffer).has_value());
            REQUIRE(device.bindIndexBuffer(command, indexBuffer, IndexFormat::uint16).has_value());
            target.setViewportAndScissor(command);
            const IndirectDrawDesc indexedDraws{
                .argumentBuffer = indexedArguments,
                .maxDrawCount = maxRecords,
                .stride = sizeof(IndexedDrawRecord),
                .countBuffer = recordCountBuffer,
                .drawIndexPrefix = true,
            };
            auto shortStride = indexedDraws;
            shortStride.stride = sizeof(DrawIndexedIndirectArguments);
            checkRejected(device.drawIndexedIndirect(command, shortStride), ErrorCode::invalidArgument);

            REQUIRE(device.drawIndexed(command, verticesPerQuad, 1, 0, 3 * verticesPerQuad, 0).has_value());
            REQUIRE(device.drawIndexedIndirect(command, indexedDraws).has_value());
            REQUIRE(device.drawIndexed(command, verticesPerQuad, 1, 0, 4 * verticesPerQuad, 0).has_value());
        });
        checkColumns(indexedColumns, {Color::red, Color::green, Color::blue, Color::red, Color::red});

        const auto plainColumns = target.render([&](CommandBuffer& command) {
            REQUIRE(device.bindGraphicsPipeline(command, *plainPipeline).has_value());
            REQUIRE(device.bindVertexBuffer(command, 0, vertexBuffer).has_value());
            target.setViewportAndScissor(command);
            CHECK(device.drawIndirect(command, IndirectDrawDesc{.argumentBuffer = emptyArguments, .maxDrawCount = 2})
                      .has_value());
            REQUIRE(device.drawIndirect(command, prefixedDraws).has_value());
        });
        checkColumns(plainColumns, {Color::magenta, Color::magenta, Color::magenta, Color::clear, Color::clear});

        PipelineHandle meshPipeline = {};
        ShaderModuleHandle meshShader = {};
        if (capabilities.meshShaders)
        {
            auto createdMesh = device.createShaderModule(ShaderModuleDesc{
                .stage = ShaderStage::mesh,
                .bytecode = bytecode.mesh,
                .entryPoint = "drawIndexMesh",
                .shaderInterface = meshInterface(),
            });
            REQUIRE(createdMesh.has_value());
            meshShader = *createdMesh;
            auto createdPipeline = device.createGraphicsPipeline(GraphicsPipelineDesc{
                .layout = *drawIndexLayout,
                .pixelShader = *pixelShader,
                .meshShader = meshShader,
                .cullMode = CullMode::none,
                .colorTargets = colorTargets,
            });
            REQUIRE(createdPipeline.has_value());
            meshPipeline = *createdPipeline;

            const auto meshColumns = target.render([&](CommandBuffer& command) {
                REQUIRE(device.bindGraphicsPipeline(command, meshPipeline).has_value());
                target.setViewportAndScissor(command);
                const IndirectMeshDispatchDesc prefixedDispatches{
                    .argumentBuffer = meshArguments,
                    .maxDispatchCount = maxRecords,
                    .countBuffer = recordCountBuffer,
                    .drawIndexPrefix = true,
                };
                checkRejected(
                    device.dispatchMeshIndirect(
                        command, IndirectMeshDispatchDesc{.argumentBuffer = meshArguments, .maxDispatchCount = 2}),
                    ErrorCode::invalidArgument);
                auto shortStride = prefixedDispatches;
                shortStride.stride = sizeof(DispatchIndirectArguments);
                checkRejected(device.dispatchMeshIndirect(command, shortStride), ErrorCode::invalidArgument);

                REQUIRE(device.dispatchMeshIndirect(command, prefixedDispatches).has_value());
                REQUIRE(device.dispatchMesh(command, 2, 1, 1).has_value());
            });
            checkColumns(meshColumns, {Color::red, Color::green, Color::blue, Color::clear, Color::red});
        }

        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);

        if (meshPipeline.isValid())
        {
            REQUIRE(device.destroyPipeline(meshPipeline).has_value());
            REQUIRE(device.destroyShaderModule(meshShader).has_value());
        }
        for (const auto buffer :
             {countBuffer, emptyArguments, meshArguments, indexedArguments, drawArguments, indexBuffer, vertexBuffer})
        {
            REQUIRE(device.destroyBuffer(buffer).has_value());
        }
        REQUIRE(device.destroyPipeline(*plainPipeline).has_value());
        REQUIRE(device.destroyPipeline(*drawIndexPipeline).has_value());
        REQUIRE(device.destroyPipelineLayout(*plainLayout).has_value());
        REQUIRE(device.destroyPipelineLayout(*drawIndexLayout).has_value());
        REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
        REQUIRE(device.destroyShaderModule(*plainVertexShader).has_value());
        REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    }

} // namespace rachis::test::draw_index
