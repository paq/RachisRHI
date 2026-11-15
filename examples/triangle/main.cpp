#include "ExampleSupport.h"
#include "TrianglePixel.h"
#include "TriangleVertex.h"

#include <Rachis/Device.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{

    constexpr std::uint32_t width = 256;
    constexpr std::uint32_t height = 256;
    constexpr std::uint32_t bytesPerPixel = 4;
    constexpr rachis::ClearColor background{.red = 0.1F, .green = 0.1F, .blue = 0.1F, .alpha = 1.0F};

    struct Pixel final
    {
        std::uint8_t red = 0;
        std::uint8_t green = 0;
        std::uint8_t blue = 0;
        std::uint8_t alpha = 0;
    };

    [[nodiscard]] Pixel pixelAt(const std::vector<std::byte>& rows,
                                std::uint32_t bytesPerRow,
                                std::uint32_t x,
                                std::uint32_t y)
    {
        const auto* texel =
            rows.data() + static_cast<std::size_t>(y) * bytesPerRow + static_cast<std::size_t>(x) * bytesPerPixel;
        return {
            std::to_integer<std::uint8_t>(texel[0]),
            std::to_integer<std::uint8_t>(texel[1]),
            std::to_integer<std::uint8_t>(texel[2]),
            std::to_integer<std::uint8_t>(texel[3]),
        };
    }

    [[nodiscard]] bool writePpm(const char* path, const std::vector<std::byte>& rows, std::uint32_t bytesPerRow)
    {
        std::FILE* file = std::fopen(path, "wb");
        if (file == nullptr)
        {
            return false;
        }
        std::fprintf(file, "P6\n%u %u\n255\n", static_cast<unsigned int>(width), static_cast<unsigned int>(height));
        std::vector<std::uint8_t> row(static_cast<std::size_t>(width) * 3);
        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                const auto pixel = pixelAt(rows, bytesPerRow, x, y);
                row[static_cast<std::size_t>(x) * 3 + 0] = pixel.red;
                row[static_cast<std::size_t>(x) * 3 + 1] = pixel.green;
                row[static_cast<std::size_t>(x) * 3 + 2] = pixel.blue;
            }
            std::fwrite(row.data(), 1, row.size(), file);
        }
        return std::fclose(file) == 0;
    }

    [[nodiscard]] bool checkImage(const std::vector<std::byte>& rows, std::uint32_t bytesPerRow)
    {
        const auto corner = pixelAt(rows, bytesPerRow, 4, 4);
        const auto top = pixelAt(rows, bytesPerRow, width / 2, 40);
        const auto bottomLeft = pixelAt(rows, bytesPerRow, 40, height - 36);
        const auto bottomRight = pixelAt(rows, bytesPerRow, width - 40, height - 36);
        const bool backgroundOk = corner.red >= 24 && corner.red <= 27 && corner.green == corner.red &&
                                  corner.blue == corner.red && corner.alpha == 255;
        const bool redOk = top.red > 200 && top.green < 60 && top.blue < 60;
        const bool blueOk = bottomLeft.blue > 200 && bottomLeft.red < 60 && bottomLeft.green < 60;
        const bool greenOk = bottomRight.green > 200 && bottomRight.red < 60 && bottomRight.blue < 60;
        if (!backgroundOk || !redOk || !blueOk || !greenOk)
        {
            std::fprintf(
                stderr,
                "Unexpected pixels: corner %u,%u,%u,%u top %u,%u,%u bottom-left %u,%u,%u bottom-right %u,%u,%u\n",
                corner.red,
                corner.green,
                corner.blue,
                corner.alpha,
                top.red,
                top.green,
                top.blue,
                bottomLeft.red,
                bottomLeft.green,
                bottomLeft.blue,
                bottomRight.red,
                bottomRight.green,
                bottomRight.blue);
            return false;
        }
        return true;
    }

} // namespace

int main(int argc, char** argv)
{
    using example::require;

    const auto options = example::parseOptions(argc, argv, "rachis-example-triangle");
    if (!options)
    {
        return EXIT_FAILURE;
    }
    const char* outputPath = options->output != nullptr ? options->output : "triangle.ppm";
    auto device = example::createDevice(options->device);
    const auto backend = device.adapterInfo()->backend;

    auto vertexShader = require(
        device.createShaderModule(rachis_shaders::reflection::TriangleVertex::moduleDesc(backend, "Triangle vertex")),
        "createShaderModule(vertex)");
    auto pixelShader = require(
        device.createShaderModule(rachis_shaders::reflection::TrianglePixel::moduleDesc(backend, "Triangle pixel")),
        "createShaderModule(pixel)");
    auto pipelineLayout = require(device.createPipelineLayout({}), "createPipelineLayout");
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = require(device.createGraphicsPipeline({
                                .layout = pipelineLayout,
                                .vertexShader = vertexShader,
                                .pixelShader = pixelShader,
                                .cullMode = rachis::CullMode::none,
                                .colorTargets = colorTargets,
                                .label = "Triangle",
                            }),
                            "createGraphicsPipeline");

    auto target = require(device.createTexture({
                              .width = width,
                              .height = height,
                              .format = rachis::Format::rgba8Unorm,
                              .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
                              .label = "Triangle target",
                          }),
                          "createTexture");
    auto targetView = require(device.createTextureView({
                                  .texture = target,
                                  .usage = rachis::TextureViewUsage::renderTarget,
                              }),
                              "createTextureView");

    const rachis::BufferTextureCopyRegion region{.extent = {.width = width, .height = height, .depth = 1}};
    const auto layout = require(device.queryTextureTransferLayout(target, region), "queryTextureTransferLayout");
    auto readback = require(device.createBuffer({
                                .size = layout.requiredBufferSize,
                                .usage = rachis::BufferUsage::copyDestination,
                                .memoryType = rachis::MemoryType::readback,
                                .label = "Triangle readback",
                            }),
                            "createBuffer");

    auto commandPool = require(device.createCommandPool({
                                   .queueType = rachis::QueueType::graphics,
                                   .maxCommandBuffers = 1,
                               }),
                               "createCommandPool");
    auto command = require(device.acquireCommandBuffer(commandPool), "acquireCommandBuffer");

    require(device.textureBarrier(command,
                                  {
                                      .texture = target,
                                      .stagesBefore = rachis::PipelineStage::none,
                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                      .accessBefore = rachis::Access::none,
                                      .accessAfter = rachis::Access::renderTarget,
                                      .layoutBefore = rachis::TextureLayout::undefined,
                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                  }),
            "textureBarrier(to render target)");
    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = background,
        },
    };
    require(device.beginRendering(command, {.colorAttachments = colorAttachments}), "beginRendering");
    require(device.bindGraphicsPipeline(command, pipeline), "bindGraphicsPipeline");
    require(device.setViewport(command, {.width = static_cast<float>(width), .height = static_cast<float>(height)}),
            "setViewport");
    require(device.setScissor(command, {.width = width, .height = height}), "setScissor");
    require(device.draw(command, 3), "draw");
    require(device.endRendering(command), "endRendering");

    require(device.textureBarrier(command,
                                  {
                                      .texture = target,
                                      .stagesBefore = rachis::PipelineStage::renderTarget,
                                      .stagesAfter = rachis::PipelineStage::copy,
                                      .accessBefore = rachis::Access::renderTarget,
                                      .accessAfter = rachis::Access::copySource,
                                      .layoutBefore = rachis::TextureLayout::renderTarget,
                                      .layoutAfter = rachis::TextureLayout::copySource,
                                  }),
            "textureBarrier(to copy source)");
    require(device.copyTextureToBuffer(command,
                                       {
                                           .buffer = readback,
                                           .bytesPerRow = layout.bytesPerRow,
                                           .rowsPerImage = layout.rowsPerImage,
                                           .texture = target,
                                           .extent = region.extent,
                                       }),
            "copyTextureToBuffer");

    const auto submitted = require(device.submit(std::move(command)), "submit");
    require(device.wait(submitted), "wait");

    std::vector<std::byte> rows(static_cast<std::size_t>(layout.requiredBufferSize));
    require(device.readBuffer(readback, 0, rows), "readBuffer");
    if (!checkImage(rows, layout.bytesPerRow))
    {
        return EXIT_FAILURE;
    }
    if (!writePpm(outputPath, rows, layout.bytesPerRow))
    {
        std::fprintf(stderr, "Could not write %s.\n", outputPath);
        return EXIT_FAILURE;
    }
    std::printf(
        "Wrote %ux%u pixels to %s\n", static_cast<unsigned int>(width), static_cast<unsigned int>(height), outputPath);

    require(device.destroyCommandPool(commandPool), "destroyCommandPool");
    require(device.destroyBuffer(readback), "destroyBuffer");
    require(device.destroyTextureView(targetView), "destroyTextureView");
    require(device.destroyTexture(target), "destroyTexture");
    require(device.destroyPipeline(pipeline), "destroyPipeline");
    require(device.destroyPipelineLayout(pipelineLayout), "destroyPipelineLayout");
    require(device.destroyShaderModule(pixelShader), "destroyShaderModule(pixel)");
    require(device.destroyShaderModule(vertexShader), "destroyShaderModule(vertex)");
    return example::finish(device);
}
