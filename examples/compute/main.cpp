#include "ExampleSupport.h"
#include "SquareShader.h"

#include <Rachis/Device.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t valueCount = 1024;
    constexpr std::uint64_t bufferSize = valueCount * sizeof(std::uint32_t);

    namespace square = rachis_shaders::reflection::SquareShader;

} // namespace

int main(int argc, char** argv)
{
    using example::require;

    const auto options = example::parseOptions(argc, argv, "rachis-example-compute");
    if (!options)
    {
        return EXIT_FAILURE;
    }
    auto device = example::createDevice(options->device);
    const auto backend = device.adapterInfo()->backend;

    auto shader = require(device.createShaderModule(square::moduleDesc(backend, "Square")), "createShaderModule");
    const std::uint32_t threadsPerGroup = square::shaderInterface(backend).workgroupSize->x;

    auto bindingLayout =
        require(device.createBindingLayout({.entries = square::bindingLayout(backend, 0), .label = "Square bindings"}),
                "createBindingLayout");
    const std::array bindingLayouts{bindingLayout};
    auto pipelineLayout =
        require(device.createPipelineLayout({.bindingLayouts = bindingLayouts}), "createPipelineLayout");
    auto pipeline = require(device.createComputePipeline({
                                .layout = pipelineLayout,
                                .computeShader = shader,
                                .label = "Square",
                            }),
                            "createComputePipeline");

    auto values = require(device.createBuffer({
                              .size = bufferSize,
                              .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource |
                                       rachis::BufferUsage::copyDestination,
                              .memoryType = rachis::MemoryType::deviceLocal,
                              .label = "Values",
                          }),
                          "createBuffer(values)");
    auto upload = require(device.createBuffer({
                              .size = bufferSize,
                              .usage = rachis::BufferUsage::copySource,
                              .memoryType = rachis::MemoryType::upload,
                              .label = "Upload",
                          }),
                          "createBuffer(upload)");
    auto readback = require(device.createBuffer({
                                .size = bufferSize,
                                .usage = rachis::BufferUsage::copyDestination,
                                .memoryType = rachis::MemoryType::readback,
                                .label = "Readback",
                            }),
                            "createBuffer(readback)");

    std::vector<std::uint32_t> input(valueCount);
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        input[index] = index;
    }
    require(device.writeBuffer(upload, 0, std::as_bytes(std::span{input})), "writeBuffer");

    auto bindingPool =
        require(device.createBindingPool({.maxBindingSets = 1, .sizes = square::bindingPoolSizes(backend, 0)}),
                "createBindingPool");
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = values,
                    .size = bufferSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = require(device.createBindingSet({
                                  .layout = bindingLayout,
                                  .writes = writes,
                                  .pool = bindingPool,
                              }),
                              "createBindingSet");

    auto commandPool = require(device.createCommandPool({
                                   .queueType = rachis::QueueType::compute,
                                   .maxCommandBuffers = 1,
                               }),
                               "createCommandPool");
    auto command = require(device.acquireCommandBuffer(commandPool), "acquireCommandBuffer");

    require(device.copyBuffer(command, {.source = upload, .destination = values, .size = bufferSize}),
            "copyBuffer(upload)");
    require(device.bufferBarrier(command,
                                 {
                                     .buffer = values,
                                     .stagesBefore = rachis::PipelineStage::copy,
                                     .stagesAfter = rachis::PipelineStage::computeShader,
                                     .accessBefore = rachis::Access::copyDestination,
                                     .accessAfter = rachis::Access::unorderedAccess,
                                 }),
            "bufferBarrier(copy to compute)");
    require(device.bindComputePipeline(command, pipeline), "bindComputePipeline");
    require(device.bindComputeBindingSet(command, 0, bindingSet), "bindComputeBindingSet");
    require(device.dispatch(command, valueCount / threadsPerGroup), "dispatch");
    require(device.bufferBarrier(command,
                                 {
                                     .buffer = values,
                                     .stagesBefore = rachis::PipelineStage::computeShader,
                                     .stagesAfter = rachis::PipelineStage::copy,
                                     .accessBefore = rachis::Access::unorderedAccess,
                                     .accessAfter = rachis::Access::copySource,
                                 }),
            "bufferBarrier(compute to copy)");
    require(device.copyBuffer(command, {.source = values, .destination = readback, .size = bufferSize}),
            "copyBuffer(readback)");

    const auto submitted = require(device.submit(std::move(command)), "submit");
    require(device.wait(submitted), "wait");

    std::vector<std::uint32_t> output(valueCount);
    require(device.readBuffer(readback, 0, std::as_writable_bytes(std::span{output})), "readBuffer");
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        if (output[index] != index * index)
        {
            std::fprintf(stderr,
                         "values[%u] is %u; expected %u.\n",
                         static_cast<unsigned int>(index),
                         static_cast<unsigned int>(output[index]),
                         static_cast<unsigned int>(index * index));
            return EXIT_FAILURE;
        }
    }
    std::printf("Squared %u values: values[%u] = %u\n",
                static_cast<unsigned int>(valueCount),
                static_cast<unsigned int>(valueCount - 1),
                static_cast<unsigned int>(output[valueCount - 1]));

    require(device.destroyCommandPool(commandPool), "destroyCommandPool");
    require(device.destroyBindingSet(bindingSet), "destroyBindingSet");
    require(device.destroyBindingPool(bindingPool), "destroyBindingPool");
    require(device.destroyBuffer(readback), "destroyBuffer(readback)");
    require(device.destroyBuffer(upload), "destroyBuffer(upload)");
    require(device.destroyBuffer(values), "destroyBuffer(values)");
    require(device.destroyPipeline(pipeline), "destroyPipeline");
    require(device.destroyPipelineLayout(pipelineLayout), "destroyPipelineLayout");
    require(device.destroyBindingLayout(bindingLayout), "destroyBindingLayout");
    require(device.destroyShaderModule(shader), "destroyShaderModule");
    return example::finish(device);
}
