#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include "Internal/VulkanResources.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);
    constexpr std::uint64_t argumentOffset = sizeof(std::uint32_t);
    constexpr std::uint32_t argumentStride = 16;
    constexpr std::uint32_t dispatchCount = 3;
    constexpr std::uint64_t argumentBufferSize = argumentOffset +
                                                 static_cast<std::uint64_t>(dispatchCount - 1) * argumentStride +
                                                 sizeof(rachis::DispatchIndirectArguments);
    constexpr std::uint64_t countOffset = sizeof(std::uint32_t);
    constexpr std::uint32_t gpuCount = 2;
    constexpr std::array countConstants{0U, 1U};

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

    [[nodiscard]] rachis::BufferBarrier outputBarrier(rachis::BufferHandle output,
                                                      rachis::PipelineStage stagesAfter,
                                                      rachis::Access accessAfter) noexcept
    {
        return rachis::BufferBarrier{
            .buffer = output,
            .stagesBefore = rachis::PipelineStage::computeShader,
            .stagesAfter = stagesAfter,
            .accessBefore = rachis::Access::unorderedAccess,
            .accessAfter = accessAfter,
        };
    }

    class DispatchBatchFixture final
    {
    public:
        explicit DispatchBatchFixture(rachis::Device& device) : device_(device)
        {
            const auto countBytecode =
                loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.countDispatch.spv");
            const auto clearBytecode =
                loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.spv");
            const auto incrementBytecode =
                loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "ConditionalExecution.spv");
            REQUIRE_FALSE(countBytecode.empty());
            REQUIRE_FALSE(clearBytecode.empty());
            REQUIRE_FALSE(incrementBytecode.empty());
            countShader_ = require(device.createShaderModule(rachis::ShaderModuleDesc{
                .stage = rachis::ShaderStage::compute,
                .bytecode = countBytecode,
                .entryPoint = "countDispatch",
                .label = "Vulkan indirect dispatch batch counter",
                .shaderInterface = rachis::test::computeWriteShaderInterface(),
            }));
            clearShader_ = require(device.createShaderModule(rachis::ShaderModuleDesc{
                .stage = rachis::ShaderStage::compute,
                .bytecode = clearBytecode,
                .entryPoint = "main",
                .label = "Vulkan indirect dispatch batch clear",
                .shaderInterface = rachis::test::computeWriteShaderInterface(),
            }));
            incrementShader_ = require(device.createShaderModule(rachis::ShaderModuleDesc{
                .stage = rachis::ShaderStage::compute,
                .bytecode = incrementBytecode,
                .entryPoint = "main",
                .label = "Vulkan indirect dispatch batch increment",
                .shaderInterface = rachis::test::computeShaderInterface(
                    {1, 1, 1},
                    std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
            }));

            output_ = require(device.createBuffer({
                .size = outputSize,
                .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
                .memoryType = rachis::MemoryType::deviceLocal,
                .label = "Vulkan indirect dispatch batch output",
            }));
            readback_ = require(device.createBuffer({
                .size = outputSize,
                .usage = rachis::BufferUsage::copyDestination,
                .memoryType = rachis::MemoryType::readback,
                .label = "Vulkan indirect dispatch batch readback",
            }));

            std::array<std::byte, argumentBufferSize> argumentBytes{};
            const rachis::DispatchIndirectArguments arguments{1, 1, 1};
            for (std::uint32_t index = 0; index < dispatchCount; ++index)
            {
                std::memcpy(argumentBytes.data() + argumentOffset + static_cast<std::uint64_t>(index) * argumentStride,
                            &arguments,
                            sizeof(arguments));
            }
            arguments_ = require(device.createBuffer({
                .size = argumentBytes.size(),
                .usage = rachis::BufferUsage::indirect,
                .memoryType = rachis::MemoryType::upload,
                .label = "Vulkan padded indirect dispatch records",
            }));
            REQUIRE(device.writeBuffer(arguments_, 0, argumentBytes).has_value());

            std::array<std::byte, countOffset + sizeof(std::uint32_t)> countBytes{};
            std::memcpy(countBytes.data() + countOffset, &gpuCount, sizeof(gpuCount));
            count_ = require(device.createBuffer({
                .size = countBytes.size(),
                .usage = rachis::BufferUsage::indirect,
                .memoryType = rachis::MemoryType::upload,
                .label = "Vulkan indirect dispatch GPU count",
            }));
            REQUIRE(device.writeBuffer(count_, 0, countBytes).has_value());

            const std::array<std::uint64_t, 2> predicateValues{0, 1};
            predicate_ = require(device.createBuffer({
                .size = sizeof(predicateValues),
                .usage = rachis::BufferUsage::predicate,
                .memoryType = rachis::MemoryType::upload,
                .label = "Vulkan indirect dispatch batch predicate",
            }));
            REQUIRE(device.writeBuffer(predicate_, 0, std::as_bytes(std::span(predicateValues))).has_value());

            const std::array bindingEntries{
                rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
            };
            bindingLayout_ = require(device.createBindingLayout({
                .entries = bindingEntries,
                .label = "Vulkan indirect dispatch batch layout",
            }));
            const std::array writes{
                rachis::BindingWrite{
                    .binding = 0,
                    .resource =
                        rachis::BufferBinding{
                            .buffer = output_,
                            .size = outputSize,
                            .structureByteStride = sizeof(std::uint32_t),
                        },
                },
            };
            bindingSet_ = require(rachis::test::createBindingSet(device,
                                                                 {
                                                                     .layout = bindingLayout_,
                                                                     .writes = writes,
                                                                     .label = "Vulkan indirect dispatch batch set",
                                                                 }));
            const std::array bindingLayouts{bindingLayout_};
            const std::array pushConstantRanges{
                rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
            };
            pipelineLayout_ = require(device.createPipelineLayout({
                .bindingLayouts = bindingLayouts,
                .pushConstantRanges = pushConstantRanges,
                .label = "Vulkan indirect dispatch batch pipeline layout",
            }));
            plainLayout_ = require(device.createPipelineLayout({
                .bindingLayouts = bindingLayouts,
                .label = "Vulkan indirect dispatch batch layout without push constants",
            }));
            countPipeline_ = require(device.createComputePipeline({
                .layout = pipelineLayout_,
                .computeShader = countShader_,
                .label = "Vulkan indirect dispatch batch pipeline",
            }));
            clearPipeline_ = require(device.createComputePipeline({
                .layout = pipelineLayout_,
                .computeShader = clearShader_,
                .label = "Vulkan indirect dispatch batch clear pipeline",
            }));
            incrementPipeline_ = require(device.createComputePipeline({
                .layout = plainLayout_,
                .computeShader = incrementShader_,
                .label = "Vulkan indirect dispatch batch increment pipeline",
            }));
        }

        DispatchBatchFixture(const DispatchBatchFixture&) = delete;
        DispatchBatchFixture& operator=(const DispatchBatchFixture&) = delete;
        DispatchBatchFixture(DispatchBatchFixture&&) = delete;
        DispatchBatchFixture& operator=(DispatchBatchFixture&&) = delete;

        ~DispatchBatchFixture()
        {
            static_cast<void>(device_.destroyPipeline(countPipeline_));
            static_cast<void>(device_.destroyPipeline(clearPipeline_));
            static_cast<void>(device_.destroyPipeline(incrementPipeline_));
            static_cast<void>(device_.destroyPipelineLayout(pipelineLayout_));
            static_cast<void>(device_.destroyPipelineLayout(plainLayout_));
            static_cast<void>(device_.destroyShaderModule(countShader_));
            static_cast<void>(device_.destroyShaderModule(clearShader_));
            static_cast<void>(device_.destroyShaderModule(incrementShader_));
            static_cast<void>(device_.destroyBindingSet(bindingSet_));
            static_cast<void>(device_.destroyBindingLayout(bindingLayout_));
            static_cast<void>(device_.destroyBuffer(output_));
            static_cast<void>(device_.destroyBuffer(readback_));
            static_cast<void>(device_.destroyBuffer(arguments_));
            static_cast<void>(device_.destroyBuffer(count_));
            static_cast<void>(device_.destroyBuffer(predicate_));
        }

        [[nodiscard]] rachis::IndirectDispatchDesc fixedBatch() const noexcept
        {
            return {
                .argumentBuffer = arguments_,
                .argumentOffset = argumentOffset,
                .maxDispatchCount = dispatchCount,
                .stride = argumentStride,
            };
        }

        [[nodiscard]] rachis::IndirectDispatchDesc countedBatch() const noexcept
        {
            auto batch = fixedBatch();
            batch.countBuffer = count_;
            batch.countOffset = countOffset;
            return batch;
        }

        [[nodiscard]] rachis::BufferHandle output() const noexcept
        {
            return output_;
        }

        [[nodiscard]] rachis::BufferHandle arguments() const noexcept
        {
            return arguments_;
        }

        [[nodiscard]] rachis::BufferHandle count() const noexcept
        {
            return count_;
        }

        [[nodiscard]] rachis::BufferHandle predicate() const noexcept
        {
            return predicate_;
        }

        void bindCounter(rachis::CommandBuffer& command, std::span<const std::uint32_t> constants) const
        {
            REQUIRE(device_.bindComputePipeline(command, countPipeline_).has_value());
            REQUIRE(device_.bindComputeBindingSet(command, 0, bindingSet_).has_value());
            if (!constants.empty())
            {
                REQUIRE(device_.setComputePushConstants(command, 0, std::as_bytes(constants)).has_value());
            }
        }

        void bindIncrement(rachis::CommandBuffer& command) const
        {
            REQUIRE(device_.bindComputePipeline(command, incrementPipeline_).has_value());
            REQUIRE(device_.bindComputeBindingSet(command, 0, bindingSet_).has_value());
        }

        void clearOutput(rachis::CommandBuffer& command) const
        {
            REQUIRE(device_
                        .bufferBarrier(command,
                                       {
                                           .buffer = output_,
                                           .stagesBefore = rachis::PipelineStage::all,
                                           .stagesAfter = rachis::PipelineStage::computeShader,
                                           .accessBefore = rachis::Access::common,
                                           .accessAfter = rachis::Access::unorderedAccess,
                                       })
                        .has_value());
            REQUIRE(device_.bindComputePipeline(command, clearPipeline_).has_value());
            REQUIRE(device_.bindComputeBindingSet(command, 0, bindingSet_).has_value());
            const std::array clearConstants{0U, 0U};
            REQUIRE(device_.setComputePushConstants(command, 0, std::as_bytes(std::span(clearConstants))).has_value());
            REQUIRE(device_.dispatch(command, 1).has_value());
            orderOutput(command);
        }

        void orderOutput(rachis::CommandBuffer& command) const
        {
            REQUIRE(
                device_
                    .bufferBarrier(
                        command,
                        outputBarrier(output_, rachis::PipelineStage::computeShader, rachis::Access::unorderedAccess))
                    .has_value());
        }

        [[nodiscard]] std::array<std::uint32_t, valueCount> finish(rachis::CommandBuffer command) const
        {
            REQUIRE(device_
                        .bufferBarrier(command,
                                       outputBarrier(output_, rachis::PipelineStage::copy, rachis::Access::copySource))
                        .has_value());
            REQUIRE(device_
                        .copyBuffer(command,
                                    {
                                        .source = output_,
                                        .destination = readback_,
                                        .size = outputSize,
                                    })
                        .has_value());
            auto submitted = device_.submit(std::move(command));
            REQUIRE(submitted.has_value());
            REQUIRE(device_.wait(*submitted).has_value());
            device_.collectGarbage();
            auto mapped = device_.mapBuffer(readback_);
            REQUIRE(mapped.has_value());
            REQUIRE(mapped->size == outputSize);
            std::array<std::uint32_t, valueCount> values{};
            std::memcpy(values.data(), mapped->data, sizeof(values));
            REQUIRE(device_.unmapBuffer(readback_).has_value());
            return values;
        }

    private:
        template<typename T>
        [[nodiscard]] static T require(rachis::Result<T> created)
        {
            REQUIRE(created.has_value());
            return *created;
        }

        rachis::Device& device_;
        rachis::ShaderModuleHandle countShader_{};
        rachis::ShaderModuleHandle clearShader_{};
        rachis::ShaderModuleHandle incrementShader_{};
        rachis::BufferHandle output_{};
        rachis::BufferHandle readback_{};
        rachis::BufferHandle arguments_{};
        rachis::BufferHandle count_{};
        rachis::BufferHandle predicate_{};
        rachis::BindingLayoutHandle bindingLayout_{};
        rachis::BindingSetHandle bindingSet_{};
        rachis::PipelineLayoutHandle pipelineLayout_{};
        rachis::PipelineLayoutHandle plainLayout_{};
        rachis::PipelineHandle countPipeline_{};
        rachis::PipelineHandle clearPipeline_{};
        rachis::PipelineHandle incrementPipeline_{};
    };

    void checkEveryValue(const std::array<std::uint32_t, valueCount>& values, std::uint32_t expected)
    {
        std::uint32_t firstMismatch = valueCount;
        std::uint32_t mismatchValue = 0;
        for (std::uint32_t index = 0; index < valueCount; ++index)
        {
            if (values[index] != expected)
            {
                firstMismatch = index;
                mismatchValue = values[index];
                break;
            }
        }
        CAPTURE(expected, firstMismatch, mismatchValue);
        CHECK(firstMismatch == valueCount);
    }

    void checkNoDiagnostics(const rachis::Device& device)
    {
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

} // namespace

TEST_CASE("Vulkan fixed indirect compute batches execute every record and reject invalid arguments",
          "[size:medium][rhi][vulkan][compute][indirect][batch]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    REQUIRE(capabilities.indirectDispatch);
    if (rachis::test::skipUnlessReferenceDesktopAdvertises(capabilities.multiDispatchIndirect, "multiDispatchIndirect"))
    {
        return;
    }
    {
        DispatchBatchFixture fixture(device);
        auto staleCountBuffer = device.createBuffer({
            .size = countOffset + sizeof(std::uint32_t),
            .usage = rachis::BufferUsage::indirect,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(staleCountBuffer.has_value());
        const auto staleCount = *staleCountBuffer;
        REQUIRE(device.destroyBuffer(*staleCountBuffer).has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        fixture.clearOutput(*command);
        fixture.bindCounter(*command, countConstants);

        const auto arguments = fixture.arguments();
        CHECK_FALSE(device.dispatchIndirect(*command, rachis::IndirectDispatchDesc{}).has_value());
        CHECK_FALSE(device.dispatchIndirect(*command, {.argumentBuffer = arguments, .argumentOffset = 2}).has_value());
        CHECK_FALSE(device
                        .dispatchIndirect(*command,
                                          {.argumentBuffer = arguments, .argumentOffset = argumentOffset, .stride = 8})
                        .has_value());
        CHECK_FALSE(device
                        .dispatchIndirect(*command,
                                          {.argumentBuffer = arguments, .argumentOffset = argumentOffset, .stride = 14})
                        .has_value());
        CHECK_FALSE(device
                        .dispatchIndirect(*command,
                                          {
                                              .argumentBuffer = arguments,
                                              .argumentOffset = argumentOffset,
                                              .countOffset = countOffset,
                                          })
                        .has_value());
        CHECK_FALSE(device.dispatchIndirect(*command, {.argumentBuffer = fixture.output()}).has_value());
        CHECK_FALSE(device
                        .dispatchIndirect(*command,
                                          {
                                              .argumentBuffer = arguments,
                                              .argumentOffset = argumentBufferSize - sizeof(std::uint32_t),
                                          })
                        .has_value());
        CHECK_FALSE(device
                        .dispatchIndirect(*command,
                                          {
                                              .argumentBuffer = arguments,
                                              .argumentOffset = argumentOffset,
                                              .maxDispatchCount = dispatchCount,
                                              .stride = argumentStride + 4,
                                          })
                        .has_value());
        CHECK_FALSE(device
                        .dispatchIndirect(*command,
                                          {
                                              .argumentBuffer = arguments,
                                              .argumentOffset = argumentOffset,
                                              .countBuffer = fixture.output(),
                                          })
                        .has_value());
        CHECK_FALSE(device
                        .dispatchIndirect(*command,
                                          {
                                              .argumentBuffer = arguments,
                                              .argumentOffset = argumentOffset,
                                              .countBuffer = fixture.count(),
                                              .countOffset = countOffset + sizeof(std::uint32_t),
                                          })
                        .has_value());
        CHECK_FALSE(device
                        .dispatchIndirect(*command,
                                          {
                                              .argumentBuffer = arguments,
                                              .argumentOffset = argumentOffset,
                                              .countBuffer = staleCount,
                                          })
                        .has_value());

        std::uint32_t expected = 0;
        REQUIRE(device.dispatchIndirect(*command, fixture.fixedBatch()).has_value());
        expected += dispatchCount;
        fixture.orderOutput(*command);
        REQUIRE(device.dispatch(*command, 1).has_value());
        expected += 1;
        fixture.orderOutput(*command);

        checkEveryValue(fixture.finish(std::move(*command)), expected);
    }
    checkNoDiagnostics(device);
}

TEST_CASE("Vulkan fixed indirect compute batches that fail to retain their arguments record nothing",
          "[size:medium][rhi][vulkan][compute][indirect][batch]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    REQUIRE(capabilities.indirectDispatch);
    if (rachis::test::skipUnlessReferenceDesktopAdvertises(capabilities.multiDispatchIndirect, "multiDispatchIndirect"))
    {
        return;
    }
    {
        DispatchBatchFixture fixture(device);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        fixture.clearOutput(*command);
        fixture.bindCounter(*command, countConstants);

        rachis::vulkan::injectVulkanBadAlloc(rachis::vulkan::VulkanBadAllocPoint::indirectArgumentRetention, true);
        const auto failed = device.dispatchIndirect(*command, fixture.fixedBatch());
        rachis::vulkan::injectVulkanBadAlloc(rachis::vulkan::VulkanBadAllocPoint::indirectArgumentRetention, false);
        REQUIRE_FALSE(failed.has_value());
        CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);

        REQUIRE(device.dispatchIndirect(*command, fixture.fixedBatch()).has_value());
        fixture.orderOutput(*command);
        checkEveryValue(fixture.finish(std::move(*command)), dispatchCount);
    }
    checkNoDiagnostics(device);
}

TEST_CASE("Vulkan GPU-counted indirect compute batches execute the GPU count and keep the bound state",
          "[size:medium][rhi][vulkan][compute][indirect][batch]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    REQUIRE(capabilities.indirectDispatch);
    if (rachis::test::skipUnlessReferenceDesktopAdvertises(capabilities.dispatchIndirectCount, "dispatchIndirectCount"))
    {
        return;
    }
    {
        DispatchBatchFixture fixture(device);

        auto unretainedPool = device.createCommandPool({
            .queueType = rachis::QueueType::compute,
            .maxCommandBuffers = 1,
            .label = "Vulkan unretained indirect dispatch pool",
            .flags = rachis::CommandPoolFlag::unretainedResources,
        });
        REQUIRE(unretainedPool.has_value());
        auto unretainedCommand = device.acquireCommandBuffer(*unretainedPool);
        REQUIRE(unretainedCommand.has_value());
        fixture.clearOutput(*unretainedCommand);
        fixture.bindCounter(*unretainedCommand, countConstants);
        REQUIRE(device.dispatchIndirect(*unretainedCommand, fixture.countedBatch()).has_value());
        checkEveryValue(fixture.finish(std::move(*unretainedCommand)), gpuCount);
        REQUIRE(device.destroyCommandPool(*unretainedPool).has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        fixture.clearOutput(*command);
        fixture.bindCounter(*command, countConstants);
        REQUIRE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = fixture.arguments(),
                                          .argumentOffset = argumentBufferSize,
                                          .maxDispatchCount = 0,
                                          .stride = argumentStride,
                                          .countBuffer = fixture.count(),
                                          .countOffset = countOffset,
                                      })
                    .has_value());

        std::uint32_t expected = 0;
        REQUIRE(device.dispatchIndirect(*command, fixture.countedBatch()).has_value());
        expected += gpuCount;
        fixture.orderOutput(*command);
        REQUIRE(device.dispatch(*command, 1).has_value());
        expected += 1;
        fixture.orderOutput(*command);

        if (capabilities.conditionalExecution)
        {
            for (const std::uint64_t predicateOffset : {std::uint64_t{0}, std::uint64_t{sizeof(std::uint64_t)}})
            {
                REQUIRE(device
                            .beginConditionalExecution(*command,
                                                       {
                                                           .buffer = fixture.predicate(),
                                                           .offset = predicateOffset,
                                                           .operation = rachis::PredicateOperation::notEqualZero,
                                                       })
                            .has_value());
                REQUIRE(device.dispatchIndirect(*command, fixture.countedBatch()).has_value());
                REQUIRE(device.dispatch(*command, 1).has_value());
                REQUIRE(device.endConditionalExecution(*command).has_value());
                fixture.orderOutput(*command);
            }
            expected += gpuCount + 1;
        }

        checkEveryValue(fixture.finish(std::move(*command)), expected);
    }
    checkNoDiagnostics(device);
}

TEST_CASE("Vulkan GPU-counted indirect dispatch clamps the count to maxDispatchCount and reads it on the GPU",
          "[size:medium][rhi][vulkan][compute][indirect][batch]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipUnlessReferenceDesktopAdvertises(
            rachis::test::requireCapabilities(device).dispatchIndirectCount, "dispatchIndirectCount"))
    {
        return;
    }

    const auto countBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.countDispatch.spv");
    REQUIRE_FALSE(countBytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = countBytecode,
        .entryPoint = "countDispatch",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(shader.has_value());

    constexpr std::uint32_t recordCount = 4;
    std::array<rachis::DispatchIndirectArguments, recordCount> records{};
    records.fill({1, 1, 1});
    auto argumentBuffer = device.createBuffer({
        .size = sizeof(records),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
    });
    auto countBuffer = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::indirect | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto countSeed = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto output = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto outputSeed = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(argumentBuffer.has_value());
    REQUIRE(countBuffer.has_value());
    REQUIRE(countSeed.has_value());
    REQUIRE(output.has_value());
    REQUIRE(outputSeed.has_value());
    REQUIRE(readback.has_value());
    const std::array<std::uint32_t, valueCount> zeros{};
    REQUIRE(device.writeBuffer(*outputSeed, 0, std::as_bytes(std::span(zeros))).has_value());
    REQUIRE(device.writeBuffer(*argumentBuffer, 0, std::as_bytes(std::span(records))).has_value());
    constexpr std::uint32_t largeCount = 100;
    REQUIRE(device.writeBuffer(*countSeed, 0, std::as_bytes(std::span(&largeCount, 1))).has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = outputSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *countBuffer,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *countSeed,
                                .destination = *countBuffer,
                                .size = sizeof(std::uint32_t),
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *countBuffer,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::indirect,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::indirectArgument,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *outputSeed,
                                .destination = *output,
                                .size = outputSize,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    const std::array constants{0U, 1U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(constants))).has_value());
    REQUIRE(device
                .dispatchIndirect(*command,
                                  {
                                      .argumentBuffer = *argumentBuffer,
                                      .maxDispatchCount = 3,
                                      .countBuffer = *countBuffer,
                                  })
                .has_value());
    REQUIRE(
        device.bufferBarrier(*command, outputBarrier(*output, rachis::PipelineStage::copy, rachis::Access::copySource))
            .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *output,
                                .destination = *readback,
                                .size = outputSize,
                            })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::array<std::uint32_t, valueCount> values{};
    std::memcpy(values.data(), mapped->data, sizeof(values));
    CHECK(values.front() == 3);
    CHECK(values.back() == 3);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*argumentBuffer).has_value());
    REQUIRE(device.destroyBuffer(*countBuffer).has_value());
    REQUIRE(device.destroyBuffer(*countSeed).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*outputSeed).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    checkNoDiagnostics(device);
}
