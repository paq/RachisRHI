#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

namespace
{

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

} // namespace

TEST_CASE("cooked specialization values select an exact immutable DXIL variant",
          "[size:medium][rhi][pipeline][specialization][library][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto variant7Bytecode = loadShader(RACHIS_TEST_SPECIALIZATION_VARIANT_7_PATH);
    const auto variant13Bytecode = loadShader(RACHIS_TEST_SPECIALIZATION_VARIANT_13_PATH);
    REQUIRE_FALSE(variant7Bytecode.empty());
    REQUIRE_FALSE(variant13Bytecode.empty());

    std::array variant7{
        rachis::SpecializationConstant::makeUnsignedInt32(0, 7),
        rachis::SpecializationConstant::makeBoolean(5, true),
    };
    std::array variant13{
        rachis::SpecializationConstant::makeUnsignedInt32(0, 13),
        rachis::SpecializationConstant::makeBoolean(5, true),
    };
    const std::array entries{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = variant7Bytecode,
            .entryPoint = "main",
            .specializationConstants = variant7,
        },
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = variant13Bytecode,
            .entryPoint = "main",
            .specializationConstants = variant13,
        },
    };
    auto library = device.createShaderLibrary({.entries = entries, .label = "Specialized compute variants"});
    REQUIRE(library.has_value());

    const std::array reorderedVariant7{
        rachis::SpecializationConstant{5, rachis::SpecializationConstantFormat::boolean, 99},
        rachis::SpecializationConstant::makeUnsignedInt32(0, 7),
    };
    const std::array duplicateEntries{
        entries[0],
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = variant13Bytecode,
            .entryPoint = "main",
            .specializationConstants = reorderedVariant7,
        },
    };
    CHECK_FALSE(device.createShaderLibrary({.entries = duplicateEntries}).has_value());

    const std::array duplicateIds{
        rachis::SpecializationConstant::makeUnsignedInt32(0, 13),
        rachis::SpecializationConstant::makeBoolean(0, true),
    };
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .entryPoint = "main",
                        .library = *library,
                        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}),
                        .specializationConstants = duplicateIds,
                    })
                    .has_value());
    const std::array invalidFormat{
        rachis::SpecializationConstant{0, static_cast<rachis::SpecializationConstantFormat>(255), 13},
    };
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .entryPoint = "main",
                        .library = *library,
                        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}),
                        .specializationConstants = invalidFormat,
                    })
                    .has_value());
    std::vector<rachis::SpecializationConstant> tooMany(rachis::maxSpecializationConstants + 1);
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .entryPoint = "main",
                        .library = *library,
                        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}),
                        .specializationConstants = tooMany,
                    })
                    .has_value());
    const std::array missingVariant{rachis::SpecializationConstant::makeUnsignedInt32(0, 99)};
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .entryPoint = "main",
                        .library = *library,
                        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}),
                        .specializationConstants = missingVariant,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = variant13Bytecode,
                        .entryPoint = "main",
                        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}),
                        .specializationConstants = variant13,
                    })
                    .has_value());

    variant13[0].valueBits = 99;
    variant13[1].valueBits = 0;
    const std::array requestedVariant13{
        rachis::SpecializationConstant{5, rachis::SpecializationConstantFormat::boolean, 7},
        rachis::SpecializationConstant::makeUnsignedInt32(0, 13),
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .entryPoint = "main",
        .library = *library,
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
        .specializationConstants = requestedVariant13,
    });
    REQUIRE(shader.has_value());

    constexpr std::uint64_t outputSize = sizeof(std::uint32_t);
    auto storage = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(storage.has_value());
    REQUIRE(readback.has_value());
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
                    .buffer = *storage,
                    .size = outputSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = bindingLayouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    REQUIRE(device.destroyShaderLibrary(*library).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());

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
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
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
        device.copyBuffer(*command, {.source = *storage, .destination = *readback, .size = outputSize}).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::uint32_t value = 0;
    std::memcpy(&value, mapped->data, sizeof(value));
    CHECK(value == 13);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
}
