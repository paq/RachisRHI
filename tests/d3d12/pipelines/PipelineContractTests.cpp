#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdio>

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

} // namespace

TEST_CASE("shader modules and pipeline layouts own portable creation state", "[size:medium][rhi][pipeline][warp]")
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

    std::array<std::byte, 8> bytecode{
        std::byte{0x44},
        std::byte{0x58},
        std::byte{0x49},
        std::byte{0x4c},
    };
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "computeMain",
        .label = "Owned compute shader",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}),
    });
    REQUIRE(shader.has_value());
    bytecode.fill(std::byte{0});

    CHECK_FALSE(device
                    .createShaderModule(rachis::ShaderModuleDesc{
                        .stage = rachis::ShaderStage::all,
                        .bytecode = bytecode,
                        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}),
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createShaderModule(rachis::ShaderModuleDesc{
                        .stage = rachis::ShaderStage::compute,
                        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}),
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createShaderModule(rachis::ShaderModuleDesc{
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = bytecode,
                        .entryPoint = {},
                        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}),
                    })
                    .has_value());

    const std::array resourceEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel},
        rachis::BindingLayoutEntry{4, rachis::BindingType::sampler, rachis::ShaderStage::pixel},
    };
    auto resources = device.createBindingLayout(rachis::BindingLayoutDesc{
        .entries = resourceEntries,
        .label = "Pipeline resources",
    });
    auto empty = device.createBindingLayout({});
    REQUIRE(resources.has_value());
    REQUIRE(empty.has_value());

    const std::array bindingLayouts{*resources, *empty};
    const std::array pushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::vertex, 0, 16},
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16},
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 16, 16},
    };
    auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstants,
        .label = "Portable pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());

    std::array<rachis::BindingLayoutHandle, rachis::maxBindingSets + 1> tooManyLayouts{};
    tooManyLayouts.fill(*empty);
    CHECK_FALSE(device
                    .createPipelineLayout(rachis::PipelineLayoutDesc{
                        .bindingLayouts = tooManyLayouts,
                    })
                    .has_value());
    const std::array misalignedPushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2},
    };
    CHECK_FALSE(device
                    .createPipelineLayout(rachis::PipelineLayoutDesc{
                        .pushConstantRanges = misalignedPushConstants,
                    })
                    .has_value());
    const std::array outOfBoundsPushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, rachis::maxPushConstantBytes, 4},
    };
    CHECK_FALSE(device
                    .createPipelineLayout(rachis::PipelineLayoutDesc{
                        .pushConstantRanges = outOfBoundsPushConstants,
                    })
                    .has_value());
    const std::array overlappingPushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16},
        rachis::PushConstantRange{rachis::ShaderStage::compute, 8, 16},
    };
    CHECK_FALSE(device
                    .createPipelineLayout(rachis::PipelineLayoutDesc{
                        .pushConstantRanges = overlappingPushConstants,
                    })
                    .has_value());

    auto emptyPipelineLayout = device.createPipelineLayout({});
    REQUIRE(emptyPipelineLayout.has_value());
    REQUIRE(device.destroyPipelineLayout(*emptyPipelineLayout).has_value());

    REQUIRE(device.destroyBindingLayout(*resources).has_value());
    REQUIRE(device.destroyBindingLayout(*empty).has_value());
    CHECK_FALSE(device
                    .createPipelineLayout(rachis::PipelineLayoutDesc{
                        .bindingLayouts = bindingLayouts,
                    })
                    .has_value());

    const auto stalePipelineLayout = *pipelineLayout;
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    CHECK_FALSE(device.destroyPipelineLayout(stalePipelineLayout).has_value());
    const auto staleShader = *shader;
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    CHECK_FALSE(device.destroyShaderModule(staleShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
