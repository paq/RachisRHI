#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
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

    [[nodiscard]] std::filesystem::path pnmsPath(std::string_view metallibName)
    {
        return std::filesystem::path(RACHIS_METALLIB_DIR) /
               std::filesystem::path(metallibName).replace_extension(".pnms");
    }

    [[nodiscard]] rachis::ShaderInterfaceDesc cubeComputeInterface(
        std::array<rachis::ShaderBindingRequirement, 2>& bindings,
        std::array<rachis::ShaderPushConstantRequirement, 1>& push) noexcept
    {
        bindings = {
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 1,
                .type = rachis::ShaderResourceType::readOnlyStorageBuffer,
                .count = 1,
            },
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 2,
                .type = rachis::ShaderResourceType::storageBuffer,
                .count = 1,
            },
        };
        push = {rachis::ShaderPushConstantRequirement{.offset = 0, .size = 16}};
        return rachis::ShaderInterfaceDesc{
            .bindings = bindings,
            .pushConstants = push,
            .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
        };
    }

    [[nodiscard]] std::uint32_t readU32(rachis::Device& device, rachis::BufferHandle buffer)
    {
        std::uint32_t value = 0;
        REQUIRE(device.readBuffer(buffer, 0, std::as_writable_bytes(std::span{&value, 1})).has_value());
        return value;
    }

} // namespace

TEST_CASE("Metal compute and graphics pipelines create from cooked metallib", "[size:large][rhi][metal][pipeline]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto cube = loadBytes(pnmsPath("BootstrapCube.cs.metallib"));
    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());

    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto computeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .label = "Metal pipeline compute",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    REQUIRE(computeShader);
    const std::array computeEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto computeBindingLayout = device.createBindingLayout({.entries = computeEntries});
    REQUIRE(computeBindingLayout);
    const std::array computeSets{*computeBindingLayout};
    const std::array computePush{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    auto computeLayout = device.createPipelineLayout({
        .bindingLayouts = computeSets,
        .pushConstantRanges = computePush,
    });
    REQUIRE(computeLayout);
    auto compute = device.createComputePipeline({
        .layout = *computeLayout,
        .computeShader = *computeShader,
        .label = "Metal cube compute pipeline",
    });
    if (!compute)
    {
        UNSCOPED_INFO(compute.error().message);
    }
    REQUIRE(compute);
    CHECK(device.diagnosticCounts().errors == 0);
    REQUIRE(device.destroyPipeline(*compute));

    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
        .label = "Metal pipeline vertex",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
        .label = "Metal pipeline pixel",
    });
    REQUIRE(vertex);
    REQUIRE(pixel);
    auto graphicsLayout = device.createPipelineLayout({});
    REQUIRE(graphicsLayout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto graphics = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .label = "Metal triangle graphics pipeline",
    });
    if (!graphics)
    {
        UNSCOPED_INFO(graphics.error().message);
    }
    REQUIRE(graphics);
    REQUIRE(device.destroyPipeline(*graphics));
    CHECK_FALSE(device.destroyPipeline(*graphics));
}

TEST_CASE("Metal pipeline creation rejects malformed state and unsupported stages",
          "[size:large][rhi][metal][pipeline]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities);
    const auto cube = loadBytes(pnmsPath("BootstrapCube.cs.metallib"));
    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());

    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto computeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(computeShader);
    REQUIRE(vertex);
    REQUIRE(pixel);

    auto emptyLayout = device.createPipelineLayout({});
    REQUIRE(emptyLayout);
    CHECK_FALSE(device
                    .createComputePipeline({
                        .layout = *emptyLayout,
                        .computeShader = *vertex,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createComputePipeline({
                        .layout = *emptyLayout,
                        .computeShader = *computeShader,
                    })
                    .has_value());

    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    CHECK_FALSE(device
                    .createGraphicsPipeline({
                        .layout = *emptyLayout,
                        .vertexShader = *vertex,
                        .pixelShader = *pixel,
                    })
                    .has_value());
    auto requireHit = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .cachePolicy = rachis::PipelineCachePolicy::requireHit,
    });
    REQUIRE_FALSE(requireHit);
    CHECK(requireHit.error().code == rachis::ErrorCode::invalidArgument);

    auto geometryRejected = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .geometryShader = *vertex,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(geometryRejected);
    CHECK(geometryRejected.error().code == rachis::ErrorCode::unsupported);

    auto meshRejected = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .meshShader = *computeShader,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(meshRejected);
    CHECK(meshRejected.error().code ==
          (capabilities->meshShaders ? rachis::ErrorCode::invalidArgument : rachis::ErrorCode::unsupported));
}

TEST_CASE("Metal pipeline layouts charge the draw index and only the stages the device has",
          "[size:large][rhi][metal][pipeline]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);

    const auto bufferLayout = [&](std::uint32_t count, rachis::ShaderStage visibility) {
        const std::array entries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, visibility, count},
        };
        auto layout = device.createBindingLayout({.entries = entries});
        REQUIRE(layout.has_value());
        return *layout;
    };

    const auto fullVertex = bufferLayout(31, rachis::ShaderStage::vertex);
    const auto almostFullVertex = bufferLayout(30, rachis::ShaderStage::vertex);
    const std::array full{fullVertex};
    const std::array almostFull{almostFullVertex};
    auto withoutDrawIndex = device.createPipelineLayout({.bindingLayouts = full});
    REQUIRE(withoutDrawIndex.has_value());
    auto overfull = device.createPipelineLayout({.bindingLayouts = full, .drawIndex = true});
    REQUIRE_FALSE(overfull.has_value());
    CHECK(overfull.error().code == rachis::ErrorCode::invalidArgument);
    auto fitting = device.createPipelineLayout({.bindingLayouts = almostFull, .drawIndex = true});
    REQUIRE(fitting.has_value());

    const auto vertexAndMesh = bufferLayout(16, rachis::ShaderStage::vertex | rachis::ShaderStage::mesh);
    const auto pixelAndMesh = bufferLayout(16, rachis::ShaderStage::pixel | rachis::ShaderStage::mesh);
    const std::array mixed{vertexAndMesh, pixelAndMesh};
    auto mixedLayout = device.createPipelineLayout({.bindingLayouts = mixed});
    CHECK(mixedLayout.has_value() == !capabilities.meshShaders);
    if (mixedLayout.has_value())
    {
        REQUIRE(device.destroyPipelineLayout(*mixedLayout).has_value());
    }
    else
    {
        CHECK(mixedLayout.error().code == rachis::ErrorCode::invalidArgument);
    }

    REQUIRE(device.destroyPipelineLayout(*fitting).has_value());
    REQUIRE(device.destroyPipelineLayout(*withoutDrawIndex).has_value());
    for (const auto layout : {pixelAndMesh, vertexAndMesh, almostFullVertex, fullVertex})
    {
        REQUIRE(device.destroyBindingLayout(layout).has_value());
    }
}

TEST_CASE("Metal shader modules check declared vertex inputs against the vertex function",
          "[size:large][rhi][metal][pipeline][vertex-input][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto vertexVs = loadBytes(pnmsPath("MetalRasterVertex.vs.metallib"));
    const auto vertexPs = loadBytes(pnmsPath("MetalRasterVertex.ps.metallib"));
    REQUIRE_FALSE(vertexVs.empty());
    REQUIRE_FALSE(vertexPs.empty());

    const auto createVertex = [&](std::span<const std::uint32_t> locations) {
        return device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vertexVs,
            .entryPoint = "vertexMain",
            .label = "Metal two-input vertex",
            .shaderInterface = rachis::test::shaderInterface({}, {}, locations),
        });
    };
    const std::array firstOnly{0U};
    auto underDeclared = createVertex(firstOnly);
    REQUIRE_FALSE(underDeclared);
    CHECK(underDeclared.error().code == rachis::ErrorCode::invalidArgument);
    const std::array overDeclared{0U, 1U, 2U};
    auto extra = createVertex(overDeclared);
    REQUIRE_FALSE(extra);
    CHECK(extra.error().code == rachis::ErrorCode::invalidArgument);
    const std::array duplicated{0U, 1U, 1U};
    auto duplicate = createVertex(duplicated);
    REQUIRE_FALSE(duplicate);
    CHECK(duplicate.error().code == rachis::ErrorCode::invalidArgument);
    auto undeclared = createVertex({});
    REQUIRE_FALSE(undeclared);
    CHECK(undeclared.error().code == rachis::ErrorCode::invalidArgument);
    auto pixelWithInputs = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = vertexPs,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, firstOnly),
    });
    REQUIRE_FALSE(pixelWithInputs);
    CHECK(pixelWithInputs.error().code == rachis::ErrorCode::invalidArgument);

    const std::array reversed{1U, 0U};
    auto vertex = createVertex(reversed);
    if (!vertex)
    {
        UNSCOPED_INFO(vertex.error().message);
    }
    REQUIRE(vertex);
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = vertexPs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(pixel);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const std::array positionAttribute{rachis::VertexAttribute{rachis::VertexFormat::float32x2, 0, 0}};
    const std::array partialLayouts{rachis::VertexBufferLayout{.stride = 20, .attributes = positionAttribute}};
    auto partial = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .vertexBuffers = partialLayouts,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(partial);
    CHECK(partial.error().code == rachis::ErrorCode::validationError);

    const std::array bothAttributes{
        rachis::VertexAttribute{rachis::VertexFormat::float32x2, 0, 0},
        rachis::VertexAttribute{rachis::VertexFormat::float32x3, 8, 1},
    };
    const std::array fullLayouts{rachis::VertexBufferLayout{.stride = 20, .attributes = bothAttributes}};
    auto full = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .vertexBuffers = fullLayouts,
        .colorTargets = colorTargets,
    });
    if (!full)
    {
        UNSCOPED_INFO(full.error().message);
    }
    REQUIRE(full);
    REQUIRE(device.destroyPipeline(*full).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal graphics pipelines report stage link rejections as validation errors",
          "[size:large][rhi][metal][pipeline][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto positionOnlyVs = loadBytes(pnmsPath("MetalHeadlessParity.direct.vs.metallib"));
    const auto colorPs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(positionOnlyVs.empty());
    REQUIRE_FALSE(colorPs.empty());
    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = positionOnlyVs,
        .entryPoint = "directVertex",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = colorPs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(vertex);
    REQUIRE(pixel);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto mismatched = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .label = "Metal unlinked stage pipeline",
    });
    REQUIRE_FALSE(mismatched);
    UNSCOPED_INFO(mismatched.error().message);
    CHECK(mismatched.error().code == rachis::ErrorCode::validationError);
}

TEST_CASE("Metal graphics pipeline creation fails when its depth-stencil state cannot be allocated",
          "[size:large][rhi][metal][pipeline][depth][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());
    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(vertex);
    REQUIRE(pixel);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    rachis::GraphicsPipelineDesc desc{
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .depthStencil =
            {
                .format = rachis::Format::d32Float,
                .depthTestEnabled = true,
                .depthWriteEnabled = true,
            },
        .label = "Metal depth-state failure pipeline",
    };

    {
        rachis::metal::ScopedFailNextDepthStencilStateCreate const failDepthState;
        auto failed = device.createGraphicsPipeline(desc);
        REQUIRE_FALSE(failed);
        CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    }
    auto recovered = device.createGraphicsPipeline(desc);
    REQUIRE(recovered);
    REQUIRE(device.destroyPipeline(*recovered).has_value());

    desc.depthStencil.format = rachis::Format::unknown;
    {
        rachis::metal::ScopedFailNextDepthStencilStateCreate const failDepthState;
        auto depthless = device.createGraphicsPipeline(desc);
        REQUIRE(depthless);
        REQUIRE(device.destroyPipeline(*depthless).has_value());
    }
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal pipeline compilation is synchronous sugar over create",
          "[size:large][rhi][metal][pipeline][compilation]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto cube = loadBytes(pnmsPath("BootstrapCube.cs.metallib"));
    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());

    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto computeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    REQUIRE(computeShader);
    const std::array computeEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto computeBindingLayout = device.createBindingLayout({.entries = computeEntries});
    REQUIRE(computeBindingLayout);
    const std::array computeSets{*computeBindingLayout};
    const std::array computePush{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    std::string computeLabel = "Metal async compute";
    auto computeLayout = device.createPipelineLayout({
        .bindingLayouts = computeSets,
        .pushConstantRanges = computePush,
        .label = "Metal async compute layout",
    });
    REQUIRE(computeLayout);
    const rachis::ComputePipelineDesc computeDesc{
        .layout = *computeLayout,
        .computeShader = *computeShader,
        .label = computeLabel,
    };
    auto compilation = device.beginComputePipelineCompilation(computeDesc);
    auto abandoned = device.beginComputePipelineCompilation(computeDesc);
    REQUIRE(compilation);
    REQUIRE(abandoned);
    const auto observed = device.pipelineCompilationState(*compilation);
    REQUIRE(observed);
    CHECK((*observed == rachis::PipelineCompilationState::pending ||
           *observed == rachis::PipelineCompilationState::succeeded));
    REQUIRE(device.destroyPipelineCompilation(*abandoned));
    CHECK_FALSE(device.destroyPipelineCompilation(*abandoned));

    computeLabel.assign("caller storage was changed");
    REQUIRE(device.destroyPipelineLayout(*computeLayout));
    REQUIRE(device.destroyShaderModule(*computeShader));
    REQUIRE(device.destroyBindingLayout(*computeBindingLayout));

    auto computePipeline = device.waitPipelineCompilation(*compilation);
    REQUIRE(computePipeline);
    CHECK_FALSE(device.pipelineCompilationState(*compilation).has_value());
    CHECK_FALSE(device.waitPipelineCompilation(*compilation).has_value());
    REQUIRE(device.destroyPipeline(*computePipeline));

    auto failed = device.beginComputePipelineCompilation({
        .layout = {},
        .computeShader = {},
    });
    REQUIRE(failed);
    CHECK_FALSE(device.waitPipelineCompilation(*failed).has_value());
    CHECK_FALSE(device.pipelineCompilationState(*failed).has_value());

    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
    });
    auto graphicsLayout = device.createPipelineLayout({});
    REQUIRE(vertex);
    REQUIRE(pixel);
    REQUIRE(graphicsLayout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto graphicsCompilation = device.beginGraphicsPipelineCompilation({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .label = "Metal async graphics",
    });
    REQUIRE(graphicsCompilation);
    auto graphics = device.waitPipelineCompilation(*graphicsCompilation);
    REQUIRE(graphics);
    REQUIRE(device.destroyPipeline(*graphics));
}

TEST_CASE("Metal specialized library variants write distinct GPU values",
          "[size:large][rhi][metal][pipeline][specialization][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto variant7 = loadBytes(pnmsPath("MetalSpecialization.7.cs.metallib"));
    const auto variant13 = loadBytes(pnmsPath("MetalSpecialization.13.cs.metallib"));
    REQUIRE_FALSE(variant7.empty());
    REQUIRE_FALSE(variant13.empty());

    std::array spec7{
        rachis::SpecializationConstant::makeUnsignedInt32(0, 7),
        rachis::SpecializationConstant::makeBoolean(5, true),
    };
    std::array spec13{
        rachis::SpecializationConstant::makeUnsignedInt32(0, 13),
        rachis::SpecializationConstant::makeBoolean(5, true),
    };
    const std::array entries{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = variant7,
            .entryPoint = "computeMain",
            .specializationConstants = spec7,
        },
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = variant13,
            .entryPoint = "computeMain",
            .specializationConstants = spec13,
        },
    };
    auto library = device.createShaderLibrary({.entries = entries, .label = "Metal specialized compute"});
    REQUIRE(library);

    const std::array reordered7{
        rachis::SpecializationConstant{5, rachis::SpecializationConstantFormat::boolean, 99},
        rachis::SpecializationConstant::makeUnsignedInt32(0, 7),
    };
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = variant13,
                        .entryPoint = "computeMain",
                        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}),
                        .specializationConstants = spec13,
                    })
                    .has_value());

    const std::array storageBinding{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader7 = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .entryPoint = "computeMain",
        .library = *library,
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, storageBinding),
        .specializationConstants = reordered7,
    });
    const std::array requested13{
        rachis::SpecializationConstant{5, rachis::SpecializationConstantFormat::boolean, 7},
        rachis::SpecializationConstant::makeUnsignedInt32(0, 13),
    };
    auto shader13 = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .entryPoint = "computeMain",
        .library = *library,
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, storageBinding),
        .specializationConstants = requested13,
    });
    REQUIRE(shader7);
    REQUIRE(shader13);

    constexpr std::uint64_t outputSize = sizeof(std::uint32_t);
    auto storage7 = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto storage13 = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto readback7 = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    auto readback13 = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(storage7);
    REQUIRE(storage13);
    REQUIRE(readback7);
    REQUIRE(readback13);

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout);
    auto set7 = rachis::test::createBindingSet(
        device,
        {.layout = *bindingLayout,
         .writes = std::array{
             rachis::BindingWrite{0, 0, rachis::BufferBinding{*storage7, 0, outputSize, sizeof(std::uint32_t)}}}});
    auto set13 = rachis::test::createBindingSet(
        device,
        {.layout = *bindingLayout,
         .writes = std::array{
             rachis::BindingWrite{0, 0, rachis::BufferBinding{*storage13, 0, outputSize, sizeof(std::uint32_t)}}}});
    REQUIRE(set7);
    REQUIRE(set13);
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = std::array{*bindingLayout}});
    REQUIRE(pipelineLayout);
    auto pipeline7 = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader7});
    auto pipeline13 = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader13});
    REQUIRE(pipeline7);
    REQUIRE(pipeline13);

    REQUIRE(device.destroyShaderLibrary(*library));
    REQUIRE(device.destroyShaderModule(*shader7));
    REQUIRE(device.destroyShaderModule(*shader13));
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
    REQUIRE(device.destroyBindingLayout(*bindingLayout));

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.bindComputePipeline(*command, *pipeline7));
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set7));
    REQUIRE(device.dispatch(*command, 1));
    REQUIRE(device.bindComputePipeline(*command, *pipeline13));
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set13));
    REQUIRE(device.dispatch(*command, 1));
    REQUIRE(device.copyBuffer(*command, {.source = *storage7, .destination = *readback7, .size = outputSize}));
    REQUIRE(device.copyBuffer(*command, {.source = *storage13, .destination = *readback13, .size = outputSize}));
    REQUIRE(device.destroyPipeline(*pipeline7));
    REQUIRE(device.destroyPipeline(*pipeline13));
    REQUIRE(device.destroyBindingSet(*set7));
    REQUIRE(device.destroyBindingSet(*set13));
    REQUIRE(device.destroyBuffer(*storage7));
    REQUIRE(device.destroyBuffer(*storage13));

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    CHECK(readU32(device, *readback7) == 7);
    CHECK(readU32(device, *readback13) == 13);
    REQUIRE(device.destroyBuffer(*readback7));
    REQUIRE(device.destroyBuffer(*readback13));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal compute push constants are captured and survive source mutation",
          "[size:large][rhi][metal][pipeline][push][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto bytecode = loadBytes(pnmsPath("MetalPushConstant.cs.metallib"));
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array shaderPush{rachis::ShaderPushConstantRequirement{0, 16}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, shaderBindings, shaderPush),
    });
    if (!shader)
    {
        UNSCOPED_INFO(shader.error().message);
    }
    REQUIRE(shader);

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
    REQUIRE(storage);
    REQUIRE(readback);
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout);
    auto set = rachis::test::createBindingSet(
        device,
        {.layout = *bindingLayout,
         .writes = std::array{
             rachis::BindingWrite{1, 0, rachis::BufferBinding{*storage, 0, outputSize, sizeof(std::uint32_t)}}}});
    REQUIRE(set);
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*bindingLayout},
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
    });
    REQUIRE(pipelineLayout);
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.bindComputePipeline(*command, *pipeline));
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set));
    std::array<std::uint32_t, 4> constants{0xA1B2C3D4U, 0, 0, 0};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span{constants})));
    constants[0] = 0xFFFFFFFFU;
    REQUIRE(device.dispatch(*command, 1));
    REQUIRE(device.copyBuffer(*command, {.source = *storage, .destination = *readback, .size = outputSize}));
    REQUIRE(device.destroyPipeline(*pipeline));
    REQUIRE(device.destroyBindingSet(*set));
    REQUIRE(device.destroyBuffer(*storage));

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    CHECK(readU32(device, *readback) == 0xA1B2C3D4U);
    REQUIRE(device.destroyBuffer(*readback));
    CHECK(device.diagnosticCounts().errors == 0);
}

#ifdef RACHIS_TESTING
TEST_CASE("Metal async pipeline compilation rejects lost device without leaking pipelines",
          "[size:large][rhi][metal][pipeline]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto cube = loadBytes(pnmsPath("BootstrapCube.cs.metallib"));
    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());

    std::array<rachis::ShaderBindingRequirement, 2> csBindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> csPush{};
    auto csShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = cubeComputeInterface(csBindings, csPush),
    });
    auto vsShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
    });
    auto psShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(csShader);
    REQUIRE(vsShader);
    REQUIRE(psShader);

    const std::array csEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto csBindingLayout = device.createBindingLayout({.entries = csEntries});
    REQUIRE(csBindingLayout);
    auto csPipeLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*csBindingLayout},
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
    });
    auto gfxPipeLayout = device.createPipelineLayout({});
    REQUIRE(csPipeLayout);
    REQUIRE(gfxPipeLayout);

    auto injected = rachis::metal::injectMetalDeviceLoss();
    REQUIRE_FALSE(injected);
    CHECK(injected.error().code == rachis::ErrorCode::deviceLost);

    auto csCompile = device.beginComputePipelineCompilation({
        .layout = *csPipeLayout,
        .computeShader = *csShader,
    });
    REQUIRE_FALSE(csCompile);
    CHECK(csCompile.error().code == rachis::ErrorCode::deviceLost);

    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto gfxCompile = device.beginGraphicsPipelineCompilation({
        .layout = *gfxPipeLayout,
        .vertexShader = *vsShader,
        .pixelShader = *psShader,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(gfxCompile);
    CHECK(gfxCompile.error().code == rachis::ErrorCode::deviceLost);
}
#endif

TEST_CASE("Metal separates compute and graphics push constants across mip generation and multi-range writes",
          "[size:large][rhi][metal][pipeline][push][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto csBytecode = loadBytes(pnmsPath("MetalPushConstant.cs.metallib"));
    REQUIRE_FALSE(csBytecode.empty());
    const std::array csShaderBindings{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array csShaderPush{rachis::ShaderPushConstantRequirement{0, 16}};
    auto csShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = csBytecode,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, csShaderBindings, csShaderPush),
    });
    REQUIRE(csShader);

    constexpr std::uint64_t outputSize = sizeof(std::uint32_t);
    auto storage = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto csReadback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(storage);
    REQUIRE(csReadback);

    const std::array csEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto csBindingLayout = device.createBindingLayout({.entries = csEntries});
    REQUIRE(csBindingLayout);
    auto csSet = rachis::test::createBindingSet(
        device,
        {.layout = *csBindingLayout,
         .writes = std::array{
             rachis::BindingWrite{1, 0, rachis::BufferBinding{*storage, 0, outputSize, sizeof(std::uint32_t)}}}});
    REQUIRE(csSet);

    const std::array splitRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 8},
        rachis::PushConstantRange{rachis::ShaderStage::compute, 8, 8},
    };
    auto csPipeLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*csBindingLayout},
        .pushConstantRanges = splitRanges,
    });
    REQUIRE(csPipeLayout);
    auto csPipeline = device.createComputePipeline({.layout = *csPipeLayout, .computeShader = *csShader});
    REQUIRE(csPipeline);

    const auto vsBytes = loadBytes(pnmsPath("MetalHeadlessParity.direct.vs.metallib"));
    const auto psBytes = loadBytes(pnmsPath("MetalHeadlessParity.ps.metallib"));
    REQUIRE_FALSE(vsBytes.empty());
    REQUIRE_FALSE(psBytes.empty());
    auto vsShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vsBytes,
        .entryPoint = "directVertex",
    });
    auto psShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = psBytes,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(vsShader);
    REQUIRE(psShader);

    auto gfxPipeLayout = device.createPipelineLayout({
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 16}},
    });
    REQUIRE(gfxPipeLayout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto gfxPipeline = device.createGraphicsPipeline({
        .layout = *gfxPipeLayout,
        .vertexShader = *vsShader,
        .pixelShader = *psShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE(gfxPipeline);

    constexpr std::uint32_t renderSize = 16;
    auto colorTarget = device.createTexture({
        .width = renderSize,
        .height = renderSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
    });
    REQUIRE(colorTarget);
    auto colorView = device.createTextureView({
        .texture = *colorTarget,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(colorView);

    const rachis::BufferTextureCopyRegion region{
        .bytesPerRow = 256,
        .rowsPerImage = renderSize,
        .extent = {.width = renderSize, .height = renderSize, .depth = 1},
    };
    auto gfxReadback = device.createBuffer({
        .size = 256 * renderSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(gfxReadback);

    auto mipTex = device.createTexture({
        .width = 4,
        .height = 4,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::copyDestination,
    });
    auto mipUpload = device.createBuffer({
        .size = 256 * 4,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(mipTex);
    REQUIRE(mipUpload);

    auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(cmd);
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *colorView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
        },
    };

    REQUIRE(device
                .textureBarrier(*cmd,
                                {
                                    .texture = *colorTarget,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device.beginRendering(*cmd, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*cmd, *gfxPipeline).has_value());
    const std::array<float, 4> greenTint{0.0F, 1.0F, 0.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*cmd, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span{greenTint}))
                .has_value());
    REQUIRE(device.endRendering(*cmd).has_value());

    REQUIRE(device
                .textureBarrier(*cmd,
                                {
                                    .texture = *mipTex,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion mipCopy{
        .bytesPerRow = 256,
        .rowsPerImage = 4,
        .extent = {.width = 4, .height = 4, .depth = 1},
    };
    REQUIRE(device.copyBufferToTexture(*cmd, *mipUpload, *mipTex, std::array{mipCopy}).has_value());
    REQUIRE(device
                .generateMipmaps(*cmd,
                                 {
                                     .texture = *mipTex,
                                     .baseLayout = rachis::TextureLayout::copyDestination,
                                     .finalLayout = rachis::TextureLayout::shaderResource,
                                 })
                .has_value());

    REQUIRE(device.bindComputePipeline(*cmd, *csPipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*cmd, 0, *csSet).has_value());
    const std::array<std::uint32_t, 4> csPushConstants{0xCAFEBABEU, 11U, 22U, 33U};
    REQUIRE(device.setComputePushConstants(*cmd, 0, std::as_bytes(std::span{csPushConstants})).has_value());

    REQUIRE(device.beginRendering(*cmd, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*cmd, *gfxPipeline).has_value());
    REQUIRE(
        device.setViewport(*cmd, {.width = static_cast<float>(renderSize), .height = static_cast<float>(renderSize)})
            .has_value());
    REQUIRE(device.setScissor(*cmd, {.x = 0, .y = 0, .width = renderSize / 2, .height = renderSize}).has_value());
    REQUIRE(device.draw(*cmd, 3, 1, 0, 0).has_value());

    const std::array<float, 4> blueTint{0.0F, 0.0F, 1.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*cmd, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span{blueTint}))
                .has_value());
    REQUIRE(device.setScissor(*cmd, {.x = renderSize / 2, .y = 0, .width = renderSize / 2, .height = renderSize})
                .has_value());
    REQUIRE(device.draw(*cmd, 3, 1, 0, 0).has_value());
    REQUIRE(device.endRendering(*cmd).has_value());

    REQUIRE(device.dispatch(*cmd, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*cmd,
                               {
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*cmd, {.source = *storage, .destination = *csReadback, .size = outputSize}).has_value());

    REQUIRE(device
                .textureBarrier(*cmd,
                                {
                                    .texture = *colorTarget,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*cmd, *colorTarget, *gfxReadback, std::array{region}).has_value());

    auto submitted = device.submit(std::move(*cmd));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    CHECK(readU32(device, *csReadback) == 0xCAFEBABEU);

    std::array<std::uint8_t, 4> leftPixel{};
    std::array<std::uint8_t, 4> rightPixel{};
    const std::uint64_t leftOffset = (renderSize / 2) * 256 + (renderSize / 2 - 2) * 4;
    const std::uint64_t rightOffset = (renderSize / 2) * 256 + (renderSize / 2 + 2) * 4;
    REQUIRE(device.readBuffer(*gfxReadback, leftOffset, std::as_writable_bytes(std::span{leftPixel})).has_value());
    REQUIRE(device.readBuffer(*gfxReadback, rightOffset, std::as_writable_bytes(std::span{rightPixel})).has_value());
    CHECK(leftPixel == std::array<std::uint8_t, 4>{0, 255, 0, 255});
    CHECK(rightPixel == std::array<std::uint8_t, 4>{0, 0, 255, 255});
    CHECK(device.diagnosticCounts().errors == 0);
}
