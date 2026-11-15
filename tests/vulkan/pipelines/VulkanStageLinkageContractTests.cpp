#include "Internal/StageLinkage.h"
#include "Internal/VulkanShaders.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace
{

    using rachis::detail::StageComponentType;
    using rachis::detail::StageInterfaceVariable;

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
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

    [[nodiscard]] std::vector<std::byte> loadStageLinkageShader(std::string_view entryPoint)
    {
        return loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) /
                         ("VulkanStageLinkage." + std::string(entryPoint) + ".spv"));
    }

    [[nodiscard]] constexpr StageInterfaceVariable variable(std::uint32_t location,
                                                            std::uint32_t firstComponent,
                                                            std::uint32_t componentCount,
                                                            StageComponentType type) noexcept
    {
        return StageInterfaceVariable{
            .location = location,
            .firstComponent = firstComponent,
            .componentCount = componentCount,
            .componentType = type,
        };
    }

#ifdef RACHIS_TESTING

    class SpirvModuleBuilder final
    {
    public:
        void instruction(std::uint32_t opcode, std::initializer_list<std::uint32_t> operands)
        {
            words_.push_back(((static_cast<std::uint32_t>(operands.size()) + 1U) << 16U) | opcode);
            words_.insert(words_.end(), operands.begin(), operands.end());
        }

        void entryPoint(std::uint32_t executionModel,
                        std::uint32_t function,
                        std::string_view name,
                        std::initializer_list<std::uint32_t> interfaceIds)
        {
            std::vector<std::uint32_t> nameWords((name.size() / 4) + 1, 0);
            std::memcpy(nameWords.data(), name.data(), name.size());
            const auto count = 3U + static_cast<std::uint32_t>(nameWords.size() + interfaceIds.size());
            words_.push_back((count << 16U) | opEntryPoint);
            words_.push_back(executionModel);
            words_.push_back(function);
            words_.insert(words_.end(), nameWords.begin(), nameWords.end());
            words_.insert(words_.end(), interfaceIds.begin(), interfaceIds.end());
        }

        void emptyFunction(std::uint32_t function, std::uint32_t voidType, std::uint32_t functionType)
        {
            instruction(opFunction, {voidType, function, 0, functionType});
            instruction(opLabel, {function + 1});
            instruction(opReturn, {});
            instruction(opFunctionEnd, {});
        }

        [[nodiscard]] std::vector<std::byte> bytes() const
        {
            std::vector<std::uint32_t> module{0x07230203U, 0x00010600U, 0U, idBound, 0U};
            module.insert(module.end(), words_.begin(), words_.end());
            std::vector<std::byte> bytes(module.size() * sizeof(std::uint32_t));
            std::memcpy(bytes.data(), module.data(), bytes.size());
            return bytes;
        }

        static constexpr std::uint32_t idBound = 4096;
        static constexpr std::uint32_t opEntryPoint = 15;
        static constexpr std::uint32_t opFunction = 54;
        static constexpr std::uint32_t opFunctionEnd = 56;
        static constexpr std::uint32_t opLabel = 248;
        static constexpr std::uint32_t opReturn = 253;

    private:
        std::vector<std::uint32_t> words_;
    };

    constexpr std::uint32_t opCapability = 17;
    constexpr std::uint32_t opMemoryModel = 14;
    constexpr std::uint32_t opDecorate = 71;
    constexpr std::uint32_t opMemberDecorate = 72;
    constexpr std::uint32_t opTypeVoid = 19;
    constexpr std::uint32_t opTypeInt = 21;
    constexpr std::uint32_t opTypeFloat = 22;
    constexpr std::uint32_t opTypeVector = 23;
    constexpr std::uint32_t opTypeArray = 28;
    constexpr std::uint32_t opTypeStruct = 30;
    constexpr std::uint32_t opTypePointer = 32;
    constexpr std::uint32_t opTypeFunction = 33;
    constexpr std::uint32_t opConstant = 43;
    constexpr std::uint32_t opVariable = 59;
    constexpr std::uint32_t decorationBlock = 2;
    constexpr std::uint32_t decorationBuiltIn = 11;
    constexpr std::uint32_t decorationLocation = 30;
    constexpr std::uint32_t decorationComponent = 31;
    constexpr std::uint32_t decorationPerVertexKHR = 5285;
    constexpr std::uint32_t storageInput = 1;
    constexpr std::uint32_t storageOutput = 3;
    constexpr std::uint32_t builtInPosition = 0;
    constexpr std::uint32_t builtInPointSize = 1;
    constexpr std::uint32_t executionModelVertex = 0;
    constexpr std::uint32_t executionModelFragment = 4;
    constexpr std::uint32_t executionModelMeshEXT = 5365;

    enum : std::uint32_t
    {
        idVoid = 1,
        idFunctionType,
        idFloat,
        idHalf,
        idInt,
        idUint,
        idLong,
        idVec2,
        idVec4,
        idUvec2,
        idUintConstant2,
        idUintConstant3,
        idFloatArray2,
        idVec2Array3,
        idVec4Array3,
        idFirstFree,
    };

    void declareCommonTypes(SpirvModuleBuilder& builder)
    {
        builder.instruction(opTypeVoid, {idVoid});
        builder.instruction(opTypeFunction, {idFunctionType, idVoid});
        builder.instruction(opTypeFloat, {idFloat, 32});
        builder.instruction(opTypeFloat, {idHalf, 16});
        builder.instruction(opTypeInt, {idInt, 32, 1});
        builder.instruction(opTypeInt, {idUint, 32, 0});
        builder.instruction(opTypeInt, {idLong, 64, 1});
        builder.instruction(opTypeVector, {idVec2, idFloat, 2});
        builder.instruction(opTypeVector, {idVec4, idFloat, 4});
        builder.instruction(opTypeVector, {idUvec2, idUint, 2});
        builder.instruction(opConstant, {idUint, idUintConstant2, 2});
        builder.instruction(opConstant, {idUint, idUintConstant3, 3});
        builder.instruction(opTypeArray, {idFloatArray2, idFloat, idUintConstant2});
        builder.instruction(opTypeArray, {idVec2Array3, idVec2, idUintConstant3});
        builder.instruction(opTypeArray, {idVec4Array3, idVec4, idUintConstant3});
    }

    void beginModule(SpirvModuleBuilder& builder)
    {
        builder.instruction(opCapability, {1});
        builder.instruction(opMemoryModel, {0, 1});
    }

#endif

    struct LinkageDevice final
    {
        rachis::Device device;
        rachis::PipelineLayoutHandle layout;
    };

    [[nodiscard]] rachis::Result<rachis::ShaderModuleHandle> createStageLinkageShader(rachis::Device& device,
                                                                                      rachis::ShaderStage stage,
                                                                                      std::string_view entryPoint)
    {
        const auto bytecode = loadStageLinkageShader(entryPoint);
        REQUIRE_FALSE(bytecode.empty());
        return device.createShaderModule(rachis::ShaderModuleDesc{
            .stage = stage,
            .bytecode = bytecode,
            .entryPoint = entryPoint,
            .label = entryPoint,
            .shaderInterface = {},
        });
    }

    [[nodiscard]] rachis::Result<rachis::PipelineHandle> createLinkedPipeline(LinkageDevice& context,
                                                                              std::string_view vertexEntryPoint,
                                                                              std::string_view pixelEntryPoint)
    {
        auto vertex = createStageLinkageShader(context.device, rachis::ShaderStage::vertex, vertexEntryPoint);
        auto pixel = createStageLinkageShader(context.device, rachis::ShaderStage::pixel, pixelEntryPoint);
        REQUIRE(vertex.has_value());
        REQUIRE(pixel.has_value());
        const std::array colorTargets{
            rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
        };
        auto pipeline = context.device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
            .layout = context.layout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
            .label = "Vulkan stage linkage pipeline",
        });
        CHECK(context.device.destroyShaderModule(*vertex).has_value());
        CHECK(context.device.destroyShaderModule(*pixel).has_value());
        return pipeline;
    }

    void requireLinkageRejection(LinkageDevice& context,
                                 std::string_view vertexEntryPoint,
                                 std::string_view pixelEntryPoint,
                                 std::string_view expectedMessage)
    {
        INFO(std::string(vertexEntryPoint) + " -> " + std::string(pixelEntryPoint));
        const auto pipeline = createLinkedPipeline(context, vertexEntryPoint, pixelEntryPoint);
        REQUIRE_FALSE(pipeline.has_value());
        CHECK(pipeline.error().code == rachis::ErrorCode::validationError);
        INFO(pipeline.error().message);
        CHECK(pipeline.error().message.find(expectedMessage) != std::string::npos);
    }

} // namespace

#ifdef RACHIS_TESTING

TEST_CASE("Vulkan reflects user-defined stage outputs and inputs from DXC SPIR-V per location",
          "[size:small][rhi][vulkan][shader][stage-linkage]")
{
    using rachis::ShaderStage;
    using rachis::vulkan::reflectVulkanStageInterface;

    const auto layoutBytecode = loadStageLinkageShader("layoutVertex");
    REQUIRE_FALSE(layoutBytecode.empty());
    const auto layout = reflectVulkanStageInterface(layoutBytecode, "layoutVertex", ShaderStage::vertex);
    REQUIRE(layout.has_value());
    CHECK(layout->inputs.empty());
    const std::vector expectedLayout{
        variable(0, 0, 2, StageComponentType::float32),
        variable(1, 0, 2, StageComponentType::float32),
        variable(2, 0, 3, StageComponentType::float32),
        variable(3, 0, 3, StageComponentType::float32),
        variable(4, 0, 4, StageComponentType::float64),
        variable(5, 0, 2, StageComponentType::float64),
    };
    CHECK(layout->outputs == expectedLayout);

    const auto uintBytecode = loadStageLinkageShader("uint4Vertex");
    REQUIRE_FALSE(uintBytecode.empty());
    const auto uint4 = reflectVulkanStageInterface(uintBytecode, "uint4Vertex", ShaderStage::vertex);
    REQUIRE(uint4.has_value());
    CHECK(uint4->outputs == std::vector{variable(0, 0, 4, StageComponentType::uint32)});

    const auto pixelBytecode = loadStageLinkageShader("float2Pixel");
    REQUIRE_FALSE(pixelBytecode.empty());
    const auto pixel = reflectVulkanStageInterface(pixelBytecode, "float2Pixel", ShaderStage::pixel);
    REQUIRE(pixel.has_value());
    CHECK(pixel->outputs.empty());
    CHECK(pixel->inputs == std::vector{variable(0, 0, 2, StageComponentType::float32)});
}

TEST_CASE("Vulkan stage-interface reflection honors Component decorations and skips built-ins",
          "[size:small][rhi][vulkan][shader][stage-linkage]")
{
    SpirvModuleBuilder builder;
    beginModule(builder);
    std::uint32_t nextId = idFirstFree;
    const auto function = nextId;
    nextId += 2;
    const auto scalarAtComponent2 = nextId++;
    const auto uvec2AtComponent0 = nextId++;
    const auto intAtComponent3 = nextId++;
    const auto longAtComponent2 = nextId++;
    const auto halfOutput = nextId++;
    const auto position = nextId++;
    const auto builtInBlock = nextId++;
    const auto locatedBlock = nextId++;
    const auto builtInBlockType = nextId++;
    const auto locatedBlockType = nextId++;
    builder.entryPoint(executionModelVertex,
                       function,
                       "main",
                       {scalarAtComponent2,
                        uvec2AtComponent0,
                        intAtComponent3,
                        longAtComponent2,
                        halfOutput,
                        position,
                        builtInBlock,
                        locatedBlock});
    builder.instruction(opDecorate, {scalarAtComponent2, decorationLocation, 1});
    builder.instruction(opDecorate, {scalarAtComponent2, decorationComponent, 2});
    builder.instruction(opDecorate, {uvec2AtComponent0, decorationLocation, 1});
    builder.instruction(opDecorate, {intAtComponent3, decorationLocation, 3});
    builder.instruction(opDecorate, {intAtComponent3, decorationComponent, 3});
    builder.instruction(opDecorate, {longAtComponent2, decorationLocation, 2});
    builder.instruction(opDecorate, {longAtComponent2, decorationComponent, 2});
    builder.instruction(opDecorate, {halfOutput, decorationLocation, 4});
    builder.instruction(opDecorate, {position, decorationBuiltIn, builtInPosition});
    builder.instruction(opDecorate, {builtInBlockType, decorationBlock});
    builder.instruction(opMemberDecorate, {builtInBlockType, 0, decorationBuiltIn, builtInPosition});
    builder.instruction(opMemberDecorate, {builtInBlockType, 1, decorationBuiltIn, builtInPointSize});
    builder.instruction(opDecorate, {locatedBlockType, decorationBlock});
    builder.instruction(opMemberDecorate, {locatedBlockType, 0, decorationLocation, 5});
    builder.instruction(opMemberDecorate, {locatedBlockType, 1, decorationLocation, 7});
    builder.instruction(opMemberDecorate, {locatedBlockType, 1, decorationComponent, 1});
    declareCommonTypes(builder);
    builder.instruction(opTypeStruct, {builtInBlockType, idVec4, idFloat});
    builder.instruction(opTypeStruct, {locatedBlockType, idVec4, idFloat});
    std::uint32_t pointerId = 150;
    const auto declare = [&](std::uint32_t variableId, std::uint32_t typeId) {
        builder.instruction(opTypePointer, {pointerId, storageOutput, typeId});
        builder.instruction(opVariable, {pointerId, variableId, storageOutput});
        ++pointerId;
    };
    declare(scalarAtComponent2, idFloat);
    declare(uvec2AtComponent0, idUvec2);
    declare(intAtComponent3, idInt);
    declare(longAtComponent2, idLong);
    declare(halfOutput, idHalf);
    declare(position, idVec4);
    declare(builtInBlock, builtInBlockType);
    declare(locatedBlock, locatedBlockType);
    builder.emptyFunction(function, idVoid, idFunctionType);

    const auto bytecode = builder.bytes();
    const auto reflected = rachis::vulkan::reflectVulkanStageInterface(bytecode, "main", rachis::ShaderStage::vertex);
    REQUIRE(reflected.has_value());
    const std::vector expected{
        variable(1, 0, 2, StageComponentType::uint32),
        variable(1, 2, 1, StageComponentType::float32),
        variable(2, 2, 2, StageComponentType::sint64),
        variable(3, 3, 1, StageComponentType::sint32),
        variable(4, 0, 1, StageComponentType::float16),
        variable(5, 0, 4, StageComponentType::float32),
        variable(7, 1, 1, StageComponentType::float32),
    };
    CHECK(reflected->outputs == expected);
    CHECK(reflected->inputs.empty());
}

TEST_CASE("Vulkan stage-interface reflection drops the per-vertex array level of mesh outputs and "
          "PerVertexKHR inputs",
          "[size:small][rhi][vulkan][shader][stage-linkage]")
{
    {
        SpirvModuleBuilder builder;
        beginModule(builder);
        std::uint32_t nextId = idFirstFree;
        const auto function = nextId;
        nextId += 2;
        const auto pointer = nextId++;
        const auto perVertexOutput = nextId++;
        builder.entryPoint(executionModelMeshEXT, function, "main", {perVertexOutput});
        builder.instruction(opDecorate, {perVertexOutput, decorationLocation, 0});
        declareCommonTypes(builder);
        builder.instruction(opTypePointer, {pointer, storageOutput, idVec4Array3});
        builder.instruction(opVariable, {pointer, perVertexOutput, storageOutput});
        builder.emptyFunction(function, idVoid, idFunctionType);

        const auto bytecode = builder.bytes();
        const auto reflected = rachis::vulkan::reflectVulkanStageInterface(bytecode, "main", rachis::ShaderStage::mesh);
        REQUIRE(reflected.has_value());
        CHECK(reflected->outputs == std::vector{variable(0, 0, 4, StageComponentType::float32)});
    }
    {
        SpirvModuleBuilder builder;
        beginModule(builder);
        std::uint32_t nextId = idFirstFree;
        const auto function = nextId;
        nextId += 2;
        const auto perVertexPointer = nextId++;
        const auto perVertexInput = nextId++;
        const auto arrayPointer = nextId++;
        const auto arrayInput = nextId++;
        builder.entryPoint(executionModelFragment, function, "main", {perVertexInput, arrayInput});
        builder.instruction(opDecorate, {perVertexInput, decorationLocation, 0});
        builder.instruction(opDecorate, {perVertexInput, decorationPerVertexKHR});
        builder.instruction(opDecorate, {arrayInput, decorationLocation, 1});
        declareCommonTypes(builder);
        builder.instruction(opTypePointer, {perVertexPointer, storageInput, idVec2Array3});
        builder.instruction(opVariable, {perVertexPointer, perVertexInput, storageInput});
        builder.instruction(opTypePointer, {arrayPointer, storageInput, idFloatArray2});
        builder.instruction(opVariable, {arrayPointer, arrayInput, storageInput});
        builder.emptyFunction(function, idVoid, idFunctionType);

        const auto bytecode = builder.bytes();
        const auto reflected =
            rachis::vulkan::reflectVulkanStageInterface(bytecode, "main", rachis::ShaderStage::pixel);
        REQUIRE(reflected.has_value());
        const std::vector expected{
            variable(0, 0, 2, StageComponentType::float32),
            variable(1, 0, 1, StageComponentType::float32),
            variable(2, 0, 1, StageComponentType::float32),
        };
        CHECK(reflected->inputs == expected);
    }
}

TEST_CASE("Vulkan stage-interface reflection rejects a component range past the end of a location",
          "[size:small][rhi][vulkan][shader][stage-linkage]")
{
    SpirvModuleBuilder builder;
    beginModule(builder);
    std::uint32_t nextId = idFirstFree;
    const auto function = nextId;
    nextId += 2;
    const auto pointer = nextId++;
    const auto output = nextId++;
    builder.entryPoint(executionModelVertex, function, "main", {output});
    builder.instruction(opDecorate, {output, decorationLocation, 0});
    builder.instruction(opDecorate, {output, decorationComponent, 3});
    declareCommonTypes(builder);
    builder.instruction(opTypePointer, {pointer, storageOutput, idVec2});
    builder.instruction(opVariable, {pointer, output, storageOutput});
    builder.emptyFunction(function, idVoid, idFunctionType);

    const auto bytecode = builder.bytes();
    const auto reflected = rachis::vulkan::reflectVulkanStageInterface(bytecode, "main", rachis::ShaderStage::vertex);
    REQUIRE_FALSE(reflected.has_value());
    CHECK(reflected.error().code == rachis::ErrorCode::invalidArgument);
}

TEST_CASE("Vulkan stage-interface reflection stops at the SPIR-V type nesting limit",
          "[size:small][rhi][vulkan][shader][stage-linkage]")
{
    constexpr std::uint32_t nestingDepth = 300;
    SpirvModuleBuilder builder;
    beginModule(builder);
    std::uint32_t nextId = idFirstFree;
    const auto function = nextId;
    nextId += 2;
    const auto pointer = nextId++;
    const auto output = nextId++;
    builder.entryPoint(executionModelVertex, function, "main", {output});
    builder.instruction(opDecorate, {output, decorationLocation, 0});
    declareCommonTypes(builder);
    auto innerType = static_cast<std::uint32_t>(idFloat);
    for (std::uint32_t level = 0; level < nestingDepth; ++level)
    {
        const auto structType = nextId++;
        builder.instruction(opTypeStruct, {structType, innerType});
        innerType = structType;
    }
    builder.instruction(opTypePointer, {pointer, storageOutput, innerType});
    builder.instruction(opVariable, {pointer, output, storageOutput});
    builder.emptyFunction(function, idVoid, idFunctionType);
    REQUIRE(nextId < SpirvModuleBuilder::idBound);

    const auto bytecode = builder.bytes();
    const auto reflected = rachis::vulkan::reflectVulkanStageInterface(bytecode, "main", rachis::ShaderStage::vertex);
    REQUIRE_FALSE(reflected.has_value());
    CHECK(reflected.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(reflected.error().message.find("stage input or output") != std::string::npos);
}

#endif

TEST_CASE("Vulkan graphics pipeline creation applies the inter-stage linkage rule",
          "[size:medium][rhi][vulkan][pipeline][stage-linkage]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto layout = created->createPipelineLayout(rachis::PipelineLayoutDesc{
        .label = "Vulkan stage linkage pipeline layout",
    });
    REQUIRE(layout.has_value());
    LinkageDevice context{.device = std::move(*created), .layout = *layout};

    SECTION("matching outputs and inputs link")
    {
        auto pipeline = createLinkedPipeline(context, "float4Vertex", "float4Pixel");
        REQUIRE(pipeline.has_value());
        CHECK(context.device.destroyPipeline(*pipeline).has_value());
    }

    SECTION("an input that reads fewer components than the output writes links")
    {
        auto pipeline = createLinkedPipeline(context, "float4Vertex", "float2Pixel");
        REQUIRE(pipeline.has_value());
        CHECK(context.device.destroyPipeline(*pipeline).has_value());
    }

    SECTION("an input at a location the vertex shader does not write is rejected")
    {
        requireLinkageRejection(context,
                                "float4Vertex",
                                "location1Pixel",
                                "The pixel shader reads location 1 component 0, which the vertex shader does not "
                                "write.");
    }

    SECTION("an input whose component type differs from the output is rejected")
    {
        requireLinkageRejection(context,
                                "uint4Vertex",
                                "float4Pixel",
                                "The pixel shader reads location 0 component 0 as float32, but the vertex shader "
                                "writes it as uint32.");
    }

    SECTION("an input that reads more components than the output writes is rejected")
    {
        requireLinkageRejection(context,
                                "float2Vertex",
                                "float4Pixel",
                                "The pixel shader reads location 0 component 2, which the vertex shader does not "
                                "write.");
    }

    SECTION("asynchronous compilation reports the same rejection")
    {
        auto vertex = createStageLinkageShader(context.device, rachis::ShaderStage::vertex, "float2Vertex");
        auto pixel = createStageLinkageShader(context.device, rachis::ShaderStage::pixel, "float4Pixel");
        REQUIRE(vertex.has_value());
        REQUIRE(pixel.has_value());
        const std::array colorTargets{
            rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
        };
        auto compilation = context.device.beginGraphicsPipelineCompilation(rachis::GraphicsPipelineDesc{
            .layout = context.layout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
            .label = "Vulkan stage linkage compilation",
        });
        REQUIRE(compilation.has_value());
        const auto pipeline = context.device.waitPipelineCompilation(*compilation);
        REQUIRE_FALSE(pipeline.has_value());
        CHECK(pipeline.error().code == rachis::ErrorCode::validationError);
    }

    CHECK(context.device.destroyPipelineLayout(context.layout).has_value());
}
