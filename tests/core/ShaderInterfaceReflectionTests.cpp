#include "GeneratedBindingArray.h"
#include "Internal/PortableValidation.h"
#include "Internal/ShaderInterfaceReflection.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{

    using rachis::shader_tool::BackendInterface;
    using rachis::shader_tool::InterfaceBackend;
    using rachis::shader_tool::InterfaceStage;
    using rachis::shader_tool::ReflectedBinding;
    using rachis::shader_tool::reflectShaderInterface;

    constexpr std::string_view computeSpirvJson = R"({
        "parameters": [
            {"name": "Constants", "binding": {"kind": "constantBuffer", "index": 0},
             "type": {"kind": "constantBuffer", "elementType": {"kind": "struct", "fields": []}}},
            {"name": "elements", "binding": {"kind": "shaderResource", "index": 1},
             "type": {"kind": "resource", "baseShape": "structuredBuffer", "resultType": {"kind": "scalar"}}},
            {"name": "values", "binding": {"kind": "unorderedAccess", "index": 2},
             "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}},
            {"name": "unusedTexture", "binding": {"kind": "shaderResource", "index": 3},
             "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "linearSampler", "binding": {"kind": "samplerState", "space": 1, "index": 4},
             "type": {"kind": "samplerState"}},
            {"name": "colorTextures", "binding": {"kind": "shaderResource", "space": 1, "index": 0},
             "type": {"kind": "array", "elementCount": 2, "elementType": {"kind": "resource", "baseShape": "texture2D"}}},
            {"name": "push", "bindings": [{"kind": "pushConstantBuffer", "index": 0},
                                          {"kind": "constantBuffer", "index": 0, "count": 0}],
             "type": {"kind": "constantBuffer",
                      "containerVarLayout": {"binding": {"kind": "pushConstantBuffer", "index": 0}},
                      "elementVarLayout": {"binding": {"kind": "uniform", "offset": 0, "size": 8}}}}
        ],
        "entryPoints": [
            {"name": "probeMain", "stage": "compute", "threadGroupSize": [8, 4, 1],
             "bindings": [
                {"name": "Constants", "binding": {"kind": "constantBuffer", "index": 0, "used": 1}},
                {"name": "elements", "binding": {"kind": "shaderResource", "index": 1, "used": 1}},
                {"name": "values", "binding": {"kind": "unorderedAccess", "index": 2, "used": 1}},
                {"name": "unusedTexture", "binding": {"kind": "shaderResource", "index": 3, "used": 0}},
                {"name": "linearSampler", "binding": {"kind": "samplerState", "space": 1, "index": 4, "used": 1}},
                {"name": "colorTextures", "binding": {"kind": "shaderResource", "space": 1, "index": 0, "used": 1}},
                {"name": "push", "bindings": [{"kind": "pushConstantBuffer", "index": 0},
                                              {"kind": "constantBuffer", "index": 0, "count": 0, "used": 1}]}
             ]}
        ]
    })";

    constexpr std::string_view computeHlslJson = R"({
        "parameters": [
            {"name": "values", "binding": {"kind": "unorderedAccess", "index": 0},
             "type": {"kind": "resource", "baseShape": "byteAddressBuffer", "access": "readWrite"}},
            {"name": "push", "binding": {"kind": "constantBuffer", "space": 4, "index": 0},
             "type": {"kind": "constantBuffer",
                      "elementVarLayout": {"binding": {"kind": "uniform", "offset": 0, "size": 16}}}},
            {"name": "bindless", "binding": {"kind": "shaderResource", "space": 2, "index": 0},
             "type": {"kind": "array", "elementCount": 0, "elementType": {"kind": "resource", "baseShape": "texture2D"}}}
        ],
        "entryPoints": [
            {"name": "main", "stage": "compute", "threadGroupSize": [64, 1, 1],
             "bindings": [
                {"name": "values", "binding": {"kind": "unorderedAccess", "index": 0, "used": 1}},
                {"name": "push", "binding": {"kind": "constantBuffer", "space": 4, "index": 0, "used": 1}},
                {"name": "bindless", "binding": {"kind": "shaderResource", "space": 2, "index": 0, "used": 1}}
             ]}
        ]
    })";

    [[nodiscard]] std::string vertexJson(std::string_view secondSemantic)
    {
        return std::string(R"({
        "parameters": [],
        "entryPoints": [
            {"name": "vertexMain", "stage": "vertex", "bindings": [],
             "parameters": [
                {"name": "vertexId", "semanticName": "SV_VERTEXID", "type": {"kind": "scalar"}},
                {"name": "position", "binding": {"kind": "varyingInput", "index": 0}, "semanticName": "TEXCOORD",
                 "type": {"kind": "vector"}},
                {"name": "transform", "binding": {"kind": "varyingInput", "index": 1, "count": 4},
                 "semanticName": ")") +
               std::string(secondSemantic) + R"(", "semanticIndex": 4, "type": {"kind": "matrix"}},
                {"name": "extra", "type": {"kind": "struct", "fields": [
                    {"name": "color", "binding": {"kind": "varyingInput", "index": 5}, "semanticName": "TEXCOORD",
                     "semanticIndex": 2, "type": {"kind": "vector"}}
                ]}}
             ]}
        ]
    })";
    }

} // namespace

TEST_CASE("Shader interface reflection keeps the resources the entry point uses",
          "[size:small][rhi][tooling][shader-interface]")
{
    const auto reflected =
        reflectShaderInterface(computeSpirvJson, InterfaceBackend::vulkan, "probeMain", InterfaceStage::compute);
    REQUIRE(reflected.has_value());
    const std::vector<ReflectedBinding> expected{
        {.name = "Constants", .set = 0, .binding = 0, .type = "uniformBuffer", .count = 1},
        {.name = "elements", .set = 0, .binding = 1, .type = "readOnlyStorageBuffer", .count = 1},
        {.name = "values", .set = 0, .binding = 2, .type = "storageBuffer", .count = 1},
        {.name = "colorTextures", .set = 1, .binding = 0, .type = "sampledTexture", .count = 2},
        {.name = "linearSampler", .set = 1, .binding = 4, .type = "sampler", .count = 1},
    };
    CHECK(reflected->bindings == expected);
    CHECK(reflected->pushConstantBytes == 8);
    REQUIRE(reflected->workgroupSize.has_value());
    CHECK(*reflected->workgroupSize == std::array<std::uint32_t, 3>{8, 4, 1});
    CHECK(reflected->vertexInputLocations.empty());

    const auto d3d12 =
        reflectShaderInterface(computeHlslJson, InterfaceBackend::d3d12, "main", InterfaceStage::compute);
    REQUIRE(d3d12.has_value());
    const std::vector<ReflectedBinding> d3d12Expected{
        {.name = "values", .set = 0, .binding = 0, .type = "storageBuffer", .count = 1},
        {.name = "bindless",
         .set = 2,
         .binding = 0,
         .type = "sampledTexture",
         .count = rachis::shader_tool::unboundedResourceCount},
    };
    CHECK(d3d12->bindings == d3d12Expected);
    CHECK(d3d12->pushConstantBytes == 16);

    const auto vulkan =
        reflectShaderInterface(computeHlslJson, InterfaceBackend::vulkan, "main", InterfaceStage::compute);
    REQUIRE_FALSE(vulkan.has_value());
    CHECK(vulkan.error().find("'push' is in binding set 4") != std::string::npos);
}

TEST_CASE("Shader interface reflection reads Boolean used flags like integer flags",
          "[size:small][rhi][tooling][shader-interface]")
{
    constexpr std::string_view json = R"({
        "parameters": [
            {"name": "results", "binding": {"kind": "unorderedAccess", "index": 0},
             "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}},
            {"name": "unused", "binding": {"kind": "unorderedAccess", "index": 1},
             "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}}
        ],
        "entryPoints": [
            {"name": "main", "stage": "compute", "threadGroupSize": [1, 1, 1],
             "bindings": [
                {"name": "results", "binding": {"kind": "unorderedAccess", "index": 0, "used": true}},
                {"name": "unused", "binding": {"kind": "unorderedAccess", "index": 1, "used": false}}
             ]}
        ]
    })";
    for (const auto backend : {InterfaceBackend::d3d12, InterfaceBackend::vulkan, InterfaceBackend::metal})
    {
        INFO("backend " << rachis::shader_tool::interfaceBackendName(backend));
        const auto reflected = reflectShaderInterface(json, backend, "main", InterfaceStage::compute);
        REQUIRE(reflected.has_value());
        const std::vector<ReflectedBinding> expected{
            {.name = "results", .set = 0, .binding = 0, .type = "storageBuffer", .count = 1},
        };
        CHECK(reflected->bindings == expected);
    }
}

TEST_CASE("Shader interface reflection reads vertex input locations per backend",
          "[size:small][rhi][tooling][shader-interface]")
{
    const auto json = vertexJson("TEXCOORD");
    const auto vulkan = reflectShaderInterface(json, InterfaceBackend::vulkan, "vertexMain", InterfaceStage::vertex);
    REQUIRE(vulkan.has_value());
    CHECK(vulkan->vertexInputLocations == std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5});
    CHECK_FALSE(vulkan->workgroupSize.has_value());

    const auto d3d12 = reflectShaderInterface(json, InterfaceBackend::d3d12, "vertexMain", InterfaceStage::vertex);
    REQUIRE(d3d12.has_value());
    CHECK(d3d12->vertexInputLocations == std::vector<std::uint32_t>{0, 2, 4, 5, 6, 7});

    const auto otherSemantic =
        reflectShaderInterface(vertexJson("COLOR"), InterfaceBackend::d3d12, "vertexMain", InterfaceStage::vertex);
    REQUIRE_FALSE(otherSemantic.has_value());
    CHECK(otherSemantic.error().find("TEXCOORD") != std::string::npos);
}

TEST_CASE("Shader interface reflection rejects what RachisRHI cannot bind",
          "[size:small][rhi][tooling][shader-interface]")
{
    CHECK_FALSE(reflectShaderInterface("{", InterfaceBackend::vulkan, "main", InterfaceStage::compute).has_value());
    CHECK_FALSE(
        reflectShaderInterface(
            R"({"parameters": [], "entryPoints": []})", InterfaceBackend::vulkan, "main", InterfaceStage::compute)
            .has_value());

    constexpr std::string_view looseUniform = R"({
        "parameters": [{"name": "scale", "binding": {"kind": "uniform", "offset": 0, "size": 4},
                        "type": {"kind": "scalar"}}],
        "entryPoints": [{"name": "main", "threadGroupSize": [1, 1, 1], "bindings": [
            {"name": "scale", "binding": {"kind": "uniform", "offset": 0, "size": 4, "used": 1}}]}]
    })";
    const auto loose = reflectShaderInterface(looseUniform, InterfaceBackend::d3d12, "main", InterfaceStage::compute);
    REQUIRE_FALSE(loose.has_value());
    CHECK(loose.error().find("constant buffer") != std::string::npos);

    constexpr std::string_view typedBuffer = R"({
        "parameters": [{"name": "texels", "binding": {"kind": "shaderResource", "index": 0},
                        "type": {"kind": "resource", "baseShape": "textureBuffer"}}],
        "entryPoints": [{"name": "main", "threadGroupSize": [1, 1, 1], "bindings": [
            {"name": "texels", "binding": {"kind": "shaderResource", "index": 0, "used": 1}}]}]
    })";
    CHECK_FALSE(
        reflectShaderInterface(typedBuffer, InterfaceBackend::vulkan, "main", InterfaceStage::compute).has_value());

    constexpr std::string_view sharedSlot = R"({
        "parameters": [
            {"name": "first", "binding": {"kind": "shaderResource", "index": 0},
             "type": {"kind": "resource", "baseShape": "texture2D"}},
            {"name": "second", "binding": {"kind": "unorderedAccess", "index": 0},
             "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}}],
        "entryPoints": [{"name": "main", "threadGroupSize": [1, 1, 1], "bindings": [
            {"name": "first", "binding": {"kind": "shaderResource", "index": 0, "used": 1}},
            {"name": "second", "binding": {"kind": "unorderedAccess", "index": 0, "used": 1}}]}]
    })";
    const auto shared = reflectShaderInterface(sharedSlot, InterfaceBackend::vulkan, "main", InterfaceStage::compute);
    REQUIRE_FALSE(shared.has_value());
    CHECK(shared.error().find("share set 0 binding 0") != std::string::npos);
}

TEST_CASE("Generated shader interface headers hold per-backend constexpr data",
          "[size:small][rhi][tooling][shader-interface]")
{
    auto vulkan =
        reflectShaderInterface(computeSpirvJson, InterfaceBackend::vulkan, "probeMain", InterfaceStage::compute);
    REQUIRE(vulkan.has_value());
    const std::array backends{BackendInterface{.backend = InterfaceBackend::vulkan, .reflected = *vulkan}};
    const auto header =
        rachis::shader_tool::generateShaderInterfaceHeader("Probe", InterfaceStage::compute, "probeMain", backends);
    REQUIRE(header.has_value());
    const auto contains = [&](std::string_view text) { return header->find(text) != std::string::npos; };
    CHECK(contains("namespace rachis_shaders::reflection::Probe"));
    CHECK(contains("inline constexpr rachis::ShaderStage stage = rachis::ShaderStage::compute;"));
    CHECK(contains("inline constexpr std::string_view entryPoint = \"probeMain\";"));
    CHECK(contains("{.set = 1, .binding = 0, .type = rachis::ShaderResourceType::sampledTexture, .count = 2}, // "
                   "colorTextures"));
    CHECK(contains("{{{.offset = 0, .size = 8}}};"));
    CHECK(contains("rachis::ShaderWorkgroupSize{8, 4, 1}"));
    CHECK(contains("inline constexpr std::uint32_t bindingSetCount = 2;"));
    CHECK(contains("{.binding = 4, .type = rachis::BindingType::sampler, .visibility = stage, .count = 1},"));
    CHECK(contains("{.type = rachis::BindingType::sampledTexture, .descriptorCount = 2},"));
    CHECK(contains("case rachis::Backend::vulkan:\n            return vulkan::shaderInterface();"));
    CHECK_FALSE(contains("rachis::Backend::d3d12"));
    CHECK_FALSE(contains("inline namespace"));

    CHECK_FALSE(rachis::shader_tool::generateShaderInterfaceHeader(
                    "not-an-identifier", InterfaceStage::compute, "probeMain", backends)
                    .has_value());

    const auto tagged = rachis::shader_tool::generateShaderInterfaceHeader(
        "Probe", InterfaceStage::compute, "probeMain", backends, "0a1b2c3d4e5f");
    REQUIRE(tagged.has_value());
    CHECK(tagged->find("namespace rachis_shaders::reflection::Probe\n{\ninline namespace t_0a1b2c3d4e5f\n{\n") !=
          std::string::npos);
    CHECK(tagged->find("} // inline namespace t_0a1b2c3d4e5f\n} // namespace rachis_shaders::reflection::Probe") !=
          std::string::npos);
    CHECK_FALSE(rachis::shader_tool::generateShaderInterfaceHeader(
                    "Probe", InterfaceStage::compute, "probeMain", backends, "not-a-tag")
                    .has_value());
}

namespace
{
    constexpr std::string_view layoutHlslJson = R"({
        "parameters": [
            {"name": "Constants", "binding": {"kind": "constantBuffer", "index": 0},
             "type": {"kind": "constantBuffer",
                      "elementVarLayout": {
                          "type": {"kind": "struct", "fields": [
                              {"name": "a", "type": {"kind": "scalar", "scalarType": "float32"},
                               "binding": {"kind": "uniform", "offset": 0, "size": 4}},
                              {"name": "weights", "type": {"kind": "array", "elementCount": 3,
                                                           "elementType": {"kind": "scalar", "scalarType": "float32"}},
                               "binding": {"kind": "uniform", "offset": 16, "size": 36, "elementStride": 16}},
                              {"name": "after", "type": {"kind": "scalar", "scalarType": "uint32"},
                               "binding": {"kind": "uniform", "offset": 52, "size": 4}},
                              {"name": "m", "type": {"kind": "matrix", "rowCount": 2, "columnCount": 3,
                                                     "elementType": {"kind": "scalar", "scalarType": "float32"}},
                               "binding": {"kind": "uniform", "offset": 64, "size": 40}}
                          ]},
                          "binding": {"kind": "uniform", "offset": 0, "size": 104}}}},
            {"name": "elements", "binding": {"kind": "shaderResource", "index": 1},
             "type": {"kind": "resource", "baseShape": "structuredBuffer",
                      "resultType": {"kind": "struct", "name": "Element", "fields": [
                          {"name": "position", "type": {"kind": "vector", "elementCount": 3,
                                                        "elementType": {"kind": "scalar", "scalarType": "float32"}},
                           "binding": {"kind": "uniform", "offset": 0, "size": 12}},
                          {"name": "flag", "type": {"kind": "scalar", "scalarType": "uint16"},
                           "binding": {"kind": "uniform", "offset": 12, "size": 2}}
                      ]}}}
        ],
        "entryPoints": [{"name": "main", "threadGroupSize": [1, 1, 1], "bindings": [
            {"name": "Constants", "binding": {"kind": "constantBuffer", "index": 0, "used": 1}},
            {"name": "elements", "binding": {"kind": "shaderResource", "index": 1, "used": 1}}]}]
    })";

} // namespace

TEST_CASE("Generated shader structs place each member at its reflected offset",
          "[size:small][rhi][tooling][shader-interface]")
{
    auto reflected = reflectShaderInterface(layoutHlslJson, InterfaceBackend::d3d12, "main", InterfaceStage::compute);
    REQUIRE(reflected.has_value());
    REQUIRE(reflected->dataResources.size() == 2);
    CHECK(reflected->dataResources[0].typeName == "Constants");
    CHECK(reflected->dataResources[1].typeName == "Element");
    REQUIRE(rachis::shader_tool::attachSlangDataLayouts(*reflected, layoutHlslJson, false).has_value());
    REQUIRE(reflected->blocks.size() == 2);
    CHECK(reflected->blocks[1].size == 16);

    const std::array backends{BackendInterface{.backend = InterfaceBackend::d3d12, .reflected = *reflected}};
    const auto header =
        rachis::shader_tool::generateShaderInterfaceHeader("Layout", InterfaceStage::compute, "main", backends);
    REQUIRE(header.has_value());
    const auto contains = [&](std::string_view text) { return header->find(text) != std::string::npos; };
    CHECK(contains("struct StridedArray final"));
    CHECK(contains("        struct Constants final\n"));
    CHECK(contains("            float a;\n            std::byte rachisPadding0[12];\n"));
    CHECK(contains("            rachis_shaders::layout::StridedArray<float, 3, 16> weights;\n"
                   "            std::uint32_t after;\n"));
    CHECK(contains("rachis_shaders::layout::StridedArray<std::array<float, 2>, 3, 16> m;"));
    CHECK(contains("static_assert(offsetof(Constants, after) == 52);"));
    CHECK(contains("static_assert(offsetof(Constants, m) == 64);"));
    CHECK(contains("static_assert(sizeof(Constants) == 104);"));
    CHECK(contains("static_assert(alignof(Constants) == 4);"));
    CHECK(contains("            std::uint16_t flag;\n        };\n"));
    CHECK(contains("static_assert(sizeof(Element) == 16);"));

    auto metal = reflectShaderInterface(layoutHlslJson, InterfaceBackend::metal, "main", InterfaceStage::compute);
    REQUIRE(metal.has_value());
    REQUIRE(rachis::shader_tool::attachSlangDataLayouts(*metal, layoutHlslJson, true).has_value());
    CHECK(metal->blocks[1].size == 16);
}

TEST_CASE("Generated shader structs reject layouts C++ cannot express", "[size:small][rhi][tooling][shader-interface]")
{
    std::string rowMajor(layoutHlslJson);
    const std::string columnMajorSize = R"("offset": 64, "size": 40)";
    rowMajor.replace(rowMajor.find(columnMajorSize), columnMajorSize.size(), R"("offset": 64, "size": 28)");
    auto reflected = reflectShaderInterface(rowMajor, InterfaceBackend::d3d12, "main", InterfaceStage::compute);
    REQUIRE(reflected.has_value());
    const auto attached = rachis::shader_tool::attachSlangDataLayouts(*reflected, rowMajor, false);
    REQUIRE_FALSE(attached.has_value());
    CHECK(attached.error().find("row_major") != std::string::npos);

    std::string overlap(layoutHlslJson);
    const std::string afterOffset = R"("offset": 52, "size": 4)";
    overlap.replace(overlap.find(afterOffset), afterOffset.size(), R"("offset": 40, "size": 4)");
    auto overlapping = reflectShaderInterface(overlap, InterfaceBackend::d3d12, "main", InterfaceStage::compute);
    REQUIRE(overlapping.has_value());
    REQUIRE(rachis::shader_tool::attachSlangDataLayouts(*overlapping, overlap, false).has_value());
    const std::array backends{BackendInterface{.backend = InterfaceBackend::d3d12, .reflected = *overlapping}};
    const auto header =
        rachis::shader_tool::generateShaderInterfaceHeader("Layout", InterfaceStage::compute, "main", backends);
    REQUIRE_FALSE(header.has_value());
    CHECK(header.error().find("overlaps") != std::string::npos);
}

namespace
{
    constexpr std::string_view drawIndexHlslJson = R"({
        "parameters": [
            {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": 5, "index": 0},
             "type": {"kind": "constantBuffer",
                      "elementVarLayout": {
                          "type": {"kind": "struct", "fields": [
                              {"name": "rachisDrawIndexValue", "type": {"kind": "scalar", "scalarType": "uint32"},
                               "binding": {"kind": "uniform", "offset": 0, "size": 4}}]},
                          "binding": {"kind": "uniform", "offset": 0, "size": 4}}}},
            {"name": "transforms", "binding": {"kind": "shaderResource", "index": 1},
             "type": {"kind": "resource", "baseShape": "structuredBuffer", "resultType": {"kind": "scalar"}}}
        ],
        "entryPoints": [
            {"name": "vertexMain", "stage": "vertex", "parameters": [],
             "bindings": [
                {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": 5, "index": 0, "used": 1}},
                {"name": "transforms", "binding": {"kind": "shaderResource", "index": 1, "used": 1}}
             ]}
        ]
    })";

    constexpr std::string_view unusedDrawIndexJson = R"({
        "parameters": [
            {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": 5, "index": 0},
             "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [
            {"name": "vertexMain", "stage": "vertex", "parameters": [],
             "bindings": [
                {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": 5, "index": 0, "used": 0}}
             ]}
        ]
    })";

    [[nodiscard]] std::vector<std::byte> vertexModuleWithBuiltIn(std::uint32_t builtIn, bool listed)
    {
        constexpr std::uint32_t mainName = 0x6E69616DU;
        std::vector<std::uint32_t> words{
            0x07230203U,
            0x00010600U,
            0U,
            9U,
            0U,
            (2U << 16U) | 17U,
            1U,
            (2U << 16U) | 17U,
            4427U,
            (3U << 16U) | 14U,
            0U,
            1U,
        };
        if (listed)
        {
            words.insert(words.end(), {(6U << 16U) | 15U, 0U, 1U, mainName, 0U, 2U});
        }
        else
        {
            words.insert(words.end(), {(5U << 16U) | 15U, 0U, 1U, mainName, 0U});
        }
        words.insert(words.end(),
                     {
                         (4U << 16U) | 71U,
                         2U,
                         11U,
                         builtIn,
                         (4U << 16U) | 21U,
                         3U,
                         32U,
                         0U,
                         (4U << 16U) | 32U,
                         4U,
                         1U,
                         3U,
                         (4U << 16U) | 59U,
                         4U,
                         2U,
                         1U,
                         (2U << 16U) | 19U,
                         5U,
                         (3U << 16U) | 33U,
                         6U,
                         5U,
                         (5U << 16U) | 54U,
                         5U,
                         1U,
                         0U,
                         6U,
                         (2U << 16U) | 248U,
                         7U,
                         (4U << 16U) | 61U,
                         3U,
                         8U,
                         2U,
                         (1U << 16U) | 253U,
                         (1U << 16U) | 56U,
                     });
        std::vector<std::byte> bytes(words.size() * sizeof(std::uint32_t));
        std::memcpy(bytes.data(), words.data(), bytes.size());
        return bytes;
    }
    [[nodiscard]] std::vector<std::byte> workgroupModuleWithDrawIndex(std::uint32_t executionModel)
    {
        constexpr std::uint32_t mainName = 0x6E69616DU;
        const std::vector<std::uint32_t> words{
            0x07230203U,
            0x00010600U,
            0U,
            9U,
            0U,
            (2U << 16U) | 17U,
            1U,
            (2U << 16U) | 17U,
            4427U,
            (3U << 16U) | 14U,
            0U,
            1U,
            (6U << 16U) | 15U,
            executionModel,
            1U,
            mainName,
            0U,
            2U,
            (6U << 16U) | 16U,
            1U,
            17U,
            4U,
            2U,
            1U,
            (4U << 16U) | 71U,
            2U,
            11U,
            4426U,
            (4U << 16U) | 21U,
            3U,
            32U,
            0U,
            (4U << 16U) | 32U,
            4U,
            1U,
            3U,
            (4U << 16U) | 59U,
            4U,
            2U,
            1U,
            (2U << 16U) | 19U,
            5U,
            (3U << 16U) | 33U,
            6U,
            5U,
            (5U << 16U) | 54U,
            5U,
            1U,
            0U,
            6U,
            (2U << 16U) | 248U,
            7U,
            (4U << 16U) | 61U,
            3U,
            8U,
            2U,
            (1U << 16U) | 253U,
            (1U << 16U) | 56U,
        };
        std::vector<std::byte> bytes(words.size() * sizeof(std::uint32_t));
        std::memcpy(bytes.data(), words.data(), bytes.size());
        return bytes;
    }

    [[nodiscard]] std::vector<std::byte> computeModuleWithSamplerInSet(std::uint32_t set)
    {
        constexpr std::uint32_t mainName = 0x6E69616DU;
        const std::vector<std::uint32_t> words{
            0x07230203U,
            0x00010600U,
            0U,
            6U,
            0U,
            (2U << 16U) | 17U,
            1U,
            (3U << 16U) | 14U,
            0U,
            1U,
            (6U << 16U) | 15U,
            5U,
            1U,
            mainName,
            0U,
            4U,
            (6U << 16U) | 16U,
            1U,
            17U,
            1U,
            1U,
            1U,
            (4U << 16U) | 71U,
            4U,
            34U,
            set,
            (4U << 16U) | 71U,
            4U,
            33U,
            0U,
            (2U << 16U) | 26U,
            2U,
            (4U << 16U) | 32U,
            3U,
            0U,
            2U,
            (4U << 16U) | 59U,
            3U,
            4U,
            0U,
        };
        std::vector<std::byte> bytes(words.size() * sizeof(std::uint32_t));
        std::memcpy(bytes.data(), words.data(), bytes.size());
        return bytes;
    }

    [[nodiscard]] std::vector<std::byte> pnmsShader(std::string_view json)
    {
        constexpr std::array metallib{std::byte{'M'}, std::byte{'T'}, std::byte{'L'}, std::byte{'B'}, std::byte{0}};
        auto packed = rachis::shader_tool::packMetalShaderManifest(
            std::span(reinterpret_cast<const std::byte*>(json.data()), json.size()), metallib);
        REQUIRE(packed.has_value());
        return *packed;
    }

    [[nodiscard]] std::vector<std::byte> dxilContainer(std::span<const std::array<std::uint32_t, 4>> resources)
    {
        std::vector<std::uint32_t> psv{0U, static_cast<std::uint32_t>(resources.size())};
        if (!resources.empty())
        {
            psv.push_back(16U);
            for (const auto& resource : resources)
            {
                psv.insert(psv.end(), resource.begin(), resource.end());
            }
        }
        const auto partSize = static_cast<std::uint32_t>(psv.size() * sizeof(std::uint32_t));
        constexpr std::uint32_t partOffset = 36;
        std::vector<std::uint32_t> words{
            0x43425844U, 0U, 0U, 0U, 0U, 1U, partOffset + 8U + partSize, 1U, partOffset, 0x30565350U, partSize};
        words.insert(words.end(), psv.begin(), psv.end());
        std::vector<std::byte> bytes(words.size() * sizeof(std::uint32_t));
        std::memcpy(bytes.data(), words.data(), bytes.size());
        return bytes;
    }
} // namespace

TEST_CASE("Shader interface reflection reads the draw index from each backend's reflection",
          "[size:small][rhi][tooling][shader-interface][draw-index]")
{
    for (const auto backend : {InterfaceBackend::d3d12, InterfaceBackend::metal})
    {
        INFO("backend " << rachis::shader_tool::interfaceBackendName(backend));
        auto reflected = reflectShaderInterface(drawIndexHlslJson, backend, "vertexMain", InterfaceStage::vertex);
        REQUIRE(reflected.has_value());
        CHECK(reflected->drawIndex);
        const std::vector<ReflectedBinding> expected{
            {.name = "transforms", .set = 0, .binding = 1, .type = "readOnlyStorageBuffer", .count = 1},
        };
        CHECK(reflected->bindings == expected);
        CHECK(std::ranges::none_of(reflected->dataResources,
                                   [](const auto& resource) { return resource.name == "RachisDrawIndex"; }));

        const auto unused = reflectShaderInterface(unusedDrawIndexJson, backend, "vertexMain", InterfaceStage::vertex);
        REQUIRE(unused.has_value());
        CHECK_FALSE(unused->drawIndex);
        CHECK(unused->bindings.empty());
    }

    const auto vulkan =
        reflectShaderInterface(drawIndexHlslJson, InterfaceBackend::vulkan, "vertexMain", InterfaceStage::vertex);
    REQUIRE_FALSE(vulkan.has_value());
    CHECK(vulkan.error().find("register space 5") != std::string::npos);

    const auto drawIndexModule = vertexModuleWithBuiltIn(4426U, true);
    const auto reads = rachis::shader_tool::spirvReadsDrawIndex(drawIndexModule, "main", InterfaceStage::vertex);
    REQUIRE(reads.has_value());
    CHECK(*reads);
    const auto spirv =
        rachis::shader_tool::reflectSpirvShaderInterface(drawIndexModule, "main", InterfaceStage::vertex);
    REQUIRE(spirv.has_value());
    CHECK(spirv->drawIndex);
    CHECK(spirv->vertexInputLocations.empty());

    const auto baseInstance =
        rachis::shader_tool::spirvReadsDrawIndex(vertexModuleWithBuiltIn(4425U, true), "main", InterfaceStage::vertex);
    REQUIRE(baseInstance.has_value());
    CHECK_FALSE(*baseInstance);
    const auto unlisted =
        rachis::shader_tool::spirvReadsDrawIndex(vertexModuleWithBuiltIn(4426U, false), "main", InterfaceStage::vertex);
    REQUIRE(unlisted.has_value());
    CHECK_FALSE(*unlisted);
    CHECK_FALSE(rachis::shader_tool::spirvReadsDrawIndex(drawIndexModule, "other", InterfaceStage::vertex).has_value());
    CHECK_FALSE(rachis::shader_tool::spirvReadsDrawIndex(drawIndexModule, "main", InterfaceStage::pixel).has_value());

    const std::array drawIndexResources{std::array<std::uint32_t, 4>{4U, 0U, 1U, 1U},
                                        std::array<std::uint32_t, 4>{2U, 5U, 0U, 0U}};
    const auto dxilReads = rachis::shader_tool::dxilReadsDrawIndex(dxilContainer(drawIndexResources));
    REQUIRE(dxilReads.has_value());
    CHECK(*dxilReads);
    const std::array otherResources{std::array<std::uint32_t, 4>{2U, 4U, 0U, 0U},
                                    std::array<std::uint32_t, 4>{2U, 5U, 1U, 1U}};
    const auto dxilOther = rachis::shader_tool::dxilReadsDrawIndex(dxilContainer(otherResources));
    REQUIRE(dxilOther.has_value());
    CHECK_FALSE(*dxilOther);
    const auto dxilEmpty = rachis::shader_tool::dxilReadsDrawIndex(dxilContainer({}));
    REQUIRE(dxilEmpty.has_value());
    CHECK_FALSE(*dxilEmpty);
    CHECK_FALSE(rachis::shader_tool::dxilReadsDrawIndex(drawIndexModule).has_value());
    auto truncated = dxilContainer(drawIndexResources);
    truncated.resize(truncated.size() - 8);
    CHECK_FALSE(rachis::shader_tool::dxilReadsDrawIndex(truncated).has_value());

    const auto pixel =
        reflectShaderInterface(drawIndexHlslJson, InterfaceBackend::d3d12, "vertexMain", InterfaceStage::pixel);
    REQUIRE(pixel.has_value());
    CHECK_FALSE(pixel->drawIndex);
    CHECK(pixel->bindings.size() == 1);

    auto d3d12 =
        reflectShaderInterface(drawIndexHlslJson, InterfaceBackend::d3d12, "vertexMain", InterfaceStage::vertex);
    REQUIRE(d3d12.has_value());
    const auto vulkanWithoutDrawIndex =
        reflectShaderInterface(unusedDrawIndexJson, InterfaceBackend::vulkan, "vertexMain", InterfaceStage::vertex);
    REQUIRE(vulkanWithoutDrawIndex.has_value());
    const std::array backends{
        BackendInterface{.backend = InterfaceBackend::d3d12, .reflected = *d3d12},
        BackendInterface{.backend = InterfaceBackend::vulkan, .reflected = *vulkanWithoutDrawIndex},
    };
    const auto header = rachis::shader_tool::generateShaderInterfaceHeader(
        "DrawIndexProbe", InterfaceStage::vertex, "vertexMain", backends);
    REQUIRE(header.has_value());
    const auto d3d12Namespace = header->find("namespace d3d12");
    const auto vulkanNamespace = header->find("namespace vulkan");
    REQUIRE(d3d12Namespace != std::string::npos);
    REQUIRE(vulkanNamespace != std::string::npos);
    CHECK(header->find("inline constexpr bool drawIndex = true;", d3d12Namespace) < vulkanNamespace);
    CHECK(header->find("inline constexpr bool drawIndex = false;", vulkanNamespace) != std::string::npos);
    CHECK(header->find("                .drawIndex = drawIndex,\n") != std::string::npos);
    CHECK(header->find("constexpr bool drawIndex(rachis::Backend backend) noexcept") != std::string::npos);
    CHECK(header->find("case rachis::Backend::d3d12:\n            return d3d12::drawIndex;") != std::string::npos);
    CHECK(header->find("RachisDrawIndex") == std::string::npos);
}
TEST_CASE("Shader interface reflection reads task and mesh entry points and their draw index",
          "[size:small][rhi][tooling][shader-interface][draw-index]")
{
    CHECK(rachis::shader_tool::parseInterfaceStage("task") == InterfaceStage::task);
    CHECK(rachis::shader_tool::parseInterfaceStage("mesh") == InterfaceStage::mesh);
    CHECK_FALSE(rachis::shader_tool::parseInterfaceStage("amplification").has_value());

    for (const auto stage : {InterfaceStage::task, InterfaceStage::mesh})
    {
        const auto name = rachis::shader_tool::interfaceStageName(stage);
        INFO("stage " << name);
        const std::string json = std::string(R"({"parameters": [
            {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": 5, "index": 0},
             "type": {"kind": "constantBuffer", "elementType": {"kind": "struct", "fields": [
                 {"name": "rachisDrawIndexValue", "type": {"kind": "scalar", "scalarType": "uint32"},
              "binding": {"kind": "uniform", "offset": 0, "size": 4}}]}}}],
            "entryPoints": [{"name": "main", "stage": ")") +
                                 (stage == InterfaceStage::task ? "amplification" : "mesh") +
                                 R"(", "threadGroupSize": [4, 2, 1], "bindings": [
                {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": 5, "index": 0, "used": 1}}
            ]}]})";
        const auto reflected = reflectShaderInterface(json, InterfaceBackend::d3d12, "main", stage);
        REQUIRE(reflected.has_value());
        CHECK(reflected->drawIndex);
        CHECK(reflected->workgroupSize == std::array<std::uint32_t, 3>{4, 2, 1});

        std::string withoutWorkgroup = json;
        withoutWorkgroup.erase(withoutWorkgroup.find(R"("threadGroupSize": [4, 2, 1], )"),
                               std::string_view(R"("threadGroupSize": [4, 2, 1], )").size());
        const auto missing = reflectShaderInterface(withoutWorkgroup, InterfaceBackend::metal, "main", stage);
        REQUIRE_FALSE(missing.has_value());
        CHECK(missing.error().find("thread-group size") != std::string::npos);

        const auto model = stage == InterfaceStage::task ? 5364U : 5365U;
        const auto module = workgroupModuleWithDrawIndex(model);
        const auto reads = rachis::shader_tool::spirvReadsDrawIndex(module, "main", stage);
        REQUIRE(reads.has_value());
        CHECK(*reads);
        const auto spirv = rachis::shader_tool::reflectSpirvShaderInterface(module, "main", stage);
        REQUIRE(spirv.has_value());
        CHECK(spirv->drawIndex);
        CHECK(spirv->workgroupSize == std::array<std::uint32_t, 3>{4, 2, 1});
        CHECK_FALSE(rachis::shader_tool::spirvReadsDrawIndex(module, "main", InterfaceStage::vertex).has_value());

        const std::array backends{BackendInterface{.backend = InterfaceBackend::d3d12, .reflected = *reflected}};
        const auto header = rachis::shader_tool::generateShaderInterfaceHeader("StageProbe", stage, "main", backends);
        REQUIRE(header.has_value());
        CHECK(header->contains("rachis::ShaderStage::" + std::string(name) + ";"));
        CHECK(header->contains("rachis::ShaderWorkgroupSize{4, 2, 1}"));
        CHECK(header->contains("inline constexpr bool drawIndex = true;"));
    }
}

TEST_CASE("Shader interface reflection reserves register space 5 for the draw index",
          "[size:small][rhi][tooling][shader-interface][draw-index]")
{
    const auto vertexJson = [](std::string_view parameter) {
        return std::string(R"({"parameters": [)") + std::string(parameter) +
               R"(], "entryPoints": [{"name": "vertexMain", "stage": "vertex", "parameters": []}]})";
    };
    constexpr std::string_view drawIndex = R"(
        {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": 5, "index": 0},
         "type": {"kind": "constantBuffer", "elementType": {"kind": "struct", "fields": [
             {"name": "rachisDrawIndexValue", "type": {"kind": "scalar", "scalarType": "uint32"},
              "binding": {"kind": "uniform", "offset": 0, "size": 4}}]}}})";
    for (const auto backend : {InterfaceBackend::d3d12, InterfaceBackend::metal})
    {
        INFO("backend " << rachis::shader_tool::interfaceBackendName(backend));
        const auto reflected =
            reflectShaderInterface(vertexJson(drawIndex), backend, "vertexMain", InterfaceStage::vertex);
        REQUIRE(reflected.has_value());
        CHECK(reflected->drawIndex);
        CHECK(reflected->bindings.empty());
    }
    const auto vulkan =
        reflectShaderInterface(vertexJson(drawIndex), InterfaceBackend::vulkan, "vertexMain", InterfaceStage::vertex);
    REQUIRE_FALSE(vulkan.has_value());
    CHECK(vulkan.error().find("register space 5") != std::string::npos);

    const std::array rejected{
        std::string(R"(
        {"name": "Offset", "binding": {"kind": "constantBuffer", "space": 5, "index": 0},
         "type": {"kind": "constantBuffer", "elementType": {"kind": "struct", "fields": [
             {"name": "index", "type": {"kind": "scalar", "scalarType": "uint32"},
              "binding": {"kind": "uniform", "offset": 16, "size": 4}}]}}})"),
        std::string(R"(
        {"name": "Other", "binding": {"kind": "constantBuffer", "space": 5, "index": 1},
         "type": {"kind": "constantBuffer", "elementType": {"kind": "struct", "fields": [
             {"name": "value", "type": {"kind": "scalar", "scalarType": "uint32"}}]}}})"),
        std::string(R"(
        {"name": "Wide", "binding": {"kind": "constantBuffer", "space": 5, "index": 0},
         "type": {"kind": "constantBuffer", "elementType": {"kind": "struct", "fields": [
             {"name": "data", "type": {"kind": "vector", "elementCount": 4,
                                       "elementType": {"kind": "scalar", "scalarType": "uint32"}}}]}}})"),
        std::string(R"(
        {"name": "Records", "binding": {"kind": "shaderResource", "space": 5, "index": 0},
         "type": {"kind": "resource", "baseShape": "structuredBuffer", "resultType": {"kind": "scalar"}}})"),
    };
    for (const auto& parameter : rejected)
    {
        INFO(parameter);
        const auto result = reflectShaderInterface(
            vertexJson(parameter), InterfaceBackend::d3d12, "vertexMain", InterfaceStage::vertex);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().find("register space 5") != std::string::npos);
    }
}

TEST_CASE("PNMS draw-index reflection checks the envelope and reads only the selected entry point",
          "[size:small][rhi][tooling][shader-interface][draw-index]")
{
    using rachis::shader_tool::metalManifestReadsDrawIndex;
    constexpr std::string_view json = R"({"parameters": [],
        "entryPoints": [
            {"name": "drawn", "stage": "vertex", "bindings": [
                {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": 5, "index": 0, "used": 1}}]},
            {"name": "plain", "stage": "vertex", "bindings": [
                {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": 5, "index": 0, "used": 0}}]},
            {"name": "meshed", "stage": "mesh", "threadGroupSize": [1, 1, 1], "bindings": [
                {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": 5, "index": 0}}]},
            {"name": "empty", "stage": "fragment"}
        ],
        "rachisBindings": [{"name": "RachisDrawIndex", "drawIndex": true}]})";
    const auto pnms = pnmsShader(json);
    const auto drawn = metalManifestReadsDrawIndex(pnms, "drawn", InterfaceStage::vertex);
    REQUIRE(drawn.has_value());
    CHECK(*drawn);
    const auto plain = metalManifestReadsDrawIndex(pnms, "plain", InterfaceStage::vertex);
    REQUIRE(plain.has_value());
    CHECK_FALSE(*plain);
    const auto meshed = metalManifestReadsDrawIndex(pnms, "meshed", InterfaceStage::mesh);
    REQUIRE(meshed.has_value());
    CHECK(*meshed);
    const auto empty = metalManifestReadsDrawIndex(pnms, "empty", InterfaceStage::pixel);
    REQUIRE(empty.has_value());
    CHECK_FALSE(*empty);
    CHECK_FALSE(metalManifestReadsDrawIndex(pnms, "missing", InterfaceStage::vertex).has_value());
    CHECK_FALSE(metalManifestReadsDrawIndex(pnms, "drawn", InterfaceStage::mesh).has_value());

    const auto truncated = std::span(pnms).first(pnms.size() - 5);
    CHECK_FALSE(metalManifestReadsDrawIndex(truncated, "drawn", InterfaceStage::vertex).has_value());
    auto longer = pnms;
    longer.push_back(std::byte{0});
    CHECK_FALSE(metalManifestReadsDrawIndex(longer, "drawn", InterfaceStage::vertex).has_value());
    auto version = pnms;
    version[4] = std::byte{2};
    CHECK_FALSE(metalManifestReadsDrawIndex(version, "drawn", InterfaceStage::vertex).has_value());
    auto library = pnms;
    library[pnms.size() - 5] = std::byte{'X'};
    CHECK_FALSE(metalManifestReadsDrawIndex(library, "drawn", InterfaceStage::vertex).has_value());
    const std::string duplicated(R"({"entryPoints": [{"name": "drawn", "stage": "vertex"},
                                                    {"name": "drawn", "stage": "vertex"}]})");
    CHECK_FALSE(metalManifestReadsDrawIndex(pnmsShader(duplicated), "drawn", InterfaceStage::vertex).has_value());
}

TEST_CASE("Generated binding layouts of unbounded arrays pass binding layout validation",
          "[size:small][rhi][tooling][shader-interface]")
{
    namespace generated = rachis_shaders::reflection::GeneratedBindingArray;
    std::size_t backends = 0;
    for (const auto backend : {rachis::Backend::d3d12, rachis::Backend::vulkan, rachis::Backend::metal})
    {
        if (rachis_shaders::GeneratedBindingArray(backend).empty())
        {
            continue;
        }
        ++backends;
        INFO("backend " << static_cast<int>(backend));
        REQUIRE(generated::bindingSetCount(backend) == 2);
        const auto entries = generated::bindingLayout(backend, 1);
        REQUIRE(entries.size() == 1);
        CHECK(entries[0].count == rachis::unboundedBindingCount);
        CHECK(entries[0].flags == rachis::BindingFlag::variableDescriptorCount);
        const auto valid = rachis::detail::validateBindingLayoutDesc(rachis::BindingLayoutDesc{.entries = entries});
        INFO((valid ? std::string{} : valid.error().message));
        CHECK(valid.has_value());
        CHECK(generated::bindingPoolSizes(backend, 1).empty());
    }
    CHECK(backends > 0);
}

TEST_CASE("Shader interface reflection rejects resources outside binding sets 0 to 3",
          "[size:small][rhi][tooling][shader-interface]")
{
    const auto inSetThree = rachis::shader_tool::reflectSpirvShaderInterface(
        computeModuleWithSamplerInSet(3), "main", InterfaceStage::compute);
    INFO((inSetThree ? std::string{} : inSetThree.error()));
    REQUIRE(inSetThree.has_value());
    REQUIRE(inSetThree->bindings.size() == 1);
    CHECK(inSetThree->bindings[0].set == 3);

    for (const std::uint32_t set : {4U, 5U})
    {
        INFO("set " << set);
        const auto rejected = rachis::shader_tool::reflectSpirvShaderInterface(
            computeModuleWithSamplerInSet(set), "main", InterfaceStage::compute);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().find("binding set " + std::to_string(set)) != std::string::npos);
    }

    const auto computeJson = [](std::string_view binding) {
        return std::string(R"({"parameters": [{"name": "table", "binding": )") + std::string(binding) +
               R"(, "type": {"kind": "resource", "baseShape": "texture2D"}}],
            "entryPoints": [{"name": "main", "stage": "compute", "threadGroupSize": [1, 1, 1]}]})";
    };
    const auto vulkan =
        reflectShaderInterface(computeJson(R"({"kind": "descriptorTableSlot", "space": 6, "index": 0})"),
                               InterfaceBackend::vulkan,
                               "main",
                               InterfaceStage::compute);
    REQUIRE_FALSE(vulkan.has_value());
    CHECK(vulkan.error().find("binding set 6") != std::string::npos);
    const auto d3d12 = reflectShaderInterface(computeJson(R"({"kind": "shaderResource", "space": 4, "index": 0})"),
                                              InterfaceBackend::d3d12,
                                              "main",
                                              InterfaceStage::compute);
    REQUIRE_FALSE(d3d12.has_value());
    CHECK(d3d12.error().find("binding set 4") != std::string::npos);
}
