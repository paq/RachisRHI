#include "Internal/MetalShaderManifest.h"
#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
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

    [[nodiscard]] std::vector<std::byte> readBytes(const std::filesystem::path& path)
    {
        INFO("shader artifact=" << path.string());
        std::ifstream stream(path, std::ios::binary);
        REQUIRE(stream);
        stream.seekg(0, std::ios::end);
        const auto size = static_cast<std::size_t>(stream.tellg());
        stream.seekg(0, std::ios::beg);
        std::vector<std::byte> bytes(size);
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
        REQUIRE(stream);
        return bytes;
    }

    [[nodiscard]] std::filesystem::path pnmsPath(std::string_view metallibName)
    {
        return std::filesystem::path(RACHIS_METALLIB_DIR) /
               std::filesystem::path(metallibName).replace_extension(".pnms");
    }

    [[nodiscard]] std::vector<std::byte> packedArtifact(std::string_view metallibName)
    {
        return readBytes(pnmsPath(metallibName));
    }

} // namespace

TEST_CASE("PNMS identity rejects truncated metadata and raw metallib", "[size:medium][rhi][metal][shader][module]")
{
    const auto packed = packedArtifact("BootstrapTriangle.vs.metallib");
    auto payload = rachis::metal::unpackShaderManifest(packed);
    REQUIRE(payload.has_value());
    CHECK(payload->metallib.size() >= 4);

    CHECK_FALSE(rachis::metal::unpackShaderManifest(payload->metallib).has_value());
    auto truncated = packed;
    truncated.resize(rachis::metal::shaderManifestHeaderBytes - 1);
    CHECK_FALSE(rachis::metal::unpackShaderManifest(truncated).has_value());
    truncated = packed;
    truncated[8] = std::byte{0};
    truncated[9] = std::byte{0};
    truncated[10] = std::byte{0};
    truncated[11] = std::byte{0};
    CHECK_FALSE(rachis::metal::unpackShaderManifest(truncated).has_value());

    auto legacy = packed;
    legacy[0] = std::byte{'R'};
    legacy[1] = std::byte{'X'};
    auto rejected = rachis::metal::unpackShaderManifest(legacy);
    REQUIRE_FALSE(rejected.has_value());
}

#if RACHIS_ENABLE_METAL
TEST_CASE("Metal createShaderModule validates portable layouts before native use",
          "[size:large][rhi][metal][shader][module]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto triangle = packedArtifact("BootstrapTriangle.vs.metallib");
    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangle,
        .entryPoint = "vertexMain",
        .label = "triangle vs",
    });
    REQUIRE(vertex.has_value());
    auto snapshot = device.shaderModuleInterface(*vertex);
    REQUIRE(snapshot.has_value());
    CHECK(snapshot->stage == rachis::ShaderStage::vertex);
    REQUIRE(device.destroyShaderModule(*vertex));
    CHECK_FALSE(device.shaderModuleInterface(*vertex).has_value());

    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::vertex,
                        .bytecode = triangle,
                        .entryPoint = "missingMain",
                    })
                    .has_value());

    const auto cube = packedArtifact("BootstrapCube.cs.metallib");
    auto cubePayload = rachis::metal::unpackShaderManifest(cube);
    REQUIRE(cubePayload.has_value());
    const auto cubeJson =
        std::string_view(reinterpret_cast<const char*>(cubePayload->json.data()), cubePayload->json.size());
    auto cubeLayout = rachis::metal::parseSlangReflectionJson(cubeJson, "computeMain", rachis::ShaderStage::compute);
    REQUIRE(cubeLayout.has_value());
    CHECK(cubeLayout->pushConstantBytes == 16);
    REQUIRE(cubeLayout->workgroupSize.has_value());
    CHECK(cubeLayout->workgroupSize->x == 64);

    const std::array bindings{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::readOnlyStorageBuffer, 1},
        rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array push{rachis::ShaderPushConstantRequirement{0, 16}};
    auto compute = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface =
            {
                .bindings = bindings,
                .pushConstants = push,
                .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
            },
    });
    if (!compute)
    {
        INFO(compute.error().message);
    }
    REQUIRE(compute.has_value());

    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = cube,
                        .entryPoint = "computeMain",
                    })
                    .has_value());

    auto truncatedPush = push;
    truncatedPush[0].size = 4;
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = cube,
                        .entryPoint = "computeMain",
                        .shaderInterface =
                            {
                                .bindings = bindings,
                                .pushConstants = truncatedPush,
                                .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
                            },
                    })
                    .has_value());

    auto oversizedPush = push;
    oversizedPush[0].size = 32;
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = cube,
                        .entryPoint = "computeMain",
                        .shaderInterface =
                            {
                                .bindings = bindings,
                                .pushConstants = oversizedPush,
                                .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
                            },
                    })
                    .has_value());

    const std::array colliding{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampler, 1},
    };
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = cube,
                        .entryPoint = "computeMain",
                        .shaderInterface =
                            {
                                .bindings = colliding,
                                .pushConstants = push,
                                .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
                            },
                    })
                    .has_value());

    const auto fixture = packedArtifact("LayoutFixture.cs.metallib");
    const std::array fixtureBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto layoutModule = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = fixture,
        .entryPoint = "computeMain",
        .shaderInterface = {.bindings = fixtureBindings, .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
    });
    REQUIRE(layoutModule.has_value());
}

TEST_CASE("Metal vertex module ignores unused file-scope texture bindings", "[size:large][rhi][metal][shader][module]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto vertexShader = packedArtifact("MetalUnusedBindings.vs.metallib");
    auto payload = rachis::metal::unpackShaderManifest(vertexShader);
    REQUIRE(payload.has_value());
    const auto json = std::string_view(reinterpret_cast<const char*>(payload->json.data()), payload->json.size());
    auto layout = rachis::metal::parseSlangReflectionJson(json, "vertexMain", rachis::ShaderStage::vertex);
    REQUIRE(layout.has_value());
    CHECK(layout->pushConstantBytes == 16);
    CHECK(layout->resources.empty());

    const std::array push{rachis::ShaderPushConstantRequirement{0, 16}};
    auto module = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexShader,
        .entryPoint = "vertexMain",
        .shaderInterface = {.pushConstants = push},
    });
    if (!module)
    {
        INFO(module.error().message);
    }
    REQUIRE(module.has_value());
}
#endif

TEST_CASE("Metal reflection parses compact JSON and scopes usage to the selected entry",
          "[size:small][rhi][metal][reflection][regression]")
{
    const std::string_view json =
        R"({"parameters":[{"name":"dispatch\"data","type":{"kind":"resource","baseShape":"structuredBuffer","access":"readWrite"}},{"name":"unused","type":{"kind":"resource","baseShape":"texture2D"}}],"entryPoints":[{"name":"vertexMain","stage":"vertex","bindings":[{"name":"dispatch\"data","bindings":[{"kind":"constantBuffer","index":3,"used":true},{"kind":"unorderedAccess","index":0,"count":0,"used":false}]},{"name":"unused","binding":{"kind":"shaderResource","index":0,"used":false}}]},{"name":"computeMain","stage":"compute","threadGroupSize":[8,4,1]}]})";
    auto vertex = rachis::metal::parseSlangReflectionJson(json, "vertexMain", rachis::ShaderStage::vertex);
    REQUIRE(vertex);
    CHECK_FALSE(vertex->workgroupSize);
    REQUIRE(vertex->resources.size() == 1);
    CHECK(vertex->resources[0].name == "dispatch\"data");
    CHECK(vertex->resources[0].metalIndex == 3);
    CHECK(vertex->resources[0].type == rachis::ShaderResourceType::storageBuffer);
    CHECK_FALSE(rachis::metal::parseSlangReflectionJson(json, "computeMain", rachis::ShaderStage::vertex));
    const auto compute = rachis::metal::parseSlangReflectionJson(json, "computeMain", rachis::ShaderStage::compute);
    REQUIRE(compute);
    REQUIRE(compute->workgroupSize);
    CHECK(compute->workgroupSize->x == 8);
    CHECK(compute->workgroupSize->y == 4);
}

TEST_CASE("Metal reflection rejects malformed JSON and invalid numeric metadata",
          "[size:small][rhi][metal][reflection][regression]")
{
    const std::array invalid{
        R"({"entryPoints":[{"name":"main","stage":"compute","threadGroupSize":[1,1,1]}])",
        R"({"entryPoints":[{"name":"main","stage":"unknown","threadGroupSize":[1,1,1]}]})",
        R"({"entryPoints":[{"name":"main","stage":"compute","threadGroupSize":[4294967297,1,1]}]})",
        R"({"entryPoints":[{"name":"main","stage":"compute","threadGroupSize":[1,0,1]}]})",
        R"({"entryPoints":[{"name":"main","stage":"compute","threadGroupSize":[true,1,1]}]})",
        R"({"entryPoints":[{"name":"main","stage":"compute"},{"name":"other","stage":"compute","threadGroupSize":[1,1,1]}]})",
    };
    for (const auto* json : invalid)
    {
        CAPTURE(json);
        CHECK_FALSE(rachis::metal::parseSlangReflectionJson(json, "main", rachis::ShaderStage::compute));
    }
}

TEST_CASE("Metal shader interfaces reject wrong buffer slots and incomplete push coverage",
          "[size:small][rhi][metal][reflection][regression]")
{
    rachis::metal::MetalReflectedLayout const reflected{
        .stage = rachis::ShaderStage::compute,
        .resources = {{.name = "output", .type = rachis::ShaderResourceType::storageBuffer, .metalIndex = 2}},
        .pushConstantBytes = 16,
    };
    const std::array correct{rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::storageBuffer, 1}};
    const std::array wrong{rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1}};
    const std::array full{rachis::ShaderPushConstantRequirement{0, 16}};
    REQUIRE(rachis::metal::validateShaderInterface(reflected, {.bindings = correct, .pushConstants = full}));
    const std::array setThree{rachis::ShaderBindingRequirement{3, 2, rachis::ShaderResourceType::storageBuffer, 1}};
    CHECK(rachis::metal::validateShaderInterface(reflected, {.bindings = setThree, .pushConstants = full}));
    const std::array ambiguous{correct[0], setThree[0]};
    CHECK_FALSE(rachis::metal::validateShaderInterface(reflected, {.bindings = ambiguous, .pushConstants = full}));
    CHECK_FALSE(rachis::metal::validateShaderInterface(reflected, {.bindings = wrong, .pushConstants = full}));
    for (const auto& range : std::array{rachis::ShaderPushConstantRequirement{4, 12},
                                        rachis::ShaderPushConstantRequirement{0xFFFFFFF0U, 32},
                                        rachis::ShaderPushConstantRequirement{0, 0},
                                        rachis::ShaderPushConstantRequirement{0, 17}})
    {
        const std::array bad{range};
        CHECK_FALSE(rachis::metal::validateShaderInterface(reflected, {.bindings = correct, .pushConstants = bad}));
    }
    const std::array overlapping{rachis::ShaderPushConstantRequirement{0, 12},
                                 rachis::ShaderPushConstantRequirement{8, 8}};
    CHECK_FALSE(rachis::metal::validateShaderInterface(reflected, {.bindings = correct, .pushConstants = overlapping}));
}

TEST_CASE("Metal shader interfaces keep explicit set 0 resources in set 0",
          "[size:small][rhi][metal][reflection][regression]")
{
    rachis::metal::MetalReflectedLayout const reflected{
        .stage = rachis::ShaderStage::compute,
        .resources = {{.name = "constants",
                       .type = rachis::ShaderResourceType::uniformBuffer,
                       .metalIndex = 0,
                       .set = 0U,
                       .binding = 0U}},
    };
    const std::array setTwo{rachis::ShaderBindingRequirement{2, 0, rachis::ShaderResourceType::uniformBuffer, 1}};
    CHECK_FALSE(rachis::metal::resolveShaderInterface(reflected, {.bindings = setTwo}).has_value());

    const std::array setZero{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}};
    const auto resolved = rachis::metal::resolveShaderInterface(reflected, {.bindings = setZero});
    REQUIRE(resolved.has_value());
    REQUIRE(resolved->resources.size() == 1U);
    CHECK(resolved->resources[0].set == 0U);
    CHECK(resolved->resources[0].binding == 0U);

    constexpr std::string_view annotatedJson = R"({
        "parameters": [
            {"name": "constants", "binding": {"kind": "constantBuffer", "index": 0}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{
            "name": "computeMain",
            "stage": "compute",
            "threadGroupSize": [1, 1, 1],
            "bindings": [{"name": "constants", "binding": {"kind": "constantBuffer", "index": 0, "used": 1}}]
        }],
        "rachisBindings": [{"name": "constants", "set": 0, "binding": 0}]
    })";
    const auto parsed =
        rachis::metal::parseSlangReflectionJson(annotatedJson, "computeMain", rachis::ShaderStage::compute);
    REQUIRE(parsed.has_value());
    CHECK_FALSE(rachis::metal::validateShaderInterface(*parsed, {.bindings = setTwo}));
    CHECK(rachis::metal::validateShaderInterface(*parsed, {.bindings = setZero}));
}

TEST_CASE("Metal shader interfaces resolve multi-set binding 0 and shifted push constants",
          "[size:small][rhi][metal][reflection][regression]")
{
    constexpr std::string_view multiSetJson = R"({
        "parameters": [
            {"name": "pushData", "binding": {"kind": "constantBuffer", "space": 4, "index": 4}, "type": {"kind": "constantBuffer", "elementType": {"kind": "struct", "sizes": [{"kind": "uniform", "value": 16}]}}},
            {"name": "set0Buf", "binding": {"kind": "constantBuffer", "index": 0}, "type": {"kind": "constantBuffer"}},
            {"name": "set1Buf", "binding": {"kind": "constantBuffer", "space": 1, "index": 1}, "type": {"kind": "constantBuffer"}},
            {"name": "set2Buf", "binding": {"kind": "constantBuffer", "space": 2, "index": 2}, "type": {"kind": "constantBuffer"}},
            {"name": "set3Buf", "binding": {"kind": "constantBuffer", "space": 3, "index": 3}, "type": {"kind": "constantBuffer"}}
        ],
        "entryPoints": [{
            "name": "computeMain",
            "stage": "compute",
            "threadGroupSize": [1, 1, 1],
            "bindings": [
                {"name": "pushData", "binding": {"kind": "constantBuffer", "space": 4, "index": 4, "used": 1}},
                {"name": "set0Buf", "binding": {"kind": "constantBuffer", "index": 0, "used": 1}},
                {"name": "set1Buf", "binding": {"kind": "constantBuffer", "space": 1, "index": 1, "used": 1}},
                {"name": "set2Buf", "binding": {"kind": "constantBuffer", "space": 2, "index": 2, "used": 1}},
                {"name": "set3Buf", "binding": {"kind": "constantBuffer", "space": 3, "index": 3, "used": 1}}
            ]
        }],
        "rachisBindings": [
            {"name": "pushData", "pushConstant": true},
            {"name": "set0Buf", "set": 0, "binding": 0},
            {"name": "set1Buf", "set": 1, "binding": 0},
            {"name": "set2Buf", "set": 2, "binding": 0},
            {"name": "set3Buf", "set": 3, "binding": 0}
        ]
    })";

    const auto reflected =
        rachis::metal::parseSlangReflectionJson(multiSetJson, "computeMain", rachis::ShaderStage::compute);
    REQUIRE(reflected.has_value());
    CHECK(reflected->pushConstantBytes == 16U);
    CHECK(reflected->pushConstantMetalIndex == 4U);
    REQUIRE(reflected->resources.size() == 4U);

    const std::array bindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        rachis::ShaderBindingRequirement{1, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        rachis::ShaderBindingRequirement{2, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        rachis::ShaderBindingRequirement{3, 0, rachis::ShaderResourceType::uniformBuffer, 1},
    };
    const std::array push{rachis::ShaderPushConstantRequirement{0, 16}};
    const auto resolved = rachis::metal::resolveShaderInterface(
        *reflected,
        {.bindings = bindings, .pushConstants = push, .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}});
    REQUIRE(resolved.has_value());
    CHECK(resolved->usesPushConstants);
    CHECK(resolved->pushConstantMetalIndex == 4U);
    REQUIRE(resolved->resources.size() == 4U);
    for (std::uint32_t setIndex = 0; setIndex < 4U; ++setIndex)
    {
        CHECK(resolved->resources[setIndex].set == setIndex);
        CHECK(resolved->resources[setIndex].binding == 0U);
        CHECK(resolved->resources[setIndex].metalIndex == setIndex);
    }

    auto wrongBinding = bindings;
    wrongBinding[2].binding = 1U;
    CHECK_FALSE(
        rachis::metal::resolveShaderInterface(
            *reflected,
            {.bindings = wrongBinding, .pushConstants = push, .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}})
            .has_value());
}

TEST_CASE("Metal reflection accepts only a one-uint32 space-5 constant buffer as the draw index",
          "[size:small][rhi][metal][reflection][draw-index]")
{
    const auto reflection = [](std::string_view space, std::string_view payloadBytes) {
        return std::string(R"({
        "parameters": [
            {"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": )") +
               std::string(space) + R"(, "index": 0},
             "type": {"kind": "constantBuffer", "elementType": {"kind": "struct", "sizes": [{"kind": "uniform", "value": )" +
               std::string(payloadBytes) + R"(}]}}}
        ],
        "entryPoints": [{
            "name": "vertexMain",
            "stage": "vertex",
            "bindings": [{"name": "RachisDrawIndex", "binding": {"kind": "constantBuffer", "space": )" +
               std::string(space) + R"(, "index": 0, "used": 1}}]
        }],
        "rachisBindings": [{"name": "RachisDrawIndex", "drawIndex": true}]
    })";
    };

    const auto accepted =
        rachis::metal::parseSlangReflectionJson(reflection("5", "4"), "vertexMain", rachis::ShaderStage::vertex);
    REQUIRE(accepted.has_value());
    CHECK(accepted->drawIndex);
    CHECK(accepted->drawIndexMetalIndex == 0);

    const auto wideConstant =
        rachis::metal::parseSlangReflectionJson(reflection("5", "16"), "vertexMain", rachis::ShaderStage::vertex);
    REQUIRE_FALSE(wideConstant.has_value());
    CHECK(wideConstant.error().code == rachis::ErrorCode::invalidArgument);

    const auto otherSpace =
        rachis::metal::parseSlangReflectionJson(reflection("1", "4"), "vertexMain", rachis::ShaderStage::vertex);
    REQUIRE_FALSE(otherSpace.has_value());
    CHECK(otherSpace.error().code == rachis::ErrorCode::invalidArgument);
}

TEST_CASE("Metal reflection applies argument-buffer arrays only to the entry points that use them",
          "[size:small][rhi][metal][reflection][binding]")
{
    constexpr std::string_view json = R"({
        "parameters": [
            {"name": "textures", "binding": {"kind": "shaderResource", "space": 1, "index": 0},
             "type": {"kind": "array", "elementCount": 0, "elementType": {"kind": "resource", "baseShape": "texture2D"}}}
        ],
        "entryPoints": [
            {
                "name": "withArray",
                "stage": "compute",
                "threadGroupSize": [1, 1, 1],
                "bindings": [{"name": "textures", "binding": {"kind": "shaderResource", "space": 1, "index": 0, "used": 1}}]
            },
            {
                "name": "unusedArray",
                "stage": "compute",
                "threadGroupSize": [1, 1, 1],
                "bindings": [{"name": "textures", "binding": {"kind": "shaderResource", "space": 1, "index": 0, "used": 0}}]
            },
            {
                "name": "withoutArray",
                "stage": "compute",
                "threadGroupSize": [1, 1, 1],
                "bindings": []
            }
        ],
        "rachisBindings": [],
        "rachisArgumentBuffers": [
            {"name": "textures", "type": "sampledTexture", "set": 1, "binding": 0, "index": 0, "capacity": 64}
        ]
    })";

    const auto withArray = rachis::metal::parseSlangReflectionJson(json, "withArray", rachis::ShaderStage::compute);
    REQUIRE(withArray.has_value());
    REQUIRE(withArray->resources.size() == 1);
    CHECK(withArray->resources.front().argumentBuffer);
    CHECK(withArray->resources.front().argumentCapacity == 64);

    for (const std::string_view entry : {std::string_view{"unusedArray"}, std::string_view{"withoutArray"}})
    {
        CAPTURE(entry);
        const auto parsed = rachis::metal::parseSlangReflectionJson(json, entry, rachis::ShaderStage::compute);
        REQUIRE(parsed.has_value());
        CHECK(parsed->resources.empty());
        CHECK(rachis::metal::validateShaderInterface(*parsed, {.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}}));
    }
}
