#include "DrawIndexContract.h"
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

    [[nodiscard]] std::uint32_t readU32(std::span<const std::byte> bytes, std::size_t offset)
    {
        REQUIRE(offset + sizeof(std::uint32_t) <= bytes.size());
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    }

    void writeU32(std::span<std::byte> bytes, std::size_t offset, std::uint32_t value)
    {
        REQUIRE(offset + sizeof(std::uint32_t) <= bytes.size());
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    [[nodiscard]] std::size_t psvPartOffset(std::span<const std::byte> container)
    {
        constexpr std::uint32_t psv0 = 0x30565350U;
        const auto partCount = readU32(container, 28);
        std::size_t found = 0;
        for (std::uint32_t index = 0; index < partCount && found == 0; ++index)
        {
            const auto partOffset = readU32(container, 32 + index * sizeof(std::uint32_t));
            if (readU32(container, partOffset) == psv0)
            {
                found = partOffset + 8;
            }
        }
        REQUIRE(found != 0);
        return found;
    }

    [[nodiscard]] rachis::Result<rachis::ShaderModuleHandle> createDrawIndexVertex(rachis::Device& device,
                                                                                   std::span<const std::byte> bytecode,
                                                                                   bool drawIndex)
    {
        static constexpr std::array locations{0U};
        return device.createShaderModule(rachis::ShaderModuleDesc{
            .stage = rachis::ShaderStage::vertex,
            .bytecode = bytecode,
            .entryPoint = "drawIndexVertex",
            .shaderInterface =
                rachis::ShaderInterfaceDesc{
                    .vertexInputLocations = locations,
                    .drawIndex = drawIndex,
                },
        });
    }

} // namespace

TEST_CASE("Direct3D 12 indirect draws and mesh dispatches give each record its draw index",
          "[size:medium][rhi][graphics][indirect][draw-index][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto vertex = loadShader(RACHIS_TEST_DRAW_INDEX_VERTEX_SHADER_PATH);
    const auto plainVertex = loadShader(RACHIS_TEST_DRAW_INDEX_PLAIN_VERTEX_SHADER_PATH);
    const auto mesh = loadShader(RACHIS_TEST_DRAW_INDEX_MESH_SHADER_PATH);
    const auto pixel = loadShader(RACHIS_TEST_DRAW_INDEX_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertex.empty());
    REQUIRE_FALSE(plainVertex.empty());
    REQUIRE_FALSE(mesh.empty());
    REQUIRE_FALSE(pixel.empty());

    rachis::test::draw_index::runContract(device,
                                          {
                                              .vertex = vertex,
                                              .plainVertex = plainVertex,
                                              .mesh = mesh,
                                              .pixel = pixel,
                                          });
}

TEST_CASE("Direct3D 12 rejects draw-index modules whose DXIL resource table cannot be read",
          "[size:medium][rhi][draw-index][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto vertex = loadShader(RACHIS_TEST_DRAW_INDEX_VERTEX_SHADER_PATH);
    REQUIRE_FALSE(vertex.empty());

    auto valid = createDrawIndexVertex(device, vertex, true);
    REQUIRE(valid.has_value());
    REQUIRE(device.destroyShaderModule(*valid).has_value());

    const auto psv = psvPartOffset(vertex);
    const auto resourceCountOffset = psv + 4 + readU32(vertex, psv);
    REQUIRE(readU32(vertex, resourceCountOffset) > 0);
    auto truncated = vertex;
    writeU32(truncated, resourceCountOffset, 0x0010'0000U);
    for (const bool drawIndex : {true, false})
    {
        INFO("Declared draw index: " << drawIndex);
        const auto rejected = createDrawIndexVertex(device, truncated, drawIndex);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    }

    auto withoutPsv = vertex;
    withoutPsv[psv - 8] = std::byte{'X'};
    for (const bool drawIndex : {true, false})
    {
        INFO("Declared draw index: " << drawIndex);
        const auto unverifiable = createDrawIndexVertex(device, withoutPsv, drawIndex);
        REQUIRE_FALSE(unverifiable.has_value());
        CHECK(unverifiable.error().code == rachis::ErrorCode::invalidArgument);
    }
}

TEST_CASE("Direct3D 12 rejects ray-tracing library entries that read the draw index",
          "[size:medium][rhi][draw-index][ray-tracing-pipeline][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto library = loadShader(RACHIS_TEST_RAY_TRACING_PIPELINE_SHADER_PATH);
    REQUIRE_FALSE(library.empty());

    struct Entry final
    {
        rachis::ShaderStage stage = rachis::ShaderStage::none;
        const char* name = nullptr;
        bool readsDrawIndex = false;
    };
    constexpr std::array entries{
        Entry{rachis::ShaderStage::rayGeneration, "rayGenerationReadingDrawIndex", true},
        Entry{rachis::ShaderStage::closestHit, "closestHitReadingDrawIndex", true},
        Entry{rachis::ShaderStage::rayGeneration, "rayGenerationWithoutRecord", false},
        Entry{rachis::ShaderStage::closestHit, "closestHitShader", false},
    };
    std::vector<rachis::ShaderLibraryEntryDesc> libraryEntries;
    for (const auto& entry : entries)
    {
        libraryEntries.push_back({.stage = entry.stage, .bytecode = library, .entryPoint = entry.name});
    }
    auto shaderLibrary = device.createShaderLibrary({.entries = libraryEntries});
    REQUIRE(shaderLibrary.has_value());

    for (const auto& entry : entries)
    {
        INFO("Entry point: " << entry.name);
        auto direct = device.createShaderModule({
            .stage = entry.stage,
            .bytecode = library,
            .entryPoint = entry.name,
        });
        auto fromLibrary = device.createShaderModule({
            .stage = entry.stage,
            .entryPoint = entry.name,
            .library = *shaderLibrary,
        });
        for (auto* module : {&direct, &fromLibrary})
        {
            if (entry.readsDrawIndex)
            {
                REQUIRE_FALSE(module->has_value());
                CHECK(module->error().code == rachis::ErrorCode::invalidArgument);
            }
            else
            {
                REQUIRE(module->has_value());
                REQUIRE(device.destroyShaderModule(**module).has_value());
            }
        }
    }
    REQUIRE(device.destroyShaderLibrary(*shaderLibrary).has_value());
}
