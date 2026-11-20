#include "D3D12/D3D12DxilSignature.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

    using rachis::d3d12::dxil::readExpectedWaveLaneCounts;
    using rachis::d3d12::dxil::readStageSignature;
    using rachis::d3d12::dxil::readThreadGroupSize;
    using rachis::d3d12::dxil::SignaturePart;
    using rachis::d3d12::dxil::ThreadGroupSize;
    using rachis::d3d12::dxil::validateComputeWaveSize;
    using rachis::d3d12::dxil::validateDrawIndexDeclaration;
    using rachis::d3d12::dxil::WaveLaneCountRange;
    using rachis::detail::StageComponentType;
    using rachis::detail::StageInterfaceVariable;

    struct Element final
    {
        std::uint8_t rows = 1;
        std::uint8_t columns = 4;
        std::uint8_t startRow = 0;
        std::uint8_t startColumn = 0;
        std::uint8_t kind = 0;
        std::uint8_t componentType = 3;
        std::uint8_t stream = 0;
    };

    void appendU32(std::vector<std::byte>& bytes, std::uint32_t value)
    {
        for (std::uint32_t shift = 0; shift < 32; shift += 8)
        {
            bytes.push_back(static_cast<std::byte>((value >> shift) & 0xFFU));
        }
    }

    void appendU8(std::vector<std::byte>& bytes, std::uint8_t value)
    {
        bytes.push_back(static_cast<std::byte>(value));
    }

    void writeU32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value)
    {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    [[nodiscard]] std::vector<std::byte> container(const std::vector<Element>& inputs,
                                                   const std::vector<Element>& outputs)
    {
        std::vector<std::byte> part;
        appendU32(part, 48);
        const auto info = part.size();
        part.resize(info + 48);
        part[info + 24] = std::byte{1};
        part[info + 28] = static_cast<std::byte>(inputs.size());
        part[info + 29] = static_cast<std::byte>(outputs.size());
        appendU32(part, 1);
        appendU32(part, 16);
        part.resize(part.size() + 16);
        appendU32(part, 4);
        appendU32(part, 0);
        appendU32(part, 1);
        appendU32(part, 0);
        appendU32(part, 16);
        for (const auto* list : {&inputs, &outputs})
        {
            for (const auto& element : *list)
            {
                appendU32(part, 0);
                appendU32(part, 0);
                appendU8(part, element.rows);
                appendU8(part, element.startRow);
                appendU8(part, static_cast<std::uint8_t>(element.columns | (element.startColumn << 4U)));
                appendU8(part, element.kind);
                appendU8(part, element.componentType);
                appendU8(part, 0);
                appendU8(part, static_cast<std::uint8_t>(element.stream << 4U));
                appendU8(part, 0);
            }
        }

        std::vector<std::byte> bytes;
        for (const char ch : std::string_view("DXBC"))
        {
            bytes.push_back(static_cast<std::byte>(ch));
        }
        bytes.resize(24);
        appendU32(bytes, 0);
        appendU32(bytes, 1);
        appendU32(bytes, 36);
        for (const char ch : std::string_view("PSV0"))
        {
            bytes.push_back(static_cast<std::byte>(ch));
        }
        appendU32(bytes, static_cast<std::uint32_t>(part.size()));
        bytes.insert(bytes.end(), part.begin(), part.end());
        writeU32(bytes, 24, static_cast<std::uint32_t>(bytes.size()));
        return bytes;
    }

} // namespace

TEST_CASE("The DXIL signature reader numbers user-defined elements in declaration order",
          "[size:small][rhi][d3d12][stage-linkage]")
{
    const auto bytes = container({{.kind = 1, .componentType = 1}},
                                 {
                                     {.kind = 3},
                                     {.columns = 2, .startRow = 1},
                                     {.columns = 3, .startRow = 2},
                                     {.columns = 1, .startRow = 1, .startColumn = 2},
                                     {.columns = 1, .startRow = 3, .componentType = 1},
                                     {.rows = 2, .columns = 4, .startRow = 4},
                                     {.columns = 2, .componentType = 9},
                                     {.columns = 1, .componentType = 0},
                                     {.columns = 4, .componentType = 2, .stream = 1},
                                 });
    const auto outputs = readStageSignature(bytes, SignaturePart::output);
    REQUIRE(outputs.has_value());
    REQUIRE(outputs->has_value());
    const auto& variables = **outputs;
    REQUIRE(variables.size() == 8);
    CHECK(variables[0].variable == StageInterfaceVariable{.location = 0, .componentCount = 2});
    CHECK(variables[1].variable == StageInterfaceVariable{.location = 1, .componentCount = 3});
    CHECK(variables[2].variable == StageInterfaceVariable{.location = 2, .componentCount = 1});
    CHECK(variables[3].variable ==
          StageInterfaceVariable{.location = 3, .componentCount = 1, .componentType = StageComponentType::uint32});
    CHECK(variables[4].variable == StageInterfaceVariable{.location = 4, .componentCount = 4});
    CHECK(variables[5].variable == StageInterfaceVariable{.location = 5, .componentCount = 4});
    CHECK(variables[6].variable ==
          StageInterfaceVariable{.location = 6, .componentCount = 2, .componentType = StageComponentType::float64});
    CHECK(variables[7].stream == 1);
    CHECK(variables[7].variable.location == 7);
    CHECK(variables[7].variable.componentType == StageComponentType::sint32);

    const auto inputs = readStageSignature(bytes, SignaturePart::input);
    REQUIRE(inputs.has_value());
    REQUIRE(inputs->has_value());
    CHECK((*inputs)->empty());
}

TEST_CASE("The DXIL signature reader rejects malformed containers", "[size:small][rhi][d3d12][stage-linkage]")
{
    const auto bytes = container({{.columns = 2}}, {});
    REQUIRE(readStageSignature(bytes, SignaturePart::input).has_value());

    CHECK_FALSE(readStageSignature(std::vector<std::byte>(8), SignaturePart::input).has_value());

    auto wrongMagic = bytes;
    wrongMagic[0] = std::byte{'X'};
    CHECK_FALSE(readStageSignature(wrongMagic, SignaturePart::input).has_value());

    auto tooManyParts = bytes;
    writeU32(tooManyParts, 28, 1000);
    CHECK_FALSE(readStageSignature(tooManyParts, SignaturePart::input).has_value());

    auto partPastEnd = bytes;
    writeU32(partPastEnd, 40, 4096);
    CHECK_FALSE(readStageSignature(partPastEnd, SignaturePart::input).has_value());

    auto resourcesPastPart = bytes;
    writeU32(resourcesPastPart, 44 + 4 + 48, 1000);
    CHECK_FALSE(readStageSignature(resourcesPastPart, SignaturePart::input).has_value());

    auto withoutPsv = bytes;
    withoutPsv[36] = std::byte{'X'};
    const auto none = readStageSignature(withoutPsv, SignaturePart::input);
    REQUIRE(none.has_value());
    CHECK_FALSE(none->has_value());
}

TEST_CASE("The DXIL reader reads a compute shader's declared wave size from the PSV0 runtime info",
          "[size:small][rhi][d3d12][subgroup]")
{
    constexpr std::size_t minimumOffset = 48 + 16;
    constexpr std::size_t maximumOffset = 48 + 20;
    auto bytes = container({}, {});

    writeU32(bytes, minimumOffset, 0);
    writeU32(bytes, maximumOffset, 0xFFFFFFFFU);
    const auto undeclared = readExpectedWaveLaneCounts(bytes);
    REQUIRE(undeclared.has_value());
    CHECK_FALSE(undeclared->has_value());

    writeU32(bytes, minimumOffset, 32);
    writeU32(bytes, maximumOffset, 32);
    const auto declared = readExpectedWaveLaneCounts(bytes);
    REQUIRE(declared.has_value());
    REQUIRE(declared->has_value());
    CHECK((*declared)->minimum == 32);
    CHECK((*declared)->maximum == 32);

    auto withoutPsv = bytes;
    withoutPsv[36] = std::byte{'X'};
    const auto noPart = readExpectedWaveLaneCounts(withoutPsv);
    REQUIRE(noPart.has_value());
    CHECK_FALSE(noPart->has_value());

    auto shortInfo = bytes;
    writeU32(shortInfo, 44, 16);
    const auto older = readExpectedWaveLaneCounts(shortInfo);
    REQUIRE(older.has_value());
    CHECK_FALSE(older->has_value());

    CHECK_FALSE(readExpectedWaveLaneCounts(std::vector<std::byte>(8)).has_value());
    auto truncatedInfo = bytes;
    writeU32(truncatedInfo, 44, 4096);
    CHECK_FALSE(readExpectedWaveLaneCounts(truncatedInfo).has_value());
}

TEST_CASE("The DXIL signature reader leaves out pixel inputs nothing reads", "[size:small][rhi][d3d12][stage-linkage]")
{
    std::vector<std::byte> psv;
    appendU32(psv, 48);
    const auto info = psv.size();
    psv.resize(info + 48);
    psv[info + 28] = std::byte{3};
    appendU32(psv, 0);
    appendU32(psv, 8);
    for (const char ch : std::string_view("A\0B\0C\0\0", 8))
    {
        psv.push_back(static_cast<std::byte>(ch));
    }
    appendU32(psv, 1);
    appendU32(psv, 0);
    appendU32(psv, 16);
    for (const std::uint32_t name : {0U, 2U, 4U})
    {
        appendU32(psv, name);
        appendU32(psv, 0);
        for (const std::uint8_t value : {std::uint8_t{1},
                                         std::uint8_t{0},
                                         std::uint8_t{2},
                                         std::uint8_t{0},
                                         std::uint8_t{3},
                                         std::uint8_t{0},
                                         std::uint8_t{0},
                                         std::uint8_t{0}})
        {
            appendU8(psv, value);
        }
    }

    std::vector<std::byte> isg;
    appendU32(isg, 3);
    appendU32(isg, 8);
    const std::uint32_t namesAt = 8 + 3 * 32;
    for (std::uint32_t element = 0; element < 3; ++element)
    {
        appendU32(isg, 0);
        appendU32(isg, namesAt + element * 2);
        appendU32(isg, 0);
        appendU32(isg, 0);
        appendU32(isg, 3);
        appendU32(isg, element + 1);
        appendU8(isg, 0x3);
        appendU8(isg, element == 1 ? 0 : 0x3);
        appendU8(isg, 0);
        appendU8(isg, 0);
        appendU32(isg, 0);
    }
    for (const char ch : std::string_view("A\0B\0C\0\0", 8))
    {
        isg.push_back(static_cast<std::byte>(ch));
    }

    std::vector<std::byte> bytes;
    for (const char ch : std::string_view("DXBC"))
    {
        bytes.push_back(static_cast<std::byte>(ch));
    }
    bytes.resize(24);
    appendU32(bytes, 0);
    appendU32(bytes, 2);
    appendU32(bytes, 40);
    appendU32(bytes, static_cast<std::uint32_t>(40 + 8 + psv.size()));
    for (const auto& [fourCC, part] :
         {std::pair{std::string_view("PSV0"), &psv}, std::pair{std::string_view("ISG1"), &isg}})
    {
        for (const char ch : fourCC)
        {
            bytes.push_back(static_cast<std::byte>(ch));
        }
        appendU32(bytes, static_cast<std::uint32_t>(part->size()));
        bytes.insert(bytes.end(), part->begin(), part->end());
    }
    writeU32(bytes, 24, static_cast<std::uint32_t>(bytes.size()));

    const auto inputs = readStageSignature(bytes, SignaturePart::input);
    REQUIRE(inputs.has_value());
    REQUIRE(inputs->has_value());
    REQUIRE((*inputs)->size() == 2);
    CHECK((**inputs)[0].variable == StageInterfaceVariable{.location = 0, .componentCount = 2});
    CHECK((**inputs)[1].variable == StageInterfaceVariable{.location = 2, .componentCount = 2});
}

namespace
{

    constexpr std::size_t runtimeInfoOffset = 48;
    constexpr std::size_t resourceCountOffset = runtimeInfoOffset + 48;
    constexpr std::size_t resourceRecordOffset = resourceCountOffset + 8;

} // namespace

TEST_CASE("Direct3D 12 draw-index declarations need a readable PSV0 resource table",
          "[size:small][rhi][d3d12][draw-index]")
{
    constexpr std::uint32_t constantBufferType = 2;
    auto binding = container({}, {});
    writeU32(binding, resourceRecordOffset, constantBufferType);
    writeU32(binding, resourceRecordOffset + 4, 5);
    CHECK(validateDrawIndexDeclaration(binding, 0, 5, true).has_value());
    const auto undeclared = validateDrawIndexDeclaration(binding, 0, 5, false);
    REQUIRE_FALSE(undeclared.has_value());
    CHECK(undeclared.error().code == rachis::ErrorCode::invalidArgument);

    const auto other = container({}, {});
    CHECK(validateDrawIndexDeclaration(other, 0, 5, false).has_value());
    const auto unbound = validateDrawIndexDeclaration(other, 0, 5, true);
    REQUIRE_FALSE(unbound.has_value());
    CHECK(unbound.error().code == rachis::ErrorCode::invalidArgument);

    auto truncated = binding;
    writeU32(truncated, resourceCountOffset, 1000);
    for (const bool declared : {true, false})
    {
        INFO("Declared: " << declared);
        const auto result = validateDrawIndexDeclaration(truncated, 0, 5, declared);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().code == rachis::ErrorCode::invalidArgument);
    }

    auto withoutPsv = binding;
    withoutPsv[36] = std::byte{'X'};
    CHECK(validateDrawIndexDeclaration(withoutPsv, 0, 5, false).has_value());
    const auto unverifiable = validateDrawIndexDeclaration(withoutPsv, 0, 5, true);
    REQUIRE_FALSE(unverifiable.has_value());
    CHECK(unverifiable.error().code == rachis::ErrorCode::invalidArgument);

    auto programWithoutPsv = binding;
    for (std::size_t index = 0; index < 4; ++index)
    {
        programWithoutPsv[36 + index] = static_cast<std::byte>(std::string_view("DXIL")[index]);
    }
    for (const bool declared : {true, false})
    {
        INFO("Declared: " << declared);
        const auto result = validateDrawIndexDeclaration(programWithoutPsv, 0, 5, declared, "rayGeneration");
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().code == rachis::ErrorCode::invalidArgument);
    }

    CHECK(validateDrawIndexDeclaration(std::vector<std::byte>(8), 0, 5, false).has_value());
    CHECK_FALSE(validateDrawIndexDeclaration(std::vector<std::byte>(8), 0, 5, true).has_value());
    auto headerOnly = std::vector<std::byte>(binding.begin(), binding.begin() + 8);
    CHECK_FALSE(validateDrawIndexDeclaration(headerOnly, 0, 5, false).has_value());
}

namespace
{

    struct LibraryResource final
    {
        std::uint32_t resourceClass = 2;
        std::uint32_t space = 5;
        std::uint32_t lowerBound = 0;
        std::uint32_t upperBound = 0;
    };

    struct LibraryFunction final
    {
        std::string_view name;
        std::uint32_t kind = 7;
        std::vector<std::uint32_t> resources;
        std::optional<std::uint32_t> list;
    };

    constexpr std::size_t libraryRuntimeDataOffset = 48;

    [[nodiscard]] std::vector<std::byte> libraryContainer(const std::vector<LibraryResource>& resources,
                                                          const std::vector<LibraryFunction>& functions)
    {
        std::vector<std::byte> strings{std::byte{0}};
        std::vector<std::uint32_t> names;
        for (const auto& function : functions)
        {
            names.push_back(static_cast<std::uint32_t>(strings.size()));
            for (const char ch : function.name)
            {
                strings.push_back(static_cast<std::byte>(ch));
            }
            strings.push_back(std::byte{0});
        }
        strings.resize((strings.size() + 3) & ~std::size_t{3});

        std::vector<std::byte> indices;
        std::vector<std::uint32_t> lists;
        for (const auto& function : functions)
        {
            if (function.resources.empty())
            {
                lists.push_back(0xFFFFFFFFU);
                continue;
            }
            lists.push_back(function.list.value_or(static_cast<std::uint32_t>(indices.size() / sizeof(std::uint32_t))));
            appendU32(indices, static_cast<std::uint32_t>(function.resources.size()));
            for (const auto resource : function.resources)
            {
                appendU32(indices, resource);
            }
        }

        std::vector<std::byte> resourceTable;
        appendU32(resourceTable, static_cast<std::uint32_t>(resources.size()));
        appendU32(resourceTable, 32);
        for (const auto& resource : resources)
        {
            for (const std::uint32_t value :
                 {resource.resourceClass, 13U, 0U, resource.space, resource.lowerBound, resource.upperBound, 0U, 0U})
            {
                appendU32(resourceTable, value);
            }
        }

        std::vector<std::byte> functionTable;
        appendU32(functionTable, static_cast<std::uint32_t>(functions.size()));
        appendU32(functionTable, 44);
        for (std::size_t index = 0; index < functions.size(); ++index)
        {
            for (const std::uint32_t value :
                 {names[index], names[index], lists[index], 0xFFFFFFFFU, functions[index].kind})
            {
                appendU32(functionTable, value);
            }
            for (std::size_t field = 0; field < 6; ++field)
            {
                appendU32(functionTable, 0);
            }
        }

        const std::array<std::pair<std::uint32_t, const std::vector<std::byte>*>, 4> parts{{
            {1U, &strings},
            {3U, &resourceTable},
            {4U, &functionTable},
            {2U, &indices},
        }};
        std::vector<std::byte> runtimeData;
        appendU32(runtimeData, 0x10);
        appendU32(runtimeData, static_cast<std::uint32_t>(parts.size()));
        std::uint32_t partOffset = 8 + static_cast<std::uint32_t>(parts.size() * sizeof(std::uint32_t));
        for (const auto& [type, part] : parts)
        {
            appendU32(runtimeData, partOffset);
            partOffset += 8 + static_cast<std::uint32_t>(part->size());
        }
        for (const auto& [type, part] : parts)
        {
            appendU32(runtimeData, type);
            appendU32(runtimeData, static_cast<std::uint32_t>(part->size()));
            runtimeData.insert(runtimeData.end(), part->begin(), part->end());
        }

        std::vector<std::byte> bytes;
        for (const char ch : std::string_view("DXBC"))
        {
            bytes.push_back(static_cast<std::byte>(ch));
        }
        bytes.resize(24);
        appendU32(bytes, 0);
        appendU32(bytes, 2);
        appendU32(bytes, 40);
        appendU32(bytes, static_cast<std::uint32_t>(40 + 8 + runtimeData.size()));
        for (const auto& [fourCC, part] : {std::pair{std::string_view("RDAT"), runtimeData},
                                           std::pair{std::string_view("DXIL"), std::vector<std::byte>(16)}})
        {
            for (const char ch : fourCC)
            {
                bytes.push_back(static_cast<std::byte>(ch));
            }
            appendU32(bytes, static_cast<std::uint32_t>(part.size()));
            bytes.insert(bytes.end(), part.begin(), part.end());
        }
        writeU32(bytes, 24, static_cast<std::uint32_t>(bytes.size()));
        return bytes;
    }

    [[nodiscard]] rachis::ErrorCode libraryDrawIndexError(const std::vector<std::byte>& library,
                                                          bool declared,
                                                          std::string_view entryPoint)
    {
        const auto result = validateDrawIndexDeclaration(library, 0, 5, declared, entryPoint);
        REQUIRE_FALSE(result.has_value());
        return result.error().code;
    }

} // namespace

TEST_CASE("Direct3D 12 reads a ray-tracing library entry's draw-index binding from the RDAT resource lists",
          "[size:small][rhi][d3d12][draw-index]")
{
    const std::vector<LibraryResource> resources{{}, {.resourceClass = 1, .space = 0}};
    const auto library = libraryContainer(resources,
                                          {
                                              {.name = "reads", .resources = {1, 0}},
                                              {.name = "plain", .resources = {1}},
                                              {.name = "empty", .kind = 10},
                                          });
    CHECK(validateDrawIndexDeclaration(library, 0, 5, false, "plain").has_value());
    CHECK(validateDrawIndexDeclaration(library, 0, 5, false, "empty").has_value());
    CHECK(libraryDrawIndexError(library, false, "reads") == rachis::ErrorCode::invalidArgument);
    CHECK(validateDrawIndexDeclaration(library, 0, 5, true, "reads").has_value());
    CHECK(libraryDrawIndexError(library, true, "plain") == rachis::ErrorCode::invalidArgument);
    CHECK(libraryDrawIndexError(library, false, "missing") == rachis::ErrorCode::invalidArgument);

    const auto helper = libraryContainer(resources,
                                         {
                                             {.name = "plain", .resources = {1}},
                                             {.name = "helper", .kind = 6, .resources = {0}},
                                         });
    CHECK(libraryDrawIndexError(helper, false, "plain") == rachis::ErrorCode::invalidArgument);

    const auto unbounded =
        libraryContainer({{.lowerBound = 0, .upperBound = 0xFFFFFFFFU}}, {{.name = "reads", .resources = {0}}});
    CHECK(libraryDrawIndexError(unbounded, false, "reads") == rachis::ErrorCode::invalidArgument);
    const auto elsewhere = libraryContainer({{.space = 4}, {.resourceClass = 0}, {.lowerBound = 1, .upperBound = 1}},
                                            {{.name = "reads", .resources = {0, 1, 2}}});
    CHECK(validateDrawIndexDeclaration(elsewhere, 0, 5, false, "reads").has_value());
    CHECK(validateDrawIndexDeclaration(elsewhere, 0, 5, false, "missing").has_value());
}

TEST_CASE("Direct3D 12 rejects ray-tracing libraries whose RDAT part cannot be read",
          "[size:small][rhi][d3d12][draw-index]")
{
    const auto library = libraryContainer({{}}, {{.name = "plain"}});
    REQUIRE(validateDrawIndexDeclaration(library, 0, 5, false, "plain").has_value());

    auto unknownVersion = library;
    writeU32(unknownVersion, libraryRuntimeDataOffset, 0x11);
    CHECK(libraryDrawIndexError(unknownVersion, false, "plain") == rachis::ErrorCode::invalidArgument);

    auto truncatedParts = library;
    writeU32(truncatedParts, libraryRuntimeDataOffset + 4, 1000);
    CHECK(libraryDrawIndexError(truncatedParts, false, "plain") == rachis::ErrorCode::invalidArgument);

    auto partOutside = library;
    writeU32(partOutside, libraryRuntimeDataOffset + 8, 0x10000);
    CHECK(libraryDrawIndexError(partOutside, false, "plain") == rachis::ErrorCode::invalidArgument);

    const auto missingResource = libraryContainer({{}}, {{.name = "plain", .resources = {7}}});
    CHECK(libraryDrawIndexError(missingResource, false, "plain") == rachis::ErrorCode::invalidArgument);

    const auto listOutside = libraryContainer({{}}, {{.name = "plain", .resources = {0}, .list = 50}});
    CHECK(libraryDrawIndexError(listOutside, false, "plain") == rachis::ErrorCode::invalidArgument);

    auto withoutRuntimeData = library;
    withoutRuntimeData[40] = std::byte{'X'};
    CHECK(libraryDrawIndexError(withoutRuntimeData, false, "plain") == rachis::ErrorCode::invalidArgument);
}

namespace
{

    constexpr std::size_t runtimeInfoSizeOffset = 44;

    [[nodiscard]] std::vector<std::byte> computeContainer(std::uint32_t minimumWave,
                                                          std::uint32_t maximumWave,
                                                          ThreadGroupSize threads)
    {
        auto bytes = container({}, {});
        writeU32(bytes, runtimeInfoOffset + 16, minimumWave);
        writeU32(bytes, runtimeInfoOffset + 20, maximumWave);
        writeU32(bytes, runtimeInfoOffset + 36, threads.x);
        writeU32(bytes, runtimeInfoOffset + 40, threads.y);
        writeU32(bytes, runtimeInfoOffset + 44, threads.z);
        return bytes;
    }

    [[nodiscard]] rachis::ErrorCode waveSizeError(const std::vector<std::byte>& bytes,
                                                  std::uint32_t required,
                                                  std::optional<ThreadGroupSize> interfaceWorkgroup,
                                                  WaveLaneCountRange device)
    {
        const auto result = validateComputeWaveSize(bytes, required, interfaceWorkgroup, device);
        REQUIRE_FALSE(result.has_value());
        return result.error().code;
    }

} // namespace

TEST_CASE("The DXIL reader reads a compute shader's thread-group size from the PSV0 runtime info",
          "[size:small][rhi][d3d12][subgroup]")
{
    auto bytes = computeContainer(0, 0, {});
    const auto unrecorded = readThreadGroupSize(bytes);
    REQUIRE(unrecorded.has_value());
    CHECK_FALSE(unrecorded->has_value());

    bytes = computeContainer(0, 0, {.x = 16, .y = 2, .z = 3});
    const auto recorded = readThreadGroupSize(bytes);
    REQUIRE(recorded.has_value());
    REQUIRE(recorded->has_value());
    CHECK(**recorded == ThreadGroupSize{.x = 16, .y = 2, .z = 3});

    auto version1 = bytes;
    writeU32(version1, runtimeInfoSizeOffset, 36);
    const auto older = readThreadGroupSize(version1);
    REQUIRE(older.has_value());
    CHECK_FALSE(older->has_value());

    auto truncated = bytes;
    writeU32(truncated, runtimeInfoSizeOffset, 4096);
    CHECK_FALSE(readThreadGroupSize(truncated).has_value());
}

TEST_CASE("Direct3D 12 accepts a [WaveSize] range that overlaps the device range when no size is required",
          "[size:small][rhi][d3d12][subgroup]")
{
    constexpr WaveLaneCountRange wave32{.minimum = 32, .maximum = 32};
    constexpr WaveLaneCountRange wave32To64{.minimum = 32, .maximum = 64};
    constexpr ThreadGroupSize threads{.x = 64, .y = 1, .z = 1};
    CHECK(validateComputeWaveSize(computeContainer(16, 64, threads), 0, std::nullopt, wave32).has_value());
    CHECK(validateComputeWaveSize(computeContainer(32, 32, threads), 0, std::nullopt, wave32).has_value());
    CHECK(validateComputeWaveSize(computeContainer(64, 128, threads), 0, std::nullopt, wave32To64).has_value());
    CHECK(validateComputeWaveSize(computeContainer(0, 0, threads), 0, std::nullopt, wave32).has_value());
    CHECK(waveSizeError(computeContainer(64, 128, threads), 0, std::nullopt, wave32) == rachis::ErrorCode::unsupported);
    CHECK(waveSizeError(computeContainer(8, 16, threads), 0, std::nullopt, wave32To64) ==
          rachis::ErrorCode::unsupported);
    CHECK(waveSizeError(computeContainer(16, 16, threads), 0, std::nullopt, wave32) == rachis::ErrorCode::unsupported);

    auto truncated = computeContainer(16, 64, threads);
    writeU32(truncated, runtimeInfoSizeOffset, 4096);
    CHECK(waveSizeError(truncated, 0, std::nullopt, wave32) == rachis::ErrorCode::invalidArgument);
    CHECK(validateComputeWaveSize(std::vector<std::byte>(8), 0, std::nullopt, wave32).has_value());
    CHECK(waveSizeError(std::vector<std::byte>(8), 32, std::nullopt, wave32) == rachis::ErrorCode::invalidArgument);
}

TEST_CASE("Direct3D 12 checks a required subgroup size against the DXIL [numthreads] size",
          "[size:small][rhi][d3d12][subgroup]")
{
    constexpr WaveLaneCountRange device{.minimum = 32, .maximum = 64};
    constexpr ThreadGroupSize wide{.x = 64, .y = 1, .z = 1};
    CHECK(validateComputeWaveSize(computeContainer(32, 32, wide), 32, wide, device).has_value());
    CHECK(validateComputeWaveSize(computeContainer(32, 32, wide), 32, std::nullopt, device).has_value());

    const auto narrow = computeContainer(32, 32, {.x = 16, .y = 1, .z = 1});
    CHECK(waveSizeError(narrow, 32, ThreadGroupSize{.x = 32, .y = 1, .z = 1}, device) ==
          rachis::ErrorCode::invalidArgument);
    CHECK(waveSizeError(narrow, 32, std::nullopt, device) == rachis::ErrorCode::invalidArgument);
    CHECK(waveSizeError(computeContainer(32, 32, {.x = 48, .y = 1, .z = 1}), 32, std::nullopt, device) ==
          rachis::ErrorCode::invalidArgument);
    CHECK(waveSizeError(computeContainer(32, 32, wide), 32, ThreadGroupSize{.x = 32, .y = 2, .z = 1}, device) ==
          rachis::ErrorCode::invalidArgument);

    auto version1 = computeContainer(32, 32, wide);
    writeU32(version1, runtimeInfoSizeOffset, 36);
    CHECK(waveSizeError(version1, 32, wide, device) == rachis::ErrorCode::invalidArgument);

    CHECK(waveSizeError(computeContainer(32, 64, wide), 32, wide, device) == rachis::ErrorCode::invalidArgument);
    CHECK(waveSizeError(computeContainer(0, 0, wide), 32, wide, device) == rachis::ErrorCode::invalidArgument);
    CHECK(waveSizeError(computeContainer(64, 64, wide), 32, wide, device) == rachis::ErrorCode::invalidArgument);
}
