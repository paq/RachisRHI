#include "D3D12/D3D12DxilSignature.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

    using rachis::d3d12::dxil::readExpectedWaveLaneCounts;
    using rachis::d3d12::dxil::readStageSignature;
    using rachis::d3d12::dxil::SignaturePart;
    using rachis::d3d12::dxil::validateDrawIndexDeclaration;
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

    CHECK_FALSE(validateDrawIndexDeclaration(std::vector<std::byte>(8), 0, 5, false).has_value());
}
