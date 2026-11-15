#include "Internal/StageLinkage.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>

namespace
{

    using rachis::ShaderStage;
    using rachis::detail::StageComponentType;
    using rachis::detail::StageInterfaceVariable;
    using rachis::detail::validateStageLinkage;

    [[nodiscard]] bool contains(const std::string& text, const std::string& part)
    {
        return text.find(part) != std::string::npos;
    }

} // namespace

TEST_CASE("stage linkage accepts inputs the producer writes with the same type", "[size:small][rhi][stage-linkage]")
{
    const std::array outputs{
        StageInterfaceVariable{.location = 0, .componentCount = 4},
        StageInterfaceVariable{.location = 1, .componentCount = 2, .componentType = StageComponentType::uint32},
    };
    const std::array inputs{
        StageInterfaceVariable{.location = 1, .componentCount = 2, .componentType = StageComponentType::uint32},
        StageInterfaceVariable{.location = 0, .componentCount = 4},
    };
    CHECK(validateStageLinkage(ShaderStage::vertex, outputs, inputs).has_value());
    CHECK(validateStageLinkage(ShaderStage::vertex, outputs, {}).has_value());
    CHECK(validateStageLinkage(ShaderStage::vertex, {}, {}).has_value());
}

TEST_CASE("stage linkage lets the pixel shader read fewer components than written", "[size:small][rhi][stage-linkage]")
{
    const std::array outputs{StageInterfaceVariable{.location = 2, .componentCount = 4}};
    const std::array inputs{StageInterfaceVariable{.location = 2, .componentCount = 3}};
    CHECK(validateStageLinkage(ShaderStage::vertex, outputs, inputs).has_value());
}

TEST_CASE("stage linkage rejects a location the producer does not write", "[size:small][rhi][stage-linkage]")
{
    const std::array outputs{StageInterfaceVariable{.location = 0, .componentCount = 4}};
    const std::array inputs{StageInterfaceVariable{.location = 3, .componentCount = 2}};
    const auto result = validateStageLinkage(ShaderStage::geometry, outputs, inputs);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == rachis::ErrorCode::validationError);
    CHECK(contains(result.error().message, "location 3 component 0"));
    CHECK(contains(result.error().message, "geometry shader does not write"));
}

TEST_CASE("stage linkage rejects reading more components than written", "[size:small][rhi][stage-linkage]")
{
    const std::array outputs{StageInterfaceVariable{.location = 1, .componentCount = 2}};
    const std::array inputs{StageInterfaceVariable{.location = 1, .componentCount = 3}};
    const auto result = validateStageLinkage(ShaderStage::vertex, outputs, inputs);
    REQUIRE_FALSE(result.has_value());
    CHECK(contains(result.error().message, "location 1 component 2"));
    CHECK(contains(result.error().message, "vertex shader does not write"));
}

TEST_CASE("stage linkage rejects a component type mismatch", "[size:small][rhi][stage-linkage]")
{
    const std::array outputs{StageInterfaceVariable{.location = 0, .componentCount = 4}};
    const std::array inputs{
        StageInterfaceVariable{.location = 0, .componentCount = 1, .componentType = StageComponentType::sint32},
    };
    const auto result = validateStageLinkage(ShaderStage::mesh, outputs, inputs);
    REQUIRE_FALSE(result.has_value());
    CHECK(contains(result.error().message, "as sint32"));
    CHECK(contains(result.error().message, "mesh shader writes it as float32"));
}

TEST_CASE("stage linkage matches outputs packed into components of one location", "[size:small][rhi][stage-linkage]")
{
    const std::array outputs{
        StageInterfaceVariable{.location = 4, .firstComponent = 0, .componentCount = 2},
        StageInterfaceVariable{
            .location = 4, .firstComponent = 2, .componentCount = 1, .componentType = StageComponentType::uint32},
    };
    const std::array packedInput{
        StageInterfaceVariable{
            .location = 4, .firstComponent = 2, .componentCount = 1, .componentType = StageComponentType::uint32},
    };
    CHECK(validateStageLinkage(ShaderStage::vertex, outputs, packedInput).has_value());

    const std::array straddlingInput{StageInterfaceVariable{.location = 4, .firstComponent = 1, .componentCount = 2}};
    const auto result = validateStageLinkage(ShaderStage::vertex, outputs, straddlingInput);
    REQUIRE_FALSE(result.has_value());
    CHECK(contains(result.error().message, "location 4 component 2 as float32"));
}

TEST_CASE("stage linkage rejects component ranges outside one location", "[size:small][rhi][stage-linkage]")
{
    const std::array badOutput{StageInterfaceVariable{.location = 0, .firstComponent = 3, .componentCount = 2}};
    CHECK_FALSE(validateStageLinkage(ShaderStage::vertex, badOutput, {}).has_value());

    const std::array outputs{StageInterfaceVariable{.location = 0}};
    const std::array emptyInput{StageInterfaceVariable{.location = 0, .componentCount = 0}};
    CHECK_FALSE(validateStageLinkage(ShaderStage::vertex, outputs, emptyInput).has_value());
}
