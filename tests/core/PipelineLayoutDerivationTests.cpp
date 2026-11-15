#include "Rachis/Pipeline.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

    using rachis::BindingFlag;
    using rachis::BindingLayoutEntry;
    using rachis::BindingType;
    using rachis::PushConstantRange;
    using rachis::ShaderBindingRequirement;
    using rachis::ShaderInterfaceSnapshot;
    using rachis::ShaderPushConstantRequirement;
    using rachis::ShaderResourceType;
    using rachis::ShaderStage;

    [[nodiscard]] bool sameEntry(const BindingLayoutEntry& left, const BindingLayoutEntry& right) noexcept
    {
        return left.binding == right.binding && left.type == right.type && left.visibility == right.visibility &&
               left.count == right.count && left.flags == right.flags;
    }

    [[nodiscard]] bool sameRanges(std::span<const PushConstantRange> actual,
                                  std::initializer_list<PushConstantRange> expected) noexcept
    {
        return actual.size() == expected.size() &&
               std::equal(actual.begin(), actual.end(), expected.begin(), [](const auto& left, const auto& right) {
                   return left.visibility == right.visibility && left.offset == right.offset && left.size == right.size;
               });
    }

    [[nodiscard]] ShaderInterfaceSnapshot pushOnly(ShaderStage stage,
                                                   std::initializer_list<ShaderPushConstantRequirement> ranges)
    {
        return ShaderInterfaceSnapshot{.stage = stage, .pushConstants = ranges};
    }

    void requireExactPerStageCoverage(std::span<const ShaderInterfaceSnapshot> shaders,
                                      std::span<const PushConstantRange> ranges)
    {
        constexpr std::uint32_t words = rachis::maxPushConstantBytes / 4;
        for (const auto& shader : shaders)
        {
            std::array<bool, words> declared{};
            std::array<bool, words> visible{};
            for (const auto& other : shaders)
            {
                if (other.stage != shader.stage)
                {
                    continue;
                }
                for (const auto& range : other.pushConstants)
                {
                    for (auto word = range.offset / 4; word < (range.offset + range.size) / 4; ++word)
                    {
                        declared[word] = true;
                    }
                }
            }
            for (const auto& range : ranges)
            {
                if (!rachis::hasAny(range.visibility, shader.stage))
                {
                    continue;
                }
                for (auto word = range.offset / 4; word < (range.offset + range.size) / 4; ++word)
                {
                    visible[word] = true;
                }
            }
            CHECK(declared == visible);
        }
        for (std::size_t index = 0; index < ranges.size(); ++index)
        {
            for (std::size_t previous = 0; previous < index; ++previous)
            {
                const auto& left = ranges[previous];
                const auto& right = ranges[index];
                const bool overlap = left.offset < right.offset + right.size && right.offset < left.offset + left.size;
                CHECK_FALSE((overlap && rachis::hasAny(left.visibility, right.visibility)));
            }
        }
    }

} // namespace

TEST_CASE("derivePipelineLayout unions binding visibility across stages and keeps sets apart",
          "[size:small][rhi][core][pipeline-layout]")
{
    const std::array vertexBindings{
        ShaderBindingRequirement{.set = 0, .binding = 0, .type = ShaderResourceType::uniformBuffer},
        ShaderBindingRequirement{.set = 3, .binding = 2, .type = ShaderResourceType::readOnlyStorageBuffer},
    };
    const std::array pixelBindings{
        ShaderBindingRequirement{.set = 0, .binding = 2, .type = ShaderResourceType::sampler},
        ShaderBindingRequirement{.set = 0, .binding = 0, .type = ShaderResourceType::uniformBuffer},
        ShaderBindingRequirement{.set = 0, .binding = 1, .type = ShaderResourceType::sampledTexture, .count = 4},
        ShaderBindingRequirement{
            .set = 1,
            .binding = 5,
            .type = ShaderResourceType::sampledTexture,
            .count = rachis::unboundedBindingCount,
        },
    };
    const std::array shaders{
        ShaderInterfaceSnapshot{
            .stage = ShaderStage::vertex,
            .bindings = {vertexBindings.begin(), vertexBindings.end()},
        },
        ShaderInterfaceSnapshot{
            .stage = ShaderStage::pixel,
            .bindings = {pixelBindings.begin(), pixelBindings.end()},
        },
    };

    const auto derived = rachis::derivePipelineLayout(shaders);
    REQUIRE(derived.has_value());
    REQUIRE(derived->bindingSets.size() == 4);
    CHECK(derived->pushConstantRanges.empty());

    const auto& set0 = derived->bindingSets[0];
    REQUIRE(set0.size() == 3);
    CHECK(sameEntry(set0[0],
                    BindingLayoutEntry{
                        .binding = 0,
                        .type = BindingType::uniformBuffer,
                        .visibility = ShaderStage::vertex | ShaderStage::pixel,
                    }));
    CHECK(sameEntry(set0[1],
                    BindingLayoutEntry{
                        .binding = 1,
                        .type = BindingType::sampledTexture,
                        .visibility = ShaderStage::pixel,
                        .count = 4,
                    }));
    CHECK(sameEntry(set0[2],
                    BindingLayoutEntry{.binding = 2, .type = BindingType::sampler, .visibility = ShaderStage::pixel}));

    const auto& set1 = derived->bindingSets[1];
    REQUIRE(set1.size() == 1);
    CHECK(sameEntry(set1[0],
                    BindingLayoutEntry{
                        .binding = 5,
                        .type = BindingType::sampledTexture,
                        .visibility = ShaderStage::pixel,
                        .count = rachis::unboundedBindingCount,
                        .flags = BindingFlag::variableDescriptorCount,
                    }));

    CHECK(derived->bindingSets[2].empty());
    const auto& set3 = derived->bindingSets[3];
    REQUIRE(set3.size() == 1);
    CHECK(sameEntry(set3[0],
                    BindingLayoutEntry{
                        .binding = 2,
                        .type = BindingType::readOnlyStorageBuffer,
                        .visibility = ShaderStage::vertex,
                    }));
}

TEST_CASE("derivePipelineLayout returns no sets or ranges for shaders without resources",
          "[size:small][rhi][core][pipeline-layout]")
{
    const std::array shaders{
        ShaderInterfaceSnapshot{.stage = ShaderStage::vertex},
        ShaderInterfaceSnapshot{.stage = ShaderStage::pixel},
    };
    const auto derived = rachis::derivePipelineLayout(shaders);
    REQUIRE(derived.has_value());
    CHECK(derived->bindingSets.empty());
    CHECK(derived->pushConstantRanges.empty());

    const auto none = rachis::derivePipelineLayout({});
    REQUIRE(none.has_value());
    CHECK(none->bindingSets.empty());
}

TEST_CASE("derivePipelineLayout reports a binding that two stages declare differently",
          "[size:small][rhi][core][pipeline-layout]")
{
    const std::array vertexBindings{
        ShaderBindingRequirement{.set = 1, .binding = 3, .type = ShaderResourceType::uniformBuffer},
    };
    SECTION("different resource types")
    {
        const std::array pixelBindings{
            ShaderBindingRequirement{.set = 1, .binding = 3, .type = ShaderResourceType::storageBuffer},
        };
        const std::array shaders{
            ShaderInterfaceSnapshot{
                .stage = ShaderStage::vertex,
                .bindings = {vertexBindings.begin(), vertexBindings.end()},
            },
            ShaderInterfaceSnapshot{
                .stage = ShaderStage::pixel,
                .bindings = {pixelBindings.begin(), pixelBindings.end()},
            },
        };
        const auto derived = rachis::derivePipelineLayout(shaders);
        REQUIRE_FALSE(derived.has_value());
        CHECK(derived.error().code == rachis::ErrorCode::validationError);
        CHECK(derived.error().message ==
              "Set 1 binding 3 is a uniform buffer with count 1 in the vertex shader, but a storage buffer with "
              "count 1 in the pixel shader.");
    }
    SECTION("different counts")
    {
        const std::array pixelBindings{
            ShaderBindingRequirement{
                .set = 1,
                .binding = 3,
                .type = ShaderResourceType::uniformBuffer,
                .count = rachis::unboundedBindingCount,
            },
        };
        const std::array shaders{
            ShaderInterfaceSnapshot{
                .stage = ShaderStage::vertex,
                .bindings = {vertexBindings.begin(), vertexBindings.end()},
            },
            ShaderInterfaceSnapshot{
                .stage = ShaderStage::pixel,
                .bindings = {pixelBindings.begin(), pixelBindings.end()},
            },
        };
        const auto derived = rachis::derivePipelineLayout(shaders);
        REQUIRE_FALSE(derived.has_value());
        CHECK(derived.error().code == rachis::ErrorCode::validationError);
        CHECK(derived.error().message ==
              "Set 1 binding 3 is a uniform buffer with count 1 in the vertex shader, but a uniform buffer with an "
              "unbounded count in the pixel shader.");
    }
}

TEST_CASE("derivePipelineLayout merges push-constant ranges per stage",
          "[size:small][rhi][core][pipeline-layout][push-constants]")
{
    SECTION("stages that declare the same bytes share one range")
    {
        const std::array shaders{
            pushOnly(ShaderStage::vertex, {{.offset = 0, .size = 16}}),
            pushOnly(ShaderStage::pixel, {{.offset = 0, .size = 16}}),
        };
        const auto derived = rachis::derivePipelineLayout(shaders);
        REQUIRE(derived.has_value());
        CHECK(sameRanges(derived->pushConstantRanges,
                         {{.visibility = ShaderStage::vertex | ShaderStage::pixel, .offset = 0, .size = 16}}));
        requireExactPerStageCoverage(shaders, derived->pushConstantRanges);
    }
    SECTION("a stage that declares more bytes keeps one range of its own")
    {
        const std::array shaders{
            pushOnly(ShaderStage::vertex, {{.offset = 0, .size = 16}}),
            pushOnly(ShaderStage::pixel, {{.offset = 0, .size = 32}}),
        };
        const auto derived = rachis::derivePipelineLayout(shaders);
        REQUIRE(derived.has_value());
        CHECK(sameRanges(derived->pushConstantRanges,
                         {
                             {.visibility = ShaderStage::vertex, .offset = 0, .size = 16},
                             {.visibility = ShaderStage::pixel, .offset = 0, .size = 32},
                         }));
        requireExactPerStageCoverage(shaders, derived->pushConstantRanges);
    }
    SECTION("adjacent and overlapping ranges of one stage merge, and disjoint ones stay apart")
    {
        const std::array shaders{
            pushOnly(ShaderStage::pixel,
                     {
                         {.offset = 0, .size = 8},
                         {.offset = 8, .size = 8},
                         {.offset = 4, .size = 8},
                         {.offset = 32, .size = 16},
                     }),
            pushOnly(ShaderStage::vertex, {{.offset = 16, .size = 16}}),
        };
        const auto derived = rachis::derivePipelineLayout(shaders);
        REQUIRE(derived.has_value());
        CHECK(sameRanges(derived->pushConstantRanges,
                         {
                             {.visibility = ShaderStage::pixel, .offset = 0, .size = 16},
                             {.visibility = ShaderStage::vertex, .offset = 16, .size = 16},
                             {.visibility = ShaderStage::pixel, .offset = 32, .size = 16},
                         }));
        requireExactPerStageCoverage(shaders, derived->pushConstantRanges);
    }
    SECTION("shaders of the same stage, as in a ray tracing pipeline, contribute the union of their bytes")
    {
        const std::array shaders{
            pushOnly(ShaderStage::rayGeneration, {{.offset = 0, .size = 8}}),
            pushOnly(ShaderStage::closestHit, {{.offset = 0, .size = 8}}),
            pushOnly(ShaderStage::closestHit, {{.offset = 8, .size = 8}}),
            pushOnly(ShaderStage::miss, {{.offset = 0, .size = 16}}),
        };
        const auto derived = rachis::derivePipelineLayout(shaders);
        REQUIRE(derived.has_value());
        CHECK(sameRanges(derived->pushConstantRanges,
                         {
                             {.visibility = ShaderStage::rayGeneration, .offset = 0, .size = 8},
                             {.visibility = ShaderStage::closestHit | ShaderStage::miss, .offset = 0, .size = 16},
                         }));
        requireExactPerStageCoverage(shaders, derived->pushConstantRanges);
    }
    SECTION("the full 128 bytes are allowed")
    {
        const std::array shaders{
            pushOnly(ShaderStage::compute, {{.offset = 0, .size = rachis::maxPushConstantBytes}}),
        };
        const auto derived = rachis::derivePipelineLayout(shaders);
        REQUIRE(derived.has_value());
        CHECK(sameRanges(derived->pushConstantRanges,
                         {{.visibility = ShaderStage::compute, .offset = 0, .size = rachis::maxPushConstantBytes}}));
    }
}

TEST_CASE("derivePipelineLayout rejects interfaces that a shader module could not have",
          "[size:small][rhi][core][pipeline-layout]")
{
    const auto rejects = [](const ShaderInterfaceSnapshot& shader) {
        const auto derived = rachis::derivePipelineLayout(std::span(&shader, 1));
        REQUIRE_FALSE(derived.has_value());
        CHECK(derived.error().code == rachis::ErrorCode::invalidArgument);
    };
    rejects(ShaderInterfaceSnapshot{.stage = ShaderStage::none});
    rejects(ShaderInterfaceSnapshot{.stage = ShaderStage::vertex | ShaderStage::pixel});
    rejects(ShaderInterfaceSnapshot{
        .stage = ShaderStage::pixel,
        .bindings = {ShaderBindingRequirement{.set = rachis::maxBindingSets, .binding = 0}},
    });
    rejects(ShaderInterfaceSnapshot{
        .stage = ShaderStage::pixel,
        .bindings = {ShaderBindingRequirement{.set = 0, .binding = 0, .count = 0}},
    });
    rejects(ShaderInterfaceSnapshot{
        .stage = ShaderStage::pixel,
        .bindings = {ShaderBindingRequirement{.type = static_cast<ShaderResourceType>(200)}},
    });
    rejects(pushOnly(ShaderStage::pixel, {{.offset = 2, .size = 4}}));
    rejects(pushOnly(ShaderStage::pixel, {{.offset = 0, .size = 0}}));
    rejects(pushOnly(ShaderStage::pixel, {{.offset = 120, .size = 16}}));
}
