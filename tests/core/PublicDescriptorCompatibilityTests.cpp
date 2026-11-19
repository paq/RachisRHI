#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace
{

    using rachis::BindingLayoutEntry;
    using rachis::BindingType;
    using rachis::BufferHandle;
    using rachis::ShaderBindingRequirement;
    using rachis::ShaderPushConstantRequirement;
    using rachis::ShaderResourceType;
    using rachis::ShaderStage;
    using rachis::ShaderWorkgroupSize;

    constexpr std::array bindingLayouts{rachis::BindingLayoutHandle{1}, rachis::BindingLayoutHandle{2}};
    constexpr std::array pushConstantRanges{rachis::PushConstantRange{ShaderStage::vertex, 4, 16}};
    constexpr std::array bindings{ShaderBindingRequirement{1, 2, ShaderResourceType::storageBuffer, 3}};
    constexpr std::array pushConstants{ShaderPushConstantRequirement{8, 24}};
    constexpr std::array vertexInputLocations{0U, 3U};

    constexpr rachis::PipelineLayoutDesc pipelineLayout{bindingLayouts, pushConstantRanges, std::string_view{"scene"}};
    static_assert(pipelineLayout.bindingLayouts.size() == 2);
    static_assert(pipelineLayout.pushConstantRanges.size() == 1);
    static_assert(pipelineLayout.label == "scene");
    static_assert(!pipelineLayout.drawIndex);

    constexpr rachis::ShaderInterfaceDesc shaderInterface{
        bindings,
        pushConstants,
        vertexInputLocations,
        ShaderWorkgroupSize{8, 4, 2},
        rachis::ShaderRecordConstantRange{1, 2, 16},
    };
    static_assert(shaderInterface.bindings.size() == 1);
    static_assert(shaderInterface.pushConstants.size() == 1);
    static_assert(shaderInterface.vertexInputLocations.size() == 2);
    static_assert(shaderInterface.workgroupSize->x == 8);
    static_assert(shaderInterface.shaderRecordConstants->size == 16);
    static_assert(!shaderInterface.drawIndex);

    constexpr rachis::ComputePipelineDesc computePipeline{
        rachis::PipelineLayoutHandle{3},
        rachis::ShaderModuleHandle{4},
        std::string_view{"cull"},
        rachis::PipelineCacheHandle{5},
        rachis::PipelineCacheKey{6, 7},
        rachis::PipelineCachePolicy::requireHit,
    };
    static_assert(computePipeline.layout.identity == 3);
    static_assert(computePipeline.computeShader.identity == 4);
    static_assert(computePipeline.label == "cull");
    static_assert(computePipeline.cache.identity == 5);
    static_assert(computePipeline.cacheKey.low == 7);
    static_assert(computePipeline.cachePolicy == rachis::PipelineCachePolicy::requireHit);
    static_assert(computePipeline.requiredSubgroupSize == 0);

    constexpr rachis::IndirectDrawDesc indirectDraw{BufferHandle{8}, 16, 3, 20, BufferHandle{9}, 4};
    static_assert(indirectDraw.argumentBuffer.identity == 8);
    static_assert(indirectDraw.argumentOffset == 16);
    static_assert(indirectDraw.maxDrawCount == 3);
    static_assert(indirectDraw.stride == 20);
    static_assert(indirectDraw.countBuffer.identity == 9);
    static_assert(indirectDraw.countOffset == 4);
    static_assert(!indirectDraw.drawIndexPrefix);

    constexpr rachis::IndirectMeshDispatchDesc indirectMesh{BufferHandle{10}, 32, 5, 16, BufferHandle{11}, 8};
    static_assert(indirectMesh.argumentBuffer.identity == 10);
    static_assert(indirectMesh.argumentOffset == 32);
    static_assert(indirectMesh.maxDispatchCount == 5);
    static_assert(indirectMesh.stride == 16);
    static_assert(indirectMesh.countBuffer.identity == 11);
    static_assert(indirectMesh.countOffset == 8);
    static_assert(!indirectMesh.drawIndexPrefix);

    constexpr rachis::SubgroupCapabilities subgroup{8, 64};
    static_assert(subgroup.minSize == 8);
    static_assert(subgroup.maxSize == 64);
    static_assert(subgroup.requiredSizeStages == ShaderStage::none);

} // namespace

TEST_CASE("Public descriptors keep the positional member order of the last release", "[size:small][rhi][core][abi]")
{
    const rachis::ShaderInterfaceSnapshot snapshot{
        ShaderStage::vertex,
        {bindings.begin(), bindings.end()},
        {pushConstants.begin(), pushConstants.end()},
        {vertexInputLocations.begin(), vertexInputLocations.end()},
        std::nullopt,
        rachis::ShaderRecordConstantRange{1, 2, 16},
    };
    CHECK(snapshot.stage == ShaderStage::vertex);
    CHECK(snapshot.bindings.size() == 1);
    CHECK(snapshot.pushConstants.size() == 1);
    CHECK(snapshot.vertexInputLocations == std::vector<std::uint32_t>{0U, 3U});
    CHECK_FALSE(snapshot.workgroupSize.has_value());
    CHECK(snapshot.shaderRecordConstants->registerSpace == 2);
    CHECK_FALSE(snapshot.drawIndex);

    const rachis::DerivedPipelineLayout derived{
        {{BindingLayoutEntry{0, BindingType::storageBuffer, ShaderStage::compute}}},
        {pushConstantRanges.begin(), pushConstantRanges.end()},
    };
    CHECK(derived.bindingSets.size() == 1);
    CHECK(derived.pushConstantRanges.size() == 1);
    CHECK_FALSE(derived.drawIndex);
}
