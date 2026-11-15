#pragma once

#include "Rachis/Binding.h"
#include "Rachis/Device.h"
#include "Rachis/Resource.h"

#include <volk.h>

namespace rachis::vulkan
{

    inline constexpr ShaderStage coreShaderStages = ShaderStage::vertex | ShaderStage::pixel | ShaderStage::compute;

    inline constexpr PipelineStage corePipelineStages =
        PipelineStage::all | PipelineStage::draw | PipelineStage::vertexInput | PipelineStage::vertexShader |
        PipelineStage::pixelShader | PipelineStage::computeShader | PipelineStage::copy | PipelineStage::resolve |
        PipelineStage::renderTarget | PipelineStage::depthStencil | PipelineStage::indirect;

    [[nodiscard]] constexpr ShaderStage enabledShaderStages(const DeviceCapabilities& capabilities) noexcept
    {
        auto stages = coreShaderStages;
        if (capabilities.geometryShaders)
        {
            stages = stages | ShaderStage::geometry;
        }
        if (capabilities.tessellationShaders)
        {
            stages = stages | ShaderStage::hull | ShaderStage::domain;
        }
        if (capabilities.taskShaders)
        {
            stages = stages | ShaderStage::task;
        }
        if (capabilities.meshShaders)
        {
            stages = stages | ShaderStage::mesh;
        }
        if (capabilities.rayTracing)
        {
            stages = stages | ShaderStage::allRayTracing;
        }
        return stages;
    }

    [[nodiscard]] constexpr PipelineStage enabledPipelineStages(ShaderStage shaderStages,
                                                                bool accelerationStructures) noexcept
    {
        auto stages = corePipelineStages;
        if (hasAny(shaderStages, ShaderStage::geometry))
        {
            stages = stages | PipelineStage::geometryShader;
        }
        if (hasAny(shaderStages, ShaderStage::hull))
        {
            stages = stages | PipelineStage::hullShader;
        }
        if (hasAny(shaderStages, ShaderStage::domain))
        {
            stages = stages | PipelineStage::domainShader;
        }
        if (hasAny(shaderStages, ShaderStage::task))
        {
            stages = stages | PipelineStage::taskShader;
        }
        if (hasAny(shaderStages, ShaderStage::mesh))
        {
            stages = stages | PipelineStage::meshShader;
        }
        if (hasAny(shaderStages, ShaderStage::allRayTracing))
        {
            stages = stages | PipelineStage::rayTracingShader;
        }
        if (accelerationStructures)
        {
            stages = stages | PipelineStage::accelerationStructureBuild;
        }
        return stages;
    }

    [[nodiscard]] constexpr VkShaderStageFlags nativeShaderStageFlags(ShaderStage stages) noexcept
    {
        VkShaderStageFlags flags = 0;
        if (hasAny(stages, ShaderStage::vertex))
        {
            flags |= VK_SHADER_STAGE_VERTEX_BIT;
        }
        if (hasAny(stages, ShaderStage::pixel))
        {
            flags |= VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        if (hasAny(stages, ShaderStage::compute))
        {
            flags |= VK_SHADER_STAGE_COMPUTE_BIT;
        }
        if (hasAny(stages, ShaderStage::geometry))
        {
            flags |= VK_SHADER_STAGE_GEOMETRY_BIT;
        }
        if (hasAny(stages, ShaderStage::hull))
        {
            flags |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        }
        if (hasAny(stages, ShaderStage::domain))
        {
            flags |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        }
        if (hasAny(stages, ShaderStage::task))
        {
            flags |= VK_SHADER_STAGE_TASK_BIT_EXT;
        }
        if (hasAny(stages, ShaderStage::mesh))
        {
            flags |= VK_SHADER_STAGE_MESH_BIT_EXT;
        }
        if (hasAny(stages, ShaderStage::rayGeneration))
        {
            flags |= VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        }
        if (hasAny(stages, ShaderStage::anyHit))
        {
            flags |= VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
        }
        if (hasAny(stages, ShaderStage::closestHit))
        {
            flags |= VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        }
        if (hasAny(stages, ShaderStage::miss))
        {
            flags |= VK_SHADER_STAGE_MISS_BIT_KHR;
        }
        if (hasAny(stages, ShaderStage::intersection))
        {
            flags |= VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
        }
        if (hasAny(stages, ShaderStage::callable))
        {
            flags |= VK_SHADER_STAGE_CALLABLE_BIT_KHR;
        }
        return flags;
    }

} // namespace rachis::vulkan
