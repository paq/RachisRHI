#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanPipelines.h"
#include "Internal/VulkanValidationHelpers.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <expected>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rachis::vulkan
{
    namespace
    {

        [[nodiscard]] constexpr VkShaderStageFlagBits nativeRayTracingStage(ShaderStage stage) noexcept
        {
            switch (stage)
            {
            case ShaderStage::rayGeneration:
                return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            case ShaderStage::anyHit:
                return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
            case ShaderStage::closestHit:
                return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            case ShaderStage::miss:
                return VK_SHADER_STAGE_MISS_BIT_KHR;
            case ShaderStage::intersection:
                return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
            case ShaderStage::callable:
                return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
            default:
                return VK_SHADER_STAGE_ALL;
            }
        }

        [[nodiscard]] constexpr bool isGeneralRayTracingStage(ShaderStage stage) noexcept
        {
            return stage == ShaderStage::rayGeneration || stage == ShaderStage::miss || stage == ShaderStage::callable;
        }

        [[nodiscard]] constexpr std::optional<ShaderResourceType> shaderResourceType(BindingType type) noexcept
        {
            switch (type)
            {
            case BindingType::uniformBuffer:
            case BindingType::dynamicUniformBuffer:
                return ShaderResourceType::uniformBuffer;
            case BindingType::readOnlyStorageBuffer:
            case BindingType::dynamicReadOnlyStorageBuffer:
                return ShaderResourceType::readOnlyStorageBuffer;
            case BindingType::storageBuffer:
            case BindingType::dynamicStorageBuffer:
                return ShaderResourceType::storageBuffer;
            case BindingType::sampledTexture:
                return ShaderResourceType::sampledTexture;
            case BindingType::storageTexture:
                return ShaderResourceType::storageTexture;
            case BindingType::sampler:
                return ShaderResourceType::sampler;
            case BindingType::accelerationStructure:
                return ShaderResourceType::accelerationStructure;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] Status validateRayTracingShaderInterface(const VulkanShaderManager::NativeShaderModule& shader,
                                                               const VulkanNativePipelineLayout& layout)
        {
            for (const auto& binding : shader.bindings)
            {
                if (binding.set >= layout.sets.size())
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "A shader-interface binding references a set absent from the pipeline layout.",
                    });
                }
                const auto& entries = layout.sets[binding.set].entries;
                const auto found = std::ranges::find(entries, binding.binding, &BindingLayoutEntry::binding);
                const auto foundType =
                    found == entries.end() ? std::optional<ShaderResourceType>{} : shaderResourceType(found->type);
                if (found == entries.end() || !foundType.has_value() || *foundType != binding.type ||
                    found->count != binding.count || !hasAny(found->visibility, shader.stage))
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "A pipeline-layout binding does not match the shader interface's type, count, or visibility.",
                    });
                }
            }
            return validation_detail::validateShaderPushConstantCoverage(
                shader.pushConstants, layout.pushConstantRanges, shader.stage);
        }

        [[nodiscard]] constexpr std::uint32_t alignedRecordSize(const RayTracingCapabilities& limits,
                                                                std::uint32_t localSize) noexcept
        {
            const auto unaligned = static_cast<std::uint64_t>(limits.shaderGroupHandleSize) + localSize;
            const auto alignment = static_cast<std::uint64_t>(limits.shaderGroupHandleAlignment);
            if (alignment == 0)
            {
                return static_cast<std::uint32_t>(unaligned);
            }
            return static_cast<std::uint32_t>(((unaligned + alignment - 1U) / alignment) * alignment);
        }

        [[nodiscard]] Status validateRecordConstants(const VulkanPipelineManager::RayTracingGroupRecord& group,
                                                     const RayTracingCapabilities& limits)
        {
            const auto modules = {group.hasGeneral ? &group.general : nullptr,
                                  group.hasClosestHit ? &group.closestHit : nullptr,
                                  group.hasAnyHit ? &group.anyHit : nullptr,
                                  group.hasIntersection ? &group.intersection : nullptr};
            if (!group.shaderRecordConstants.has_value())
            {
                for (const auto* shader : modules)
                {
                    if (shader != nullptr && shader->shaderRecordConstants.has_value())
                    {
                        return std::unexpected(Error{
                            ErrorCode::validationError,
                            0,
                            "A shader-record constant range does not match the shader interface metadata.",
                        });
                    }
                }
                return {};
            }

            const auto& constants = *group.shaderRecordConstants;
            if (constants.size == 0 || constants.size % sizeof(std::uint32_t) != 0 ||
                limits.shaderGroupHandleSize > limits.maxShaderGroupStride ||
                constants.size > limits.maxShaderGroupStride - limits.shaderGroupHandleSize ||
                alignedRecordSize(limits, constants.size) > limits.maxShaderGroupStride)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader-record constants must be non-empty, DWORD aligned, and fit the maximum record stride.",
                });
            }
            for (const auto* shader : modules)
            {
                if (shader != nullptr && shader->shaderRecordConstants.has_value() &&
                    shader->shaderRecordConstants != group.shaderRecordConstants)
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "A shader-record constant range does not match the shader interface metadata.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Status resolveGroupShader(VulkanShaderManager& shaders,
                                                const VulkanNativePipelineLayout& layout,
                                                ShaderModuleHandle handle,
                                                VulkanShaderManager::NativeShaderModule& module,
                                                bool& present)
        {
            if (!handle.isValid())
            {
                return {};
            }
            auto native = shaders.nativeShaderModule(handle);
            if (!native)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A ray-tracing shader group references an invalid or stale module.",
                });
            }
            if (auto interfaceValidation = validateRayTracingShaderInterface(*native, layout); !interfaceValidation)
            {
                return interfaceValidation;
            }
            module = std::move(*native);
            present = true;
            return {};
        }

        [[nodiscard]] bool hasCorrectlyStagedHitShaders(
            const VulkanPipelineManager::RayTracingGroupRecord& group) noexcept
        {
            return (!group.hasClosestHit || group.closestHit.stage == ShaderStage::closestHit) &&
                   (!group.hasAnyHit || group.anyHit.stage == ShaderStage::anyHit);
        }

        [[nodiscard]] bool isValidGeneralGroup(const VulkanPipelineManager::RayTracingGroupRecord& group) noexcept
        {
            return group.hasGeneral && !group.hasClosestHit && !group.hasAnyHit && !group.hasIntersection &&
                   isGeneralRayTracingStage(group.general.stage);
        }

        [[nodiscard]] bool isValidTrianglesHitGroup(const VulkanPipelineManager::RayTracingGroupRecord& group) noexcept
        {
            return !group.hasGeneral && !group.hasIntersection && (group.hasClosestHit || group.hasAnyHit) &&
                   hasCorrectlyStagedHitShaders(group);
        }

        [[nodiscard]] bool isValidProceduralHitGroup(const VulkanPipelineManager::RayTracingGroupRecord& group) noexcept
        {
            return !group.hasGeneral && group.hasIntersection &&
                   group.intersection.stage == ShaderStage::intersection && hasCorrectlyStagedHitShaders(group);
        }

        [[nodiscard]] Status validateGroupShaderStages(const VulkanPipelineManager::RayTracingGroupRecord& group)
        {
            switch (group.type)
            {
            case RayTracingShaderGroupType::general:
                if (!isValidGeneralGroup(group))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A general ray-tracing group requires exactly one ray-generation, miss, or callable module.",
                    });
                }
                break;
            case RayTracingShaderGroupType::trianglesHitGroup:
                if (!isValidTrianglesHitGroup(group))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A triangle hit group requires at least one correctly staged hit shader and no general or "
                        "intersection shader.",
                    });
                }
                break;
            case RayTracingShaderGroupType::proceduralHitGroup:
                if (!isValidProceduralHitGroup(group))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A procedural hit group requires an intersection module and correctly staged optional hit "
                        "modules.",
                    });
                }
                break;
            default:
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, "A ray-tracing shader group has an invalid type."});
            }
            return {};
        }

        [[nodiscard]] Result<VulkanPipelineManager::RayTracingGroupRecord> resolveGroup(
            VulkanShaderManager& shaders,
            const VulkanNativePipelineLayout& layout,
            const RayTracingShaderGroupDesc& desc,
            const RayTracingCapabilities& limits)
        {
            VulkanPipelineManager::RayTracingGroupRecord group{
                .type = desc.type,
                .shaderRecordConstants = desc.shaderRecordConstants,
            };
            if (auto status = resolveGroupShader(shaders, layout, desc.generalShader, group.general, group.hasGeneral);
                !status)
            {
                return std::unexpected(std::move(status.error()));
            }
            if (auto status =
                    resolveGroupShader(shaders, layout, desc.closestHitShader, group.closestHit, group.hasClosestHit);
                !status)
            {
                return std::unexpected(std::move(status.error()));
            }
            if (auto status = resolveGroupShader(shaders, layout, desc.anyHitShader, group.anyHit, group.hasAnyHit);
                !status)
            {
                return std::unexpected(std::move(status.error()));
            }
            if (auto status = resolveGroupShader(
                    shaders, layout, desc.intersectionShader, group.intersection, group.hasIntersection);
                !status)
            {
                return std::unexpected(std::move(status.error()));
            }
            if (auto stages = validateGroupShaderStages(group); !stages)
            {
                return std::unexpected(std::move(stages.error()));
            }
            if (auto constants = validateRecordConstants(group, limits); !constants)
            {
                return std::unexpected(std::move(constants.error()));
            }
            group.localConstantSize = group.shaderRecordConstants.has_value() ? group.shaderRecordConstants->size : 0U;
            group.recordSize = alignedRecordSize(limits, group.localConstantSize);
            return group;
        }

        [[nodiscard]] bool isRayGenerationGroup(const VulkanPipelineManager::RayTracingGroupRecord& group) noexcept
        {
            return group.hasGeneral && group.general.stage == ShaderStage::rayGeneration;
        }

        [[nodiscard]] Status validateRayTracingPipelineDesc(const RayTracingPipelineDesc& desc,
                                                            const RayTracingCapabilities& limits)
        {
            const bool hasNoGroups = desc.libraries.empty() && desc.groups.empty();
            const bool hasMisalignedShaderData =
                desc.maxPayloadSize % sizeof(std::uint32_t) != 0 || desc.maxAttributeSize % sizeof(std::uint32_t) != 0;
            const bool exceedsAttributeLimit = desc.maxAttributeSize > limits.maxAttributeSize;
            const bool hasUnsupportedRecursionDepth =
                desc.maxRecursionDepth == 0 || desc.maxRecursionDepth > limits.maxRecursionDepth;
            const bool lacksShaderGroupHandles = limits.shaderGroupHandleSize == 0;
            if (hasNoGroups || hasMisalignedShaderData || exceedsAttributeLimit || hasUnsupportedRecursionDepth ||
                lacksShaderGroupHandles)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A ray-tracing pipeline requires linked or local groups, scalar-aligned shader data, and supported "
                    "attribute and recursion limits.",
                });
            }
            return {};
        }

        [[nodiscard]] bool matchesPipelineInterface(const VulkanPipelineManager::RayTracingLibraryRecord& library,
                                                    const VulkanNativePipelineLayout& layout,
                                                    const RayTracingPipelineDesc& desc) noexcept
        {
            return library.pipelineLayout == layout.pipelineLayout && library.maxPayloadSize == desc.maxPayloadSize &&
                   library.maxAttributeSize == desc.maxAttributeSize;
        }

        [[nodiscard]] Result<std::vector<VulkanPipelineManager::RayTracingGroupRecord>> resolvePipelineGroups(
            VulkanShaderManager& shaders,
            const VulkanNativePipelineLayout& layout,
            const RayTracingPipelineDesc& desc,
            const RayTracingCapabilities& limits,
            std::span<const std::shared_ptr<VulkanPipelineManager::RayTracingLibraryRecord>> libraries)
        {
            std::vector<VulkanPipelineManager::RayTracingGroupRecord> groups;
            bool hasRayGeneration = false;
            for (const auto& library : libraries)
            {
                hasRayGeneration = hasRayGeneration || library->hasRayGeneration;
                groups.insert(groups.end(), library->groups.begin(), library->groups.end());
            }
            for (const auto& groupDesc : desc.groups)
            {
                auto group = resolveGroup(shaders, layout, groupDesc, limits);
                if (!group)
                {
                    return std::unexpected(std::move(group.error()));
                }
                hasRayGeneration = hasRayGeneration || isRayGenerationGroup(*group);
                groups.push_back(std::move(*group));
            }
            if (groups.empty() || !hasRayGeneration)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A linked ray-tracing pipeline requires at least one ray-generation group.",
                });
            }
            return groups;
        }

        struct StageKey final
        {
            VkShaderModule module = VK_NULL_HANDLE;
            ShaderStage stage = ShaderStage::none;
            std::string entryPoint;
            std::vector<SpecializationConstant> specializationConstants;

            friend bool operator==(const StageKey&, const StageKey&) = default;
        };

        struct StageKeyHash final
        {
            [[nodiscard]] std::size_t operator()(const StageKey& key) const noexcept
            {
                auto hash = std::hash<VkShaderModule>{}(key.module) ^
                            (std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(key.stage)) << 1U) ^
                            (std::hash<std::string>{}(key.entryPoint) << 2U);
                for (const auto& constant : key.specializationConstants)
                {
                    hash ^= std::hash<std::uint32_t>{}(constant.id) + (hash << 6U) + (hash >> 2U);
                    hash ^= std::hash<std::uint32_t>{}(constant.valueBits) + (hash << 6U) + (hash >> 2U);
                }
                return hash;
            }
        };

        struct StageSpecialization final
        {
            std::vector<VkSpecializationMapEntry> entries;
            std::vector<std::uint32_t> values;
            VkSpecializationInfo info{};
        };

        struct UniqueShaderStages final
        {
            std::vector<VkPipelineShaderStageCreateInfo> createInfos;
            std::deque<StageSpecialization> specializations;
            std::unordered_map<StageKey, std::uint32_t, StageKeyHash> indices;
        };

        [[nodiscard]] const VkSpecializationInfo* storeSpecialization(
            std::deque<StageSpecialization>& specializations, const std::vector<SpecializationConstant>& constants)
        {
            if (constants.empty())
            {
                return nullptr;
            }
            auto& storage = specializations.emplace_back();
            for (const auto& constant : constants)
            {
                storage.entries.push_back({constant.id,
                                           static_cast<std::uint32_t>(storage.values.size() * sizeof(std::uint32_t)),
                                           sizeof(std::uint32_t)});
                storage.values.push_back(constant.valueBits);
            }
            storage.info = {static_cast<std::uint32_t>(storage.entries.size()),
                            storage.entries.data(),
                            storage.values.size() * sizeof(std::uint32_t),
                            storage.values.data()};
            return &storage.info;
        }

        [[nodiscard]] std::uint32_t addUniqueStage(UniqueShaderStages& stages,
                                                   const VulkanShaderManager::NativeShaderModule& shader,
                                                   bool present)
        {
            if (!present)
            {
                return VK_SHADER_UNUSED_KHR;
            }
            const StageKey key{
                .module = shader.module,
                .stage = shader.stage,
                .entryPoint = shader.entryPoint,
                .specializationConstants = shader.specializationConstants,
            };
            if (const auto found = stages.indices.find(key); found != stages.indices.end())
            {
                return found->second;
            }
            const auto index = static_cast<std::uint32_t>(stages.createInfos.size());
            const auto* specialization = storeSpecialization(stages.specializations, shader.specializationConstants);
            stages.createInfos.push_back(VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .stage = nativeRayTracingStage(shader.stage),
                .module = shader.module,
                .pName = shader.entryPoint.c_str(),
                .pSpecializationInfo = specialization,
            });
            stages.indices.emplace(key, index);
            return index;
        }

        [[nodiscard]] VkRayTracingShaderGroupCreateInfoKHR nativeShaderGroup(
            UniqueShaderStages& stages, const VulkanPipelineManager::RayTracingGroupRecord& group)
        {
            VkRayTracingShaderGroupCreateInfoKHR native{
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .pNext = {},
                .type = {},
                .generalShader = VK_SHADER_UNUSED_KHR,
                .closestHitShader = VK_SHADER_UNUSED_KHR,
                .anyHitShader = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
                .pShaderGroupCaptureReplayHandle = {},
            };
            switch (group.type)
            {
            case RayTracingShaderGroupType::general:
                native.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
                native.generalShader = addUniqueStage(stages, group.general, group.hasGeneral);
                break;
            case RayTracingShaderGroupType::trianglesHitGroup:
                native.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
                native.closestHitShader = addUniqueStage(stages, group.closestHit, group.hasClosestHit);
                native.anyHitShader = addUniqueStage(stages, group.anyHit, group.hasAnyHit);
                break;
            case RayTracingShaderGroupType::proceduralHitGroup:
                native.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
                native.closestHitShader = addUniqueStage(stages, group.closestHit, group.hasClosestHit);
                native.anyHitShader = addUniqueStage(stages, group.anyHit, group.hasAnyHit);
                native.intersectionShader = addUniqueStage(stages, group.intersection, group.hasIntersection);
                break;
            }
            return native;
        }

        [[nodiscard]] Result<VkPipeline> createNativeRayTracingPipeline(
            const VolkDeviceTable& table,
            VkDevice device,
            VkPipelineLayout layout,
            std::span<const VulkanPipelineManager::RayTracingGroupRecord> groups,
            std::uint32_t maxRecursionDepth)
        {
            UniqueShaderStages stages;
            std::vector<VkRayTracingShaderGroupCreateInfoKHR> nativeGroups;
            nativeGroups.reserve(groups.size());
            for (const auto& group : groups)
            {
                nativeGroups.push_back(nativeShaderGroup(stages, group));
            }
            const VkRayTracingPipelineCreateInfoKHR createInfo{
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
                .pNext = {},
                .flags = {},
                .stageCount = static_cast<std::uint32_t>(stages.createInfos.size()),
                .pStages = stages.createInfos.data(),
                .groupCount = static_cast<std::uint32_t>(nativeGroups.size()),
                .pGroups = nativeGroups.data(),
                .maxPipelineRayRecursionDepth = maxRecursionDepth,
                .pLibraryInfo = {},
                .pLibraryInterface = {},
                .pDynamicState = {},
                .layout = layout,
                .basePipelineHandle = {},
                .basePipelineIndex = {},
            };
            VkPipeline pipeline = VK_NULL_HANDLE;
            const auto created = table.vkCreateRayTracingPipelinesKHR(
                device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateRayTracingPipelinesKHR"));
            }
            return pipeline;
        }

        [[nodiscard]] std::expected<std::vector<std::byte>, VkResult> readShaderGroupHandles(
            const VolkDeviceTable& table,
            VkDevice device,
            VkPipeline pipeline,
            std::uint32_t handleSize,
            std::size_t groupCount)
        {
            const auto handleBytes = static_cast<std::size_t>(handleSize) * groupCount;
            std::vector<std::byte> handles(handleBytes);
            const auto copied = table.vkGetRayTracingShaderGroupHandlesKHR(
                device, pipeline, 0, static_cast<std::uint32_t>(groupCount), handleBytes, handles.data());
            if (copied != VK_SUCCESS)
            {
                return std::unexpected(copied);
            }
            return handles;
        }

        [[nodiscard]] std::vector<std::shared_ptr<void>> collectShaderOwners(
            std::span<const VulkanPipelineManager::RayTracingGroupRecord> groups,
            std::span<const std::shared_ptr<VulkanPipelineManager::RayTracingLibraryRecord>> libraries)
        {
            std::vector<std::shared_ptr<void>> owners;
            for (const auto& group : groups)
            {
                if (group.hasGeneral)
                {
                    owners.push_back(group.general.owner);
                }
                if (group.hasClosestHit)
                {
                    owners.push_back(group.closestHit.owner);
                }
                if (group.hasAnyHit)
                {
                    owners.push_back(group.anyHit.owner);
                }
                if (group.hasIntersection)
                {
                    owners.push_back(group.intersection.owner);
                }
            }
            for (const auto& library : libraries)
            {
                owners.push_back(library);
            }
            return owners;
        }

        [[nodiscard]] VulkanNativePipeline describeRayTracingPipeline(std::shared_ptr<void> owner,
                                                                      const VulkanNativePipelineLayout& layout,
                                                                      VkPipeline pipeline)
        {
            std::vector<std::uint8_t> setRequired;
            std::vector<std::uint32_t> setDynamicCounts;
            std::vector<VkDescriptorSetLayout> setLayouts;
            setRequired.reserve(layout.sets.size());
            setDynamicCounts.reserve(layout.sets.size());
            setLayouts.reserve(layout.sets.size());
            for (const auto& set : layout.sets)
            {
                setRequired.push_back(set.entries.empty() ? std::uint8_t{0} : std::uint8_t{1});
                setDynamicCounts.push_back(set.dynamicBufferCount);
                setLayouts.push_back(set.setLayout);
            }
            return VulkanNativePipeline{
                .owner = std::move(owner),
                .layoutOwner = layout.owner,
                .pipeline = pipeline,
                .layout = layout.pipelineLayout,
                .compute = false,
                .rayTracing = true,
                .rasterizationEnabled = false,
                .colorFormats = {},
                .vertexStrides = {},
                .vertexAlignments = {},
                .pushConstantRanges = layout.pushConstantRanges,
                .nativePushConstantRanges = layout.nativePushConstantRanges,
                .setRequired = std::move(setRequired),
                .setDynamicCounts = std::move(setDynamicCounts),
                .setLayouts = std::move(setLayouts),
            };
        }

        struct ShaderGroupRecordSizes final
        {
            std::vector<std::uint32_t> recordSizes;
            std::vector<std::uint32_t> localConstantSizes;
        };

        [[nodiscard]] ShaderGroupRecordSizes collectShaderGroupRecordSizes(
            std::span<const VulkanPipelineManager::RayTracingGroupRecord> groups)
        {
            ShaderGroupRecordSizes sizes;
            sizes.recordSizes.reserve(groups.size());
            sizes.localConstantSizes.reserve(groups.size());
            for (const auto& group : groups)
            {
                sizes.recordSizes.push_back(group.recordSize);
                sizes.localConstantSizes.push_back(group.localConstantSize);
            }
            return sizes;
        }
    } // namespace

    Result<RayTracingPipelineLibraryHandle> VulkanPipelineManager::createRayTracingPipelineLibrary(
        const RayTracingPipelineLibraryDesc& desc)
    {
        if (!capabilities_.rayTracing)
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Ray-tracing pipeline libraries require ray-tracing pipelines."});
        }
        const auto& limits = capabilities_.rayTracingLimits;
        if (desc.groups.empty() || desc.maxPayloadSize % sizeof(std::uint32_t) != 0 ||
            desc.maxAttributeSize % sizeof(std::uint32_t) != 0 || desc.maxAttributeSize > limits.maxAttributeSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A ray-tracing pipeline library requires at least one shader group and scalar-aligned shader data.",
            });
        }
        if (shaders_ == nullptr || bindings_ == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidState, 0, "The Vulkan pipeline manager is not initialized."});
        }
        auto layout = bindings_->nativePipelineLayout(desc.layout);
        if (!layout)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A ray-tracing pipeline library references an invalid or stale layout.",
            });
        }

        auto record = std::make_shared<RayTracingLibraryRecord>();
        record->pipelineLayout = layout->pipelineLayout;
        record->layoutOwner = layout->owner;
        record->maxPayloadSize = desc.maxPayloadSize;
        record->maxAttributeSize = desc.maxAttributeSize;
        record->label = std::string(desc.label);
        record->groups.reserve(desc.groups.size());
        for (const auto& groupDesc : desc.groups)
        {
            auto group = resolveGroup(*shaders_, *layout, groupDesc, limits);
            if (!group)
            {
                return std::unexpected(std::move(group.error()));
            }
            record->hasRayGeneration = record->hasRayGeneration || isRayGenerationGroup(*group);
            record->groups.push_back(std::move(*group));
        }

        const std::scoped_lock lock(mutex_);
        return rayTracingLibraries_.emplace(std::move(record));
    }

    Status VulkanPipelineManager::destroyRayTracingPipelineLibrary(RayTracingPipelineLibraryHandle library)
    {
        const std::scoped_lock lock(mutex_);
        if (!rayTracingLibraries_.erase(library))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The ray-tracing pipeline library handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<std::vector<std::shared_ptr<VulkanPipelineManager::RayTracingLibraryRecord>>> VulkanPipelineManager::
        resolveLinkedRayTracingLibraries(const RayTracingPipelineDesc& desc,
                                         const VulkanNativePipelineLayout& layout) const
    {
        std::vector<std::shared_ptr<RayTracingLibraryRecord>> libraries;
        libraries.reserve(desc.libraries.size());
        const std::scoped_lock lock(mutex_);
        for (const auto library : desc.libraries)
        {
            const auto* found = rayTracingLibraries_.find(library);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A ray-tracing pipeline references an invalid or stale library.",
                });
            }
            if (std::ranges::contains(libraries, *found))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A ray-tracing pipeline cannot link the same library more than once.",
                });
            }
            if (!matchesPipelineInterface(**found, layout, desc))
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "Every ray-tracing pipeline library must match the final layout and shader interface exactly.",
                });
            }
            libraries.push_back(*found);
        }
        return libraries;
    }

    Result<PipelineHandle> VulkanPipelineManager::createRayTracingPipeline(const RayTracingPipelineDesc& desc)
    {
        if (!capabilities_.rayTracing || table_ == nullptr || table_->vkCreateRayTracingPipelinesKHR == nullptr ||
            table_->vkGetRayTracingShaderGroupHandlesKHR == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Ray-tracing pipeline creation is unavailable on this device."});
        }
        const auto& limits = capabilities_.rayTracingLimits;
        if (auto validation = validateRayTracingPipelineDesc(desc, limits); !validation)
        {
            return std::unexpected(std::move(validation.error()));
        }
        if (shaders_ == nullptr || bindings_ == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidState, 0, "The Vulkan pipeline manager is not initialized."});
        }

        auto layout = bindings_->nativePipelineLayout(desc.layout);
        if (!layout)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A ray-tracing pipeline references an invalid or stale layout.",
            });
        }
        auto libraries = resolveLinkedRayTracingLibraries(desc, *layout);
        if (!libraries)
        {
            return std::unexpected(std::move(libraries.error()));
        }
        auto groups = resolvePipelineGroups(*shaders_, *layout, desc, limits, *libraries);
        if (!groups)
        {
            return std::unexpected(std::move(groups.error()));
        }

        auto pipeline =
            createNativeRayTracingPipeline(*table_, device_, layout->pipelineLayout, *groups, desc.maxRecursionDepth);
        if (!pipeline)
        {
            return std::unexpected(std::move(pipeline.error()));
        }
        auto handles =
            readShaderGroupHandles(*table_, device_, *pipeline, limits.shaderGroupHandleSize, groups->size());
        if (!handles)
        {
            const auto copied = handles.error();
            table_->vkDestroyPipeline(device_, *pipeline, nullptr);
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(copied), copied, "vkGetRayTracingShaderGroupHandlesKHR"));
        }

        auto keepAlive = makePipelineKeepAlive(*pipeline, layout->owner);
        keepAlive->shaderOwners = collectShaderOwners(*groups, *libraries);
        if (!desc.label.empty())
        {
            const std::string label(desc.label);
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_PIPELINE,
                                reinterpret_cast<std::uint64_t>(*pipeline),
                                label.c_str());
        }

        auto sizes = collectShaderGroupRecordSizes(*groups);
        return publishPipeline(PipelineRecord{
            .native = std::make_shared<VulkanNativePipeline>(describeRayTracingPipeline(keepAlive, *layout, *pipeline)),
            .shaderGroupHandles = std::move(*handles),
            .shaderGroupRecordSizes = std::move(sizes.recordSizes),
            .shaderGroupLocalConstantSizes = std::move(sizes.localConstantSizes),
        });
    }

    Status VulkanPipelineManager::getRayTracingShaderGroupHandles(PipelineHandle pipeline,
                                                                  std::uint32_t firstGroup,
                                                                  std::uint32_t groupCount,
                                                                  std::span<std::byte> destination) const
    {
        const std::shared_lock lock(mutex_);
        const auto* found = pipelines_.find(pipeline);
        if (found == nullptr || found->native == nullptr || !found->native->rayTracing)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shader-group handle retrieval requires a live ray-tracing pipeline.",
            });
        }
        const auto handleSize = static_cast<std::size_t>(capabilities_.rayTracingLimits.shaderGroupHandleSize);
        const auto groupTotal = handleSize == 0 ? 0 : found->shaderGroupHandles.size() / handleSize;
        if (groupCount == 0 || firstGroup >= groupTotal || groupCount > groupTotal - firstGroup ||
            destination.size() != groupCount * handleSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shader-group handle retrieval requires a non-empty in-range group span and exact output size.",
            });
        }
        std::memcpy(destination.data(),
                    found->shaderGroupHandles.data() + static_cast<std::size_t>(firstGroup) * handleSize,
                    destination.size());
        return {};
    }

    Result<std::uint32_t> VulkanPipelineManager::rayTracingShaderGroupRecordSize(PipelineHandle pipeline,
                                                                                 std::uint32_t groupIndex) const
    {
        const std::shared_lock lock(mutex_);
        const auto* found = pipelines_.find(pipeline);
        if (found == nullptr || found->native == nullptr || !found->native->rayTracing ||
            groupIndex >= found->shaderGroupRecordSizes.size())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shader-group record-size retrieval requires a live ray-tracing pipeline and in-range group index.",
            });
        }
        return found->shaderGroupRecordSizes[groupIndex];
    }

    Status VulkanPipelineManager::writeRayTracingShaderGroupRecord(PipelineHandle pipeline,
                                                                   std::uint32_t groupIndex,
                                                                   std::span<const std::byte> localConstants,
                                                                   std::span<std::byte> destination) const
    {
        const std::shared_lock lock(mutex_);
        const auto* found = pipelines_.find(pipeline);
        if (found == nullptr || found->native == nullptr || !found->native->rayTracing)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shader-group record encoding requires a live ray-tracing pipeline.",
            });
        }
        const auto handleSize = static_cast<std::size_t>(capabilities_.rayTracingLimits.shaderGroupHandleSize);
        if (groupIndex >= found->shaderGroupRecordSizes.size() ||
            groupIndex >= found->shaderGroupLocalConstantSizes.size() ||
            localConstants.size() != found->shaderGroupLocalConstantSizes[groupIndex] ||
            destination.size() != found->shaderGroupRecordSizes[groupIndex] || handleSize > destination.size())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shader-group record encoding requires an in-range group and exact local-data and output sizes.",
            });
        }
        std::ranges::fill(destination, std::byte{0});
        std::memcpy(destination.data(),
                    found->shaderGroupHandles.data() + static_cast<std::size_t>(groupIndex) * handleSize,
                    handleSize);
        if (!localConstants.empty())
        {
            std::memcpy(destination.data() + handleSize, localConstants.data(), localConstants.size());
        }
        return {};
    }

} // namespace rachis::vulkan
