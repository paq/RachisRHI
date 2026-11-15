#include "D3D12/D3D12PipelineInternal.h"

namespace rachis::d3d12
{
    using Microsoft::WRL::ComPtr;
    using namespace pipeline_detail;

    namespace
    {

        using ShaderModuleTable = detail::HandleTable<std::shared_ptr<ShaderModuleRecord>, ShaderModuleHandle>;
        using RayTracingPipelineLibraryTable =
            detail::HandleTable<std::shared_ptr<RayTracingPipelineLibraryRecord>, RayTracingPipelineLibraryHandle>;

        [[nodiscard]] std::uint64_t alignedShaderRecordSize(const RayTracingCapabilities& limits,
                                                            std::uint32_t localSize)
        {
            const auto unalignedSize = static_cast<std::uint64_t>(limits.shaderGroupHandleSize) + localSize;
            return ((unalignedSize + limits.shaderGroupHandleAlignment - 1U) / limits.shaderGroupHandleAlignment) *
                   limits.shaderGroupHandleAlignment;
        }

        [[nodiscard]] std::array<std::shared_ptr<ShaderModuleRecord>, 4> groupShaders(
            const ResolvedRayTracingGroup& group)
        {
            return {group.general, group.closestHit, group.anyHit, group.intersection};
        }

        [[nodiscard]] constexpr bool isGeneralStage(ShaderStage stage) noexcept
        {
            return stage == ShaderStage::rayGeneration || stage == ShaderStage::miss || stage == ShaderStage::callable;
        }

        [[nodiscard]] bool isAbsentOrHasStage(const std::shared_ptr<ShaderModuleRecord>& shader,
                                              ShaderStage stage) noexcept
        {
            return shader == nullptr || shader->stage == stage;
        }

        [[nodiscard]] bool isWellFormedGeneralGroup(const ResolvedRayTracingGroup& group) noexcept
        {
            return group.general != nullptr && group.closestHit == nullptr && group.anyHit == nullptr &&
                   group.intersection == nullptr && isGeneralStage(group.general->stage);
        }

        [[nodiscard]] bool isWellFormedTrianglesHitGroup(const ResolvedRayTracingGroup& group) noexcept
        {
            return group.general == nullptr && group.intersection == nullptr &&
                   (group.closestHit != nullptr || group.anyHit != nullptr) &&
                   isAbsentOrHasStage(group.closestHit, ShaderStage::closestHit) &&
                   isAbsentOrHasStage(group.anyHit, ShaderStage::anyHit);
        }

        [[nodiscard]] bool isWellFormedProceduralHitGroup(const ResolvedRayTracingGroup& group) noexcept
        {
            return group.general == nullptr && group.intersection != nullptr &&
                   group.intersection->stage == ShaderStage::intersection &&
                   isAbsentOrHasStage(group.closestHit, ShaderStage::closestHit) &&
                   isAbsentOrHasStage(group.anyHit, ShaderStage::anyHit);
        }

        [[nodiscard]] Status validateGroupShaderStages(const ResolvedRayTracingGroup& group)
        {
            switch (group.type)
            {
            case RayTracingShaderGroupType::general:
                if (!isWellFormedGeneralGroup(group))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A general ray-tracing group requires exactly one ray-generation, miss, or callable module.",
                    });
                }
                return {};
            case RayTracingShaderGroupType::trianglesHitGroup:
                if (!isWellFormedTrianglesHitGroup(group))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A triangle hit group requires at least one correctly staged hit shader and no general or "
                        "intersection shader.",
                    });
                }
                return {};
            case RayTracingShaderGroupType::proceduralHitGroup:
                if (!isWellFormedProceduralHitGroup(group))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A procedural hit group requires an intersection module and correctly staged optional hit "
                        "modules.",
                    });
                }
                return {};
            default:
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A ray-tracing shader group has an invalid type.",
                });
            }
        }

        [[nodiscard]] bool anyShaderInterfaceDeclaresRecordConstants(const ResolvedRayTracingGroup& group)
        {
            return std::ranges::any_of(groupShaders(group), [](const std::shared_ptr<ShaderModuleRecord>& shader) {
                return shader != nullptr && shader->shaderInterface.shaderRecordConstants.has_value();
            });
        }

        [[nodiscard]] bool hasValidShaderRecordConstantSize(const ShaderRecordConstantRange& constants,
                                                            const RayTracingCapabilities& limits)
        {
            return constants.size != 0 && constants.size % sizeof(std::uint32_t) == 0 &&
                   limits.shaderGroupHandleSize <= limits.maxShaderGroupStride &&
                   constants.size <= limits.maxShaderGroupStride - limits.shaderGroupHandleSize &&
                   alignedShaderRecordSize(limits, constants.size) <= limits.maxShaderGroupStride;
        }

        [[nodiscard]] bool matchesShaderInterfaceMetadata(const ResolvedRayTracingGroup& group)
        {
            return std::ranges::none_of(groupShaders(group), [&](const std::shared_ptr<ShaderModuleRecord>& shader) {
                return shader != nullptr && shader->shaderInterface.shaderRecordConstants.has_value() &&
                       shader->shaderInterface.shaderRecordConstants != group.shaderRecordConstants;
            });
        }

        [[nodiscard]] bool overlapsGlobalConstantBuffer(const ShaderRecordConstantRange& constants,
                                                        const PipelineLayoutRecord& layout)
        {
            if (constants.registerSpace >= layout.slots.size())
            {
                return false;
            }
            const auto coversShaderRegister = [&](const BindingLayoutEntry& entry) {
                if (shaderResourceType(entry.type) != ShaderResourceType::uniformBuffer ||
                    entry.binding > constants.shaderRegister)
                {
                    return false;
                }
                return entry.count == unboundedBindingCount || constants.shaderRegister - entry.binding < entry.count;
            };
            return std::ranges::any_of(layout.slots[constants.registerSpace].entries, coversShaderRegister);
        }

        [[nodiscard]] bool overlapsGlobalPushConstants(const ShaderRecordConstantRange& constants,
                                                       const PipelineLayoutRecord& layout) noexcept
        {
            return constants.shaderRegister == pushConstantRegister &&
                   constants.registerSpace == pushConstantRegisterSpace &&
                   layout.pushConstantRootParameter != invalidRootParameter;
        }

        [[nodiscard]] Status validateShaderRecordConstants(const ResolvedRayTracingGroup& group,
                                                           const RayTracingCapabilities& limits,
                                                           const PipelineLayoutRecord& layout)
        {
            if (!group.shaderRecordConstants.has_value())
            {
                if (anyShaderInterfaceDeclaresRecordConstants(group))
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "Shader interface metadata requires shader-record constants absent from its group.",
                    });
                }
                return {};
            }

            const auto& constants = *group.shaderRecordConstants;
            if (!hasValidShaderRecordConstantSize(constants, limits))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader-record constants must be non-empty, DWORD aligned, and fit the maximum record stride.",
                });
            }
            if (!matchesShaderInterfaceMetadata(group))
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "A shader-record constant range does not match the shader interface metadata.",
                });
            }
            if (overlapsGlobalConstantBuffer(constants, layout))
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "Shader-record constants overlap a global constant-buffer binding of any stage visibility.",
                });
            }
            if (overlapsGlobalPushConstants(constants, layout))
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "Shader-record constants overlap the global push-constant binding of any stage visibility.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateShaderGroups(std::span<const ResolvedRayTracingGroup> groups,
                                                  const RayTracingCapabilities& limits,
                                                  const PipelineLayoutRecord& layout)
        {
            for (const auto& group : groups)
            {
                auto stageValidation = validateGroupShaderStages(group);
                if (!stageValidation)
                {
                    return std::unexpected(std::move(stageValidation.error()));
                }
                auto constantValidation = validateShaderRecordConstants(group, limits, layout);
                if (!constantValidation)
                {
                    return std::unexpected(std::move(constantValidation.error()));
                }
            }
            return {};
        }

        [[nodiscard]] bool isRayGenerationGroup(const ResolvedRayTracingGroup& group) noexcept
        {
            return group.type == RayTracingShaderGroupType::general &&
                   group.general->stage == ShaderStage::rayGeneration;
        }

        [[nodiscard]] std::vector<std::shared_ptr<ShaderModuleRecord>> collectDistinctShaders(
            std::span<const ResolvedRayTracingGroup> groups)
        {
            std::vector<std::shared_ptr<ShaderModuleRecord>> shaders;
            for (const auto& group : groups)
            {
                for (const auto& shader : groupShaders(group))
                {
                    if (shader != nullptr && !std::ranges::contains(shaders, shader))
                    {
                        shaders.push_back(shader);
                    }
                }
            }
            return shaders;
        }

        void appendGroupRecordSizes(RayTracingPipelineLibraryRecord& record,
                                    std::span<const ResolvedRayTracingGroup> groups,
                                    const RayTracingCapabilities& limits)
        {
            record.groupLocalConstantSizes.reserve(groups.size());
            record.groupRecordSizes.reserve(groups.size());
            for (const auto& group : groups)
            {
                const auto localSize = group.shaderRecordConstants.has_value() ? group.shaderRecordConstants->size : 0U;
                record.groupLocalConstantSizes.push_back(localSize);
                record.groupRecordSizes.push_back(
                    static_cast<std::uint32_t>(alignedShaderRecordSize(limits, localSize)));
            }
        }

        [[nodiscard]] bool isValidExportName(std::string_view entryPoint)
        {
            return !entryPoint.empty() && !entryPoint.contains('\0') &&
                   std::ranges::none_of(entryPoint, [](unsigned char character) { return character >= 0x80U; });
        }

        [[nodiscard]] Status validateCollectionShaders(std::span<const std::shared_ptr<ShaderModuleRecord>> shaders,
                                                       const PipelineLayoutRecord& layout)
        {
            for (const auto& shader : shaders)
            {
                auto interfaceValidation = validateShaderInterface(*shader, layout);
                if (!interfaceValidation)
                {
                    return std::unexpected(std::move(interfaceValidation.error()));
                }
                if (!isValidExportName(shader->entryPoint))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "D3D12 ray-tracing export names must be non-empty ASCII strings without embedded nulls.",
                    });
                }
            }
            return {};
        }

        struct HitShaderExport final
        {
            std::size_t shader = 0;
            std::optional<ShaderRecordConstantRange> constants = {};
            std::size_t variant = 0;
            std::wstring name = {};
        };

        struct ShaderExportNames final
        {
            std::span<const std::shared_ptr<ShaderModuleRecord>> shaders = {};
            std::vector<std::wstring> source = {};
            std::vector<std::wstring> exported = {};
            std::vector<HitShaderExport> hitExports = {};
        };

        [[nodiscard]] std::size_t shaderExportIndex(const ShaderExportNames& names,
                                                    const std::shared_ptr<ShaderModuleRecord>& shader)
        {
            return static_cast<std::size_t>(std::ranges::find(names.shaders, shader) - names.shaders.begin());
        }

        [[nodiscard]] bool isRenamedHitShaderVariant(const HitShaderExport& hitExport) noexcept
        {
            return hitExport.variant != 0;
        }

        void appendHitShaderExports(ShaderExportNames& names, std::span<const ResolvedRayTracingGroup> groups)
        {
            for (const auto& group : groups)
            {
                if (group.type == RayTracingShaderGroupType::general)
                {
                    continue;
                }
                for (const auto& shader : groupShaders(group))
                {
                    if (shader == nullptr)
                    {
                        continue;
                    }
                    const auto index = shaderExportIndex(names, shader);
                    const auto exportsShader = [&](const HitShaderExport& hitExport) {
                        return hitExport.shader == index;
                    };
                    const bool hasLayout = std::ranges::any_of(names.hitExports, [&](const HitShaderExport& hitExport) {
                        return exportsShader(hitExport) && hitExport.constants == group.shaderRecordConstants;
                    });
                    if (hasLayout)
                    {
                        continue;
                    }
                    const auto variant =
                        static_cast<std::size_t>(std::ranges::count_if(names.hitExports, exportsShader));
                    names.hitExports.push_back({
                        .shader = index,
                        .constants = group.shaderRecordConstants,
                        .variant = variant,
                        .name = variant == 0 ? names.exported[index]
                                             : names.exported[index] + L"_v" + std::to_wstring(variant),
                    });
                }
            }
        }

        [[nodiscard]] ShaderExportNames buildShaderExportNames(
            std::span<const std::shared_ptr<ShaderModuleRecord>> shaders,
            std::span<const ResolvedRayTracingGroup> groups,
            std::wstring_view exportNamespace)
        {
            ShaderExportNames names{.shaders = shaders};
            names.source.reserve(shaders.size());
            names.exported.reserve(shaders.size());
            for (std::size_t index = 0; index < shaders.size(); ++index)
            {
                const auto& shader = shaders[index];
                auto& sourceName = names.source.emplace_back();
                sourceName.reserve(shader->entryPoint.size());
                std::ranges::transform(shader->entryPoint, std::back_inserter(sourceName), [](unsigned char character) {
                    return static_cast<wchar_t>(character);
                });
                names.exported.push_back(std::wstring{exportNamespace} + L"_shader_" + std::to_wstring(index));
            }
            appendHitShaderExports(names, groups);
            return names;
        }

        [[nodiscard]] std::size_t renamedHitShaderVariantCount(const ShaderExportNames& names)
        {
            return static_cast<std::size_t>(std::ranges::count_if(names.hitExports, isRenamedHitShaderVariant));
        }

        [[nodiscard]] const std::wstring& exportedShaderName(const ShaderExportNames& names,
                                                             const std::shared_ptr<ShaderModuleRecord>& shader)
        {
            return names.exported[shaderExportIndex(names, shader)];
        }

        [[nodiscard]] LPCWSTR optionalHitShaderImport(const ShaderExportNames& names,
                                                      const std::shared_ptr<ShaderModuleRecord>& shader,
                                                      const std::optional<ShaderRecordConstantRange>& constants)
        {
            if (shader == nullptr)
            {
                return nullptr;
            }
            const auto index = shaderExportIndex(names, shader);
            const auto found = std::ranges::find_if(names.hitExports, [&](const HitShaderExport& hitExport) {
                return hitExport.shader == index && hitExport.constants == constants;
            });
            return found->name.c_str();
        }

        [[nodiscard]] std::vector<std::wstring> buildGroupExportNames(std::span<const ResolvedRayTracingGroup> groups,
                                                                      const ShaderExportNames& shaderNames,
                                                                      std::wstring_view exportNamespace)
        {
            std::vector<std::wstring> groupNames;
            groupNames.reserve(groups.size());
            for (std::size_t index = 0; index < groups.size(); ++index)
            {
                if (groups[index].type == RayTracingShaderGroupType::general)
                {
                    groupNames.push_back(exportedShaderName(shaderNames, groups[index].general));
                    continue;
                }
                groupNames.push_back(std::wstring{exportNamespace} + L"_hit_group_" + std::to_wstring(index));
            }
            return groupNames;
        }

        [[nodiscard]] Status validateRepeatedExportConstants(std::span<const std::wstring> groupNames,
                                                             std::span<const ResolvedRayTracingGroup> groups)
        {
            for (std::size_t index = 0; index < groupNames.size(); ++index)
            {
                for (std::size_t previous = 0; previous < index; ++previous)
                {
                    if (groupNames[previous] == groupNames[index] &&
                        groups[previous].shaderRecordConstants != groups[index].shaderRecordConstants)
                    {
                        return std::unexpected(Error{
                            ErrorCode::validationError,
                            0,
                            "Repeated shader exports must use one consistent shader-record constant layout.",
                        });
                    }
                }
            }
            return {};
        }

        struct CollectionSubobjectStorage final
        {
            CollectionSubobjectStorage() = default;
            CollectionSubobjectStorage(const CollectionSubobjectStorage&) = delete;
            CollectionSubobjectStorage& operator=(const CollectionSubobjectStorage&) = delete;
            CollectionSubobjectStorage(CollectionSubobjectStorage&&) = delete;
            CollectionSubobjectStorage& operator=(CollectionSubobjectStorage&&) = delete;

            std::vector<D3D12_EXPORT_DESC> exports;
            std::vector<D3D12_DXIL_LIBRARY_DESC> libraries;
            std::vector<D3D12_HIT_GROUP_DESC> hitGroups;
            std::vector<D3D12_LOCAL_ROOT_SIGNATURE> localRootSignatures;
            std::vector<D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION> localRootSignatureAssociations;
            std::vector<D3D12_STATE_SUBOBJECT> subobjects;
        };

        void appendDxilLibrary(CollectionSubobjectStorage& storage,
                               std::size_t library,
                               const ShaderModuleRecord& shader,
                               const std::wstring& exportedName,
                               const std::wstring& sourceName)
        {
            storage.exports[library] = {
                .Name = exportedName.c_str(),
                .ExportToRename = sourceName.c_str(),
                .Flags = D3D12_EXPORT_FLAG_NONE,
            };
            storage.libraries[library] = {
                .DXILLibrary =
                    {
                        .pShaderBytecode = shader.bytecode->data(),
                        .BytecodeLength = shader.bytecode->size(),
                    },
                .NumExports = 1,
                .pExports = &storage.exports[library],
            };
            storage.subobjects.push_back({
                .Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY,
                .pDesc = &storage.libraries[library],
            });
        }

        void appendDxilLibraries(CollectionSubobjectStorage& storage, const ShaderExportNames& shaderNames)
        {
            const auto shaderCount = shaderNames.shaders.size();
            const auto libraryCount = shaderCount + renamedHitShaderVariantCount(shaderNames);
            storage.exports.resize(libraryCount);
            storage.libraries.resize(libraryCount);
            for (std::size_t index = 0; index < shaderCount; ++index)
            {
                appendDxilLibrary(storage,
                                  index,
                                  *shaderNames.shaders[index],
                                  shaderNames.exported[index],
                                  shaderNames.source[index]);
            }
            auto library = shaderCount;
            for (const auto& hitExport : shaderNames.hitExports)
            {
                if (!isRenamedHitShaderVariant(hitExport))
                {
                    continue;
                }
                appendDxilLibrary(storage,
                                  library++,
                                  *shaderNames.shaders[hitExport.shader],
                                  hitExport.name,
                                  shaderNames.source[hitExport.shader]);
            }
        }

        void appendHitGroups(CollectionSubobjectStorage& storage,
                             std::span<const ResolvedRayTracingGroup> groups,
                             std::span<const std::wstring> groupNames,
                             const ShaderExportNames& shaderNames)
        {
            storage.hitGroups.reserve(groups.size());
            for (std::size_t index = 0; index < groups.size(); ++index)
            {
                const auto& group = groups[index];
                if (group.type == RayTracingShaderGroupType::general)
                {
                    continue;
                }
                storage.hitGroups.push_back({
                    .HitGroupExport = groupNames[index].c_str(),
                    .Type = group.type == RayTracingShaderGroupType::trianglesHitGroup
                                ? D3D12_HIT_GROUP_TYPE_TRIANGLES
                                : D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE,
                    .AnyHitShaderImport =
                        optionalHitShaderImport(shaderNames, group.anyHit, group.shaderRecordConstants),
                    .ClosestHitShaderImport =
                        optionalHitShaderImport(shaderNames, group.closestHit, group.shaderRecordConstants),
                    .IntersectionShaderImport =
                        optionalHitShaderImport(shaderNames, group.intersection, group.shaderRecordConstants),
                });
                storage.subobjects.push_back({
                    .Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP,
                    .pDesc = &storage.hitGroups.back(),
                });
            }
        }

        struct LocalRootSignatureBuild final
        {
            ShaderRecordConstantRange constants = {};
            ComPtr<ID3D12RootSignature> rootSignature = {};
            std::vector<LPCWSTR> exports = {};
        };

        [[nodiscard]] std::vector<LocalRootSignatureBuild> groupExportsByShaderRecordConstants(
            std::span<const ResolvedRayTracingGroup> groups, std::span<const std::wstring> groupNames)
        {
            std::vector<LocalRootSignatureBuild> localRootSignatures;
            localRootSignatures.reserve(groups.size());
            for (std::size_t index = 0; index < groups.size(); ++index)
            {
                if (!groups[index].shaderRecordConstants.has_value())
                {
                    continue;
                }
                const auto& constants = *groups[index].shaderRecordConstants;
                auto found = std::ranges::find(localRootSignatures, constants, &LocalRootSignatureBuild::constants);
                if (found == localRootSignatures.end())
                {
                    found = localRootSignatures.insert(localRootSignatures.end(),
                                                       LocalRootSignatureBuild{.constants = constants});
                }
                const auto* const exportName = groupNames[index].c_str();
                const bool alreadyExported = std::ranges::any_of(found->exports, [&](LPCWSTR existing) {
                    return std::wstring_view{existing} == std::wstring_view{exportName};
                });
                if (!alreadyExported)
                {
                    found->exports.push_back(exportName);
                }
            }
            return localRootSignatures;
        }

        [[nodiscard]] Result<ComPtr<ID3DBlob>> serializeLocalRootSignature(const ShaderRecordConstantRange& constants)
        {
            const D3D12_ROOT_PARAMETER1 rootParameter{
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                .Constants =
                    {
                        .ShaderRegister = constants.shaderRegister,
                        .RegisterSpace = constants.registerSpace,
                        .Num32BitValues = static_cast<UINT>(constants.size / sizeof(std::uint32_t)),
                    },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
            };
            const D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootDesc{
                .Version = D3D_ROOT_SIGNATURE_VERSION_1_1,
                .Desc_1_1 =
                    {
                        .NumParameters = 1,
                        .pParameters = &rootParameter,
                        .NumStaticSamplers = 0,
                        .pStaticSamplers = nullptr,
                        .Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE,
                    },
            };
            ComPtr<ID3DBlob> serialized;
            ComPtr<ID3DBlob> diagnostics;
            const HRESULT serializeResult = D3D12SerializeVersionedRootSignature(&rootDesc, &serialized, &diagnostics);
            if (FAILED(serializeResult))
            {
                std::string message = "D3D12SerializeVersionedRootSignature(local ray tracing) failed.";
                if (diagnostics != nullptr && diagnostics->GetBufferPointer() != nullptr)
                {
                    message.append(" ");
                    message.append(static_cast<const char*>(diagnostics->GetBufferPointer()),
                                   diagnostics->GetBufferSize());
                }
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    static_cast<std::int64_t>(serializeResult),
                    std::move(message),
                });
            }
            return serialized;
        }

        [[nodiscard]] Result<ComPtr<ID3D12RootSignature>> createLocalRootSignature(
            ID3D12Device& device, const ShaderRecordConstantRange& constants)
        {
            auto serialized = serializeLocalRootSignature(constants);
            if (!serialized)
            {
                return std::unexpected(std::move(serialized.error()));
            }
            ComPtr<ID3D12RootSignature> rootSignature;
            const HRESULT rootResult = device.CreateRootSignature(
                0, (*serialized)->GetBufferPointer(), (*serialized)->GetBufferSize(), IID_PPV_ARGS(&rootSignature));
            if (FAILED(rootResult))
            {
                return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(rootResult)),
                                                   rootResult,
                                                   "ID3D12Device::CreateRootSignature(local ray tracing)"));
            }
            return rootSignature;
        }

        [[nodiscard]] Result<std::vector<LocalRootSignatureBuild>> createLocalRootSignatures(
            ID3D12Device& device,
            std::span<const ResolvedRayTracingGroup> groups,
            std::span<const std::wstring> groupNames,
            std::string_view label)
        {
            auto localRootSignatures = groupExportsByShaderRecordConstants(groups, groupNames);
            for (std::size_t index = 0; index < localRootSignatures.size(); ++index)
            {
                auto& local = localRootSignatures[index];
                auto rootSignature = createLocalRootSignature(device, local.constants);
                if (!rootSignature)
                {
                    return std::unexpected(std::move(rootSignature.error()));
                }
                local.rootSignature = std::move(*rootSignature);
                const auto localLabel = label.empty() ? std::string{"RHI ray-tracing local root signature"}
                                                      : std::string{label} + " local root " + std::to_string(index);
                setDebugName(*local.rootSignature.Get(), localLabel);
            }
            return localRootSignatures;
        }

        void retainLocalRootSignatures(RayTracingPipelineLibraryRecord& record,
                                       std::span<const LocalRootSignatureBuild> localRootSignatures)
        {
            record.localRootSignatures.reserve(localRootSignatures.size());
            for (const auto& local : localRootSignatures)
            {
                record.localRootSignatures.push_back(local.rootSignature);
            }
        }

        void appendLocalRootSignatures(CollectionSubobjectStorage& storage,
                                       std::span<LocalRootSignatureBuild> localRootSignatures)
        {
            storage.localRootSignatures.reserve(localRootSignatures.size());
            storage.localRootSignatureAssociations.reserve(localRootSignatures.size());
            for (auto& local : localRootSignatures)
            {
                storage.localRootSignatures.push_back({.pLocalRootSignature = local.rootSignature.Get()});
                storage.subobjects.push_back({
                    .Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE,
                    .pDesc = &storage.localRootSignatures.back(),
                });
                const auto* localSubobject = &storage.subobjects.back();
                storage.localRootSignatureAssociations.push_back({
                    .pSubobjectToAssociate = localSubobject,
                    .NumExports = static_cast<UINT>(local.exports.size()),
                    .pExports = local.exports.data(),
                });
                storage.subobjects.push_back({
                    .Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION,
                    .pDesc = &storage.localRootSignatureAssociations.back(),
                });
            }
        }

        [[nodiscard]] Result<ComPtr<ID3D12StateObject>> createStateObject(
            ID3D12Device5& rayTracingDevice,
            D3D12_STATE_OBJECT_TYPE type,
            std::span<const D3D12_STATE_SUBOBJECT> subobjects,
            std::string_view operation)
        {
            const D3D12_STATE_OBJECT_DESC nativeDesc{
                .Type = type,
                .NumSubobjects = static_cast<UINT>(subobjects.size()),
                .pSubobjects = subobjects.data(),
            };
            ComPtr<ID3D12StateObject> stateObject;
            const HRESULT result = rayTracingDevice.CreateStateObject(&nativeDesc, IID_PPV_ARGS(&stateObject));
            if (FAILED(result))
            {
                const auto code = result == E_INVALIDARG
                                      ? ErrorCode::validationError
                                      : classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result));
                return std::unexpected(nativeError(code, result, operation));
            }
            return stateObject;
        }

        [[nodiscard]] Status buildCollectionStateObject(ID3D12Device& device,
                                                        ID3D12Device5& rayTracingDevice,
                                                        RayTracingPipelineLibraryRecord& record,
                                                        std::span<const ResolvedRayTracingGroup> groups,
                                                        std::wstring_view exportNamespace)
        {
            const auto shaderNames = buildShaderExportNames(record.shaders, groups, exportNamespace);
            record.groupNames = buildGroupExportNames(groups, shaderNames, exportNamespace);
            auto exportValidation = validateRepeatedExportConstants(record.groupNames, groups);
            if (!exportValidation)
            {
                return std::unexpected(std::move(exportValidation.error()));
            }

            CollectionSubobjectStorage storage;
            storage.subobjects.reserve(record.shaders.size() + renamedHitShaderVariantCount(shaderNames) +
                                       groups.size() * 3 + 1);
            appendDxilLibraries(storage, shaderNames);
            appendHitGroups(storage, groups, record.groupNames, shaderNames);

            auto localRootSignatures = createLocalRootSignatures(device, groups, record.groupNames, record.label);
            if (!localRootSignatures)
            {
                return std::unexpected(std::move(localRootSignatures.error()));
            }
            retainLocalRootSignatures(record, *localRootSignatures);
            appendLocalRootSignatures(storage, *localRootSignatures);

            const D3D12_STATE_OBJECT_CONFIG stateObjectConfig{
                .Flags = D3D12_STATE_OBJECT_FLAG_ALLOW_LOCAL_DEPENDENCIES_ON_EXTERNAL_DEFINITIONS,
            };
            storage.subobjects.push_back({
                .Type = D3D12_STATE_SUBOBJECT_TYPE_STATE_OBJECT_CONFIG,
                .pDesc = &stateObjectConfig,
            });
            auto stateObject = createStateObject(rayTracingDevice,
                                                 D3D12_STATE_OBJECT_TYPE_COLLECTION,
                                                 storage.subobjects,
                                                 "ID3D12Device5::CreateStateObject(collection)");
            if (!stateObject)
            {
                return std::unexpected(std::move(stateObject.error()));
            }
            record.stateObject = std::move(*stateObject);
            setDebugName(*record.stateObject.Get(), record.label);
            return {};
        }

        struct RayTracingCollectionDesc final
        {
            std::shared_ptr<PipelineLayoutRecord> layout = {};
            std::span<const ResolvedRayTracingGroup> groups = {};
            std::uint32_t maxPayloadSize = 0;
            std::uint32_t maxAttributeSize = 0;
            std::wstring_view exportNamespace = {};
            std::string_view label = {};
        };

        [[nodiscard]] Result<std::shared_ptr<RayTracingPipelineLibraryRecord>> buildRayTracingCollection(
            ID3D12Device& device,
            ID3D12Device5& rayTracingDevice,
            const RayTracingCapabilities& limits,
            RayTracingCollectionDesc desc)
        {
            if (desc.groups.empty() || desc.groups.size() > std::numeric_limits<UINT>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A ray-tracing pipeline library requires at least one shader group.",
                });
            }

            auto record = std::make_shared<RayTracingPipelineLibraryRecord>();
            record->layout = std::move(desc.layout);
            record->maxPayloadSize = desc.maxPayloadSize;
            record->maxAttributeSize = desc.maxAttributeSize;
            record->label = desc.label;

            auto groupValidation = validateShaderGroups(desc.groups, limits, *record->layout);
            if (!groupValidation)
            {
                return std::unexpected(std::move(groupValidation.error()));
            }
            record->hasRayGeneration = std::ranges::any_of(desc.groups, isRayGenerationGroup);
            record->shaders = collectDistinctShaders(desc.groups);
            appendGroupRecordSizes(*record, desc.groups, limits);
            auto shaderValidation = validateCollectionShaders(record->shaders, *record->layout);
            if (!shaderValidation)
            {
                return std::unexpected(std::move(shaderValidation.error()));
            }

            auto stateObjectCreation =
                buildCollectionStateObject(device, rayTracingDevice, *record, desc.groups, desc.exportNamespace);
            if (!stateObjectCreation)
            {
                return std::unexpected(std::move(stateObjectCreation.error()));
            }
            return record;
        }

        [[nodiscard]] Status validateShaderRecordLimits(const RayTracingCapabilities& limits)
        {
            if (limits.shaderGroupHandleSize != D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES ||
                limits.shaderGroupHandleAlignment == 0 || limits.maxShaderGroupStride < limits.shaderGroupHandleSize)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "The D3D12 device exposed incoherent ray-tracing shader-record limits.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateRayTracingPipelineLibraryDesc(const RayTracingPipelineLibraryDesc& desc,
                                                                   const RayTracingCapabilities& limits)
        {
            auto limitValidation = validateShaderRecordLimits(limits);
            if (!limitValidation)
            {
                return std::unexpected(std::move(limitValidation.error()));
            }
            const bool hasGroups = !desc.groups.empty();
            const bool groupCountFitsNativeRange = desc.groups.size() <= std::numeric_limits<UINT>::max();
            const bool hasScalarAlignedShaderData =
                desc.maxPayloadSize % sizeof(std::uint32_t) == 0 && desc.maxAttributeSize % sizeof(std::uint32_t) == 0;
            const bool hasSupportedAttributeSize = desc.maxAttributeSize <= limits.maxAttributeSize;
            if (!hasGroups || !groupCountFitsNativeRange || !hasScalarAlignedShaderData || !hasSupportedAttributeSize)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A ray-tracing pipeline library requires groups and supported scalar-aligned interface sizes.",
                });
            }
            return {};
        }

        [[nodiscard]] std::shared_ptr<ShaderModuleRecord> findShaderModule(const ShaderModuleTable& shaderModules,
                                                                           ShaderModuleHandle handle)
        {
            if (!handle.isValid())
            {
                return {};
            }
            const auto* found = shaderModules.find(handle);
            return found != nullptr ? *found : std::shared_ptr<ShaderModuleRecord>{};
        }

        [[nodiscard]] std::optional<ResolvedRayTracingGroup> resolveShaderGroup(const ShaderModuleTable& shaderModules,
                                                                                const RayTracingShaderGroupDesc& group)
        {
            auto resolved = ResolvedRayTracingGroup{
                .type = group.type,
                .general = findShaderModule(shaderModules, group.generalShader),
                .closestHit = findShaderModule(shaderModules, group.closestHitShader),
                .anyHit = findShaderModule(shaderModules, group.anyHitShader),
                .intersection = findShaderModule(shaderModules, group.intersectionShader),
                .shaderRecordConstants = group.shaderRecordConstants,
            };
            const auto isResolved = [](ShaderModuleHandle handle, const std::shared_ptr<ShaderModuleRecord>& shader) {
                return !handle.isValid() || shader != nullptr;
            };
            if (!isResolved(group.generalShader, resolved.general) ||
                !isResolved(group.closestHitShader, resolved.closestHit) ||
                !isResolved(group.anyHitShader, resolved.anyHit) ||
                !isResolved(group.intersectionShader, resolved.intersection))
            {
                return std::nullopt;
            }
            return resolved;
        }

        [[nodiscard]] Result<std::vector<ResolvedRayTracingGroup>> resolveShaderGroups(
            const ShaderModuleTable& shaderModules,
            std::span<const RayTracingShaderGroupDesc> groups,
            std::string_view staleModuleMessage)
        {
            std::vector<ResolvedRayTracingGroup> resolvedGroups;
            resolvedGroups.reserve(groups.size());
            for (const auto& group : groups)
            {
                auto resolved = resolveShaderGroup(shaderModules, group);
                if (!resolved)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        std::string{staleModuleMessage},
                    });
                }
                resolvedGroups.push_back(std::move(*resolved));
            }
            return resolvedGroups;
        }

        [[nodiscard]] Result<std::wstring> allocateCollectionExportNamespace(std::uint64_t& nextExportNamespace)
        {
            if (nextExportNamespace == std::numeric_limits<std::uint64_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "The ray-tracing export namespace is exhausted.",
                });
            }
            return L"__rhi_collection_" + std::to_wstring(nextExportNamespace++);
        }

    } // namespace

    Result<RayTracingPipelineLibraryHandle> PipelineManager::createRayTracingPipelineLibrary(
        const RayTracingPipelineLibraryDesc& desc)
    {
        if (!impl_->rayTracing || impl_->rayTracingDevice == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Ray-tracing pipeline-library creation is unavailable on this device.",
            });
        }
        const auto& limits = impl_->rayTracingCapabilities;
        auto descValidation = validateRayTracingPipelineLibraryDesc(desc, limits);
        if (!descValidation)
        {
            return std::unexpected(std::move(descValidation.error()));
        }

        std::shared_ptr<PipelineLayoutRecord> layout;
        std::vector<ResolvedRayTracingGroup> groups;
        std::wstring exportNamespace;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* foundLayout = impl_->pipelineLayouts.find(desc.layout);
            if (foundLayout == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A ray-tracing pipeline library references an invalid or stale layout.",
                });
            }
            layout = *foundLayout;
            auto resolvedGroups =
                resolveShaderGroups(impl_->shaderModules,
                                    desc.groups,
                                    "A ray-tracing pipeline-library group references an invalid or stale module.");
            if (!resolvedGroups)
            {
                return std::unexpected(std::move(resolvedGroups.error()));
            }
            groups = std::move(*resolvedGroups);
            auto allocatedNamespace = allocateCollectionExportNamespace(impl_->nextRayTracingExportNamespace);
            if (!allocatedNamespace)
            {
                return std::unexpected(std::move(allocatedNamespace.error()));
            }
            exportNamespace = std::move(*allocatedNamespace);
        }

        auto record = buildRayTracingCollection(*impl_->device,
                                                *impl_->rayTracingDevice.Get(),
                                                limits,
                                                RayTracingCollectionDesc{
                                                    .layout = std::move(layout),
                                                    .groups = groups,
                                                    .maxPayloadSize = desc.maxPayloadSize,
                                                    .maxAttributeSize = desc.maxAttributeSize,
                                                    .exportNamespace = exportNamespace,
                                                    .label = desc.label,
                                                });
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }

        const std::scoped_lock lock(impl_->mutex);
        return impl_->rayTracingPipelineLibraries.emplace(std::move(*record));
    }

    Status PipelineManager::destroyRayTracingPipelineLibrary(RayTracingPipelineLibraryHandle library)
    {
        const std::scoped_lock lock(impl_->mutex);
        if (!impl_->rayTracingPipelineLibraries.erase(library))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Cannot destroy an invalid or stale ray-tracing pipeline library.",
            });
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] Status validateRayTracingPipelineDesc(const RayTracingPipelineDesc& desc,
                                                            const RayTracingCapabilities& limits)
        {
            auto limitValidation = validateShaderRecordLimits(limits);
            if (!limitValidation)
            {
                return std::unexpected(std::move(limitValidation.error()));
            }
            const bool hasGroups = !desc.libraries.empty() || !desc.groups.empty();
            const bool countsFitNativeRange = desc.libraries.size() <= std::numeric_limits<UINT>::max() &&
                                              desc.groups.size() <= std::numeric_limits<UINT>::max();
            const bool hasScalarAlignedShaderData =
                desc.maxPayloadSize % sizeof(std::uint32_t) == 0 && desc.maxAttributeSize % sizeof(std::uint32_t) == 0;
            const bool hasSupportedAttributeSize = desc.maxAttributeSize <= limits.maxAttributeSize;
            const bool hasSupportedRecursionDepth =
                desc.maxRecursionDepth != 0 && desc.maxRecursionDepth <= limits.maxRecursionDepth;
            if (!hasGroups || !countsFitNativeRange || !hasScalarAlignedShaderData || !hasSupportedAttributeSize ||
                !hasSupportedRecursionDepth)
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

        struct CollectionsToLink final
        {
            std::vector<std::shared_ptr<RayTracingPipelineLibraryRecord>> collections;
            std::size_t groupCount = 0;
            bool hasRayGeneration = false;
        };

        [[nodiscard]] Status addCollectionToLink(CollectionsToLink& collectionsToLink,
                                                 std::shared_ptr<RayTracingPipelineLibraryRecord> collection)
        {
            if (collection->groupNames.size() > std::numeric_limits<UINT>::max() - collectionsToLink.groupCount)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The linked ray-tracing shader-group count exceeds the portable limit.",
                });
            }
            collectionsToLink.groupCount += collection->groupNames.size();
            collectionsToLink.hasRayGeneration = collectionsToLink.hasRayGeneration || collection->hasRayGeneration;
            collectionsToLink.collections.push_back(std::move(collection));
            return {};
        }

        [[nodiscard]] Status addLibrariesToLink(CollectionsToLink& collectionsToLink,
                                                const RayTracingPipelineLibraryTable& libraries,
                                                const RayTracingPipelineDesc& desc,
                                                const std::shared_ptr<PipelineLayoutRecord>& layout)
        {
            for (const auto library : desc.libraries)
            {
                const auto* found = libraries.find(library);
                if (found == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A ray-tracing pipeline references an invalid or stale library.",
                    });
                }
                if (std::ranges::contains(collectionsToLink.collections, *found))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A ray-tracing pipeline cannot link the same library more than once.",
                    });
                }
                if ((*found)->layout != layout || (*found)->maxPayloadSize != desc.maxPayloadSize ||
                    (*found)->maxAttributeSize != desc.maxAttributeSize)
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "Every ray-tracing pipeline library must match the final layout and shader interface exactly.",
                    });
                }
                auto addition = addCollectionToLink(collectionsToLink, *found);
                if (!addition)
                {
                    return std::unexpected(std::move(addition.error()));
                }
            }
            return {};
        }

        [[nodiscard]] std::shared_ptr<RayTracingPipelineRecord> makeRayTracingPipelineRecord(
            std::shared_ptr<PipelineLayoutRecord> layout,
            const CollectionsToLink& collectionsToLink,
            std::string_view label)
        {
            auto record = std::make_shared<RayTracingPipelineRecord>();
            record->layout = std::move(layout);
            record->collections = collectionsToLink.collections;
            record->label = label;
            record->groupLocalConstantSizes.reserve(collectionsToLink.groupCount);
            record->groupRecordSizes.reserve(collectionsToLink.groupCount);
            for (const auto& collection : collectionsToLink.collections)
            {
                record->groupLocalConstantSizes.insert(record->groupLocalConstantSizes.end(),
                                                       collection->groupLocalConstantSizes.begin(),
                                                       collection->groupLocalConstantSizes.end());
                record->groupRecordSizes.insert(record->groupRecordSizes.end(),
                                                collection->groupRecordSizes.begin(),
                                                collection->groupRecordSizes.end());
            }
            return record;
        }

        [[nodiscard]] std::vector<LPCWSTR> linkedGroupNames(const CollectionsToLink& collectionsToLink)
        {
            std::vector<LPCWSTR> groupNames;
            groupNames.reserve(collectionsToLink.groupCount);
            for (const auto& collection : collectionsToLink.collections)
            {
                std::ranges::transform(collection->groupNames,
                                       std::back_inserter(groupNames),
                                       [](const std::wstring& name) { return name.c_str(); });
            }
            return groupNames;
        }

        [[nodiscard]] Result<ComPtr<ID3D12StateObject>> createLinkedStateObject(ID3D12Device5& rayTracingDevice,
                                                                                const RayTracingPipelineRecord& record,
                                                                                const RayTracingPipelineDesc& desc)
        {
            std::vector<D3D12_EXISTING_COLLECTION_DESC> existingCollections;
            std::vector<D3D12_STATE_SUBOBJECT> subobjects;
            existingCollections.reserve(record.collections.size());
            subobjects.reserve(record.collections.size() + 4);
            for (const auto& collection : record.collections)
            {
                existingCollections.push_back({
                    .pExistingCollection = collection->stateObject.Get(),
                    .NumExports = 0,
                    .pExports = nullptr,
                });
                subobjects.push_back({
                    .Type = D3D12_STATE_SUBOBJECT_TYPE_EXISTING_COLLECTION,
                    .pDesc = &existingCollections.back(),
                });
            }

            const D3D12_STATE_OBJECT_CONFIG stateObjectConfig{
                .Flags = D3D12_STATE_OBJECT_FLAG_ALLOW_EXTERNAL_DEPENDENCIES_ON_LOCAL_DEFINITIONS,
            };
            const D3D12_GLOBAL_ROOT_SIGNATURE globalRootSignature{
                .pGlobalRootSignature = record.layout->rootSignature.Get(),
            };
            const D3D12_RAYTRACING_SHADER_CONFIG shaderConfig{
                .MaxPayloadSizeInBytes = desc.maxPayloadSize,
                .MaxAttributeSizeInBytes = desc.maxAttributeSize,
            };
            const D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig{
                .MaxTraceRecursionDepth = desc.maxRecursionDepth,
            };
            subobjects.push_back({
                .Type = D3D12_STATE_SUBOBJECT_TYPE_STATE_OBJECT_CONFIG,
                .pDesc = &stateObjectConfig,
            });
            subobjects.push_back({
                .Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE,
                .pDesc = &globalRootSignature,
            });
            subobjects.push_back({
                .Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG,
                .pDesc = &shaderConfig,
            });
            subobjects.push_back({
                .Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
                .pDesc = &pipelineConfig,
            });
            return createStateObject(rayTracingDevice,
                                     D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
                                     subobjects,
                                     "ID3D12Device5::CreateStateObject(ray tracing)");
        }

        [[nodiscard]] Status copyShaderGroupHandles(RayTracingPipelineRecord& record,
                                                    std::span<const LPCWSTR> groupNames)
        {
            record.groupHandles.reserve(groupNames.size());
            for (const auto* const name : groupNames)
            {
                const void* identifier = record.properties->GetShaderIdentifier(name);
                if (identifier == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::nativeFailure,
                        0,
                        "The linked D3D12 state object did not expose a requested shader-group identifier.",
                    });
                }
                auto& handle = record.groupHandles.emplace_back();
                std::memcpy(handle.data(), identifier, handle.size());
            }
            return {};
        }

        [[nodiscard]] Result<std::shared_ptr<RayTracingPipelineRecord>> linkRayTracingPipeline(
            ID3D12Device5& rayTracingDevice,
            std::shared_ptr<PipelineLayoutRecord> layout,
            const CollectionsToLink& collectionsToLink,
            const RayTracingPipelineDesc& desc)
        {
            if (!collectionsToLink.hasRayGeneration)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A linked ray-tracing pipeline requires at least one ray-generation group.",
                });
            }

            auto record = makeRayTracingPipelineRecord(std::move(layout), collectionsToLink, desc.label);
            auto stateObject = createLinkedStateObject(rayTracingDevice, *record, desc);
            if (!stateObject)
            {
                return std::unexpected(std::move(stateObject.error()));
            }
            record->stateObject = std::move(*stateObject);
            setDebugName(*record->stateObject.Get(), desc.label);
            const HRESULT propertiesResult = record->stateObject.As(&record->properties);
            if (FAILED(propertiesResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, propertiesResult, "ID3D12StateObjectProperties query"));
            }
            const auto groupNames = linkedGroupNames(collectionsToLink);
            auto handleCopy = copyShaderGroupHandles(*record, groupNames);
            if (!handleCopy)
            {
                return std::unexpected(std::move(handleCopy.error()));
            }
            return record;
        }

    } // namespace

    Result<PipelineHandle> PipelineManager::createRayTracingPipeline(const RayTracingPipelineDesc& desc)
    {
        if (!impl_->rayTracing || impl_->rayTracingDevice == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Ray-tracing pipeline creation is unavailable on this device.",
            });
        }
        const auto& limits = impl_->rayTracingCapabilities;
        auto descValidation = validateRayTracingPipelineDesc(desc, limits);
        if (!descValidation)
        {
            return std::unexpected(std::move(descValidation.error()));
        }

        std::shared_ptr<PipelineLayoutRecord> layout;
        CollectionsToLink collectionsToLink;
        collectionsToLink.collections.reserve(desc.libraries.size() + (desc.groups.empty() ? 0U : 1U));
        std::vector<ResolvedRayTracingGroup> localGroups;
        std::wstring localExportNamespace;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* foundLayout = impl_->pipelineLayouts.find(desc.layout);
            if (foundLayout == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A ray-tracing pipeline references an invalid or stale layout.",
                });
            }
            layout = *foundLayout;
            auto libraryAddition =
                addLibrariesToLink(collectionsToLink, impl_->rayTracingPipelineLibraries, desc, layout);
            if (!libraryAddition)
            {
                return std::unexpected(std::move(libraryAddition.error()));
            }
            auto resolvedGroups =
                resolveShaderGroups(impl_->shaderModules,
                                    desc.groups,
                                    "A local ray-tracing shader group references an invalid or stale module.");
            if (!resolvedGroups)
            {
                return std::unexpected(std::move(resolvedGroups.error()));
            }
            localGroups = std::move(*resolvedGroups);
            if (!localGroups.empty())
            {
                auto allocatedNamespace = allocateCollectionExportNamespace(impl_->nextRayTracingExportNamespace);
                if (!allocatedNamespace)
                {
                    return std::unexpected(std::move(allocatedNamespace.error()));
                }
                localExportNamespace = std::move(*allocatedNamespace);
            }
        }

        if (!localGroups.empty())
        {
            auto localCollection = buildRayTracingCollection(*impl_->device,
                                                             *impl_->rayTracingDevice.Get(),
                                                             limits,
                                                             RayTracingCollectionDesc{
                                                                 .layout = layout,
                                                                 .groups = localGroups,
                                                                 .maxPayloadSize = desc.maxPayloadSize,
                                                                 .maxAttributeSize = desc.maxAttributeSize,
                                                                 .exportNamespace = localExportNamespace,
                                                                 .label = desc.label,
                                                             });
            if (!localCollection)
            {
                return std::unexpected(std::move(localCollection.error()));
            }
            auto localAddition = addCollectionToLink(collectionsToLink, std::move(*localCollection));
            if (!localAddition)
            {
                return std::unexpected(std::move(localAddition.error()));
            }
        }

        auto record =
            linkRayTracingPipeline(*impl_->rayTracingDevice.Get(), std::move(layout), collectionsToLink, desc);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }

        const std::scoped_lock lock(impl_->mutex);
        return impl_->pipelines.emplace(PipelineRecord{std::move(*record)});
    }

    Status PipelineManager::getRayTracingShaderGroupHandles(PipelineHandle pipeline,
                                                            std::uint32_t firstGroup,
                                                            std::uint32_t groupCount,
                                                            std::span<std::byte> destination)
    {
        std::shared_ptr<RayTracingPipelineRecord> record;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* found = impl_->pipelines.find(pipeline);
            const auto* rayTracing =
                found != nullptr ? std::get_if<std::shared_ptr<RayTracingPipelineRecord>>(found) : nullptr;
            if (rayTracing == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader-group handle retrieval requires a live ray-tracing pipeline.",
                });
            }
            record = *rayTracing;
        }
        const auto handleSize = static_cast<std::size_t>(impl_->rayTracingCapabilities.shaderGroupHandleSize);
        if (groupCount == 0 || firstGroup >= record->groupHandles.size() ||
            groupCount > record->groupHandles.size() - firstGroup ||
            groupCount > std::numeric_limits<std::size_t>::max() / handleSize ||
            destination.size() != static_cast<std::size_t>(groupCount) * handleSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shader-group handle retrieval requires a non-empty in-range group span and exact output size.",
            });
        }
        for (std::size_t index = 0; index < groupCount; ++index)
        {
            const auto& handle = record->groupHandles[firstGroup + index];
            std::memcpy(destination.data() + index * handleSize, handle.data(), handleSize);
        }
        return {};
    }

    Result<std::uint32_t> PipelineManager::rayTracingShaderGroupRecordSize(PipelineHandle pipeline,
                                                                           std::uint32_t groupIndex)
    {
        const std::scoped_lock lock(impl_->mutex);
        const auto* found = impl_->pipelines.find(pipeline);
        const auto* rayTracing =
            found != nullptr ? std::get_if<std::shared_ptr<RayTracingPipelineRecord>>(found) : nullptr;
        if (rayTracing == nullptr || groupIndex >= (*rayTracing)->groupRecordSizes.size())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shader-group record-size retrieval requires a live ray-tracing pipeline and in-range group index.",
            });
        }
        return (*rayTracing)->groupRecordSizes[groupIndex];
    }

    Status PipelineManager::writeRayTracingShaderGroupRecord(PipelineHandle pipeline,
                                                             std::uint32_t groupIndex,
                                                             std::span<const std::byte> localConstants,
                                                             std::span<std::byte> destination)
    {
        std::shared_ptr<RayTracingPipelineRecord> record;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* found = impl_->pipelines.find(pipeline);
            const auto* rayTracing =
                found != nullptr ? std::get_if<std::shared_ptr<RayTracingPipelineRecord>>(found) : nullptr;
            if (rayTracing == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader-group record encoding requires a live ray-tracing pipeline.",
                });
            }
            record = *rayTracing;
        }
        if (groupIndex >= record->groupHandles.size() || groupIndex >= record->groupLocalConstantSizes.size() ||
            groupIndex >= record->groupRecordSizes.size() ||
            localConstants.size() != record->groupLocalConstantSizes[groupIndex] ||
            destination.size() != record->groupRecordSizes[groupIndex])
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shader-group record encoding requires an in-range group and exact local-data and output sizes.",
            });
        }

        std::ranges::fill(destination, std::byte{0});
        const auto handleSize = static_cast<std::size_t>(impl_->rayTracingCapabilities.shaderGroupHandleSize);
        std::memcpy(destination.data(), record->groupHandles[groupIndex].data(), handleSize);
        if (!localConstants.empty())
        {
            std::memcpy(destination.data() + handleSize, localConstants.data(), localConstants.size());
        }
        return {};
    }

    Status PipelineManager::bindRayTracingPipeline(QueueManager& queues,
                                                   BindingManager& bindings,
                                                   detail::BackendCommandBuffer& commandBuffer,
                                                   PipelineHandle pipeline)
    {
        if (commandBuffer.queueType() == QueueType::copy || commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A ray-tracing pipeline requires a graphics or compute command buffer outside a rendering scope.",
            });
        }

        std::shared_ptr<RayTracingPipelineRecord> record;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* found = impl_->pipelines.find(pipeline);
            const auto* rayTracing =
                found != nullptr ? std::get_if<std::shared_ptr<RayTracingPipelineRecord>>(found) : nullptr;
            if (rayTracing == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline is not a live ray-tracing pipeline.",
                });
            }
            record = *rayTracing;
        }

        auto commandList = queues.commandList4(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, record);
        if (!retained)
        {
            return retained;
        }
        const auto heaps = bindings.nativeDescriptorHeaps();
        ID3D12DescriptorHeap* const nativeHeaps[]{heaps.resources, heaps.samplers};
        (*commandList)->SetDescriptorHeaps(static_cast<UINT>(std::size(nativeHeaps)), nativeHeaps);
        (*commandList)->SetComputeRootSignature(record->layout->rootSignature.Get());
        (*commandList)->SetPipelineState1(record->stateObject.Get());

        commandBuffer.computeState() = {};
        commandBuffer.rayTracingState() = {};
        auto& state = commandBuffer.rayTracingState();
        state.pipelineOwner = std::move(record);
        state.boundBindingSets.clear();
        const auto activePipeline = std::static_pointer_cast<RayTracingPipelineRecord>(state.pipelineOwner);
        state.boundBindingSets.reserve(activePipeline->layout->slots.size());
        for (const auto& slot : activePipeline->layout->slots)
        {
            state.boundBindingSets.push_back(!slot.requiresBindingSet);
        }
        return {};
    }

    Status PipelineManager::bindRayTracingBindingSet(QueueManager& queues,
                                                     BindingManager& bindings,
                                                     ResourceManager& resources,
                                                     detail::BackendCommandBuffer& commandBuffer,
                                                     std::uint32_t setIndex,
                                                     BindingSetHandle bindingSet,
                                                     std::span<const std::uint32_t> dynamicOffsets)
    {
        const std::array bindingSets{bindingSet};
        return bindRayTracingBindingSets(
            queues, bindings, resources, commandBuffer, setIndex, bindingSets, dynamicOffsets);
    }

    Status PipelineManager::bindRayTracingBindingSets(QueueManager& queues,
                                                      BindingManager& bindings,
                                                      ResourceManager& resources,
                                                      detail::BackendCommandBuffer& commandBuffer,
                                                      std::uint32_t firstSet,
                                                      std::span<const BindingSetHandle> bindingSets,
                                                      std::span<const std::uint32_t> dynamicOffsets)
    {
        auto& state = commandBuffer.rayTracingState();
        if (state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A ray-tracing pipeline must be bound before binding sets.",
            });
        }
        const auto pipeline = std::static_pointer_cast<RayTracingPipelineRecord>(state.pipelineOwner);
        auto prepared = prepareBindingSetBatch(
            queues, commandBuffer, bindings, resources, pipeline->layout->slots, firstSet, bindingSets, dynamicOffsets);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        auto commandList = queues.commandList4(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = retainBindingSetBatchDependencies(queues, commandBuffer, *prepared);
        if (!retained)
        {
            return retained;
        }
        const auto activeSets = (*prepared)->activeSets();
        for (std::size_t index = 0; index < activeSets.size(); ++index)
        {
            const auto& slot = pipeline->layout->slots[firstSet + index];
            applyComputeRootBindings(**commandList, slot, activeSets[index]);
            state.boundBindingSets[firstSet + index] = true;
        }
        return {};
    }

    Status PipelineManager::setRayTracingPushConstants(QueueManager& queues,
                                                       detail::BackendCommandBuffer& commandBuffer,
                                                       std::uint32_t offset,
                                                       std::span<const std::byte> data)
    {
        auto& state = commandBuffer.rayTracingState();
        if (state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A ray-tracing pipeline must be bound before setting push constants.",
            });
        }
        if (data.empty() || offset % sizeof(std::uint32_t) != 0 || data.size() % sizeof(std::uint32_t) != 0 ||
            data.size() > maxPushConstantBytes || offset > maxPushConstantBytes - data.size())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Ray-tracing push constants require a non-empty, 4-byte aligned range within 128 bytes.",
            });
        }
        const auto pipeline = std::static_pointer_cast<RayTracingPipelineRecord>(state.pipelineOwner);
        const auto end = static_cast<std::uint64_t>(offset) + data.size();
        const bool declared = std::ranges::any_of(pipeline->layout->pushConstantRanges, [&](const auto& range) {
            return hasAny(range.visibility, ShaderStage::allRayTracing) && range.offset <= offset &&
                   end <= static_cast<std::uint64_t>(range.offset) + range.size;
        });
        if (!declared || pipeline->layout->pushConstantRootParameter == invalidRootParameter)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The push-constant write is not contained in a ray-tracing-visible pipeline layout range.",
            });
        }
        auto commandList = queues.commandList4(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)
            ->SetComputeRoot32BitConstants(pipeline->layout->pushConstantRootParameter,
                                           static_cast<UINT>(data.size() / sizeof(std::uint32_t)),
                                           data.data(),
                                           offset / sizeof(std::uint32_t));
        return {};
    }

    namespace
    {

        struct PreparedRayTracingShaderTableRegion final
        {
            BufferBindingSource source;
            D3D12_GPU_VIRTUAL_ADDRESS address = 0;
            std::uint64_t size = 0;
            std::uint64_t stride = 0;
        };

        struct PreparedRayTracingShaderTables final
        {
            PreparedRayTracingShaderTableRegion rayGeneration;
            std::optional<PreparedRayTracingShaderTableRegion> miss;
            std::optional<PreparedRayTracingShaderTableRegion> hitGroup;
            std::optional<PreparedRayTracingShaderTableRegion> callable;
        };

        struct RayTracingShaderTableRetention final
        {
            std::vector<std::shared_ptr<void>> owners;
            std::vector<std::shared_ptr<ResidencyRecord>> residencies;
        };

        [[nodiscard]] Status validateRayTracingDispatchState(detail::BackendCommandBuffer& commandBuffer)
        {
            const auto& state = commandBuffer.rayTracingState();
            if (state.pipelineOwner == nullptr || commandBuffer.graphicsState().renderingOwner != nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Ray dispatch requires a bound ray-tracing pipeline outside a rendering scope.",
                });
            }
            if (!state.boundBindingSets.all())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Every required ray-tracing binding set must be bound before dispatch.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<PreparedRayTracingShaderTables> prepareRayTracingShaderTables(
            ResourceManager& resources,
            const RayTracingCapabilities& limits,
            const RayTracingShaderTableRegion& rayGenerationRegion,
            const RayTracingShaderTableRegion& missRegion,
            const RayTracingShaderTableRegion& hitGroupRegion,
            const RayTracingShaderTableRegion& callableRegion)
        {
            const auto prepareRegion = [&](const RayTracingShaderTableRegion& region, bool required, bool rayGeneration)
                -> Result<std::optional<PreparedRayTracingShaderTableRegion>> {
                if (!region.buffer.isValid())
                {
                    if (!required && region.offset == 0 && region.size == 0 && region.stride == 0)
                    {
                        return std::optional<PreparedRayTracingShaderTableRegion>{};
                    }
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A shader-table region has an invalid buffer or non-zero metadata for an omitted table.",
                    });
                }
                auto source = resources.bufferBindingSource(region.buffer);
                if (!source)
                {
                    return std::unexpected(std::move(source.error()));
                }
                const bool rangeValid =
                    region.offset <= source->desc.size && region.size <= source->desc.size - region.offset;
                const bool strideValid = region.stride >= limits.shaderGroupHandleSize &&
                                         region.stride <= limits.maxShaderGroupStride &&
                                         region.stride % limits.shaderGroupHandleAlignment == 0;
                if (!hasAny(source->desc.usage, BufferUsage::shaderBindingTable) || region.size == 0 || !rangeValid ||
                    region.offset % limits.shaderGroupBaseAlignment != 0 || !strideValid ||
                    (rayGeneration ? region.size != region.stride : region.size % region.stride != 0))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A shader-table region violates its usage, range, base-alignment, record-stride, or table-size "
                        "contract.",
                    });
                }
                const auto address = source->resource->GetGPUVirtualAddress() + region.offset;
                return std::optional<PreparedRayTracingShaderTableRegion>{PreparedRayTracingShaderTableRegion{
                    .source = std::move(*source),
                    .address = address,
                    .size = region.size,
                    .stride = region.stride,
                }};
            };

            auto rayGeneration = prepareRegion(rayGenerationRegion, true, true);
            auto miss = prepareRegion(missRegion, false, false);
            auto hitGroup = prepareRegion(hitGroupRegion, false, false);
            auto callable = prepareRegion(callableRegion, false, false);
            if (!rayGeneration)
            {
                return std::unexpected(std::move(rayGeneration.error()));
            }
            if (!miss)
            {
                return std::unexpected(std::move(miss.error()));
            }
            if (!hitGroup)
            {
                return std::unexpected(std::move(hitGroup.error()));
            }
            if (!callable)
            {
                return std::unexpected(std::move(callable.error()));
            }
            if (!rayGeneration->has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "Validated ray dispatch lost its required ray-generation table.",
                });
            }
            return PreparedRayTracingShaderTables{
                .rayGeneration = std::move(rayGeneration->value()),
                .miss = std::move(*miss),
                .hitGroup = std::move(*hitGroup),
                .callable = std::move(*callable),
            };
        }

        [[nodiscard]] D3D12_DISPATCH_RAYS_DESC nativeRayTracingDispatchDesc(
            const PreparedRayTracingShaderTables& tables,
            std::uint32_t width,
            std::uint32_t height,
            std::uint32_t depth)
        {
            const auto nativeTable = [](const std::optional<PreparedRayTracingShaderTableRegion>& region) {
                return region.has_value()
                   ? D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE{
                         .StartAddress = region->address,
                         .SizeInBytes = region->size,
                         .StrideInBytes = region->stride,
                     }
                   : D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE{};
            };
            return D3D12_DISPATCH_RAYS_DESC{
                .RayGenerationShaderRecord =
                    {
                        .StartAddress = tables.rayGeneration.address,
                        .SizeInBytes = tables.rayGeneration.size,
                    },
                .MissShaderTable = nativeTable(tables.miss),
                .HitGroupTable = nativeTable(tables.hitGroup),
                .CallableShaderTable = nativeTable(tables.callable),
                .Width = width,
                .Height = height,
                .Depth = depth,
            };
        }

        [[nodiscard]] std::pair<std::array<const BufferBindingSource*, 4>, std::size_t> rayTracingShaderTableSources(
            const PreparedRayTracingShaderTables& tables)
        {
            std::array<const BufferBindingSource*, 4> sources{};
            std::size_t count = 0;
            sources[count++] = &tables.rayGeneration.source;
            for (const auto* region : {&tables.miss, &tables.hitGroup, &tables.callable})
            {
                if (region->has_value())
                {
                    sources[count++] = &region->value().source;
                }
            }
            return {sources, count};
        }

    } // namespace

    Status PipelineManager::dispatchRays(QueueManager& queues,
                                         ResourceManager& resources,
                                         detail::BackendCommandBuffer& commandBuffer,
                                         const RayTracingDispatchDesc& desc)
    {
        auto state = validateRayTracingDispatchState(commandBuffer);
        if (!state)
        {
            return state;
        }
        const auto invocationCount = static_cast<std::uint64_t>(desc.width) * desc.height * desc.depth;
        if (desc.width == 0 || desc.height == 0 || desc.depth == 0 ||
            invocationCount > impl_->rayTracingCapabilities.maxDispatchInvocations)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Ray-dispatch dimensions must be non-zero and fit the device's aggregate invocation limit.",
            });
        }
        auto tables = prepareRayTracingShaderTables(
            resources, impl_->rayTracingCapabilities, desc.rayGeneration, desc.miss, desc.hitGroup, desc.callable);
        if (!tables)
        {
            return std::unexpected(std::move(tables.error()));
        }
        auto commandList = queues.commandList4(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }

        const auto [sources, sourceCount] = rayTracingShaderTableSources(*tables);
        auto retention = std::make_shared<RayTracingShaderTableRetention>();
        retention->owners.reserve(sourceCount);
        retention->residencies.reserve(sourceCount);
        for (const auto* source : std::span{sources}.first(sourceCount))
        {
            retention->owners.push_back(source->owner);
            if (source->residency != nullptr)
            {
                retention->residencies.push_back(source->residency);
            }
        }
        auto retained = queues.retainResource(commandBuffer, retention, retention->residencies);
        if (!retained)
        {
            return retained;
        }

        const auto nativeDesc = nativeRayTracingDispatchDesc(*tables, desc.width, desc.height, desc.depth);
        (*commandList)->DispatchRays(&nativeDesc);
        return {};
    }

    Status PipelineManager::dispatchRaysIndirect(QueueManager& queues,
                                                 ResourceManager& resources,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const RayTracingIndirectDispatchDesc& desc)
    {
        auto state = validateRayTracingDispatchState(commandBuffer);
        if (!state)
        {
            return state;
        }
        auto signature = impl_->indirectRayTracingDispatchSignature();
        if (!signature)
        {
            return std::unexpected(std::move(signature.error()));
        }
        auto tables = prepareRayTracingShaderTables(
            resources, impl_->rayTracingCapabilities, desc.rayGeneration, desc.miss, desc.hitGroup, desc.callable);
        if (!tables)
        {
            return std::unexpected(std::move(tables.error()));
        }
        const auto nativeTemplate = nativeRayTracingDispatchDesc(*tables, 0, 0, 0);
        const auto [sources, sourceCount] = rayTracingShaderTableSources(*tables);
        auto nativeArguments =
            resources.recordRayTracingIndirectDispatchArguments(queues,
                                                                commandBuffer,
                                                                nativeTemplate,
                                                                desc.argumentBuffer,
                                                                desc.argumentOffset,
                                                                std::span{sources}.first(sourceCount));
        if (!nativeArguments)
        {
            return std::unexpected(std::move(nativeArguments.error()));
        }
        auto commandList = queues.commandList4(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)
            ->ExecuteIndirect(
                signature->Get(), 1, nativeArguments->argumentResource, nativeArguments->argumentOffset, nullptr, 0);
        test::noteIndirectExecutionRecorded();
        return {};
    }

} // namespace rachis::d3d12
