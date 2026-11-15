#include "Internal/VulkanShaders.h"

#include "Internal/StageLinkage.h"
#include "Internal/VulkanDiagnostics.h"

#ifdef RACHIS_TESTING
#include "Vulkan/VulkanTestHooks.h"
#endif

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rachis::vulkan
{
    namespace
    {

#ifdef RACHIS_TESTING
        std::mutex boundShaderMutex;
        VulkanShaderManager* boundShaders = nullptr;
#endif

        using SpecializationSet = std::vector<SpecializationConstant>;

        [[nodiscard]] constexpr bool isSingleShaderStage(ShaderStage stage) noexcept
        {
            return stage == ShaderStage::vertex || stage == ShaderStage::pixel || stage == ShaderStage::compute ||
                   stage == ShaderStage::geometry || stage == ShaderStage::hull || stage == ShaderStage::domain ||
                   stage == ShaderStage::task || stage == ShaderStage::mesh || stage == ShaderStage::rayGeneration ||
                   stage == ShaderStage::anyHit || stage == ShaderStage::closestHit || stage == ShaderStage::miss ||
                   stage == ShaderStage::intersection || stage == ShaderStage::callable;
        }

        [[nodiscard]] constexpr bool isValidShaderResourceType(ShaderResourceType type) noexcept
        {
            return type == ShaderResourceType::uniformBuffer || type == ShaderResourceType::readOnlyStorageBuffer ||
                   type == ShaderResourceType::storageBuffer || type == ShaderResourceType::sampledTexture ||
                   type == ShaderResourceType::storageTexture || type == ShaderResourceType::sampler ||
                   type == ShaderResourceType::accelerationStructure || type == ShaderResourceType::minMipFeedback ||
                   type == ShaderResourceType::mipRegionUsedFeedback;
        }

        [[nodiscard]] constexpr std::size_t shaderResourceClass(ShaderResourceType type) noexcept
        {
            switch (type)
            {
            case ShaderResourceType::uniformBuffer:
                return 0;
            case ShaderResourceType::readOnlyStorageBuffer:
            case ShaderResourceType::sampledTexture:
            case ShaderResourceType::accelerationStructure:
                return 1;
            case ShaderResourceType::storageBuffer:
            case ShaderResourceType::storageTexture:
            case ShaderResourceType::minMipFeedback:
            case ShaderResourceType::mipRegionUsedFeedback:
                return 2;
            default:
                return 3;
            }
        }

        [[nodiscard]] constexpr bool bindingRequirementsOverlap(const ShaderBindingRequirement& left,
                                                                const ShaderBindingRequirement& right) noexcept
        {
            if (left.set != right.set || shaderResourceClass(left.type) != shaderResourceClass(right.type))
            {
                return false;
            }
            const auto leftEnd = left.count == unboundedBindingCount
                                     ? std::numeric_limits<std::uint64_t>::max()
                                     : static_cast<std::uint64_t>(left.binding) + left.count;
            const auto rightEnd = right.count == unboundedBindingCount
                                      ? std::numeric_limits<std::uint64_t>::max()
                                      : static_cast<std::uint64_t>(right.binding) + right.count;
            return left.binding < rightEnd && right.binding < leftEnd;
        }

        [[nodiscard]] constexpr bool pushConstantRequirementsOverlap(
            const ShaderPushConstantRequirement& left, const ShaderPushConstantRequirement& right) noexcept
        {
            const auto leftEnd = static_cast<std::uint64_t>(left.offset) + left.size;
            const auto rightEnd = static_cast<std::uint64_t>(right.offset) + right.size;
            return left.offset < rightEnd && right.offset < leftEnd;
        }

        [[nodiscard]] constexpr bool fitsInvocationLimit(const ShaderWorkgroupSize& size, std::uint32_t limit) noexcept
        {
            if (size.x == 0 || size.y == 0 || size.z == 0 || limit == 0 || size.x > limit || size.y > limit / size.x)
            {
                return false;
            }
            const auto xy = size.x * size.y;
            return size.z <= limit / xy;
        }

        [[nodiscard]] Result<SpecializationSet> buildSpecializationSet(
            std::span<const SpecializationConstant> constants)
        {
            if (constants.size() > maxSpecializationConstants)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A shader specialization set exceeds the portable constant-count limit.",
                });
            }
            SpecializationSet result(constants.begin(), constants.end());
            for (auto& constant : result)
            {
                switch (constant.format)
                {
                case SpecializationConstantFormat::boolean:
                    constant.valueBits = constant.valueBits != 0 ? 1U : 0U;
                    break;
                case SpecializationConstantFormat::float32:
                case SpecializationConstantFormat::signedInt32:
                case SpecializationConstantFormat::unsignedInt32:
                    break;
                default:
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A shader specialization constant has an invalid format.",
                    });
                }
            }
            std::ranges::sort(result, {}, &SpecializationConstant::id);
            if (std::ranges::adjacent_find(result, {}, &SpecializationConstant::id) != result.end())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader specialization constant IDs must be unique within one value set.",
                });
            }
            return result;
        }

        struct InterfaceRecord final
        {
            std::vector<ShaderBindingRequirement> bindings;
            std::vector<ShaderPushConstantRequirement> pushConstants;
            std::vector<std::uint32_t> vertexInputLocations;
            std::optional<ShaderWorkgroupSize> workgroupSize;
            std::optional<ShaderRecordConstantRange> shaderRecordConstants;
        };

        [[nodiscard]] constexpr bool isValidInterfaceBinding(const ShaderBindingRequirement& binding) noexcept
        {
            return binding.set < maxBindingSets && isValidShaderResourceType(binding.type) && binding.count != 0 &&
                   (binding.count == unboundedBindingCount ||
                    binding.binding <= std::numeric_limits<std::uint32_t>::max() - (binding.count - 1));
        }

        [[nodiscard]] Status validateInterfaceBindings(std::span<const ShaderBindingRequirement> bindings)
        {
            for (std::size_t index = 0; index < bindings.size(); ++index)
            {
                const auto& binding = bindings[index];
                if (!isValidInterfaceBinding(binding))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Shader-interface bindings require a valid set, resource type, and non-empty register range.",
                    });
                }
                const auto conflictsWithBinding = [&](const ShaderBindingRequirement& existing) {
                    return (existing.set == binding.set && existing.binding == binding.binding) ||
                           bindingRequirementsOverlap(existing, binding);
                };
                if (std::ranges::any_of(bindings.first(index), conflictsWithBinding))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Shader-interface bindings must be unique and non-overlapping within each register class.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] constexpr bool isValidPushConstantRange(const ShaderPushConstantRequirement& range) noexcept
        {
            return range.size != 0 && range.offset % sizeof(std::uint32_t) == 0 &&
                   range.size % sizeof(std::uint32_t) == 0 && range.offset <= maxPushConstantBytes &&
                   range.size <= maxPushConstantBytes - range.offset;
        }

        [[nodiscard]] Status validateInterfacePushConstants(
            std::span<const ShaderPushConstantRequirement> pushConstants)
        {
            for (std::size_t index = 0; index < pushConstants.size(); ++index)
            {
                const auto& range = pushConstants[index];
                if (!isValidPushConstantRange(range))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Shader-interface push constants require non-empty, 4-byte aligned ranges within 128 bytes.",
                    });
                }
                const auto overlapsRange = [&](const ShaderPushConstantRequirement& existing) {
                    return pushConstantRequirementsOverlap(range, existing);
                };
                if (std::ranges::any_of(pushConstants.first(index), overlapsRange))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Shader-interface push-constant ranges cannot overlap.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Result<std::vector<std::uint32_t>> buildInterfaceVertexInputLocations(
            ShaderStage stage, std::span<const std::uint32_t> locations)
        {
            std::vector<std::uint32_t> sorted(locations.begin(), locations.end());
            std::ranges::sort(sorted);
            if ((stage != ShaderStage::vertex && !sorted.empty()) ||
                std::ranges::any_of(sorted, [](std::uint32_t location) { return location >= maxVertexAttributes; }) ||
                std::ranges::adjacent_find(sorted) != sorted.end())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Only vertex interfaces may declare unique portable input locations within the device limit.",
                });
            }
            return sorted;
        }

        [[nodiscard]] bool fitsStageWorkgroupLimits(ShaderStage stage,
                                                    const ShaderWorkgroupSize& size,
                                                    const DeviceCapabilities& capabilities)
        {
            if (stage == ShaderStage::compute)
            {
                return size.x <= capabilities.limits.maxComputeWorkgroupSize[0] &&
                       size.y <= capabilities.limits.maxComputeWorkgroupSize[1] &&
                       size.z <= capabilities.limits.maxComputeWorkgroupSize[2] &&
                       fitsInvocationLimit(size, capabilities.limits.maxComputeWorkgroupInvocations);
            }
            if (stage == ShaderStage::task)
            {
                return fitsInvocationLimit(size, capabilities.meshShader.maxTaskWorkgroupInvocations);
            }
            return fitsInvocationLimit(size, capabilities.meshShader.maxMeshWorkgroupInvocations);
        }

        [[nodiscard]] Status validateInterfaceWorkgroupSize(ShaderStage stage,
                                                            const std::optional<ShaderWorkgroupSize>& workgroupSize,
                                                            const DeviceCapabilities& capabilities)
        {
            const bool requiresWorkgroup =
                stage == ShaderStage::compute || stage == ShaderStage::task || stage == ShaderStage::mesh;
            if (workgroupSize.has_value() != requiresWorkgroup)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Compute, task, and mesh interfaces require a workgroup size; other shader stages must omit it.",
                });
            }
            if (!workgroupSize.has_value())
            {
                return {};
            }
            const auto& size = *workgroupSize;
            if (size.x == 0 || size.y == 0 || size.z == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader-interface workgroup dimensions must be non-zero.",
                });
            }
            if (!fitsStageWorkgroupLimits(stage, size, capabilities))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The shader-interface workgroup size exceeds the device's stage limits.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateInterfaceShaderRecordConstants(
            ShaderStage stage, const std::optional<ShaderRecordConstantRange>& shaderRecordConstants)
        {
            if (!shaderRecordConstants.has_value())
            {
                return {};
            }
            const auto& constants = *shaderRecordConstants;
            if (!hasAny(stage, ShaderStage::allRayTracing) || constants.size == 0 ||
                constants.size % sizeof(std::uint32_t) != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader-record constants require a ray-tracing stage and a non-empty, 4-byte aligned range.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<InterfaceRecord> buildShaderInterface(ShaderStage stage,
                                                                   const ShaderInterfaceDesc& desc,
                                                                   const DeviceCapabilities& capabilities)
        {
            if (auto bindings = validateInterfaceBindings(desc.bindings); !bindings)
            {
                return std::unexpected(std::move(bindings.error()));
            }
            if (auto pushConstants = validateInterfacePushConstants(desc.pushConstants); !pushConstants)
            {
                return std::unexpected(std::move(pushConstants.error()));
            }
            auto vertexInputLocations = buildInterfaceVertexInputLocations(stage, desc.vertexInputLocations);
            if (!vertexInputLocations)
            {
                return std::unexpected(std::move(vertexInputLocations.error()));
            }
            if (auto workgroup = validateInterfaceWorkgroupSize(stage, desc.workgroupSize, capabilities); !workgroup)
            {
                return std::unexpected(std::move(workgroup.error()));
            }
            if (auto recordConstants = validateInterfaceShaderRecordConstants(stage, desc.shaderRecordConstants);
                !recordConstants)
            {
                return std::unexpected(std::move(recordConstants.error()));
            }

            InterfaceRecord record;
            record.bindings.assign(desc.bindings.begin(), desc.bindings.end());
            record.pushConstants.assign(desc.pushConstants.begin(), desc.pushConstants.end());
            record.vertexInputLocations = std::move(*vertexInputLocations);
            record.workgroupSize = desc.workgroupSize;
            record.shaderRecordConstants = desc.shaderRecordConstants;
            return record;
        }

        constexpr std::uint32_t spirvMagicNumber = 0x07230203U;
        constexpr std::uint32_t spirvOpUndef = 1;
        constexpr std::uint32_t spirvOpName = 5;
        constexpr std::uint32_t spirvOpMemberName = 6;
        constexpr std::uint32_t spirvOpLine = 8;
        constexpr std::uint32_t spirvOpExtension = 10;
        constexpr std::uint32_t spirvOpExtInstImport = 11;
        constexpr std::uint32_t spirvOpExtInst = 12;
        constexpr std::uint32_t spirvOpMemoryModel = 14;
        constexpr std::uint32_t spirvOpCapability = 17;
        constexpr std::uint32_t spirvOpEntryPoint = 15;
        constexpr std::uint32_t spirvOpExecutionMode = 16;
        constexpr std::uint32_t spirvOpExecutionModeId = 331;
        constexpr std::uint32_t spirvOpTypeVoid = 19;
        constexpr std::uint32_t spirvOpTypeInt = 21;
        constexpr std::uint32_t spirvOpTypeFloat = 22;
        constexpr std::uint32_t spirvOpTypeVector = 23;
        constexpr std::uint32_t spirvOpTypeMatrix = 24;
        constexpr std::uint32_t spirvOpTypeImage = 25;
        constexpr std::uint32_t spirvOpTypeSampler = 26;
        constexpr std::uint32_t spirvOpTypeSampledImage = 27;
        constexpr std::uint32_t spirvOpTypeArray = 28;
        constexpr std::uint32_t spirvOpTypeRuntimeArray = 29;
        constexpr std::uint32_t spirvOpTypeStruct = 30;
        constexpr std::uint32_t spirvOpTypePointer = 32;
        constexpr std::uint32_t spirvOpTypeForwardPointer = 39;
        constexpr std::uint32_t spirvOpTypeRayQuery = 4472;
        constexpr std::uint32_t spirvOpTypeAccelerationStructure = 5341;
        constexpr std::uint32_t spirvOpConstantTrue = 41;
        constexpr std::uint32_t spirvOpConstantFalse = 42;
        constexpr std::uint32_t spirvOpConstant = 43;
        constexpr std::uint32_t spirvOpConstantComposite = 44;
        constexpr std::uint32_t spirvOpConstantNull = 46;
        constexpr std::uint32_t spirvOpSpecConstantTrue = 48;
        constexpr std::uint32_t spirvOpSpecConstantFalse = 49;
        constexpr std::uint32_t spirvOpSpecConstant = 50;
        constexpr std::uint32_t spirvOpSpecConstantComposite = 51;
        constexpr std::uint32_t spirvOpSpecConstantOp = 52;
        constexpr std::uint32_t spirvOpFunction = 54;
        constexpr std::uint32_t spirvOpFunctionParameter = 55;
        constexpr std::uint32_t spirvOpFunctionEnd = 56;
        constexpr std::uint32_t spirvOpFunctionCall = 57;
        constexpr std::uint32_t spirvOpVariable = 59;
        constexpr std::uint32_t spirvOpImageTexelPointer = 60;
        constexpr std::uint32_t spirvOpLoad = 61;
        constexpr std::uint32_t spirvOpStore = 62;
        constexpr std::uint32_t spirvOpCopyMemory = 63;
        constexpr std::uint32_t spirvOpCopyMemorySized = 64;
        constexpr std::uint32_t spirvOpAccessChain = 65;
        constexpr std::uint32_t spirvOpInBoundsAccessChain = 66;
        constexpr std::uint32_t spirvOpArrayLength = 68;
        constexpr std::uint32_t spirvOpVectorShuffle = 79;
        constexpr std::uint32_t spirvOpCompositeExtract = 81;
        constexpr std::uint32_t spirvOpCompositeInsert = 82;
        constexpr std::uint32_t spirvOpImageSampleImplicitLod = 87;
        constexpr std::uint32_t spirvOpImageSampleExplicitLod = 88;
        constexpr std::uint32_t spirvOpImageSampleDrefImplicitLod = 89;
        constexpr std::uint32_t spirvOpImageSampleDrefExplicitLod = 90;
        constexpr std::uint32_t spirvOpImageSampleProjImplicitLod = 91;
        constexpr std::uint32_t spirvOpImageSampleProjExplicitLod = 92;
        constexpr std::uint32_t spirvOpImageSampleProjDrefImplicitLod = 93;
        constexpr std::uint32_t spirvOpImageSampleProjDrefExplicitLod = 94;
        constexpr std::uint32_t spirvOpImageFetch = 95;
        constexpr std::uint32_t spirvOpImageGather = 96;
        constexpr std::uint32_t spirvOpImageDrefGather = 97;
        constexpr std::uint32_t spirvOpImageRead = 98;
        constexpr std::uint32_t spirvOpImageWrite = 99;
        constexpr std::uint32_t spirvOpUConvert = 113;
        constexpr std::uint32_t spirvOpSConvert = 114;
        constexpr std::uint32_t spirvOpSNegate = 126;
        constexpr std::uint32_t spirvOpIAdd = 128;
        constexpr std::uint32_t spirvOpISub = 130;
        constexpr std::uint32_t spirvOpIMul = 132;
        constexpr std::uint32_t spirvOpUDiv = 134;
        constexpr std::uint32_t spirvOpSDiv = 135;
        constexpr std::uint32_t spirvOpUMod = 137;
        constexpr std::uint32_t spirvOpSRem = 138;
        constexpr std::uint32_t spirvOpSMod = 139;
        constexpr std::uint32_t spirvOpLogicalEqual = 164;
        constexpr std::uint32_t spirvOpLogicalNotEqual = 165;
        constexpr std::uint32_t spirvOpLogicalOr = 166;
        constexpr std::uint32_t spirvOpLogicalAnd = 167;
        constexpr std::uint32_t spirvOpLogicalNot = 168;
        constexpr std::uint32_t spirvOpSelect = 169;
        constexpr std::uint32_t spirvOpIEqual = 170;
        constexpr std::uint32_t spirvOpINotEqual = 171;
        constexpr std::uint32_t spirvOpUGreaterThan = 172;
        constexpr std::uint32_t spirvOpSGreaterThan = 173;
        constexpr std::uint32_t spirvOpUGreaterThanEqual = 174;
        constexpr std::uint32_t spirvOpSGreaterThanEqual = 175;
        constexpr std::uint32_t spirvOpULessThan = 176;
        constexpr std::uint32_t spirvOpSLessThan = 177;
        constexpr std::uint32_t spirvOpULessThanEqual = 178;
        constexpr std::uint32_t spirvOpSLessThanEqual = 179;
        constexpr std::uint32_t spirvOpShiftRightLogical = 194;
        constexpr std::uint32_t spirvOpShiftRightArithmetic = 195;
        constexpr std::uint32_t spirvOpShiftLeftLogical = 196;
        constexpr std::uint32_t spirvOpBitwiseOr = 197;
        constexpr std::uint32_t spirvOpBitwiseXor = 198;
        constexpr std::uint32_t spirvOpBitwiseAnd = 199;
        constexpr std::uint32_t spirvOpNot = 200;
        constexpr std::uint32_t spirvOpEmitVertex = 218;
        constexpr std::uint32_t spirvOpEndPrimitive = 219;
        constexpr std::uint32_t spirvOpEmitStreamVertex = 220;
        constexpr std::uint32_t spirvOpEndStreamPrimitive = 221;
        constexpr std::uint32_t spirvOpControlBarrier = 224;
        constexpr std::uint32_t spirvOpMemoryBarrier = 225;
        constexpr std::uint32_t spirvOpAtomicLoad = 227;
        constexpr std::uint32_t spirvOpAtomicStore = 228;
        constexpr std::uint32_t spirvOpAtomicXor = 242;
        constexpr std::uint32_t spirvOpPhi = 245;
        constexpr std::uint32_t spirvOpLoopMerge = 246;
        constexpr std::uint32_t spirvOpSelectionMerge = 247;
        constexpr std::uint32_t spirvOpLabel = 248;
        constexpr std::uint32_t spirvOpBranch = 249;
        constexpr std::uint32_t spirvOpBranchConditional = 250;
        constexpr std::uint32_t spirvOpSwitch = 251;
        constexpr std::uint32_t spirvOpKill = 252;
        constexpr std::uint32_t spirvOpReturn = 253;
        constexpr std::uint32_t spirvOpReturnValue = 254;
        constexpr std::uint32_t spirvOpUnreachable = 255;
        constexpr std::uint32_t spirvOpLifetimeStart = 256;
        constexpr std::uint32_t spirvOpLifetimeStop = 257;
        constexpr std::uint32_t spirvOpGenericCastToPtrExplicit = 123;
        constexpr std::uint32_t spirvOpGroupIAdd = 264;
        constexpr std::uint32_t spirvOpGroupSMax = 271;
        constexpr std::uint32_t spirvOpImageSparseSampleImplicitLod = 305;
        constexpr std::uint32_t spirvOpImageSparseSampleExplicitLod = 306;
        constexpr std::uint32_t spirvOpImageSparseSampleDrefImplicitLod = 307;
        constexpr std::uint32_t spirvOpImageSparseSampleDrefExplicitLod = 308;
        constexpr std::uint32_t spirvOpImageSparseSampleProjImplicitLod = 309;
        constexpr std::uint32_t spirvOpImageSparseSampleProjExplicitLod = 310;
        constexpr std::uint32_t spirvOpImageSparseSampleProjDrefImplicitLod = 311;
        constexpr std::uint32_t spirvOpImageSparseSampleProjDrefExplicitLod = 312;
        constexpr std::uint32_t spirvOpImageSparseFetch = 313;
        constexpr std::uint32_t spirvOpImageSparseGather = 314;
        constexpr std::uint32_t spirvOpImageSparseDrefGather = 315;
        constexpr std::uint32_t spirvOpNoLine = 317;
        constexpr std::uint32_t spirvOpAtomicFlagClear = 319;
        constexpr std::uint32_t spirvOpImageSparseRead = 320;
        constexpr std::uint32_t spirvOpMemoryNamedBarrier = 329;
        constexpr std::uint32_t spirvOpModuleProcessed = 330;
        constexpr std::uint32_t spirvOpGroupNonUniformBallotBitCount = 342;
        constexpr std::uint32_t spirvOpGroupNonUniformIAdd = 349;
        constexpr std::uint32_t spirvOpGroupNonUniformLogicalXor = 364;
        constexpr std::uint32_t spirvOpTerminateInvocation = 4416;
        constexpr std::uint32_t spirvOpExtInstWithForwardRefsKHR = 4433;
        constexpr std::uint32_t spirvOpTraceRayKHR = 4445;
        constexpr std::uint32_t spirvOpExecuteCallableKHR = 4446;
        constexpr std::uint32_t spirvOpIgnoreIntersectionKHR = 4448;
        constexpr std::uint32_t spirvOpTerminateRayKHR = 4449;
        constexpr std::uint32_t spirvOpSDot = 4450;
        constexpr std::uint32_t spirvOpUDot = 4451;
        constexpr std::uint32_t spirvOpSUDot = 4452;
        constexpr std::uint32_t spirvOpSDotAccSat = 4453;
        constexpr std::uint32_t spirvOpUDotAccSat = 4454;
        constexpr std::uint32_t spirvOpSUDotAccSat = 4455;
        constexpr std::uint32_t spirvOpRayQueryInitializeKHR = 4473;
        constexpr std::uint32_t spirvOpRayQueryTerminateKHR = 4474;
        constexpr std::uint32_t spirvOpRayQueryGenerateIntersectionKHR = 4475;
        constexpr std::uint32_t spirvOpRayQueryConfirmIntersectionKHR = 4476;
        constexpr std::uint32_t spirvOpGroupIAddNonUniformAMD = 5000;
        constexpr std::uint32_t spirvOpGroupSMaxNonUniformAMD = 5007;
        constexpr std::uint32_t spirvOpImageSampleFootprintNV = 5283;
        constexpr std::uint32_t spirvOpEmitMeshTasksEXT = 5294;
        constexpr std::uint32_t spirvOpSetMeshOutputsEXT = 5295;
        constexpr std::uint32_t spirvOpWritePackedPrimitiveIndices4x8NV = 5299;
        constexpr std::uint32_t spirvOpIgnoreIntersectionNV = 5335;
        constexpr std::uint32_t spirvOpTerminateRayNV = 5336;
        constexpr std::uint32_t spirvOpTraceNV = 5337;
        constexpr std::uint32_t spirvOpTraceMotionNV = 5338;
        constexpr std::uint32_t spirvOpTraceRayMotionNV = 5339;
        constexpr std::uint32_t spirvOpExecuteCallableNV = 5344;
        constexpr std::uint32_t spirvOpCooperativeMatrixLoadNV = 5359;
        constexpr std::uint32_t spirvOpCooperativeMatrixStoreNV = 5360;
        constexpr std::uint32_t spirvOpBeginInvocationInterlockEXT = 5364;
        constexpr std::uint32_t spirvOpEndInvocationInterlockEXT = 5365;
        constexpr std::uint32_t spirvOpDemoteToHelperInvocation = 5380;
        constexpr std::uint32_t spirvOpSamplerImageAddressingModeNV = 5397;
        constexpr std::uint32_t spirvOpAssumeTrueKHR = 5630;
        constexpr std::uint32_t spirvOpGroupIMulKHR = 6401;
        constexpr std::uint32_t spirvOpGroupLogicalXorKHR = 6408;
        constexpr std::uint32_t spirvOpDecorate = 71;
        constexpr std::uint32_t spirvOpMemberDecorate = 72;
        constexpr std::uint32_t spirvOpDecorationGroup = 73;
        constexpr std::uint32_t spirvOpGroupDecorate = 74;
        constexpr std::uint32_t spirvOpGroupMemberDecorate = 75;
        constexpr std::uint32_t spirvOpDecorateId = 332;
        constexpr std::uint32_t spirvOpDecorateString = 5632;
        constexpr std::uint32_t spirvOpMemberDecorateString = 5633;
        constexpr std::uint32_t spirvExecutionModelVertex = 0;
        constexpr std::uint32_t spirvExecutionModelTessellationControl = 1;
        constexpr std::uint32_t spirvExecutionModelTessellationEvaluation = 2;
        constexpr std::uint32_t spirvExecutionModelGeometry = 3;
        constexpr std::uint32_t spirvExecutionModelFragment = 4;
        constexpr std::uint32_t spirvExecutionModelGLCompute = 5;
        constexpr std::uint32_t spirvExecutionModelTaskEXT = 5364;
        constexpr std::uint32_t spirvExecutionModelMeshEXT = 5365;
        constexpr std::uint32_t spirvExecutionModelRayGenerationKHR = 5313;
        constexpr std::uint32_t spirvExecutionModelIntersectionKHR = 5314;
        constexpr std::uint32_t spirvExecutionModelAnyHitKHR = 5315;
        constexpr std::uint32_t spirvExecutionModelClosestHitKHR = 5316;
        constexpr std::uint32_t spirvExecutionModelMissKHR = 5317;
        constexpr std::uint32_t spirvExecutionModelCallableKHR = 5318;
        constexpr std::uint32_t spirvExecutionModeLocalSize = 17;
        constexpr std::uint32_t spirvExecutionModeLocalSizeId = 38;
        constexpr std::uint32_t spirvDecorationSpecId = 1;
        constexpr std::uint32_t spirvDecorationBlock = 2;
        constexpr std::uint32_t spirvDecorationBufferBlock = 3;
        constexpr std::uint32_t spirvDecorationRowMajor = 4;
        constexpr std::uint32_t spirvDecorationArrayStride = 6;
        constexpr std::uint32_t spirvDecorationMatrixStride = 7;
        constexpr std::uint32_t spirvDecorationBuiltIn = 11;
        constexpr std::uint32_t spirvDecorationNonWritable = 24;
        constexpr std::uint32_t spirvDecorationLocation = 30;
        constexpr std::uint32_t spirvDecorationComponent = 31;
        constexpr std::uint32_t spirvDecorationBinding = 33;
        constexpr std::uint32_t spirvDecorationDescriptorSet = 34;
        constexpr std::uint32_t spirvDecorationOffset = 35;
        constexpr std::uint32_t spirvDecorationPerVertexKHR = 5285;
        constexpr std::uint32_t spirvBuiltInPointSize = 1;
        constexpr std::uint32_t spirvStorageClassUniformConstant = 0;
        constexpr std::uint32_t spirvStorageClassInput = 1;
        constexpr std::uint32_t spirvStorageClassUniform = 2;
        constexpr std::uint32_t spirvStorageClassOutput = 3;
        constexpr std::uint32_t spirvStorageClassWorkgroup = 4;
        constexpr std::uint32_t spirvStorageClassFunction = 7;
        constexpr std::uint32_t spirvStorageClassPushConstant = 9;
        constexpr std::uint32_t spirvStorageClassImage = 11;
        constexpr std::uint32_t spirvStorageClassStorageBuffer = 12;
        constexpr std::uint32_t spirvCapabilityGeometry = 2;
        constexpr std::uint32_t spirvCapabilityTessellation = 3;
        constexpr std::uint32_t spirvCapabilityFloat16 = 9;
        constexpr std::uint32_t spirvCapabilityFloat64 = 10;
        constexpr std::uint32_t spirvCapabilityInt64 = 11;
        constexpr std::uint32_t spirvCapabilityInt64Atomics = 12;
        constexpr std::uint32_t spirvCapabilityInt16 = 22;
        constexpr std::uint32_t spirvCapabilityImageGatherExtended = 25;
        constexpr std::uint32_t spirvCapabilityClipDistance = 32;
        constexpr std::uint32_t spirvCapabilityCullDistance = 33;
        constexpr std::uint32_t spirvCapabilitySampleRateShading = 35;
        constexpr std::uint32_t spirvCapabilityMinLod = 42;
        constexpr std::uint32_t spirvCapabilityMultiViewport = 57;
        constexpr std::uint32_t spirvCapabilityGroupNonUniform = 61;
        constexpr std::uint32_t spirvCapabilityGroupNonUniformArithmetic = 63;
        constexpr std::uint32_t spirvCapabilityDrawParameters = 4427;
        constexpr std::uint32_t spirvCapabilityStorageBuffer16BitAccess = 4433;
        constexpr std::uint32_t spirvCapabilityUniformAndStorageBuffer16BitAccess = 4434;
        constexpr std::uint32_t spirvCapabilityStoragePushConstant16 = 4435;
        constexpr std::uint32_t spirvCapabilityStorageInputOutput16 = 4436;
        constexpr std::uint32_t spirvCapabilityMultiView = 4439;
        constexpr std::uint32_t spirvCapabilityRayQueryKHR = 4472;
        constexpr std::uint32_t spirvCapabilityRayTracingKHR = 4479;
        constexpr std::uint32_t spirvCapabilityInt64ImageExt = 5016;
        constexpr std::uint32_t spirvCapabilityShaderViewportIndexLayerEXT = 5254;
        constexpr std::uint32_t spirvCapabilityMeshShadingEXT = 5283;
        constexpr std::uint32_t spirvCapabilityFragmentBarycentricKHR = 5284;

        [[nodiscard]] std::optional<ShaderStage> shaderStageFromExecutionModel(std::uint32_t model) noexcept
        {
            switch (model)
            {
            case spirvExecutionModelVertex:
                return ShaderStage::vertex;
            case spirvExecutionModelTessellationControl:
                return ShaderStage::hull;
            case spirvExecutionModelTessellationEvaluation:
                return ShaderStage::domain;
            case spirvExecutionModelGeometry:
                return ShaderStage::geometry;
            case spirvExecutionModelFragment:
                return ShaderStage::pixel;
            case spirvExecutionModelGLCompute:
                return ShaderStage::compute;
            case spirvExecutionModelTaskEXT:
                return ShaderStage::task;
            case spirvExecutionModelMeshEXT:
                return ShaderStage::mesh;
            case spirvExecutionModelRayGenerationKHR:
                return ShaderStage::rayGeneration;
            case spirvExecutionModelAnyHitKHR:
                return ShaderStage::anyHit;
            case spirvExecutionModelClosestHitKHR:
                return ShaderStage::closestHit;
            case spirvExecutionModelMissKHR:
                return ShaderStage::miss;
            case spirvExecutionModelIntersectionKHR:
                return ShaderStage::intersection;
            case spirvExecutionModelCallableKHR:
                return ShaderStage::callable;
            default:
                return std::nullopt;
            }
        }

        struct SpirvReflection final
        {
            std::string entryPoint = {};
            ShaderStage stage = ShaderStage::none;
            std::vector<ShaderBindingRequirement> bindings = {};
            std::uint32_t pushConstantBytes = 0;
            std::vector<std::uint32_t> vertexInputLocations = {};
            std::optional<ShaderWorkgroupSize> workgroupSize = {};
            std::unordered_set<std::uint32_t> specIds = {};
            std::unordered_set<std::uint32_t> non32BitSpecIds = {};
            std::uint32_t readOnlyStorageArrayStride = 0;
            std::uint32_t storageArrayStride = 0;
            bool writesPointSize = false;
            bool writableStorage = false;
            std::vector<detail::StageInterfaceVariable> stageOutputs = {};
            std::vector<detail::StageInterfaceVariable> stageInputs = {};
        };

        struct TypeInfo final
        {
            enum class Kind : std::uint8_t
            {
                unknown,
                scalar,
                vector,
                matrix,
                image,
                sampler,
                sampledImage,
                pointer,
                array,
                runtimeArray,
                structure,
                acceleration,
            };

            enum class ScalarKind : std::uint8_t
            {
                none,
                floatingPoint,
                signedInteger,
                unsignedInteger,
            };

            Kind kind = Kind::unknown;
            ScalarKind scalarKind = ScalarKind::none;
            std::uint32_t pointee = 0;
            std::uint32_t storageClass = 0;
            std::uint32_t element = 0;
            std::uint32_t sampled = 0;
            std::uint32_t arrayLength = 1;
            std::uint32_t arrayLengthId = 0;
            std::uint32_t scalarBytes = 0;
            bool runtimeArray = false;
            std::vector<std::uint32_t> members;
        };

        struct FunctionReflectionInfo final
        {
            std::vector<std::uint32_t> callees;
            std::vector<std::uint32_t> referencedIds;
            std::vector<std::uint32_t> writtenPointers;
        };

        constexpr std::size_t spirvHeaderWordCount = 5;
        constexpr std::size_t spirvHeaderIdBoundWord = 3;

        struct SpirvInstruction final
        {
            std::uint32_t opcode = 0;
            std::uint32_t wordCount = 0;
            const std::uint32_t* words = nullptr;
        };

        [[nodiscard]] Result<std::vector<std::uint32_t>> copySpirvWords(std::span<const std::byte> bytecode)
        {
            if (bytecode.size() < spirvHeaderWordCount * sizeof(std::uint32_t) ||
                bytecode.size() % sizeof(std::uint32_t) != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "SPIR-V bytecode must be a non-empty multiple of four bytes.",
                });
            }
            std::uint32_t magic = 0;
            std::memcpy(&magic, bytecode.data(), sizeof(magic));
            if (magic != spirvMagicNumber)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader bytecode is not a SPIR-V module.",
                });
            }
            std::vector<std::uint32_t> words(bytecode.size() / sizeof(std::uint32_t));
            std::memcpy(words.data(), bytecode.data(), bytecode.size());
            return words;
        }

        [[nodiscard]] Result<SpirvInstruction> decodeInstruction(std::span<const std::uint32_t> moduleWords,
                                                                 std::size_t index)
        {
            const auto first = moduleWords[index];
            const auto count = first >> 16U;
            const auto opcode = first & 0xFFFFU;
            if (count == 0 || index + count > moduleWords.size())
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, "SPIR-V instruction is truncated."});
            }
            return SpirvInstruction{.opcode = opcode, .wordCount = count, .words = moduleWords.data() + index};
        }

        [[nodiscard]] constexpr bool hasGroupOperationLiteralOperand(std::uint32_t opcode) noexcept
        {
            return (opcode >= spirvOpGroupIAdd && opcode <= spirvOpGroupSMax) ||
                   opcode == spirvOpGroupNonUniformBallotBitCount ||
                   (opcode >= spirvOpGroupNonUniformIAdd && opcode <= spirvOpGroupNonUniformLogicalXor) ||
                   (opcode >= spirvOpGroupIAddNonUniformAMD && opcode <= spirvOpGroupSMaxNonUniformAMD) ||
                   (opcode >= spirvOpGroupIMulKHR && opcode <= spirvOpGroupLogicalXorKHR);
        }

        [[nodiscard]] constexpr bool referencesNoTrackedIds(std::uint32_t opcode) noexcept
        {
            return opcode <= 8 || opcode == spirvOpNoLine || opcode == spirvOpModuleProcessed ||
                   opcode == spirvOpFunction || opcode == spirvOpFunctionParameter || opcode == spirvOpFunctionEnd ||
                   opcode == spirvOpLabel || opcode == spirvOpLoopMerge || opcode == spirvOpSelectionMerge ||
                   opcode == spirvOpBranch || opcode == spirvOpKill || opcode == spirvOpReturn ||
                   opcode == spirvOpUnreachable || opcode == spirvOpTerminateInvocation ||
                   opcode == spirvOpDemoteToHelperInvocation || opcode == spirvOpEmitVertex ||
                   opcode == spirvOpEndPrimitive || opcode == spirvOpIgnoreIntersectionKHR ||
                   opcode == spirvOpTerminateRayKHR || opcode == spirvOpIgnoreIntersectionNV ||
                   opcode == spirvOpTerminateRayNV || opcode == spirvOpBeginInvocationInterlockEXT ||
                   opcode == spirvOpEndInvocationInterlockEXT || opcode == spirvOpSamplerImageAddressingModeNV;
        }

        constexpr std::uint32_t noTrailingIdOperands = std::numeric_limits<std::uint32_t>::max();

        struct IdOperandLayout final
        {
            std::uint32_t minimumWordCount = 0;
            std::uint32_t leadingIdCount = 0;
            std::uint32_t trailingIdsBegin = noTrailingIdOperands;
        };

        [[nodiscard]] constexpr std::optional<IdOperandLayout> resultlessIdOperandLayout(std::uint32_t opcode) noexcept
        {
            switch (opcode)
            {
            case spirvOpBranchConditional:
            case spirvOpSwitch:
            case spirvOpReturnValue:
            case spirvOpLifetimeStart:
            case spirvOpLifetimeStop:
                return IdOperandLayout{.minimumWordCount = 2, .leadingIdCount = 1};
            case spirvOpStore:
            case spirvOpCopyMemory:
                return IdOperandLayout{.minimumWordCount = 3, .leadingIdCount = 2};
            case spirvOpCopyMemorySized:
                return IdOperandLayout{.minimumWordCount = 4, .leadingIdCount = 3};
            case spirvOpCooperativeMatrixStoreNV:
                return IdOperandLayout{.minimumWordCount = 5, .leadingIdCount = 4};
            case spirvOpImageWrite:
                return IdOperandLayout{.minimumWordCount = 4, .leadingIdCount = 3, .trailingIdsBegin = 5};
            case spirvOpControlBarrier:
            case spirvOpMemoryBarrier:
            case spirvOpMemoryNamedBarrier:
            case spirvOpAtomicStore:
            case spirvOpAtomicFlagClear:
            case spirvOpEmitStreamVertex:
            case spirvOpEndStreamPrimitive:
            case spirvOpTraceRayKHR:
            case spirvOpExecuteCallableKHR:
            case spirvOpRayQueryInitializeKHR:
            case spirvOpRayQueryTerminateKHR:
            case spirvOpRayQueryGenerateIntersectionKHR:
            case spirvOpRayQueryConfirmIntersectionKHR:
            case spirvOpEmitMeshTasksEXT:
            case spirvOpSetMeshOutputsEXT:
            case spirvOpWritePackedPrimitiveIndices4x8NV:
            case spirvOpTraceNV:
            case spirvOpTraceMotionNV:
            case spirvOpTraceRayMotionNV:
            case spirvOpExecuteCallableNV:
            case spirvOpAssumeTrueKHR:
                return IdOperandLayout{.minimumWordCount = 2, .leadingIdCount = 0, .trailingIdsBegin = 1};
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] constexpr IdOperandLayout resultIdOperandLayout(std::uint32_t opcode) noexcept
        {
            if (hasGroupOperationLiteralOperand(opcode))
            {
                return IdOperandLayout{.minimumWordCount = 6, .leadingIdCount = 1, .trailingIdsBegin = 5};
            }
            switch (opcode)
            {
            case spirvOpLoad:
            case spirvOpArrayLength:
            case spirvOpCompositeExtract:
            case spirvOpGenericCastToPtrExplicit:
                return IdOperandLayout{.minimumWordCount = 4, .leadingIdCount = 1};
            case spirvOpVectorShuffle:
            case spirvOpCompositeInsert:
            case spirvOpSDot:
            case spirvOpUDot:
            case spirvOpSUDot:
                return IdOperandLayout{.minimumWordCount = 5, .leadingIdCount = 2};
            case spirvOpSDotAccSat:
            case spirvOpUDotAccSat:
            case spirvOpSUDotAccSat:
            case spirvOpCooperativeMatrixLoadNV:
                return IdOperandLayout{.minimumWordCount = 6, .leadingIdCount = 3};
            case spirvOpImageSampleImplicitLod:
            case spirvOpImageSampleExplicitLod:
            case spirvOpImageSampleProjImplicitLod:
            case spirvOpImageSampleProjExplicitLod:
            case spirvOpImageFetch:
            case spirvOpImageRead:
            case spirvOpImageSparseSampleImplicitLod:
            case spirvOpImageSparseSampleExplicitLod:
            case spirvOpImageSparseSampleProjImplicitLod:
            case spirvOpImageSparseSampleProjExplicitLod:
            case spirvOpImageSparseFetch:
            case spirvOpImageSparseRead:
                return IdOperandLayout{.minimumWordCount = 5, .leadingIdCount = 2, .trailingIdsBegin = 6};
            case spirvOpImageSampleDrefImplicitLod:
            case spirvOpImageSampleDrefExplicitLod:
            case spirvOpImageSampleProjDrefImplicitLod:
            case spirvOpImageSampleProjDrefExplicitLod:
            case spirvOpImageGather:
            case spirvOpImageDrefGather:
            case spirvOpImageSparseSampleDrefImplicitLod:
            case spirvOpImageSparseSampleDrefExplicitLod:
            case spirvOpImageSparseSampleProjDrefImplicitLod:
            case spirvOpImageSparseSampleProjDrefExplicitLod:
            case spirvOpImageSparseGather:
            case spirvOpImageSparseDrefGather:
                return IdOperandLayout{.minimumWordCount = 6, .leadingIdCount = 3, .trailingIdsBegin = 7};
            case spirvOpImageSampleFootprintNV:
                return IdOperandLayout{.minimumWordCount = 7, .leadingIdCount = 4, .trailingIdsBegin = 8};
            default:
                return IdOperandLayout{.minimumWordCount = 3, .leadingIdCount = 0, .trailingIdsBegin = 3};
            }
        }

        void recordResultlessIdOperands(const IdOperandLayout& layout,
                                        const SpirvInstruction& instruction,
                                        FunctionReflectionInfo& function)
        {
            if (instruction.wordCount < layout.minimumWordCount)
            {
                return;
            }
            for (std::uint32_t i = 1; i <= layout.leadingIdCount; ++i)
            {
                function.referencedIds.push_back(instruction.words[i]);
            }
            for (std::uint32_t i = layout.trailingIdsBegin; i < instruction.wordCount; ++i)
            {
                function.referencedIds.push_back(instruction.words[i]);
            }
        }

        void recordDerivedOperand(std::uint32_t resultId,
                                  std::uint32_t operandId,
                                  FunctionReflectionInfo& function,
                                  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& derivedFrom)
        {
            function.referencedIds.push_back(operandId);
            if (resultId != 0)
            {
                derivedFrom[resultId].push_back(operandId);
            }
        }

        void recordResultIdOperands(const IdOperandLayout& layout,
                                    const SpirvInstruction& instruction,
                                    FunctionReflectionInfo& function,
                                    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& derivedFrom)
        {
            if (instruction.wordCount < layout.minimumWordCount)
            {
                return;
            }
            const auto* words = instruction.words;
            for (std::uint32_t i = 3; i < 3 + layout.leadingIdCount; ++i)
            {
                recordDerivedOperand(words[2], words[i], function, derivedFrom);
            }
            for (std::uint32_t i = layout.trailingIdsBegin; i < instruction.wordCount; ++i)
            {
                recordDerivedOperand(words[2], words[i], function, derivedFrom);
            }
        }

        void recordFunctionInstruction(const SpirvInstruction& instruction,
                                       FunctionReflectionInfo& function,
                                       std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& derivedFrom,
                                       const std::unordered_set<std::uint32_t>& nonSemanticExtSets)
        {
            const auto opcode = instruction.opcode;
            const auto wordCount = instruction.wordCount;
            const auto* words = instruction.words;
            if (wordCount <= 1 || referencesNoTrackedIds(opcode))
            {
                return;
            }
            if (const auto resultless = resultlessIdOperandLayout(opcode); resultless.has_value())
            {
                recordResultlessIdOperands(*resultless, instruction, function);
                return;
            }

            switch (opcode)
            {
            case spirvOpVariable:
                if (wordCount >= 5)
                {
                    recordDerivedOperand(words[2], words[4], function, derivedFrom);
                }
                return;
            case spirvOpFunctionCall:
                if (wordCount >= 4)
                {
                    function.callees.push_back(words[3]);
                    for (std::uint32_t i = 4; i < wordCount; ++i)
                    {
                        recordDerivedOperand(words[2], words[i], function, derivedFrom);
                    }
                }
                return;
            case spirvOpExtInst: {
                const bool isNonSemantic = wordCount >= 5 && nonSemanticExtSets.contains(words[3]);
                const IdOperandLayout layout{
                    .minimumWordCount = 5,
                    .leadingIdCount = 0,
                    .trailingIdsBegin = isNonSemantic ? noTrailingIdOperands : 5U,
                };
                recordResultIdOperands(layout, instruction, function, derivedFrom);
                return;
            }
            case spirvOpPhi:
                if (wordCount >= 3)
                {
                    for (std::uint32_t i = 3; i < wordCount; i += 2)
                    {
                        recordDerivedOperand(words[2], words[i], function, derivedFrom);
                    }
                }
                return;
            default:
                recordResultIdOperands(resultIdOperandLayout(opcode), instruction, function, derivedFrom);
                return;
            }
        }

        struct SpecConstantOpRecord final
        {
            std::uint32_t resultTypeId = 0;
            std::uint32_t resultId = 0;
            std::uint32_t innerOpcode = 0;
            std::vector<std::uint32_t> operands;
        };

        struct SpirvConstantTable final
        {
            std::unordered_map<std::uint32_t, std::uint32_t> scalars;
            std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> composites;
            std::unordered_map<std::uint32_t, std::uint32_t> valueTypeIds;
        };

        void recordSpecConstantOpDependencies(
            std::uint32_t resultId,
            std::uint32_t innerOpcode,
            const std::uint32_t* operands,
            std::uint32_t operandCount,
            std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& derivedFrom)
        {
            if (operandCount == 0)
            {
                return;
            }
            switch (innerOpcode)
            {
            case spirvOpCompositeExtract:
            case spirvOpGenericCastToPtrExplicit:
                derivedFrom[resultId].push_back(operands[0]);
                return;
            case spirvOpVectorShuffle:
            case spirvOpCompositeInsert:
                derivedFrom[resultId].push_back(operands[0]);
                if (operandCount >= 2)
                {
                    derivedFrom[resultId].push_back(operands[1]);
                }
                return;
            default:
                for (std::uint32_t i = 0; i < operandCount; ++i)
                {
                    derivedFrom[resultId].push_back(operands[i]);
                }
                return;
            }
        }

        [[nodiscard]] std::optional<std::uint32_t> findScalarConstant(
            const std::unordered_map<std::uint32_t, std::uint32_t>& constants, std::uint32_t id)
        {
            const auto it = constants.find(id);
            return it != constants.end() ? std::optional<std::uint32_t>{it->second} : std::nullopt;
        }

        [[nodiscard]] std::uint32_t scalarByteWidth(const std::unordered_map<std::uint32_t, TypeInfo>& types,
                                                    std::uint32_t typeId) noexcept
        {
            const auto typeIt = types.find(typeId);
            return typeIt != types.end() && typeIt->second.kind == TypeInfo::Kind::scalar ? typeIt->second.scalarBytes
                                                                                          : 0;
        }

        [[nodiscard]] std::uint32_t constantByteWidth(
            const std::unordered_map<std::uint32_t, TypeInfo>& types,
            const std::unordered_map<std::uint32_t, std::uint32_t>& valueTypeIds,
            std::uint32_t constantId) noexcept
        {
            const auto typeIdIt = valueTypeIds.find(constantId);
            return typeIdIt != valueTypeIds.end() ? scalarByteWidth(types, typeIdIt->second) : 0;
        }

        [[nodiscard]] constexpr std::uint32_t scalarWidthMask(std::uint32_t scalarBytes) noexcept
        {
            if (scalarBytes == 1)
            {
                return 0xFFU;
            }
            if (scalarBytes == 2)
            {
                return 0xFFFFU;
            }
            return 0xFFFF'FFFFU;
        }

        [[nodiscard]] constexpr std::int32_t signExtendScalar(std::uint32_t value, std::uint32_t scalarBytes) noexcept
        {
            if (scalarBytes == 1)
            {
                return static_cast<std::int32_t>(static_cast<std::int8_t>(value & 0xFFU));
            }
            if (scalarBytes == 2)
            {
                return static_cast<std::int32_t>(static_cast<std::int16_t>(value & 0xFFFFU));
            }
            return std::bit_cast<std::int32_t>(value);
        }

        void evaluateSpecConstantVectorShuffle(
            const SpecConstantOpRecord& specOp,
            std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& compositeConstants)
        {
            const auto& operands = specOp.operands;
            const auto vec1It = compositeConstants.find(operands[0]);
            const auto vec2It = compositeConstants.find(operands[1]);
            if (vec1It == compositeConstants.end() || vec2It == compositeConstants.end())
            {
                return;
            }
            const auto& vec1 = vec1It->second;
            const auto& vec2 = vec2It->second;
            std::vector<std::uint32_t> shuffled;
            shuffled.reserve(operands.size() - 2);
            for (const auto comp : std::span<const std::uint32_t>(operands).subspan(2))
            {
                if (comp == 0xFFFFFFFFU)
                {
                    shuffled.push_back(0U);
                }
                else if (comp < vec1.size())
                {
                    shuffled.push_back(vec1[comp]);
                }
                else if (comp - vec1.size() < vec2.size())
                {
                    shuffled.push_back(vec2[comp - vec1.size()]);
                }
                else
                {
                    return;
                }
            }
            compositeConstants[specOp.resultId] = std::move(shuffled);
        }

        [[nodiscard]] std::optional<std::vector<std::uint32_t>> cloneCompositeWithInsertedObject(
            std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& compositeConstants,
            std::uint32_t compositeId,
            std::span<const std::uint32_t> indexPath,
            std::uint32_t objectId,
            std::uint32_t& nextSyntheticId)
        {
            const auto compIt = compositeConstants.find(compositeId);
            if (compIt == compositeConstants.end())
            {
                return std::nullopt;
            }
            const auto idx = indexPath.front();
            if (idx >= compIt->second.size())
            {
                return std::nullopt;
            }
            auto copy = compIt->second;
            if (indexPath.size() == 1)
            {
                copy[idx] = objectId;
                return copy;
            }
            auto nested = cloneCompositeWithInsertedObject(
                compositeConstants, copy[idx], indexPath.subspan(1), objectId, nextSyntheticId);
            if (!nested.has_value())
            {
                return std::nullopt;
            }
            const auto synthId = nextSyntheticId++;
            compositeConstants[synthId] = std::move(*nested);
            copy[idx] = synthId;
            return copy;
        }

        void evaluateSpecConstantCompositeInsert(
            const SpecConstantOpRecord& specOp,
            std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& compositeConstants)
        {
            std::uint32_t nextSyntheticId = 0x8000'0000U + static_cast<std::uint32_t>(compositeConstants.size());
            const auto objectId = specOp.operands[0];
            const auto compositeId = specOp.operands[1];
            const auto indexPath = std::span<const std::uint32_t>(specOp.operands).subspan(2);
            auto updated =
                cloneCompositeWithInsertedObject(compositeConstants, compositeId, indexPath, objectId, nextSyntheticId);
            if (updated.has_value())
            {
                compositeConstants[specOp.resultId] = std::move(*updated);
            }
        }

        void evaluateSpecConstantCompositeExtract(const SpecConstantOpRecord& specOp, SpirvConstantTable& constants)
        {
            std::uint32_t currentId = specOp.operands[0];
            for (const auto memberIndex : std::span<const std::uint32_t>(specOp.operands).subspan(1))
            {
                const auto compIt = constants.composites.find(currentId);
                if (compIt == constants.composites.end() || memberIndex >= compIt->second.size())
                {
                    return;
                }
                currentId = compIt->second[memberIndex];
            }
            if (const auto compIt = constants.composites.find(currentId); compIt != constants.composites.end())
            {
                constants.composites[specOp.resultId] = compIt->second;
            }
            if (const auto val = findScalarConstant(constants.scalars, currentId); val.has_value())
            {
                constants.scalars[specOp.resultId] = *val;
            }
        }

        [[nodiscard]] constexpr std::optional<std::uint32_t> evaluateUnarySpecConstantOp(
            std::uint32_t innerOpcode,
            std::uint32_t operand,
            std::uint32_t operandBytes,
            std::uint32_t resultMask) noexcept
        {
            switch (innerOpcode)
            {
            case spirvOpUConvert:
                return operand & scalarWidthMask(operandBytes) & resultMask;
            case spirvOpSConvert:
                return std::bit_cast<std::uint32_t>(signExtendScalar(operand, operandBytes)) & resultMask;
            case spirvOpSNegate:
                return (0U - operand) & resultMask;
            case spirvOpLogicalNot:
                return operand == 0 ? 1U : 0U;
            case spirvOpNot:
                return ~operand & resultMask;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] constexpr std::optional<bool> evaluateSpecConstantComparison(std::uint32_t innerOpcode,
                                                                                   std::uint32_t lhs,
                                                                                   std::uint32_t rhs) noexcept
        {
            const auto sLhs = std::bit_cast<std::int32_t>(lhs);
            const auto sRhs = std::bit_cast<std::int32_t>(rhs);
            switch (innerOpcode)
            {
            case spirvOpLogicalEqual:
                return (lhs != 0) == (rhs != 0);
            case spirvOpLogicalNotEqual:
                return (lhs != 0) != (rhs != 0);
            case spirvOpLogicalOr:
                return lhs != 0 || rhs != 0;
            case spirvOpLogicalAnd:
                return lhs != 0 && rhs != 0;
            case spirvOpIEqual:
                return lhs == rhs;
            case spirvOpINotEqual:
                return lhs != rhs;
            case spirvOpUGreaterThan:
                return lhs > rhs;
            case spirvOpSGreaterThan:
                return sLhs > sRhs;
            case spirvOpUGreaterThanEqual:
                return lhs >= rhs;
            case spirvOpSGreaterThanEqual:
                return sLhs >= sRhs;
            case spirvOpULessThan:
                return lhs < rhs;
            case spirvOpSLessThan:
                return sLhs < sRhs;
            case spirvOpULessThanEqual:
                return lhs <= rhs;
            case spirvOpSLessThanEqual:
                return sLhs <= sRhs;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] constexpr bool signedDivisionOverflows(std::int32_t lhs, std::int32_t rhs) noexcept
        {
            return lhs == (std::numeric_limits<std::int32_t>::min)() && rhs == -1;
        }

        [[nodiscard]] constexpr std::optional<std::uint32_t> signedQuotient(std::int32_t lhs,
                                                                            std::int32_t rhs,
                                                                            std::uint32_t resultMask) noexcept
        {
            if (rhs == 0)
            {
                return std::nullopt;
            }
            if (signedDivisionOverflows(lhs, rhs))
            {
                return std::bit_cast<std::uint32_t>(lhs) & resultMask;
            }
            return std::bit_cast<std::uint32_t>(lhs / rhs) & resultMask;
        }

        [[nodiscard]] constexpr std::optional<std::uint32_t> signedRemainder(std::int32_t lhs,
                                                                             std::int32_t rhs,
                                                                             std::uint32_t resultMask) noexcept
        {
            if (rhs == 0)
            {
                return std::nullopt;
            }
            if (signedDivisionOverflows(lhs, rhs))
            {
                return 0U;
            }
            return std::bit_cast<std::uint32_t>(lhs % rhs) & resultMask;
        }

        [[nodiscard]] constexpr std::optional<std::uint32_t> signedModulo(std::int32_t lhs,
                                                                          std::int32_t rhs,
                                                                          std::uint32_t resultMask) noexcept
        {
            if (rhs == 0)
            {
                return std::nullopt;
            }
            if (signedDivisionOverflows(lhs, rhs))
            {
                return 0U;
            }
            std::int32_t rem = lhs % rhs;
            if (rem != 0 && ((rem < 0) != (rhs < 0)))
            {
                rem += rhs;
            }
            return std::bit_cast<std::uint32_t>(rem) & resultMask;
        }

        [[nodiscard]] constexpr std::optional<std::uint32_t> evaluateSpecConstantArithmetic(
            std::uint32_t innerOpcode, std::uint32_t lhs, std::uint32_t rhs, std::uint32_t resultMask) noexcept
        {
            const auto sLhs = std::bit_cast<std::int32_t>(lhs);
            const auto sRhs = std::bit_cast<std::int32_t>(rhs);
            switch (innerOpcode)
            {
            case spirvOpIAdd:
                return (lhs + rhs) & resultMask;
            case spirvOpISub:
                return (lhs - rhs) & resultMask;
            case spirvOpIMul:
                return (lhs * rhs) & resultMask;
            case spirvOpUDiv:
                return rhs != 0 ? std::optional<std::uint32_t>{(lhs / rhs) & resultMask} : std::nullopt;
            case spirvOpSDiv:
                return signedQuotient(sLhs, sRhs, resultMask);
            case spirvOpUMod:
                return rhs != 0 ? std::optional<std::uint32_t>{(lhs % rhs) & resultMask} : std::nullopt;
            case spirvOpSRem:
                return signedRemainder(sLhs, sRhs, resultMask);
            case spirvOpSMod:
                return signedModulo(sLhs, sRhs, resultMask);
            case spirvOpShiftRightLogical:
                return rhs < 32U ? (lhs >> rhs) : 0U;
            case spirvOpShiftRightArithmetic:
                return (rhs < 32U ? std::bit_cast<std::uint32_t>(sLhs >> rhs) : (sLhs < 0 ? ~0U : 0U)) & resultMask;
            case spirvOpShiftLeftLogical:
                return rhs < 32U ? (lhs << rhs) & resultMask : 0U;
            case spirvOpBitwiseOr:
                return (lhs | rhs) & resultMask;
            case spirvOpBitwiseXor:
                return (lhs ^ rhs) & resultMask;
            case spirvOpBitwiseAnd:
                return (lhs & rhs) & resultMask;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] constexpr std::optional<std::uint32_t> evaluateBinarySpecConstantOp(
            std::uint32_t innerOpcode, std::uint32_t lhs, std::uint32_t rhs, std::uint32_t resultMask) noexcept
        {
            if (const auto comparison = evaluateSpecConstantComparison(innerOpcode, lhs, rhs); comparison.has_value())
            {
                return *comparison ? 1U : 0U;
            }
            return evaluateSpecConstantArithmetic(innerOpcode, lhs, rhs, resultMask);
        }

        void evaluateSpecConstantSelect(const SpecConstantOpRecord& specOp, SpirvConstantTable& constants)
        {
            const auto cond = findScalarConstant(constants.scalars, specOp.operands[0]);
            if (!cond.has_value())
            {
                return;
            }
            const auto chosenId = *cond != 0 ? specOp.operands[1] : specOp.operands[2];
            if (const auto chosenComp = constants.composites.find(chosenId); chosenComp != constants.composites.end())
            {
                constants.composites[specOp.resultId] = chosenComp->second;
            }
            if (const auto chosenScalar = findScalarConstant(constants.scalars, chosenId); chosenScalar.has_value())
            {
                constants.scalars[specOp.resultId] = *chosenScalar;
            }
        }

        void evaluateSpecConstantOp(const SpecConstantOpRecord& specOp,
                                    const std::unordered_map<std::uint32_t, TypeInfo>& types,
                                    SpirvConstantTable& constants)
        {
            const auto innerOpcode = specOp.innerOpcode;
            const auto& operands = specOp.operands;
            if (innerOpcode == spirvOpVectorShuffle && operands.size() >= 2)
            {
                evaluateSpecConstantVectorShuffle(specOp, constants.composites);
                return;
            }
            if (innerOpcode == spirvOpCompositeInsert && operands.size() >= 3)
            {
                evaluateSpecConstantCompositeInsert(specOp, constants.composites);
                return;
            }
            if (innerOpcode == spirvOpCompositeExtract && operands.size() >= 2)
            {
                evaluateSpecConstantCompositeExtract(specOp, constants);
                return;
            }
            if (innerOpcode == spirvOpSelect && operands.size() == 3)
            {
                evaluateSpecConstantSelect(specOp, constants);
                return;
            }

            const auto resultMask = scalarWidthMask(scalarByteWidth(types, specOp.resultTypeId));
            std::optional<std::uint32_t> resolved;
            if (operands.size() == 1)
            {
                if (const auto operand = findScalarConstant(constants.scalars, operands[0]); operand.has_value())
                {
                    const auto operandBytes = constantByteWidth(types, constants.valueTypeIds, operands[0]);
                    resolved = evaluateUnarySpecConstantOp(innerOpcode, *operand, operandBytes, resultMask);
                }
            }
            else if (operands.size() == 2)
            {
                const auto lhs = findScalarConstant(constants.scalars, operands[0]);
                const auto rhs = findScalarConstant(constants.scalars, operands[1]);
                if (lhs.has_value() && rhs.has_value())
                {
                    resolved = evaluateBinarySpecConstantOp(innerOpcode, *lhs, *rhs, resultMask);
                }
            }
            if (resolved.has_value())
            {
                constants.scalars[specOp.resultId] = *resolved;
            }
        }

        struct SpirvSpecializationTable final
        {
            std::unordered_map<std::uint32_t, std::uint32_t> overrideBySpecId;
            std::unordered_map<std::uint32_t, std::uint32_t> specIdByResult;
            std::unordered_set<std::uint32_t> specIds;
            std::unordered_set<std::uint32_t> non32BitSpecIds;
            std::unordered_set<std::uint32_t> boolResultIds;
            std::unordered_set<std::uint32_t> scalar32ResultIds;
            std::vector<SpecConstantOpRecord> constantOps;
        };

        struct SpirvDecorationTable final
        {
            std::unordered_map<std::uint32_t, std::uint32_t> binding;
            std::unordered_map<std::uint32_t, std::uint32_t> descriptorSet;
            std::unordered_map<std::uint32_t, std::uint32_t> location;
            std::unordered_map<std::uint32_t, std::uint32_t> component;
            std::unordered_set<std::uint32_t> builtIn;
            std::unordered_set<std::uint32_t> perVertex;
            std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> memberLocations;
            std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> memberComponents;
            std::unordered_set<std::uint32_t> nonWritable;
            std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>> nonWritableMembers;
            std::unordered_set<std::uint32_t> block;
            std::unordered_set<std::uint32_t> bufferBlock;
            std::unordered_map<std::uint32_t, std::uint32_t> arrayStride;
            std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> memberOffsets;
            std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> memberMatrixStrides;
            std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>> rowMajorMembers;
            std::unordered_set<std::uint32_t> pointSizeVariables;
            std::unordered_map<std::uint32_t, std::uint32_t> pointSizeMembers;
        };

        struct SpirvVariableTable final
        {
            std::unordered_map<std::uint32_t, std::uint32_t> typeIds;
            std::unordered_map<std::uint32_t, std::uint32_t> storageClasses;
        };

        struct SpirvAccessChain final
        {
            std::uint32_t base = 0;
            std::optional<std::uint32_t> firstIndex;
        };

        struct SpirvEntryPointRecord final
        {
            std::optional<std::uint32_t> functionId;
            std::string name;
            ShaderStage stage = ShaderStage::none;
            std::unordered_set<std::uint32_t> interfaceIds;
        };

        struct SpirvLocalSizeTable final
        {
            std::unordered_map<std::uint32_t, ShaderWorkgroupSize> literalByEntry;
            std::unordered_map<std::uint32_t, std::array<std::uint32_t, 3>> constantIdsByEntry;
        };

        struct SpirvModule final
        {
            std::unordered_map<std::uint32_t, TypeInfo> types;
            SpirvConstantTable constants;
            SpirvSpecializationTable specialization;
            SpirvDecorationTable decorations;
            SpirvVariableTable variables;
            SpirvEntryPointRecord entryPoint;
            SpirvLocalSizeTable localSizes;
            std::unordered_map<std::uint32_t, FunctionReflectionInfo> functions;
            std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> derivedFrom;
            std::unordered_set<std::uint32_t> nonSemanticExtSets;
            std::unordered_map<std::uint32_t, SpirvAccessChain> accessChains;
        };

        [[nodiscard]] constexpr bool hasTruncatedTypeOperands(const SpirvInstruction& instruction) noexcept
        {
            const auto opcode = instruction.opcode;
            const bool lacksResultId = (opcode == spirvOpTypeSampler || opcode == spirvOpTypeAccelerationStructure ||
                                        opcode == spirvOpTypeStruct) &&
                                       instruction.wordCount < 2;
            const bool lacksImageType = opcode == spirvOpTypeSampledImage && instruction.wordCount < 3;
            return lacksResultId || lacksImageType;
        }

        [[nodiscard]] std::size_t boundedStringLength(const char* text, std::size_t maxBytes) noexcept
        {
            std::size_t length = 0;
            while (length < maxBytes && text[length] != '\0')
            {
                ++length;
            }
            return length;
        }

        [[nodiscard]] Status recordEntryPoint(const SpirvInstruction& instruction,
                                              std::string_view entryPoint,
                                              ShaderStage stage,
                                              SpirvEntryPointRecord& record)
        {
            if (instruction.opcode != spirvOpEntryPoint)
            {
                return {};
            }
            if (instruction.wordCount < 4)
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, "SPIR-V OpEntryPoint is truncated."});
            }
            const auto model = shaderStageFromExecutionModel(instruction.words[1]);
            const char* name = reinterpret_cast<const char*>(instruction.words + 3);
            const auto remaining = (instruction.wordCount - 3) * sizeof(std::uint32_t);
            const auto nameLength = boundedStringLength(name, remaining);
            if (nameLength == remaining)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, "SPIR-V OpEntryPoint name is unterminated."});
            }
            const std::size_t nameWords = (nameLength / sizeof(std::uint32_t)) + 1U;
            if (3U + nameWords > instruction.wordCount)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, "SPIR-V OpEntryPoint operands are truncated."});
            }
            const bool isRequestedEntryPoint =
                model.has_value() && *model == stage && std::string_view(name, nameLength) == entryPoint;
            if (!isRequestedEntryPoint)
            {
                return {};
            }
            if (record.functionId.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The SPIR-V module declares duplicate entry points for the requested stage and name.",
                });
            }
            record.functionId = instruction.words[2];
            record.name = std::string(entryPoint);
            record.stage = *model;
            for (std::size_t ifaceIdx = 3U + nameWords; ifaceIdx < instruction.wordCount; ++ifaceIdx)
            {
                record.interfaceIds.insert(instruction.words[ifaceIdx]);
            }
            return {};
        }

        void recordNonSemanticExtInstImport(const SpirvInstruction& instruction,
                                            std::unordered_set<std::uint32_t>& nonSemanticExtSets)
        {
            if (instruction.opcode != spirvOpExtInstImport || instruction.wordCount < 3)
            {
                return;
            }
            const char* setName = reinterpret_cast<const char*>(instruction.words + 2);
            const auto maxBytes = (instruction.wordCount - 2) * sizeof(std::uint32_t);
            const auto nameLen = boundedStringLength(setName, maxBytes);
            if (nameLen < maxBytes && std::string_view(setName, nameLen).starts_with("NonSemantic."))
            {
                nonSemanticExtSets.insert(instruction.words[1]);
            }
        }

        void recordLocalSizeExecutionMode(const SpirvInstruction& instruction, SpirvLocalSizeTable& localSizes)
        {
            const bool isExecutionMode =
                instruction.opcode == spirvOpExecutionMode || instruction.opcode == spirvOpExecutionModeId;
            if (!isExecutionMode || instruction.wordCount < 6)
            {
                return;
            }
            const auto entryPointId = instruction.words[1];
            const auto mode = instruction.words[2];
            if (instruction.opcode == spirvOpExecutionMode && mode == spirvExecutionModeLocalSize)
            {
                localSizes.literalByEntry[entryPointId] =
                    ShaderWorkgroupSize{instruction.words[3], instruction.words[4], instruction.words[5]};
            }
            else if (instruction.opcode == spirvOpExecutionModeId && mode == spirvExecutionModeLocalSizeId)
            {
                localSizes.constantIdsByEntry[entryPointId] =
                    std::array<std::uint32_t, 3>{instruction.words[3], instruction.words[4], instruction.words[5]};
            }
        }

        void recordDecoration(const SpirvInstruction& instruction,
                              SpirvDecorationTable& decorations,
                              SpirvSpecializationTable& specialization)
        {
            if (instruction.opcode != spirvOpDecorate || instruction.wordCount < 3)
            {
                return;
            }
            const auto target = instruction.words[1];
            const auto decoration = instruction.words[2];
            const bool hasLiteral = instruction.wordCount >= 4;
            if (decoration == spirvDecorationBinding && hasLiteral)
            {
                decorations.binding[target] = instruction.words[3];
            }
            else if (decoration == spirvDecorationDescriptorSet && hasLiteral)
            {
                decorations.descriptorSet[target] = instruction.words[3];
            }
            else if (decoration == spirvDecorationLocation && hasLiteral)
            {
                decorations.location[target] = instruction.words[3];
            }
            else if (decoration == spirvDecorationComponent && hasLiteral)
            {
                decorations.component[target] = instruction.words[3];
            }
            else if (decoration == spirvDecorationBuiltIn)
            {
                decorations.builtIn.insert(target);
                if (hasLiteral && instruction.words[3] == spirvBuiltInPointSize)
                {
                    decorations.pointSizeVariables.insert(target);
                }
            }
            else if (decoration == spirvDecorationPerVertexKHR)
            {
                decorations.perVertex.insert(target);
            }
            else if (decoration == spirvDecorationSpecId && hasLiteral)
            {
                specialization.specIds.insert(instruction.words[3]);
                specialization.specIdByResult[target] = instruction.words[3];
            }
            else if (decoration == spirvDecorationNonWritable)
            {
                decorations.nonWritable.insert(target);
            }
            else if (decoration == spirvDecorationBlock)
            {
                decorations.block.insert(target);
            }
            else if (decoration == spirvDecorationBufferBlock)
            {
                decorations.bufferBlock.insert(target);
            }
            else if (decoration == spirvDecorationArrayStride && hasLiteral)
            {
                decorations.arrayStride[target] = instruction.words[3];
            }
        }

        void recordMemberDecoration(const SpirvInstruction& instruction, SpirvDecorationTable& decorations)
        {
            if (instruction.opcode != spirvOpMemberDecorate || instruction.wordCount < 4)
            {
                return;
            }
            const auto structId = instruction.words[1];
            const auto member = instruction.words[2];
            const auto decoration = instruction.words[3];
            const bool hasLiteral = instruction.wordCount >= 5;
            if (decoration == spirvDecorationOffset && hasLiteral)
            {
                decorations.memberOffsets[structId][member] = instruction.words[4];
            }
            else if (decoration == spirvDecorationRowMajor)
            {
                decorations.rowMajorMembers[structId].insert(member);
            }
            else if (decoration == spirvDecorationNonWritable)
            {
                decorations.nonWritableMembers[structId].insert(member);
            }
            else if (decoration == spirvDecorationMatrixStride && hasLiteral)
            {
                decorations.memberMatrixStrides[structId][member] = instruction.words[4];
            }
            else if (decoration == spirvDecorationLocation && hasLiteral)
            {
                decorations.memberLocations[structId][member] = instruction.words[4];
            }
            else if (decoration == spirvDecorationComponent && hasLiteral)
            {
                decorations.memberComponents[structId][member] = instruction.words[4];
            }
            else if (decoration == spirvDecorationBuiltIn && hasLiteral &&
                     instruction.words[4] == spirvBuiltInPointSize)
            {
                decorations.pointSizeMembers[structId] = member;
            }
        }

        void recordType(const SpirvInstruction& instruction,
                        std::unordered_map<std::uint32_t, TypeInfo>& types,
                        const std::unordered_map<std::uint32_t, std::uint32_t>& scalarConstants)
        {
            const auto* words = instruction.words;
            const auto wordCount = instruction.wordCount;
            switch (instruction.opcode)
            {
            case spirvOpTypeInt:
            case spirvOpTypeFloat:
                if (wordCount >= 3)
                {
                    auto& type = types[words[1]];
                    type.kind = TypeInfo::Kind::scalar;
                    type.scalarBytes = words[2] / 8;
                    if (instruction.opcode == spirvOpTypeFloat)
                    {
                        type.scalarKind = TypeInfo::ScalarKind::floatingPoint;
                    }
                    else if (wordCount >= 4)
                    {
                        type.scalarKind =
                            words[3] != 0 ? TypeInfo::ScalarKind::signedInteger : TypeInfo::ScalarKind::unsignedInteger;
                    }
                }
                break;
            case spirvOpTypeVector:
            case spirvOpTypeMatrix:
                if (wordCount >= 4)
                {
                    auto& type = types[words[1]];
                    type.kind =
                        instruction.opcode == spirvOpTypeVector ? TypeInfo::Kind::vector : TypeInfo::Kind::matrix;
                    type.element = words[2];
                    type.arrayLength = words[3];
                }
                break;
            case spirvOpTypeImage:
                if (wordCount >= 3)
                {
                    auto& type = types[words[1]];
                    type.kind = TypeInfo::Kind::image;
                    if (wordCount >= 8)
                    {
                        type.sampled = words[7];
                    }
                }
                break;
            case spirvOpTypeSampler:
                types[words[1]].kind = TypeInfo::Kind::sampler;
                break;
            case spirvOpTypeSampledImage: {
                auto& type = types[words[1]];
                type.kind = TypeInfo::Kind::sampledImage;
                type.element = words[2];
                break;
            }
            case spirvOpTypeAccelerationStructure:
                types[words[1]].kind = TypeInfo::Kind::acceleration;
                break;
            case spirvOpTypePointer:
                if (wordCount >= 4)
                {
                    auto& type = types[words[1]];
                    type.kind = TypeInfo::Kind::pointer;
                    type.storageClass = words[2];
                    type.pointee = words[3];
                }
                break;
            case spirvOpTypeArray:
                if (wordCount >= 4)
                {
                    auto& type = types[words[1]];
                    type.kind = TypeInfo::Kind::array;
                    type.element = words[2];
                    type.arrayLengthId = words[3];
                    const auto lengthIt = scalarConstants.find(words[3]);
                    type.arrayLength = lengthIt == scalarConstants.end() ? 1 : lengthIt->second;
                }
                break;
            case spirvOpTypeRuntimeArray:
                if (wordCount >= 3)
                {
                    auto& type = types[words[1]];
                    type.kind = TypeInfo::Kind::runtimeArray;
                    type.element = words[2];
                    type.runtimeArray = true;
                    type.arrayLength = unboundedBindingCount;
                }
                break;
            case spirvOpTypeStruct: {
                auto& type = types[words[1]];
                type.kind = TypeInfo::Kind::structure;
                type.members.assign(words + 2, words + wordCount);
                break;
            }
            default:
                break;
            }
        }

        [[nodiscard]] constexpr bool isConstantDeclaration(std::uint32_t opcode) noexcept
        {
            return opcode == spirvOpConstantTrue || opcode == spirvOpConstantFalse || opcode == spirvOpConstantNull ||
                   opcode == spirvOpConstant || opcode == spirvOpConstantComposite ||
                   opcode == spirvOpSpecConstantComposite;
        }

        void recordConstant(const SpirvInstruction& instruction,
                            SpirvConstantTable& constants,
                            std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& derivedFrom)
        {
            if (!isConstantDeclaration(instruction.opcode) || instruction.wordCount < 3)
            {
                return;
            }
            const auto typeId = instruction.words[1];
            const auto resultId = instruction.words[2];
            switch (instruction.opcode)
            {
            case spirvOpConstantTrue:
            case spirvOpConstantFalse:
                constants.valueTypeIds[resultId] = typeId;
                constants.scalars[resultId] = instruction.opcode == spirvOpConstantTrue ? 1U : 0U;
                break;
            case spirvOpConstantNull:
                constants.valueTypeIds[resultId] = typeId;
                constants.scalars[resultId] = 0U;
                break;
            case spirvOpConstant:
                if (instruction.wordCount >= 4)
                {
                    constants.valueTypeIds[resultId] = typeId;
                    constants.scalars[resultId] = instruction.words[3];
                }
                break;
            case spirvOpConstantComposite:
            case spirvOpSpecConstantComposite:
                constants.valueTypeIds[resultId] = typeId;
                constants.composites[resultId].assign(instruction.words + 3, instruction.words + instruction.wordCount);
                for (std::uint32_t i = 3; i < instruction.wordCount; ++i)
                {
                    derivedFrom[resultId].push_back(instruction.words[i]);
                }
                break;
            default:
                break;
            }
        }

        [[nodiscard]] std::optional<std::uint32_t> findSpecializationOverride(
            const SpirvSpecializationTable& specialization, std::uint32_t resultId)
        {
            const auto specIt = specialization.specIdByResult.find(resultId);
            if (specIt == specialization.specIdByResult.end())
            {
                return std::nullopt;
            }
            const auto overrideIt = specialization.overrideBySpecId.find(specIt->second);
            if (overrideIt == specialization.overrideBySpecId.end())
            {
                return std::nullopt;
            }
            return overrideIt->second;
        }

        void recordSpecConstant(const SpirvInstruction& instruction,
                                const std::unordered_map<std::uint32_t, TypeInfo>& types,
                                SpirvSpecializationTable& specialization,
                                SpirvConstantTable& constants)
        {
            const bool isBoolean =
                instruction.opcode == spirvOpSpecConstantTrue || instruction.opcode == spirvOpSpecConstantFalse;
            if ((!isBoolean && instruction.opcode != spirvOpSpecConstant) || instruction.wordCount < 3)
            {
                return;
            }
            const auto resultTypeId = instruction.words[1];
            const auto resultId = instruction.words[2];
            constants.valueTypeIds[resultId] = resultTypeId;
            const auto specIt = specialization.specIdByResult.find(resultId);
            const bool hasSpecId = specIt != specialization.specIdByResult.end();
            if (hasSpecId)
            {
                specialization.specIds.insert(specIt->second);
            }
            const auto overrideBits = findSpecializationOverride(specialization, resultId);
            const bool hasOneLiteralWord = instruction.wordCount == 4;
            if (isBoolean)
            {
                specialization.boolResultIds.insert(resultId);
                const bool value =
                    overrideBits.has_value() ? *overrideBits != 0 : instruction.opcode == spirvOpSpecConstantTrue;
                constants.scalars[resultId] = value ? 1U : 0U;
            }
            else if (hasOneLiteralWord && scalarByteWidth(types, resultTypeId) == sizeof(std::uint32_t))
            {
                specialization.scalar32ResultIds.insert(resultId);
                constants.scalars[resultId] = overrideBits.value_or(instruction.words[3]);
            }
            else
            {
                if (hasSpecId)
                {
                    specialization.non32BitSpecIds.insert(specIt->second);
                }
                if (hasOneLiteralWord)
                {
                    constants.scalars[resultId] = instruction.words[3];
                }
            }
        }

        void recordSpecConstantOp(const SpirvInstruction& instruction, SpirvModule& module)
        {
            if (instruction.opcode != spirvOpSpecConstantOp || instruction.wordCount < 4)
            {
                return;
            }
            const auto* words = instruction.words;
            const auto operandCount = instruction.wordCount - 4;
            module.constants.valueTypeIds[words[2]] = words[1];
            recordSpecConstantOpDependencies(words[2], words[3], words + 4, operandCount, module.derivedFrom);
            SpecConstantOpRecord specOp{
                .resultTypeId = words[1],
                .resultId = words[2],
                .innerOpcode = words[3],
                .operands = std::vector<std::uint32_t>(words + 4, words + instruction.wordCount),
            };
            evaluateSpecConstantOp(specOp, module.types, module.constants);
            module.specialization.constantOps.push_back(std::move(specOp));
        }

        [[nodiscard]] Status recordModuleDeclaration(const SpirvInstruction& instruction,
                                                     std::string_view entryPoint,
                                                     ShaderStage stage,
                                                     SpirvModule& module)
        {
            if (auto recorded = recordEntryPoint(instruction, entryPoint, stage, module.entryPoint); !recorded)
            {
                return recorded;
            }
            recordNonSemanticExtInstImport(instruction, module.nonSemanticExtSets);
            recordLocalSizeExecutionMode(instruction, module.localSizes);
            recordDecoration(instruction, module.decorations, module.specialization);
            recordMemberDecoration(instruction, module.decorations);
            recordType(instruction, module.types, module.constants.scalars);
            recordConstant(instruction, module.constants, module.derivedFrom);
            recordSpecConstant(instruction, module.types, module.specialization, module.constants);
            recordSpecConstantOp(instruction, module);
            return {};
        }

        [[nodiscard]] constexpr std::uint32_t declaredIdOperandCount(std::uint32_t opcode) noexcept
        {
            if ((opcode >= spirvOpTypeVoid && opcode <= spirvOpTypeForwardPointer) ||
                opcode == spirvOpTypeAccelerationStructure || opcode == spirvOpExtInstImport || opcode == spirvOpLabel)
            {
                return 1;
            }
            const bool declaresTypedResult = isConstantDeclaration(opcode) || opcode == spirvOpSpecConstantTrue ||
                                             opcode == spirvOpSpecConstantFalse || opcode == spirvOpSpecConstant ||
                                             opcode == spirvOpSpecConstantOp || opcode == spirvOpVariable ||
                                             opcode == spirvOpFunction || opcode == spirvOpFunctionParameter;
            return declaresTypedResult ? 2 : 0;
        }

        [[nodiscard]] bool declaresIdOutsideBound(const SpirvInstruction& instruction, std::uint32_t idBound) noexcept
        {
            const auto idCount = std::min(declaredIdOperandCount(instruction.opcode), instruction.wordCount - 1);
            for (std::uint32_t i = 1; i <= idCount; ++i)
            {
                if (instruction.words[i] == 0 || instruction.words[i] >= idBound)
                {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] constexpr bool isBlockTerminator(std::uint32_t opcode) noexcept
        {
            return (opcode >= spirvOpBranch && opcode <= spirvOpUnreachable) || opcode == spirvOpTerminateInvocation ||
                   opcode == spirvOpIgnoreIntersectionKHR || opcode == spirvOpTerminateRayKHR ||
                   opcode == spirvOpEmitMeshTasksEXT;
        }

        [[nodiscard]] constexpr bool mayTrailBlockTerminator(std::uint32_t opcode) noexcept
        {
            return opcode == spirvOpLine || opcode == spirvOpNoLine || opcode == spirvOpExtInst ||
                   opcode == spirvOpExtInstWithForwardRefsKHR;
        }

        struct SpirvFunctionScope final
        {
            std::optional<std::uint32_t> functionId;
            bool hasBlocks = false;
            std::uint32_t lastBodyOpcode = 0;
        };

        [[nodiscard]] Status openFunction(const SpirvInstruction& instruction,
                                          std::unordered_map<std::uint32_t, FunctionReflectionInfo>& functions,
                                          SpirvFunctionScope& scope)
        {
            if (scope.functionId.has_value())
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, "A SPIR-V OpFunction begins inside another function."});
            }
            if (instruction.wordCount < 5)
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, "SPIR-V OpFunction is truncated."});
            }
            scope = SpirvFunctionScope{.functionId = instruction.words[2]};
            functions.try_emplace(instruction.words[2]);
            return {};
        }

        [[nodiscard]] Status closeFunction(SpirvFunctionScope& scope)
        {
            if (!scope.functionId.has_value())
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, "A SPIR-V OpFunctionEnd appears outside a function."});
            }
            if (scope.hasBlocks && !isBlockTerminator(scope.lastBodyOpcode))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The last block of a SPIR-V function does not end with a termination instruction.",
                });
            }
            scope = {};
            return {};
        }

        void trackFunctionBody(const SpirvInstruction& instruction, SpirvFunctionScope& scope)
        {
            if (!scope.functionId.has_value())
            {
                return;
            }
            scope.hasBlocks = scope.hasBlocks || instruction.opcode == spirvOpLabel;
            if (!mayTrailBlockTerminator(instruction.opcode))
            {
                scope.lastBodyOpcode = instruction.opcode;
            }
        }

        [[nodiscard]] Status trackEnclosingFunction(
            const SpirvInstruction& instruction,
            std::unordered_map<std::uint32_t, FunctionReflectionInfo>& functions,
            SpirvFunctionScope& scope)
        {
            switch (instruction.opcode)
            {
            case spirvOpFunction:
                return openFunction(instruction, functions, scope);
            case spirvOpFunctionEnd:
                return closeFunction(scope);
            default:
                trackFunctionBody(instruction, scope);
                return {};
            }
        }

        struct SpirvDefinitionRequirements final
        {
            std::unordered_set<std::uint32_t> entryPointFunctionIds;
            std::unordered_set<std::uint32_t> namedOrDecoratedIds;
        };

        [[nodiscard]] constexpr bool namesOrDecoratesTarget(std::uint32_t opcode) noexcept
        {
            return opcode == spirvOpName || opcode == spirvOpMemberName || opcode == spirvOpDecorate ||
                   opcode == spirvOpMemberDecorate || opcode == spirvOpDecorateId || opcode == spirvOpDecorateString ||
                   opcode == spirvOpMemberDecorateString;
        }

        [[nodiscard]] constexpr bool refersToIdsWithoutDefiningOne(std::uint32_t opcode) noexcept
        {
            return namesOrDecoratesTarget(opcode) || opcode == spirvOpGroupDecorate ||
                   opcode == spirvOpGroupMemberDecorate || opcode == spirvOpEntryPoint ||
                   opcode == spirvOpExecutionMode || opcode == spirvOpExecutionModeId;
        }

        [[nodiscard]] Status recordDefinitionRequirement(const SpirvInstruction& instruction,
                                                         SpirvDefinitionRequirements& requirements)
        {
            const auto opcode = instruction.opcode;
            if (opcode == spirvOpEntryPoint && instruction.wordCount >= 3)
            {
                requirements.entryPointFunctionIds.insert(instruction.words[2]);
            }
            else if (opcode == spirvOpExecutionMode || opcode == spirvOpExecutionModeId)
            {
                if (instruction.wordCount < 3)
                {
                    return std::unexpected(
                        Error{ErrorCode::invalidArgument, 0, "SPIR-V OpExecutionMode is truncated."});
                }
                requirements.entryPointFunctionIds.insert(instruction.words[1]);
            }
            else if (namesOrDecoratesTarget(opcode) && instruction.wordCount >= 2)
            {
                requirements.namedOrDecoratedIds.insert(instruction.words[1]);
            }
            return {};
        }

        [[nodiscard]] Status validateFunctionDefinitions(
            const SpirvFunctionScope& scope,
            const std::unordered_map<std::uint32_t, FunctionReflectionInfo>& functions,
            const std::unordered_set<std::uint32_t>& entryPointFunctionIds)
        {
            if (scope.functionId.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The SPIR-V module ends inside a function; it is truncated or lacks an OpFunctionEnd.",
                });
            }
            const bool declaresUndefinedEntryPoint =
                std::ranges::any_of(entryPointFunctionIds, [&](std::uint32_t entryFunctionId) {
                    return !functions.contains(entryFunctionId);
                });
            if (declaresUndefinedEntryPoint)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A SPIR-V OpEntryPoint or OpExecutionMode names a function the module does not define; it may be "
                    "truncated.",
                });
            }
            for (const auto& [functionId, function] : functions)
            {
                static_cast<void>(functionId);
                const bool callsUndefinedFunction = std::ranges::any_of(
                    function.callees, [&](std::uint32_t calleeId) { return !functions.contains(calleeId); });
                if (callsUndefinedFunction)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A SPIR-V OpFunctionCall targets a function the module does not define; it may be truncated.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Status validateNamedAndDecoratedIds(std::span<const std::uint32_t> words,
                                                          std::unordered_set<std::uint32_t> undefinedIds)
        {
            std::size_t index = spirvHeaderWordCount;
            while (!undefinedIds.empty() && index < words.size())
            {
                auto instruction = decodeInstruction(words, index);
                if (!instruction)
                {
                    return std::unexpected(std::move(instruction.error()));
                }
                if (!refersToIdsWithoutDefiningOne(instruction->opcode))
                {
                    const auto resultOperandCount = std::min(instruction->wordCount - 1U, 2U);
                    for (std::uint32_t operand = 1; operand <= resultOperandCount; ++operand)
                    {
                        undefinedIds.erase(instruction->words[operand]);
                    }
                }
                index += instruction->wordCount;
            }
            if (!undefinedIds.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A SPIR-V name or decoration targets an ID the module does not define; it may be truncated.",
                });
            }
            return {};
        }

        void recordVariable(const SpirvInstruction& instruction, bool insideFunction, SpirvModule& module)
        {
            if (instruction.opcode != spirvOpVariable || instruction.wordCount < 4)
            {
                return;
            }
            const auto variableId = instruction.words[2];
            module.variables.typeIds[variableId] = instruction.words[1];
            module.variables.storageClasses[variableId] = instruction.words[3];
            if (instruction.wordCount >= 5 && !insideFunction)
            {
                module.derivedFrom[variableId].push_back(instruction.words[4]);
            }
        }

        void recordPointerWrite(const SpirvInstruction& instruction,
                                FunctionReflectionInfo& function,
                                std::unordered_map<std::uint32_t, SpirvAccessChain>& accessChains)
        {
            const auto* words = instruction.words;
            const auto wordCount = instruction.wordCount;
            switch (instruction.opcode)
            {
            case spirvOpStore:
            case spirvOpCopyMemory:
            case spirvOpCopyMemorySized:
                if (wordCount >= 3)
                {
                    function.writtenPointers.push_back(words[1]);
                }
                return;
            case spirvOpFunctionCall:
                for (std::uint32_t i = 4; i < wordCount; ++i)
                {
                    function.writtenPointers.push_back(words[i]);
                }
                return;
            case spirvOpAccessChain:
            case spirvOpInBoundsAccessChain:
                if (wordCount >= 4)
                {
                    accessChains[words[2]] = SpirvAccessChain{
                        .base = words[3],
                        .firstIndex = wordCount >= 5 ? std::optional<std::uint32_t>{words[4]} : std::nullopt,
                    };
                }
                return;
            default:
                return;
            }
        }

        [[nodiscard]] Result<SpirvModule> scanSpirvModule(
            std::span<const std::uint32_t> words,
            std::string_view entryPoint,
            ShaderStage stage,
            std::span<const SpecializationConstant> specializationOverrides)
        {
            SpirvModule module;
            module.specialization.overrideBySpecId.reserve(specializationOverrides.size());
            for (const auto& spec : specializationOverrides)
            {
                module.specialization.overrideBySpecId.emplace(spec.id, spec.valueBits);
            }

            const auto idBound = words[spirvHeaderIdBoundWord];
            SpirvFunctionScope functionScope;
            SpirvDefinitionRequirements requiredDefinitions;
            std::size_t index = spirvHeaderWordCount;
            while (index < words.size())
            {
                auto instruction = decodeInstruction(words, index);
                if (!instruction)
                {
                    return std::unexpected(std::move(instruction.error()));
                }
                if (hasTruncatedTypeOperands(*instruction))
                {
                    return std::unexpected(Error{ErrorCode::invalidArgument, 0, "SPIR-V type operands are truncated."});
                }
                if (declaresIdOutsideBound(*instruction, idBound))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A SPIR-V declaration uses an ID of zero or one not below the module header's ID bound.",
                    });
                }
                if (auto recorded = recordModuleDeclaration(*instruction, entryPoint, stage, module); !recorded)
                {
                    return std::unexpected(std::move(recorded.error()));
                }
                if (auto required = recordDefinitionRequirement(*instruction, requiredDefinitions); !required)
                {
                    return std::unexpected(std::move(required.error()));
                }
                if (auto tracked = trackEnclosingFunction(*instruction, module.functions, functionScope); !tracked)
                {
                    return std::unexpected(std::move(tracked.error()));
                }
                recordVariable(*instruction, functionScope.functionId.has_value(), module);
                if (functionScope.functionId.has_value())
                {
                    auto& function = module.functions[*functionScope.functionId];
                    recordFunctionInstruction(*instruction, function, module.derivedFrom, module.nonSemanticExtSets);
                    recordPointerWrite(*instruction, function, module.accessChains);
                }
                index += instruction->wordCount;
            }
            if (auto defined = validateFunctionDefinitions(
                    functionScope, module.functions, requiredDefinitions.entryPointFunctionIds);
                !defined)
            {
                return std::unexpected(std::move(defined.error()));
            }
            if (auto defined = validateNamedAndDecoratedIds(words, std::move(requiredDefinitions.namedOrDecoratedIds));
                !defined)
            {
                return std::unexpected(std::move(defined.error()));
            }
            return module;
        }

        [[nodiscard]] bool declaresRequestedEntryPoint(const SpirvModule& module,
                                                       std::string_view entryPoint,
                                                       ShaderStage stage)
        {
            const auto& found = module.entryPoint;
            return found.functionId.has_value() && found.name == entryPoint && found.stage == stage &&
                   module.functions.contains(*found.functionId);
        }

        void applySpecializationOverrides(const SpirvSpecializationTable& specialization,
                                          std::unordered_map<std::uint32_t, std::uint32_t>& scalarConstants)
        {
            for (const auto& [resultId, specId] : specialization.specIdByResult)
            {
                const auto overrideIt = specialization.overrideBySpecId.find(specId);
                if (overrideIt == specialization.overrideBySpecId.end())
                {
                    continue;
                }
                if (specialization.boolResultIds.contains(resultId))
                {
                    scalarConstants[resultId] = overrideIt->second != 0 ? 1U : 0U;
                }
                else if (specialization.scalar32ResultIds.contains(resultId))
                {
                    scalarConstants[resultId] = overrideIt->second;
                }
            }
        }

        void resolveArrayLengths(std::unordered_map<std::uint32_t, TypeInfo>& types,
                                 const std::unordered_map<std::uint32_t, std::uint32_t>& scalarConstants)
        {
            for (auto& [typeId, type] : types)
            {
                static_cast<void>(typeId);
                if (type.kind != TypeInfo::Kind::array || type.arrayLengthId == 0)
                {
                    continue;
                }
                if (const auto lengthIt = scalarConstants.find(type.arrayLengthId); lengthIt != scalarConstants.end())
                {
                    type.arrayLength = lengthIt->second;
                }
            }
        }

        void applySpecialization(SpirvModule& module)
        {
            applySpecializationOverrides(module.specialization, module.constants.scalars);
            for (const auto& specOp : module.specialization.constantOps)
            {
                evaluateSpecConstantOp(specOp, module.types, module.constants);
            }
            resolveArrayLengths(module.types, module.constants.scalars);
        }

        [[nodiscard]] Result<std::optional<ShaderWorkgroupSize>> resolveWorkgroupSize(const SpirvModule& module)
        {
            const auto entryPointId = *module.entryPoint.functionId;
            const auto& constants = module.constants.scalars;
            const auto& constantIdsByEntry = module.localSizes.constantIdsByEntry;
            const auto& literalByEntry = module.localSizes.literalByEntry;
            std::optional<ShaderWorkgroupSize> workgroup;
            if (const auto sizeIdsIt = constantIdsByEntry.find(entryPointId); sizeIdsIt != constantIdsByEntry.end())
            {
                const auto& ids = sizeIdsIt->second;
                const auto xIt = constants.find(ids[0]);
                const auto yIt = constants.find(ids[1]);
                const auto zIt = constants.find(ids[2]);
                if (xIt == constants.end() || yIt == constants.end() || zIt == constants.end())
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "The SPIR-V LocalSizeId execution mode references an unresolved constant.",
                    });
                }
                workgroup = ShaderWorkgroupSize{xIt->second, yIt->second, zIt->second};
            }
            else if (const auto sizeIt = literalByEntry.find(entryPointId); sizeIt != literalByEntry.end())
            {
                workgroup = sizeIt->second;
            }
            if (workgroup.has_value() && (workgroup->x == 0 || workgroup->y == 0 || workgroup->z == 0))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "SPIR-V workgroup dimensions must be non-zero.",
                });
            }
            return workgroup;
        }

        [[nodiscard]] std::unordered_set<std::uint32_t> collectReachableFunctions(
            const std::unordered_map<std::uint32_t, FunctionReflectionInfo>& functions, std::uint32_t entryPointId)
        {
            std::unordered_set<std::uint32_t> reachableFunctions{entryPointId};
            std::vector<std::uint32_t> worklist{entryPointId};
            while (!worklist.empty())
            {
                const auto fnId = worklist.back();
                worklist.pop_back();
                const auto fnIt = functions.find(fnId);
                if (fnIt == functions.end())
                {
                    continue;
                }
                for (const auto calleeId : fnIt->second.callees)
                {
                    if (reachableFunctions.insert(calleeId).second)
                    {
                        worklist.push_back(calleeId);
                    }
                }
            }
            return reachableFunctions;
        }

        [[nodiscard]] std::unordered_set<std::uint32_t> collectUsedIds(
            const SpirvModule& module, const std::unordered_set<std::uint32_t>& reachableFunctions)
        {
            std::unordered_set<std::uint32_t> usedIds;
            std::vector<std::uint32_t> worklist;
            const auto markUsed = [&](std::uint32_t id) {
                if (usedIds.insert(id).second)
                {
                    worklist.push_back(id);
                }
            };
            for (const auto fnId : reachableFunctions)
            {
                const auto fnIt = module.functions.find(fnId);
                if (fnIt == module.functions.end())
                {
                    continue;
                }
                std::ranges::for_each(fnIt->second.referencedIds, markUsed);
            }
            while (!worklist.empty())
            {
                const auto currentId = worklist.back();
                worklist.pop_back();
                const auto derivedIt = module.derivedFrom.find(currentId);
                if (derivedIt == module.derivedFrom.end())
                {
                    continue;
                }
                std::ranges::for_each(derivedIt->second, markUsed);
            }
            return usedIds;
        }

        [[nodiscard]] bool isEntryPointVertexInput(const SpirvModule& module,
                                                   const std::unordered_set<std::uint32_t>& usedIds,
                                                   ShaderStage stage,
                                                   std::uint32_t variableId)
        {
            return stage == ShaderStage::vertex && module.decorations.location.contains(variableId) &&
                   (module.entryPoint.interfaceIds.contains(variableId) || usedIds.contains(variableId));
        }

        struct EntryPointUsage final
        {
            std::unordered_set<std::uint32_t> usedIds;
            bool writesPointSize = false;
        };

        [[nodiscard]] bool pointerWritesPointSize(const SpirvModule& module, std::uint32_t pointer)
        {
            std::optional<std::uint32_t> memberIndexId;
            auto root = pointer;
            for (std::size_t depth = 0; depth <= module.accessChains.size(); ++depth)
            {
                const auto chain = module.accessChains.find(root);
                if (chain == module.accessChains.end())
                {
                    break;
                }
                if (chain->second.firstIndex.has_value())
                {
                    memberIndexId = chain->second.firstIndex;
                }
                root = chain->second.base;
            }
            const auto storage = module.variables.storageClasses.find(root);
            const auto variableType = module.variables.typeIds.find(root);
            if (storage == module.variables.storageClasses.end() || storage->second != spirvStorageClassOutput ||
                variableType == module.variables.typeIds.end())
            {
                return false;
            }
            if (module.decorations.pointSizeVariables.contains(root))
            {
                return true;
            }
            const auto pointerType = module.types.find(variableType->second);
            const auto pointeeType = pointerType != module.types.end() ? pointerType->second.pointee : 0U;
            const auto member = module.decorations.pointSizeMembers.find(pointeeType);
            if (member == module.decorations.pointSizeMembers.end())
            {
                return false;
            }
            if (!memberIndexId.has_value())
            {
                return true;
            }
            const auto index = module.constants.scalars.find(*memberIndexId);
            return index != module.constants.scalars.end() && index->second == member->second;
        }

        [[nodiscard]] bool entryPointWritesPointSize(const SpirvModule& module,
                                                     const std::unordered_set<std::uint32_t>& reachableFunctions)
        {
            return std::ranges::any_of(reachableFunctions, [&](std::uint32_t fnId) {
                const auto fnIt = module.functions.find(fnId);
                return fnIt != module.functions.end() &&
                       std::ranges::any_of(fnIt->second.writtenPointers, [&](std::uint32_t pointer) {
                           return pointerWritesPointSize(module, pointer);
                       });
            });
        }

        [[nodiscard]] EntryPointUsage analyzeEntryPointUsage(const SpirvModule& module)
        {
            const auto reachableFunctions = collectReachableFunctions(module.functions, *module.entryPoint.functionId);
            return EntryPointUsage{
                .usedIds = collectUsedIds(module, reachableFunctions),
                .writesPointSize = entryPointWritesPointSize(module, reachableFunctions),
            };
        }

        [[nodiscard]] std::uint32_t variablePointeeTypeId(const SpirvModule& module, std::uint32_t variableId)
        {
            const auto typeId = module.variables.typeIds.at(variableId);
            const auto pointerIt = module.types.find(typeId);
            return pointerIt != module.types.end() ? pointerIt->second.pointee : typeId;
        }

        constexpr std::size_t maxSpirvTypeNestingDepth = 255;

        struct SpirvMatrixLayout final
        {
            std::uint32_t stride = 0;
            bool rowMajor = false;

            friend bool operator==(const SpirvMatrixLayout&, const SpirvMatrixLayout&) = default;
        };

        struct SpirvTypeExtentKey final
        {
            std::uint32_t id = 0;
            SpirvMatrixLayout matrixLayout = {};

            friend bool operator==(const SpirvTypeExtentKey&, const SpirvTypeExtentKey&) = default;
        };

        struct SpirvTypeExtentKeyHash final
        {
            [[nodiscard]] std::size_t operator()(const SpirvTypeExtentKey& key) const noexcept
            {
                const auto packed = (static_cast<std::uint64_t>(key.id) << 32U) | key.matrixLayout.stride;
                return std::hash<std::uint64_t>{}(packed) ^ static_cast<std::size_t>(key.matrixLayout.rowMajor);
            }
        };

        using SpirvTypeExtentCache =
            std::unordered_map<SpirvTypeExtentKey, std::optional<std::uint64_t>, SpirvTypeExtentKeyHash>;

        void countSpirvTypeVisit() noexcept
        {
#ifdef RACHIS_TESTING
            test::spirvReflectionTypeVisits().fetch_add(1, std::memory_order_relaxed);
#endif
        }

        [[nodiscard]] std::uint32_t resolveArrayStride(const SpirvModule& module,
                                                       std::uint32_t typeId,
                                                       std::unordered_set<std::uint32_t>& visited,
                                                       std::size_t depth)
        {
            if (depth > maxSpirvTypeNestingDepth || !visited.insert(typeId).second)
            {
                return 0;
            }
            countSpirvTypeVisit();
            const auto& arrayStride = module.decorations.arrayStride;
            if (arrayStride.contains(typeId))
            {
                return arrayStride.at(typeId);
            }
            const auto found = module.types.find(typeId);
            if (found == module.types.end())
            {
                return 0;
            }
            const auto& type = found->second;
            if (type.kind == TypeInfo::Kind::pointer)
            {
                return resolveArrayStride(module, type.pointee, visited, depth + 1);
            }
            if (type.kind == TypeInfo::Kind::array || type.kind == TypeInfo::Kind::runtimeArray)
            {
                return arrayStride.contains(type.element) ? arrayStride.at(type.element) : 0;
            }
            if (type.kind == TypeInfo::Kind::structure)
            {
                for (const auto member : type.members)
                {
                    if (const auto stride = resolveArrayStride(module, member, visited, depth + 1); stride != 0)
                    {
                        return stride;
                    }
                }
            }
            return 0;
        }

        [[nodiscard]] std::uint32_t resolveArrayStride(const SpirvModule& module, std::uint32_t typeId)
        {
            std::unordered_set<std::uint32_t> visited;
            return resolveArrayStride(module, typeId, visited, 0);
        }

        [[nodiscard]] std::optional<std::uint64_t> typeExtent(const SpirvModule& module,
                                                              SpirvTypeExtentCache& cache,
                                                              std::uint32_t id,
                                                              SpirvMatrixLayout matrixLayout,
                                                              std::size_t depth);

        [[nodiscard]] std::optional<std::uint64_t> structureExtent(const SpirvModule& module,
                                                                   SpirvTypeExtentCache& cache,
                                                                   std::uint32_t id,
                                                                   const TypeInfo& type,
                                                                   std::size_t depth)
        {
            const auto& memberOffsets = module.decorations.memberOffsets;
            const auto& memberMatrixStrides = module.decorations.memberMatrixStrides;
            const auto& rowMajorMembers = module.decorations.rowMajorMembers;
            std::uint64_t extent = 0;
            for (std::uint32_t member = 0; member < type.members.size(); ++member)
            {
                if (!memberOffsets.contains(id) || !memberOffsets.at(id).contains(member))
                {
                    return std::nullopt;
                }
                const SpirvMatrixLayout matrixLayout{
                    .stride = memberMatrixStrides.contains(id) && memberMatrixStrides.at(id).contains(member)
                                  ? memberMatrixStrides.at(id).at(member)
                                  : 0,
                    .rowMajor = rowMajorMembers.contains(id) && rowMajorMembers.at(id).contains(member),
                };
                const auto size = typeExtent(module, cache, type.members[member], matrixLayout, depth + 1);
                if (!size || *size > maxPushConstantBytes)
                {
                    return std::nullopt;
                }
                extent = (std::max)(extent, memberOffsets.at(id).at(member) + *size);
            }
            return extent;
        }

        [[nodiscard]] std::optional<std::uint64_t> rowMajorMatrixExtent(const SpirvModule& module,
                                                                        SpirvTypeExtentCache& cache,
                                                                        const TypeInfo& type,
                                                                        std::uint32_t matrixStride,
                                                                        std::size_t depth)
        {
            const auto column = module.types.find(type.element);
            if (column == module.types.end() || column->second.kind != TypeInfo::Kind::vector ||
                column->second.arrayLength == 0 || type.arrayLength == 0)
            {
                return std::nullopt;
            }
            const auto componentSize = typeExtent(module, cache, column->second.element, {}, depth + 2);
            if (!componentSize || *componentSize > maxPushConstantBytes)
            {
                return std::nullopt;
            }
            const auto rowSize = *componentSize * type.arrayLength;
            if (matrixStride < rowSize)
            {
                return std::nullopt;
            }
            return (std::uint64_t{matrixStride} * (column->second.arrayLength - 1ULL)) + rowSize;
        }

        [[nodiscard]] std::optional<std::uint64_t> elementSequenceExtent(const SpirvModule& module,
                                                                         SpirvTypeExtentCache& cache,
                                                                         std::uint32_t id,
                                                                         const TypeInfo& type,
                                                                         SpirvMatrixLayout matrixLayout,
                                                                         std::size_t depth)
        {
            if (type.kind == TypeInfo::Kind::matrix && matrixLayout.rowMajor)
            {
                return rowMajorMatrixExtent(module, cache, type, matrixLayout.stride, depth);
            }
            const auto& arrayStride = module.decorations.arrayStride;
            const auto elementSize = typeExtent(module, cache, type.element, matrixLayout, depth + 1);
            if (!elementSize || *elementSize > maxPushConstantBytes || type.arrayLength == 0)
            {
                return std::nullopt;
            }
            std::uint64_t stride = *elementSize;
            if (type.kind == TypeInfo::Kind::matrix)
            {
                stride = matrixLayout.stride;
            }
            else if (type.kind == TypeInfo::Kind::array)
            {
                stride = arrayStride.contains(id) ? arrayStride.at(id) : 0;
            }
            if (stride < *elementSize)
            {
                return std::nullopt;
            }
            return stride * (type.arrayLength - 1ULL) + *elementSize;
        }

        [[nodiscard]] std::optional<std::uint64_t> measureTypeExtent(const SpirvModule& module,
                                                                     SpirvTypeExtentCache& cache,
                                                                     std::uint32_t id,
                                                                     const TypeInfo& type,
                                                                     SpirvMatrixLayout matrixLayout,
                                                                     std::size_t depth)
        {
            switch (type.kind)
            {
            case TypeInfo::Kind::scalar:
                return type.scalarBytes == 0 ? std::nullopt : std::optional<std::uint64_t>{type.scalarBytes};
            case TypeInfo::Kind::structure:
                return structureExtent(module, cache, id, type, depth);
            case TypeInfo::Kind::vector:
            case TypeInfo::Kind::matrix:
            case TypeInfo::Kind::array:
                return elementSequenceExtent(module, cache, id, type, matrixLayout, depth);
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] std::optional<std::uint64_t> typeExtent(const SpirvModule& module,
                                                              SpirvTypeExtentCache& cache,
                                                              std::uint32_t id,
                                                              SpirvMatrixLayout matrixLayout,
                                                              std::size_t depth)
        {
            const auto found = module.types.find(id);
            if (found == module.types.end() || depth > maxSpirvTypeNestingDepth)
            {
                return std::nullopt;
            }
            const SpirvTypeExtentKey key{.id = id, .matrixLayout = matrixLayout};
            if (const auto cached = cache.find(key); cached != cache.end())
            {
                return cached->second;
            }
            countSpirvTypeVisit();
            const auto extent = measureTypeExtent(module, cache, id, found->second, matrixLayout, depth);
            cache.emplace(key, extent);
            return extent;
        }

        [[nodiscard]] Result<std::uint32_t> reflectPushConstantBytes(const SpirvModule& module,
                                                                     std::uint32_t variableId)
        {
            SpirvTypeExtentCache cache;
            const auto extent = typeExtent(module, cache, variablePointeeTypeId(module, variableId), {}, 0);
            if (!extent || *extent == 0 || *extent > maxPushConstantBytes)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The SPIR-V push-constant block has an invalid or unsupported extent.",
                });
            }
            return static_cast<std::uint32_t>(*extent);
        }

        struct DescriptorShape final
        {
            std::uint32_t pointeeTypeId = 0;
            std::uint32_t elementTypeId = 0;
            std::uint32_t count = 1;
            const TypeInfo* element = nullptr;
        };

        [[nodiscard]] DescriptorShape resolveDescriptorShape(const SpirvModule& module, std::uint32_t variableId)
        {
            const auto pointee = variablePointeeTypeId(module, variableId);
            DescriptorShape shape{.pointeeTypeId = pointee, .elementTypeId = pointee};
            const auto pointeeIt = module.types.find(pointee);
            const auto* pointeeType = pointeeIt != module.types.end() ? &pointeeIt->second : nullptr;
            if (pointeeType != nullptr &&
                (pointeeType->kind == TypeInfo::Kind::array || pointeeType->kind == TypeInfo::Kind::runtimeArray))
            {
                shape.count = pointeeType->runtimeArray ? unboundedBindingCount : pointeeType->arrayLength;
                shape.elementTypeId = pointeeType->element;
            }
            const auto elementIt = module.types.find(shape.elementTypeId);
            shape.element = elementIt != module.types.end() ? &elementIt->second : pointeeType;
            return shape;
        }

        [[nodiscard]] bool isReadOnlyStorageBlock(const SpirvModule& module,
                                                  std::uint32_t variableId,
                                                  const DescriptorShape& shape)
        {
            if (module.decorations.nonWritable.contains(variableId))
            {
                return true;
            }
            if (shape.element == nullptr || shape.element->kind != TypeInfo::Kind::structure ||
                shape.element->members.empty())
            {
                return false;
            }
            const auto members = module.decorations.nonWritableMembers.find(shape.elementTypeId);
            if (members == module.decorations.nonWritableMembers.end())
            {
                return false;
            }
            const auto memberCount = static_cast<std::uint32_t>(shape.element->members.size());
            for (std::uint32_t member = 0; member < memberCount; ++member)
            {
                if (!members->second.contains(member))
                {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] std::optional<ShaderResourceType> classifyDescriptorResource(const SpirvModule& module,
                                                                                   std::uint32_t variableId,
                                                                                   std::uint32_t storageClass,
                                                                                   const DescriptorShape& shape)
        {
            const auto elementKind = shape.element != nullptr ? shape.element->kind : TypeInfo::Kind::unknown;
            switch (elementKind)
            {
            case TypeInfo::Kind::sampler:
                return ShaderResourceType::sampler;
            case TypeInfo::Kind::sampledImage:
                return ShaderResourceType::sampledTexture;
            case TypeInfo::Kind::image:
                return shape.element->sampled == 2 ? ShaderResourceType::storageTexture
                                                   : ShaderResourceType::sampledTexture;
            case TypeInfo::Kind::acceleration:
                return ShaderResourceType::accelerationStructure;
            default:
                break;
            }
            const auto& decorations = module.decorations;
            if (storageClass == spirvStorageClassStorageBuffer ||
                decorations.bufferBlock.contains(shape.pointeeTypeId) ||
                decorations.bufferBlock.contains(shape.elementTypeId))
            {
                return isReadOnlyStorageBlock(module, variableId, shape) ? ShaderResourceType::readOnlyStorageBuffer
                                                                         : ShaderResourceType::storageBuffer;
            }
            if (storageClass == spirvStorageClassUniform || decorations.block.contains(shape.pointeeTypeId))
            {
                return ShaderResourceType::uniformBuffer;
            }
            return std::nullopt;
        }

        [[nodiscard]] Status reflectDescriptorBinding(const SpirvModule& module,
                                                      std::uint32_t variableId,
                                                      std::uint32_t storageClass,
                                                      SpirvReflection& reflection)
        {
            const auto shape = resolveDescriptorShape(module, variableId);
            const auto type = classifyDescriptorResource(module, variableId, storageClass, shape);
            if (!type.has_value())
            {
                return {};
            }
            if (*type == ShaderResourceType::readOnlyStorageBuffer)
            {
                reflection.readOnlyStorageArrayStride = resolveArrayStride(module, shape.pointeeTypeId);
            }
            else if (*type == ShaderResourceType::storageBuffer)
            {
                reflection.storageArrayStride = resolveArrayStride(module, shape.pointeeTypeId);
            }
            if ((*type == ShaderResourceType::storageBuffer || *type == ShaderResourceType::storageTexture) &&
                !isReadOnlyStorageBlock(module, variableId, shape))
            {
                reflection.writableStorage = true;
            }
            const auto& descriptorSet = module.decorations.descriptorSet;
            const ShaderBindingRequirement requirement{
                .set = descriptorSet.contains(variableId) ? descriptorSet.at(variableId) : 0U,
                .binding = module.decorations.binding.at(variableId),
                .type = *type,
                .count = shape.count,
            };
            if (requirement.set >= maxBindingSets)
            {
                return std::unexpected(Error{ErrorCode::invalidArgument,
                                             0,
                                             "A SPIR-V descriptor set exceeds the portable binding-set limit. Push "
                                             "constants require the PushConstant storage class."});
            }
            reflection.bindings.push_back(requirement);
            return {};
        }

        [[nodiscard]] constexpr std::uint32_t saturatingProduct(std::uint32_t left, std::uint32_t right) noexcept
        {
            const auto product = static_cast<std::uint64_t>(left) * right;
            return product > (std::numeric_limits<std::uint32_t>::max)() ? (std::numeric_limits<std::uint32_t>::max)()
                                                                         : static_cast<std::uint32_t>(product);
        }

        [[nodiscard]] std::optional<std::uint32_t> vertexInputLocationCount(const SpirvModule& module,
                                                                            std::uint32_t typeId,
                                                                            std::size_t depth)
        {
            const auto found = module.types.find(typeId);
            if (found == module.types.end() || depth > maxSpirvTypeNestingDepth)
            {
                return std::nullopt;
            }
            const auto& type = found->second;
            switch (type.kind)
            {
            case TypeInfo::Kind::scalar:
                return 1U;
            case TypeInfo::Kind::vector:
                return scalarByteWidth(module.types, type.element) == sizeof(std::uint64_t) && type.arrayLength > 2
                           ? 2U
                           : 1U;
            case TypeInfo::Kind::matrix:
            case TypeInfo::Kind::array: {
                const auto element = vertexInputLocationCount(module, type.element, depth + 1);
                if (!element || type.arrayLength == 0)
                {
                    return std::nullopt;
                }
                return saturatingProduct(*element, type.arrayLength);
            }
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] Status reflectVertexInputLocations(const SpirvModule& module,
                                                         std::uint32_t variableId,
                                                         SpirvReflection& reflection)
        {
            const auto count = vertexInputLocationCount(module, variablePointeeTypeId(module, variableId), 0);
            if (!count)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A SPIR-V vertex input must be a scalar, vector, matrix, or array of them.",
                });
            }
            const auto first = module.decorations.location.at(variableId);
            if (first >= maxVertexAttributes || *count > maxVertexAttributes - first)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A SPIR-V vertex input uses a location beyond the portable vertex attribute limit.",
                });
            }
            for (std::uint32_t location = first; location < first + *count; ++location)
            {
                reflection.vertexInputLocations.push_back(location);
            }
            return {};
        }

        [[nodiscard]] Status reflectVariables(const SpirvModule& module,
                                              const std::unordered_set<std::uint32_t>& usedIds,
                                              ShaderStage stage,
                                              SpirvReflection& reflection)
        {
            for (const auto& [id, storage] : module.variables.storageClasses)
            {
                if (storage == spirvStorageClassFunction || storage == spirvStorageClassOutput)
                {
                    continue;
                }
                if (storage == spirvStorageClassInput)
                {
                    if (!isEntryPointVertexInput(module, usedIds, stage, id))
                    {
                        continue;
                    }
                    if (auto reflected = reflectVertexInputLocations(module, id, reflection); !reflected)
                    {
                        return reflected;
                    }
                    continue;
                }
                if (!usedIds.contains(id))
                {
                    continue;
                }
                if (storage == spirvStorageClassPushConstant)
                {
                    auto pushConstantBytes = reflectPushConstantBytes(module, id);
                    if (!pushConstantBytes)
                    {
                        return std::unexpected(std::move(pushConstantBytes.error()));
                    }
                    reflection.pushConstantBytes = (std::max)(reflection.pushConstantBytes, *pushConstantBytes);
                    continue;
                }
                if (!module.decorations.binding.contains(id))
                {
                    continue;
                }
                if (auto reflected = reflectDescriptorBinding(module, id, storage, reflection); !reflected)
                {
                    return reflected;
                }
            }
            return {};
        }

        constexpr std::uint32_t maxReflectedStageLocations = 1024;

        [[nodiscard]] std::optional<detail::StageComponentType> stageComponentType(const TypeInfo& scalar) noexcept
        {
            using detail::StageComponentType;
            using ScalarKind = TypeInfo::ScalarKind;
            constexpr std::array<std::array<StageComponentType, 3>, 3> byKindAndWidth{{
                {StageComponentType::float16, StageComponentType::float32, StageComponentType::float64},
                {StageComponentType::sint16, StageComponentType::sint32, StageComponentType::sint64},
                {StageComponentType::uint16, StageComponentType::uint32, StageComponentType::uint64},
            }};
            if (scalar.kind != TypeInfo::Kind::scalar)
            {
                return std::nullopt;
            }
            std::size_t kind = 0;
            switch (scalar.scalarKind)
            {
            case ScalarKind::floatingPoint:
                kind = 0;
                break;
            case ScalarKind::signedInteger:
                kind = 1;
                break;
            case ScalarKind::unsignedInteger:
                kind = 2;
                break;
            default:
                return std::nullopt;
            }
            std::size_t width = 0;
            switch (scalar.scalarBytes)
            {
            case 2:
                width = 0;
                break;
            case 4:
                width = 1;
                break;
            case 8:
                width = 2;
                break;
            default:
                return std::nullopt;
            }
            return byKindAndWidth[kind][width];
        }

        [[nodiscard]] std::optional<std::uint32_t> appendStageInterfaceVector(
            const TypeInfo& scalar,
            std::uint32_t elementCount,
            std::uint32_t location,
            std::uint32_t firstComponent,
            std::vector<detail::StageInterfaceVariable>& variables)
        {
            const auto componentType = stageComponentType(scalar);
            if (!componentType.has_value() || elementCount == 0 || elementCount > 4 ||
                firstComponent >= detail::stageComponentsPerLocation)
            {
                return std::nullopt;
            }
            const auto slots = elementCount * (scalar.scalarBytes == 8 ? 2U : 1U);
            if (slots <= detail::stageComponentsPerLocation &&
                firstComponent + slots > detail::stageComponentsPerLocation)
            {
                return std::nullopt;
            }
            if (slots > detail::stageComponentsPerLocation && firstComponent != 0)
            {
                return std::nullopt;
            }
            std::uint32_t consumed = 0;
            for (std::uint32_t remaining = slots; remaining != 0; ++consumed)
            {
                const auto start = consumed == 0 ? firstComponent : 0U;
                const auto count = (std::min)(remaining, detail::stageComponentsPerLocation - start);
                variables.push_back(detail::StageInterfaceVariable{
                    .location = location + consumed,
                    .firstComponent = start,
                    .componentCount = count,
                    .componentType = *componentType,
                });
                remaining -= count;
            }
            return consumed;
        }

        [[nodiscard]] std::optional<std::uint32_t> appendStageInterfaceType(
            const SpirvModule& module,
            std::uint32_t typeId,
            std::uint32_t location,
            std::uint32_t firstComponent,
            std::size_t depth,
            std::vector<detail::StageInterfaceVariable>& variables)
        {
            const auto found = module.types.find(typeId);
            if (found == module.types.end() || depth > maxSpirvTypeNestingDepth ||
                location >= maxReflectedStageLocations)
            {
                return std::nullopt;
            }
            const auto& type = found->second;
            if (type.kind == TypeInfo::Kind::scalar)
            {
                return appendStageInterfaceVector(type, 1, location, firstComponent, variables);
            }
            if (type.kind == TypeInfo::Kind::vector)
            {
                const auto element = module.types.find(type.element);
                if (element == module.types.end())
                {
                    return std::nullopt;
                }
                return appendStageInterfaceVector(
                    element->second, type.arrayLength, location, firstComponent, variables);
            }
            const bool sequence = type.kind == TypeInfo::Kind::matrix || type.kind == TypeInfo::Kind::array;
            if (!sequence && type.kind != TypeInfo::Kind::structure)
            {
                return std::nullopt;
            }
            const auto elementCount = sequence ? type.arrayLength : static_cast<std::uint32_t>(type.members.size());
            const auto elementComponent = type.kind == TypeInfo::Kind::array ? firstComponent : 0U;
            std::uint32_t consumed = 0;
            for (std::uint32_t index = 0; index < elementCount; ++index)
            {
                const auto elementType = sequence ? type.element : type.members[index];
                const auto elementLocations = appendStageInterfaceType(
                    module, elementType, location + consumed, elementComponent, depth + 1, variables);
                if (!elementLocations.has_value())
                {
                    return std::nullopt;
                }
                consumed += *elementLocations;
            }
            return consumed;
        }

        [[nodiscard]] Status reflectStageInterfaceVariable(const SpirvModule& module,
                                                           std::uint32_t variableId,
                                                           bool arrayed,
                                                           std::vector<detail::StageInterfaceVariable>& variables)
        {
            const auto& decorations = module.decorations;
            if (decorations.builtIn.contains(variableId))
            {
                return {};
            }
            auto typeId = variablePointeeTypeId(module, variableId);
            if (arrayed)
            {
                const auto found = module.types.find(typeId);
                if (found != module.types.end() && found->second.kind == TypeInfo::Kind::array)
                {
                    typeId = found->second.element;
                }
            }
            const Error unsupported{
                ErrorCode::invalidArgument,
                0,
                "A SPIR-V stage input or output has a type, location, or component that the stage interface cannot "
                "hold.",
            };
            if (const auto location = decorations.location.find(variableId); location != decorations.location.end())
            {
                const auto component = decorations.component.find(variableId);
                const auto firstComponent = component != decorations.component.end() ? component->second : 0U;
                if (!appendStageInterfaceType(module, typeId, location->second, firstComponent, 0, variables))
                {
                    return std::unexpected(unsupported);
                }
                return {};
            }
            const auto block = module.types.find(typeId);
            const auto memberLocations = decorations.memberLocations.find(typeId);
            if (block == module.types.end() || block->second.kind != TypeInfo::Kind::structure ||
                memberLocations == decorations.memberLocations.end())
            {
                return {};
            }
            const auto memberComponents = decorations.memberComponents.find(typeId);
            for (const auto& [member, location] : memberLocations->second)
            {
                if (member >= block->second.members.size())
                {
                    return std::unexpected(unsupported);
                }
                std::uint32_t firstComponent = 0;
                if (memberComponents != decorations.memberComponents.end())
                {
                    if (const auto found = memberComponents->second.find(member);
                        found != memberComponents->second.end())
                    {
                        firstComponent = found->second;
                    }
                }
                if (!appendStageInterfaceType(
                        module, block->second.members[member], location, firstComponent, 0, variables))
                {
                    return std::unexpected(unsupported);
                }
            }
            return {};
        }

        [[nodiscard]] constexpr bool precedesByLocation(const detail::StageInterfaceVariable& left,
                                                        const detail::StageInterfaceVariable& right) noexcept
        {
            return left.location < right.location ||
                   (left.location == right.location && left.firstComponent < right.firstComponent);
        }

        [[nodiscard]] Status reflectStageInterface(const SpirvModule& module,
                                                   ShaderStage stage,
                                                   SpirvReflection& reflection)
        {
            const bool producer = stage == ShaderStage::vertex || stage == ShaderStage::domain ||
                                  stage == ShaderStage::geometry || stage == ShaderStage::mesh;
            if (!producer && stage != ShaderStage::pixel)
            {
                return {};
            }
            const auto storageClass = producer ? spirvStorageClassOutput : spirvStorageClassInput;
            auto& variables = producer ? reflection.stageOutputs : reflection.stageInputs;
            for (const auto variableId : module.entryPoint.interfaceIds)
            {
                const auto storage = module.variables.storageClasses.find(variableId);
                if (storage == module.variables.storageClasses.end() || storage->second != storageClass)
                {
                    continue;
                }
                const bool arrayed = stage == ShaderStage::mesh || module.decorations.perVertex.contains(variableId);
                if (auto reflected = reflectStageInterfaceVariable(module, variableId, arrayed, variables); !reflected)
                {
                    return reflected;
                }
            }
            std::ranges::sort(variables, precedesByLocation);
            return {};
        }

        [[nodiscard]] constexpr bool precedesBySetAndBinding(const ShaderBindingRequirement& left,
                                                             const ShaderBindingRequirement& right) noexcept
        {
            return left.set < right.set || (left.set == right.set && left.binding < right.binding);
        }

        [[nodiscard]] Result<SpirvReflection> reflectSpirv(
            std::span<const std::uint32_t> words,
            std::string_view entryPoint,
            ShaderStage stage,
            std::span<const SpecializationConstant> specializationOverrides = {})
        {
            auto module = scanSpirvModule(words, entryPoint, stage, specializationOverrides);
            if (!module)
            {
                return std::unexpected(std::move(module.error()));
            }
            if (!declaresRequestedEntryPoint(*module, entryPoint, stage))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The SPIR-V module does not contain the requested entry point and stage.",
                });
            }
            applySpecialization(*module);

            auto workgroup = resolveWorkgroupSize(*module);
            if (!workgroup)
            {
                return std::unexpected(std::move(workgroup.error()));
            }
            const auto usage = analyzeEntryPointUsage(*module);

            SpirvReflection reflection{
                .entryPoint = module->entryPoint.name,
                .stage = module->entryPoint.stage,
                .workgroupSize = *workgroup,
                .specIds = std::move(module->specialization.specIds),
                .non32BitSpecIds = std::move(module->specialization.non32BitSpecIds),
                .writesPointSize = usage.writesPointSize,
            };
            if (auto reflected = reflectVariables(*module, usage.usedIds, stage, reflection); !reflected)
            {
                return std::unexpected(std::move(reflected.error()));
            }
            if (auto reflected = reflectStageInterface(*module, stage, reflection); !reflected)
            {
                return std::unexpected(std::move(reflected.error()));
            }
            std::ranges::sort(reflection.bindings, precedesBySetAndBinding);
            std::ranges::sort(reflection.vertexInputLocations);
            const auto duplicates = std::ranges::unique(reflection.vertexInputLocations);
            reflection.vertexInputLocations.erase(duplicates.begin(), duplicates.end());
            return reflection;
        }

        [[nodiscard]] Status compareInterface(const InterfaceRecord& declared, const SpirvReflection& reflected)
        {
            if (declared.workgroupSize.has_value() != reflected.workgroupSize.has_value() ||
                (declared.workgroupSize.has_value() && (declared.workgroupSize->x != reflected.workgroupSize->x ||
                                                        declared.workgroupSize->y != reflected.workgroupSize->y ||
                                                        declared.workgroupSize->z != reflected.workgroupSize->z)))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Declared workgroup size does not match the SPIR-V LocalSize execution mode.",
                });
            }
            if (declared.bindings.size() != reflected.bindings.size())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Declared shader bindings do not match the reflected SPIR-V descriptor set.",
                });
            }
            auto declaredBindings = declared.bindings;
            std::ranges::sort(declaredBindings, precedesBySetAndBinding);
            for (std::size_t index = 0; index < declaredBindings.size(); ++index)
            {
                const auto& left = declaredBindings[index];
                const auto& right = reflected.bindings[index];
                if (left.set != right.set || left.binding != right.binding || left.type != right.type ||
                    left.count != right.count)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A declared shader binding does not match the reflected SPIR-V set, binding, type, or count.",
                    });
                }
            }

            std::uint32_t declaredPush = 0;
            for (const auto& range : declared.pushConstants)
            {
                declaredPush = (std::max)(declaredPush, range.offset + range.size);
            }
            if ((declaredPush == 0) != (reflected.pushConstantBytes == 0))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Declared push-constant ranges do not match the reflected SPIR-V push-constant block.",
                });
            }
            if (declaredPush != 0 && declaredPush < reflected.pushConstantBytes)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Declared push-constant ranges are smaller than the reflected SPIR-V block.",
                });
            }

            if (declared.vertexInputLocations != reflected.vertexInputLocations)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Declared vertex input locations do not match the reflected SPIR-V locations.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr bool isAtomicOpcode(std::uint32_t opcode) noexcept
        {
            return opcode >= spirvOpAtomicLoad && opcode <= spirvOpAtomicXor;
        }

        struct SpirvFeatureRequirements final
        {
            bool float16 = false;
            bool float64 = false;
            bool int16 = false;
            bool int64 = false;
            bool storageBuffer16Bit = false;
            bool uniformAndStorageBuffer16Bit = false;
            bool pushConstant16Bit = false;
            bool inputOutput16Bit = false;
            bool subgroups = false;
            bool groupSharedInt64Atomics = false;
            bool typedInt64Atomics = false;
            bool storageInt64Atomics = false;
            bool geometry = false;
            bool tessellation = false;
            bool clipDistance = false;
            bool cullDistance = false;
            bool multiViewport = false;
            bool multiview = false;
            bool layeredOrViewportIndex = false;
            bool fragmentBarycentric = false;
            bool drawParameters = false;
            bool rayTracing = false;
            bool rayQuery = false;
            bool meshShading = false;
            bool localSizeId = false;
            bool imageGatherExtended = false;
            bool sampleRateShading = false;
            bool minLod = false;
        };

        struct SpirvAtomicOperandTypes final
        {
            std::unordered_map<std::uint32_t, std::uint32_t> integerBitWidthByType;
            std::unordered_map<std::uint32_t, std::uint32_t> storageClassByPointerType;
            std::unordered_map<std::uint32_t, std::uint32_t> pointeeByPointerType;
            std::unordered_map<std::uint32_t, std::uint32_t> typeByResultId;
            bool declaresInt64Atomics = false;
        };

        void recordCapabilityRequirement(std::uint32_t capability, SpirvFeatureRequirements& requirements) noexcept
        {
            switch (capability)
            {
            case spirvCapabilityFloat16:
                requirements.float16 = true;
                break;
            case spirvCapabilityFloat64:
                requirements.float64 = true;
                break;
            case spirvCapabilityInt16:
                requirements.int16 = true;
                break;
            case spirvCapabilityStorageBuffer16BitAccess:
                requirements.storageBuffer16Bit = true;
                break;
            case spirvCapabilityUniformAndStorageBuffer16BitAccess:
                requirements.uniformAndStorageBuffer16Bit = true;
                break;
            case spirvCapabilityStoragePushConstant16:
                requirements.pushConstant16Bit = true;
                break;
            case spirvCapabilityStorageInputOutput16:
                requirements.inputOutput16Bit = true;
                break;
            case spirvCapabilityInt64:
            case spirvCapabilityInt64Atomics:
                requirements.int64 = true;
                break;
            case spirvCapabilityInt64ImageExt:
                requirements.int64 = true;
                requirements.typedInt64Atomics = true;
                break;
            case spirvCapabilityGroupNonUniform:
            case spirvCapabilityGroupNonUniformArithmetic:
                requirements.subgroups = true;
                break;
            case spirvCapabilityGeometry:
                requirements.geometry = true;
                break;
            case spirvCapabilityTessellation:
                requirements.tessellation = true;
                break;
            case spirvCapabilityClipDistance:
                requirements.clipDistance = true;
                break;
            case spirvCapabilityCullDistance:
                requirements.cullDistance = true;
                break;
            case spirvCapabilityMultiViewport:
                requirements.multiViewport = true;
                break;
            case spirvCapabilityDrawParameters:
                requirements.drawParameters = true;
                break;
            case spirvCapabilityMultiView:
                requirements.multiview = true;
                break;
            case spirvCapabilityShaderViewportIndexLayerEXT:
                requirements.layeredOrViewportIndex = true;
                break;
            case spirvCapabilityFragmentBarycentricKHR:
                requirements.fragmentBarycentric = true;
                break;
            case spirvCapabilityRayTracingKHR:
                requirements.rayTracing = true;
                break;
            case spirvCapabilityRayQueryKHR:
                requirements.rayQuery = true;
                break;
            case spirvCapabilityMeshShadingEXT:
                requirements.meshShading = true;
                break;
            case spirvCapabilityImageGatherExtended:
                requirements.imageGatherExtended = true;
                break;
            case spirvCapabilitySampleRateShading:
                requirements.sampleRateShading = true;
                break;
            case spirvCapabilityMinLod:
                requirements.minLod = true;
                break;
            default:
                break;
            }
        }

        [[nodiscard]] std::uint32_t pointeeIntegerBitWidth(const SpirvAtomicOperandTypes& operandTypes,
                                                           std::uint32_t pointerType)
        {
            const auto pointeeIt = operandTypes.pointeeByPointerType.find(pointerType);
            if (pointeeIt == operandTypes.pointeeByPointerType.end())
            {
                return 0;
            }
            const auto widthIt = operandTypes.integerBitWidthByType.find(pointeeIt->second);
            return widthIt != operandTypes.integerBitWidthByType.end() ? widthIt->second : 0;
        }

        void recordInt64AtomicRequirement(const SpirvInstruction& instruction,
                                          const SpirvAtomicOperandTypes& operandTypes,
                                          SpirvFeatureRequirements& requirements)
        {
            const auto pointerId =
                instruction.opcode == spirvOpAtomicStore ? instruction.words[1] : instruction.words[3];
            const auto typeIt = operandTypes.typeByResultId.find(pointerId);
            const auto pointerType = typeIt != operandTypes.typeByResultId.end() ? typeIt->second : pointerId;
            const auto storageIt = operandTypes.storageClassByPointerType.find(pointerType);
            const auto storage = storageIt != operandTypes.storageClassByPointerType.end() ? storageIt->second : 0;
            if (pointeeIntegerBitWidth(operandTypes, pointerType) != 64 && !operandTypes.declaresInt64Atomics)
            {
                return;
            }
            if (storage == spirvStorageClassWorkgroup)
            {
                requirements.groupSharedInt64Atomics = true;
            }
            else if (storage == spirvStorageClassImage || storage == spirvStorageClassUniformConstant)
            {
                requirements.typedInt64Atomics = true;
            }
            else if (storage == spirvStorageClassStorageBuffer || storage == spirvStorageClassUniform)
            {
                requirements.storageInt64Atomics = true;
            }
        }

        [[nodiscard]] constexpr bool declaresUntypedModuleEntity(std::uint32_t opcode) noexcept
        {
            return opcode == spirvOpExtension || opcode == spirvOpExtInstImport || opcode == spirvOpMemoryModel ||
                   opcode == spirvOpCapability || opcode == spirvOpDecorationGroup ||
                   (opcode >= spirvOpTypeVoid && opcode <= spirvOpTypeForwardPointer) ||
                   opcode == spirvOpTypeRayQuery || opcode == spirvOpTypeAccelerationStructure;
        }

        [[nodiscard]] constexpr bool declaresResultTypeAndId(std::uint32_t opcode) noexcept
        {
            if (opcode == spirvOpUndef || opcode == spirvOpFunction || opcode == spirvOpFunctionParameter)
            {
                return true;
            }
            return !referencesNoTrackedIds(opcode) && !resultlessIdOperandLayout(opcode).has_value() &&
                   !refersToIdsWithoutDefiningOne(opcode) && !declaresUntypedModuleEntity(opcode);
        }

        void recordFeatureRequirement(const SpirvInstruction& instruction,
                                      SpirvAtomicOperandTypes& operandTypes,
                                      SpirvFeatureRequirements& requirements)
        {
            const auto opcode = instruction.opcode;
            const auto wordCount = instruction.wordCount;
            const auto* words = instruction.words;
            switch (opcode)
            {
            case spirvOpCapability:
                if (wordCount >= 2)
                {
                    recordCapabilityRequirement(words[1], requirements);
                    if (words[1] == spirvCapabilityInt64Atomics)
                    {
                        operandTypes.declaresInt64Atomics = true;
                    }
                }
                break;
            case spirvOpExecutionModeId:
                if (wordCount >= 3 && words[2] == spirvExecutionModeLocalSizeId)
                {
                    requirements.localSizeId = true;
                }
                break;
            case spirvOpTypeInt:
                if (wordCount >= 4)
                {
                    operandTypes.integerBitWidthByType[words[1]] = words[2];
                }
                break;
            case spirvOpTypePointer:
                if (wordCount >= 4)
                {
                    operandTypes.storageClassByPointerType[words[1]] = words[2];
                    operandTypes.pointeeByPointerType[words[1]] = words[3];
                }
                break;
            case spirvOpVariable:
                if (wordCount >= 4)
                {
                    operandTypes.typeByResultId[words[2]] = words[1];
                }
                break;
            case spirvOpImageTexelPointer:
                if (wordCount >= 3)
                {
                    operandTypes.typeByResultId[words[2]] = words[1];
                }
                break;
            default:
                if (isAtomicOpcode(opcode) && wordCount >= 4)
                {
                    recordInt64AtomicRequirement(instruction, operandTypes, requirements);
                }
                else if (wordCount >= 3 && declaresResultTypeAndId(opcode) &&
                         operandTypes.storageClassByPointerType.contains(words[1]))
                {
                    operandTypes.typeByResultId[words[2]] = words[1];
                }
                break;
            }
        }

        [[nodiscard]] Result<SpirvFeatureRequirements> scanSpirvFeatureRequirements(
            std::span<const std::uint32_t> words)
        {
            SpirvFeatureRequirements requirements;
            SpirvAtomicOperandTypes operandTypes;
            std::size_t index = spirvHeaderWordCount;
            while (index < words.size())
            {
                auto instruction = decodeInstruction(words, index);
                if (!instruction)
                {
                    return std::unexpected(std::move(instruction.error()));
                }
                recordFeatureRequirement(*instruction, operandTypes, requirements);
                index += instruction->wordCount;
            }
            return requirements;
        }

        struct FeatureSupportCheck final
        {
            bool unsupported = false;
            const char* message = nullptr;
        };

        [[nodiscard]] Status validateFeatureSupport(const SpirvFeatureRequirements& requirements,
                                                    const DeviceCapabilities& capabilities,
                                                    bool maintenance4,
                                                    const VulkanShaderFeatureSupport& features)
        {
            const std::array checks{
                FeatureSupportCheck{requirements.float16 && !capabilities.shaderFloat16,
                                    "16-bit float shader arithmetic is not available on this device."},
                FeatureSupportCheck{requirements.int16 && !capabilities.shaderInt16,
                                    "16-bit integer shader arithmetic is not available on this device."},
                FeatureSupportCheck{requirements.int64 && !capabilities.shaderInt64,
                                    "64-bit integer shader arithmetic is not available on this device."},
                FeatureSupportCheck{requirements.float64 && !features.shaderFloat64,
                                    "64-bit float shader arithmetic is not available on this device."},
                FeatureSupportCheck{requirements.storageBuffer16Bit && !features.storageBuffer16BitAccess,
                                    "16-bit storage-buffer access is not available on this device."},
                FeatureSupportCheck{requirements.uniformAndStorageBuffer16Bit &&
                                        !features.uniformAndStorageBuffer16BitAccess,
                                    "16-bit uniform-buffer access is not available on this device."},
                FeatureSupportCheck{requirements.pushConstant16Bit && !features.storagePushConstant16,
                                    "16-bit push-constant access is not available on this device."},
                FeatureSupportCheck{requirements.inputOutput16Bit && !features.storageInputOutput16,
                                    "16-bit shader stage inputs and outputs are not available on this device."},
                FeatureSupportCheck{requirements.subgroups && !capabilities.subgroupOperations,
                                    "Shader subgroup operations are not available on this device."},
                FeatureSupportCheck{requirements.groupSharedInt64Atomics && !capabilities.shaderInt64AtomicGroupShared,
                                    "Group-shared 64-bit integer atomics are not available on this device."},
                FeatureSupportCheck{requirements.typedInt64Atomics && !capabilities.shaderInt64AtomicTypedResource,
                                    "Typed-resource 64-bit integer atomics are not available on this device."},
                FeatureSupportCheck{requirements.storageInt64Atomics,
                                    "Storage-buffer 64-bit integer atomics are not a portable RHI capability."},
                FeatureSupportCheck{requirements.geometry && !capabilities.geometryShaders &&
                                        !capabilities.fragmentPrimitiveId,
                                    "Geometry-shader or fragment PrimitiveId support is not available on this device."},
                FeatureSupportCheck{requirements.tessellation && !capabilities.tessellationShaders,
                                    "Tessellation shaders are not available on this device."},
                FeatureSupportCheck{requirements.clipDistance && capabilities.limits.maxClipDistances == 0,
                                    "Shader clip distances are not available on this device."},
                FeatureSupportCheck{requirements.cullDistance && capabilities.limits.maxCullDistances == 0,
                                    "Shader cull distances are not available on this device."},
                FeatureSupportCheck{requirements.multiViewport && !capabilities.multiViewport,
                                    "Multiple viewports are not available on this device."},
                FeatureSupportCheck{requirements.multiview && !capabilities.multiviewRendering,
                                    "Multiview rendering is not available on this device."},
                FeatureSupportCheck{requirements.layeredOrViewportIndex &&
                                        capabilities.layeredRenderingModes == LayeredRenderingMode::none &&
                                        !capabilities.multiViewport,
                                    "Layered rendering and viewport-array index are not available on this device."},
                FeatureSupportCheck{requirements.fragmentBarycentric && !capabilities.fragmentBarycentricCoordinates,
                                    "Fragment barycentric coordinates are not available on this device."},
                FeatureSupportCheck{requirements.rayTracing && !capabilities.rayTracing,
                                    "Ray-tracing pipelines are not available on this device."},
                FeatureSupportCheck{requirements.rayQuery && !capabilities.rayQuery,
                                    "Inline ray queries are not available on this device."},
                FeatureSupportCheck{requirements.meshShading && !capabilities.meshShaders,
                                    "Mesh shaders are not available on this device."},
                FeatureSupportCheck{requirements.localSizeId && !maintenance4,
                                    "SPIR-V LocalSizeId execution mode requires the Vulkan maintenance4 feature."},
                FeatureSupportCheck{requirements.imageGatherExtended && !features.shaderImageGatherExtended,
                                    "Extended image gather offsets are not available on this device."},
                FeatureSupportCheck{requirements.sampleRateShading && !features.sampleRateShading,
                                    "Sample-rate shading is not available on this device."},
                FeatureSupportCheck{requirements.minLod && !features.shaderResourceMinLod,
                                    "Shader minimum-LOD clamps are not available on this device."},
                FeatureSupportCheck{
                    requirements.drawParameters && !features.shaderDrawParameters,
                    "SPIR-V DrawParameters capability requires the Vulkan shaderDrawParameters feature."},
            };
            for (const auto& check : checks)
            {
                if (check.unsupported)
                {
                    return std::unexpected(Error{ErrorCode::unsupported, 0, check.message});
                }
            }
            return {};
        }

        [[nodiscard]] Status validateSpirvFeatureRequirements(std::span<const std::uint32_t> words,
                                                              const DeviceCapabilities& capabilities,
                                                              bool maintenance4,
                                                              const VulkanShaderFeatureSupport& features)
        {
            auto requirements = scanSpirvFeatureRequirements(words);
            if (!requirements)
            {
                return std::unexpected(std::move(requirements.error()));
            }
            return validateFeatureSupport(*requirements, capabilities, maintenance4, features);
        }

        [[nodiscard]] constexpr bool isPreRasterizationStoreStage(ShaderStage stage) noexcept
        {
            return stage == ShaderStage::vertex || stage == ShaderStage::hull || stage == ShaderStage::domain ||
                   stage == ShaderStage::geometry;
        }

        [[nodiscard]] Status validatePreRasterizationStores(ShaderStage stage,
                                                            const SpirvReflection& reflected,
                                                            const VulkanShaderFeatureSupport& features)
        {
            if (reflected.writableStorage && isPreRasterizationStoreStage(stage) &&
                !features.vertexPipelineStoresAndAtomics)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Writable storage in vertex, tessellation, and geometry shaders is not available on this device.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateSpecialization(const SpecializationSet& constants,
                                                    const SpirvReflection& reflected)
        {
            for (const auto& constant : constants)
            {
                if (!reflected.specIds.contains(constant.id))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A specialization constant ID is not present in the SPIR-V module.",
                    });
                }
                if (reflected.non32BitSpecIds.contains(constant.id))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A specialization value overrides a SPIR-V spec constant that is not a bool or a 32-bit "
                        "scalar.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Status validateShaderModuleSource(const ShaderModuleDesc& desc)
        {
            const bool hasDirectBytecode = !desc.bytecode.empty();
            const bool hasLibrary = desc.library.isValid();
            if (!isSingleShaderStage(desc.stage) || desc.entryPoint.empty() || hasDirectBytecode == hasLibrary)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A shader module requires exactly one stage and one bytecode source: direct bytes or a live "
                    "library.",
                });
            }
            if (hasDirectBytecode && !desc.specializationConstants.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Specialization values require a shader library containing precompiled backend variants.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<VkShaderModule> createNativeShaderModule(VkDevice device,
                                                                      const VolkDeviceTable* table,
                                                                      std::span<const std::uint32_t> words)
        {
            if (table == nullptr || table->vkCreateShaderModule == nullptr || device == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The Vulkan shader manager is not initialized.",
                });
            }
            VkShaderModuleCreateInfo const createInfo{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .codeSize = words.size_bytes(),
                .pCode = words.data(),
            };
            VkShaderModule module = VK_NULL_HANDLE;
            const auto created = table->vkCreateShaderModule(device, &createInfo, nullptr, &module);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateShaderModule"));
            }
            return module;
        }

        [[nodiscard]] VulkanShaderLayoutProbe shaderLayoutProbe(const SpirvReflection& reflected) noexcept
        {
            return VulkanShaderLayoutProbe{
                .pushConstantBytes = reflected.pushConstantBytes,
                .readOnlyStorageArrayStride = reflected.readOnlyStorageArrayStride,
                .storageArrayStride = reflected.storageArrayStride,
            };
        }

    } // namespace

    VulkanShaderManager::ShaderKeepAlive::~ShaderKeepAlive()
    {
        if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroyShaderModule != nullptr &&
            module != VK_NULL_HANDLE)
        {
            table->vkDestroyShaderModule(device, module, nullptr);
        }
    }

    VulkanShaderManager::~VulkanShaderManager()
    {
        shutdown();
    }

    void VulkanShaderManager::initialize(VkDevice device,
                                         const VolkDeviceTable* table,
                                         const DeviceCapabilities& capabilities,
                                         const VulkanInstanceTable* instanceTable,
                                         bool maintenance4,
                                         VulkanShaderFeatureSupport features,
                                         bool retainSpirv) noexcept
    {
        device_ = device;
        table_ = table;
        instanceTable_ = instanceTable;
        capabilities_ = capabilities;
        maintenance4_ = maintenance4;
        features_ = features;
        retainSpirv_ = retainSpirv;
#ifdef RACHIS_TESTING
        const std::scoped_lock lock(boundShaderMutex);
        boundShaders = this;
#endif
    }

    void VulkanShaderManager::shutdown() noexcept
    {
#ifdef RACHIS_TESTING
        const std::scoped_lock bound(boundShaderMutex);
        if (boundShaders == this)
        {
            boundShaders = nullptr;
        }
#endif
        const std::scoped_lock lock(mutex_);
        modules_.clear();
        libraries_.clear();
        device_ = VK_NULL_HANDLE;
        table_ = nullptr;
        instanceTable_ = nullptr;
        maintenance4_ = false;
        features_ = {};
        retainSpirv_ = false;
    }

    Result<ShaderLibraryHandle> VulkanShaderManager::createShaderLibrary(const ShaderLibraryDesc& desc)
    {
        if (desc.entries.empty())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shader library requires at least one staged bytecode entry.",
            });
        }
        auto record = std::make_shared<ShaderLibraryRecord>();
        record->label = std::string(desc.label);
        record->entries.reserve(desc.entries.size());
        for (const auto& entry : desc.entries)
        {
            if (!isSingleShaderStage(entry.stage) || entry.bytecode.empty() || entry.entryPoint.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Every shader-library entry requires exactly one stage, non-empty bytecode, and a non-empty name.",
                });
            }
            auto specializationConstants = buildSpecializationSet(entry.specializationConstants);
            if (!specializationConstants)
            {
                return std::unexpected(std::move(specializationConstants.error()));
            }
            if (std::ranges::any_of(record->entries, [&](const ShaderLibraryEntryRecord& existing) {
                    return existing.stage == entry.stage && existing.entryPoint == entry.entryPoint &&
                           existing.specializationConstants == *specializationConstants;
                }))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader-library entries must be unique by stage, name, and specialization values.",
                });
            }
            auto words = copySpirvWords(entry.bytecode);
            if (!words)
            {
                return std::unexpected(std::move(words.error()));
            }
            auto reflected = reflectSpirv(*words, entry.entryPoint, entry.stage, *specializationConstants);
            if (!reflected)
            {
                return std::unexpected(std::move(reflected.error()));
            }
            auto specialization = validateSpecialization(*specializationConstants, *reflected);
            if (!specialization)
            {
                return std::unexpected(std::move(specialization.error()));
            }
            record->entries.push_back({
                .stage = entry.stage,
                .bytecode = std::vector<std::byte>(entry.bytecode.begin(), entry.bytecode.end()),
                .entryPoint = std::string(entry.entryPoint),
                .specializationConstants = std::move(*specializationConstants),
            });
        }
        const std::scoped_lock lock(mutex_);
        return libraries_.emplace(std::move(record));
    }

    Status VulkanShaderManager::destroyShaderLibrary(ShaderLibraryHandle shaderLibrary)
    {
        const std::scoped_lock lock(mutex_);
        if (!libraries_.erase(shaderLibrary))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader library handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<VulkanShaderManager::ShaderBytecodeSource> VulkanShaderManager::resolveShaderBytecode(
        const ShaderModuleDesc& desc, const std::vector<SpecializationConstant>& specializationConstants)
    {
        if (!desc.library.isValid())
        {
            return ShaderBytecodeSource{.library = nullptr, .bytecode = desc.bytecode};
        }
        const std::scoped_lock lock(mutex_);
        auto* foundLibrary = libraries_.find(desc.library);
        if (foundLibrary == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shader module references an invalid or stale shader library.",
            });
        }
        const auto foundEntry =
            std::ranges::find_if((*foundLibrary)->entries, [&](const ShaderLibraryEntryRecord& entry) {
                return entry.stage == desc.stage && entry.entryPoint == desc.entryPoint &&
                       entry.specializationConstants == specializationConstants;
            });
        if (foundEntry == (*foundLibrary)->entries.end())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader library does not contain the requested stage, entry point, and specialization values.",
            });
        }
        return ShaderBytecodeSource{.library = *foundLibrary, .bytecode = foundEntry->bytecode};
    }

    Result<ShaderModuleHandle> VulkanShaderManager::createShaderModule(const ShaderModuleDesc& desc)
    {
        if (auto source = validateShaderModuleSource(desc); !source)
        {
            return std::unexpected(std::move(source.error()));
        }
        auto specializationConstants = buildSpecializationSet(desc.specializationConstants);
        if (!specializationConstants)
        {
            return std::unexpected(std::move(specializationConstants.error()));
        }
        auto shaderInterface = buildShaderInterface(desc.stage, desc.shaderInterface, capabilities_);
        if (!shaderInterface)
        {
            return std::unexpected(std::move(shaderInterface.error()));
        }
        auto bytecodeSource = resolveShaderBytecode(desc, *specializationConstants);
        if (!bytecodeSource)
        {
            return std::unexpected(std::move(bytecodeSource.error()));
        }
        const auto bytecode = bytecodeSource->bytecode;
        auto words = copySpirvWords(bytecode);
        if (!words)
        {
            return std::unexpected(std::move(words.error()));
        }

        auto reflected = reflectSpirv(*words, desc.entryPoint, desc.stage, *specializationConstants);
        if (!reflected)
        {
            return std::unexpected(std::move(reflected.error()));
        }
        auto matched = compareInterface(*shaderInterface, *reflected);
        if (!matched)
        {
            return std::unexpected(std::move(matched.error()));
        }
        auto features = validateSpirvFeatureRequirements(*words, capabilities_, maintenance4_, features_);
        if (!features)
        {
            return std::unexpected(std::move(features.error()));
        }
        auto stores = validatePreRasterizationStores(desc.stage, *reflected, features_);
        if (!stores)
        {
            return std::unexpected(std::move(stores.error()));
        }
        auto specialization = validateSpecialization(*specializationConstants, *reflected);
        if (!specialization)
        {
            return std::unexpected(std::move(specialization.error()));
        }
        auto module = createNativeShaderModule(device_, table_, *words);
        if (!module)
        {
            return std::unexpected(std::move(module.error()));
        }

        ShaderModuleRecord record;
        record.stage = desc.stage;
        record.entryPoint = std::string(desc.entryPoint);
        record.label = std::string(desc.label);
        record.shaderInterface.bindings = std::move(shaderInterface->bindings);
        record.shaderInterface.pushConstants = std::move(shaderInterface->pushConstants);
        record.shaderInterface.vertexInputLocations = std::move(shaderInterface->vertexInputLocations);
        record.shaderInterface.workgroupSize = shaderInterface->workgroupSize;
        record.shaderInterface.shaderRecordConstants = shaderInterface->shaderRecordConstants;
        record.specializationConstants = std::move(*specializationConstants);
        record.stageOutputs = std::move(reflected->stageOutputs);
        record.stageInputs = std::move(reflected->stageInputs);
        record.layout = shaderLayoutProbe(*reflected);
        record.writesPointSize = reflected->writesPointSize;
        record.library = std::move(bytecodeSource->library);
        if (!desc.library.isValid())
        {
            record.directBytecode.assign(bytecode.begin(), bytecode.end());
        }
        record.keepAlive = std::make_shared<ShaderKeepAlive>();
        record.keepAlive->device = device_;
        record.keepAlive->table = table_;
        record.keepAlive->module = *module;
        if (retainSpirv_)
        {
            record.keepAlive->spirv = std::move(*words);
        }
        if (!record.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_SHADER_MODULE,
                                reinterpret_cast<std::uint64_t>(*module),
                                record.label.c_str());
        }
        const std::scoped_lock lock(mutex_);
        return modules_.emplace(std::move(record));
    }

    Result<ShaderInterfaceSnapshot> VulkanShaderManager::shaderModuleInterface(ShaderModuleHandle shaderModule) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* found = modules_.find(shaderModule);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader module handle is invalid or stale.",
            });
        }
        return ShaderInterfaceSnapshot{
            .stage = found->stage,
            .bindings = found->shaderInterface.bindings,
            .pushConstants = found->shaderInterface.pushConstants,
            .vertexInputLocations = found->shaderInterface.vertexInputLocations,
            .workgroupSize = found->shaderInterface.workgroupSize,
            .shaderRecordConstants = found->shaderInterface.shaderRecordConstants,
        };
    }

    Status VulkanShaderManager::destroyShaderModule(ShaderModuleHandle shaderModule)
    {
        const std::scoped_lock lock(mutex_);
        if (!modules_.erase(shaderModule))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader module handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<VulkanShaderManager::NativeShaderModule> VulkanShaderManager::nativeShaderModule(
        ShaderModuleHandle shaderModule) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* found = modules_.find(shaderModule);
        if (found == nullptr || found->keepAlive == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader module handle is invalid or stale.",
            });
        }
        return NativeShaderModule{
            .owner = found->keepAlive,
            .module = found->keepAlive->module,
            .spirv = found->keepAlive->spirv,
            .stage = found->stage,
            .entryPoint = found->entryPoint,
            .bindings = found->shaderInterface.bindings,
            .pushConstants = found->shaderInterface.pushConstants,
            .vertexInputLocations = found->shaderInterface.vertexInputLocations,
            .specializationConstants = found->specializationConstants,
            .shaderRecordConstants = found->shaderInterface.shaderRecordConstants,
            .writesPointSize = found->writesPointSize,
            .stageOutputs = found->stageOutputs,
            .stageInputs = found->stageInputs,
        };
    }

#ifdef RACHIS_TESTING
    Result<VulkanShaderLayoutProbe> VulkanShaderManager::queryLayout(ShaderModuleHandle shaderModule) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* found = modules_.find(shaderModule);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader module handle is invalid or stale.",
            });
        }
        return found->layout;
    }

    Result<VulkanShaderLayoutProbe> queryVulkanShaderLayout(ShaderModuleHandle shaderModule)
    {
        const std::scoped_lock lock(boundShaderMutex);
        if (boundShaders == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No Vulkan shader manager is bound for RACHIS_TESTING.",
            });
        }
        return boundShaders->queryLayout(shaderModule);
    }

    Result<VulkanStageInterfaceProbe> reflectVulkanStageInterface(std::span<const std::byte> bytecode,
                                                                  std::string_view entryPoint,
                                                                  ShaderStage stage)
    {
        auto words = copySpirvWords(bytecode);
        if (!words)
        {
            return std::unexpected(std::move(words.error()));
        }
        auto reflected = reflectSpirv(*words, entryPoint, stage);
        if (!reflected)
        {
            return std::unexpected(std::move(reflected.error()));
        }
        return VulkanStageInterfaceProbe{
            .outputs = std::move(reflected->stageOutputs),
            .inputs = std::move(reflected->stageInputs),
        };
    }
#endif

} // namespace rachis::vulkan
