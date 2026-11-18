#include "Rachis/Device.h"
#include "RhiTestSupport.h"
#include "ShaderInt64AtomicStorageBuffer.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/VulkanShaders.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#ifndef RACHIS_TESTING
#error Vulkan shader-module tests require RACHIS_TESTING.
#endif

namespace
{

    [[nodiscard]] rachis::DeviceDesc vulkanDesc(
        rachis::ValidationLevel validation = rachis::ValidationLevel::standard) noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = validation,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadBytes(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        return stream ? bytes : std::vector<std::byte>{};
    }

    [[nodiscard]] rachis::ShaderInterfaceDesc cubeComputeInterface(
        std::array<rachis::ShaderBindingRequirement, 2>& bindings,
        std::array<rachis::ShaderPushConstantRequirement, 1>& push) noexcept
    {
        bindings = {
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 0,
                .type = rachis::ShaderResourceType::readOnlyStorageBuffer,
                .count = 1,
            },
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 1,
                .type = rachis::ShaderResourceType::storageBuffer,
                .count = 1,
            },
        };
        push = {rachis::ShaderPushConstantRequirement{.offset = 0, .size = 16}};
        return rachis::ShaderInterfaceDesc{
            .bindings = bindings,
            .pushConstants = push,
            .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
        };
    }

    [[nodiscard]] bool rejectsAsInvalidArgument(rachis::Device& device,
                                                std::span<const std::byte> bytecode,
                                                std::string_view entryPoint,
                                                const rachis::ShaderInterfaceDesc& shaderInterface)
    {
        const std::array entries{rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = entryPoint,
        }};
        auto library = device.createShaderLibrary({.entries = entries});
        auto module = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = entryPoint,
            .shaderInterface = shaderInterface,
        });
        if (library)
        {
            CHECK(device.destroyShaderLibrary(*library).has_value());
        }
        if (module)
        {
            CHECK(device.destroyShaderModule(*module).has_value());
        }
        return !library && library.error().code == rachis::ErrorCode::invalidArgument && !module &&
               module.error().code == rachis::ErrorCode::invalidArgument;
    }

    constexpr std::uint32_t spirvMagicNumber = 0x07230203U;
    constexpr std::uint32_t spirvVersion1_0 = 0x00010000U;
    constexpr std::uint32_t spirvVersion1_3 = 0x00010300U;
    constexpr std::uint32_t spirvVersion1_4 = 0x00010400U;
    constexpr std::size_t spirvHeaderIdBoundWord = 3;
    constexpr std::uint32_t spirvOpName = 5;
    constexpr std::uint32_t spirvOpString = 7;
    constexpr std::uint32_t spirvOpLine = 8;
    constexpr std::uint32_t spirvOpExtension = 10;
    constexpr std::uint32_t spirvOpMemoryModel = 14;
    constexpr std::uint32_t spirvOpEntryPoint = 15;
    constexpr std::uint32_t spirvOpExecutionMode = 16;
    constexpr std::uint32_t spirvOpCapability = 17;
    constexpr std::uint32_t spirvOpTypeVoid = 19;
    constexpr std::uint32_t spirvOpTypeBool = 20;
    constexpr std::uint32_t spirvOpTypeInt = 21;
    constexpr std::uint32_t spirvOpTypeFloat = 22;
    constexpr std::uint32_t spirvOpTypeVector = 23;
    constexpr std::uint32_t spirvOpTypeMatrix = 24;
    constexpr std::uint32_t spirvOpTypeImage = 25;
    constexpr std::uint32_t spirvOpTypeArray = 28;
    constexpr std::uint32_t spirvOpTypeRuntimeArray = 29;
    constexpr std::uint32_t spirvOpTypeStruct = 30;
    constexpr std::uint32_t spirvOpTypePointer = 32;
    constexpr std::uint32_t spirvOpTypeFunction = 33;
    constexpr std::uint32_t spirvOpConstant = 43;
    constexpr std::uint32_t spirvOpConstantComposite = 44;
    constexpr std::uint32_t spirvOpSpecConstantTrue = 48;
    constexpr std::uint32_t spirvOpSpecConstant = 50;
    constexpr std::uint32_t spirvOpFunction = 54;
    constexpr std::uint32_t spirvOpFunctionEnd = 56;
    constexpr std::uint32_t spirvOpFunctionCall = 57;
    constexpr std::uint32_t spirvOpVariable = 59;
    constexpr std::uint32_t spirvOpLoad = 61;
    constexpr std::uint32_t spirvOpStore = 62;
    constexpr std::uint32_t spirvOpCopyMemory = 63;
    constexpr std::uint32_t spirvOpAccessChain = 65;
    constexpr std::uint32_t spirvOpDecorate = 71;
    constexpr std::uint32_t spirvOpMemberDecorate = 72;
    constexpr std::uint32_t spirvOpConvertFToU = 109;
    constexpr std::uint32_t spirvOpUConvert = 113;
    constexpr std::uint32_t spirvOpBitcast = 124;
    constexpr std::uint32_t spirvOpFAdd = 129;
    constexpr std::uint32_t spirvOpSelect = 169;
    constexpr std::uint32_t spirvOpAtomicIAdd = 234;
    constexpr std::uint32_t spirvOpLabel = 248;
    constexpr std::uint32_t spirvOpReturn = 253;
    constexpr std::uint32_t spirvCapabilityShader = 1;
    constexpr std::uint32_t spirvCapabilityFloat16 = 9;
    constexpr std::uint32_t spirvCapabilityFloat64 = 10;
    constexpr std::uint32_t spirvCapabilityInt64 = 11;
    constexpr std::uint32_t spirvCapabilityInt64Atomics = 12;
    constexpr std::uint32_t spirvCapabilityInt16 = 22;
    constexpr std::uint32_t spirvCapabilityImageGatherExtended = 25;
    constexpr std::uint32_t spirvCapabilityImageCubeArray = 34;
    constexpr std::uint32_t spirvCapabilitySampleRateShading = 35;
    constexpr std::uint32_t spirvCapabilityInt8 = 39;
    constexpr std::uint32_t spirvCapabilityMinLod = 42;
    constexpr std::uint32_t spirvCapabilityMultiViewport = 57;
    constexpr std::uint32_t spirvCapabilityDrawParameters = 4427;
    constexpr std::uint32_t spirvCapabilityStorageBuffer16BitAccess = 4433;
    constexpr std::uint32_t spirvCapabilityUniformAndStorageBuffer16BitAccess = 4434;
    constexpr std::uint32_t spirvCapabilityStoragePushConstant16 = 4435;
    constexpr std::uint32_t spirvCapabilityStorageInputOutput16 = 4436;
    constexpr std::uint32_t spirvCapabilityRayQueryKHR = 4472;
    constexpr std::uint32_t spirvCapabilityRayTracingKHR = 4479;
    constexpr std::uint32_t spirvCapabilityMeshShadingEXT = 5283;
    constexpr std::uint32_t spirvAddressingLogical = 0;
    constexpr std::uint32_t spirvMemoryModelGlsl450 = 1;
    constexpr std::uint32_t spirvExecutionModelVertex = 0;
    constexpr std::uint32_t spirvExecutionModelTessellationControl = 1;
    constexpr std::uint32_t spirvExecutionModelTessellationEvaluation = 2;
    constexpr std::uint32_t spirvExecutionModelGeometry = 3;
    constexpr std::uint32_t spirvExecutionModelFragment = 4;
    constexpr std::uint32_t spirvExecutionModelGlCompute = 5;
    constexpr std::uint32_t spirvExecutionModeLocalSize = 17;
    constexpr std::uint32_t spirvStorageClassUniformConstant = 0;
    constexpr std::uint32_t spirvStorageClassInput = 1;
    constexpr std::uint32_t spirvStorageClassUniform = 2;
    constexpr std::uint32_t spirvStorageClassOutput = 3;
    constexpr std::uint32_t spirvStorageClassWorkgroup = 4;
    constexpr std::uint32_t spirvStorageClassFunction = 7;
    constexpr std::uint32_t spirvStorageClassStorageBuffer = 12;
    constexpr std::uint32_t spirvDim2D = 1;
    constexpr std::uint32_t spirvImageFormatR32f = 3;
    constexpr std::uint32_t spirvDecorationRelaxedPrecision = 0;
    constexpr std::uint32_t spirvDecorationSpecId = 1;
    constexpr std::uint32_t spirvDecorationBlock = 2;
    constexpr std::uint32_t spirvDecorationBufferBlock = 3;
    constexpr std::uint32_t spirvDecorationArrayStride = 6;
    constexpr std::uint32_t spirvDecorationBuiltIn = 11;
    constexpr std::uint32_t spirvDecorationNonWritable = 24;
    constexpr std::uint32_t spirvDecorationLocation = 30;
    constexpr std::uint32_t spirvDecorationComponent = 31;
    constexpr std::uint32_t spirvDecorationBinding = 33;
    constexpr std::uint32_t spirvDecorationDescriptorSet = 34;
    constexpr std::uint32_t spirvDecorationOffset = 35;
    constexpr std::uint32_t spirvBuiltInPosition = 0;
    constexpr std::uint32_t spirvBuiltInPointSize = 1;
    constexpr std::uint32_t spirvBuiltInVertexIndex = 42;
    constexpr std::uint32_t spirvScopeWorkgroup = 2;
    constexpr std::uint32_t spirvMemorySemanticsRelaxed = 0;
    constexpr std::uint32_t spirvFunctionControlNone = 0;
    constexpr std::uint32_t spirvFloatOneBits = 0x3F800000U;

    [[nodiscard]] std::vector<std::uint32_t> spirvInstruction(std::uint32_t opcode,
                                                              std::initializer_list<std::uint32_t> operands)
    {
        std::vector<std::uint32_t> words{(static_cast<std::uint32_t>(operands.size() + 1U) << 16U) | opcode};
        words.insert(words.end(), operands.begin(), operands.end());
        return words;
    }

    [[nodiscard]] std::vector<std::uint32_t> spirvInstructionWithString(std::uint32_t opcode,
                                                                        std::initializer_list<std::uint32_t> operands,
                                                                        std::string_view text)
    {
        std::vector<std::uint32_t> stringWords((text.size() / sizeof(std::uint32_t)) + 1U, 0U);
        std::memcpy(stringWords.data(), text.data(), text.size());
        auto words = spirvInstruction(opcode, operands);
        words.insert(words.end(), stringWords.begin(), stringWords.end());
        words[0] = (static_cast<std::uint32_t>(words.size()) << 16U) | opcode;
        return words;
    }

    [[nodiscard]] std::vector<std::uint32_t> joinSpirvWords(std::initializer_list<std::span<const std::uint32_t>> parts)
    {
        std::vector<std::uint32_t> words;
        for (const auto part : parts)
        {
            words.insert(words.end(), part.begin(), part.end());
        }
        return words;
    }

    [[nodiscard]] std::vector<std::byte> joinSpirvBytes(std::initializer_list<std::span<const std::uint32_t>> parts)
    {
        const auto words = joinSpirvWords(parts);
        const auto bytes = std::as_bytes(std::span(words));
        return {bytes.begin(), bytes.end()};
    }

    struct CallingComputeModule final
    {
        static constexpr std::uint32_t callerId = 1;
        static constexpr std::uint32_t voidTypeId = 2;
        static constexpr std::uint32_t functionTypeId = 3;
        static constexpr std::uint32_t callerLabelId = 4;
        static constexpr std::uint32_t callResultId = 5;
        static constexpr std::uint32_t calleeId = 6;
        static constexpr std::uint32_t calleeLabelId = 7;
        static constexpr std::uint32_t idBound = 8;

        std::vector<std::uint32_t> header{spirvMagicNumber, spirvVersion1_0, 0U, idBound, 0U};
        std::vector<std::uint32_t> declarations = joinSpirvWords({
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
            spirvInstructionWithString(spirvOpEntryPoint, {spirvExecutionModelGlCompute, callerId}, "main"),
            spirvInstruction(spirvOpExecutionMode, {callerId, spirvExecutionModeLocalSize, 1U, 1U, 1U}),
            spirvInstruction(spirvOpTypeVoid, {voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {functionTypeId, voidTypeId}),
        });
        std::vector<std::uint32_t> callerBody = joinSpirvWords({
            spirvInstruction(spirvOpFunction, {voidTypeId, callerId, spirvFunctionControlNone, functionTypeId}),
            spirvInstruction(spirvOpLabel, {callerLabelId}),
            spirvInstruction(spirvOpFunctionCall, {voidTypeId, callResultId, calleeId}),
            spirvInstruction(spirvOpReturn, {}),
        });
        std::vector<std::uint32_t> callerEnd = spirvInstruction(spirvOpFunctionEnd, {});
        std::vector<std::uint32_t> calleeBody = joinSpirvWords({
            spirvInstruction(spirvOpFunction, {voidTypeId, calleeId, spirvFunctionControlNone, functionTypeId}),
            spirvInstruction(spirvOpLabel, {calleeLabelId}),
        });
        std::vector<std::uint32_t> calleeReturn = spirvInstruction(spirvOpReturn, {});
        std::vector<std::uint32_t> calleeEnd = spirvInstruction(spirvOpFunctionEnd, {});
    };

    [[nodiscard]] std::vector<std::uint32_t> emptyComputeFunction(std::uint32_t voidTypeId,
                                                                  std::uint32_t functionId,
                                                                  std::uint32_t functionTypeId,
                                                                  std::uint32_t labelId)
    {
        return joinSpirvWords({
            spirvInstruction(spirvOpFunction, {voidTypeId, functionId, spirvFunctionControlNone, functionTypeId}),
            spirvInstruction(spirvOpLabel, {labelId}),
            spirvInstruction(spirvOpReturn, {}),
            spirvInstruction(spirvOpFunctionEnd, {}),
        });
    }

    struct MultiEntryComputeModule final
    {
        static constexpr std::uint32_t mainId = 1;
        static constexpr std::uint32_t voidTypeId = 2;
        static constexpr std::uint32_t functionTypeId = 3;
        static constexpr std::uint32_t mainLabelId = 4;
        static constexpr std::uint32_t otherId = 5;
        static constexpr std::uint32_t otherLabelId = 6;
        static constexpr std::uint32_t helperId = 7;
        static constexpr std::uint32_t helperLabelId = 8;
        static constexpr std::uint32_t floatTypeId = 9;
        static constexpr std::uint32_t oneId = 10;
        static constexpr std::uint32_t sumId = 11;
        static constexpr std::uint32_t idBound = 12;

        std::vector<std::uint32_t> header{spirvMagicNumber, spirvVersion1_0, 0U, idBound, 0U};
        std::vector<std::uint32_t> capabilities = joinSpirvWords({
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
        });
        std::vector<std::uint32_t> mainEntryPoint =
            spirvInstructionWithString(spirvOpEntryPoint, {spirvExecutionModelGlCompute, mainId}, "main");
        std::vector<std::uint32_t> otherEntryPoint =
            spirvInstructionWithString(spirvOpEntryPoint, {spirvExecutionModelGlCompute, otherId}, "other");
        std::vector<std::uint32_t> mainExecutionMode =
            spirvInstruction(spirvOpExecutionMode, {mainId, spirvExecutionModeLocalSize, 1U, 1U, 1U});
        std::vector<std::uint32_t> otherExecutionMode =
            spirvInstruction(spirvOpExecutionMode, {otherId, spirvExecutionModeLocalSize, 1U, 1U, 1U});
        std::vector<std::uint32_t> helperName = spirvInstructionWithString(spirvOpName, {helperId}, "helper");
        std::vector<std::uint32_t> sumDecoration =
            spirvInstruction(spirvOpDecorate, {sumId, spirvDecorationRelaxedPrecision});
        std::vector<std::uint32_t> types = joinSpirvWords({
            spirvInstruction(spirvOpTypeVoid, {voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {functionTypeId, voidTypeId}),
            spirvInstruction(spirvOpTypeFloat, {floatTypeId, 32U}),
            spirvInstruction(spirvOpConstant, {floatTypeId, oneId, spirvFloatOneBits}),
        });
        std::vector<std::uint32_t> mainFunction = emptyComputeFunction(voidTypeId, mainId, functionTypeId, mainLabelId);
        std::vector<std::uint32_t> otherFunction =
            emptyComputeFunction(voidTypeId, otherId, functionTypeId, otherLabelId);
        std::vector<std::uint32_t> helperFunction = joinSpirvWords({
            spirvInstruction(spirvOpFunction, {voidTypeId, helperId, spirvFunctionControlNone, functionTypeId}),
            spirvInstruction(spirvOpLabel, {helperLabelId}),
            spirvInstruction(spirvOpFAdd, {floatTypeId, sumId, oneId, oneId}),
            spirvInstruction(spirvOpReturn, {}),
            spirvInstruction(spirvOpFunctionEnd, {}),
        });
    };

    [[nodiscard]] std::vector<std::byte> computeModuleDeclaringCapability(std::uint32_t capability,
                                                                          std::string_view extension = {})
    {
        constexpr std::uint32_t mainId = 1;
        constexpr std::uint32_t voidTypeId = 2;
        constexpr std::uint32_t functionTypeId = 3;
        constexpr std::uint32_t labelId = 4;
        constexpr std::uint32_t idBound = 5;
        const auto version = extension.empty() ? spirvVersion1_0 : spirvVersion1_4;
        const std::vector<std::uint32_t> header{spirvMagicNumber, version, 0U, idBound, 0U};
        const auto extensionDeclaration = extension.empty()
                                              ? std::vector<std::uint32_t>{}
                                              : spirvInstructionWithString(spirvOpExtension, {}, extension);
        return joinSpirvBytes({
            header,
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            spirvInstruction(spirvOpCapability, {capability}),
            extensionDeclaration,
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
            spirvInstructionWithString(spirvOpEntryPoint, {spirvExecutionModelGlCompute, mainId}, "main"),
            spirvInstruction(spirvOpExecutionMode, {mainId, spirvExecutionModeLocalSize, 1U, 1U, 1U}),
            spirvInstruction(spirvOpTypeVoid, {voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {functionTypeId, voidTypeId}),
            emptyComputeFunction(voidTypeId, mainId, functionTypeId, labelId),
        });
    }

    struct GroupSharedInt64AtomicModule final
    {
        static constexpr std::uint32_t mainId = 1;
        static constexpr std::uint32_t voidTypeId = 2;
        static constexpr std::uint32_t functionTypeId = 3;
        static constexpr std::uint32_t labelId = 4;
        static constexpr std::uint32_t uintTypeId = 5;
        static constexpr std::uint32_t ulongTypeId = 6;
        static constexpr std::uint32_t workgroupPointerTypeId = 7;
        static constexpr std::uint32_t accumulatorId = 8;
        static constexpr std::uint32_t scopeId = 9;
        static constexpr std::uint32_t semanticsId = 10;
        static constexpr std::uint32_t oneId = 11;
        static constexpr std::uint32_t atomicResultId = 12;
        static constexpr std::uint32_t fileId = 13;
        static constexpr std::uint32_t functionPointerTypeId = 14;
        static constexpr std::uint32_t localId = 15;
        static constexpr std::uint32_t idBound = 16;

        std::vector<std::uint32_t> header{spirvMagicNumber, spirvVersion1_0, 0U, idBound, 0U};
        std::vector<std::uint32_t> declarations = joinSpirvWords({
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            spirvInstruction(spirvOpCapability, {spirvCapabilityInt64}),
            spirvInstruction(spirvOpCapability, {spirvCapabilityInt64Atomics}),
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
            spirvInstructionWithString(spirvOpEntryPoint, {spirvExecutionModelGlCompute, mainId}, "main"),
            spirvInstruction(spirvOpExecutionMode, {mainId, spirvExecutionModeLocalSize, 1U, 1U, 1U}),
            spirvInstructionWithString(spirvOpString, {fileId}, "atomic.hlsl"),
            spirvInstruction(spirvOpTypeVoid, {voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {functionTypeId, voidTypeId}),
            spirvInstruction(spirvOpTypeInt, {uintTypeId, 32U, 0U}),
            spirvInstruction(spirvOpTypeInt, {ulongTypeId, 64U, 0U}),
            spirvInstruction(spirvOpTypePointer, {workgroupPointerTypeId, spirvStorageClassWorkgroup, ulongTypeId}),
            spirvInstruction(spirvOpTypePointer, {functionPointerTypeId, spirvStorageClassFunction, ulongTypeId}),
            spirvInstruction(spirvOpVariable, {workgroupPointerTypeId, accumulatorId, spirvStorageClassWorkgroup}),
            spirvInstruction(spirvOpConstant, {uintTypeId, scopeId, spirvScopeWorkgroup}),
            spirvInstruction(spirvOpConstant, {uintTypeId, semanticsId, spirvMemorySemanticsRelaxed}),
            spirvInstruction(spirvOpConstant, {ulongTypeId, oneId, 1U, 0U}),
        });
        std::vector<std::uint32_t> functionStart = joinSpirvWords({
            spirvInstruction(spirvOpFunction, {voidTypeId, mainId, spirvFunctionControlNone, functionTypeId}),
            spirvInstruction(spirvOpLabel, {labelId}),
            spirvInstruction(spirvOpVariable, {functionPointerTypeId, localId, spirvStorageClassFunction}),
        });
        std::vector<std::uint32_t> atomicAndReturn = joinSpirvWords({
            spirvInstruction(spirvOpAtomicIAdd,
                             {ulongTypeId, atomicResultId, accumulatorId, scopeId, semanticsId, oneId}),
            spirvInstruction(spirvOpReturn, {}),
            spirvInstruction(spirvOpFunctionEnd, {}),
        });
    };

    [[nodiscard]] std::vector<std::byte> int64AtomicsModuleWith32BitGroupSharedAtomic()
    {
        constexpr std::uint32_t mainId = 1;
        constexpr std::uint32_t voidTypeId = 2;
        constexpr std::uint32_t functionTypeId = 3;
        constexpr std::uint32_t labelId = 4;
        constexpr std::uint32_t uintTypeId = 5;
        constexpr std::uint32_t workgroupPointerTypeId = 6;
        constexpr std::uint32_t counterId = 7;
        constexpr std::uint32_t scopeId = 8;
        constexpr std::uint32_t semanticsId = 9;
        constexpr std::uint32_t oneId = 10;
        constexpr std::uint32_t atomicResultId = 11;
        constexpr std::uint32_t idBound = 12;
        const std::vector<std::uint32_t> header{spirvMagicNumber, spirvVersion1_0, 0U, idBound, 0U};
        return joinSpirvBytes({
            header,
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            spirvInstruction(spirvOpCapability, {spirvCapabilityInt64}),
            spirvInstruction(spirvOpCapability, {spirvCapabilityInt64Atomics}),
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
            spirvInstructionWithString(spirvOpEntryPoint, {spirvExecutionModelGlCompute, mainId}, "main"),
            spirvInstruction(spirvOpExecutionMode, {mainId, spirvExecutionModeLocalSize, 1U, 1U, 1U}),
            spirvInstruction(spirvOpTypeVoid, {voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {functionTypeId, voidTypeId}),
            spirvInstruction(spirvOpTypeInt, {uintTypeId, 32U, 0U}),
            spirvInstruction(spirvOpTypePointer, {workgroupPointerTypeId, spirvStorageClassWorkgroup, uintTypeId}),
            spirvInstruction(spirvOpVariable, {workgroupPointerTypeId, counterId, spirvStorageClassWorkgroup}),
            spirvInstruction(spirvOpConstant, {uintTypeId, scopeId, spirvScopeWorkgroup}),
            spirvInstruction(spirvOpConstant, {uintTypeId, semanticsId, spirvMemorySemanticsRelaxed}),
            spirvInstruction(spirvOpConstant, {uintTypeId, oneId, 1U}),
            spirvInstruction(spirvOpFunction, {voidTypeId, mainId, spirvFunctionControlNone, functionTypeId}),
            spirvInstruction(spirvOpLabel, {labelId}),
            spirvInstruction(spirvOpAtomicIAdd, {uintTypeId, atomicResultId, counterId, scopeId, semanticsId, oneId}),
            spirvInstruction(spirvOpReturn, {}),
            spirvInstruction(spirvOpFunctionEnd, {}),
        });
    }

    struct VertexInputModule final
    {
        static constexpr std::uint32_t mainId = 1;
        static constexpr std::uint32_t voidTypeId = 2;
        static constexpr std::uint32_t functionTypeId = 3;
        static constexpr std::uint32_t labelId = 4;
        static constexpr std::uint32_t uintTypeId = 5;
        static constexpr std::uint32_t twoId = 6;
        static constexpr std::uint32_t floatTypeId = 7;
        static constexpr std::uint32_t float2TypeId = 8;
        static constexpr std::uint32_t float4TypeId = 9;
        static constexpr std::uint32_t float4x4TypeId = 10;
        static constexpr std::uint32_t float4Array2TypeId = 11;
        static constexpr std::uint32_t doubleTypeId = 12;
        static constexpr std::uint32_t double2TypeId = 13;
        static constexpr std::uint32_t double3TypeId = 14;
        static constexpr std::uint32_t double4TypeId = 15;
        static constexpr std::uint32_t double4x2TypeId = 16;
        static constexpr std::uint32_t structTypeId = 17;
        static constexpr std::uint32_t firstInputId = 18;
    };

    struct VertexInputVariable final
    {
        std::uint32_t typeId = 0;
        std::uint32_t location = 0;
        std::uint32_t component = 0;
        std::optional<std::uint32_t> builtIn = {};
    };

    [[nodiscard]] std::vector<std::byte> vertexModuleWithInputs(std::span<const VertexInputVariable> inputs)
    {
        using Module = VertexInputModule;
        std::vector<std::uint32_t> interfaceIds;
        std::vector<std::uint32_t> decorations;
        std::vector<std::uint32_t> variables;
        for (std::size_t index = 0; index < inputs.size(); ++index)
        {
            const auto& input = inputs[index];
            const auto pointerTypeId = Module::firstInputId + static_cast<std::uint32_t>(2 * index);
            const auto variableId = pointerTypeId + 1U;
            interfaceIds.push_back(variableId);
            const auto decoration =
                input.builtIn
                    ? spirvInstruction(spirvOpDecorate, {variableId, spirvDecorationBuiltIn, *input.builtIn})
                    : spirvInstruction(spirvOpDecorate, {variableId, spirvDecorationLocation, input.location});
            decorations.insert(decorations.end(), decoration.begin(), decoration.end());
            if (input.component != 0)
            {
                const auto component =
                    spirvInstruction(spirvOpDecorate, {variableId, spirvDecorationComponent, input.component});
                decorations.insert(decorations.end(), component.begin(), component.end());
            }
            const auto declarations = joinSpirvWords({
                spirvInstruction(spirvOpTypePointer, {pointerTypeId, spirvStorageClassInput, input.typeId}),
                spirvInstruction(spirvOpVariable, {pointerTypeId, variableId, spirvStorageClassInput}),
            });
            variables.insert(variables.end(), declarations.begin(), declarations.end());
        }
        const auto idBound = Module::firstInputId + static_cast<std::uint32_t>(2 * inputs.size());
        const std::vector<std::uint32_t> header{spirvMagicNumber, spirvVersion1_0, 0U, idBound, 0U};
        auto entryPoint =
            spirvInstructionWithString(spirvOpEntryPoint, {spirvExecutionModelVertex, Module::mainId}, "main");
        entryPoint.insert(entryPoint.end(), interfaceIds.begin(), interfaceIds.end());
        entryPoint[0] = (static_cast<std::uint32_t>(entryPoint.size()) << 16U) | spirvOpEntryPoint;
        return joinSpirvBytes({
            header,
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            spirvInstruction(spirvOpCapability, {spirvCapabilityFloat64}),
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
            entryPoint,
            decorations,
            spirvInstruction(spirvOpTypeVoid, {Module::voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {Module::functionTypeId, Module::voidTypeId}),
            spirvInstruction(spirvOpTypeInt, {Module::uintTypeId, 32U, 0U}),
            spirvInstruction(spirvOpConstant, {Module::uintTypeId, Module::twoId, 2U}),
            spirvInstruction(spirvOpTypeFloat, {Module::floatTypeId, 32U}),
            spirvInstruction(spirvOpTypeVector, {Module::float2TypeId, Module::floatTypeId, 2U}),
            spirvInstruction(spirvOpTypeVector, {Module::float4TypeId, Module::floatTypeId, 4U}),
            spirvInstruction(spirvOpTypeMatrix, {Module::float4x4TypeId, Module::float4TypeId, 4U}),
            spirvInstruction(spirvOpTypeArray, {Module::float4Array2TypeId, Module::float4TypeId, Module::twoId}),
            spirvInstruction(spirvOpTypeFloat, {Module::doubleTypeId, 64U}),
            spirvInstruction(spirvOpTypeVector, {Module::double2TypeId, Module::doubleTypeId, 2U}),
            spirvInstruction(spirvOpTypeVector, {Module::double3TypeId, Module::doubleTypeId, 3U}),
            spirvInstruction(spirvOpTypeVector, {Module::double4TypeId, Module::doubleTypeId, 4U}),
            spirvInstruction(spirvOpTypeMatrix, {Module::double4x2TypeId, Module::double4TypeId, 2U}),
            spirvInstruction(spirvOpTypeStruct, {Module::structTypeId, Module::float4TypeId}),
            variables,
            emptyComputeFunction(Module::voidTypeId, Module::mainId, Module::functionTypeId, Module::labelId),
        });
    }

    struct StorageBlockModule final
    {
        static constexpr std::uint32_t mainId = 1;
        static constexpr std::uint32_t voidTypeId = 2;
        static constexpr std::uint32_t functionTypeId = 3;
        static constexpr std::uint32_t labelId = 4;
        static constexpr std::uint32_t uintTypeId = 5;
        static constexpr std::uint32_t zeroId = 6;
        static constexpr std::uint32_t fourId = 7;
        static constexpr std::uint32_t blockTypeId = 8;
        static constexpr std::uint32_t blockArrayTypeId = 9;
        static constexpr std::uint32_t blockRuntimeArrayTypeId = 10;
        static constexpr std::uint32_t variablePointerTypeId = 11;
        static constexpr std::uint32_t memberPointerTypeId = 12;
        static constexpr std::uint32_t variableId = 13;
        static constexpr std::uint32_t memberId = 14;
        static constexpr std::uint32_t idBound = 15;
    };

    enum class StorageBlockArray : std::uint8_t
    {
        none,
        fixed,
        runtime,
    };

    struct StorageBlockVariant final
    {
        std::string_view name;
        std::vector<std::uint32_t> nonWritableMembers;
        bool nonWritableVariable = false;
        bool bufferBlock = false;
        StorageBlockArray array = StorageBlockArray::none;
        rachis::ShaderResourceType reflected = rachis::ShaderResourceType::storageBuffer;
    };

    [[nodiscard]] std::vector<std::byte> storageBlockModule(const StorageBlockVariant& variant)
    {
        using Module = StorageBlockModule;
        const auto storageClass = variant.bufferBlock ? spirvStorageClassUniform : spirvStorageClassStorageBuffer;
        auto decorations = joinSpirvWords({
            spirvInstruction(spirvOpDecorate, {Module::variableId, spirvDecorationDescriptorSet, 0U}),
            spirvInstruction(spirvOpDecorate, {Module::variableId, spirvDecorationBinding, 0U}),
            spirvInstruction(
                spirvOpDecorate,
                {Module::blockTypeId, variant.bufferBlock ? spirvDecorationBufferBlock : spirvDecorationBlock}),
            spirvInstruction(spirvOpMemberDecorate, {Module::blockTypeId, 0U, spirvDecorationOffset, 0U}),
            spirvInstruction(spirvOpMemberDecorate, {Module::blockTypeId, 1U, spirvDecorationOffset, 4U}),
        });
        if (variant.nonWritableVariable)
        {
            const auto decoration = spirvInstruction(spirvOpDecorate, {Module::variableId, spirvDecorationNonWritable});
            decorations.insert(decorations.end(), decoration.begin(), decoration.end());
        }
        for (const auto member : variant.nonWritableMembers)
        {
            const auto decoration =
                spirvInstruction(spirvOpMemberDecorate, {Module::blockTypeId, member, spirvDecorationNonWritable});
            decorations.insert(decorations.end(), decoration.begin(), decoration.end());
        }
        auto variableTypeId = Module::blockTypeId;
        if (variant.array == StorageBlockArray::fixed)
        {
            variableTypeId = Module::blockArrayTypeId;
        }
        else if (variant.array == StorageBlockArray::runtime)
        {
            variableTypeId = Module::blockRuntimeArrayTypeId;
        }
        const auto memberAccess =
            variant.array == StorageBlockArray::none
                ? spirvInstruction(spirvOpAccessChain,
                                   {Module::memberPointerTypeId, Module::memberId, Module::variableId, Module::zeroId})
                : spirvInstruction(spirvOpAccessChain,
                                   {Module::memberPointerTypeId,
                                    Module::memberId,
                                    Module::variableId,
                                    Module::zeroId,
                                    Module::zeroId});
        const std::vector<std::uint32_t> header{spirvMagicNumber, spirvVersion1_0, 0U, Module::idBound, 0U};
        return joinSpirvBytes({
            header,
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
            spirvInstructionWithString(spirvOpEntryPoint, {spirvExecutionModelGlCompute, Module::mainId}, "main"),
            spirvInstruction(spirvOpExecutionMode, {Module::mainId, spirvExecutionModeLocalSize, 1U, 1U, 1U}),
            decorations,
            spirvInstruction(spirvOpTypeVoid, {Module::voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {Module::functionTypeId, Module::voidTypeId}),
            spirvInstruction(spirvOpTypeInt, {Module::uintTypeId, 32U, 0U}),
            spirvInstruction(spirvOpConstant, {Module::uintTypeId, Module::zeroId, 0U}),
            spirvInstruction(spirvOpConstant, {Module::uintTypeId, Module::fourId, 4U}),
            spirvInstruction(spirvOpTypeStruct, {Module::blockTypeId, Module::uintTypeId, Module::uintTypeId}),
            spirvInstruction(spirvOpTypeArray, {Module::blockArrayTypeId, Module::blockTypeId, Module::fourId}),
            spirvInstruction(spirvOpTypeRuntimeArray, {Module::blockRuntimeArrayTypeId, Module::blockTypeId}),
            spirvInstruction(spirvOpTypePointer, {Module::variablePointerTypeId, storageClass, variableTypeId}),
            spirvInstruction(spirvOpTypePointer, {Module::memberPointerTypeId, storageClass, Module::uintTypeId}),
            spirvInstruction(spirvOpVariable, {Module::variablePointerTypeId, Module::variableId, storageClass}),
            spirvInstruction(spirvOpFunction,
                             {Module::voidTypeId, Module::mainId, spirvFunctionControlNone, Module::functionTypeId}),
            spirvInstruction(spirvOpLabel, {Module::labelId}),
            memberAccess,
            spirvInstruction(spirvOpReturn, {}),
            spirvInstruction(spirvOpFunctionEnd, {}),
        });
    }

    struct SpecConstantModule final
    {
        static constexpr std::uint32_t mainId = 1;
        static constexpr std::uint32_t voidTypeId = 2;
        static constexpr std::uint32_t functionTypeId = 3;
        static constexpr std::uint32_t labelId = 4;
        static constexpr std::uint32_t uintTypeId = 5;
        static constexpr std::uint32_t zeroId = 6;
        static constexpr std::uint32_t oneId = 7;
        static constexpr std::uint32_t runtimeArrayTypeId = 8;
        static constexpr std::uint32_t blockTypeId = 9;
        static constexpr std::uint32_t blockPointerTypeId = 10;
        static constexpr std::uint32_t outputId = 11;
        static constexpr std::uint32_t elementPointerTypeId = 12;
        static constexpr std::uint32_t specTypeId = 13;
        static constexpr std::uint32_t specConstantId = 14;
        static constexpr std::uint32_t elementId = 15;
        static constexpr std::uint32_t convertedId = 16;
        static constexpr std::uint32_t idBound = 17;
        static constexpr std::uint32_t specId = 3;
    };

    struct SpecConstantVariant final
    {
        std::string_view name;
        std::uint32_t typeOpcode = spirvOpTypeInt;
        std::vector<std::uint32_t> typeOperands;
        std::uint32_t capability = 0;
        std::vector<std::uint32_t> defaultWords;
        std::uint32_t conversionOpcode = 0;
    };

    [[nodiscard]] std::vector<std::uint32_t> spirvInstructionWithOperands(std::uint32_t opcode,
                                                                          std::span<const std::uint32_t> operands)
    {
        std::vector<std::uint32_t> words{(static_cast<std::uint32_t>(operands.size() + 1U) << 16U) | opcode};
        words.insert(words.end(), operands.begin(), operands.end());
        return words;
    }

    [[nodiscard]] std::vector<std::byte> specConstantModule(const SpecConstantVariant& variant)
    {
        using Module = SpecConstantModule;
        const bool sharesUintType =
            variant.typeOpcode == spirvOpTypeInt && std::ranges::equal(variant.typeOperands, std::array{32U, 0U});
        const auto specTypeId = sharesUintType ? Module::uintTypeId : Module::specTypeId;
        const auto capability = variant.capability == 0 ? std::vector<std::uint32_t>{}
                                                        : spirvInstruction(spirvOpCapability, {variant.capability});
        std::vector<std::uint32_t> specType;
        if (!sharesUintType)
        {
            std::vector<std::uint32_t> operands{specTypeId};
            operands.insert(operands.end(), variant.typeOperands.begin(), variant.typeOperands.end());
            specType = spirvInstructionWithOperands(variant.typeOpcode, operands);
        }
        std::vector<std::uint32_t> specConstantOperands{specTypeId, Module::specConstantId};
        specConstantOperands.insert(
            specConstantOperands.end(), variant.defaultWords.begin(), variant.defaultWords.end());
        const auto specConstant = spirvInstructionWithOperands(
            variant.defaultWords.empty() ? spirvOpSpecConstantTrue : spirvOpSpecConstant, specConstantOperands);
        std::vector<std::uint32_t> conversion;
        if (variant.conversionOpcode == spirvOpSelect)
        {
            conversion = spirvInstruction(
                spirvOpSelect,
                {Module::uintTypeId, Module::convertedId, Module::specConstantId, Module::oneId, Module::zeroId});
        }
        else if (variant.conversionOpcode != 0)
        {
            conversion = spirvInstruction(variant.conversionOpcode,
                                          {Module::uintTypeId, Module::convertedId, Module::specConstantId});
        }
        const auto storedId = variant.conversionOpcode == 0 ? Module::specConstantId : Module::convertedId;
        const std::vector<std::uint32_t> header{spirvMagicNumber, spirvVersion1_3, 0U, Module::idBound, 0U};
        return joinSpirvBytes({
            header,
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            capability,
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
            spirvInstructionWithString(spirvOpEntryPoint, {spirvExecutionModelGlCompute, Module::mainId}, "main"),
            spirvInstruction(spirvOpExecutionMode, {Module::mainId, spirvExecutionModeLocalSize, 1U, 1U, 1U}),
            spirvInstruction(spirvOpDecorate, {Module::runtimeArrayTypeId, spirvDecorationArrayStride, 4U}),
            spirvInstruction(spirvOpDecorate, {Module::blockTypeId, spirvDecorationBlock}),
            spirvInstruction(spirvOpMemberDecorate, {Module::blockTypeId, 0U, spirvDecorationOffset, 0U}),
            spirvInstruction(spirvOpDecorate, {Module::outputId, spirvDecorationDescriptorSet, 0U}),
            spirvInstruction(spirvOpDecorate, {Module::outputId, spirvDecorationBinding, 0U}),
            spirvInstruction(spirvOpDecorate, {Module::specConstantId, spirvDecorationSpecId, Module::specId}),
            spirvInstruction(spirvOpTypeVoid, {Module::voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {Module::functionTypeId, Module::voidTypeId}),
            spirvInstruction(spirvOpTypeInt, {Module::uintTypeId, 32U, 0U}),
            spirvInstruction(spirvOpConstant, {Module::uintTypeId, Module::zeroId, 0U}),
            spirvInstruction(spirvOpConstant, {Module::uintTypeId, Module::oneId, 1U}),
            spirvInstruction(spirvOpTypeRuntimeArray, {Module::runtimeArrayTypeId, Module::uintTypeId}),
            spirvInstruction(spirvOpTypeStruct, {Module::blockTypeId, Module::runtimeArrayTypeId}),
            spirvInstruction(spirvOpTypePointer,
                             {Module::blockPointerTypeId, spirvStorageClassStorageBuffer, Module::blockTypeId}),
            spirvInstruction(spirvOpVariable,
                             {Module::blockPointerTypeId, Module::outputId, spirvStorageClassStorageBuffer}),
            spirvInstruction(spirvOpTypePointer,
                             {Module::elementPointerTypeId, spirvStorageClassStorageBuffer, Module::uintTypeId}),
            specType,
            specConstant,
            spirvInstruction(spirvOpFunction,
                             {Module::voidTypeId, Module::mainId, spirvFunctionControlNone, Module::functionTypeId}),
            spirvInstruction(spirvOpLabel, {Module::labelId}),
            spirvInstruction(
                spirvOpAccessChain,
                {Module::elementPointerTypeId, Module::elementId, Module::outputId, Module::zeroId, Module::zeroId}),
            conversion,
            spirvInstruction(spirvOpStore, {Module::elementId, storedId}),
            spirvInstruction(spirvOpReturn, {}),
            spirvInstruction(spirvOpFunctionEnd, {}),
        });
    }

    [[nodiscard]] std::vector<SpecConstantVariant> specConstantVariants()
    {
        constexpr std::uint32_t seven = 7;
        return {
            {.name = "bool",
             .typeOpcode = spirvOpTypeBool,
             .typeOperands = {},
             .defaultWords = {},
             .conversionOpcode = spirvOpSelect},
            {.name = "uint32", .typeOperands = {32U, 0U}, .defaultWords = {seven}},
            {.name = "int32", .typeOperands = {32U, 1U}, .defaultWords = {seven}, .conversionOpcode = spirvOpBitcast},
            {.name = "float32",
             .typeOpcode = spirvOpTypeFloat,
             .typeOperands = {32U},
             .defaultWords = {std::bit_cast<std::uint32_t>(7.0F)},
             .conversionOpcode = spirvOpConvertFToU},
            {.name = "uint8",
             .typeOperands = {8U, 0U},
             .capability = spirvCapabilityInt8,
             .defaultWords = {seven},
             .conversionOpcode = spirvOpUConvert},
            {.name = "uint16",
             .typeOperands = {16U, 0U},
             .capability = spirvCapabilityInt16,
             .defaultWords = {seven},
             .conversionOpcode = spirvOpUConvert},
            {.name = "uint64",
             .typeOperands = {64U, 0U},
             .capability = spirvCapabilityInt64,
             .defaultWords = {seven, 0U},
             .conversionOpcode = spirvOpUConvert},
            {.name = "float16",
             .typeOpcode = spirvOpTypeFloat,
             .typeOperands = {16U},
             .capability = spirvCapabilityFloat16,
             .defaultWords = {0x4700U},
             .conversionOpcode = spirvOpConvertFToU},
            {.name = "float64",
             .typeOpcode = spirvOpTypeFloat,
             .typeOperands = {64U},
             .capability = spirvCapabilityFloat64,
             .defaultWords = {0U, 0x401C0000U},
             .conversionOpcode = spirvOpConvertFToU},
        };
    }

    [[nodiscard]] SpecConstantVariant specConstantVariant(std::string_view name)
    {
        auto variants = specConstantVariants();
        const auto found = std::ranges::find(variants, name, &SpecConstantVariant::name);
        REQUIRE(found != variants.end());
        return std::move(*found);
    }

    VKAPI_ATTR VkResult VKAPI_CALL fakeCreateShaderModule(VkDevice,
                                                          const VkShaderModuleCreateInfo*,
                                                          const VkAllocationCallbacks*,
                                                          VkShaderModule* module)
    {
        *module = reinterpret_cast<VkShaderModule>(std::uintptr_t{0x5000});
        return VK_SUCCESS;
    }

    struct CapturedShaderModuleCode final
    {
        std::vector<std::byte> bytes;
        bool aligned = false;
    };

    [[nodiscard]] CapturedShaderModuleCode& capturedShaderModuleCode() noexcept
    {
        static CapturedShaderModuleCode captured;
        return captured;
    }

    VKAPI_ATTR VkResult VKAPI_CALL capturingCreateShaderModule(VkDevice,
                                                               const VkShaderModuleCreateInfo* createInfo,
                                                               const VkAllocationCallbacks*,
                                                               VkShaderModule* module)
    {
        auto& captured = capturedShaderModuleCode();
        captured.aligned = reinterpret_cast<std::uintptr_t>(createInfo->pCode) % alignof(std::uint32_t) == 0;
        const auto* code = reinterpret_cast<const std::byte*>(createInfo->pCode);
        captured.bytes.assign(code, code + createInfo->codeSize);
        *module = reinterpret_cast<VkShaderModule>(std::uintptr_t{0x5001});
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL fakeDestroyShaderModule(VkDevice, VkShaderModule, const VkAllocationCallbacks*)
    {
    }

    [[nodiscard]] std::vector<std::uint32_t> vertexEntryPoint(std::uint32_t mainId,
                                                              std::initializer_list<std::uint32_t> interfaceIds)
    {
        auto words = spirvInstructionWithString(spirvOpEntryPoint, {spirvExecutionModelVertex, mainId}, "main");
        words.insert(words.end(), interfaceIds.begin(), interfaceIds.end());
        words[0] = (static_cast<std::uint32_t>(words.size()) << 16U) | spirvOpEntryPoint;
        return words;
    }

    [[nodiscard]] std::vector<std::byte> vertexModuleWithPointSizeVariable(bool writePointSize)
    {
        constexpr std::uint32_t mainId = 1;
        constexpr std::uint32_t voidTypeId = 2;
        constexpr std::uint32_t functionTypeId = 3;
        constexpr std::uint32_t labelId = 4;
        constexpr std::uint32_t floatTypeId = 5;
        constexpr std::uint32_t vectorTypeId = 6;
        constexpr std::uint32_t vectorOutputTypeId = 7;
        constexpr std::uint32_t floatOutputTypeId = 8;
        constexpr std::uint32_t positionId = 9;
        constexpr std::uint32_t pointSizeId = 10;
        constexpr std::uint32_t oneId = 11;
        constexpr std::uint32_t onesId = 12;
        constexpr std::uint32_t idBound = 13;
        const std::vector<std::uint32_t> header{spirvMagicNumber, spirvVersion1_0, 0U, idBound, 0U};
        const auto pointSizeStore =
            writePointSize ? spirvInstruction(spirvOpStore, {pointSizeId, oneId}) : std::vector<std::uint32_t>{};
        return joinSpirvBytes({
            header,
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
            vertexEntryPoint(mainId, {positionId, pointSizeId}),
            spirvInstruction(spirvOpDecorate, {positionId, spirvDecorationBuiltIn, spirvBuiltInPosition}),
            spirvInstruction(spirvOpDecorate, {pointSizeId, spirvDecorationBuiltIn, spirvBuiltInPointSize}),
            spirvInstruction(spirvOpTypeVoid, {voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {functionTypeId, voidTypeId}),
            spirvInstruction(spirvOpTypeFloat, {floatTypeId, 32U}),
            spirvInstruction(spirvOpTypeVector, {vectorTypeId, floatTypeId, 4U}),
            spirvInstruction(spirvOpTypePointer, {vectorOutputTypeId, spirvStorageClassOutput, vectorTypeId}),
            spirvInstruction(spirvOpTypePointer, {floatOutputTypeId, spirvStorageClassOutput, floatTypeId}),
            spirvInstruction(spirvOpVariable, {vectorOutputTypeId, positionId, spirvStorageClassOutput}),
            spirvInstruction(spirvOpVariable, {floatOutputTypeId, pointSizeId, spirvStorageClassOutput}),
            spirvInstruction(spirvOpConstant, {floatTypeId, oneId, spirvFloatOneBits}),
            spirvInstruction(spirvOpConstantComposite, {vectorTypeId, onesId, oneId, oneId, oneId, oneId}),
            spirvInstruction(spirvOpFunction, {voidTypeId, mainId, spirvFunctionControlNone, functionTypeId}),
            spirvInstruction(spirvOpLabel, {labelId}),
            spirvInstruction(spirvOpStore, {positionId, onesId}),
            pointSizeStore,
            spirvInstruction(spirvOpReturn, {}),
            spirvInstruction(spirvOpFunctionEnd, {}),
        });
    }

    [[nodiscard]] std::vector<std::byte> vertexModuleWritingPerVertexMember(std::uint32_t member)
    {
        constexpr std::uint32_t mainId = 1;
        constexpr std::uint32_t voidTypeId = 2;
        constexpr std::uint32_t functionTypeId = 3;
        constexpr std::uint32_t labelId = 4;
        constexpr std::uint32_t floatTypeId = 5;
        constexpr std::uint32_t vectorTypeId = 6;
        constexpr std::uint32_t intTypeId = 7;
        constexpr std::uint32_t blockTypeId = 8;
        constexpr std::uint32_t blockOutputTypeId = 9;
        constexpr std::uint32_t vectorOutputTypeId = 10;
        constexpr std::uint32_t floatOutputTypeId = 11;
        constexpr std::uint32_t perVertexId = 12;
        constexpr std::uint32_t oneId = 13;
        constexpr std::uint32_t onesId = 14;
        constexpr std::uint32_t memberIndexId = 15;
        constexpr std::uint32_t memberPointerId = 16;
        constexpr std::uint32_t idBound = 17;
        const std::vector<std::uint32_t> header{spirvMagicNumber, spirvVersion1_0, 0U, idBound, 0U};
        const bool writesPosition = member == 0;
        return joinSpirvBytes({
            header,
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
            vertexEntryPoint(mainId, {perVertexId}),
            spirvInstruction(spirvOpMemberDecorate, {blockTypeId, 0U, spirvDecorationBuiltIn, spirvBuiltInPosition}),
            spirvInstruction(spirvOpMemberDecorate, {blockTypeId, 1U, spirvDecorationBuiltIn, spirvBuiltInPointSize}),
            spirvInstruction(spirvOpDecorate, {blockTypeId, spirvDecorationBlock}),
            spirvInstruction(spirvOpTypeVoid, {voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {functionTypeId, voidTypeId}),
            spirvInstruction(spirvOpTypeFloat, {floatTypeId, 32U}),
            spirvInstruction(spirvOpTypeVector, {vectorTypeId, floatTypeId, 4U}),
            spirvInstruction(spirvOpTypeInt, {intTypeId, 32U, 1U}),
            spirvInstruction(spirvOpTypeStruct, {blockTypeId, vectorTypeId, floatTypeId}),
            spirvInstruction(spirvOpTypePointer, {blockOutputTypeId, spirvStorageClassOutput, blockTypeId}),
            spirvInstruction(spirvOpTypePointer, {vectorOutputTypeId, spirvStorageClassOutput, vectorTypeId}),
            spirvInstruction(spirvOpTypePointer, {floatOutputTypeId, spirvStorageClassOutput, floatTypeId}),
            spirvInstruction(spirvOpVariable, {blockOutputTypeId, perVertexId, spirvStorageClassOutput}),
            spirvInstruction(spirvOpConstant, {floatTypeId, oneId, spirvFloatOneBits}),
            spirvInstruction(spirvOpConstantComposite, {vectorTypeId, onesId, oneId, oneId, oneId, oneId}),
            spirvInstruction(spirvOpConstant, {intTypeId, memberIndexId, member}),
            spirvInstruction(spirvOpFunction, {voidTypeId, mainId, spirvFunctionControlNone, functionTypeId}),
            spirvInstruction(spirvOpLabel, {labelId}),
            spirvInstruction(
                spirvOpAccessChain,
                {writesPosition ? vectorOutputTypeId : floatOutputTypeId, memberPointerId, perVertexId, memberIndexId}),
            spirvInstruction(spirvOpStore, {memberPointerId, writesPosition ? onesId : oneId}),
            spirvInstruction(spirvOpReturn, {}),
            spirvInstruction(spirvOpFunctionEnd, {}),
        });
    }

    [[nodiscard]] rachis::Result<rachis::ShaderModuleHandle> createWithoutDevice(
        std::span<const std::byte> bytecode,
        const rachis::DeviceCapabilities& capabilities,
        rachis::ShaderStage stage = rachis::ShaderStage::compute,
        const rachis::ShaderInterfaceDesc& shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
        const rachis::vulkan::VulkanShaderFeatureSupport& features = {})
    {
        VolkDeviceTable table{};
        table.vkCreateShaderModule = &fakeCreateShaderModule;
        table.vkDestroyShaderModule = &fakeDestroyShaderModule;
        rachis::vulkan::VulkanShaderManager shaders;
        shaders.initialize(
            reinterpret_cast<VkDevice>(std::uintptr_t{0x4000}), &table, capabilities, nullptr, false, features);
        auto module = shaders.createShaderModule({
            .stage = stage,
            .bytecode = bytecode,
            .entryPoint = "main",
            .shaderInterface = shaderInterface,
        });
        if (module)
        {
            CHECK(shaders.destroyShaderModule(*module).has_value());
        }
        return module;
    }

    [[nodiscard]] rachis::Result<rachis::ShaderModuleHandle> createWithoutDevice(
        std::span<const std::byte> bytecode,
        const rachis::DeviceCapabilities& capabilities,
        const rachis::vulkan::VulkanShaderFeatureSupport& features,
        rachis::ShaderStage stage = rachis::ShaderStage::compute,
        std::span<const rachis::ShaderBindingRequirement> bindings = {})
    {
        std::optional<rachis::ShaderWorkgroupSize> workgroupSize;
        if (stage == rachis::ShaderStage::compute)
        {
            workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1};
        }
        return createWithoutDevice(
            bytecode, capabilities, stage, {.bindings = bindings, .workgroupSize = workgroupSize}, features);
    }

    [[nodiscard]] rachis::Result<rachis::ShaderLibraryHandle> createLibraryWithoutDevice(
        std::span<const std::byte> bytecode, rachis::ShaderStage stage)
    {
        VolkDeviceTable table{};
        table.vkCreateShaderModule = &fakeCreateShaderModule;
        table.vkDestroyShaderModule = &fakeDestroyShaderModule;
        rachis::vulkan::VulkanShaderManager shaders;
        shaders.initialize(reinterpret_cast<VkDevice>(std::uintptr_t{0x4000}), &table, rachis::DeviceCapabilities{});
        const std::array entries{rachis::ShaderLibraryEntryDesc{
            .stage = stage,
            .bytecode = bytecode,
            .entryPoint = "main",
        }};
        auto library = shaders.createShaderLibrary({.entries = entries});
        if (library)
        {
            CHECK(shaders.destroyShaderLibrary(*library).has_value());
        }
        return library;
    }

    [[nodiscard]] rachis::Result<rachis::ShaderModuleHandle> createVertexWithoutDevice(
        std::span<const VertexInputVariable> inputs, std::span<const std::uint32_t> locations)
    {
        const auto bytecode = vertexModuleWithInputs(inputs);
        return createWithoutDevice(bytecode,
                                   rachis::DeviceCapabilities{},
                                   rachis::ShaderStage::vertex,
                                   {.vertexInputLocations = locations},
                                   {.shaderFloat64 = true});
    }

    enum class StorageDeclaration : std::uint8_t
    {
        writableBuffer,
        nonWritableBuffer,
        nonWritableBufferMembers,
        writableImage,
    };

    [[nodiscard]] std::vector<std::byte> moduleUsingStorage(std::uint32_t executionModel,
                                                            StorageDeclaration declaration)
    {
        constexpr std::uint32_t mainId = 1;
        constexpr std::uint32_t voidTypeId = 2;
        constexpr std::uint32_t functionTypeId = 3;
        constexpr std::uint32_t labelId = 4;
        constexpr std::uint32_t uintTypeId = 5;
        constexpr std::uint32_t runtimeArrayTypeId = 6;
        constexpr std::uint32_t blockTypeId = 7;
        constexpr std::uint32_t blockPointerTypeId = 8;
        constexpr std::uint32_t bufferId = 9;
        constexpr std::uint32_t zeroId = 10;
        constexpr std::uint32_t elementPointerTypeId = 11;
        constexpr std::uint32_t elementId = 12;
        constexpr std::uint32_t floatTypeId = 13;
        constexpr std::uint32_t imageTypeId = 14;
        constexpr std::uint32_t imagePointerTypeId = 15;
        constexpr std::uint32_t imageId = 16;
        constexpr std::uint32_t loadedImageId = 17;
        constexpr std::uint32_t idBound = 18;
        const bool image = declaration == StorageDeclaration::writableImage;
        const auto resourceId = image ? imageId : bufferId;
        const std::vector<std::uint32_t> header{spirvMagicNumber, spirvVersion1_3, 0U, idBound, 0U};
        std::vector<std::uint32_t> executionMode;
        if (executionModel == spirvExecutionModelGlCompute)
        {
            executionMode = spirvInstruction(spirvOpExecutionMode, {mainId, spirvExecutionModeLocalSize, 1U, 1U, 1U});
        }
        std::vector<std::uint32_t> readOnly;
        if (declaration == StorageDeclaration::nonWritableBuffer)
        {
            readOnly = spirvInstruction(spirvOpDecorate, {bufferId, spirvDecorationNonWritable});
        }
        else if (declaration == StorageDeclaration::nonWritableBufferMembers)
        {
            readOnly = spirvInstruction(spirvOpMemberDecorate, {blockTypeId, 0U, spirvDecorationNonWritable});
        }
        const auto variable =
            image ? spirvInstruction(spirvOpVariable, {imagePointerTypeId, imageId, spirvStorageClassUniformConstant})
                  : spirvInstruction(spirvOpVariable, {blockPointerTypeId, bufferId, spirvStorageClassStorageBuffer});
        const auto access =
            image ? spirvInstruction(spirvOpLoad, {imageTypeId, loadedImageId, imageId})
                  : spirvInstruction(spirvOpAccessChain, {elementPointerTypeId, elementId, bufferId, zeroId, zeroId});
        return joinSpirvBytes({
            header,
            spirvInstruction(spirvOpCapability, {spirvCapabilityShader}),
            spirvInstruction(spirvOpMemoryModel, {spirvAddressingLogical, spirvMemoryModelGlsl450}),
            spirvInstructionWithString(spirvOpEntryPoint, {executionModel, mainId}, "main"),
            executionMode,
            spirvInstruction(spirvOpDecorate, {runtimeArrayTypeId, spirvDecorationArrayStride, 4U}),
            spirvInstruction(spirvOpMemberDecorate, {blockTypeId, 0U, spirvDecorationOffset, 0U}),
            spirvInstruction(spirvOpDecorate, {blockTypeId, spirvDecorationBlock}),
            readOnly,
            spirvInstruction(spirvOpDecorate, {resourceId, spirvDecorationDescriptorSet, 0U}),
            spirvInstruction(spirvOpDecorate, {resourceId, spirvDecorationBinding, 0U}),
            spirvInstruction(spirvOpTypeVoid, {voidTypeId}),
            spirvInstruction(spirvOpTypeFunction, {functionTypeId, voidTypeId}),
            spirvInstruction(spirvOpTypeInt, {uintTypeId, 32U, 0U}),
            spirvInstruction(spirvOpTypeRuntimeArray, {runtimeArrayTypeId, uintTypeId}),
            spirvInstruction(spirvOpTypeStruct, {blockTypeId, runtimeArrayTypeId}),
            spirvInstruction(spirvOpTypePointer, {blockPointerTypeId, spirvStorageClassStorageBuffer, blockTypeId}),
            spirvInstruction(spirvOpTypePointer, {elementPointerTypeId, spirvStorageClassStorageBuffer, uintTypeId}),
            spirvInstruction(spirvOpTypeFloat, {floatTypeId, 32U}),
            spirvInstruction(spirvOpTypeImage,
                             {imageTypeId, floatTypeId, spirvDim2D, 0U, 0U, 0U, 2U, spirvImageFormatR32f}),
            spirvInstruction(spirvOpTypePointer, {imagePointerTypeId, spirvStorageClassUniformConstant, imageTypeId}),
            spirvInstruction(spirvOpConstant, {uintTypeId, zeroId, 0U}),
            variable,
            spirvInstruction(spirvOpFunction, {voidTypeId, mainId, spirvFunctionControlNone, functionTypeId}),
            spirvInstruction(spirvOpLabel, {labelId}),
            access,
            spirvInstruction(spirvOpReturn, {}),
            spirvInstruction(spirvOpFunctionEnd, {}),
        });
    }

} // namespace

TEST_CASE("Vulkan shader modules gate SPIR-V MultiViewport, not ImageCubeArray, on multi-viewport support",
          "[size:small][rhi][vulkan][shader][module]")
{
    rachis::DeviceCapabilities capabilities{};
    capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
    capabilities.limits.maxComputeWorkgroupInvocations = 1024;
    const auto multiViewport = computeModuleDeclaringCapability(spirvCapabilityMultiViewport);
    const auto imageCubeArray = computeModuleDeclaringCapability(spirvCapabilityImageCubeArray);

    capabilities.multiViewport = false;
    auto rejected = createWithoutDevice(multiViewport, capabilities);
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
    CHECK(rejected.error().message == "Multiple viewports are not available on this device.");

    auto cubeArray = createWithoutDevice(imageCubeArray, capabilities);
    if (!cubeArray)
    {
        UNSCOPED_INFO(cubeArray.error().message);
    }
    CHECK(cubeArray.has_value());

    capabilities.multiViewport = true;
    auto accepted = createWithoutDevice(multiViewport, capabilities);
    if (!accepted)
    {
        UNSCOPED_INFO(accepted.error().message);
    }
    CHECK(accepted.has_value());
}

TEST_CASE("Vulkan shader modules record whether the vertex entry point writes PointSize",
          "[size:small][rhi][vulkan][shader][module]")
{
    VolkDeviceTable table{};
    table.vkCreateShaderModule = &fakeCreateShaderModule;
    table.vkDestroyShaderModule = &fakeDestroyShaderModule;
    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(reinterpret_cast<VkDevice>(std::uintptr_t{0x4000}),
                       &table,
                       rachis::DeviceCapabilities{},
                       nullptr,
                       false,
                       {.shaderDrawParameters = true});
    const auto writesPointSize = [&](std::span<const std::byte> bytecode, std::string_view entryPoint) {
        auto module = shaders.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = bytecode,
            .entryPoint = entryPoint,
            .shaderInterface = {},
        });
        if (!module)
        {
            UNSCOPED_INFO(module.error().message);
        }
        REQUIRE(module);
        auto native = shaders.nativeShaderModule(*module);
        REQUIRE(native);
        const bool writes = native->writesPointSize;
        CHECK(shaders.destroyShaderModule(*module).has_value());
        return writes;
    };

    CHECK_FALSE(writesPointSize(vertexModuleWithPointSizeVariable(false), "main"));
    CHECK(writesPointSize(vertexModuleWithPointSizeVariable(true), "main"));
    CHECK_FALSE(writesPointSize(vertexModuleWritingPerVertexMember(0), "main"));
    CHECK(writesPointSize(vertexModuleWritingPerVertexMember(1), "main"));

    const auto unsized = loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / "PointList.vs.spv");
    const auto sized = loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / "PointList.sized.vs.spv");
    REQUIRE_FALSE(unsized.empty());
    REQUIRE_FALSE(sized.empty());
    CHECK_FALSE(writesPointSize(unsized, "vertexMain"));
    CHECK(writesPointSize(sized, "sizedVertexMain"));
}

TEST_CASE("Vulkan shader modules gate SPIR-V RayTracingKHR, RayQueryKHR, and MeshShadingEXT on enabled features",
          "[size:small][rhi][vulkan][shader][module]")
{
    struct CapabilityGate final
    {
        std::uint32_t capability = 0;
        std::string_view extension;
        bool rachis::DeviceCapabilities::* feature = nullptr;
        std::string_view message;
    };
    const std::array gates{
        CapabilityGate{
            .capability = spirvCapabilityRayTracingKHR,
            .extension = "SPV_KHR_ray_tracing",
            .feature = &rachis::DeviceCapabilities::rayTracing,
            .message = "Ray-tracing pipelines are not available on this device.",
        },
        CapabilityGate{
            .capability = spirvCapabilityRayQueryKHR,
            .extension = "SPV_KHR_ray_query",
            .feature = &rachis::DeviceCapabilities::rayQuery,
            .message = "Inline ray queries are not available on this device.",
        },
        CapabilityGate{
            .capability = spirvCapabilityMeshShadingEXT,
            .extension = "SPV_EXT_mesh_shader",
            .feature = &rachis::DeviceCapabilities::meshShaders,
            .message = "Mesh shaders are not available on this device.",
        },
    };
    for (const auto& gate : gates)
    {
        INFO(gate.extension);
        rachis::DeviceCapabilities capabilities{};
        capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
        capabilities.limits.maxComputeWorkgroupInvocations = 1024;
        capabilities.rayTracing = true;
        capabilities.rayQuery = true;
        capabilities.meshShaders = true;
        const auto bytecode = computeModuleDeclaringCapability(gate.capability, gate.extension);

        capabilities.*gate.feature = false;
        auto rejected = createWithoutDevice(bytecode, capabilities);
        CHECK_FALSE(rejected);
        if (!rejected)
        {
            CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
            CHECK(rejected.error().message == gate.message);
        }

        capabilities.*gate.feature = true;
        auto accepted = createWithoutDevice(bytecode, capabilities);
        if (!accepted)
        {
            UNSCOPED_INFO(accepted.error().message);
        }
        CHECK(accepted.has_value());
    }
}

TEST_CASE("Vulkan shader modules classify a group-shared 64-bit atomic after OpLine and OpCopyMemory name its pointer",
          "[size:small][rhi][vulkan][shader][module]")
{
    using Module = GroupSharedInt64AtomicModule;
    const Module module;
    const std::vector<std::uint32_t> noInstruction;
    const auto line = spirvInstruction(spirvOpLine, {Module::fileId, Module::accumulatorId, 1U});
    const auto copy = spirvInstruction(spirvOpCopyMemory, {Module::localId, Module::accumulatorId});
    const std::array<std::pair<std::string_view, std::span<const std::uint32_t>>, 3> variants{{
        {"no instruction", noInstruction},
        {"OpLine", line},
        {"OpCopyMemory", copy},
    }};
    for (const auto& [name, beforeAtomic] : variants)
    {
        INFO(name);
        const auto bytecode = joinSpirvBytes(
            {module.header, module.declarations, module.functionStart, beforeAtomic, module.atomicAndReturn});
        rachis::DeviceCapabilities capabilities{};
        capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
        capabilities.limits.maxComputeWorkgroupInvocations = 1024;
        capabilities.shaderInt64 = true;

        capabilities.shaderInt64AtomicGroupShared = true;
        capabilities.shaderInt64AtomicTypedResource = false;
        auto accepted = createWithoutDevice(bytecode, capabilities);
        if (!accepted)
        {
            UNSCOPED_INFO(accepted.error().message);
        }
        CHECK(accepted.has_value());

        capabilities.shaderInt64AtomicGroupShared = false;
        capabilities.shaderInt64AtomicTypedResource = true;
        auto rejected = createWithoutDevice(bytecode, capabilities);
        CHECK_FALSE(rejected);
        if (!rejected)
        {
            CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
            CHECK(rejected.error().message == "Group-shared 64-bit integer atomics are not available on this device.");
        }
    }
}

TEST_CASE("Vulkan shader modules gate storage-buffer 64-bit atomics on shaderInt64AtomicStorageBuffer alone",
          "[size:small][rhi][vulkan][shader][module]")
{
    const auto bytecode = rachis_shaders::ShaderInt64AtomicStorageBuffer(rachis::Backend::vulkan);
    REQUIRE_FALSE(bytecode.empty());
    const std::array bindings{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}};
    const rachis::ShaderInterfaceDesc shaderInterface{
        .bindings = bindings,
        .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
    };
    rachis::DeviceCapabilities capabilities{};
    capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
    capabilities.limits.maxComputeWorkgroupInvocations = 1024;
    capabilities.shaderInt64 = true;

    capabilities.shaderInt64AtomicGroupShared = true;
    capabilities.shaderInt64AtomicTypedResource = true;
    capabilities.shaderInt64AtomicStorageBuffer = false;
    auto rejected = createWithoutDevice(bytecode, capabilities, rachis::ShaderStage::compute, shaderInterface);
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
    CHECK(rejected.error().message == "Storage-buffer 64-bit integer atomics are not available on this device.");

    capabilities.shaderInt64AtomicGroupShared = false;
    capabilities.shaderInt64AtomicTypedResource = false;
    capabilities.shaderInt64AtomicStorageBuffer = true;
    auto accepted = createWithoutDevice(bytecode, capabilities, rachis::ShaderStage::compute, shaderInterface);
    if (!accepted)
    {
        UNSCOPED_INFO(accepted.error().message);
    }
    CHECK(accepted.has_value());
}

TEST_CASE("Vulkan shader modules need no 64-bit atomic capability for a 32-bit atomic beside Int64Atomics",
          "[size:small][rhi][vulkan][shader][module]")
{
    rachis::DeviceCapabilities capabilities{};
    capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
    capabilities.limits.maxComputeWorkgroupInvocations = 1024;
    capabilities.shaderInt64 = true;
    auto accepted = createWithoutDevice(int64AtomicsModuleWith32BitGroupSharedAtomic(), capabilities);
    if (!accepted)
    {
        UNSCOPED_INFO(accepted.error().message);
    }
    CHECK(accepted.has_value());
}

TEST_CASE("Vulkan shader modules and libraries read SPIR-V bytecode at any byte alignment",
          "[size:small][rhi][vulkan][shader][module]")
{
    rachis::DeviceCapabilities capabilities{};
    capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
    capabilities.limits.maxComputeWorkgroupInvocations = 1024;
    const auto spirv = computeModuleDeclaringCapability(spirvCapabilityImageCubeArray);
    std::vector<std::byte> storage(spirv.size() + 1U);
    std::ranges::copy(spirv, storage.begin() + 1);
    const std::span<const std::byte> unaligned(storage.data() + 1, spirv.size());
    REQUIRE(reinterpret_cast<std::uintptr_t>(unaligned.data()) % alignof(std::uint32_t) != 0);

    VolkDeviceTable table{};
    table.vkCreateShaderModule = &capturingCreateShaderModule;
    table.vkDestroyShaderModule = &fakeDestroyShaderModule;
    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(reinterpret_cast<VkDevice>(std::uintptr_t{0x4000}), &table, capabilities);
    const rachis::ShaderInterfaceDesc shaderInterface{.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}};
    auto& captured = capturedShaderModuleCode();

    captured = {};
    auto direct = shaders.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = unaligned,
        .entryPoint = "main",
        .shaderInterface = shaderInterface,
    });
    if (!direct)
    {
        UNSCOPED_INFO(direct.error().message);
    }
    REQUIRE(direct.has_value());
    CHECK(captured.aligned);
    CHECK(captured.bytes == spirv);

    const std::array entries{rachis::ShaderLibraryEntryDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = unaligned,
        .entryPoint = "main",
    }};
    auto library = shaders.createShaderLibrary({.entries = entries});
    if (!library)
    {
        UNSCOPED_INFO(library.error().message);
    }
    REQUIRE(library.has_value());
    captured = {};
    auto fromLibrary = shaders.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .entryPoint = "main",
        .library = *library,
        .shaderInterface = shaderInterface,
    });
    if (!fromLibrary)
    {
        UNSCOPED_INFO(fromLibrary.error().message);
    }
    REQUIRE(fromLibrary.has_value());
    CHECK(captured.aligned);
    CHECK(captured.bytes == spirv);

    CHECK(shaders.destroyShaderModule(*fromLibrary).has_value());
    CHECK(shaders.destroyShaderLibrary(*library).has_value());
    CHECK(shaders.destroyShaderModule(*direct).has_value());
}

TEST_CASE("Vulkan shader modules reflect every location that matrix, array, and 64-bit vertex inputs consume",
          "[size:small][rhi][vulkan][shader][module]")
{
    using Module = VertexInputModule;
    struct LocationCase final
    {
        std::string_view name;
        std::vector<VertexInputVariable> inputs;
        std::vector<std::uint32_t> locations;
        std::vector<std::uint32_t> baseLocations;
    };
    const std::array cases{
        LocationCase{
            .name = "float2 and float4x4",
            .inputs = {{.typeId = Module::float2TypeId, .location = 0},
                       {.typeId = Module::float4x4TypeId, .location = 1}},
            .locations = {0, 1, 2, 3, 4},
            .baseLocations = {0, 1},
        },
        LocationCase{
            .name = "float2 and float4[2]",
            .inputs = {{.typeId = Module::float2TypeId, .location = 0},
                       {.typeId = Module::float4Array2TypeId, .location = 1}},
            .locations = {0, 1, 2},
            .baseLocations = {0, 1},
        },
        LocationCase{
            .name = "double4, double3, and double2",
            .inputs = {{.typeId = Module::double4TypeId, .location = 0},
                       {.typeId = Module::double3TypeId, .location = 2},
                       {.typeId = Module::double2TypeId, .location = 4}},
            .locations = {0, 1, 2, 3, 4},
            .baseLocations = {0, 2, 4},
        },
        LocationCase{
            .name = "matrix of two double4 columns",
            .inputs = {{.typeId = Module::double4x2TypeId, .location = 1}},
            .locations = {1, 2, 3, 4},
            .baseLocations = {1},
        },
        LocationCase{
            .name = "float4x4 at the last four locations",
            .inputs = {{.typeId = Module::float4x4TypeId, .location = 12}},
            .locations = {12, 13, 14, 15},
            .baseLocations = {12},
        },
        LocationCase{
            .name = "two float2 inputs packed into one location",
            .inputs = {{.typeId = Module::float2TypeId, .location = 3},
                       {.typeId = Module::float2TypeId, .location = 3, .component = 2}},
            .locations = {3},
            .baseLocations = {3},
        },
        LocationCase{
            .name = "BuiltIn VertexIndex beside a float4",
            .inputs = {{.typeId = Module::uintTypeId, .builtIn = spirvBuiltInVertexIndex},
                       {.typeId = Module::float4TypeId, .location = 0}},
            .locations = {0},
            .baseLocations = {0},
        },
    };
    for (const auto& testCase : cases)
    {
        INFO(testCase.name);
        auto accepted = createVertexWithoutDevice(testCase.inputs, testCase.locations);
        if (!accepted)
        {
            UNSCOPED_INFO(accepted.error().message);
        }
        CHECK(accepted.has_value());
        if (testCase.baseLocations == testCase.locations)
        {
            continue;
        }
        auto rejected = createVertexWithoutDevice(testCase.inputs, testCase.baseLocations);
        CHECK_FALSE(rejected);
        if (!rejected)
        {
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
            CHECK(rejected.error().message ==
                  "Declared vertex input locations do not match the reflected SPIR-V locations.");
        }
    }
}

TEST_CASE("Vulkan shader modules and libraries reject vertex inputs past the portable limit or of struct type",
          "[size:small][rhi][vulkan][shader][module]")
{
    using Module = VertexInputModule;
    struct RejectedCase final
    {
        std::string_view name;
        std::vector<VertexInputVariable> inputs;
        std::vector<std::uint32_t> locations;
        std::string_view message;
    };
    constexpr std::string_view beyondLimit =
        "A SPIR-V vertex input uses a location beyond the portable vertex attribute limit.";
    const std::array cases{
        RejectedCase{
            .name = "float4 at location 16",
            .inputs = {{.typeId = Module::float2TypeId, .location = 0},
                       {.typeId = Module::float4TypeId, .location = 16}},
            .locations = {0},
            .message = beyondLimit,
        },
        RejectedCase{
            .name = "float4x4 at location 13",
            .inputs = {{.typeId = Module::float4x4TypeId, .location = 13}},
            .locations = {13, 14, 15},
            .message = beyondLimit,
        },
        RejectedCase{
            .name = "struct at location 0",
            .inputs = {{.typeId = Module::structTypeId, .location = 0}},
            .locations = {0},
            .message = "A SPIR-V vertex input must be a scalar, vector, matrix, or array of them.",
        },
    };
    for (const auto& testCase : cases)
    {
        INFO(testCase.name);
        auto module = createVertexWithoutDevice(testCase.inputs, testCase.locations);
        CHECK_FALSE(module);
        if (!module)
        {
            CHECK(module.error().code == rachis::ErrorCode::invalidArgument);
            CHECK(module.error().message == testCase.message);
        }
        const auto bytecode = vertexModuleWithInputs(testCase.inputs);
        auto library = createLibraryWithoutDevice(bytecode, rachis::ShaderStage::vertex);
        CHECK_FALSE(library);
        if (!library)
        {
            CHECK(library.error().code == rachis::ErrorCode::invalidArgument);
            CHECK(library.error().message == testCase.message);
        }
    }
}

TEST_CASE("Vulkan shader modules reflect a storage block as read-only when its variable or every member is NonWritable",
          "[size:small][rhi][vulkan][shader][module]")
{
    constexpr auto readOnly = rachis::ShaderResourceType::readOnlyStorageBuffer;
    constexpr auto writable = rachis::ShaderResourceType::storageBuffer;
    const std::array variants{
        StorageBlockVariant{.name = "no NonWritable member", .nonWritableMembers = {}, .reflected = writable},
        StorageBlockVariant{
            .name = "only the second of two members NonWritable", .nonWritableMembers = {1}, .reflected = writable},
        StorageBlockVariant{.name = "both members NonWritable", .nonWritableMembers = {0, 1}, .reflected = readOnly},
        StorageBlockVariant{.name = "NonWritable variable",
                            .nonWritableMembers = {},
                            .nonWritableVariable = true,
                            .reflected = readOnly},
        StorageBlockVariant{.name = "NonWritable BufferBlock variable",
                            .nonWritableMembers = {},
                            .nonWritableVariable = true,
                            .bufferBlock = true,
                            .reflected = readOnly},
        StorageBlockVariant{.name = "BufferBlock with both members NonWritable",
                            .nonWritableMembers = {0, 1},
                            .bufferBlock = true,
                            .reflected = readOnly},
        StorageBlockVariant{.name = "array of four blocks with both members NonWritable",
                            .nonWritableMembers = {0, 1},
                            .array = StorageBlockArray::fixed,
                            .reflected = readOnly},
        StorageBlockVariant{.name = "array of four blocks with only the first member NonWritable",
                            .nonWritableMembers = {0},
                            .array = StorageBlockArray::fixed,
                            .reflected = writable},
        StorageBlockVariant{.name = "runtime array of blocks with both members NonWritable",
                            .nonWritableMembers = {1, 0},
                            .array = StorageBlockArray::runtime,
                            .reflected = readOnly},
    };
    rachis::DeviceCapabilities capabilities{};
    capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
    capabilities.limits.maxComputeWorkgroupInvocations = 1024;
    for (const auto& variant : variants)
    {
        INFO(variant.name);
        const auto bytecode = storageBlockModule(variant);
        std::uint32_t count = 1;
        if (variant.array == StorageBlockArray::fixed)
        {
            count = 4;
        }
        else if (variant.array == StorageBlockArray::runtime)
        {
            count = rachis::unboundedBindingCount;
        }
        const std::array matching{
            rachis::ShaderBindingRequirement{.set = 0, .binding = 0, .type = variant.reflected, .count = count},
        };
        auto accepted =
            createWithoutDevice(bytecode,
                                capabilities,
                                rachis::ShaderStage::compute,
                                {.bindings = matching, .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}});
        if (!accepted)
        {
            UNSCOPED_INFO(accepted.error().message);
        }
        CHECK(accepted.has_value());

        const std::array mismatched{
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 0,
                .type = variant.reflected == readOnly ? writable : readOnly,
                .count = count,
            },
        };
        auto rejected =
            createWithoutDevice(bytecode,
                                capabilities,
                                rachis::ShaderStage::compute,
                                {.bindings = mismatched, .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}});
        CHECK_FALSE(rejected);
        if (!rejected)
        {
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
            CHECK(rejected.error().message ==
                  "A declared shader binding does not match the reflected SPIR-V set, binding, type, or count.");
        }
    }
}

TEST_CASE("Vulkan shader modules gate SPIR-V ImageGatherExtended, SampleRateShading, and MinLod on their device "
          "features",
          "[size:small][rhi][vulkan][shader][module]")
{
    rachis::DeviceCapabilities capabilities{};
    capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
    capabilities.limits.maxComputeWorkgroupInvocations = 1024;
    constexpr rachis::vulkan::VulkanShaderFeatureSupport allFeatures{
        .sampleRateShading = true,
        .vertexPipelineStoresAndAtomics = true,
        .shaderImageGatherExtended = true,
        .shaderResourceMinLod = true,
    };

    struct GatedCapability final
    {
        std::uint32_t capability = 0;
        bool rachis::vulkan::VulkanShaderFeatureSupport::* feature = nullptr;
        std::string_view message;
    };
    const std::array gated{
        GatedCapability{
            .capability = spirvCapabilityImageGatherExtended,
            .feature = &rachis::vulkan::VulkanShaderFeatureSupport::shaderImageGatherExtended,
            .message = "Extended image gather offsets are not available on this device.",
        },
        GatedCapability{
            .capability = spirvCapabilitySampleRateShading,
            .feature = &rachis::vulkan::VulkanShaderFeatureSupport::sampleRateShading,
            .message = "Sample-rate shading is not available on this device.",
        },
        GatedCapability{
            .capability = spirvCapabilityMinLod,
            .feature = &rachis::vulkan::VulkanShaderFeatureSupport::shaderResourceMinLod,
            .message = "Shader minimum-LOD clamps are not available on this device.",
        },
    };
    for (const auto& entry : gated)
    {
        INFO("SPIR-V capability " << entry.capability);
        const auto bytecode = computeModuleDeclaringCapability(entry.capability);

        auto withoutAny = createWithoutDevice(bytecode, capabilities);
        REQUIRE_FALSE(withoutAny);
        CHECK(withoutAny.error().code == rachis::ErrorCode::unsupported);
        CHECK(withoutAny.error().message == entry.message);

        auto othersOnly = allFeatures;
        othersOnly.*entry.feature = false;
        auto withoutFeature = createWithoutDevice(bytecode, capabilities, othersOnly);
        REQUIRE_FALSE(withoutFeature);
        CHECK(withoutFeature.error().code == rachis::ErrorCode::unsupported);
        CHECK(withoutFeature.error().message == entry.message);

        rachis::vulkan::VulkanShaderFeatureSupport onlyFeature{};
        onlyFeature.*entry.feature = true;
        auto accepted = createWithoutDevice(bytecode, capabilities, onlyFeature);
        if (!accepted)
        {
            UNSCOPED_INFO(accepted.error().message);
        }
        CHECK(accepted.has_value());
    }
}

TEST_CASE("Vulkan shader modules gate SPIR-V Float64 and the 16-bit storage capabilities on their device features",
          "[size:small][rhi][vulkan][shader][module]")
{
    rachis::DeviceCapabilities capabilities{};
    capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
    capabilities.limits.maxComputeWorkgroupInvocations = 1024;
    capabilities.shaderFloat16 = true;
    capabilities.shaderInt16 = true;
    capabilities.shaderInt64 = true;
    constexpr rachis::vulkan::VulkanShaderFeatureSupport allFeatures{
        .sampleRateShading = true,
        .vertexPipelineStoresAndAtomics = true,
        .shaderImageGatherExtended = true,
        .shaderResourceMinLod = true,
        .shaderFloat64 = true,
        .storageBuffer16BitAccess = true,
        .uniformAndStorageBuffer16BitAccess = true,
        .storagePushConstant16 = true,
        .storageInputOutput16 = true,
    };

    struct GatedCapability final
    {
        std::uint32_t capability = 0;
        bool rachis::vulkan::VulkanShaderFeatureSupport::* feature = nullptr;
        std::string_view message;
    };
    const std::array gated{
        GatedCapability{
            .capability = spirvCapabilityFloat64,
            .feature = &rachis::vulkan::VulkanShaderFeatureSupport::shaderFloat64,
            .message = "64-bit float shader arithmetic is not available on this device.",
        },
        GatedCapability{
            .capability = spirvCapabilityStorageBuffer16BitAccess,
            .feature = &rachis::vulkan::VulkanShaderFeatureSupport::storageBuffer16BitAccess,
            .message = "16-bit storage-buffer access is not available on this device.",
        },
        GatedCapability{
            .capability = spirvCapabilityUniformAndStorageBuffer16BitAccess,
            .feature = &rachis::vulkan::VulkanShaderFeatureSupport::uniformAndStorageBuffer16BitAccess,
            .message = "16-bit uniform-buffer access is not available on this device.",
        },
        GatedCapability{
            .capability = spirvCapabilityStoragePushConstant16,
            .feature = &rachis::vulkan::VulkanShaderFeatureSupport::storagePushConstant16,
            .message = "16-bit push-constant access is not available on this device.",
        },
        GatedCapability{
            .capability = spirvCapabilityStorageInputOutput16,
            .feature = &rachis::vulkan::VulkanShaderFeatureSupport::storageInputOutput16,
            .message = "16-bit shader stage inputs and outputs are not available on this device.",
        },
    };
    for (const auto& entry : gated)
    {
        INFO("SPIR-V capability " << entry.capability);
        const auto bytecode = computeModuleDeclaringCapability(entry.capability);

        auto withoutAny = createWithoutDevice(bytecode, capabilities);
        REQUIRE_FALSE(withoutAny);
        CHECK(withoutAny.error().code == rachis::ErrorCode::unsupported);
        CHECK(withoutAny.error().message == entry.message);

        auto othersOnly = allFeatures;
        othersOnly.*entry.feature = false;
        auto withoutFeature = createWithoutDevice(bytecode, capabilities, othersOnly);
        REQUIRE_FALSE(withoutFeature);
        CHECK(withoutFeature.error().code == rachis::ErrorCode::unsupported);
        CHECK(withoutFeature.error().message == entry.message);

        rachis::vulkan::VulkanShaderFeatureSupport onlyFeature{};
        onlyFeature.*entry.feature = true;
        auto accepted = createWithoutDevice(bytecode, capabilities, onlyFeature);
        if (!accepted)
        {
            UNSCOPED_INFO(accepted.error().message);
        }
        CHECK(accepted.has_value());
    }

    for (const auto capability : {spirvCapabilityFloat16, spirvCapabilityInt16})
    {
        INFO("arithmetic-only SPIR-V capability " << capability);
        auto accepted = createWithoutDevice(computeModuleDeclaringCapability(capability), capabilities);
        if (!accepted)
        {
            UNSCOPED_INFO(accepted.error().message);
        }
        CHECK(accepted.has_value());
    }
}

TEST_CASE("Vulkan shader modules reject writable storage in vertex, tessellation, and geometry stages without "
          "vertexPipelineStoresAndAtomics",
          "[size:small][rhi][vulkan][shader][module]")
{
    rachis::DeviceCapabilities capabilities{};
    capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
    capabilities.limits.maxComputeWorkgroupInvocations = 1024;
    capabilities.geometryShaders = true;
    capabilities.tessellationShaders = true;
    capabilities.tessellation.maxPatchControlPoints = 32;
    constexpr rachis::vulkan::VulkanShaderFeatureSupport withoutStores{
        .sampleRateShading = true,
        .vertexPipelineStoresAndAtomics = false,
        .shaderImageGatherExtended = true,
        .shaderResourceMinLod = true,
    };
    constexpr rachis::vulkan::VulkanShaderFeatureSupport withStores{
        .sampleRateShading = true,
        .vertexPipelineStoresAndAtomics = true,
        .shaderImageGatherExtended = true,
        .shaderResourceMinLod = true,
    };
    const std::array storageBuffer{rachis::ShaderBindingRequirement{
        .set = 0,
        .binding = 0,
        .type = rachis::ShaderResourceType::storageBuffer,
        .count = 1,
    }};
    const std::array readOnlyStorageBuffer{rachis::ShaderBindingRequirement{
        .set = 0,
        .binding = 0,
        .type = rachis::ShaderResourceType::readOnlyStorageBuffer,
        .count = 1,
    }};
    const std::array storageTexture{rachis::ShaderBindingRequirement{
        .set = 0,
        .binding = 0,
        .type = rachis::ShaderResourceType::storageTexture,
        .count = 1,
    }};

    struct StageModel final
    {
        rachis::ShaderStage stage = rachis::ShaderStage::none;
        std::uint32_t executionModel = 0;
    };
    const std::array preRasterization{
        StageModel{rachis::ShaderStage::vertex, spirvExecutionModelVertex},
        StageModel{rachis::ShaderStage::hull, spirvExecutionModelTessellationControl},
        StageModel{rachis::ShaderStage::domain, spirvExecutionModelTessellationEvaluation},
        StageModel{rachis::ShaderStage::geometry, spirvExecutionModelGeometry},
    };
    for (const auto& [stage, model] : preRasterization)
    {
        INFO("execution model " << model);
        const auto writableBuffer = moduleUsingStorage(model, StorageDeclaration::writableBuffer);
        const auto writableImage = moduleUsingStorage(model, StorageDeclaration::writableImage);
        const auto nonWritableBuffer = moduleUsingStorage(model, StorageDeclaration::nonWritableBuffer);
        const auto nonWritableMembers = moduleUsingStorage(model, StorageDeclaration::nonWritableBufferMembers);

        auto rejectedBuffer = createWithoutDevice(writableBuffer, capabilities, withoutStores, stage, storageBuffer);
        REQUIRE_FALSE(rejectedBuffer);
        CHECK(rejectedBuffer.error().code == rachis::ErrorCode::unsupported);
        CHECK(rejectedBuffer.error().message ==
              "Writable storage in vertex, tessellation, and geometry shaders is not available on this device.");

        auto rejectedImage = createWithoutDevice(writableImage, capabilities, withoutStores, stage, storageTexture);
        REQUIRE_FALSE(rejectedImage);
        CHECK(rejectedImage.error().code == rachis::ErrorCode::unsupported);

        auto storesEnabled = createWithoutDevice(writableBuffer, capabilities, withStores, stage, storageBuffer);
        if (!storesEnabled)
        {
            UNSCOPED_INFO(storesEnabled.error().message);
        }
        CHECK(storesEnabled.has_value());

        auto readOnlyVariable =
            createWithoutDevice(nonWritableBuffer, capabilities, withoutStores, stage, readOnlyStorageBuffer);
        if (!readOnlyVariable)
        {
            UNSCOPED_INFO(readOnlyVariable.error().message);
        }
        CHECK(readOnlyVariable.has_value());

        auto readOnlyMembers =
            createWithoutDevice(nonWritableMembers, capabilities, withoutStores, stage, readOnlyStorageBuffer);
        if (!readOnlyMembers)
        {
            UNSCOPED_INFO(readOnlyMembers.error().message);
        }
        CHECK(readOnlyMembers.has_value());
    }

    const std::array otherStages{
        StageModel{rachis::ShaderStage::pixel, spirvExecutionModelFragment},
        StageModel{rachis::ShaderStage::compute, spirvExecutionModelGlCompute},
    };
    for (const auto& [stage, model] : otherStages)
    {
        INFO("execution model " << model);
        auto accepted = createWithoutDevice(moduleUsingStorage(model, StorageDeclaration::writableBuffer),
                                            capabilities,
                                            withoutStores,
                                            stage,
                                            storageBuffer);
        if (!accepted)
        {
            UNSCOPED_INFO(accepted.error().message);
        }
        CHECK(accepted.has_value());
    }
}

TEST_CASE("Vulkan pipelines that need sample-rate shading, min-LOD, extended gather, or vertex stores are valid or "
          "unsupported",
          "[size:medium][rhi][vulkan][shader][module]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto spirvDir = std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR);

    const std::array storageRequirements{rachis::ShaderBindingRequirement{
        .set = 0,
        .binding = 2,
        .type = rachis::ShaderResourceType::storageBuffer,
        .count = 1,
    }};
    const std::array pulledRequirements{rachis::ShaderBindingRequirement{
        .set = 0,
        .binding = 3,
        .type = rachis::ShaderResourceType::readOnlyStorageBuffer,
        .count = 1,
    }};
    const std::array textureRequirements{
        rachis::ShaderBindingRequirement{
            .set = 0,
            .binding = 0,
            .type = rachis::ShaderResourceType::sampledTexture,
            .count = 1,
        },
        rachis::ShaderBindingRequirement{
            .set = 0,
            .binding = 1,
            .type = rachis::ShaderResourceType::sampler,
            .count = 1,
        },
    };
    const std::array storageEntries{rachis::BindingLayoutEntry{
        .binding = 2,
        .type = rachis::BindingType::storageBuffer,
        .visibility = rachis::ShaderStage::vertex,
    }};
    const std::array pulledEntries{rachis::BindingLayoutEntry{
        .binding = 3,
        .type = rachis::BindingType::readOnlyStorageBuffer,
        .visibility = rachis::ShaderStage::vertex,
    }};
    const std::array textureEntries{
        rachis::BindingLayoutEntry{
            .binding = 0,
            .type = rachis::BindingType::sampledTexture,
            .visibility = rachis::ShaderStage::pixel,
        },
        rachis::BindingLayoutEntry{
            .binding = 1,
            .type = rachis::BindingType::sampler,
            .visibility = rachis::ShaderStage::pixel,
        },
    };

    struct FeatureCase final
    {
        std::string_view name;
        std::string_view vertexFile;
        std::string_view vertexEntry;
        std::span<const rachis::ShaderBindingRequirement> vertexBindings;
        std::string_view pixelFile;
        std::string_view pixelEntry;
        std::span<const rachis::ShaderBindingRequirement> pixelBindings;
        std::span<const rachis::BindingLayoutEntry> layoutEntries;
        std::uint32_t sampleCount = 1;
        bool mayBeUnsupported = true;
    };
    const std::array cases{
        FeatureCase{
            .name = "vertex-stage storage writes",
            .vertexFile = "VulkanBaseShaderFeatures.store.vs.spv",
            .vertexEntry = "storeVertex",
            .vertexBindings = storageRequirements,
            .pixelFile = "VulkanBaseShaderFeatures.solid.ps.spv",
            .pixelEntry = "solidPixel",
            .pixelBindings = {},
            .layoutEntries = storageEntries,
        },
        FeatureCase{
            .name = "read-only vertex pulling",
            .vertexFile = "VulkanBaseShaderFeatures.pulled.vs.spv",
            .vertexEntry = "pulledVertex",
            .vertexBindings = pulledRequirements,
            .pixelFile = "VulkanBaseShaderFeatures.solid.ps.spv",
            .pixelEntry = "solidPixel",
            .pixelBindings = {},
            .layoutEntries = pulledEntries,
            .mayBeUnsupported = false,
        },
        FeatureCase{
            .name = "sample-rate shading",
            .vertexFile = "VulkanBaseShaderFeatures.fullscreen.vs.spv",
            .vertexEntry = "fullscreenVertex",
            .vertexBindings = {},
            .pixelFile = "VulkanBaseShaderFeatures.sample-index.ps.spv",
            .pixelEntry = "sampleIndexPixel",
            .pixelBindings = {},
            .layoutEntries = {},
            .sampleCount = 4,
        },
        FeatureCase{
            .name = "minimum-LOD clamp",
            .vertexFile = "VulkanBaseShaderFeatures.fullscreen.vs.spv",
            .vertexEntry = "fullscreenVertex",
            .vertexBindings = {},
            .pixelFile = "VulkanBaseShaderFeatures.min-lod.ps.spv",
            .pixelEntry = "minLodPixel",
            .pixelBindings = textureRequirements,
            .layoutEntries = textureEntries,
        },
        FeatureCase{
            .name = "extended gather offsets",
            .vertexFile = "VulkanBaseShaderFeatures.fullscreen.vs.spv",
            .vertexEntry = "fullscreenVertex",
            .vertexBindings = {},
            .pixelFile = "VulkanBaseShaderFeatures.gather-offset.ps.spv",
            .pixelEntry = "gatherOffsetPixel",
            .pixelBindings = textureRequirements,
            .layoutEntries = textureEntries,
        },
    };

    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    for (const auto& featureCase : cases)
    {
        INFO(featureCase.name);
        const auto vertexBytes = loadBytes(spirvDir / featureCase.vertexFile);
        const auto pixelBytes = loadBytes(spirvDir / featureCase.pixelFile);
        REQUIRE_FALSE(vertexBytes.empty());
        REQUIRE_FALSE(pixelBytes.empty());
        const auto errorsBefore = device.diagnosticCounts().errors;

        auto vertex = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vertexBytes,
            .entryPoint = featureCase.vertexEntry,
            .shaderInterface = {.bindings = featureCase.vertexBindings},
        });
        auto pixel = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = pixelBytes,
            .entryPoint = featureCase.pixelEntry,
            .shaderInterface = {.bindings = featureCase.pixelBindings},
        });
        if (!vertex || !pixel)
        {
            CHECK(featureCase.mayBeUnsupported);
            for (const auto* module : {&vertex, &pixel})
            {
                if (*module)
                {
                    CHECK(device.destroyShaderModule(**module).has_value());
                    continue;
                }
                UNSCOPED_INFO(module->error().message);
                CHECK(module->error().code == rachis::ErrorCode::unsupported);
            }
            CHECK(device.diagnosticCounts().errors == errorsBefore);
            continue;
        }

        std::optional<rachis::BindingLayoutHandle> bindingLayout;
        if (!featureCase.layoutEntries.empty())
        {
            auto createdLayout = device.createBindingLayout({.entries = featureCase.layoutEntries});
            REQUIRE(createdLayout);
            bindingLayout = *createdLayout;
        }
        const auto bindingLayouts =
            bindingLayout.has_value() ? std::span(&*bindingLayout, 1) : std::span<const rachis::BindingLayoutHandle>{};
        auto layout = device.createPipelineLayout({.bindingLayouts = bindingLayouts});
        REQUIRE(layout);
        auto pipeline = device.createGraphicsPipeline({
            .layout = *layout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
            .sampleCount = featureCase.sampleCount,
        });
        if (!pipeline)
        {
            UNSCOPED_INFO(pipeline.error().message);
        }
        CHECK(pipeline.has_value());
        CHECK(device.diagnosticCounts().errors == errorsBefore);

        if (pipeline)
        {
            CHECK(device.destroyPipeline(*pipeline).has_value());
        }
        CHECK(device.destroyPipelineLayout(*layout).has_value());
        if (bindingLayout.has_value())
        {
            CHECK(device.destroyBindingLayout(*bindingLayout).has_value());
        }
        CHECK(device.destroyShaderModule(*vertex).has_value());
        CHECK(device.destroyShaderModule(*pixel).has_value());
    }
}

TEST_CASE("Vulkan shader modules gate SPIR-V DrawParameters on the shaderDrawParameters feature",
          "[size:small][rhi][vulkan][shader][module]")
{
    rachis::DeviceCapabilities capabilities{};
    capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
    capabilities.limits.maxComputeWorkgroupInvocations = 1024;
    const auto drawParameters = computeModuleDeclaringCapability(spirvCapabilityDrawParameters);

    auto rejected = createWithoutDevice(drawParameters, capabilities, rachis::vulkan::VulkanShaderFeatureSupport{});
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
    CHECK(rejected.error().message ==
          "SPIR-V DrawParameters capability requires the Vulkan shaderDrawParameters feature.");

    auto accepted = createWithoutDevice(
        drawParameters, capabilities, rachis::vulkan::VulkanShaderFeatureSupport{.shaderDrawParameters = true});
    if (!accepted)
    {
        UNSCOPED_INFO(accepted.error().message);
    }
    CHECK(accepted.has_value());
}

TEST_CASE("Vulkan shader modules reject malformed SPIR-V and interface mismatches",
          "[size:medium][rhi][vulkan][shader][module]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto cube = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapCube.cs.spv");
    const auto triangleVs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.vs.spv");
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(triangleVs.empty());

    for (const auto opcode : {26U, 27U, 30U, 5341U})
    {
        auto truncatedType = triangleVs;
        const std::uint32_t instruction = (1U << 16) | opcode;
        const auto instructionBytes = std::as_bytes(std::span(&instruction, 1));
        truncatedType.insert(truncatedType.end(), instructionBytes.begin(), instructionBytes.end());
        auto malformed = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = truncatedType,
            .entryPoint = "vertexMain",
        });
        REQUIRE_FALSE(malformed);
        CHECK(malformed.error().code == rachis::ErrorCode::invalidArgument);
    }
    CHECK(device.diagnosticCounts().errors == 0);

    CHECK_FALSE(device.createShaderLibrary({}).has_value());
    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto compute = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .label = "Vulkan cube compute",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    REQUIRE(compute);

    auto snapshot = device.shaderModuleInterface(*compute);
    REQUIRE(snapshot);
    CHECK(snapshot->stage == rachis::ShaderStage::compute);
    REQUIRE(snapshot->bindings.size() == 2);
    CHECK(snapshot->bindings[0].type == rachis::ShaderResourceType::readOnlyStorageBuffer);
    CHECK(snapshot->pushConstants[0].size == 16);
    REQUIRE(snapshot->workgroupSize.has_value());
    CHECK(snapshot->workgroupSize->x == 64);

    const auto fullInterface = cubeComputeInterface(bindings, push);
    push[0].size = 12;
    auto truncatedPush = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = fullInterface,
    });
    REQUIRE_FALSE(truncatedPush);
    CHECK(truncatedPush.error().code == rachis::ErrorCode::invalidArgument);

    std::vector<std::uint32_t> uniformWords(cube.size() / sizeof(std::uint32_t));
    std::memcpy(uniformWords.data(), cube.data(), cube.size());
    std::uint32_t pushVariable = 0;
    std::size_t firstType = uniformWords.size();
    for (std::size_t index = 5; index < uniformWords.size(); index += uniformWords[index] >> 16)
    {
        const auto opcode = uniformWords[index] & 0xFFFFU;
        if (opcode >= 19 && opcode <= 39 && firstType == uniformWords.size())
        {
            firstType = index;
        }
        if (opcode == 32 && uniformWords[index + 2] == 9)
        {
            uniformWords[index + 2] = 2;
        }
        if (opcode == 59 && uniformWords[index + 3] == 9)
        {
            pushVariable = uniformWords[index + 2];
            uniformWords[index + 3] = 2;
        }
    }
    REQUIRE(pushVariable != 0);
    REQUIRE(firstType < uniformWords.size());
    const std::array decorations{(4U << 16) | 71U, pushVariable, 34U, 4U, (4U << 16) | 71U, pushVariable, 33U, 0U};
    uniformWords.insert(
        uniformWords.begin() + static_cast<std::ptrdiff_t>(firstType), decorations.begin(), decorations.end());
    auto uniformPush = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = std::as_bytes(std::span(uniformWords)),
        .entryPoint = "computeMain",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    REQUIRE_FALSE(uniformPush);
    CHECK(uniformPush.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(device.diagnosticCounts().errors == 0);

    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = cube,
                        .entryPoint = "computeMain",
                        .shaderInterface = {},
                    })
                    .has_value());
    bindings[0].type = rachis::ShaderResourceType::uniformBuffer;
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = cube,
                        .entryPoint = "computeMain",
                        .shaderInterface =
                            rachis::ShaderInterfaceDesc{
                                .bindings = bindings,
                                .pushConstants = push,
                                .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
                            },
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = cube,
                        .entryPoint = "missingMain",
                        .shaderInterface = cubeComputeInterface(bindings, push),
                    })
                    .has_value());

    constexpr std::array notSpirv{std::byte{0x11}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}};
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::vertex,
                        .bytecode = notSpirv,
                        .entryPoint = "vertexMain",
                    })
                    .has_value());

    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
        .label = "Vulkan triangle vertex",
    });
    if (!vertex)
    {
        UNSCOPED_INFO(vertex.error().message);
    }
    REQUIRE(vertex);
    REQUIRE(device.destroyShaderModule(*compute).has_value());
    CHECK_FALSE(device.shaderModuleInterface(*compute).has_value());
    auto surviving = device.shaderModuleInterface(*vertex);
    REQUIRE(surviving);
    CHECK(surviving->stage == rachis::ShaderStage::vertex);
    REQUIRE(device.destroyShaderModule(*vertex).has_value());
}

TEST_CASE("Vulkan shader modules reflect dxc StructuredBuffer and ByteAddressBuffer bindings as read-only storage",
          "[size:medium][rhi][vulkan][shader][module]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto bytecode = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "ReadOnlyStorageBuffers.spv");
    REQUIRE_FALSE(bytecode.empty());

    constexpr auto readOnly = rachis::ShaderResourceType::readOnlyStorageBuffer;
    constexpr auto writable = rachis::ShaderResourceType::storageBuffer;
    constexpr std::array bindings{
        rachis::ShaderBindingRequirement{.set = 0, .binding = 0, .type = readOnly, .count = 1},
        rachis::ShaderBindingRequirement{.set = 0, .binding = 1, .type = writable, .count = 1},
        rachis::ShaderBindingRequirement{.set = 0, .binding = 2, .type = readOnly, .count = 1},
        rachis::ShaderBindingRequirement{.set = 0, .binding = 3, .type = readOnly, .count = 4},
    };
    for (std::size_t flipped = 0; flipped < bindings.size(); ++flipped)
    {
        INFO("binding " << bindings[flipped].binding);
        auto mismatched = bindings;
        mismatched[flipped].type = mismatched[flipped].type == readOnly ? writable : readOnly;
        auto rejected = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}, mismatched),
        });
        CHECK_FALSE(rejected);
        if (!rejected)
        {
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
            CHECK(rejected.error().message ==
                  "A declared shader binding does not match the reflected SPIR-V set, binding, type, or count.");
        }
        else
        {
            CHECK(device.destroyShaderModule(*rejected).has_value());
        }
    }

    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Vulkan dxc read-only storage",
        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}, bindings),
    });
    if (!shader)
    {
        UNSCOPED_INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());

    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint32_t regionCount = 6;
    constexpr std::uint64_t regionSize = valueCount * sizeof(std::uint32_t);
    std::vector<std::uint32_t> inputs(static_cast<std::size_t>(valueCount) * regionCount);
    for (std::uint32_t region = 0; region < regionCount; ++region)
    {
        for (std::uint32_t index = 0; index < valueCount; ++index)
        {
            inputs[(region * valueCount) + index] = (region * 1000U) + (index * 3U) + 1U;
        }
    }
    const auto input = [&](std::uint32_t region, std::uint32_t index) { return inputs[(region * valueCount) + index]; };

    auto inputBuffer = device.createBuffer({
        .size = regionSize * regionCount,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan dxc read-only inputs",
    });
    auto outputBuffer = device.createBuffer({
        .size = regionSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan dxc read-only output",
    });
    auto upload = device.createBuffer({
        .size = regionSize * regionCount,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan dxc read-only upload",
    });
    auto readback = device.createBuffer({
        .size = regionSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan dxc read-only readback",
    });
    REQUIRE(inputBuffer.has_value());
    REQUIRE(outputBuffer.has_value());
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(device.writeBuffer(*upload, 0, std::as_bytes(std::span(inputs))).has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{3, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute, 4},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Vulkan dxc read-only layout"});
    REQUIRE(bindingLayout.has_value());
    const auto inputRegion = [&](std::uint32_t region, std::uint32_t stride) {
        return rachis::BufferBinding{
            .buffer = *inputBuffer,
            .offset = region * regionSize,
            .size = regionSize,
            .structureByteStride = stride,
        };
    };
    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = inputRegion(0, sizeof(std::uint32_t))},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *outputBuffer,
                    .size = regionSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
        rachis::BindingWrite{.binding = 2, .resource = inputRegion(1, 0)},
        rachis::BindingWrite{.binding = 3, .arrayElement = 0, .resource = inputRegion(2, sizeof(std::uint32_t))},
        rachis::BindingWrite{.binding = 3, .arrayElement = 1, .resource = inputRegion(3, sizeof(std::uint32_t))},
        rachis::BindingWrite{.binding = 3, .arrayElement = 2, .resource = inputRegion(4, sizeof(std::uint32_t))},
        rachis::BindingWrite{.binding = 3, .arrayElement = 3, .resource = inputRegion(5, sizeof(std::uint32_t))},
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     rachis::BindingSetDesc{
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Vulkan dxc read-only set",
                                                     });
    if (!bindingSet)
    {
        UNSCOPED_INFO(bindingSet.error().message);
    }
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = bindingLayouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Vulkan dxc read-only pipeline",
    });
    if (!pipeline)
    {
        UNSCOPED_INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *upload,
                                .destination = *inputBuffer,
                                .size = regionSize * regionCount,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *inputBuffer,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::shaderResource,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *outputBuffer,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *outputBuffer,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *outputBuffer,
                                .destination = *readback,
                                .size = regionSize,
                            })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    std::vector<std::uint32_t> output(valueCount, 0U);
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(output))).has_value());
    std::uint32_t firstMismatch = valueCount;
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        const auto arrayed = input(2, index) + (input(3, index) * 2U) + (input(4, index) * 3U) + (input(5, index) * 4U);
        if (output[index] != input(0, index) + (input(1, index) * 5U) + (arrayed * 7U))
        {
            firstMismatch = index;
            break;
        }
    }
    CAPTURE(firstMismatch);
    CHECK(firstMismatch == valueCount);

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*outputBuffer).has_value());
    REQUIRE(device.destroyBuffer(*inputBuffer).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan shader libraries select immutable SPIR-V by stage, name, and specialization",
          "[size:medium][rhi][vulkan][shader][library]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto cube = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapCube.cs.spv");
    const auto spec = loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / "VulkanSpecialization.spv");
    const auto triangleVs = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapTriangle.vs.spv");
    REQUIRE_FALSE(cube.empty());
    REQUIRE_FALSE(spec.empty());
    REQUIRE_FALSE(triangleVs.empty());

    const auto one = rachis::SpecializationConstant::makeUnsignedInt32(7, 1);
    const auto two = rachis::SpecializationConstant::makeUnsignedInt32(7, 2);
    std::array firstSpec{one};
    std::array secondSpec{two};
    const std::array entries{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = spec,
            .entryPoint = "main",
            .specializationConstants = firstSpec,
        },
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = spec,
            .entryPoint = "main",
            .specializationConstants = secondSpec,
        },
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::vertex,
            .bytecode = triangleVs,
            .entryPoint = "vertexMain",
        },
    };
    auto library = device.createShaderLibrary({
        .entries = entries,
        .label = "Vulkan specialized library",
    });
    REQUIRE(library);

    const std::array duplicate{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = spec,
            .entryPoint = "main",
            .specializationConstants = firstSpec,
        },
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = spec,
            .entryPoint = "main",
            .specializationConstants = firstSpec,
        },
    };
    CHECK_FALSE(device.createShaderLibrary({.entries = duplicate}).has_value());

    std::array specBinding{
        rachis::ShaderBindingRequirement{
            .set = 0,
            .binding = 0,
            .type = rachis::ShaderResourceType::storageBuffer,
            .count = 1,
        },
    };
    auto selected = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .entryPoint = "main",
        .library = *library,
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = specBinding,
                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
            },
        .specializationConstants = secondSpec,
    });
    REQUIRE(selected);
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = spec,
                        .entryPoint = "main",
                        .shaderInterface =
                            rachis::ShaderInterfaceDesc{
                                .bindings = specBinding,
                                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
                            },
                        .specializationConstants = firstSpec,
                    })
                    .has_value());

    std::array<rachis::ShaderBindingRequirement, 2> cubeBindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> cubePush{};
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .entryPoint = "computeMain",
                        .library = *library,
                        .shaderInterface = cubeComputeInterface(cubeBindings, cubePush),
                    })
                    .has_value());

    REQUIRE(device.destroyShaderModule(*selected).has_value());
    REQUIRE(device.destroyShaderLibrary(*library).has_value());
    CHECK_FALSE(device.destroyShaderLibrary(*library).has_value());
}

TEST_CASE("Vulkan shader libraries reject specialization values for spec constants that are not bool or 32-bit",
          "[size:small][rhi][vulkan][shader][library]")
{
    struct OverrideCase final
    {
        std::string_view variant;
        rachis::SpecializationConstant value;
        bool accepted = false;
    };
    constexpr auto specId = SpecConstantModule::specId;
    const std::array cases{
        OverrideCase{"bool", rachis::SpecializationConstant::makeBoolean(specId, false), true},
        OverrideCase{"uint32", rachis::SpecializationConstant::makeUnsignedInt32(specId, 5), true},
        OverrideCase{"int32", rachis::SpecializationConstant::makeSignedInt32(specId, -5), true},
        OverrideCase{"float32", rachis::SpecializationConstant::makeFloat32(specId, 5.0F), true},
        OverrideCase{"uint8", rachis::SpecializationConstant::makeUnsignedInt32(specId, 5), false},
        OverrideCase{"uint16", rachis::SpecializationConstant::makeUnsignedInt32(specId, 5), false},
        OverrideCase{"uint64", rachis::SpecializationConstant::makeUnsignedInt32(specId, 5), false},
        OverrideCase{"float16", rachis::SpecializationConstant::makeFloat32(specId, 5.0F), false},
        OverrideCase{"float64", rachis::SpecializationConstant::makeFloat32(specId, 5.0F), false},
    };
    rachis::DeviceCapabilities capabilities{};
    capabilities.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
    capabilities.limits.maxComputeWorkgroupInvocations = 1024;
    capabilities.shaderFloat16 = true;
    capabilities.shaderInt16 = true;
    capabilities.shaderInt64 = true;
    VolkDeviceTable table{};
    table.vkCreateShaderModule = &fakeCreateShaderModule;
    table.vkDestroyShaderModule = &fakeDestroyShaderModule;
    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(reinterpret_cast<VkDevice>(std::uintptr_t{0x4000}),
                       &table,
                       capabilities,
                       nullptr,
                       false,
                       {.shaderFloat64 = true});
    const std::array bindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const rachis::ShaderInterfaceDesc shaderInterface{
        .bindings = bindings,
        .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
    };
    const auto createLibrary = [&](std::span<const std::byte> bytecode,
                                   std::span<const rachis::SpecializationConstant> constants) {
        const std::array entries{rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .specializationConstants = constants,
        }};
        return shaders.createShaderLibrary({.entries = entries});
    };
    const auto createModuleFromLibrary = [&](std::span<const std::byte> bytecode,
                                             std::span<const rachis::SpecializationConstant> constants) {
        auto library = createLibrary(bytecode, constants);
        if (!library)
        {
            UNSCOPED_INFO(library.error().message);
            return false;
        }
        auto module = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .entryPoint = "main",
            .library = *library,
            .shaderInterface = shaderInterface,
            .specializationConstants = constants,
        });
        if (!module)
        {
            UNSCOPED_INFO(module.error().message);
        }
        else
        {
            CHECK(shaders.destroyShaderModule(*module).has_value());
        }
        CHECK(shaders.destroyShaderLibrary(*library).has_value());
        return module.has_value();
    };
    for (const auto& testCase : cases)
    {
        INFO(testCase.variant);
        const auto bytecode = specConstantModule(specConstantVariant(testCase.variant));
        CHECK(createModuleFromLibrary(bytecode, {}));
        const std::array values{testCase.value};
        if (testCase.accepted)
        {
            CHECK(createModuleFromLibrary(bytecode, values));
            continue;
        }
        auto rejected = createLibrary(bytecode, values);
        CHECK_FALSE(rejected);
        if (!rejected)
        {
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
            CHECK(rejected.error().message ==
                  "A specialization value overrides a SPIR-V spec constant that is not a bool or a 32-bit scalar.");
        }
        else
        {
            CHECK(shaders.destroyShaderLibrary(*rejected).has_value());
        }
    }

    const auto wide = specConstantModule(specConstantVariant("uint64"));
    const std::array unknown{rachis::SpecializationConstant::makeUnsignedInt32(specId + 1U, 5)};
    auto unknownLibrary = createLibrary(wide, unknown);
    CHECK_FALSE(unknownLibrary);
    if (!unknownLibrary)
    {
        CHECK(unknownLibrary.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(unknownLibrary.error().message == "A specialization constant ID is not present in the SPIR-V module.");
    }
    else
    {
        CHECK(shaders.destroyShaderLibrary(*unknownLibrary).has_value());
    }
}

TEST_CASE("Vulkan compute pipelines run 16-bit and 64-bit spec constants at their defaults beside 32-bit overrides",
          "[size:medium][rhi][vulkan][shader][library]")
{
    struct DispatchCase final
    {
        std::string_view variant;
        bool rachis::DeviceCapabilities::* feature = nullptr;
        const char* featureName = nullptr;
        bool overridden = false;
        std::uint32_t expected = 0;
    };
    const auto testCase =
        GENERATE(DispatchCase{"uint32", nullptr, nullptr, true, 5},
                 DispatchCase{"uint16", &rachis::DeviceCapabilities::shaderInt16, "shaderInt16", false, 7},
                 DispatchCase{"uint64", &rachis::DeviceCapabilities::shaderInt64, "shaderInt64", false, 7});
    INFO(testCase.variant);
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (testCase.feature != nullptr &&
        rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(device).*testCase.feature,
                                            testCase.featureName))
    {
        return;
    }
    const auto bytecode = specConstantModule(specConstantVariant(testCase.variant));
    const std::array five{rachis::SpecializationConstant::makeUnsignedInt32(SpecConstantModule::specId, 5)};
    const auto libraryEntries = [&](std::span<const rachis::SpecializationConstant> constants) {
        return std::array{rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .specializationConstants = constants,
        }};
    };
    if (!testCase.overridden)
    {
        const auto rejectedEntries = libraryEntries(five);
        auto rejected = device.createShaderLibrary({.entries = rejectedEntries});
        CHECK_FALSE(rejected);
        if (!rejected)
        {
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
        }
        else
        {
            CHECK(device.destroyShaderLibrary(*rejected).has_value());
        }
    }

    const auto constants = testCase.overridden ? std::span<const rachis::SpecializationConstant>(five)
                                               : std::span<const rachis::SpecializationConstant>{};
    const auto entries = libraryEntries(constants);
    auto library = device.createShaderLibrary({.entries = entries, .label = "Vulkan spec-constant width library"});
    if (!library)
    {
        UNSCOPED_INFO(library.error().message);
    }
    REQUIRE(library.has_value());
    const std::array bindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .entryPoint = "main",
        .label = "Vulkan spec-constant width shader",
        .library = *library,
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = bindings,
                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
            },
        .specializationConstants = constants,
    });
    if (!shader)
    {
        UNSCOPED_INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());

    constexpr std::uint64_t outputBytes = sizeof(std::uint32_t);
    auto output = device.createBuffer({
        .size = outputBytes,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan spec-constant width output",
    });
    auto readback = device.createBuffer({
        .size = outputBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan spec-constant width readback",
    });
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = outputBytes,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    if (!bindingSet)
    {
        UNSCOPED_INFO(bindingSet.error().message);
    }
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = bindingLayouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    if (!pipeline)
    {
        UNSCOPED_INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *output,
                                .destination = *readback,
                                .size = outputBytes,
                            })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    std::uint32_t written = 0;
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(&written, 1))).has_value());
    CHECK(written == testCase.expected);

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyShaderLibrary(*library).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan shader modules reject every truncation of a cooked SPIR-V module",
          "[size:medium][rhi][vulkan][shader][module]")
{
    const auto validation = GENERATE(rachis::ValidationLevel::disabled, rachis::ValidationLevel::standard);
    auto created = rachis::createDevice(vulkanDesc(validation));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto cube = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "BootstrapCube.cs.spv");
    REQUIRE_FALSE(cube.empty());
    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    const auto shaderInterface = cubeComputeInterface(bindings, push);

    std::vector<std::size_t> acceptedLengths;
    for (std::size_t length = 0; length < cube.size(); ++length)
    {
        if (!rejectsAsInvalidArgument(device, std::span(cube).first(length), "computeMain", shaderInterface))
        {
            acceptedLengths.push_back(length);
        }
    }
    CHECK(acceptedLengths == std::vector<std::size_t>{});
    CHECK(device.diagnosticCounts().errors == 0);

    auto complete = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = shaderInterface,
    });
    REQUIRE(complete);
    REQUIRE(device.destroyShaderModule(*complete).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan shader modules reject SPIR-V whose function structure is cut or broken",
          "[size:medium][rhi][vulkan][shader][module]")
{
    const auto validation = GENERATE(rachis::ValidationLevel::disabled, rachis::ValidationLevel::standard);
    auto created = rachis::createDevice(vulkanDesc(validation));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const CallingComputeModule parts;
    const rachis::ShaderInterfaceDesc shaderInterface{.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}};

    const auto completeModuleWithBound = [&](std::uint32_t bound) {
        auto header = parts.header;
        header[spirvHeaderIdBoundWord] = bound;
        return joinSpirvBytes({header,
                               parts.declarations,
                               parts.callerBody,
                               parts.callerEnd,
                               parts.calleeBody,
                               parts.calleeReturn,
                               parts.calleeEnd});
    };
    const auto complete = completeModuleWithBound(CallingComputeModule::idBound);
    auto accepted = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = complete,
        .entryPoint = "main",
        .shaderInterface = shaderInterface,
    });
    REQUIRE(accepted);
    REQUIRE(device.destroyShaderModule(*accepted).has_value());

    const std::array<std::pair<std::string_view, std::vector<std::byte>>, 9> damaged{{
        {"cut at the function boundary before the callee is defined",
         joinSpirvBytes({parts.header, parts.declarations, parts.callerBody, parts.callerEnd})},
        {"cut inside the caller", joinSpirvBytes({parts.header, parts.declarations, parts.callerBody})},
        {"cut inside the callee",
         joinSpirvBytes({parts.header,
                         parts.declarations,
                         parts.callerBody,
                         parts.callerEnd,
                         parts.calleeBody,
                         parts.calleeReturn})},
        {"cut inside the callee's OpLabel",
         joinSpirvBytes({parts.header,
                         parts.declarations,
                         parts.callerBody,
                         parts.callerEnd,
                         std::span(parts.calleeBody).first(parts.calleeBody.size() - 1)})},
        {"an OpFunction inside another function",
         joinSpirvBytes({parts.header,
                         parts.declarations,
                         parts.callerBody,
                         parts.calleeBody,
                         parts.calleeReturn,
                         parts.calleeEnd})},
        {"an OpFunctionEnd outside any function",
         joinSpirvBytes({parts.header,
                         parts.declarations,
                         parts.callerBody,
                         parts.callerEnd,
                         parts.callerEnd,
                         parts.calleeBody,
                         parts.calleeReturn,
                         parts.calleeEnd})},
        {"a last block without a terminator",
         joinSpirvBytes(
             {parts.header, parts.declarations, parts.callerBody, parts.callerEnd, parts.calleeBody, parts.calleeEnd})},
        {"a callee label at the ID bound", completeModuleWithBound(CallingComputeModule::calleeLabelId)},
        {"a zero ID bound", completeModuleWithBound(0)},
    }};
    for (const auto& [damage, bytecode] : damaged)
    {
        INFO(damage);
        CHECK(rejectsAsInvalidArgument(device, bytecode, "main", shaderInterface));
    }
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan shader modules reject SPIR-V cut before a function or ID that the module refers to",
          "[size:medium][rhi][vulkan][shader][module]")
{
    const auto validation = GENERATE(rachis::ValidationLevel::disabled, rachis::ValidationLevel::standard);
    auto created = rachis::createDevice(vulkanDesc(validation));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const MultiEntryComputeModule parts;
    const rachis::ShaderInterfaceDesc shaderInterface{.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}};

    const auto declarations = joinSpirvWords({parts.header,
                                              parts.capabilities,
                                              parts.mainEntryPoint,
                                              parts.otherEntryPoint,
                                              parts.mainExecutionMode,
                                              parts.otherExecutionMode,
                                              parts.helperName,
                                              parts.sumDecoration,
                                              parts.types});
    const auto complete = joinSpirvBytes({declarations, parts.mainFunction, parts.otherFunction, parts.helperFunction});
    const std::array completeEntries{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = complete,
            .entryPoint = "main",
        },
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = complete,
            .entryPoint = "other",
        },
    };
    auto library = device.createShaderLibrary({.entries = completeEntries});
    REQUIRE(library);
    REQUIRE(device.destroyShaderLibrary(*library).has_value());
    for (const std::string_view entryPoint : {std::string_view("main"), std::string_view("other")})
    {
        INFO(entryPoint);
        auto accepted = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = complete,
            .entryPoint = entryPoint,
            .shaderInterface = shaderInterface,
        });
        REQUIRE(accepted);
        REQUIRE(device.destroyShaderModule(*accepted).has_value());
    }
    CHECK(device.diagnosticCounts().errors == 0);

    const std::array<std::pair<std::string_view, std::vector<std::byte>>, 6> damaged{{
        {"an entry point whose function the cut removed",
         joinSpirvBytes({parts.header,
                         parts.capabilities,
                         parts.mainEntryPoint,
                         parts.otherEntryPoint,
                         parts.mainExecutionMode,
                         parts.types,
                         parts.mainFunction})},
        {"an execution mode for a function the module does not define",
         joinSpirvBytes({parts.header,
                         parts.capabilities,
                         parts.mainEntryPoint,
                         parts.mainExecutionMode,
                         parts.otherExecutionMode,
                         parts.types,
                         parts.mainFunction})},
        {"a name whose function the cut removed",
         joinSpirvBytes({parts.header,
                         parts.capabilities,
                         parts.mainEntryPoint,
                         parts.mainExecutionMode,
                         parts.helperName,
                         parts.types,
                         parts.mainFunction})},
        {"a decoration whose target the cut removed",
         joinSpirvBytes({parts.header,
                         parts.capabilities,
                         parts.mainEntryPoint,
                         parts.mainExecutionMode,
                         parts.sumDecoration,
                         parts.types,
                         parts.mainFunction})},
        {"the whole module cut after the first entry point's function",
         joinSpirvBytes({declarations, parts.mainFunction})},
        {"the whole module cut after the second entry point's function",
         joinSpirvBytes({declarations, parts.mainFunction, parts.otherFunction})},
    }};
    for (const auto& [damage, bytecode] : damaged)
    {
        INFO(damage);
        CHECK(rejectsAsInvalidArgument(device, bytecode, "main", shaderInterface));
    }
    CHECK(device.diagnosticCounts().errors == 0);
}
