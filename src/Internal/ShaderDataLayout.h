#pragma once

#include "Internal/ShaderBinaryTool.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rachis::shader_tool
{

    enum class DataUsage : std::uint8_t
    {
        constantBuffer,
        pushConstants,
        structuredBuffer,
    };

    struct LayoutType;

    struct LayoutField final
    {
        std::string name;
        std::uint32_t offset = 0;
        std::shared_ptr<const LayoutType> type;
    };

    struct LayoutType final
    {
        enum class Kind : std::uint8_t
        {
            scalar,
            vector,
            matrix,
            array,
            structure,
        };

        Kind kind = Kind::scalar;
        std::string_view scalar;
        std::uint32_t scalarSize = 4;
        std::uint32_t components = 1;
        std::uint32_t vectorCount = 1;
        std::uint32_t stride = 0;
        std::uint32_t count = 0;
        std::shared_ptr<const LayoutType> element;
        std::string name;
        std::vector<LayoutField> fields;
    };

    struct DataResource final
    {
        std::string name;
        std::string typeName;
        DataUsage usage = DataUsage::constantBuffer;
        std::uint32_t set = 0;
        std::uint32_t binding = 0;
    };

    struct DataBlock final
    {
        DataResource resource;
        std::shared_ptr<const LayoutType> type;
        std::uint32_t size = 0;
    };

    namespace detail
    {

        struct ScalarInfo final
        {
            std::string_view cppType;
            std::uint32_t size = 4;
        };

        [[nodiscard]] inline std::optional<ScalarInfo> slangScalar(std::string_view name) noexcept
        {
            if (name == "float32")
            {
                return ScalarInfo{"float", 4};
            }
            if (name == "int32")
            {
                return ScalarInfo{"std::int32_t", 4};
            }
            if (name == "uint32" || name == "bool")
            {
                return ScalarInfo{"std::uint32_t", 4};
            }
            if (name == "float16" || name == "uint16")
            {
                return ScalarInfo{"std::uint16_t", 2};
            }
            if (name == "int16")
            {
                return ScalarInfo{"std::int16_t", 2};
            }
            if (name == "float64")
            {
                return ScalarInfo{"double", 8};
            }
            if (name == "int64")
            {
                return ScalarInfo{"std::int64_t", 8};
            }
            if (name == "uint64")
            {
                return ScalarInfo{"std::uint64_t", 8};
            }
            return std::nullopt;
        }

        [[nodiscard]] inline std::string_view layoutJsonString(const JsonValue* value) noexcept
        {
            const auto* text = value != nullptr ? value->asString() : nullptr;
            return text != nullptr ? std::string_view(*text) : std::string_view{};
        }

        [[nodiscard]] inline std::uint32_t layoutJsonUint(const JsonValue* object, std::string_view key) noexcept
        {
            const auto* value = object != nullptr ? object->find(key) : nullptr;
            return value != nullptr ? value->asUint32().value_or(0) : 0;
        }

        [[nodiscard]] inline std::uint32_t cppSize(const LayoutType& type);

        [[nodiscard]] inline std::uint32_t cppAlignment(const LayoutType& type)
        {
            switch (type.kind)
            {
            case LayoutType::Kind::scalar:
            case LayoutType::Kind::vector:
            case LayoutType::Kind::matrix:
                return type.scalarSize;
            case LayoutType::Kind::array:
                return type.element != nullptr ? cppAlignment(*type.element) : 1U;
            case LayoutType::Kind::structure: {
                std::uint32_t alignment = 1;
                for (const auto& field : type.fields)
                {
                    alignment = std::max(alignment, cppAlignment(*field.type));
                }
                return alignment;
            }
            }
            return 1;
        }

        [[nodiscard]] constexpr std::uint32_t roundUp(std::uint32_t value, std::uint32_t alignment) noexcept
        {
            return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
        }

        [[nodiscard]] inline std::uint32_t structureEnd(const LayoutType& type)
        {
            std::uint32_t end = 0;
            for (const auto& field : type.fields)
            {
                end = std::max(end, field.offset + cppSize(*field.type));
            }
            return end;
        }

        inline std::uint32_t cppSize(const LayoutType& type)
        {
            switch (type.kind)
            {
            case LayoutType::Kind::scalar:
                return type.scalarSize;
            case LayoutType::Kind::vector:
                return type.scalarSize * type.components;
            case LayoutType::Kind::matrix: {
                const auto vectorSize = type.scalarSize * type.components;
                return type.vectorCount <= 1 ? vectorSize : (type.vectorCount - 1) * type.stride + vectorSize;
            }
            case LayoutType::Kind::array: {
                const auto elementSize = cppSize(*type.element);
                return type.count == 0 ? 0 : (type.count - 1) * std::max(type.stride, elementSize) + elementSize;
            }
            case LayoutType::Kind::structure:
                return roundUp(structureEnd(type), cppAlignment(type));
            }
            return 0;
        }

        [[nodiscard]] inline std::expected<std::shared_ptr<const LayoutType>, std::string> slangLayoutType(
            const JsonValue& type, const JsonValue* binding, std::size_t depth)
        {
            if (depth > 32)
            {
                return std::unexpected("A buffer type nests too deeply.");
            }
            auto result = std::make_shared<LayoutType>();
            const auto kind = layoutJsonString(type.find("kind"));
            const auto scalarOf = [](const JsonValue& scalarType) -> std::optional<ScalarInfo> {
                return slangScalar(layoutJsonString(scalarType.find("scalarType")));
            };
            if (kind == "scalar")
            {
                const auto scalar = scalarOf(type);
                if (!scalar.has_value())
                {
                    return std::unexpected("A buffer holds the unsupported scalar type '" +
                                           std::string(layoutJsonString(type.find("scalarType"))) + "'.");
                }
                result->kind = LayoutType::Kind::scalar;
                result->scalar = scalar->cppType;
                result->scalarSize = scalar->size;
                return result;
            }
            if (kind == "vector" || kind == "matrix")
            {
                const auto* elementType = type.find("elementType");
                const auto scalar = elementType != nullptr ? scalarOf(*elementType) : std::nullopt;
                if (!scalar.has_value())
                {
                    return std::unexpected("A buffer holds a vector or matrix of an unsupported scalar type.");
                }
                result->scalar = scalar->cppType;
                result->scalarSize = scalar->size;
                if (kind == "vector")
                {
                    result->kind = LayoutType::Kind::vector;
                    result->components = layoutJsonUint(&type, "elementCount");
                    return result;
                }
                result->kind = LayoutType::Kind::matrix;
                result->components = layoutJsonUint(&type, "rowCount");
                result->vectorCount = layoutJsonUint(&type, "columnCount");
                const auto vectorSize = result->components * result->scalarSize;
                const auto size = layoutJsonUint(binding, "size");
                if (binding == nullptr || size < vectorSize)
                {
                    return std::unexpected("A matrix in a buffer has no reflected size.");
                }
                if (result->vectorCount <= 1)
                {
                    result->stride = vectorSize;
                    return result;
                }
                const auto span = size - vectorSize;
                result->stride = span / (result->vectorCount - 1);
                if (span % (result->vectorCount - 1) != 0 || result->stride < vectorSize ||
                    result->stride % result->scalarSize != 0)
                {
                    return std::unexpected(
                        "A buffer holds a row_major matrix. rachis_add_shader supports column-major matrices only.");
                }
                return result;
            }
            if (kind == "array")
            {
                const auto* elementType = type.find("elementType");
                result->kind = LayoutType::Kind::array;
                result->count = layoutJsonUint(&type, "elementCount");
                result->stride = layoutJsonUint(binding, "elementStride");
                if (elementType == nullptr || result->count == 0)
                {
                    return std::unexpected("A buffer holds an array without a size, which C++ cannot lay out.");
                }
                if (layoutJsonString(elementType->find("kind")) == "matrix")
                {
                    return std::unexpected("Arrays of matrices in buffers are not supported.");
                }
                auto element = slangLayoutType(*elementType, nullptr, depth + 1);
                if (!element)
                {
                    return std::unexpected(std::move(element.error()));
                }
                result->element = std::move(*element);
                if (result->stride == 0)
                {
                    result->stride = cppSize(*result->element);
                }
                return result;
            }
            if (kind == "struct")
            {
                result->kind = LayoutType::Kind::structure;
                result->name = std::string(layoutJsonString(type.find("name")));
                const auto* fields = type.find("fields");
                if (fields == nullptr || fields->asArray() == nullptr)
                {
                    return std::unexpected("A structure in a buffer has no fields.");
                }
                for (const auto& field : *fields->asArray())
                {
                    const auto* fieldType = field.find("type");
                    const auto* fieldBinding = field.find("binding");
                    if (fieldType == nullptr || fieldBinding == nullptr)
                    {
                        return std::unexpected("A structure field in a buffer has no layout.");
                    }
                    auto converted = slangLayoutType(*fieldType, fieldBinding, depth + 1);
                    if (!converted)
                    {
                        return std::unexpected(std::move(converted.error()));
                    }
                    result->fields.push_back(LayoutField{
                        .name = std::string(layoutJsonString(field.find("name"))),
                        .offset = layoutJsonUint(fieldBinding, "offset"),
                        .type = std::move(*converted),
                    });
                }
                return result;
            }
            return std::unexpected("A buffer holds the unsupported type kind '" + std::string(kind) + "'.");
        }

        [[nodiscard]] inline std::uint32_t slangUniformSize(const JsonValue& type) noexcept
        {
            const auto* sizes = type.find("sizes");
            if (sizes == nullptr || sizes->asArray() == nullptr)
            {
                return 0;
            }
            for (const auto& item : *sizes->asArray())
            {
                const auto* kind = item.find("kind");
                if (kind != nullptr && kind->asString() != nullptr && *kind->asString() == "uniform")
                {
                    return layoutJsonUint(&item, "value");
                }
            }
            return 0;
        }

    } // namespace detail

    [[nodiscard]] inline std::expected<DataBlock, std::string> slangDataBlock(const detail::JsonValue& parameter,
                                                                              const DataResource& resource,
                                                                              bool metal)
    {
        const auto* type = parameter.find("type");
        if (type == nullptr)
        {
            return std::unexpected("The resource '" + resource.name + "' has no reflected type.");
        }
        const detail::JsonValue* structType = nullptr;
        const detail::JsonValue* binding = nullptr;
        if (resource.usage == DataUsage::structuredBuffer)
        {
            structType = type->find("resultType");
        }
        else if (const auto* element = type->find("elementVarLayout"); element != nullptr)
        {
            structType = element->find("type");
            binding = element->find("binding");
        }
        if (structType == nullptr)
        {
            structType = type->find("elementType");
        }
        if (structType == nullptr)
        {
            return std::unexpected("The resource '" + resource.name + "' has no reflected structure.");
        }
        auto layout = detail::slangLayoutType(*structType, binding, 0);
        if (!layout)
        {
            return std::unexpected("The resource '" + resource.name + "': " + layout.error());
        }
        if ((*layout)->kind != LayoutType::Kind::structure)
        {
            return std::unexpected("The resource '" + resource.name + "' does not hold a structure.");
        }
        auto named = std::make_shared<LayoutType>(**layout);
        named->name = resource.typeName;
        std::uint32_t size = 0;
        if (resource.usage == DataUsage::structuredBuffer)
        {
            const auto packed = detail::roundUp(detail::structureEnd(*named), detail::cppAlignment(*named));
            const auto reflected = metal ? detail::slangUniformSize(*structType) : 0U;
            size = reflected != 0 ? reflected : packed;
        }
        else
        {
            size = std::max(detail::layoutJsonUint(binding, "size"), detail::structureEnd(*named));
        }
        return DataBlock{.resource = resource, .type = std::move(named), .size = size};
    }

    namespace detail
    {
        class SpirvLayoutModule final
        {
        public:
            struct Type final
            {
                std::uint32_t opcode = 0;
                std::vector<std::uint32_t> operands;
            };

            struct Variable final
            {
                std::uint32_t pointerType = 0;
                std::uint32_t storageClass = 0;
            };

            struct EntryPoint final
            {
                std::uint32_t executionModel = 0;
                std::uint32_t function = 0;
                std::string name;
                std::vector<std::uint32_t> interfaceIds;
            };

            struct InterfaceBinding final
            {
                std::string name;
                std::uint32_t set = 0;
                std::uint32_t binding = 0;
                std::string_view type;
                std::uint32_t count = 1;
            };

            struct Interface final
            {
                std::vector<InterfaceBinding> bindings;
                std::vector<DataResource> dataResources;
                std::vector<std::uint32_t> vertexInputLocations;
                std::optional<std::array<std::uint32_t, 3>> workgroupSize;
            };

            static constexpr std::uint32_t executionModelVertex = 0;
            static constexpr std::uint32_t executionModelFragment = 4;
            static constexpr std::uint32_t executionModelGLCompute = 5;

            [[nodiscard]] static std::expected<SpirvLayoutModule, std::string> parse(std::span<const std::byte> bytes)
            {
                if (bytes.size() % 4 != 0 || bytes.size() < 20)
                {
                    return std::unexpected("The SPIR-V module is truncated.");
                }
                std::vector<std::uint32_t> words(bytes.size() / 4);
                std::memcpy(words.data(), bytes.data(), bytes.size());
                if (words[0] != 0x07230203U)
                {
                    return std::unexpected("The Vulkan layout source is not a SPIR-V module.");
                }
                SpirvLayoutModule module;
                module.version_ = words[1];
                for (std::size_t cursor = 5; cursor < words.size();)
                {
                    const auto wordCount = words[cursor] >> 16U;
                    const auto opcode = words[cursor] & 0xFFFFU;
                    if (wordCount == 0 || cursor + wordCount > words.size())
                    {
                        return std::unexpected("The SPIR-V module has a malformed instruction.");
                    }
                    module.instruction(opcode, std::span(words).subspan(cursor + 1, wordCount - 1));
                    cursor += wordCount;
                }
                return module;
            }

            [[nodiscard]] std::expected<std::shared_ptr<const LayoutType>, std::string> layoutOf(
                std::uint32_t typeId, std::uint32_t matrixStride, bool rowMajor, std::size_t depth) const
            {
                if (depth > 32)
                {
                    return std::unexpected("A SPIR-V buffer type nests too deeply.");
                }
                const auto found = types_.find(typeId);
                if (found == types_.end())
                {
                    return std::unexpected("A SPIR-V buffer member has an unknown type.");
                }
                const auto& type = found->second;
                auto result = std::make_shared<LayoutType>();
                switch (type.opcode)
                {
                case opTypeBool:
                case opTypeInt:
                case opTypeFloat: {
                    const auto scalar = scalarInfo(type);
                    if (!scalar.has_value())
                    {
                        return std::unexpected("A SPIR-V buffer holds an unsupported scalar type.");
                    }
                    result->kind = LayoutType::Kind::scalar;
                    result->scalar = scalar->cppType;
                    result->scalarSize = scalar->size;
                    return result;
                }
                case opTypeVector: {
                    auto component = layoutOf(type.operands.at(0), 0, false, depth + 1);
                    if (!component || (*component)->kind != LayoutType::Kind::scalar)
                    {
                        return std::unexpected("A SPIR-V vector in a buffer has no scalar components.");
                    }
                    result->kind = LayoutType::Kind::vector;
                    result->scalar = (*component)->scalar;
                    result->scalarSize = (*component)->scalarSize;
                    result->components = type.operands.at(1);
                    return result;
                }
                case opTypeMatrix: {
                    auto column = layoutOf(type.operands.at(0), 0, false, depth + 1);
                    if (!column || (*column)->kind != LayoutType::Kind::vector || matrixStride == 0)
                    {
                        return std::unexpected("A SPIR-V matrix in a buffer has no column type or matrix stride.");
                    }
                    result->kind = LayoutType::Kind::matrix;
                    result->scalar = (*column)->scalar;
                    result->scalarSize = (*column)->scalarSize;
                    result->components = rowMajor ? type.operands.at(1) : (*column)->components;
                    result->vectorCount = rowMajor ? (*column)->components : type.operands.at(1);
                    result->stride = matrixStride;
                    return result;
                }
                case opTypeArray: {
                    const auto length = constants_.find(type.operands.at(1));
                    const auto stride = arrayStrides_.find(typeId);
                    if (length == constants_.end() || stride == arrayStrides_.end())
                    {
                        return std::unexpected("A SPIR-V array in a buffer has no constant length or stride.");
                    }
                    auto element = layoutOf(type.operands.at(0), matrixStride, rowMajor, depth + 1);
                    if (!element)
                    {
                        return std::unexpected(std::move(element.error()));
                    }
                    if ((*element)->kind == LayoutType::Kind::matrix)
                    {
                        return std::unexpected("Arrays of matrices in buffers are not supported.");
                    }
                    result->kind = LayoutType::Kind::array;
                    result->count = length->second;
                    result->stride = stride->second;
                    result->element = std::move(*element);
                    return result;
                }
                case opTypeStruct: {
                    if (const auto wrapped = wrappedArray(typeId, type); wrapped.has_value())
                    {
                        return layoutOf(*wrapped, matrixStride, rowMajor, depth + 1);
                    }
                    result->kind = LayoutType::Kind::structure;
                    result->name = structName(typeId);
                    for (std::uint32_t member = 0; member < type.operands.size(); ++member)
                    {
                        const auto key = std::pair{typeId, member};
                        const auto offset = memberOffsets_.find(key);
                        if (offset == memberOffsets_.end())
                        {
                            return std::unexpected("A SPIR-V structure member in a buffer has no offset.");
                        }
                        const auto stride = memberMatrixStrides_.find(key);
                        auto converted = layoutOf(type.operands[member],
                                                  stride != memberMatrixStrides_.end() ? stride->second : 0,
                                                  rowMajorMembers_.contains(key),
                                                  depth + 1);
                        if (!converted)
                        {
                            return std::unexpected(std::move(converted.error()));
                        }
                        const auto name = memberNames_.find(key);
                        result->fields.push_back(LayoutField{
                            .name = name != memberNames_.end() ? name->second : "member" + std::to_string(member),
                            .offset = offset->second,
                            .type = std::move(*converted),
                        });
                    }
                    return result;
                }
                default:
                    return std::unexpected("A SPIR-V buffer holds an unsupported type.");
                }
            }

            [[nodiscard]] std::optional<std::uint32_t> blockType(bool pushConstants,
                                                                 std::uint32_t set,
                                                                 std::uint32_t binding) const
            {
                for (const auto& [id, variable] : variables_)
                {
                    const bool push = variable.storageClass == storagePushConstant;
                    const bool buffer =
                        variable.storageClass == storageUniform || variable.storageClass == storageStorageBuffer;
                    if (pushConstants ? !push : !buffer)
                    {
                        continue;
                    }
                    if (!pushConstants)
                    {
                        const auto foundSet = descriptorSets_.find(id);
                        const auto foundBinding = bindings_.find(id);
                        if (foundBinding == bindings_.end() || foundBinding->second != binding ||
                            (foundSet != descriptorSets_.end() ? foundSet->second : 0U) != set)
                        {
                            continue;
                        }
                    }
                    const auto pointer = types_.find(variable.pointerType);
                    if (pointer == types_.end() || pointer->second.opcode != opTypePointer)
                    {
                        return std::nullopt;
                    }
                    return pointer->second.operands.at(1);
                }
                return std::nullopt;
            }

            [[nodiscard]] std::optional<std::pair<std::uint32_t, std::uint32_t>> runtimeArrayElement(
                std::uint32_t blockId) const
            {
                const auto block = types_.find(blockId);
                if (block == types_.end() || block->second.opcode != opTypeStruct || block->second.operands.empty())
                {
                    return std::nullopt;
                }
                const auto arrayId = block->second.operands.back();
                const auto array = types_.find(arrayId);
                const auto stride = arrayStrides_.find(arrayId);
                if (array == types_.end() || array->second.opcode != opTypeRuntimeArray ||
                    stride == arrayStrides_.end())
                {
                    return std::nullopt;
                }
                return std::pair{array->second.operands.at(0), stride->second};
            }

            [[nodiscard]] std::expected<Interface, std::string> interfaceOf(std::string_view entryName,
                                                                            std::uint32_t executionModel) const
            {
                const EntryPoint* entry = nullptr;
                for (const auto& candidate : entryPoints_)
                {
                    if (candidate.name == entryName && candidate.executionModel == executionModel)
                    {
                        entry = &candidate;
                        break;
                    }
                }
                if (entry == nullptr)
                {
                    return std::unexpected("The SPIR-V module has no entry point '" + std::string(entryName) +
                                           "' for the shader's stage.");
                }
                const bool listsResources = version_ >= 0x00010400U;
                const auto uses = [entry](std::uint32_t id) {
                    return std::ranges::find(entry->interfaceIds, id) != entry->interfaceIds.end();
                };
                Interface result;
                for (const auto& [id, variable] : variables_)
                {
                    const auto storage = variable.storageClass;
                    if (storage == storageInput)
                    {
                        if (executionModel == executionModelVertex && uses(id) && !builtIns_.contains(id))
                        {
                            if (auto appended = appendVertexInput(id, variable, result.vertexInputLocations); !appended)
                            {
                                return std::unexpected(std::move(appended.error()));
                            }
                        }
                        continue;
                    }
                    const bool push = storage == storagePushConstant;
                    if (!push && storage != storageUniformConstant && storage != storageUniform &&
                        storage != storageStorageBuffer)
                    {
                        continue;
                    }
                    if (listsResources && !uses(id))
                    {
                        continue;
                    }
                    const auto pointee = pointeeType(variable);
                    if (!pointee.has_value())
                    {
                        return std::unexpected("A SPIR-V resource has no pointer type.");
                    }
                    const auto name = resourceName(id, *pointee);
                    if (push)
                    {
                        result.dataResources.push_back(DataResource{
                            .name = name,
                            .typeName = structNameOr(*pointee, name),
                            .usage = DataUsage::pushConstants,
                        });
                        continue;
                    }
                    if (auto appended = appendBinding(id, storage, *pointee, name, result); !appended)
                    {
                        return std::unexpected(std::move(appended.error()));
                    }
                }
                if (executionModel == executionModelGLCompute)
                {
                    auto size = workgroupSize(entry->function);
                    if (!size)
                    {
                        return std::unexpected(std::move(size.error()));
                    }
                    result.workgroupSize = *size;
                }
                std::ranges::sort(result.vertexInputLocations);
                if (std::ranges::adjacent_find(result.vertexInputLocations) != result.vertexInputLocations.end())
                {
                    return std::unexpected("Two vertex inputs share a location.");
                }
                return result;
            }

        private:
            static constexpr std::uint32_t opName = 5;
            static constexpr std::uint32_t opMemberName = 6;
            static constexpr std::uint32_t opEntryPoint = 15;
            static constexpr std::uint32_t opExecutionMode = 16;
            static constexpr std::uint32_t opTypeBool = 20;
            static constexpr std::uint32_t opTypeInt = 21;
            static constexpr std::uint32_t opTypeFloat = 22;
            static constexpr std::uint32_t opTypeVector = 23;
            static constexpr std::uint32_t opTypeMatrix = 24;
            static constexpr std::uint32_t opTypeImage = 25;
            static constexpr std::uint32_t opTypeSampler = 26;
            static constexpr std::uint32_t opTypeSampledImage = 27;
            static constexpr std::uint32_t opTypeArray = 28;
            static constexpr std::uint32_t opTypeRuntimeArray = 29;
            static constexpr std::uint32_t opTypeStruct = 30;
            static constexpr std::uint32_t opTypePointer = 32;
            static constexpr std::uint32_t opConstant = 43;
            static constexpr std::uint32_t opConstantComposite = 44;
            static constexpr std::uint32_t opSpecConstant = 50;
            static constexpr std::uint32_t opSpecConstantComposite = 51;
            static constexpr std::uint32_t opVariable = 59;
            static constexpr std::uint32_t opDecorate = 71;
            static constexpr std::uint32_t opMemberDecorate = 72;
            static constexpr std::uint32_t opExecutionModeId = 331;
            static constexpr std::uint32_t opTypeAccelerationStructure = 5341;
            static constexpr std::uint32_t decorationBlock = 2;
            static constexpr std::uint32_t decorationBufferBlock = 3;
            static constexpr std::uint32_t decorationRowMajor = 4;
            static constexpr std::uint32_t decorationArrayStride = 6;
            static constexpr std::uint32_t decorationMatrixStride = 7;
            static constexpr std::uint32_t decorationBuiltIn = 11;
            static constexpr std::uint32_t decorationNonWritable = 24;
            static constexpr std::uint32_t decorationLocation = 30;
            static constexpr std::uint32_t decorationBinding = 33;
            static constexpr std::uint32_t decorationDescriptorSet = 34;
            static constexpr std::uint32_t decorationOffset = 35;
            static constexpr std::uint32_t builtInWorkgroupSize = 25;
            static constexpr std::uint32_t executionModeLocalSize = 17;
            static constexpr std::uint32_t executionModeLocalSizeId = 38;
            static constexpr std::uint32_t imageDimBuffer = 5;
            static constexpr std::uint32_t imageDimSubpassData = 6;
            static constexpr std::uint32_t storageUniformConstant = 0;
            static constexpr std::uint32_t storageInput = 1;
            static constexpr std::uint32_t storageUniform = 2;
            static constexpr std::uint32_t storagePushConstant = 9;
            static constexpr std::uint32_t storageStorageBuffer = 12;
            static constexpr std::uint32_t unboundedBindingCount = 0xFFFFFFFFU;

            [[nodiscard]] std::optional<std::uint32_t> pointeeType(const Variable& variable) const
            {
                const auto pointer = types_.find(variable.pointerType);
                if (pointer == types_.end() || pointer->second.opcode != opTypePointer ||
                    pointer->second.operands.size() < 2)
                {
                    return std::nullopt;
                }
                return pointer->second.operands[1];
            }

            [[nodiscard]] const Type* typeOf(std::uint32_t id) const
            {
                const auto found = types_.find(id);
                return found != types_.end() ? &found->second : nullptr;
            }

            [[nodiscard]] std::string structNameOr(std::uint32_t typeId, std::string_view fallback) const
            {
                auto name = structName(typeId);
                return name.empty() ? std::string(fallback) : name;
            }

            [[nodiscard]] std::string resourceName(std::uint32_t variableId, std::uint32_t pointee) const
            {
                if (const auto found = names_.find(variableId); found != names_.end() && !found->second.empty())
                {
                    return found->second;
                }
                std::uint32_t element = pointee;
                if (const auto* type = typeOf(pointee);
                    type != nullptr && (type->opcode == opTypeArray || type->opcode == opTypeRuntimeArray))
                {
                    element = type->operands.at(0);
                }
                auto name = structName(element);
                return name.empty() ? "resource" + std::to_string(variableId) : name;
            }

            [[nodiscard]] bool isReadOnlyBlock(std::uint32_t variableId, std::uint32_t blockId) const
            {
                if (nonWritable_.contains(variableId))
                {
                    return true;
                }
                const auto* block = typeOf(blockId);
                const auto members = nonWritableMembers_.find(blockId);
                if (block == nullptr || block->opcode != opTypeStruct || block->operands.empty() ||
                    members == nonWritableMembers_.end())
                {
                    return false;
                }
                for (std::uint32_t member = 0; member < block->operands.size(); ++member)
                {
                    if (!members->second.contains(member))
                    {
                        return false;
                    }
                }
                return true;
            }

            [[nodiscard]] std::expected<void, std::string> appendBinding(std::uint32_t id,
                                                                         std::uint32_t storage,
                                                                         std::uint32_t pointee,
                                                                         const std::string& name,
                                                                         Interface& result) const
            {
                const auto binding = bindings_.find(id);
                if (binding == bindings_.end())
                {
                    return std::unexpected("The resource '" + name + "' has no binding.");
                }
                const auto set = descriptorSets_.find(id);
                std::uint32_t count = 1;
                std::uint32_t elementId = pointee;
                const auto* element = typeOf(pointee);
                if (element != nullptr && element->opcode == opTypeArray)
                {
                    const auto length = constants_.find(element->operands.at(1));
                    if (length == constants_.end())
                    {
                        return std::unexpected("The resource array '" + name + "' has no constant length.");
                    }
                    count = length->second;
                }
                else if (element != nullptr && element->opcode == opTypeRuntimeArray)
                {
                    count = unboundedBindingCount;
                }
                if (element != nullptr && (element->opcode == opTypeArray || element->opcode == opTypeRuntimeArray))
                {
                    elementId = element->operands.at(0);
                    element = typeOf(elementId);
                    if (element != nullptr && (element->opcode == opTypeArray || element->opcode == opTypeRuntimeArray))
                    {
                        return std::unexpected("The resource array '" + name + "' has more than one dimension.");
                    }
                }
                if (element == nullptr)
                {
                    return std::unexpected("The resource '" + name + "' has an unknown type.");
                }
                std::string_view type;
                std::optional<DataUsage> usage;
                std::string typeName = structNameOr(elementId, name);
                switch (element->opcode)
                {
                case opTypeSampler:
                    type = "sampler";
                    break;
                case opTypeSampledImage:
                    return std::unexpected("The resource '" + name +
                                           "' combines a texture and a sampler, which RachisRHI cannot bind. Declare "
                                           "a texture and a sampler and combine them in the shader.");
                case opTypeImage: {
                    const auto dim = element->operands.at(1);
                    if (dim == imageDimBuffer || dim == imageDimSubpassData)
                    {
                        return std::unexpected("The resource '" + name +
                                               "' is a texel buffer or subpass input, which RachisRHI cannot bind.");
                    }
                    type = element->operands.at(5) == 2 ? "storageTexture" : "sampledTexture";
                    break;
                }
                case opTypeAccelerationStructure:
                    type = "accelerationStructure";
                    break;
                case opTypeStruct:
                    if (storage == storageStorageBuffer || bufferBlocks_.contains(elementId))
                    {
                        type = isReadOnlyBlock(id, elementId) ? "readOnlyStorageBuffer" : "storageBuffer";
                        if (element->operands.size() == 1)
                        {
                            if (const auto* array = typeOf(element->operands[0]);
                                array != nullptr && array->opcode == opTypeRuntimeArray)
                            {
                                if (const auto* item = typeOf(array->operands.at(0));
                                    item != nullptr && item->opcode == opTypeStruct)
                                {
                                    usage = DataUsage::structuredBuffer;
                                    typeName = structNameOr(array->operands[0], name);
                                }
                            }
                        }
                    }
                    else if (storage == storageUniform && blocks_.contains(elementId))
                    {
                        type = "uniformBuffer";
                        usage = DataUsage::constantBuffer;
                    }
                    break;
                default:
                    break;
                }
                if (type.empty())
                {
                    return std::unexpected("The resource '" + name + "' has a type RachisRHI cannot bind.");
                }
                const InterfaceBinding reflected{
                    .name = name,
                    .set = set != descriptorSets_.end() ? set->second : 0U,
                    .binding = binding->second,
                    .type = type,
                    .count = count,
                };
                if (usage.has_value() && count == 1)
                {
                    result.dataResources.push_back(DataResource{
                        .name = name,
                        .typeName = std::move(typeName),
                        .usage = *usage,
                        .set = reflected.set,
                        .binding = reflected.binding,
                    });
                }
                result.bindings.push_back(reflected);
                return {};
            }

            [[nodiscard]] std::expected<std::uint32_t, std::string> locationCount(std::uint32_t typeId,
                                                                                  std::size_t depth) const
            {
                const auto* type = depth <= 8 ? typeOf(typeId) : nullptr;
                if (type == nullptr)
                {
                    return std::unexpected("A vertex input has an unknown type.");
                }
                switch (type->opcode)
                {
                case opTypeBool:
                case opTypeInt:
                case opTypeFloat:
                    return 1U;
                case opTypeVector: {
                    const auto* component = typeOf(type->operands.at(0));
                    const bool wide =
                        component != nullptr && !component->operands.empty() && component->operands[0] == 64;
                    return wide && type->operands.at(1) > 2 ? 2U : 1U;
                }
                case opTypeMatrix:
                case opTypeArray: {
                    std::uint32_t elements = type->operands.at(1);
                    if (type->opcode == opTypeArray)
                    {
                        const auto length = constants_.find(elements);
                        if (length == constants_.end())
                        {
                            return std::unexpected("A vertex input array has no constant length.");
                        }
                        elements = length->second;
                    }
                    auto each = locationCount(type->operands.at(0), depth + 1);
                    if (!each)
                    {
                        return each;
                    }
                    return elements * *each;
                }
                default:
                    return std::unexpected("A vertex input is a structure or another type that takes no locations.");
                }
            }

            [[nodiscard]] std::expected<void, std::string> appendVertexInput(
                std::uint32_t id, const Variable& variable, std::vector<std::uint32_t>& locations) const
            {
                const auto location = locations_.find(id);
                const auto pointee = pointeeType(variable);
                if (location == locations_.end() || !pointee.has_value())
                {
                    return std::unexpected("A vertex input has no location.");
                }
                auto count = locationCount(*pointee, 0);
                if (!count)
                {
                    return std::unexpected(std::move(count.error()));
                }
                for (std::uint32_t offset = 0; offset < *count; ++offset)
                {
                    locations.push_back(location->second + offset);
                }
                return {};
            }

            [[nodiscard]] std::optional<std::uint32_t> constantValue(std::uint32_t id) const
            {
                const auto found = constants_.find(id);
                return found != constants_.end() ? std::optional{found->second} : std::nullopt;
            }

            [[nodiscard]] std::expected<std::array<std::uint32_t, 3>, std::string> workgroupSize(
                std::uint32_t function) const
            {
                std::optional<std::array<std::uint32_t, 3>> size;
                for (const auto& [id, builtIn] : builtIns_)
                {
                    const auto composite = composites_.find(id);
                    if (builtIn != builtInWorkgroupSize || composite == composites_.end() ||
                        composite->second.size() != 3)
                    {
                        continue;
                    }
                    std::array<std::uint32_t, 3> components{};
                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        const auto value = constantValue(composite->second[axis]);
                        if (!value.has_value())
                        {
                            return std::unexpected("The SPIR-V WorkgroupSize built-in is not constant.");
                        }
                        components[axis] = *value;
                    }
                    size = components;
                }
                if (!size.has_value())
                {
                    if (const auto literal = localSizes_.find(function); literal != localSizes_.end())
                    {
                        size = literal->second;
                    }
                    else if (const auto ids = localSizeIds_.find(function); ids != localSizeIds_.end())
                    {
                        std::array<std::uint32_t, 3> components{};
                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            const auto value = constantValue(ids->second[axis]);
                            if (!value.has_value())
                            {
                                return std::unexpected("The SPIR-V LocalSizeId operands are not constant.");
                            }
                            components[axis] = *value;
                        }
                        size = components;
                    }
                }
                if (!size.has_value())
                {
                    return std::unexpected("The compute entry point has no workgroup size.");
                }
                if (std::ranges::any_of(*size, [](std::uint32_t value) { return value == 0; }))
                {
                    return std::unexpected("The compute entry point's workgroup size is not a positive integer.");
                }
                return *size;
            }

            [[nodiscard]] static std::string literalString(std::span<const std::uint32_t> words)
            {
                std::string text;
                for (const auto word : words)
                {
                    for (std::uint32_t byte = 0; byte < 4; ++byte)
                    {
                        const auto ch = static_cast<char>((word >> (byte * 8U)) & 0xFFU);
                        if (ch == '\0')
                        {
                            return text;
                        }
                        text.push_back(ch);
                    }
                }
                return text;
            }

            void instruction(std::uint32_t opcode, std::span<const std::uint32_t> operands)
            {
                switch (opcode)
                {
                case opName:
                    if (!operands.empty())
                    {
                        names_[operands[0]] = literalString(operands.subspan(1));
                    }
                    break;
                case opMemberName:
                    if (operands.size() >= 2)
                    {
                        memberNames_[{operands[0], operands[1]}] = literalString(operands.subspan(2));
                    }
                    break;
                case opEntryPoint:
                    if (operands.size() >= 3)
                    {
                        EntryPoint entry{
                            .executionModel = operands[0],
                            .function = operands[1],
                            .name = literalString(operands.subspan(2)),
                        };
                        const auto nameWords = entry.name.size() / 4 + 1;
                        for (auto index = 2 + nameWords; index < operands.size(); ++index)
                        {
                            entry.interfaceIds.push_back(operands[index]);
                        }
                        entryPoints_.push_back(std::move(entry));
                    }
                    break;
                case opExecutionMode:
                    if (operands.size() >= 5 && operands[1] == executionModeLocalSize)
                    {
                        localSizes_[operands[0]] = {operands[2], operands[3], operands[4]};
                    }
                    break;
                case opExecutionModeId:
                    if (operands.size() >= 5 && operands[1] == executionModeLocalSizeId)
                    {
                        localSizeIds_[operands[0]] = {operands[2], operands[3], operands[4]};
                    }
                    break;
                case opTypeBool:
                case opTypeInt:
                case opTypeFloat:
                case opTypeVector:
                case opTypeMatrix:
                case opTypeImage:
                case opTypeSampler:
                case opTypeSampledImage:
                case opTypeArray:
                case opTypeRuntimeArray:
                case opTypeStruct:
                case opTypePointer:
                case opTypeAccelerationStructure:
                    if (!operands.empty())
                    {
                        types_[operands[0]] = Type{opcode, {operands.begin() + 1, operands.end()}};
                    }
                    break;
                case opConstant:
                case opSpecConstant:
                    if (operands.size() >= 3)
                    {
                        constants_[operands[1]] = operands[2];
                    }
                    break;
                case opConstantComposite:
                case opSpecConstantComposite:
                    if (operands.size() >= 2)
                    {
                        composites_[operands[1]] = {operands.begin() + 2, operands.end()};
                    }
                    break;
                case opVariable:
                    if (operands.size() >= 3)
                    {
                        variables_.emplace_back(operands[1], Variable{operands[0], operands[2]});
                    }
                    break;
                case opDecorate:
                    if (operands.size() >= 3 && operands[1] == decorationArrayStride)
                    {
                        arrayStrides_[operands[0]] = operands[2];
                    }
                    else if (operands.size() >= 3 && operands[1] == decorationBinding)
                    {
                        bindings_[operands[0]] = operands[2];
                    }
                    else if (operands.size() >= 3 && operands[1] == decorationDescriptorSet)
                    {
                        descriptorSets_[operands[0]] = operands[2];
                    }
                    else if (operands.size() >= 3 && operands[1] == decorationLocation)
                    {
                        locations_[operands[0]] = operands[2];
                    }
                    else if (operands.size() >= 3 && operands[1] == decorationBuiltIn)
                    {
                        builtIns_[operands[0]] = operands[2];
                    }
                    else if (operands.size() >= 2 && operands[1] == decorationNonWritable)
                    {
                        nonWritable_.insert(operands[0]);
                    }
                    else if (operands.size() >= 2 && operands[1] == decorationBlock)
                    {
                        blocks_.insert(operands[0]);
                    }
                    else if (operands.size() >= 2 && operands[1] == decorationBufferBlock)
                    {
                        bufferBlocks_.insert(operands[0]);
                    }
                    break;
                case opMemberDecorate:
                    if (operands.size() >= 3)
                    {
                        const auto key = std::pair{operands[0], operands[1]};
                        if (operands[2] == decorationOffset && operands.size() >= 4)
                        {
                            memberOffsets_[key] = operands[3];
                        }
                        else if (operands[2] == decorationMatrixStride && operands.size() >= 4)
                        {
                            memberMatrixStrides_[key] = operands[3];
                        }
                        else if (operands[2] == decorationRowMajor)
                        {
                            rowMajorMembers_[key] = true;
                        }
                        else if (operands[2] == decorationNonWritable)
                        {
                            nonWritableMembers_[operands[0]].insert(operands[1]);
                        }
                    }
                    break;
                default:
                    break;
                }
            }

            [[nodiscard]] std::optional<ScalarInfo> scalarInfo(const Type& type) const
            {
                if (type.opcode == opTypeBool)
                {
                    return ScalarInfo{"std::uint32_t", 4};
                }
                const auto width = type.operands.empty() ? 0U : type.operands[0];
                if (type.opcode == opTypeFloat)
                {
                    return width == 32   ? std::optional{ScalarInfo{"float", 4}}
                           : width == 64 ? std::optional{ScalarInfo{"double", 8}}
                           : width == 16 ? std::optional{ScalarInfo{"std::uint16_t", 2}}
                                         : std::nullopt;
                }
                const bool isSigned = type.operands.size() >= 2 && type.operands[1] != 0;
                switch (width)
                {
                case 16:
                    return ScalarInfo{isSigned ? "std::int16_t" : "std::uint16_t", 2};
                case 32:
                    return ScalarInfo{isSigned ? "std::int32_t" : "std::uint32_t", 4};
                case 64:
                    return ScalarInfo{isSigned ? "std::int64_t" : "std::uint64_t", 8};
                default:
                    return std::nullopt;
                }
            }

            [[nodiscard]] std::string structName(std::uint32_t typeId) const
            {
                const auto found = names_.find(typeId);
                if (found == names_.end())
                {
                    return {};
                }
                std::string_view name = found->second;
                for (const std::string_view suffix : {"_std140", "_std430"})
                {
                    if (name.size() > suffix.size() && name.ends_with(suffix))
                    {
                        name.remove_suffix(suffix.size());
                        break;
                    }
                }
                return std::string(name);
            }

            [[nodiscard]] std::optional<std::uint32_t> wrappedArray(std::uint32_t typeId, const Type& type) const
            {
                const auto name = names_.find(typeId);
                if (name == names_.end() || !name->second.starts_with("_Array_") || type.operands.size() != 1)
                {
                    return std::nullopt;
                }
                const auto member = memberNames_.find({typeId, 0});
                const auto offset = memberOffsets_.find({typeId, 0});
                const auto* array = typeOf(type.operands[0]);
                if (member == memberNames_.end() || member->second != "data" || offset == memberOffsets_.end() ||
                    offset->second != 0 || array == nullptr || array->opcode != opTypeArray)
                {
                    return std::nullopt;
                }
                return type.operands[0];
            }

            std::unordered_map<std::uint32_t, Type> types_;
            std::unordered_map<std::uint32_t, std::string> names_;
            std::map<std::pair<std::uint32_t, std::uint32_t>, std::string> memberNames_;
            std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> memberOffsets_;
            std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> memberMatrixStrides_;
            std::map<std::pair<std::uint32_t, std::uint32_t>, bool> rowMajorMembers_;
            std::unordered_map<std::uint32_t, std::uint32_t> arrayStrides_;
            std::unordered_map<std::uint32_t, std::uint32_t> constants_;
            std::unordered_map<std::uint32_t, std::uint32_t> bindings_;
            std::unordered_map<std::uint32_t, std::uint32_t> descriptorSets_;
            std::vector<std::pair<std::uint32_t, Variable>> variables_;
            std::uint32_t version_ = 0;
            std::vector<EntryPoint> entryPoints_;
            std::unordered_map<std::uint32_t, std::array<std::uint32_t, 3>> localSizes_;
            std::unordered_map<std::uint32_t, std::array<std::uint32_t, 3>> localSizeIds_;
            std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> composites_;
            std::unordered_map<std::uint32_t, std::uint32_t> locations_;
            std::map<std::uint32_t, std::uint32_t> builtIns_;
            std::set<std::uint32_t> nonWritable_;
            std::map<std::uint32_t, std::set<std::uint32_t>> nonWritableMembers_;
            std::set<std::uint32_t> blocks_;
            std::set<std::uint32_t> bufferBlocks_;
        };

    } // namespace detail

    [[nodiscard]] inline std::expected<std::vector<DataBlock>, std::string> spirvDataBlocks(
        std::span<const std::byte> spirv, std::span<const DataResource> resources)
    {
        auto module = detail::SpirvLayoutModule::parse(spirv);
        if (!module)
        {
            return std::unexpected(std::move(module.error()));
        }
        std::vector<DataBlock> blocks;
        for (const auto& resource : resources)
        {
            const auto blockId =
                module->blockType(resource.usage == DataUsage::pushConstants, resource.set, resource.binding);
            if (!blockId.has_value())
            {
                return std::unexpected("The SPIR-V module has no block for the resource '" + resource.name + "'.");
            }
            std::uint32_t typeId = *blockId;
            std::uint32_t stride = 0;
            if (resource.usage == DataUsage::structuredBuffer)
            {
                const auto element = module->runtimeArrayElement(*blockId);
                if (!element.has_value())
                {
                    return std::unexpected("The SPIR-V block of '" + resource.name + "' is not a structured buffer.");
                }
                typeId = element->first;
                stride = element->second;
            }
            auto layout = module->layoutOf(typeId, 0, false, 0);
            if (!layout)
            {
                return std::unexpected("The resource '" + resource.name + "': " + layout.error());
            }
            if ((*layout)->kind != LayoutType::Kind::structure)
            {
                return std::unexpected("The resource '" + resource.name + "' does not hold a structure.");
            }
            auto named = std::make_shared<LayoutType>(**layout);
            named->name = resource.typeName;
            const auto size = resource.usage == DataUsage::structuredBuffer ? stride : detail::structureEnd(*named);
            blocks.push_back(DataBlock{.resource = resource, .type = std::move(named), .size = size});
        }
        return blocks;
    }

    namespace detail
    {

        [[nodiscard]] inline bool isCppKeyword(std::string_view name) noexcept
        {
            static constexpr std::array keywords{
                "alignas", "alignof", "and",       "asm",     "auto",     "bool",      "break",    "case",
                "catch",   "char",    "class",     "concept", "const",    "constexpr", "continue", "default",
                "delete",  "do",      "double",    "else",    "enum",     "explicit",  "export",   "extern",
                "false",   "float",   "for",       "friend",  "goto",     "if",        "inline",   "int",
                "long",    "mutable", "namespace", "new",     "noexcept", "not",       "nullptr",  "operator",
                "or",      "private", "protected", "public",  "register", "requires",  "return",   "short",
                "signed",  "sizeof",  "static",    "struct",  "switch",   "template",  "this",     "throw",
                "true",    "try",     "typedef",   "typeid",  "typename", "union",     "unsigned", "using",
                "virtual", "void",    "volatile",  "while",   "xor",
            };
            return std::ranges::find(keywords, name) != keywords.end();
        }

        [[nodiscard]] inline std::string cppIdentifier(std::string_view name)
        {
            std::string identifier(name);
            if (identifier.empty() || !isValidCppIdentifier(identifier) || isCppKeyword(identifier) ||
                identifier.starts_with("rachisPadding"))
            {
                identifier = "field_" + identifier;
                std::ranges::replace_if(identifier, [](char ch) { return !isAsciiIdentifierContinue(ch); }, '_');
            }
            return identifier;
        }

        [[nodiscard]] inline bool layoutsMatch(const LayoutType& left, const LayoutType& right)
        {
            if (left.kind != right.kind || left.scalar != right.scalar || left.components != right.components ||
                left.vectorCount != right.vectorCount || left.stride != right.stride || left.count != right.count ||
                left.name != right.name || left.fields.size() != right.fields.size())
            {
                return false;
            }
            if ((left.element == nullptr) != (right.element == nullptr) ||
                (left.element != nullptr && !layoutsMatch(*left.element, *right.element)))
            {
                return false;
            }
            for (std::size_t index = 0; index < left.fields.size(); ++index)
            {
                if (left.fields[index].name != right.fields[index].name ||
                    left.fields[index].offset != right.fields[index].offset ||
                    !layoutsMatch(*left.fields[index].type, *right.fields[index].type))
                {
                    return false;
                }
            }
            return true;
        }

        class DataStructEmitter final
        {
        public:
            explicit DataStructEmitter(std::string& text) : text_(text)
            {
            }

            [[nodiscard]] std::expected<void, std::string> emitBlock(const DataBlock& block, std::string_view name)
            {
                nested_.clear();
                nestedOrder_.clear();
                pendingAsserts_.clear();
                topName_ = std::string(name);
                std::string body;
                std::vector<std::string> asserts;
                const auto target = block.resource.usage == DataUsage::structuredBuffer
                                        ? block.size
                                        : roundUp(block.size, cppAlignment(*block.type));
                auto members = emitMembers(*block.type, topName_, target, "        ", body, asserts);
                if (!members)
                {
                    return members;
                }
                text_ += "        // ";
                text_ += usageText(block.resource.usage);
                text_ += " '" + block.resource.name + "'.\n";
                text_ += "        struct " + topName_ + " final\n        {\n";
                for (const auto& declaration : nestedOrder_)
                {
                    text_ += declaration;
                }
                text_ += body;
                text_ += "        };\n";
                for (const auto& line : pendingAsserts_)
                {
                    text_ += "        " + line + "\n";
                }
                for (const auto& line : asserts)
                {
                    text_ += "        " + line + "\n";
                }
                text_ += "\n";
                return {};
            }

        private:
            [[nodiscard]] static std::string_view usageText(DataUsage usage) noexcept
            {
                switch (usage)
                {
                case DataUsage::constantBuffer:
                    return "The constant buffer";
                case DataUsage::pushConstants:
                    return "The push constants";
                case DataUsage::structuredBuffer:
                    return "One element of the structured buffer";
                }
                return "The buffer";
            }

            [[nodiscard]] std::expected<std::string, std::string> cppType(const LayoutType& type)
            {
                switch (type.kind)
                {
                case LayoutType::Kind::scalar:
                    return std::string(type.scalar);
                case LayoutType::Kind::vector:
                    return "std::array<" + std::string(type.scalar) + ", " + std::to_string(type.components) + ">";
                case LayoutType::Kind::matrix: {
                    const auto vector =
                        "std::array<" + std::string(type.scalar) + ", " + std::to_string(type.components) + ">";
                    return arrayOf(vector, type.scalarSize * type.components, type.vectorCount, type.stride);
                }
                case LayoutType::Kind::array: {
                    auto element = cppType(*type.element);
                    if (!element)
                    {
                        return element;
                    }
                    return arrayOf(*element, cppSize(*type.element), type.count, type.stride);
                }
                case LayoutType::Kind::structure:
                    return declareNested(type);
                }
                return std::unexpected("A buffer member has an unsupported type.");
            }

            [[nodiscard]] static std::string arrayOf(const std::string& element,
                                                     std::uint32_t elementSize,
                                                     std::uint32_t count,
                                                     std::uint32_t stride)
            {
                if (count <= 1 || stride <= elementSize)
                {
                    return "std::array<" + element + ", " + std::to_string(count) + ">";
                }
                return "rachis_shaders::layout::StridedArray<" + element + ", " + std::to_string(count) + ", " +
                       std::to_string(stride) + ">";
            }

            [[nodiscard]] std::expected<std::string, std::string> declareNested(const LayoutType& type)
            {
                const auto base = cppIdentifier(type.name.empty() ? std::string("Nested") : type.name);
                std::string body;
                std::vector<std::string> asserts;
                std::string name = base;
                for (std::uint32_t suffix = 1;; ++suffix)
                {
                    const auto found = nested_.find(name);
                    if (found == nested_.end())
                    {
                        break;
                    }
                    if (found->second == &type || layoutsMatch(*found->second, type))
                    {
                        return topName_ + "::" + name;
                    }
                    name = base + "_" + std::to_string(suffix);
                }
                nested_.emplace(name, &type);
                const auto qualified = topName_ + "::" + name;
                auto members = emitMembers(type, qualified, 0, "            ", body, asserts);
                if (!members)
                {
                    return std::unexpected(std::move(members.error()));
                }
                std::string declaration =
                    "            struct " + name + " final\n            {\n" + body + "            };\n";
                nestedOrder_.push_back(std::move(declaration));
                pendingAsserts_.insert(pendingAsserts_.end(), asserts.begin(), asserts.end());
                return qualified;
            }

            [[nodiscard]] std::expected<void, std::string> emitMembers(const LayoutType& type,
                                                                       const std::string& qualified,
                                                                       std::uint32_t targetSize,
                                                                       std::string_view indent,
                                                                       std::string& body,
                                                                       std::vector<std::string>& asserts)
            {
                auto fields = type.fields;
                std::ranges::stable_sort(fields, {}, &LayoutField::offset);
                std::uint32_t cursor = 0;
                std::uint32_t padding = 0;
                for (const auto& field : fields)
                {
                    const auto name = cppIdentifier(field.name);
                    if (field.offset < cursor)
                    {
                        return std::unexpected("The member '" + field.name + "' of '" + qualified +
                                               "' overlaps the member before it.");
                    }
                    if (field.offset > cursor)
                    {
                        body += std::string(indent) + "    std::byte rachisPadding" + std::to_string(padding++) + "[" +
                                std::to_string(field.offset - cursor) + "];\n";
                    }
                    auto spelled = cppType(*field.type);
                    if (!spelled)
                    {
                        return std::unexpected(std::move(spelled.error()));
                    }
                    body += std::string(indent) + "    " + *spelled + " " + name + ";\n";
                    asserts.push_back("static_assert(offsetof(" + qualified + ", " + name +
                                      ") == " + std::to_string(field.offset) + ");");
                    cursor = field.offset + cppSize(*field.type);
                }
                const auto alignment = cppAlignment(type);
                if (targetSize != 0)
                {
                    if (targetSize < cursor || targetSize % alignment != 0)
                    {
                        return std::unexpected("The stride of '" + qualified + "' does not fit its members.");
                    }
                    if (roundUp(cursor, alignment) < targetSize)
                    {
                        body += std::string(indent) + "    std::byte rachisPadding" + std::to_string(padding++) + "[" +
                                std::to_string(targetSize - cursor) + "];\n";
                        cursor = targetSize;
                    }
                }
                asserts.push_back("static_assert(sizeof(" + qualified +
                                  ") == " + std::to_string(roundUp(cursor, alignment)) + ");");
                asserts.push_back("static_assert(alignof(" + qualified + ") == " + std::to_string(alignment) + ");");
                return {};
            }

            std::string& text_;
            std::string topName_;
            std::map<std::string, const LayoutType*> nested_;
            std::vector<std::string> nestedOrder_;
            std::vector<std::string> pendingAsserts_;
        };

    } // namespace detail

    inline constexpr std::string_view stridedArraySupport = R"(#ifndef RACHIS_SHADERS_LAYOUT_STRIDED_ARRAY
#define RACHIS_SHADERS_LAYOUT_STRIDED_ARRAY
namespace rachis_shaders::layout
{
    // N values of T, Stride bytes apart, as a shader lays out an array, or the vectors of a matrix. The last value
    // has no padding after it, so the next member can follow it as closely as the shader packs it.
    template<typename T, std::size_t N, std::size_t Stride>
    struct StridedArray final
    {
        static_assert(N > 1 && Stride > sizeof(T) && Stride % alignof(T) == 0);

        struct Element final
        {
            T value;
            std::byte padding[Stride - sizeof(T)];
        };

        std::array<Element, N - 1> leading;
        T last;

        [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept
        {
            return index + 1 < N ? leading[index].value : last;
        }
        [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept
        {
            return index + 1 < N ? leading[index].value : last;
        }
        [[nodiscard]] static constexpr std::size_t size() noexcept
        {
            return N;
        }
    };
} // namespace rachis_shaders::layout
#endif
)";

} // namespace rachis::shader_tool
