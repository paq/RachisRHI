#pragma once

#include "Internal/ShaderBinaryTool.h"
#include "Internal/ShaderDataLayout.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rachis::shader_tool
{

    enum class InterfaceBackend : std::uint8_t
    {
        d3d12,
        vulkan,
        metal,
    };

    enum class InterfaceStage : std::uint8_t
    {
        vertex,
        pixel,
        compute,
        task,
        mesh,
    };

    inline constexpr std::uint32_t d3d12PushConstantRegister = 0;
    inline constexpr std::uint32_t d3d12PushConstantSpace = 4;
    inline constexpr std::uint32_t drawIndexRegister = 0;
    inline constexpr std::uint32_t drawIndexSpace = 5;
    inline constexpr std::uint32_t bindingSetLimit = 4;
    inline constexpr std::uint32_t unboundedResourceCount = 0xFFFFFFFFU;

    struct ReflectedBinding final
    {
        std::string name;
        std::uint32_t set = 0;
        std::uint32_t binding = 0;
        std::string_view type;
        std::uint32_t count = 1;

        [[nodiscard]] bool operator==(const ReflectedBinding&) const = default;
    };

    struct ReflectedInterface final
    {
        std::vector<ReflectedBinding> bindings;
        std::uint32_t pushConstantBytes = 0;
        std::vector<std::uint32_t> vertexInputLocations;
        std::optional<std::array<std::uint32_t, 3>> workgroupSize;
        std::vector<DataResource> dataResources;
        std::vector<DataBlock> blocks;
        bool drawIndex = false;
    };

    struct BackendInterface final
    {
        InterfaceBackend backend = InterfaceBackend::vulkan;
        ReflectedInterface reflected;
    };

    [[nodiscard]] constexpr std::optional<InterfaceBackend> parseInterfaceBackend(std::string_view name) noexcept
    {
        if (name == "d3d12")
        {
            return InterfaceBackend::d3d12;
        }
        if (name == "vulkan")
        {
            return InterfaceBackend::vulkan;
        }
        if (name == "metal")
        {
            return InterfaceBackend::metal;
        }
        return std::nullopt;
    }

    [[nodiscard]] constexpr std::optional<InterfaceStage> parseInterfaceStage(std::string_view name) noexcept
    {
        if (name == "vertex")
        {
            return InterfaceStage::vertex;
        }
        if (name == "pixel")
        {
            return InterfaceStage::pixel;
        }
        if (name == "compute")
        {
            return InterfaceStage::compute;
        }
        if (name == "task")
        {
            return InterfaceStage::task;
        }
        if (name == "mesh")
        {
            return InterfaceStage::mesh;
        }
        return std::nullopt;
    }

    [[nodiscard]] constexpr bool interfaceStageHasWorkgroup(InterfaceStage stage) noexcept
    {
        return stage == InterfaceStage::compute || stage == InterfaceStage::task || stage == InterfaceStage::mesh;
    }

    [[nodiscard]] constexpr bool interfaceStageReadsDrawIndex(InterfaceStage stage) noexcept
    {
        return stage == InterfaceStage::vertex || stage == InterfaceStage::task || stage == InterfaceStage::mesh;
    }

    [[nodiscard]] constexpr std::string_view interfaceStageName(InterfaceStage stage) noexcept
    {
        switch (stage)
        {
        case InterfaceStage::vertex:
            return "vertex";
        case InterfaceStage::pixel:
            return "pixel";
        case InterfaceStage::compute:
            return "compute";
        case InterfaceStage::task:
            return "task";
        case InterfaceStage::mesh:
            return "mesh";
        }
        return "compute";
    }

    [[nodiscard]] constexpr std::string_view interfaceBackendName(InterfaceBackend backend) noexcept
    {
        switch (backend)
        {
        case InterfaceBackend::d3d12:
            return "d3d12";
        case InterfaceBackend::vulkan:
            return "vulkan";
        case InterfaceBackend::metal:
            return "metal";
        }
        return "vulkan";
    }

    namespace detail
    {

        [[nodiscard]] inline std::string_view jsonString(const JsonValue* value) noexcept
        {
            const auto* text = value != nullptr ? value->asString() : nullptr;
            return text != nullptr ? std::string_view(*text) : std::string_view{};
        }

        [[nodiscard]] inline std::optional<std::uint32_t> jsonUint32(const JsonValue& object, std::string_view key)
        {
            const auto* value = object.find(key);
            return value != nullptr ? value->asUint32() : std::nullopt;
        }

        [[nodiscard]] inline std::vector<const JsonValue*> bindingRecords(const JsonValue& parameter)
        {
            std::vector<const JsonValue*> records;
            if (const auto* single = parameter.find("binding"); single != nullptr && single->asObject() != nullptr)
            {
                records.push_back(single);
            }
            if (const auto* list = parameter.find("bindings"); list != nullptr && list->asArray() != nullptr)
            {
                for (const auto& item : *list->asArray())
                {
                    records.push_back(&item);
                }
            }
            return records;
        }

        [[nodiscard]] inline const JsonValue* selectEntryPoint(const JsonValue& root, std::string_view entryPoint)
        {
            const auto* entries = root.find("entryPoints");
            if (entries == nullptr || entries->asArray() == nullptr)
            {
                return nullptr;
            }
            for (const auto& entry : *entries->asArray())
            {
                if (jsonString(entry.find("name")) == entryPoint)
                {
                    return &entry;
                }
            }
            return nullptr;
        }

        [[nodiscard]] inline bool isUsedRecord(const JsonValue& record)
        {
            const auto* used = record.find("used");
            if (used == nullptr)
            {
                return false;
            }
            if (const auto* flag = used->asBool(); flag != nullptr)
            {
                return *flag;
            }
            const auto value = used->asUint32();
            return value.has_value() && *value != 0;
        }

        [[nodiscard]] inline std::optional<std::vector<std::string>> usedParameterNames(const JsonValue& entry)
        {
            const auto* bindings = entry.find("bindings");
            if (bindings == nullptr || bindings->asArray() == nullptr)
            {
                return std::nullopt;
            }
            std::vector<std::string> used;
            for (const auto& item : *bindings->asArray())
            {
                const auto records = bindingRecords(item);
                const bool isUsed =
                    std::ranges::any_of(records, [](const JsonValue* record) { return isUsedRecord(*record); });
                if (isUsed)
                {
                    used.emplace_back(jsonString(item.find("name")));
                }
            }
            return used;
        }

        [[nodiscard]] inline bool isPushConstantParameter(const JsonValue& parameter, InterfaceBackend backend)
        {
            if (backend == InterfaceBackend::d3d12)
            {
                return std::ranges::any_of(bindingRecords(parameter), [](const JsonValue* record) {
                    return jsonString(record->find("kind")) == "constantBuffer" &&
                           jsonUint32(*record, "space").value_or(0) == d3d12PushConstantSpace &&
                           jsonUint32(*record, "index").value_or(0) == d3d12PushConstantRegister;
                });
            }
            if (std::ranges::any_of(bindingRecords(parameter), [](const JsonValue* record) {
                    return jsonString(record->find("kind")) == "pushConstantBuffer";
                }))
            {
                return true;
            }
            const auto* type = parameter.find("type");
            const auto* container = type != nullptr ? type->find("containerVarLayout") : nullptr;
            const auto* binding = container != nullptr ? container->find("binding") : nullptr;
            return binding != nullptr && jsonString(binding->find("kind")) == "pushConstantBuffer";
        }

        [[nodiscard]] inline std::expected<std::uint32_t, std::string> pushConstantBlockBytes(
            const JsonValue& parameter)
        {
            const auto* type = parameter.find("type");
            const auto* element = type != nullptr ? type->find("elementVarLayout") : nullptr;
            const auto* binding = element != nullptr ? element->find("binding") : nullptr;
            const auto size = binding != nullptr ? jsonUint32(*binding, "size") : std::nullopt;
            if (!size.has_value() || *size == 0)
            {
                return std::unexpected("The push-constant block '" + std::string(jsonString(parameter.find("name"))) +
                                       "' has no reflected size.");
            }
            return *size;
        }

        struct RegisterSlot final
        {
            std::uint32_t set = 0;
            std::uint32_t binding = 0;
        };

        [[nodiscard]] inline std::optional<RegisterSlot> resourceSlot(const JsonValue& parameter,
                                                                      InterfaceBackend backend)
        {
            const auto records = bindingRecords(parameter);
            if (backend != InterfaceBackend::d3d12)
            {
                for (const auto* record : records)
                {
                    if (jsonString(record->find("kind")) == "descriptorTableSlot")
                    {
                        return RegisterSlot{jsonUint32(*record, "space").value_or(0),
                                            jsonUint32(*record, "index").value_or(0)};
                    }
                }
            }
            for (const auto* record : records)
            {
                const auto kind = jsonString(record->find("kind"));
                if (kind == "constantBuffer" || kind == "shaderResource" || kind == "unorderedAccess" ||
                    kind == "samplerState")
                {
                    return RegisterSlot{jsonUint32(*record, "space").value_or(0),
                                        jsonUint32(*record, "index").value_or(0)};
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] inline std::expected<bool, std::string> classifyDrawIndexSpace(
            const JsonValue& parameter, InterfaceBackend backend, const std::optional<RegisterSlot>& slot)
        {
            if (!slot.has_value() || slot->set != drawIndexSpace)
            {
                return false;
            }
            const auto* type = parameter.find("type");
            if (backend != InterfaceBackend::vulkan && slot->binding == drawIndexRegister && type != nullptr &&
                isDrawIndexPayload(*type))
            {
                return true;
            }
            return std::unexpected(
                "The resource '" + std::string(jsonString(parameter.find("name"))) +
                "' uses register space 5, which is reserved for the draw index. Only the constant buffer of "
                "Rachis/DrawIndex.hlsli, one uint at register(b0, space5), may use it, and only on Direct3D 12 and "
                "Metal.");
        }

        [[nodiscard]] inline bool isTextureShape(std::string_view shape) noexcept
        {
            return shape == "texture1D" || shape == "texture2D" || shape == "texture3D" || shape == "textureCube";
        }

        struct ResourceShape final
        {
            std::string_view type;
            std::uint32_t count = 1;
        };

        [[nodiscard]] inline std::expected<ResourceShape, std::string> classifyResource(const JsonValue& parameter)
        {
            const std::string name(jsonString(parameter.find("name")));
            const auto* type = parameter.find("type");
            std::uint32_t count = 1;
            for (std::size_t depth = 0; type != nullptr && jsonString(type->find("kind")) == "array"; ++depth)
            {
                if (depth > 0)
                {
                    return std::unexpected("The resource array '" + name + "' has more than one dimension.");
                }
                const auto elements = jsonUint32(*type, "elementCount").value_or(0);
                count = elements == 0 ? unboundedResourceCount : elements;
                type = type->find("elementType");
            }
            if (type == nullptr)
            {
                return std::unexpected("The parameter '" + name + "' has no reflected type.");
            }
            const auto kind = jsonString(type->find("kind"));
            if (kind == "constantBuffer")
            {
                return ResourceShape{"uniformBuffer", count};
            }
            if (kind == "samplerState")
            {
                return ResourceShape{"sampler", count};
            }
            if (kind == "resource")
            {
                const auto shape = jsonString(type->find("baseShape"));
                const auto access = jsonString(type->find("access"));
                const bool writable = !access.empty() && access != "read";
                if (shape == "structuredBuffer" || shape == "byteAddressBuffer")
                {
                    return ResourceShape{writable ? "storageBuffer" : "readOnlyStorageBuffer", count};
                }
                if (isTextureShape(shape))
                {
                    return ResourceShape{writable ? "storageTexture" : "sampledTexture", count};
                }
                if (shape == "accelerationStructure")
                {
                    return ResourceShape{"accelerationStructure", count};
                }
                return std::unexpected("The resource '" + name + "' has the shape '" + std::string(shape) +
                                       "', which RachisRHI cannot bind.");
            }
            return std::unexpected("The parameter '" + name + "' has the kind '" + std::string(kind) +
                                   "', which RachisRHI cannot bind. Put loose uniforms in a constant buffer.");
        }

        [[nodiscard]] inline std::optional<DataUsage> dataUsage(const JsonValue& parameter, std::string_view type)
        {
            if (type == "uniformBuffer")
            {
                return DataUsage::constantBuffer;
            }
            if (type != "readOnlyStorageBuffer" && type != "storageBuffer")
            {
                return std::nullopt;
            }
            const auto* reflected = parameter.find("type");
            const auto* result = reflected != nullptr ? reflected->find("resultType") : nullptr;
            const bool structured =
                reflected != nullptr && jsonString(reflected->find("baseShape")) == "structuredBuffer";
            return structured && result != nullptr && jsonString(result->find("kind")) == "struct"
                       ? std::optional{DataUsage::structuredBuffer}
                       : std::nullopt;
        }

        [[nodiscard]] inline std::string dataTypeName(const JsonValue& parameter, std::string_view fallback)
        {
            const auto* type = parameter.find("type");
            if (type != nullptr)
            {
                for (const auto* candidate :
                     {type->find("resultType"),
                      type->find("elementVarLayout") != nullptr ? type->find("elementVarLayout")->find("type")
                                                                : nullptr,
                      type->find("elementType")})
                {
                    if (candidate != nullptr && jsonString(candidate->find("kind")) == "struct")
                    {
                        if (const auto name = jsonString(candidate->find("name")); !name.empty())
                        {
                            return std::string(name);
                        }
                    }
                }
            }
            return std::string(fallback);
        }

        [[nodiscard]] inline bool equalsIgnoringCase(std::string_view left, std::string_view right) noexcept
        {
            return left.size() == right.size() && std::ranges::equal(left, right, [](char a, char b) {
                       const auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c; };
                       return lower(a) == lower(b);
                   });
        }

        [[nodiscard]] inline std::expected<void, std::string> appendVertexInput(const JsonValue& variable,
                                                                                InterfaceBackend backend,
                                                                                std::vector<std::uint32_t>& locations,
                                                                                std::size_t depth)
        {
            if (depth > 16)
            {
                return std::unexpected("A vertex input nests structures too deeply.");
            }
            const auto* binding = findBindingOfKind(variable, "varyingInput");
            const auto semantic = jsonString(variable.find("semanticName"));
            const auto* type = variable.find("type");
            if (semantic.empty() && type != nullptr && jsonString(type->find("kind")) == "struct")
            {
                if (const auto* fields = type->find("fields"); fields != nullptr && fields->asArray() != nullptr)
                {
                    for (const auto& field : *fields->asArray())
                    {
                        if (auto appended = appendVertexInput(field, backend, locations, depth + 1); !appended)
                        {
                            return appended;
                        }
                    }
                }
                return {};
            }
            if (binding == nullptr)
            {
                return {};
            }
            const auto count = jsonUint32(*binding, "count").value_or(1);
            std::uint32_t first = jsonUint32(*binding, "index").value_or(0);
            if (backend == InterfaceBackend::d3d12)
            {
                if (!equalsIgnoringCase(semantic, "TEXCOORD"))
                {
                    return std::unexpected("The vertex input '" + std::string(jsonString(variable.find("name"))) +
                                           "' uses the semantic '" + std::string(semantic) +
                                           "'. RachisRHI binds vertex location n to TEXCOORDn on Direct3D 12.");
                }
                first = jsonUint32(variable, "semanticIndex").value_or(0);
            }
            for (std::uint32_t offset = 0; offset < count; ++offset)
            {
                locations.push_back(first + offset);
            }
            return {};
        }

        [[nodiscard]] inline std::expected<void, std::string> validateBindingSets(
            const std::vector<ReflectedBinding>& bindings)
        {
            for (const auto& binding : bindings)
            {
                if (binding.set >= bindingSetLimit)
                {
                    return std::unexpected("The resource '" + binding.name + "' is in binding set " +
                                           std::to_string(binding.set) +
                                           ". RachisRHI binding sets are 0 to 3; register space 4 is reserved for "
                                           "push constants and space 5 for the draw index.");
                }
            }
            return {};
        }

        [[nodiscard]] inline std::expected<void, std::string> sortBindings(std::vector<ReflectedBinding>& bindings)
        {
            if (auto sets = validateBindingSets(bindings); !sets)
            {
                return sets;
            }
            std::ranges::sort(bindings, [](const ReflectedBinding& left, const ReflectedBinding& right) {
                return std::pair{left.set, left.binding} < std::pair{right.set, right.binding};
            });
            for (std::size_t index = 1; index < bindings.size(); ++index)
            {
                const auto& previous = bindings[index - 1];
                const auto& current = bindings[index];
                if (previous.set == current.set && previous.binding == current.binding)
                {
                    return std::unexpected("The resources '" + previous.name + "' and '" + current.name +
                                           "' share set " + std::to_string(current.set) + " binding " +
                                           std::to_string(current.binding) + ".");
                }
            }
            return {};
        }

    } // namespace detail

    [[nodiscard]] inline std::expected<ReflectedInterface, std::string> reflectShaderInterface(
        std::string_view reflectionJson, InterfaceBackend backend, std::string_view entryPoint, InterfaceStage stage)
    {
        auto root = detail::JsonParser(reflectionJson).parseDocument();
        if (!root.has_value())
        {
            return std::unexpected("The Slang reflection JSON does not parse: " + root.error());
        }
        const auto* entry = detail::selectEntryPoint(*root, entryPoint);
        if (entry == nullptr)
        {
            return std::unexpected("The Slang reflection JSON has no entry point '" + std::string(entryPoint) + "'.");
        }
        const auto used = detail::usedParameterNames(*entry);

        ReflectedInterface reflected;
        if (const auto* parameters = root->find("parameters");
            parameters != nullptr && parameters->asArray() != nullptr)
        {
            for (const auto& parameter : *parameters->asArray())
            {
                const std::string name(detail::jsonString(parameter.find("name")));
                if (used.has_value() && std::ranges::find(*used, name) == used->end())
                {
                    continue;
                }
                if (detail::isPushConstantParameter(parameter, backend))
                {
                    auto bytes = detail::pushConstantBlockBytes(parameter);
                    if (!bytes)
                    {
                        return std::unexpected(std::move(bytes.error()));
                    }
                    reflected.pushConstantBytes = std::max(reflected.pushConstantBytes, *bytes);
                    reflected.dataResources.push_back(DataResource{
                        .name = name,
                        .typeName = detail::dataTypeName(parameter, name),
                        .usage = DataUsage::pushConstants,
                    });
                    continue;
                }
                const auto slot = detail::resourceSlot(parameter, backend);
                const auto drawIndex = detail::classifyDrawIndexSpace(parameter, backend, slot);
                if (!drawIndex)
                {
                    return std::unexpected(drawIndex.error());
                }
                if (*drawIndex)
                {
                    reflected.drawIndex = true;
                    continue;
                }
                if (!slot.has_value())
                {
                    if (const auto* binding = parameter.find("binding");
                        binding != nullptr && detail::jsonString(binding->find("kind")) == "uniform")
                    {
                        return std::unexpected("The global '" + name +
                                               "' is a loose uniform. Put it in a constant buffer.");
                    }
                    continue;
                }
                auto shape = detail::classifyResource(parameter);
                if (!shape)
                {
                    return std::unexpected(std::move(shape.error()));
                }
                reflected.bindings.push_back(ReflectedBinding{
                    .name = name,
                    .set = slot->set,
                    .binding = slot->binding,
                    .type = shape->type,
                    .count = shape->count,
                });
                if (const auto usage = detail::dataUsage(parameter, shape->type);
                    usage.has_value() && shape->count == 1)
                {
                    reflected.dataResources.push_back(DataResource{
                        .name = name,
                        .typeName = detail::dataTypeName(parameter, name),
                        .usage = *usage,
                        .set = slot->set,
                        .binding = slot->binding,
                    });
                }
            }
        }
        if (auto sorted = detail::sortBindings(reflected.bindings); !sorted)
        {
            return std::unexpected(std::move(sorted.error()));
        }

        if (interfaceStageHasWorkgroup(stage))
        {
            const auto* size = entry->find("threadGroupSize");
            const auto* items = size != nullptr ? size->asArray() : nullptr;
            if (items == nullptr || items->size() != 3)
            {
                return std::unexpected("The " + std::string(interfaceStageName(stage)) + " entry point '" +
                                       std::string(entryPoint) + "' has no reflected thread-group size." +
                                       (stage == InterfaceStage::compute
                                            ? std::string{}
                                            : std::string(" Slang reflects the thread-group size of task "
                                                          "and mesh shaders from version 2026.20.")));
            }
            std::array<std::uint32_t, 3> workgroup{};
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                const auto value = (*items)[axis].asUint32();
                if (!value.has_value() || *value == 0)
                {
                    return std::unexpected("The " + std::string(interfaceStageName(stage)) +
                                           " entry point's thread-group size is not a positive integer.");
                }
                workgroup[axis] = *value;
            }
            reflected.workgroupSize = workgroup;
        }

        if (stage == InterfaceStage::vertex)
        {
            if (const auto* parameters = entry->find("parameters");
                parameters != nullptr && parameters->asArray() != nullptr)
            {
                for (const auto& parameter : *parameters->asArray())
                {
                    if (auto appended =
                            detail::appendVertexInput(parameter, backend, reflected.vertexInputLocations, 0);
                        !appended)
                    {
                        return std::unexpected(std::move(appended.error()));
                    }
                }
            }
            std::ranges::sort(reflected.vertexInputLocations);
            if (std::ranges::adjacent_find(reflected.vertexInputLocations) != reflected.vertexInputLocations.end())
            {
                return std::unexpected("Two vertex inputs share a location.");
            }
        }
        reflected.drawIndex = reflected.drawIndex && interfaceStageReadsDrawIndex(stage);
        return reflected;
    }

    namespace detail
    {
        inline void fitPushConstants(ReflectedInterface& reflected)
        {
            for (const auto& block : reflected.blocks)
            {
                if (block.resource.usage == DataUsage::pushConstants)
                {
                    reflected.pushConstantBytes = roundUp(block.size, cppAlignment(*block.type));
                }
            }
        }

    } // namespace detail

    [[nodiscard]] inline std::expected<void, std::string> attachSlangDataLayouts(ReflectedInterface& reflected,
                                                                                 std::string_view layoutJson,
                                                                                 bool metal)
    {
        auto root = detail::JsonParser(layoutJson).parseDocument();
        if (!root.has_value())
        {
            return std::unexpected("The Slang layout reflection does not parse: " + root.error());
        }
        const auto* parameters = root->find("parameters");
        for (const auto& resource : reflected.dataResources)
        {
            const detail::JsonValue* match = nullptr;
            if (parameters != nullptr && parameters->asArray() != nullptr)
            {
                for (const auto& parameter : *parameters->asArray())
                {
                    if (detail::jsonString(parameter.find("name")) == resource.name)
                    {
                        match = &parameter;
                        break;
                    }
                }
            }
            if (match == nullptr)
            {
                return std::unexpected("The layout reflection has no resource '" + resource.name + "'.");
            }
            auto block = slangDataBlock(*match, resource, metal);
            if (!block)
            {
                return std::unexpected(std::move(block.error()));
            }
            reflected.blocks.push_back(std::move(*block));
        }
        detail::fitPushConstants(reflected);
        return {};
    }

    [[nodiscard]] inline std::expected<void, std::string> attachSpirvDataLayouts(ReflectedInterface& reflected,
                                                                                 std::span<const std::byte> spirv)
    {
        auto blocks = spirvDataBlocks(spirv, reflected.dataResources);
        if (!blocks)
        {
            return std::unexpected(std::move(blocks.error()));
        }
        reflected.blocks = std::move(*blocks);
        detail::fitPushConstants(reflected);
        return {};
    }

    [[nodiscard]] inline std::expected<bool, std::string> dxilReadsDrawIndex(std::span<const std::byte> container)
    {
        constexpr std::size_t headerSize = 32;
        constexpr std::uint32_t constantBufferType = 2;
        constexpr std::uint32_t bindInfoSize = 16;
        const auto readU32 = [](std::span<const std::byte> bytes, std::size_t offset) {
            std::uint32_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return value;
        };
        if (container.size() < headerSize || std::memcmp(container.data(), "DXBC", 4) != 0)
        {
            return std::unexpected("The Direct3D 12 bytecode is not a DXIL container.");
        }
        const auto size = std::min<std::size_t>(readU32(container, 24), container.size());
        const auto parts = readU32(container, 28);
        if (size < headerSize || parts > (size - headerSize) / sizeof(std::uint32_t))
        {
            return std::unexpected("The DXIL container has an invalid size or part count.");
        }
        for (std::uint32_t part = 0; part < parts; ++part)
        {
            const std::size_t offset = readU32(container, headerSize + part * sizeof(std::uint32_t));
            if (offset > size || size - offset < 8)
            {
                return std::unexpected("The DXIL container has a part outside the container.");
            }
            const std::size_t partSize = readU32(container, offset + 4);
            if (partSize > size - offset - 8)
            {
                return std::unexpected("The DXIL container has a part that runs past the container.");
            }
            if (std::memcmp(container.data() + offset, "PSV0", 4) != 0)
            {
                continue;
            }
            const auto data = container.subspan(offset + 8, partSize);
            if (data.size() < sizeof(std::uint32_t))
            {
                return std::unexpected("The DXIL PSV0 part is truncated.");
            }
            const std::size_t infoSize = readU32(data, 0);
            if (infoSize > data.size() - 4 || data.size() - 4 - infoSize < sizeof(std::uint32_t))
            {
                return std::unexpected("The DXIL PSV0 part is truncated.");
            }
            const auto resources = readU32(data, 4 + infoSize);
            if (resources == 0)
            {
                return false;
            }
            const std::size_t table = 4 + infoSize + 4;
            if (data.size() - table < sizeof(std::uint32_t))
            {
                return std::unexpected("The DXIL PSV0 resource table is truncated.");
            }
            const std::size_t recordSize = readU32(data, table);
            if (recordSize < bindInfoSize || resources > (data.size() - table - 4) / recordSize)
            {
                return std::unexpected("The DXIL PSV0 resource table is truncated.");
            }
            for (std::uint32_t resource = 0; resource < resources; ++resource)
            {
                const auto record = table + 4 + resource * recordSize;
                if (readU32(data, record) == constantBufferType && readU32(data, record + 4) == drawIndexSpace &&
                    readU32(data, record + 8) <= drawIndexRegister && drawIndexRegister <= readU32(data, record + 12))
                {
                    return true;
                }
            }
            return false;
        }
        return std::unexpected("The DXIL container has no PSV0 part.");
    }

    [[nodiscard]] constexpr std::uint32_t spirvExecutionModel(InterfaceStage stage) noexcept
    {
        switch (stage)
        {
        case InterfaceStage::vertex:
            return detail::SpirvLayoutModule::executionModelVertex;
        case InterfaceStage::pixel:
            return detail::SpirvLayoutModule::executionModelFragment;
        case InterfaceStage::compute:
            return detail::SpirvLayoutModule::executionModelGLCompute;
        case InterfaceStage::task:
            return detail::SpirvLayoutModule::executionModelTaskEXT;
        case InterfaceStage::mesh:
            return detail::SpirvLayoutModule::executionModelMeshEXT;
        }
        return detail::SpirvLayoutModule::executionModelGLCompute;
    }

    [[nodiscard]] inline std::expected<bool, std::string> spirvReadsDrawIndex(std::span<const std::byte> spirv,
                                                                              std::string_view entryPoint,
                                                                              InterfaceStage stage)
    {
        auto module = detail::SpirvLayoutModule::parse(spirv);
        if (!module)
        {
            return std::unexpected(std::move(module.error()));
        }
        auto found = module->interfaceOf(entryPoint, spirvExecutionModel(stage));
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        return found->drawIndex;
    }

    namespace detail
    {
        [[nodiscard]] constexpr bool reflectedStageMatches(std::string_view reflected, InterfaceStage stage) noexcept
        {
            switch (stage)
            {
            case InterfaceStage::vertex:
                return reflected == "vertex";
            case InterfaceStage::pixel:
                return reflected == "fragment" || reflected == "pixel";
            case InterfaceStage::compute:
                return reflected == "compute";
            case InterfaceStage::task:
                return reflected == "amplification";
            case InterfaceStage::mesh:
                return reflected == "mesh";
            }
            return false;
        }

        [[nodiscard]] inline bool isAnnotatedDrawIndex(const JsonValue* annotations, std::string_view name)
        {
            if (annotations == nullptr || annotations->asArray() == nullptr)
            {
                return false;
            }
            return std::ranges::any_of(*annotations->asArray(), [&](const JsonValue& annotation) {
                const auto* flag = annotation.find("drawIndex");
                return jsonString(annotation.find("name")) == name && flag != nullptr && flag->asBool() != nullptr &&
                       *flag->asBool();
            });
        }
    } // namespace detail

    [[nodiscard]] inline std::expected<bool, std::string> metalManifestReadsDrawIndex(std::span<const std::byte> pnms,
                                                                                      std::string_view entryPoint,
                                                                                      InterfaceStage stage)
    {
        constexpr std::size_t headerBytes = 16;
        constexpr std::uint32_t manifestVersion = 1;
        if (pnms.size() < headerBytes || std::memcmp(pnms.data(), "PNMS", 4) != 0)
        {
            return std::unexpected("The Metal shader does not start with a PNMS header.");
        }
        const auto readU32 = [&](std::size_t offset) {
            std::uint32_t value = 0;
            std::memcpy(&value, pnms.data() + offset, sizeof(value));
            return value;
        };
        if (readU32(4) != manifestVersion)
        {
            return std::unexpected("The PNMS header has an unsupported version.");
        }
        const std::size_t jsonBytes = readU32(8);
        const std::size_t metallibBytes = readU32(12);
        if (jsonBytes > pnms.size() - headerBytes || metallibBytes != pnms.size() - headerBytes - jsonBytes)
        {
            return std::unexpected("The PNMS section sizes do not match the size of the shader.");
        }
        if (jsonBytes == 0 || metallibBytes < 4 || std::memcmp(pnms.data() + headerBytes + jsonBytes, "MTLB", 4) != 0)
        {
            return std::unexpected("The PNMS shader lacks its reflection JSON or its Metal library.");
        }
        const std::string_view json(reinterpret_cast<const char*>(pnms.data() + headerBytes), jsonBytes);
        auto root = detail::JsonParser(json).parseDocument();
        if (!root.has_value())
        {
            return std::unexpected("The PNMS reflection does not parse: " + root.error());
        }
        const auto* entries = root->find("entryPoints");
        if (root->asObject() == nullptr || entries == nullptr || entries->asArray() == nullptr)
        {
            return std::unexpected("The PNMS reflection has no entry points.");
        }
        const detail::JsonValue* entry = nullptr;
        for (const auto& candidate : *entries->asArray())
        {
            if (detail::jsonString(candidate.find("name")) != entryPoint)
            {
                continue;
            }
            if (entry != nullptr)
            {
                return std::unexpected("The PNMS reflection lists the entry point '" + std::string(entryPoint) +
                                       "' twice.");
            }
            entry = &candidate;
        }
        if (entry == nullptr)
        {
            return std::unexpected("The PNMS reflection has no entry point '" + std::string(entryPoint) + "'.");
        }
        if (!detail::reflectedStageMatches(detail::jsonString(entry->find("stage")), stage))
        {
            return std::unexpected("The PNMS entry point '" + std::string(entryPoint) + "' is not a " +
                                   std::string(interfaceStageName(stage)) + " shader.");
        }
        const auto* active = entry->find("bindings");
        if (active == nullptr)
        {
            return false;
        }
        if (active->asArray() == nullptr)
        {
            return std::unexpected("The PNMS entry point bindings are not an array.");
        }
        const auto* annotations = root->find("rachisBindings");
        return std::ranges::any_of(*active->asArray(), [&](const detail::JsonValue& binding) {
            const auto name = detail::jsonString(binding.find("name"));
            if (!detail::isAnnotatedDrawIndex(annotations, name))
            {
                return false;
            }
            const auto* record = detail::findBindingOfKind(binding, "constantBuffer");
            const auto* used = record != nullptr ? record->find("used") : nullptr;
            if (used == nullptr)
            {
                return record != nullptr;
            }
            if (const auto* flag = used->asBool(); flag != nullptr)
            {
                return *flag;
            }
            const auto value = used->asUint32();
            return value.has_value() && *value != 0;
        });
    }

    [[nodiscard]] inline std::expected<ReflectedInterface, std::string> reflectSpirvShaderInterface(
        std::span<const std::byte> spirv, std::string_view entryPoint, InterfaceStage stage)
    {
        auto module = detail::SpirvLayoutModule::parse(spirv);
        if (!module)
        {
            return std::unexpected(std::move(module.error()));
        }
        auto found = module->interfaceOf(entryPoint, spirvExecutionModel(stage));
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        ReflectedInterface reflected;
        for (auto& binding : found->bindings)
        {
            reflected.bindings.push_back(ReflectedBinding{
                .name = std::move(binding.name),
                .set = binding.set,
                .binding = binding.binding,
                .type = binding.type,
                .count = binding.count,
            });
        }
        if (auto sorted = detail::sortBindings(reflected.bindings); !sorted)
        {
            return std::unexpected(std::move(sorted.error()));
        }
        reflected.dataResources = std::move(found->dataResources);
        reflected.vertexInputLocations = std::move(found->vertexInputLocations);
        reflected.workgroupSize = found->workgroupSize;
        reflected.drawIndex = found->drawIndex;
        if (auto attached = attachSpirvDataLayouts(reflected, spirv); !attached)
        {
            return std::unexpected(std::move(attached.error()));
        }
        return reflected;
    }

    namespace detail
    {
        [[nodiscard]] inline std::expected<void, std::string> appendDataStructs(std::string& text,
                                                                                const std::vector<DataBlock>& blocks)
        {
            std::vector<std::pair<std::string, const DataBlock*>> emitted;
            DataStructEmitter emitter(text);
            for (const auto& block : blocks)
            {
                auto name = cppIdentifier(block.type->name);
                bool shared = false;
                for (const auto& [existing, other] : emitted)
                {
                    if (existing != name)
                    {
                        continue;
                    }
                    const bool sameTarget =
                        (other->resource.usage == DataUsage::structuredBuffer) ==
                            (block.resource.usage == DataUsage::structuredBuffer) &&
                        (block.resource.usage != DataUsage::structuredBuffer || other->size == block.size);
                    if (sameTarget && layoutsMatch(*other->type, *block.type))
                    {
                        shared = true;
                    }
                    else
                    {
                        name += "_" + cppIdentifier(block.resource.name);
                    }
                    break;
                }
                if (shared)
                {
                    continue;
                }
                if (auto written = emitter.emitBlock(block, name); !written)
                {
                    return written;
                }
                emitted.emplace_back(name, &block);
            }
            return {};
        }

        [[nodiscard]] constexpr std::string_view stageEnumerator(InterfaceStage stage) noexcept
        {
            return interfaceStageName(stage);
        }

        [[nodiscard]] inline std::string countText(std::uint32_t count)
        {
            return count == unboundedResourceCount ? std::string("rachis::unboundedBindingCount")
                                                   : std::to_string(count);
        }

        inline void appendArrayHeader(std::string& text, std::string_view type, std::size_t size, std::string_view name)
        {
            text += "        inline constexpr std::array<";
            text += type;
            text += ", ";
            text += std::to_string(size);
            text += "> ";
            text += name;
        }

        struct PoolSize final
        {
            std::string_view type;
            std::uint32_t count = 0;
        };

        [[nodiscard]] inline std::uint32_t bindingSetCount(const ReflectedInterface& reflected) noexcept
        {
            std::uint32_t sets = 0;
            for (const auto& binding : reflected.bindings)
            {
                sets = std::max(sets, binding.set + 1);
            }
            return sets;
        }

        [[nodiscard]] inline std::expected<void, std::string> appendBackendNamespace(std::string& text,
                                                                                     const BackendInterface& backend)
        {
            const auto& reflected = backend.reflected;
            text += "    namespace ";
            text += interfaceBackendName(backend.backend);
            text += "\n    {\n";
            if (auto structs = appendDataStructs(text, reflected.blocks); !structs)
            {
                return std::unexpected("The " + std::string(interfaceBackendName(backend.backend)) +
                                       " layout: " + structs.error());
            }

            appendArrayHeader(text, "rachis::ShaderBindingRequirement", reflected.bindings.size(), "bindings");
            if (reflected.bindings.empty())
            {
                text += "{};\n";
            }
            else
            {
                text += "{{\n";
                for (const auto& binding : reflected.bindings)
                {
                    text += "            {.set = " + std::to_string(binding.set) +
                            ", .binding = " + std::to_string(binding.binding) +
                            ", .type = rachis::ShaderResourceType::" + std::string(binding.type) +
                            ", .count = " + countText(binding.count) + "}, // " + binding.name + "\n";
                }
                text += "        }};\n";
            }

            const auto pushCount = reflected.pushConstantBytes == 0 ? 0U : 1U;
            appendArrayHeader(text, "rachis::ShaderPushConstantRequirement", pushCount, "pushConstants");
            text += pushCount == 0
                        ? std::string("{};\n")
                        : "{{{.offset = 0, .size = " + std::to_string(reflected.pushConstantBytes) + "}}};\n";
            appendArrayHeader(text, "rachis::PushConstantRange", pushCount, "pushConstantRanges");
            text += pushCount == 0 ? std::string("{};\n")
                                   : "{{{.visibility = stage, .offset = 0, .size = " +
                                         std::to_string(reflected.pushConstantBytes) + "}}};\n";

            appendArrayHeader(text, "std::uint32_t", reflected.vertexInputLocations.size(), "vertexInputLocations");
            text += "{";
            for (std::size_t index = 0; index < reflected.vertexInputLocations.size(); ++index)
            {
                text += index == 0 ? "" : ", ";
                text += std::to_string(reflected.vertexInputLocations[index]);
            }
            text += "};\n";

            text += "        inline constexpr std::optional<rachis::ShaderWorkgroupSize> workgroupSize";
            if (reflected.workgroupSize.has_value())
            {
                const auto& size = *reflected.workgroupSize;
                text += "{rachis::ShaderWorkgroupSize{" + std::to_string(size[0]) + ", " + std::to_string(size[1]) +
                        ", " + std::to_string(size[2]) + "}};\n";
            }
            else
            {
                text += "{};\n";
            }
            text += "        inline constexpr bool drawIndex = ";
            text += reflected.drawIndex ? "true;\n" : "false;\n";

            const auto sets = bindingSetCount(reflected);
            text += "        inline constexpr std::uint32_t bindingSetCount = " + std::to_string(sets) + ";\n";
            for (std::uint32_t set = 0; set < sets; ++set)
            {
                std::vector<const ReflectedBinding*> entries;
                std::vector<PoolSize> pool;
                for (const auto& binding : reflected.bindings)
                {
                    if (binding.set != set)
                    {
                        continue;
                    }
                    entries.push_back(&binding);
                    if (binding.count == unboundedResourceCount)
                    {
                        continue;
                    }
                    auto found = std::ranges::find(pool, binding.type, &PoolSize::type);
                    if (found == pool.end())
                    {
                        pool.push_back(PoolSize{binding.type, binding.count});
                    }
                    else
                    {
                        found->count += binding.count;
                    }
                }
                const auto suffix = std::to_string(set);
                appendArrayHeader(text, "rachis::BindingLayoutEntry", entries.size(), "bindingLayout" + suffix);
                if (entries.empty())
                {
                    text += "{};\n";
                }
                else
                {
                    text += "{{\n";
                    for (const auto* binding : entries)
                    {
                        text += "            {.binding = " + std::to_string(binding->binding) +
                                ", .type = rachis::BindingType::" + std::string(binding->type) +
                                ", .visibility = stage, .count = " + countText(binding->count);
                        if (binding->count == unboundedResourceCount)
                        {
                            text += ", .flags = rachis::BindingFlag::variableDescriptorCount";
                        }
                        text += "},\n";
                    }
                    text += "        }};\n";
                }
                appendArrayHeader(text, "rachis::BindingPoolSize", pool.size(), "bindingPoolSizes" + suffix);
                if (pool.empty())
                {
                    text += "{};\n";
                }
                else
                {
                    text += "{{\n";
                    for (const auto& size : pool)
                    {
                        text += "            {.type = rachis::BindingType::" + std::string(size.type) +
                                ", .descriptorCount = " + std::to_string(size.count) + "},\n";
                    }
                    text += "        }};\n";
                }
            }

            text += "\n        [[nodiscard]] constexpr std::span<const rachis::BindingLayoutEntry> bindingLayout("
                    "std::uint32_t set) noexcept\n        {\n";
            text += sets == 0 ? "            static_cast<void>(set);\n" : "            switch (set)\n            {\n";
            for (std::uint32_t set = 0; set < sets; ++set)
            {
                text += "            case " + std::to_string(set) + ":\n                return bindingLayout" +
                        std::to_string(set) + ";\n";
            }
            text += sets == 0 ? "            return {};\n"
                              : "            default:\n                return {};\n            }\n";
            text += "        }\n";

            text += "        [[nodiscard]] constexpr std::span<const rachis::BindingPoolSize> bindingPoolSizes("
                    "std::uint32_t set) noexcept\n        {\n";
            text += sets == 0 ? "            static_cast<void>(set);\n" : "            switch (set)\n            {\n";
            for (std::uint32_t set = 0; set < sets; ++set)
            {
                text += "            case " + std::to_string(set) + ":\n                return bindingPoolSizes" +
                        std::to_string(set) + ";\n";
            }
            text += sets == 0 ? "            return {};\n"
                              : "            default:\n                return {};\n            }\n";
            text += "        }\n";

            text += "        [[nodiscard]] constexpr rachis::ShaderInterfaceDesc shaderInterface() noexcept\n"
                    "        {\n"
                    "            return {\n"
                    "                .bindings = bindings,\n"
                    "                .pushConstants = pushConstants,\n"
                    "                .vertexInputLocations = vertexInputLocations,\n"
                    "                .workgroupSize = workgroupSize,\n"
                    "                .drawIndex = drawIndex,\n"
                    "            };\n"
                    "        }\n";
            text += "    } // namespace ";
            text += interfaceBackendName(backend.backend);
            text += "\n\n";
            return {};
        }

        inline void appendDispatch(std::string& text,
                                   std::span<const BackendInterface> backends,
                                   std::string_view signature,
                                   std::string_view call,
                                   std::string_view fallback)
        {
            text += "    ";
            text += signature;
            text += "\n    {\n        switch (backend)\n        {\n";
            for (const auto& backend : backends)
            {
                const auto name = interfaceBackendName(backend.backend);
                text += "        case rachis::Backend::";
                text += name;
                text += ":\n            return ";
                text += name;
                text += "::";
                text += call;
                text += ";\n";
            }
            text += "        default:\n            return ";
            text += fallback;
            text += ";\n        }\n    }\n\n";
        }

    } // namespace detail

    [[nodiscard]] inline std::expected<std::string, std::string> generateShaderInterfaceHeader(
        std::string_view name,
        InterfaceStage stage,
        std::string_view entryPoint,
        std::span<const BackendInterface> backends,
        std::string_view tag = {})
    {
        if (!detail::isValidCppIdentifier(name))
        {
            return std::unexpected("The shader name must be a C++ identifier.");
        }
        if (!tag.empty() && !detail::isValidCppIdentifier("t_" + std::string(tag)))
        {
            return std::unexpected("The tag must make t_<tag> a C++ identifier.");
        }
        if (!detail::isValidCppIdentifier(entryPoint))
        {
            return std::unexpected("The entry point must be a C++ identifier.");
        }
        if (backends.empty())
        {
            return std::unexpected("At least one backend's reflection is required.");
        }

        std::string text;
        text += "// Generated by RachisShaderBinaryTool reflect-interface from the reflection of each backend's\n"
                "// compilation. Do not edit.\n"
                "#pragma once\n\n"
                "#include <Rachis/Binding.h>\n"
                "#include <Rachis/Device.h>\n"
                "#include <Rachis/Pipeline.h>\n\n"
                "#include <array>\n"
                "#include <cstddef>\n"
                "#include <cstdint>\n"
                "#include <optional>\n"
                "#include <span>\n"
                "#include <string_view>\n\n";
        text += stridedArraySupport;
        text += "\n// The structs hold each buffer's data as the backend lays it out. Vectors are std::array, and a\n"
                "// matrix is an array of the vectors it stores: its columns, since shaders are column-major.\n";
        text += "namespace rachis_shaders::reflection::";
        text += name;
        text += "\n{\n";
        if (!tag.empty())
        {
            text += "inline namespace t_";
            text += tag;
            text += "\n{\n";
        }
        text += "    inline constexpr rachis::ShaderStage stage = rachis::ShaderStage::";
        text += detail::stageEnumerator(stage);
        text += ";\n    inline constexpr std::string_view entryPoint = \"";
        text += entryPoint;
        text += "\";\n\n";
        for (const auto& backend : backends)
        {
            if (auto appended = detail::appendBackendNamespace(text, backend); !appended)
            {
                return std::unexpected(std::move(appended.error()));
            }
        }
        detail::appendDispatch(text,
                               backends,
                               "// The shader interface for ShaderModuleDesc::shaderInterface.\n"
                               "    [[nodiscard]] constexpr rachis::ShaderInterfaceDesc shaderInterface("
                               "rachis::Backend backend) noexcept",
                               "shaderInterface()",
                               "{}");
        detail::appendDispatch(text,
                               backends,
                               "// One past the highest binding set the shader uses.\n"
                               "    [[nodiscard]] constexpr std::uint32_t bindingSetCount("
                               "rachis::Backend backend) noexcept",
                               "bindingSetCount",
                               "0");
        detail::appendDispatch(text,
                               backends,
                               "// The binding-layout entries of one set, visible to this shader's stage.\n"
                               "    [[nodiscard]] constexpr std::span<const rachis::BindingLayoutEntry> "
                               "bindingLayout(rachis::Backend backend, std::uint32_t set) noexcept",
                               "bindingLayout(set)",
                               "{}");
        detail::appendDispatch(text,
                               backends,
                               "// The pool sizes one binding set of `set` needs. Unbounded arrays are left out.\n"
                               "    [[nodiscard]] constexpr std::span<const rachis::BindingPoolSize> "
                               "bindingPoolSizes(rachis::Backend backend, std::uint32_t set) noexcept",
                               "bindingPoolSizes(set)",
                               "{}");
        detail::appendDispatch(text,
                               backends,
                               "// The push-constant ranges for PipelineLayoutDesc, visible to this shader's stage.\n"
                               "    [[nodiscard]] constexpr std::span<const rachis::PushConstantRange> "
                               "pushConstantRanges(rachis::Backend backend) noexcept",
                               "pushConstantRanges",
                               "{}");
        detail::appendDispatch(text,
                               backends,
                               "// Whether the shader reads the draw index, for PipelineLayoutDesc::drawIndex.\n"
                               "    [[nodiscard]] constexpr bool drawIndex(rachis::Backend backend) noexcept",
                               "drawIndex",
                               "false");
        if (!tag.empty())
        {
            text += "} // inline namespace t_";
            text += tag;
            text += "\n";
        }
        text += "} // namespace rachis_shaders::reflection::";
        text += name;
        text += "\n";
        return text;
    }

} // namespace rachis::shader_tool
