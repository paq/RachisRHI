#import <Foundation/Foundation.h>

#include "Internal/MetalShaderManifest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace rachis::metal
{
    namespace
    {

        [[nodiscard]] Error manifestError(std::string message)
        {
            return Error{ErrorCode::invalidArgument, 0, std::move(message)};
        }

        [[nodiscard]] std::uint32_t readU32(std::span<const std::byte> bytes, std::size_t offset)
        {
            std::uint32_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return value;
        }

        [[nodiscard]] NSDictionary* dictionary(id value)
        {
            return [value isKindOfClass:[NSDictionary class]] ? value : nil;
        }

        [[nodiscard]] NSArray* array(id value)
        {
            return [value isKindOfClass:[NSArray class]] ? value : nil;
        }

        [[nodiscard]] NSString* string(id value)
        {
            return [value isKindOfClass:[NSString class]] ? value : nil;
        }

        [[nodiscard]] std::optional<std::uint32_t> uintValue(id value)
        {
            if (![value isKindOfClass:[NSNumber class]] ||
                CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID())
            {
                return std::nullopt;
            }
            const double number = [value doubleValue];
            if (!std::isfinite(number) || number < 0 || number > std::numeric_limits<std::uint32_t>::max() ||
                std::floor(number) != number)
            {
                return std::nullopt;
            }
            return static_cast<std::uint32_t>(number);
        }

        [[nodiscard]] NSDictionary* bindingOfKind(NSDictionary* parameter, NSString* kind)
        {
            if (kind == nil)
            {
                return nil;
            }
            NSDictionary* const single = dictionary(parameter[@"binding"]);
            if ([string(single[@"kind"]) isEqualToString:kind])
            {
                return single;
            }

            for (id item in array(parameter[@"bindings"]))
            {
                NSDictionary* const binding = dictionary(item);
                if ([string(binding[@"kind"]) isEqualToString:kind])
                {
                    return binding;
                }
            }
            return nil;
        }

        [[nodiscard]] std::optional<std::uint32_t> uniformSize(NSDictionary* type)
        {
            for (id item in array(type[@"sizes"]))
            {
                NSDictionary* const size = dictionary(item);
                if ([string(size[@"kind"]) isEqualToString:@"uniform"])
                {
                    return uintValue(size[@"value"]);
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] bool collectFields(NSDictionary* type,
                                         std::uint32_t base,
                                         std::vector<MetalReflectedField>& fields,
                                         std::uint32_t depth = 0)
        {
            if (depth > 64)
            {
                return false;
            }

            for (id item in array(type[@"fields"]))
            {
                NSDictionary* const field = dictionary(item);
                NSString* const name = string(field[@"name"]);
                NSDictionary* const binding = bindingOfKind(field, @"uniform");
                const auto offset = uintValue(binding[@"offset"]);
                const auto size = uintValue(binding[@"size"]);
                if (name == nil || !offset || !size || *offset > std::numeric_limits<std::uint32_t>::max() - base ||
                    *size > std::numeric_limits<std::uint32_t>::max() - base - *offset)
                {
                    return false;
                }
                const char* const fieldName = name.UTF8String;
                if (fieldName == nullptr)
                {
                    return false;
                }
                fields.push_back({.path = fieldName, .offset = base + *offset, .size = *size});
                if (!collectFields(dictionary(field[@"type"]), base + *offset, fields, depth + 1))
                {
                    return false;
                }
            }
            if (NSDictionary* const element = dictionary(type[@"elementType"]))
            {
                return collectFields(element, base, fields, depth + 1);
            }
            return true;
        }

    } // namespace

    std::vector<std::byte> packShaderManifest(std::span<const std::byte> json, std::span<const std::byte> metallib)
    {
        if (json.size() > std::numeric_limits<std::uint32_t>::max() ||
            metallib.size() > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::length_error("Metal shader manifest sections exceed the PNMS size fields.");
        }
        std::vector<std::byte> packed(shaderManifestHeaderBytes + json.size() + metallib.size());
        packed[0] = std::byte{'P'};
        packed[1] = std::byte{'N'};
        packed[2] = std::byte{'M'};
        packed[3] = std::byte{'S'};
        const std::uint32_t version = shaderManifestVersion;
        const auto jsonSize = static_cast<std::uint32_t>(json.size());
        const auto metallibSize = static_cast<std::uint32_t>(metallib.size());
        std::memcpy(packed.data() + 4, &version, sizeof(version));
        std::memcpy(packed.data() + 8, &jsonSize, sizeof(jsonSize));
        std::memcpy(packed.data() + 12, &metallibSize, sizeof(metallibSize));
        if (!json.empty())
        {
            std::memcpy(packed.data() + shaderManifestHeaderBytes, json.data(), json.size());
        }
        if (!metallib.empty())
        {
            std::memcpy(packed.data() + shaderManifestHeaderBytes + json.size(), metallib.data(), metallib.size());
        }
        return packed;
    }

    Result<MetalShaderPayload> unpackShaderManifest(std::span<const std::byte> bytecode)
    {
        if (bytecode.size() < shaderManifestHeaderBytes)
        {
            return std::unexpected(manifestError("Metal shader bytecode is missing the PNMS identity header."));
        }
        if (bytecode[0] != std::byte{'P'} || bytecode[1] != std::byte{'N'} || bytecode[2] != std::byte{'M'} ||
            bytecode[3] != std::byte{'S'})
        {
            return std::unexpected(manifestError(
                "Metal shader bytecode is not a PNMS identity blob. Raw metallib without reflection is rejected."));
        }
        const auto version = readU32(bytecode, 4);
        if (version != shaderManifestVersion)
        {
            return std::unexpected(manifestError("Unsupported Metal shader identity version."));
        }
        const auto jsonSize = readU32(bytecode, 8);
        const auto metallibSize = readU32(bytecode, 12);
        if (shaderManifestHeaderBytes + static_cast<std::size_t>(jsonSize) + static_cast<std::size_t>(metallibSize) !=
            bytecode.size())
        {
            return std::unexpected(manifestError("Metal shader identity sizes do not match the bytecode length."));
        }
        if (jsonSize == 0 || metallibSize < 4)
        {
            return std::unexpected(
                manifestError("Metal shader identity is missing reflection JSON or metallib bytes."));
        }
        const auto json = bytecode.subspan(shaderManifestHeaderBytes, jsonSize);
        const auto metallib = bytecode.subspan(shaderManifestHeaderBytes + jsonSize, metallibSize);
        if (metallib[0] != std::byte{'M'} || metallib[1] != std::byte{'T'} || metallib[2] != std::byte{'L'} ||
            metallib[3] != std::byte{'B'})
        {
            return std::unexpected(manifestError("PNMS metallib payload is not a Metal library."));
        }
        return MetalShaderPayload{.json = json, .metallib = metallib};
    }

    namespace
    {

        [[nodiscard]] Result<NSDictionary*> findReflectedEntryPoint(NSDictionary* root, NSString* requested)
        {
            NSDictionary* entry = nil;

            for (id item in array(root[@"entryPoints"]))
            {
                NSDictionary* const candidate = dictionary(item);
                if ([string(candidate[@"name"]) isEqualToString:requested])
                {
                    if (entry != nil)
                    {
                        return std::unexpected(manifestError("Metal reflection contains a duplicate entry point."));
                    }
                    entry = candidate;
                }
            }
            if (entry == nil)
            {
                return std::unexpected(manifestError("Metal reflection does not contain the requested entry point."));
            }
            return entry;
        }

        [[nodiscard]] ShaderStage reflectedShaderStage(NSString* stageName)
        {
            if ([stageName isEqualToString:@"compute"])
            {
                return ShaderStage::compute;
            }
            if ([stageName isEqualToString:@"vertex"])
            {
                return ShaderStage::vertex;
            }
            if ([stageName isEqualToString:@"fragment"] || [stageName isEqualToString:@"pixel"])
            {
                return ShaderStage::pixel;
            }
            if ([stageName isEqualToString:@"mesh"])
            {
                return ShaderStage::mesh;
            }
            if ([stageName isEqualToString:@"amplification"])
            {
                return ShaderStage::task;
            }
            return ShaderStage::none;
        }

        [[nodiscard]] Result<ShaderWorkgroupSize> reflectedWorkgroupSize(NSDictionary* entry)
        {
            NSArray* const group = array(entry[@"threadGroupSize"]);
            if (group.count != 3)
            {
                return std::unexpected(manifestError("Metal compute reflection requires three workgroup dimensions."));
            }
            const auto x = uintValue(group[0]);
            const auto y = uintValue(group[1]);
            const auto z = uintValue(group[2]);
            if (!x || !y || !z || *x == 0 || *y == 0 || *z == 0)
            {
                return std::unexpected(manifestError("Metal workgroup dimensions must be positive 32-bit integers."));
            }
            return ShaderWorkgroupSize{*x, *y, *z};
        }

        struct LogicalBindingAnnotation final
        {
            std::string name;
            bool pushConstant = false;
            std::uint32_t set = 0;
            std::uint32_t binding = 0;
        };

        [[nodiscard]] Result<LogicalBindingAnnotation> parseLogicalBindingAnnotation(
            NSDictionary* node, const std::vector<LogicalBindingAnnotation>& rachisBindings)
        {
            NSString* const annotatedName = string(node[@"name"]);
            const char* const utf8 = annotatedName != nil ? annotatedName.UTF8String : nullptr;
            if (node == nil || utf8 == nullptr)
            {
                return std::unexpected(manifestError("Metal rachisBindings entry is missing a valid UTF-8 name."));
            }
            if (std::ranges::any_of(rachisBindings, [&](const auto& other) { return other.name == utf8; }))
            {
                return std::unexpected(manifestError("Metal rachisBindings contains a duplicate resource name."));
            }
            id const pushFlag = node[@"pushConstant"];
            if (pushFlag != nil)
            {
                if (![pushFlag isKindOfClass:[NSNumber class]] ||
                    CFGetTypeID((__bridge CFTypeRef)pushFlag) != CFBooleanGetTypeID() || ![pushFlag boolValue])
                {
                    return std::unexpected(manifestError("Metal rachisBindings pushConstant flag must be true."));
                }
                return LogicalBindingAnnotation{.name = utf8, .pushConstant = true};
            }
            const auto logicalSet = uintValue(node[@"set"]);
            const auto logicalBinding = uintValue(node[@"binding"]);
            if (!logicalSet || !logicalBinding || *logicalSet >= maxBindingSets)
            {
                return std::unexpected(
                    manifestError("Metal rachisBindings entries require set in 0..3 and an unsigned binding index."));
            }
            return LogicalBindingAnnotation{
                .name = utf8,
                .pushConstant = false,
                .set = *logicalSet,
                .binding = *logicalBinding,
            };
        }

        [[nodiscard]] Result<std::vector<LogicalBindingAnnotation>> parseLogicalBindingAnnotations(NSDictionary* root)
        {
            std::vector<LogicalBindingAnnotation> rachisBindings;
            if (root[@"rachisBindings"] == nil)
            {
                return rachisBindings;
            }
            NSArray* const annotations = array(root[@"rachisBindings"]);
            if (annotations == nil)
            {
                return std::unexpected(manifestError("Metal rachisBindings metadata must be an array."));
            }

            for (id item in annotations)
            {
                auto annotation = parseLogicalBindingAnnotation(dictionary(item), rachisBindings);
                if (!annotation)
                {
                    return std::unexpected(std::move(annotation.error()));
                }
                rachisBindings.push_back(std::move(*annotation));
            }
            return rachisBindings;
        }

        [[nodiscard]] std::optional<rachis::detail::StageComponentType> stageComponentType(NSString* name)
        {
            const char* const utf8 = name != nil ? name.UTF8String : nullptr;
            if (utf8 == nullptr)
            {
                return std::nullopt;
            }
            const std::string_view text(utf8);
            for (auto type : {rachis::detail::StageComponentType::float16,
                              rachis::detail::StageComponentType::float32,
                              rachis::detail::StageComponentType::float64,
                              rachis::detail::StageComponentType::sint16,
                              rachis::detail::StageComponentType::sint32,
                              rachis::detail::StageComponentType::sint64,
                              rachis::detail::StageComponentType::uint16,
                              rachis::detail::StageComponentType::uint32,
                              rachis::detail::StageComponentType::uint64})
            {
                if (rachis::detail::stageComponentTypeName(type) == text)
                {
                    return type;
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] Result<std::vector<rachis::detail::StageInterfaceVariable>> parseStageInterfaceVariables(id value)
        {
            NSArray* const entries = array(value);
            if (entries == nil)
            {
                return std::unexpected(manifestError("Metal rachisStageInterface inputs and outputs must be arrays."));
            }
            std::vector<rachis::detail::StageInterfaceVariable> variables;
            variables.reserve(entries.count);
            for (id item in entries)
            {
                NSDictionary* const entry = dictionary(item);
                const auto location = uintValue(entry[@"location"]);
                const auto component = uintValue(entry[@"component"]);
                const auto componentCount = uintValue(entry[@"componentCount"]);
                const auto componentType = stageComponentType(string(entry[@"componentType"]));
                if (!location || !component || !componentCount || !componentType)
                {
                    return std::unexpected(manifestError("Metal rachisStageInterface entries require a location, "
                                                         "component, componentType and componentCount."));
                }
                const rachis::detail::StageInterfaceVariable variable{
                    .location = *location,
                    .firstComponent = *component,
                    .componentCount = *componentCount,
                    .componentType = *componentType,
                };
                if (!rachis::detail::isValidStageInterfaceVariable(variable))
                {
                    return std::unexpected(
                        manifestError("Metal rachisStageInterface entries must fit in one four-component location."));
                }
                variables.push_back(variable);
            }
            return variables;
        }

        [[nodiscard]] Result<std::optional<MetalReflectedStageInterface>> parseStageInterfaceAnnotation(
            NSDictionary* root)
        {
            id const value = root[@"rachisStageInterface"];
            if (value == nil)
            {
                return std::optional<MetalReflectedStageInterface>{};
            }
            NSDictionary* const node = dictionary(value);
            if (node == nil)
            {
                return std::unexpected(manifestError("Metal rachisStageInterface metadata must be an object."));
            }
            auto inputs = parseStageInterfaceVariables(node[@"inputs"]);
            if (!inputs)
            {
                return std::unexpected(std::move(inputs.error()));
            }
            auto outputs = parseStageInterfaceVariables(node[@"outputs"]);
            if (!outputs)
            {
                return std::unexpected(std::move(outputs.error()));
            }
            return MetalReflectedStageInterface{.inputs = std::move(*inputs), .outputs = std::move(*outputs)};
        }

        [[nodiscard]] Result<NSDictionary*> findReflectedParameter(NSDictionary* root, NSString* name)
        {
            NSDictionary* parameter = nil;

            for (id global in array(root[@"parameters"]))
            {
                NSDictionary* const candidate = dictionary(global);
                if ([string(candidate[@"name"]) isEqualToString:name])
                {
                    if (parameter != nil)
                    {
                        return std::unexpected(manifestError("Metal reflection contains duplicate resource names."));
                    }
                    parameter = candidate;
                }
            }
            if (name == nil || parameter == nil)
            {
                return std::unexpected(manifestError("Metal reflection is missing a resource type."));
            }
            return parameter;
        }

        struct ReflectedResourceElement final
        {
            NSDictionary* type = nil;
            std::uint32_t count = 1;
        };

        [[nodiscard]] Result<ReflectedResourceElement> unwrapReflectedResourceArrays(NSDictionary* parameter)
        {
            NSDictionary* type = dictionary(parameter[@"type"]);
            std::uint32_t count = 1;
            while ([string(type[@"kind"]) isEqualToString:@"array"])
            {
                const auto elements = uintValue(type[@"elementCount"]);
                if (!elements || *elements == 0 || *elements > 128 / count)
                {
                    return std::unexpected(manifestError("Metal resource arrays must fit the argument table."));
                }
                count *= *elements;
                type = dictionary(type[@"elementType"]);
            }
            return ReflectedResourceElement{.type = type, .count = count};
        }

        [[nodiscard]] NSString* classifyReflectedResource(NSDictionary* type, MetalReflectedResource& resource)
        {
            NSString* nativeKind = nil;
            if ([string(type[@"kind"]) isEqualToString:@"constantBuffer"])
            {
                resource.type = ShaderResourceType::uniformBuffer;
                nativeKind = @"constantBuffer";
            }
            else if ([string(type[@"kind"]) isEqualToString:@"samplerState"])
            {
                resource.type = ShaderResourceType::sampler;
                nativeKind = @"samplerState";
            }
            else if ([string(type[@"kind"]) isEqualToString:@"resource"])
            {
                const bool writable = [string(type[@"access"]) isEqualToString:@"readWrite"];
                if ([string(type[@"baseShape"]) hasPrefix:@"texture"])
                {
                    resource.type = writable ? ShaderResourceType::storageTexture : ShaderResourceType::sampledTexture;
                    nativeKind = @"shaderResource";
                }
                else if ([string(type[@"baseShape"]) isEqualToString:@"structuredBuffer"] ||
                         [string(type[@"baseShape"]) isEqualToString:@"byteAddressBuffer"])
                {
                    resource.type =
                        writable ? ShaderResourceType::storageBuffer : ShaderResourceType::readOnlyStorageBuffer;
                    nativeKind = @"constantBuffer";
                }
            }
            return nativeKind;
        }

        [[nodiscard]] Status recordReflectedPushConstant(NSDictionary* type,
                                                         const MetalReflectedResource& resource,
                                                         bool isLegacySpace4PushConstant,
                                                         MetalReflectedLayout& layout)
        {
            const auto size = uniformSize(dictionary(type[@"elementType"]));
            if (resource.type != ShaderResourceType::uniformBuffer ||
                (isLegacySpace4PushConstant && resource.metalIndex != reservedPushConstantBufferIndex) ||
                resource.metalIndex >= 31 || resource.count != 1 || !size || *size == 0 ||
                layout.pushConstantBytes != 0)
            {
                return std::unexpected(manifestError("Metal push-constant reflection has an invalid layout."));
            }
            layout.pushConstantBytes = *size;
            layout.pushConstantMetalIndex = resource.metalIndex;
            return {};
        }

        [[nodiscard]] Status recordReflectedResource(MetalReflectedResource& resource,
                                                     const LogicalBindingAnnotation* annotation,
                                                     std::uint32_t space,
                                                     MetalReflectedLayout& layout)
        {
            if (annotation != nullptr)
            {
                if (space >= maxBindingSets)
                {
                    return std::unexpected(
                        manifestError("Metal resource register spaces exceed the portable four-set limit."));
                }
                resource.set = annotation->set;
                resource.binding = annotation->binding;
            }
            else if (space != 0)
            {
                return std::unexpected(
                    manifestError("Metal resource register spaces require an explicit native mapping."));
            }
            if (std::ranges::any_of(layout.resources, [&](const auto& other) { return other.name == resource.name; }))
            {
                return std::unexpected(manifestError("Metal reflection contains a duplicate active resource."));
            }
            layout.resources.push_back(std::move(resource));
            return {};
        }

        [[nodiscard]] Status reflectActiveBinding(NSDictionary* root,
                                                  NSDictionary* active,
                                                  const std::vector<LogicalBindingAnnotation>& rachisBindings,
                                                  MetalReflectedLayout& layout)
        {
            NSString* const name = string(active[@"name"]);
            if (name == nil)
            {
                return std::unexpected(manifestError("Metal reflection is missing a resource name."));
            }
            auto parameter = findReflectedParameter(root, name);
            if (!parameter)
            {
                return std::unexpected(std::move(parameter.error()));
            }
            auto element = unwrapReflectedResourceArrays(*parameter);
            if (!element)
            {
                return std::unexpected(std::move(element.error()));
            }
            NSDictionary* const type = element->type;
            MetalReflectedResource resource;
            const char* const resourceName = name.UTF8String;
            if (resourceName == nullptr)
            {
                return std::unexpected(manifestError("Metal resource names must be valid UTF-8."));
            }
            resource.name = resourceName;
            resource.count = element->count;
            NSString* const nativeKind = classifyReflectedResource(type, resource);
            NSDictionary* const binding = bindingOfKind(active, nativeKind);
            if (nativeKind == nil || binding == nil)
            {
                return std::unexpected(manifestError("Metal reflection contains an unsupported resource mapping."));
            }
            id const used = binding[@"used"];
            if (used != nil &&
                (![used isKindOfClass:[NSNumber class]] || ([used doubleValue] != 0 && [used doubleValue] != 1)))
            {
                return std::unexpected(manifestError("Metal resource usage must be a boolean flag."));
            }

            if (used != nil && ![used boolValue])
            {
                return {};
            }
            const auto index = uintValue(binding[@"index"]);
            const auto space =
                binding[@"space"] == nil ? std::optional<std::uint32_t>{0} : uintValue(binding[@"space"]);
            if (!index || !space)
            {
                return std::unexpected(manifestError("Metal binding indices must be unsigned 32-bit integers."));
            }
            resource.metalIndex = *index;
            const auto foundAnnotation = std::ranges::find_if(
                rachisBindings, [&](const auto& candidate) { return candidate.name == resource.name; });
            const LogicalBindingAnnotation* const annotation =
                foundAnnotation != rachisBindings.end() ? &*foundAnnotation : nullptr;
            const bool isAnnotatedPushConstant = annotation != nullptr && annotation->pushConstant;
            const bool isLegacySpace4PushConstant =
                annotation == nullptr && resource.type == ShaderResourceType::uniformBuffer && *space == 4;
            auto recorded = isAnnotatedPushConstant || isLegacySpace4PushConstant
                                ? recordReflectedPushConstant(type, resource, isLegacySpace4PushConstant, layout)
                                : recordReflectedResource(resource, annotation, *space, layout);
            if (!recorded)
            {
                return recorded;
            }
            if (!collectFields(type, 0, layout.uniformFields))
            {
                return std::unexpected(manifestError("Metal reflection contains an invalid uniform field layout."));
            }
            return {};
        }

    } // namespace

    Result<MetalReflectedLayout> parseSlangReflectionJson(std::string_view json,
                                                          std::string_view entryPoint,
                                                          ShaderStage stage)
    {
        @autoreleasepool
        {
            NSData* const data = [NSData dataWithBytes:json.data() length:json.size()];
            NSDictionary* const root = dictionary([NSJSONSerialization JSONObjectWithData:data
                                                                                  options:0
                                                                                    error:nullptr]);
            NSString* const requested = [[NSString alloc] initWithBytes:entryPoint.data()
                                                                 length:entryPoint.size()
                                                               encoding:NSUTF8StringEncoding];
            if (root == nil || requested.length == 0 || array(root[@"entryPoints"]) == nil)
            {
                return std::unexpected(manifestError("Metal reflection requires valid JSON and an entry point."));
            }
            auto found = findReflectedEntryPoint(root, requested);
            if (!found)
            {
                return std::unexpected(std::move(found.error()));
            }
            NSDictionary* const entry = *found;

            MetalReflectedLayout layout;
            layout.entryPoint = std::string(entryPoint);
            layout.stage = reflectedShaderStage(string(entry[@"stage"]));
            if (layout.stage == ShaderStage::none || layout.stage != stage)
            {
                return std::unexpected(
                    manifestError("The reflected Metal entry point stage does not match the request."));
            }
            if (layout.stage == ShaderStage::compute || layout.stage == ShaderStage::task ||
                layout.stage == ShaderStage::mesh)
            {
                auto workgroupSize = reflectedWorkgroupSize(entry);
                if (!workgroupSize)
                {
                    return std::unexpected(std::move(workgroupSize.error()));
                }
                layout.workgroupSize = *workgroupSize;
            }
            if (entry[@"bindings"] != nil && array(entry[@"bindings"]) == nil)
            {
                return std::unexpected(manifestError("Metal entry point bindings must be an array."));
            }
            auto rachisBindings = parseLogicalBindingAnnotations(root);
            if (!rachisBindings)
            {
                return std::unexpected(std::move(rachisBindings.error()));
            }
            auto stageInterface = parseStageInterfaceAnnotation(root);
            if (!stageInterface)
            {
                return std::unexpected(std::move(stageInterface.error()));
            }
            layout.stageInterface = std::move(*stageInterface);

            for (id item in array(entry[@"bindings"]))
            {
                if (auto reflected = reflectActiveBinding(root, dictionary(item), *rachisBindings, layout); !reflected)
                {
                    return std::unexpected(std::move(reflected.error()));
                }
            }
            return layout;
        }
    }

    namespace
    {

        [[nodiscard]] Status validateReflectedStageAndLimits(const MetalReflectedLayout& layout,
                                                             const ShaderInterfaceDesc& declared)
        {
            if (layout.stage == ShaderStage::none)
            {
                return std::unexpected(manifestError("Reflected Metal shader stage is missing."));
            }
            if (declared.workgroupSize)
            {
                if (!layout.workgroupSize || declared.workgroupSize->x != layout.workgroupSize->x ||
                    declared.workgroupSize->y != layout.workgroupSize->y ||
                    declared.workgroupSize->z != layout.workgroupSize->z)
                {
                    return std::unexpected(manifestError("Declared workgroup size does not match Metal reflection."));
                }
            }
            if (layout.pushConstantBytes > maxPushConstantBytes)
            {
                return std::unexpected(manifestError("Metal push constants exceed the supported size."));
            }
            if (layout.pushConstantBytes > 0 && layout.pushConstantMetalIndex >= 31)
            {
                return std::unexpected(
                    manifestError("Reflected Metal push-constant buffer exceeds the argument table."));
            }
            for (const auto& host : declared.bindings)
            {
                if (host.set >= maxBindingSets)
                {
                    return std::unexpected(
                        manifestError("Declared shader binding set exceeds the portable four-set limit."));
                }
            }
            return {};
        }

        [[nodiscard]] Status validateDeclaredPushConstantCoverage(const MetalReflectedLayout& layout,
                                                                  const ShaderInterfaceDesc& declared)
        {
            std::array<bool, maxPushConstantBytes> covered{};
            for (const auto& range : declared.pushConstants)
            {
                if (range.size == 0 || range.offset % 4 != 0 || range.size % 4 != 0 ||
                    range.offset > layout.pushConstantBytes || range.size > layout.pushConstantBytes - range.offset)
                {
                    return std::unexpected(
                        manifestError("Declared Metal push constants are misaligned or outside reflection."));
                }
                for (std::uint32_t index = range.offset; index < range.offset + range.size; ++index)
                {
                    if (covered[index])
                    {
                        return std::unexpected(manifestError("Declared Metal push constants overlap."));
                    }
                    covered[index] = true;
                }
            }
            if (!std::ranges::all_of(std::span{covered}.first(layout.pushConstantBytes),
                                     [](bool value) { return value; }))
            {
                return std::unexpected(manifestError("Declared Metal push constants omit required bytes."));
            }
            return {};
        }

        using MetalArgumentTableOccupancy = std::array<std::array<bool, 128>, 3>;

        [[nodiscard]] Status claimMetalArgumentRange(const MetalReflectedResource& resource,
                                                     MetalArgumentTableOccupancy& occupied)
        {
            std::size_t kind = 0;
            if (resource.type == ShaderResourceType::sampler)
            {
                kind = 2;
            }
            else if (resource.type == ShaderResourceType::sampledTexture ||
                     resource.type == ShaderResourceType::storageTexture)
            {
                kind = 1;
            }
            constexpr std::array<std::uint32_t, 3> limits{31, 128, 16};
            const auto limit = limits[kind];
            if (resource.metalIndex >= limit || resource.count == 0 || resource.count > limit - resource.metalIndex)
            {
                return std::unexpected(manifestError("Reflected Metal resource range exceeds the argument table."));
            }
            for (std::uint32_t index = resource.metalIndex; index < resource.metalIndex + resource.count; ++index)
            {
                if (occupied[kind][index])
                {
                    return std::unexpected(manifestError("Reflected Metal argument ranges collide."));
                }
                occupied[kind][index] = true;
            }
            return {};
        }

        [[nodiscard]] Result<std::optional<std::size_t>> findDeclaredBindingInReflectedSet(
            const MetalReflectedResource& resource, std::uint32_t expectedBinding, const ShaderInterfaceDesc& declared)
        {
            std::optional<std::size_t> matchedIndex{};
            if (!resource.set.has_value())
            {
                return matchedIndex;
            }
            for (std::size_t index = 0; index < declared.bindings.size(); ++index)
            {
                const auto& host = declared.bindings[index];
                if (host.set == *resource.set && host.binding == expectedBinding && host.type == resource.type &&
                    host.count == resource.count)
                {
                    if (matchedIndex.has_value())
                    {
                        return std::unexpected(
                            manifestError("Declared shader interface contains duplicate resource bindings."));
                    }
                    matchedIndex = index;
                }
            }
            return matchedIndex;
        }

        [[nodiscard]] Result<std::optional<std::size_t>> findDeclaredBindingInAnySet(
            const MetalReflectedResource& resource, std::uint32_t expectedBinding, const ShaderInterfaceDesc& declared)
        {
            std::optional<std::size_t> matchedIndex{};
            for (std::size_t index = 0; index < declared.bindings.size(); ++index)
            {
                const auto& host = declared.bindings[index];
                if (host.binding == expectedBinding && host.type == resource.type && host.count == resource.count)
                {
                    if (matchedIndex.has_value())
                    {
                        return std::unexpected(
                            manifestError("A Metal resource maps to more than one host binding set."));
                    }
                    matchedIndex = index;
                }
            }
            return matchedIndex;
        }

        [[nodiscard]] Result<std::size_t> matchDeclaredHostBinding(const MetalReflectedResource& resource,
                                                                   const ShaderInterfaceDesc& declared,
                                                                   const std::vector<bool>& claimedHostBinding)
        {
            const std::uint32_t expectedBinding = resource.binding.value_or(resource.metalIndex);
            auto matchedIndex = findDeclaredBindingInReflectedSet(resource, expectedBinding, declared);
            if (!matchedIndex)
            {
                return std::unexpected(std::move(matchedIndex.error()));
            }
            if (!matchedIndex->has_value() && !resource.set.has_value())
            {
                matchedIndex = findDeclaredBindingInAnySet(resource, expectedBinding, declared);
                if (!matchedIndex)
                {
                    return std::unexpected(std::move(matchedIndex.error()));
                }
            }
            if (!matchedIndex->has_value() || claimedHostBinding[**matchedIndex])
            {
                return std::unexpected(
                    manifestError("Host shader interface is missing a resource required by Metal reflection."));
            }
            return **matchedIndex;
        }

    } // namespace

    Result<MetalResolvedStageBindings> resolveShaderInterface(const MetalReflectedLayout& layout,
                                                              const ShaderInterfaceDesc& declared)
    {
        if (auto valid = validateReflectedStageAndLimits(layout, declared); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        if (auto valid = validateDeclaredPushConstantCoverage(layout, declared); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }

        MetalResolvedStageBindings resolved{};
        resolved.usesPushConstants = layout.pushConstantBytes > 0;
        resolved.pushConstantBytes = layout.pushConstantBytes;
        resolved.pushConstantMetalIndex = layout.pushConstantMetalIndex;
        resolved.resources.reserve(layout.resources.size());

        MetalArgumentTableOccupancy occupied{};
        if (layout.pushConstantBytes > 0)
        {
            occupied[0][layout.pushConstantMetalIndex] = true;
        }

        std::vector<bool> claimedHostBinding(declared.bindings.size(), false);

        for (const auto& resource : layout.resources)
        {
            if (auto claimed = claimMetalArgumentRange(resource, occupied); !claimed)
            {
                return std::unexpected(std::move(claimed.error()));
            }
            auto matchedIndex = matchDeclaredHostBinding(resource, declared, claimedHostBinding);
            if (!matchedIndex)
            {
                return std::unexpected(std::move(matchedIndex.error()));
            }
            claimedHostBinding[*matchedIndex] = true;
            const auto& host = declared.bindings[*matchedIndex];
            resolved.resources.push_back(MetalResolvedResourceBinding{
                .set = host.set,
                .binding = host.binding,
                .type = resource.type,
                .count = resource.count,
                .metalIndex = resource.metalIndex,
            });
        }
        return resolved;
    }

    Status validateShaderInterface(const MetalReflectedLayout& layout, const ShaderInterfaceDesc& declared)
    {
        auto resolved = resolveShaderInterface(layout, declared);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        return {};
    }

} // namespace rachis::metal
