#include "Rachis/Pipeline.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace rachis
{
    namespace
    {

        constexpr std::uint32_t pushConstantWordBytes = sizeof(std::uint32_t);
        constexpr std::uint32_t pushConstantWordCount = maxPushConstantBytes / pushConstantWordBytes;
        constexpr std::uint32_t shaderStageBitCount = 14;

        [[nodiscard]] constexpr std::optional<BindingType> bindingTypeFor(ShaderResourceType type) noexcept
        {
            switch (type)
            {
            case ShaderResourceType::uniformBuffer:
                return BindingType::uniformBuffer;
            case ShaderResourceType::readOnlyStorageBuffer:
                return BindingType::readOnlyStorageBuffer;
            case ShaderResourceType::storageBuffer:
                return BindingType::storageBuffer;
            case ShaderResourceType::sampledTexture:
                return BindingType::sampledTexture;
            case ShaderResourceType::storageTexture:
                return BindingType::storageTexture;
            case ShaderResourceType::sampler:
                return BindingType::sampler;
            case ShaderResourceType::accelerationStructure:
                return BindingType::accelerationStructure;
            case ShaderResourceType::minMipFeedback:
                return BindingType::minMipFeedback;
            case ShaderResourceType::mipRegionUsedFeedback:
                return BindingType::mipRegionUsedFeedback;
            }
            return std::nullopt;
        }

        [[nodiscard]] constexpr std::string_view resourceTypeName(ShaderResourceType type) noexcept
        {
            switch (type)
            {
            case ShaderResourceType::uniformBuffer:
                return "uniform buffer";
            case ShaderResourceType::readOnlyStorageBuffer:
                return "read-only storage buffer";
            case ShaderResourceType::storageBuffer:
                return "storage buffer";
            case ShaderResourceType::sampledTexture:
                return "sampled texture";
            case ShaderResourceType::storageTexture:
                return "storage texture";
            case ShaderResourceType::sampler:
                return "sampler";
            case ShaderResourceType::accelerationStructure:
                return "acceleration structure";
            case ShaderResourceType::minMipFeedback:
                return "min-mip feedback texture";
            case ShaderResourceType::mipRegionUsedFeedback:
                return "mip-region-used feedback texture";
            }
            return "unknown resource";
        }

        [[nodiscard]] constexpr std::string_view stageName(ShaderStage stage) noexcept
        {
            switch (stage)
            {
            case ShaderStage::vertex:
                return "vertex";
            case ShaderStage::pixel:
                return "pixel";
            case ShaderStage::compute:
                return "compute";
            case ShaderStage::geometry:
                return "geometry";
            case ShaderStage::hull:
                return "hull";
            case ShaderStage::domain:
                return "domain";
            case ShaderStage::task:
                return "task";
            case ShaderStage::mesh:
                return "mesh";
            case ShaderStage::rayGeneration:
                return "ray-generation";
            case ShaderStage::anyHit:
                return "any-hit";
            case ShaderStage::closestHit:
                return "closest-hit";
            case ShaderStage::miss:
                return "miss";
            case ShaderStage::intersection:
                return "intersection";
            case ShaderStage::callable:
                return "callable";
            default:
                return "unknown";
            }
        }

        [[nodiscard]] std::string countText(std::uint32_t count)
        {
            return count == unboundedBindingCount ? std::string("an unbounded count")
                                                  : "count " + std::to_string(count);
        }

        struct DerivedBinding final
        {
            BindingLayoutEntry entry;
            ShaderResourceType type = ShaderResourceType::uniformBuffer;
            ShaderStage firstStage = ShaderStage::none;
        };

        [[nodiscard]] Error invalidInterface(std::string message)
        {
            return Error{ErrorCode::invalidArgument, 0, std::move(message)};
        }

        [[nodiscard]] Status addBinding(std::array<std::vector<DerivedBinding>, maxBindingSets>& sets,
                                        ShaderStage stage,
                                        const ShaderBindingRequirement& binding)
        {
            const auto bindingType = bindingTypeFor(binding.type);
            if (binding.set >= maxBindingSets || !bindingType.has_value() || binding.count == 0)
            {
                return std::unexpected(invalidInterface(
                    "A shader interface declares a binding with an invalid set, resource type, or count."));
            }
            auto& entries = sets[binding.set];
            const auto existing = std::ranges::find_if(
                entries, [&](const DerivedBinding& derived) { return derived.entry.binding == binding.binding; });
            if (existing == entries.end())
            {
                entries.push_back(DerivedBinding{
                    .entry =
                        BindingLayoutEntry{
                            .binding = binding.binding,
                            .type = *bindingType,
                            .visibility = stage,
                            .count = binding.count,
                            .flags = binding.count == unboundedBindingCount ? BindingFlag::variableDescriptorCount
                                                                            : BindingFlag::none,
                        },
                    .type = binding.type,
                    .firstStage = stage,
                });
                return {};
            }
            if (existing->type != binding.type || existing->entry.count != binding.count)
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "Set " + std::to_string(binding.set) + " binding " + std::to_string(binding.binding) + " is a " +
                        std::string(resourceTypeName(existing->type)) + " with " + countText(existing->entry.count) +
                        " in the " + std::string(stageName(existing->firstStage)) + " shader, but a " +
                        std::string(resourceTypeName(binding.type)) + " with " + countText(binding.count) + " in the " +
                        std::string(stageName(stage)) + " shader.",
                });
            }
            existing->entry.visibility = existing->entry.visibility | stage;
            return {};
        }

        [[nodiscard]] Status addPushConstants(std::array<ShaderStage, pushConstantWordCount>& wordStages,
                                              ShaderStage stage,
                                              const ShaderPushConstantRequirement& range)
        {
            if (range.size == 0 || range.offset % pushConstantWordBytes != 0 ||
                range.size % pushConstantWordBytes != 0 || range.offset > maxPushConstantBytes ||
                range.size > maxPushConstantBytes - range.offset)
            {
                return std::unexpected(invalidInterface(
                    "A shader interface declares a push-constant range that is empty, not 4-byte aligned, or past "
                    "128 bytes."));
            }
            const auto first = range.offset / pushConstantWordBytes;
            const auto last = (range.offset + range.size) / pushConstantWordBytes;
            for (auto word = first; word < last; ++word)
            {
                wordStages[word] = wordStages[word] | stage;
            }
            return {};
        }

        [[nodiscard]] std::vector<PushConstantRange> mergePushConstants(
            const std::array<ShaderStage, pushConstantWordCount>& wordStages)
        {
            std::vector<PushConstantRange> ranges;
            for (std::uint32_t bit = 0; bit < shaderStageBitCount; ++bit)
            {
                const auto stage = static_cast<ShaderStage>(1U << bit);
                std::uint32_t word = 0;
                while (word < pushConstantWordCount)
                {
                    if (!hasAny(wordStages[word], stage))
                    {
                        ++word;
                        continue;
                    }
                    const auto first = word;
                    while (word < pushConstantWordCount && hasAny(wordStages[word], stage))
                    {
                        ++word;
                    }
                    const auto offset = first * pushConstantWordBytes;
                    const auto size = (word - first) * pushConstantWordBytes;
                    const auto shared = std::ranges::find_if(ranges, [&](const PushConstantRange& range) {
                        return range.offset == offset && range.size == size;
                    });
                    if (shared != ranges.end())
                    {
                        shared->visibility = shared->visibility | stage;
                    }
                    else
                    {
                        ranges.push_back(PushConstantRange{.visibility = stage, .offset = offset, .size = size});
                    }
                }
            }
            std::ranges::sort(ranges, [](const PushConstantRange& left, const PushConstantRange& right) {
                return left.offset < right.offset || (left.offset == right.offset && left.size < right.size);
            });
            return ranges;
        }

    } // namespace

    Result<DerivedPipelineLayout> derivePipelineLayout(std::span<const ShaderInterfaceSnapshot> shaders)
    {
        std::array<std::vector<DerivedBinding>, maxBindingSets> sets{};
        std::array<ShaderStage, pushConstantWordCount> wordStages{};
        for (const auto& shader : shaders)
        {
            const auto stageBits = static_cast<std::underlying_type_t<ShaderStage>>(shader.stage);
            if (!std::has_single_bit(stageBits) || !hasAny(shader.stage, ShaderStage::all))
            {
                return std::unexpected(invalidInterface("Each shader interface must name exactly one shader stage."));
            }
            for (const auto& binding : shader.bindings)
            {
                if (auto added = addBinding(sets, shader.stage, binding); !added)
                {
                    return std::unexpected(std::move(added.error()));
                }
            }
            for (const auto& range : shader.pushConstants)
            {
                if (auto added = addPushConstants(wordStages, shader.stage, range); !added)
                {
                    return std::unexpected(std::move(added.error()));
                }
            }
        }

        DerivedPipelineLayout layout;
        std::size_t setCount = 0;
        for (std::size_t set = 0; set < sets.size(); ++set)
        {
            if (!sets[set].empty())
            {
                setCount = set + 1U;
            }
        }
        layout.bindingSets.resize(setCount);
        for (std::size_t set = 0; set < setCount; ++set)
        {
            auto& entries = layout.bindingSets[set];
            entries.reserve(sets[set].size());
            for (const auto& derived : sets[set])
            {
                entries.push_back(derived.entry);
            }
            std::ranges::sort(entries, {}, &BindingLayoutEntry::binding);
        }
        layout.pushConstantRanges = mergePushConstants(wordStages);
        return layout;
    }

} // namespace rachis
