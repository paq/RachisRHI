#pragma once

#include "Rachis/Binding.h"
#include "Rachis/Pipeline.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rachis::d3d12
{

    inline constexpr std::uint32_t maxRootSignatureDwords = 64;
    inline constexpr std::uint32_t rootDescriptorDwordCost = 2;
    inline constexpr std::uint32_t descriptorTableDwordCost = 1;
    inline constexpr std::uint32_t unpromotedRootParameter = UINT32_MAX;

    enum class RootDescriptorKind : std::uint8_t
    {
        cbv = 0,
        srv = 1,
        uav = 2,
    };

    struct RootSignatureSlotEntryInput final
    {
        BindingLayoutEntry entry{};
        std::uint32_t descriptorOffset = 0;
    };

    struct RootSignatureSlotInput final
    {
        std::span<const RootSignatureSlotEntryInput> entries{};
    };

    struct PromotedDynamicRootDescriptor final
    {
        std::uint32_t setIndex = 0;
        std::uint32_t dynamicIndex = 0;
        std::uint32_t entryIndex = 0;
        std::uint32_t shaderRegister = 0;
        std::uint32_t rootParameter = unpromotedRootParameter;
        RootDescriptorKind kind = RootDescriptorKind::cbv;
        ShaderStage visibility = ShaderStage::none;
    };

    struct RootSignatureSlotPlan final
    {
        std::array<std::uint32_t, maxDynamicBufferOffsetsPerBindingSet> dynamicRootParameters{};
        std::array<RootDescriptorKind, maxDynamicBufferOffsetsPerBindingSet> dynamicRootKinds{};
        std::array<std::uint32_t, maxDynamicBufferOffsetsPerBindingSet> dynamicTableOffsets{};
        std::vector<std::uint8_t> entryPromoted{};
        std::uint32_t resourceRootParameter = unpromotedRootParameter;
        std::uint32_t samplerRootParameter = unpromotedRootParameter;
        std::uint32_t dynamicTableRootParameter = unpromotedRootParameter;
        std::uint32_t dynamicBufferCount = 0;
        std::uint32_t promotedDynamicBufferCount = 0;
        std::uint32_t dynamicTableDescriptorCount = 0;
        bool hasResourceDescriptorTable = false;
        bool hasSamplerDescriptorTable = false;
        bool hasDynamicDescriptorTable = false;
        bool hasStaticResourceDescriptors = false;

        constexpr RootSignatureSlotPlan()
        {
            dynamicRootParameters.fill(unpromotedRootParameter);
            dynamicRootKinds.fill(RootDescriptorKind::cbv);
            dynamicTableOffsets.fill(unpromotedRootParameter);
        }

        [[nodiscard]] constexpr bool allDynamicBuffersPromoted() const noexcept
        {
            return promotedDynamicBufferCount == dynamicBufferCount;
        }

        [[nodiscard]] constexpr bool resourceTableIsPerBind() const noexcept
        {
            return hasResourceDescriptorTable && !hasStaticResourceDescriptors;
        }
    };

    struct RootSignaturePlan final
    {
        std::vector<RootSignatureSlotPlan> slots{};
        std::vector<PromotedDynamicRootDescriptor> promotedRootDescriptors{};
        std::uint32_t pushConstantRootParameter = unpromotedRootParameter;
        std::uint32_t pushConstantDwords = 0;
        std::uint32_t drawIndexRootParameter = unpromotedRootParameter;
        std::uint32_t drawIndexDwords = 0;
        std::uint32_t totalRootParameters = 0;
        std::uint32_t totalRootDwords = 0;
    };

    [[nodiscard]] constexpr bool isDynamicBufferBindingType(BindingType type) noexcept
    {
        return type == BindingType::dynamicUniformBuffer || type == BindingType::dynamicReadOnlyStorageBuffer ||
               type == BindingType::dynamicStorageBuffer;
    }

    [[nodiscard]] constexpr RootDescriptorKind rootDescriptorKindForBindingType(BindingType type) noexcept
    {
        switch (type)
        {
        case BindingType::dynamicUniformBuffer:
        case BindingType::uniformBuffer:
            return RootDescriptorKind::cbv;
        case BindingType::dynamicReadOnlyStorageBuffer:
        case BindingType::readOnlyStorageBuffer:
            return RootDescriptorKind::srv;
        case BindingType::dynamicStorageBuffer:
        case BindingType::storageBuffer:
        default:
            return RootDescriptorKind::uav;
        }
    }

    [[nodiscard]] constexpr bool isPromotableDynamicRootBuffer(const BindingLayoutEntry& entry) noexcept
    {
        return entry.type == BindingType::dynamicUniformBuffer && entry.count == 1 && entry.flags == BindingFlag::none;
    }

    namespace root_signature_detail
    {

        struct BindingSetSummary final
        {
            std::uint32_t eligibleDynamicCount = 0;
            bool hasIneligibleDynamic = false;
            bool hasStaticResource = false;
            bool hasDynamic = false;
            bool hasSamplers = false;
        };

        [[nodiscard]] constexpr bool isEligibleForRootDescriptor(const BindingLayoutEntry& entry,
                                                                 std::uint32_t dynamicIndex) noexcept
        {
            return isPromotableDynamicRootBuffer(entry) && dynamicIndex < maxDynamicBufferOffsetsPerBindingSet;
        }

        inline void recordDynamicRootKinds(RootSignatureSlotPlan& slotPlan,
                                           std::uint32_t firstDynamicIndex,
                                           const BindingLayoutEntry& entry) noexcept
        {
            const auto kind = rootDescriptorKindForBindingType(entry.type);
            for (std::uint32_t element = 0;
                 element < entry.count && (firstDynamicIndex + element) < maxDynamicBufferOffsetsPerBindingSet;
                 ++element)
            {
                slotPlan.dynamicRootKinds[firstDynamicIndex + element] = kind;
            }
        }

        [[nodiscard]] inline BindingSetSummary initializeSlotPlan(std::span<const RootSignatureSlotEntryInput> entries,
                                                                  RootSignatureSlotPlan& slotPlan)
        {
            BindingSetSummary summary;
            slotPlan.entryPromoted.assign(entries.size(), 0U);

            std::uint32_t dynamicCursor = 0;
            for (const auto& input : entries)
            {
                const auto& item = input.entry;
                if (item.type == BindingType::sampler)
                {
                    summary.hasSamplers = true;
                    continue;
                }

                if (!isDynamicBufferBindingType(item.type))
                {
                    summary.hasStaticResource = true;
                    continue;
                }

                summary.hasDynamic = true;
                recordDynamicRootKinds(slotPlan, dynamicCursor, item);
                if (isEligibleForRootDescriptor(item, dynamicCursor))
                {
                    ++summary.eligibleDynamicCount;
                }
                else
                {
                    summary.hasIneligibleDynamic = true;
                }
                dynamicCursor += item.count;
            }

            slotPlan.dynamicBufferCount = dynamicCursor;
            slotPlan.hasSamplerDescriptorTable = summary.hasSamplers;
            slotPlan.hasStaticResourceDescriptors = summary.hasStaticResource;
            slotPlan.hasResourceDescriptorTable = summary.hasStaticResource || summary.hasDynamic;
            slotPlan.hasDynamicDescriptorTable = summary.hasStaticResource && summary.hasDynamic;
            return summary;
        }

        [[nodiscard]] constexpr std::uint32_t descriptorTableDwords(const BindingSetSummary& summary) noexcept
        {
            std::uint32_t dwords = 0;
            if (summary.hasStaticResource)
            {
                dwords += descriptorTableDwordCost;
            }
            if (summary.hasDynamic)
            {
                dwords += descriptorTableDwordCost;
            }
            if (summary.hasSamplers)
            {
                dwords += descriptorTableDwordCost;
            }
            return dwords;
        }

        [[nodiscard]] constexpr std::uint32_t fullSetPromotionDwordDelta(const BindingSetSummary& summary) noexcept
        {
            return summary.hasIneligibleDynamic
                       ? (summary.eligibleDynamicCount * rootDescriptorDwordCost)
                       : (summary.eligibleDynamicCount * rootDescriptorDwordCost - descriptorTableDwordCost);
        }

        [[nodiscard]] constexpr PromotedDynamicRootDescriptor makePromotedRootDescriptor(
            std::size_t setIndex,
            std::size_t entryIndex,
            std::uint32_t dynamicIndex,
            const BindingLayoutEntry& entry) noexcept
        {
            return PromotedDynamicRootDescriptor{
                .setIndex = static_cast<std::uint32_t>(setIndex),
                .dynamicIndex = dynamicIndex,
                .entryIndex = static_cast<std::uint32_t>(entryIndex),
                .shaderRegister = entry.binding,
                .rootParameter = unpromotedRootParameter,
                .kind = rootDescriptorKindForBindingType(entry.type),
                .visibility = entry.visibility,
            };
        }

        [[nodiscard]] inline std::uint32_t promoteBindingSetDynamicBuffers(
            RootSignaturePlan& plan,
            std::size_t setIndex,
            std::span<const RootSignatureSlotEntryInput> entries,
            const BindingSetSummary& summary,
            std::uint32_t totalDwords)
        {
            if (summary.eligibleDynamicCount == 0 || totalDwords > maxRootSignatureDwords)
            {
                return totalDwords;
            }

            auto& slotPlan = plan.slots[setIndex];
            const std::uint32_t fullSetDelta = fullSetPromotionDwordDelta(summary);
            const bool promoteFullSet = fullSetDelta <= (maxRootSignatureDwords - totalDwords);

            std::uint32_t dynamicCursor = 0;
            for (std::size_t entryIndex = 0; entryIndex < entries.size(); ++entryIndex)
            {
                const auto& item = entries[entryIndex].entry;
                if (!isDynamicBufferBindingType(item.type))
                {
                    continue;
                }

                const bool fitsSingleRootDescriptor = rootDescriptorDwordCost <= (maxRootSignatureDwords - totalDwords);
                if (isEligibleForRootDescriptor(item, dynamicCursor) && (promoteFullSet || fitsSingleRootDescriptor))
                {
                    slotPlan.entryPromoted[entryIndex] = 1U;
                    ++slotPlan.promotedDynamicBufferCount;
                    plan.promotedRootDescriptors.push_back(
                        makePromotedRootDescriptor(setIndex, entryIndex, dynamicCursor, item));
                    if (!promoteFullSet)
                    {
                        totalDwords += rootDescriptorDwordCost;
                    }
                }
                dynamicCursor += item.count;
            }

            if (promoteFullSet)
            {
                totalDwords += fullSetDelta;
                if (!summary.hasIneligibleDynamic)
                {
                    slotPlan.hasDynamicDescriptorTable = false;
                    slotPlan.hasResourceDescriptorTable = summary.hasStaticResource;
                }
            }
            return totalDwords;
        }

        inline void assignDynamicTableOffsets(RootSignatureSlotPlan& slotPlan,
                                              std::span<const RootSignatureSlotEntryInput> entries) noexcept
        {
            std::uint32_t dynamicCursor = 0;
            std::uint32_t tableCursor = 0;
            for (std::size_t entryIndex = 0; entryIndex < entries.size(); ++entryIndex)
            {
                const auto& item = entries[entryIndex].entry;
                if (!isDynamicBufferBindingType(item.type))
                {
                    continue;
                }
                if (slotPlan.entryPromoted[entryIndex] == 0)
                {
                    for (std::uint32_t element = 0;
                         element < item.count && (dynamicCursor + element) < maxDynamicBufferOffsetsPerBindingSet;
                         ++element)
                    {
                        slotPlan.dynamicTableOffsets[dynamicCursor + element] = tableCursor++;
                    }
                }
                dynamicCursor += item.count;
            }
            slotPlan.dynamicTableDescriptorCount = tableCursor;
        }

        inline void assignRootParameterIndices(RootSignaturePlan& plan) noexcept
        {
            std::uint32_t nextRootParameter = 0;
            if (plan.pushConstantDwords != 0)
            {
                plan.pushConstantRootParameter = nextRootParameter++;
            }
            if (plan.drawIndexDwords != 0)
            {
                plan.drawIndexRootParameter = nextRootParameter++;
            }
            for (auto& promoted : plan.promotedRootDescriptors)
            {
                promoted.rootParameter = nextRootParameter++;
                plan.slots[promoted.setIndex].dynamicRootParameters[promoted.dynamicIndex] = promoted.rootParameter;
            }
            for (auto& slotPlan : plan.slots)
            {
                if (slotPlan.hasResourceDescriptorTable)
                {
                    slotPlan.resourceRootParameter = nextRootParameter++;
                }
                if (slotPlan.hasSamplerDescriptorTable)
                {
                    slotPlan.samplerRootParameter = nextRootParameter++;
                }
                if (slotPlan.hasDynamicDescriptorTable)
                {
                    slotPlan.dynamicTableRootParameter = nextRootParameter++;
                }
                else if (slotPlan.dynamicTableDescriptorCount != 0)
                {
                    slotPlan.dynamicTableRootParameter = slotPlan.resourceRootParameter;
                }
            }
            plan.totalRootParameters = nextRootParameter;
        }

    } // namespace root_signature_detail

    [[nodiscard]] inline RootSignaturePlan planRootSignature(std::span<const RootSignatureSlotInput> slots,
                                                             std::uint32_t pushConstantBytes,
                                                             bool drawIndex = false)
    {
        RootSignaturePlan plan;
        plan.pushConstantDwords = (pushConstantBytes + 3U) / 4U;
        plan.drawIndexDwords = drawIndex ? 1U : 0U;
        plan.slots.resize(slots.size());

        std::vector<root_signature_detail::BindingSetSummary> summaries(slots.size());
        std::uint32_t totalDwords = plan.pushConstantDwords + plan.drawIndexDwords;
        std::size_t totalEligibleDynamic = 0;
        for (std::size_t setIndex = 0; setIndex < slots.size(); ++setIndex)
        {
            auto& summary = summaries[setIndex];
            summary = root_signature_detail::initializeSlotPlan(slots[setIndex].entries, plan.slots[setIndex]);
            totalEligibleDynamic += summary.eligibleDynamicCount;
            totalDwords += root_signature_detail::descriptorTableDwords(summary);
        }

        plan.promotedRootDescriptors.reserve(totalEligibleDynamic);
        for (std::size_t setIndex = 0; setIndex < slots.size(); ++setIndex)
        {
            totalDwords = root_signature_detail::promoteBindingSetDynamicBuffers(
                plan, setIndex, slots[setIndex].entries, summaries[setIndex], totalDwords);
            root_signature_detail::assignDynamicTableOffsets(plan.slots[setIndex], slots[setIndex].entries);
        }

        root_signature_detail::assignRootParameterIndices(plan);
        plan.totalRootDwords = totalDwords;
        return plan;
    }

} // namespace rachis::d3d12
