#pragma once

#include "Common/HandleTable.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanResources.h"
#include "Rachis/Binding.h"
#include "Rachis/Device.h"
#include "Rachis/Pipeline.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <variant>
#include <vector>
#include <volk.h>

namespace rachis::vulkan
{

    class VulkanAccelerationManager;

    struct VulkanBufferBindingLimits final
    {
        std::uint64_t minUniformBufferOffsetAlignment = 256;
        std::uint64_t minStorageBufferOffsetAlignment = 16;
        std::uint64_t maxUniformBufferRange = 65536;
        std::uint64_t maxStorageBufferRange = 1ULL << 30;
    };

    struct VulkanDescriptorLimits final
    {
        std::uint32_t maxPerStageDescriptorSamplers = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxPerStageDescriptorUniformBuffers = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxPerStageDescriptorStorageBuffers = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxPerStageDescriptorSampledImages = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxPerStageDescriptorStorageImages = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxPerStageDescriptorAccelerationStructures = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxPerStageResources = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxDescriptorSetSamplers = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxDescriptorSetUniformBuffers = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxDescriptorSetUniformBuffersDynamic = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxDescriptorSetStorageBuffers = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxDescriptorSetStorageBuffersDynamic = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxDescriptorSetSampledImages = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxDescriptorSetStorageImages = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxDescriptorSetAccelerationStructures = std::numeric_limits<std::uint32_t>::max();
    };

    struct VulkanDynamicBufferBinding final
    {
        BindingType type = BindingType::dynamicUniformBuffer;
        std::uint64_t offset = 0;
        std::uint64_t range = 0;
        std::uint64_t bufferSize = 0;
    };

    struct VulkanNativeBindingLayout final
    {
        std::shared_ptr<void> owner;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        std::vector<BindingLayoutEntry> entries;
        std::uint32_t dynamicBufferCount = 0;
    };

    struct VulkanNativeBindingSet final
    {
        std::shared_ptr<void> owner;
        VulkanCommandUse commandUse = {};
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        std::uint32_t dynamicBufferCount = 0;
        std::span<const VulkanDynamicBufferBinding> dynamicBuffers = {};
        std::span<const VulkanAccelerationStructureUse> accelerationStructureUses = {};
        std::span<const std::shared_ptr<VulkanSwapchainImageLease>> swapchainLeases = {};
    };

    struct VulkanNativePipelineLayout final
    {
        std::shared_ptr<void> owner = {};
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        std::uint32_t pushConstantBytes = 0;
        std::vector<PushConstantRange> pushConstantRanges = {};
        std::vector<PushConstantRange> nativePushConstantRanges = {};
        std::vector<VulkanNativeBindingLayout> sets = {};
        bool drawIndex = false;
        std::uint64_t fragmentOutputCapacity = std::numeric_limits<std::uint64_t>::max();
    };

    class VulkanBindingManager final
    {
    public:
        VulkanBindingManager() = default;
        ~VulkanBindingManager();

        VulkanBindingManager(const VulkanBindingManager&) = delete;
        VulkanBindingManager& operator=(const VulkanBindingManager&) = delete;
        VulkanBindingManager(VulkanBindingManager&&) = delete;
        VulkanBindingManager& operator=(VulkanBindingManager&&) = delete;

        void initialize(VkDevice device,
                        const VolkDeviceTable* table,
                        const DeviceCapabilities& capabilities,
                        const VulkanInstanceTable* instanceTable = nullptr,
                        VulkanBufferBindingLimits bufferLimits = {},
                        VulkanDescriptorLimits descriptorLimits = {},
                        VulkanDescriptorLimits updateAfterBindLimits = {}) noexcept;
        void setAccelerationManager(VulkanAccelerationManager* acceleration) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] Result<BindingLayoutHandle> createBindingLayout(const BindingLayoutDesc& desc);
        [[nodiscard]] Status destroyBindingLayout(BindingLayoutHandle layout);
        [[nodiscard]] Result<BindingPoolHandle> createBindingPool(const BindingPoolDesc& desc);
        [[nodiscard]] Status destroyBindingPool(BindingPoolHandle pool);
        [[nodiscard]] Result<BindingSetHandle> createBindingSet(VulkanResourceAllocator& resources,
                                                                const BindingSetDesc& desc);
        [[nodiscard]] Status updateBindingSet(VulkanResourceAllocator& resources,
                                              BindingSetHandle set,
                                              std::span<const BindingWrite> writes);
        [[nodiscard]] Status destroyBindingSet(BindingSetHandle set);
        [[nodiscard]] Result<PipelineLayoutHandle> createPipelineLayout(const PipelineLayoutDesc& desc);
        [[nodiscard]] Status destroyPipelineLayout(PipelineLayoutHandle pipelineLayout);
        [[nodiscard]] Result<VulkanNativeBindingLayout> nativeBindingLayout(BindingLayoutHandle layout) const;
        [[nodiscard]] Result<VulkanNativeBindingSet> nativeBindingSet(BindingSetHandle set) const;
        [[nodiscard]] Result<VulkanNativeBindingSet> retainBindingSetForCommand(BindingSetHandle set,
                                                                                bool retainResources) const;
        [[nodiscard]] Result<VulkanNativePipelineLayout> nativePipelineLayout(PipelineLayoutHandle layout) const;
        [[nodiscard]] BindingUpdateStatistics statistics() const noexcept;

    private:
        struct LayoutKeepAlive final
        {
            VkDevice device = VK_NULL_HANDLE;
            const VolkDeviceTable* table = nullptr;
            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            ~LayoutKeepAlive();
            LayoutKeepAlive() = default;
            LayoutKeepAlive(const LayoutKeepAlive&) = delete;
            LayoutKeepAlive& operator=(const LayoutKeepAlive&) = delete;
            LayoutKeepAlive(LayoutKeepAlive&&) = delete;
            LayoutKeepAlive& operator=(LayoutKeepAlive&&) = delete;
        };

        struct BindingLayoutRecord final
        {
            std::vector<BindingLayoutEntry> entries;
            std::uint32_t resourceDescriptorCount = 0;
            std::uint32_t samplerDescriptorCount = 0;
            std::uint32_t dynamicBufferCount = 0;
            std::optional<std::uint32_t> variableBinding;
            std::uint32_t variableDescriptorCapacity = 0;
            bool updateAfterBind = false;
            std::string label;
            std::shared_ptr<LayoutKeepAlive> keepAlive;
        };

        struct BindingPoolState;
        struct BindingSetVersion;

        struct BindingPoolSetLease final
        {
            BindingPoolSetLease() = default;
            BindingPoolSetLease(const BindingPoolSetLease&) = delete;
            BindingPoolSetLease& operator=(const BindingPoolSetLease&) = delete;
            BindingPoolSetLease(BindingPoolSetLease&&) = delete;
            BindingPoolSetLease& operator=(BindingPoolSetLease&&) = delete;
            std::shared_ptr<BindingPoolState> state;
            std::atomic<std::uint64_t> activeCommandRetentions{0};
            bool armed = false;
            ~BindingPoolSetLease();
        };

        struct BindingPoolVersionLease final
        {
            BindingPoolVersionLease() = default;
            BindingPoolVersionLease(const BindingPoolVersionLease&) = delete;
            BindingPoolVersionLease& operator=(const BindingPoolVersionLease&) = delete;
            BindingPoolVersionLease(BindingPoolVersionLease&&) = delete;
            BindingPoolVersionLease& operator=(BindingPoolVersionLease&&) = delete;
            std::shared_ptr<BindingPoolState> state;
            std::array<std::uint32_t, bindingTypeCount> counts{};
            VkDescriptorPool nativePool = VK_NULL_HANDLE;
            VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
            bool armed = false;
            ~BindingPoolVersionLease();
        };

        struct BindingPoolState final
        {
            BindingPoolState() = default;
            BindingPoolState(const BindingPoolState&) = delete;
            BindingPoolState& operator=(const BindingPoolState&) = delete;
            BindingPoolState(BindingPoolState&&) = delete;
            BindingPoolState& operator=(BindingPoolState&&) = delete;
            std::mutex mutex;
            VkDevice device = VK_NULL_HANDLE;
            const VolkDeviceTable* table = nullptr;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            VkDescriptorPool updateAfterBindPool = VK_NULL_HANDLE;
            std::vector<VkDescriptorPoolSize> nativeSizes;
            std::uint32_t nativeMaxSets = 0;
            std::uint32_t maxBindingSets = 0;
            std::uint32_t liveBindingSets = 0;
            std::array<std::uint32_t, bindingTypeCount> descriptorCapacity{};
            std::array<std::uint32_t, bindingTypeCount> liveDescriptors{};
            bool logicalAlive = true;
            std::string label;
            ~BindingPoolState();
        };

    public:
        struct BindingSlot final
        {
            BindingType type = BindingType::uniformBuffer;
            bool swapchainView = false;
            std::variant<std::monostate, BufferBinding, TextureViewHandle, SamplerHandle, AccelerationStructureHandle>
                resource = {};
            std::shared_ptr<void> keepAlive = {};
            VkBuffer buffer = VK_NULL_HANDLE;
            VkImageView imageView = VK_NULL_HANDLE;
            VkSampler sampler = VK_NULL_HANDLE;
            VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
            VulkanAccelerationStructureUse accelerationStructureUse = {};
            std::uint64_t range = 0;
            std::uint64_t bufferSize = 0;
        };

        struct SwapchainViewLeases final
        {
            std::vector<std::shared_ptr<VulkanSwapchainImageLease>> leases;
            std::vector<std::size_t> slotCounts;
        };

        struct StagedSlot final
        {
            std::size_t index = 0;
            std::size_t write = 0;
            BindingSlot slot;
            std::shared_ptr<VulkanSwapchainImageLease> replacedSwapchainLease;
        };

    private:
        struct BindingSetVersion final
        {
            BindingSetVersion() = default;
            BindingSetVersion(const BindingSetVersion&) = delete;
            BindingSetVersion& operator=(const BindingSetVersion&) = delete;
            BindingSetVersion(BindingSetVersion&&) = delete;
            BindingSetVersion& operator=(BindingSetVersion&&) = delete;
            ~BindingSetVersion();
            std::shared_ptr<BindingLayoutRecord> layout;
            std::shared_ptr<BindingPoolState> pool;
            std::shared_ptr<BindingPoolSetLease> poolSet;
            std::shared_ptr<std::vector<BindingSlot>> slots;
            std::vector<VulkanDynamicBufferBinding> dynamicBuffers;
            std::uint32_t variableDescriptorCount = 0;
            std::shared_ptr<BindingPoolVersionLease> versionLease;
            std::shared_ptr<const std::vector<VulkanAccelerationStructureUse>> accelerationStructureUses;
            std::shared_ptr<const SwapchainViewLeases> swapchainViews;
            std::vector<BindingSlot> superseded;
            std::shared_ptr<BindingSetVersion> next;
        };

        struct BindingSetRecord final
        {
            std::shared_ptr<BindingLayoutRecord> layout;
            std::shared_ptr<BindingPoolSetLease> poolSet;
            std::shared_ptr<BindingSetVersion> version;
            std::string label;
        };

        struct PipelineLayoutKeepAlive final
        {
            VkDevice device = VK_NULL_HANDLE;
            const VolkDeviceTable* table = nullptr;
            VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
            std::vector<std::shared_ptr<LayoutKeepAlive>> setLayouts;
            ~PipelineLayoutKeepAlive();
            PipelineLayoutKeepAlive() = default;
            PipelineLayoutKeepAlive(const PipelineLayoutKeepAlive&) = delete;
            PipelineLayoutKeepAlive& operator=(const PipelineLayoutKeepAlive&) = delete;
            PipelineLayoutKeepAlive(PipelineLayoutKeepAlive&&) = delete;
            PipelineLayoutKeepAlive& operator=(PipelineLayoutKeepAlive&&) = delete;
        };

        struct PipelineLayoutRecord final
        {
            std::vector<std::shared_ptr<BindingLayoutRecord>> sets;
            std::vector<PushConstantRange> pushConstantRanges;
            std::vector<PushConstantRange> nativePushConstantRanges;
            std::uint32_t pushConstantBytes = 0;
            bool drawIndex = false;
            std::uint64_t fragmentOutputCapacity = std::numeric_limits<std::uint64_t>::max();
            std::string label;
            std::shared_ptr<PipelineLayoutKeepAlive> keepAlive;
        };

        struct LayoutAndPool;

        [[nodiscard]] Result<std::shared_ptr<LayoutKeepAlive>> createNativeSetLayout(
            const BindingLayoutRecord& record) const;
        [[nodiscard]] Status createNativeDescriptorPool(BindingPoolState& state,
                                                        std::span<const VkDescriptorPoolSize> poolSizes,
                                                        std::uint32_t maxSets,
                                                        VkDescriptorPoolCreateFlags extraFlags,
                                                        VkDescriptorPool& target) const;
        [[nodiscard]] Result<VkDescriptorPool> nativePoolLocked(BindingPoolState& pool, bool updateAfterBind) const;
        [[nodiscard]] Result<LayoutAndPool> findLayoutAndPool(BindingLayoutHandle layout, BindingPoolHandle pool) const;
        [[nodiscard]] static std::shared_ptr<BindingSetVersion> makeUnallocatedVersion(
            std::shared_ptr<BindingLayoutRecord> layout,
            std::shared_ptr<BindingPoolState> pool,
            std::shared_ptr<BindingPoolSetLease> poolSet,
            std::vector<BindingSlot> slots,
            std::uint32_t variableDescriptorCount);
        [[nodiscard]] Status allocateNativeSetLocked(BindingSetVersion& version) const;
        [[nodiscard]] Status allocateVersionLocked(BindingSetVersion& version) const;
        [[nodiscard]] Status allocateFirstVersion(BindingSetVersion& version) const;
        [[nodiscard]] Result<BindingSetHandle> publishFirstVersion(std::string_view label,
                                                                   std::shared_ptr<BindingLayoutRecord> layout,
                                                                   std::shared_ptr<BindingPoolSetLease> poolSet,
                                                                   std::shared_ptr<BindingSetVersion> version);
        [[nodiscard]] Result<std::shared_ptr<BindingSetRecord>> findBindingSetRecord(BindingSetHandle set) const;
        [[nodiscard]] Status allocateReplacementVersion(const BindingSetRecord& current,
                                                        BindingSetVersion& version) const;
        [[nodiscard]] static VulkanNativeBindingSet makeNativeBindingSet(const BindingSetRecord& record);
        [[nodiscard]] Status publishReplacementVersion(BindingSetHandle set,
                                                       const std::shared_ptr<BindingSetRecord>& current,
                                                       std::shared_ptr<BindingSetVersion> version,
                                                       std::span<const BindingWrite> writes);
        [[nodiscard]] Status publishInPlaceUpdate(BindingSetHandle set,
                                                  const std::shared_ptr<BindingSetRecord>& current,
                                                  std::vector<StagedSlot> staged,
                                                  std::span<const BindingWrite> writes);
        [[nodiscard]] std::vector<BindingSlot> snapshotSlots(const BindingSetRecord& record) const;
        [[nodiscard]] Result<std::vector<StagedSlot>> stageSlots(VulkanResourceAllocator& resources,
                                                                 const BindingSetRecord& record,
                                                                 std::span<const BindingWrite> writes) const;
        [[nodiscard]] Result<std::vector<std::shared_ptr<BindingLayoutRecord>>> findPipelineSetLayouts(
            std::span<const BindingLayoutHandle> bindingLayouts) const;
        [[nodiscard]] Result<std::shared_ptr<PipelineLayoutKeepAlive>> createNativePipelineLayout(
            const PipelineLayoutRecord& record) const;

        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        const VulkanInstanceTable* instanceTable_ = nullptr;
        bool partiallyBoundSupported_ = false;
        bool variableDescriptorCountSupported_ = false;
        bool updateAfterBindSupported_ = false;
        bool accelerationStructures_ = false;
        ShaderStage enabledShaderStages_ = coreShaderStages;
        VulkanAccelerationManager* accelerationManager_ = nullptr;
        std::uint32_t maxVariableResourceDescriptors_ = 0;
        std::uint32_t maxVariableSamplerDescriptors_ = 0;
        std::uint32_t maxVariableBufferDescriptors_ = 0;
        VulkanBufferBindingLimits bufferLimits_{};
        VulkanDescriptorLimits descriptorLimits_{};
        VulkanDescriptorLimits updateAfterBindLimits_{};
        mutable std::shared_mutex mutex_{};
        BindingUpdateStatistics statistics_{};
        detail::HandleTable<std::shared_ptr<BindingLayoutRecord>, BindingLayoutHandle> layouts_{};
        detail::HandleTable<std::shared_ptr<BindingPoolState>, BindingPoolHandle> pools_{};
        detail::HandleTable<std::shared_ptr<BindingSetRecord>, BindingSetHandle> sets_{};
        detail::HandleTable<std::shared_ptr<PipelineLayoutRecord>, PipelineLayoutHandle> pipelineLayouts_{};
    };

} // namespace rachis::vulkan
