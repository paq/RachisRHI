#pragma once

#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanWindowsProfile.h"
#include "Rachis/Device.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>
#include <volk.h>

namespace rachis::vulkan
{

    struct VulkanInstanceInfo final
    {
        VkInstance instance = VK_NULL_HANDLE;
        VulkanInstanceTable table{};
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        std::unique_ptr<VulkanDiagnosticSink> sink;
        VulkanInstanceReport report{};
        bool validationEnabled = false;
        bool validationRequested = false;
        bool validationMissing = false;
        bool debugUtils = false;

        VulkanInstanceInfo() = default;
        VulkanInstanceInfo(VulkanInstanceInfo&& other) noexcept;
        VulkanInstanceInfo& operator=(VulkanInstanceInfo&& other) noexcept;
        ~VulkanInstanceInfo();

        VulkanInstanceInfo(const VulkanInstanceInfo&) = delete;
        VulkanInstanceInfo& operator=(const VulkanInstanceInfo&) = delete;

        void reset() noexcept;
    };

    void destroyVulkanInstance(VulkanInstanceInfo& info) noexcept;

    struct VulkanQueueFamilies final
    {
        std::uint32_t graphics = VK_QUEUE_FAMILY_IGNORED;
        std::uint32_t compute = VK_QUEUE_FAMILY_IGNORED;
        std::uint32_t transfer = VK_QUEUE_FAMILY_IGNORED;
        std::uint32_t graphicsIndex = 0;
        std::uint32_t computeIndex = 0;
        std::uint32_t transferIndex = 0;
    };

    struct VulkanMultiviewStageSupport final
    {
        bool geometry = false;
        bool tessellation = false;
        bool mesh = false;
    };

    struct VulkanShaderFeatureSupport final
    {
        bool sampleRateShading = false;
        bool vertexPipelineStoresAndAtomics = false;
        bool shaderImageGatherExtended = false;
        bool shaderResourceMinLod = false;
        bool shaderFloat64 = false;
        bool storageBuffer16BitAccess = false;
        bool uniformAndStorageBuffer16BitAccess = false;
        bool storagePushConstant16 = false;
        bool storageInputOutput16 = false;
        bool shaderDrawParameters = false;
    };

    struct VulkanQueueFamilyTraits final
    {
        VkQueueFlags queueFlags = 0;
        std::uint32_t timestampValidBits = 0;
    };

    struct VulkanQueueFamilyTraitSet final
    {
        VulkanQueueFamilyTraits graphics{};
        VulkanQueueFamilyTraits compute{};
        VulkanQueueFamilyTraits transfer{};
    };

    struct VulkanQueueHandles final
    {
        VkQueue graphics = VK_NULL_HANDLE;
        VkQueue compute = VK_NULL_HANDLE;
        VkQueue copy = VK_NULL_HANDLE;
        VkQueue present = VK_NULL_HANDLE;
        VulkanQueueFamilies families{};
        VulkanPresentCompletionStrategy presentCompletion = VulkanPresentCompletionStrategy::imageLifetimeAssociation;
        bool deviceFaultEnabled = false;
        bool presentWaitEnabled = false;
        bool maintenance4Enabled = false;
        bool maintenance5Enabled = false;
        bool retireOnSubmissionFence = false;
        VulkanMultiviewStageSupport multiviewStagesEnabled{};
        VulkanShaderFeatureSupport shaderFeaturesEnabled{};
    };

    struct VulkanPhysicalDeviceRecord final
    {
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        AdapterInfo adapter{};
        std::array<std::uint8_t, VK_UUID_SIZE> deviceUuid{};
        std::array<std::uint8_t, VK_LUID_SIZE> deviceLuid{};
        bool deviceLuidValid = false;
        VulkanPhysicalDeviceReport report{};
        VulkanQueueFamilies queues{};
        VulkanQueueFamilyTraitSet queueTraits{};
        std::string deviceName;
        std::string driverId;
        bool moltenVk = false;
        VkPhysicalDeviceProperties properties{};
        VkPhysicalDeviceMemoryProperties memory{};
        VkResolveModeFlags depthResolveModes = 0;
        VkResolveModeFlags stencilResolveModes = 0;
        bool independentResolveNone = false;
        bool independentResolve = false;
        bool deviceFault = false;
        bool swapchainMaintenance1 = false;
        bool presentId = false;
        bool presentWait = false;
        bool runtimeDescriptorArray = false;
        bool descriptorBindingPartiallyBound = false;
        bool descriptorBindingVariableDescriptorCount = false;
        bool descriptorBindingUpdateUnusedWhilePending = false;
        bool descriptorBindingSampledImageUpdateAfterBind = false;
        bool descriptorBindingStorageImageUpdateAfterBind = false;
        bool descriptorBindingStorageBufferUpdateAfterBind = false;
        bool shaderStorageBufferArrayNonUniformIndexing = false;
        bool shaderSampledImageArrayNonUniformIndexing = false;
        bool shaderStorageImageArrayNonUniformIndexing = false;
        bool shaderFloat16 = false;
        bool shaderInt16 = false;
        bool shaderInt64 = false;
        bool shaderSharedInt64Atomics = false;
        bool shaderBufferInt64Atomics = false;
        bool shaderImageInt64Atomics = false;
        bool subgroupOperations = false;
        std::uint32_t subgroupMinSize = 0;
        std::uint32_t subgroupMaxSize = 0;
        bool geometryShaders = false;
        bool tessellationShaders = false;
        std::uint32_t tessellationMaxPatchControlPoints = 0;
        bool shaderOutputLayer = false;
        bool multiview = false;
        bool multiviewGeometryShader = false;
        bool multiviewTessellationShader = false;
        std::uint32_t maxMultiviewViewCount = 0;
        bool multiViewport = false;
        bool depthBounds = false;
        bool fillModeNonSolid = false;
        bool depthClamp = false;
        bool depthBiasClamp = false;
        bool separateDepthStencilLayouts = false;
        bool logicOp = false;
        bool shaderClipDistance = false;
        bool shaderCullDistance = false;
        bool samplerFilterMinmax = false;
        bool conservativeRasterization = false;
        bool fragmentBarycentric = false;
        bool shaderDrawParameters = false;
        bool occlusionQueryPrecise = false;
        bool pipelineStatisticsQuery = false;
        bool hostQueryReset = false;
        bool calibratedTimestamps = false;
        bool fragmentStoresAndAtomics = false;
        bool fragmentShaderPixelInterlock = false;
        bool sharedBufferHandles = false;
        bool sharedTextureHandles = false;
        bool sharedTimelineHandles = false;
        bool imageViewFormatSwizzle = true;
        std::uint32_t minVertexInputBindingStrideAlignment = 1;
        bool memoryBudget = false;
        bool memoryPriority = false;
        bool meshShaders = false;
        bool taskShaders = false;
        bool meshShaderQueries = false;
        bool multiviewMeshShader = false;
        bool multiDrawIndirect = false;
        bool drawIndirectCount = false;
        bool conditionalRendering = false;
        bool maintenance4 = false;
        bool maintenance5 = false;
        MeshShaderCapabilities meshShader{};
        bool variableRateShading = false;
        bool additionalShadingRates = false;
        bool variableRateShadingTier2 = false;
        bool perPrimitiveShadingRateWithViewportIndexing = false;
        std::uint32_t shadingRateImageTileSize = 0;
        bool bufferDeviceAddress = false;
        bool accelerationStructures = false;
        bool rayTracingPipeline = false;
        bool rayQuery = false;
        bool indirectRayDispatch = false;
        bool deferredHostOperations = false;
        RayTracingCapabilities rayTracingLimits{};
        std::uint32_t minAccelerationStructureScratchOffsetAlignment = 256;
        std::uint32_t maxPerStageDescriptorAccelerationStructures = 0;
        std::uint32_t maxDescriptorSetAccelerationStructures = 0;
        std::uint32_t maxPerStageDescriptorUpdateAfterBindAccelerationStructures = 0;
        std::uint32_t maxDescriptorSetUpdateAfterBindAccelerationStructures = 0;
        std::uint32_t maxPerStageDescriptorUpdateAfterBindSamplers = 0;
        std::uint32_t maxPerStageDescriptorUpdateAfterBindUniformBuffers = 0;
        std::uint32_t maxPerStageDescriptorUpdateAfterBindStorageBuffers = 0;
        std::uint32_t maxPerStageDescriptorUpdateAfterBindSampledImages = 0;
        std::uint32_t maxPerStageDescriptorUpdateAfterBindStorageImages = 0;
        std::uint32_t maxPerStageUpdateAfterBindResources = 0;
        std::uint32_t maxDescriptorSetUpdateAfterBindSamplers = 0;
        std::uint32_t maxDescriptorSetUpdateAfterBindUniformBuffers = 0;
        std::uint32_t maxDescriptorSetUpdateAfterBindUniformBuffersDynamic = 0;
        std::uint32_t maxDescriptorSetUpdateAfterBindStorageBuffers = 0;
        std::uint32_t maxDescriptorSetUpdateAfterBindStorageBuffersDynamic = 0;
        std::uint32_t maxDescriptorSetUpdateAfterBindSampledImages = 0;
        std::uint32_t maxDescriptorSetUpdateAfterBindStorageImages = 0;
    };

    [[nodiscard]] Status checkVulkanInstanceVersion(PFN_vkEnumerateInstanceVersion enumerateInstanceVersion);
    [[nodiscard]] Result<VulkanInstanceInfo> createVulkanInstance(const DeviceDesc& desc);

    void loadVulkanDeviceTable(VolkDeviceTable& table,
                               VkDevice device,
                               PFN_vkGetDeviceProcAddr getDeviceProcAddr) noexcept;

    [[nodiscard]] Result<std::vector<VulkanPhysicalDeviceRecord>> discoverVulkanPhysicalDevices(
        VkInstance instance, const VulkanInstanceTable& instanceTable, const VulkanInstanceReport& instanceReport);

    [[nodiscard]] VulkanQueueFamilies selectVulkanQueueFamilies(std::span<const VkQueueFamilyProperties> families);

    [[nodiscard]] const VulkanPhysicalDeviceRecord* selectVulkanPhysicalDevice(
        const std::vector<VulkanPhysicalDeviceRecord>& devices,
        const VulkanInstanceReport& instanceReport,
        const DeviceDesc& desc);

    void reportVulkanAdapterIdCollisions(const std::vector<VulkanPhysicalDeviceRecord>& devices,
                                         VulkanDiagnosticSink& sink);

    [[nodiscard]] Result<VkDevice> createVulkanLogicalDevice(const VulkanInstanceTable& instanceTable,
                                                             const VulkanPhysicalDeviceRecord& record,
                                                             const VulkanInstanceReport& instanceReport,
                                                             VulkanQueueHandles& queues,
                                                             bool enableDeviceFault);

    inline constexpr std::uint32_t maxVariableArrayDescriptorReserve = 1024;

    [[nodiscard]] constexpr std::uint32_t variableArrayDescriptorLimit(std::uint32_t deviceLimit) noexcept
    {
        const auto reserve = deviceLimit / 4U;
        return deviceLimit -
               (reserve < maxVariableArrayDescriptorReserve ? reserve : maxVariableArrayDescriptorReserve);
    }

    [[nodiscard]] DeviceCapabilities publishVulkanCapabilities(const VulkanInstanceTable& instanceTable,
                                                               const VulkanPhysicalDeviceRecord& record,
                                                               bool debugMarkers) noexcept;
    void recordVulkanQueueFamilies(VulkanPhysicalDeviceRecord& record,
                                   std::span<const VkQueueFamilyProperties> families);
    [[nodiscard]] QueueProperties publishVulkanQueueProperties(const VulkanPhysicalDeviceRecord& record,
                                                               QueueType queueType) noexcept;
    [[nodiscard]] FormatProperties queryVulkanFormatProperties(const VulkanInstanceTable& instanceTable,
                                                               VkPhysicalDevice physical,
                                                               Format format) noexcept;

    class VulkanFormatPropertiesCache final
    {
    public:
        static constexpr std::size_t kFormatCount = static_cast<std::size_t>(Format::bc6hRgbSfloat) + 1U;

        VulkanFormatPropertiesCache() = default;
        VulkanFormatPropertiesCache(const VulkanFormatPropertiesCache&) = delete;
        VulkanFormatPropertiesCache& operator=(const VulkanFormatPropertiesCache&) = delete;
        VulkanFormatPropertiesCache(VulkanFormatPropertiesCache&&) = delete;
        VulkanFormatPropertiesCache& operator=(VulkanFormatPropertiesCache&&) = delete;

        void reset() noexcept
        {
            const std::scoped_lock lock(mutex_);
            for (auto& slot : slots_)
            {
                slot.properties = {};
                slot.ready.store(false, std::memory_order_relaxed);
            }
        }

        [[nodiscard]] FormatProperties query(const VulkanInstanceTable* instanceTable,
                                             VkPhysicalDevice physical,
                                             Format format) const noexcept
        {
            const auto index = static_cast<std::size_t>(format);
            if (format == Format::unknown || index >= kFormatCount || physical == VK_NULL_HANDLE)
            {
                return {};
            }
            auto& slot = slots_[index];
            if (slot.ready.load(std::memory_order_acquire))
            {
                return slot.properties;
            }
            const std::scoped_lock lock(mutex_);
            if (!slot.ready.load(std::memory_order_relaxed))
            {
                static const VulkanInstanceTable emptyInstanceTable{};
                slot.properties = queryVulkanFormatProperties(
                    instanceTable != nullptr ? *instanceTable : emptyInstanceTable, physical, format);
                slot.ready.store(true, std::memory_order_release);
            }
            return slot.properties;
        }

    private:
        struct Slot final
        {
            std::atomic<bool> ready{false};
            FormatProperties properties{};
        };

        mutable std::mutex mutex_{};
        mutable std::array<Slot, kFormatCount> slots_{};
    };

    [[nodiscard]] VkFormat vulkanFormat(Format format) noexcept;
    [[nodiscard]] VkFormat vulkanVertexFormat(VertexFormat format) noexcept;

} // namespace rachis::vulkan
