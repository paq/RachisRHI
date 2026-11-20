#pragma once

#include "Common/HandleTable.h"
#include "Internal/VertexInputValidation.h"
#include "Internal/VulkanBindings.h"
#include "Internal/VulkanRuntime.h"
#include "Internal/VulkanShaders.h"
#include "Rachis/Device.h"
#include "Rachis/Pipeline.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <span>
#include <string>
#include <thread>
#include <variant>
#include <vector>
#include <volk.h>

namespace rachis::vulkan
{

    struct VulkanNativePipeline final
    {
        std::shared_ptr<void> owner;
        std::shared_ptr<void> layoutOwner;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        bool compute = false;
        bool rayTracing = false;
        bool mesh = false;
        bool rasterizationEnabled = true;
        bool stencilEnabled = false;
        bool usesBlendConstants = false;
        bool dynamicDepthBias = false;
        bool depthBoundsTest = false;
        bool readsDrawIndex = false;
        std::uint32_t sampleCount = 1;
        std::uint32_t viewMask = 0;
        Format depthStencilFormat = Format::unknown;
        std::vector<Format> colorFormats;
        std::vector<std::uint32_t> vertexStrides;
        std::vector<std::uint32_t> vertexAlignments;
        std::vector<PushConstantRange> pushConstantRanges;
        std::vector<PushConstantRange> nativePushConstantRanges;
        std::vector<std::uint8_t> setRequired;
        std::vector<std::uint32_t> setDynamicCounts;
        std::vector<VkDescriptorSetLayout> setLayouts;
    };

    class VulkanPipelineManager final
    {
    public:
        VulkanPipelineManager() = default;
        ~VulkanPipelineManager();

        VulkanPipelineManager(const VulkanPipelineManager&) = delete;
        VulkanPipelineManager& operator=(const VulkanPipelineManager&) = delete;
        VulkanPipelineManager(VulkanPipelineManager&&) = delete;
        VulkanPipelineManager& operator=(VulkanPipelineManager&&) = delete;

        void initialize(VkDevice device,
                        const VolkDeviceTable* table,
                        VkPhysicalDevice physical,
                        const DeviceCapabilities& capabilities,
                        VulkanShaderManager* shaders,
                        VulkanBindingManager* bindings,
                        const VulkanInstanceTable* instanceTable = nullptr,
                        bool depthBiasClamp = true,
                        VulkanFormatPropertiesCache* sharedFormatCache = nullptr,
                        bool maintenance5 = false,
                        VulkanMultiviewStageSupport multiviewStages = {},
                        std::uint32_t minVertexInputBindingStrideAlignment = 1,
                        bool isolateDynamicBufferShaders = false) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] Result<PipelineCacheHandle> createPipelineCache(const PipelineCacheDesc& desc);
        [[nodiscard]] Status destroyPipelineCache(PipelineCacheHandle pipelineCache);
        [[nodiscard]] Result<std::vector<std::byte>> pipelineCacheData(PipelineCacheHandle pipelineCache);
        [[nodiscard]] Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc& desc);
        [[nodiscard]] Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc& desc);
        [[nodiscard]] Result<RayTracingPipelineLibraryHandle> createRayTracingPipelineLibrary(
            const RayTracingPipelineLibraryDesc& desc);
        [[nodiscard]] Status destroyRayTracingPipelineLibrary(RayTracingPipelineLibraryHandle library);
        [[nodiscard]] Result<PipelineHandle> createRayTracingPipeline(const RayTracingPipelineDesc& desc);
        [[nodiscard]] Status getRayTracingShaderGroupHandles(PipelineHandle pipeline,
                                                             std::uint32_t firstGroup,
                                                             std::uint32_t groupCount,
                                                             std::span<std::byte> destination) const;
        [[nodiscard]] Result<std::uint32_t> rayTracingShaderGroupRecordSize(PipelineHandle pipeline,
                                                                            std::uint32_t groupIndex) const;
        [[nodiscard]] Status writeRayTracingShaderGroupRecord(PipelineHandle pipeline,
                                                              std::uint32_t groupIndex,
                                                              std::span<const std::byte> localConstants,
                                                              std::span<std::byte> destination) const;
        [[nodiscard]] Result<PipelineCompilationHandle> beginComputePipelineCompilation(
            const ComputePipelineDesc& desc);
        [[nodiscard]] Result<PipelineCompilationHandle> beginGraphicsPipelineCompilation(
            const GraphicsPipelineDesc& desc);
        [[nodiscard]] Result<PipelineCompilationState> pipelineCompilationState(PipelineCompilationHandle compilation);
        [[nodiscard]] Result<PipelineHandle> waitPipelineCompilation(PipelineCompilationHandle compilation);
        [[nodiscard]] Status destroyPipelineCompilation(PipelineCompilationHandle compilation);
        [[nodiscard]] Status destroyPipeline(PipelineHandle pipeline);
        [[nodiscard]] Result<std::shared_ptr<const VulkanNativePipeline>> nativePipeline(PipelineHandle pipeline) const;

        enum class PipelineCacheEntryKind : std::uint32_t
        {
            compute,
            graphics,
        };

        struct PipelineCacheEntry final
        {
            PipelineCacheKey key{};
            PipelineCacheEntryKind kind = PipelineCacheEntryKind::compute;
            std::uint32_t reserved = 0;

            friend constexpr auto operator<=>(const PipelineCacheEntry&, const PipelineCacheEntry&) noexcept = default;
        };

    private:
        struct CacheKeepAlive final
        {
            VkDevice device = VK_NULL_HANDLE;
            const VolkDeviceTable* table = nullptr;
            VkPipelineCache cache = VK_NULL_HANDLE;

            CacheKeepAlive() = default;
            ~CacheKeepAlive();
            CacheKeepAlive(const CacheKeepAlive&) = delete;
            CacheKeepAlive& operator=(const CacheKeepAlive&) = delete;
            CacheKeepAlive(CacheKeepAlive&&) = delete;
            CacheKeepAlive& operator=(CacheKeepAlive&&) = delete;
        };

        struct CacheRecord final
        {
            std::shared_ptr<CacheKeepAlive> keepAlive;
            std::set<PipelineCacheEntry> entries;
            std::string label;
            std::mutex mutex;
        };

        struct PipelineKeepAlive final
        {
            VkDevice device = VK_NULL_HANDLE;
            const VolkDeviceTable* table = nullptr;
            VkPipeline pipeline = VK_NULL_HANDLE;
            std::shared_ptr<void> layoutOwner;
            std::vector<std::shared_ptr<void>> shaderOwners;

            PipelineKeepAlive() = default;
            ~PipelineKeepAlive();
            PipelineKeepAlive(const PipelineKeepAlive&) = delete;
            PipelineKeepAlive& operator=(const PipelineKeepAlive&) = delete;
            PipelineKeepAlive(PipelineKeepAlive&&) = delete;
            PipelineKeepAlive& operator=(PipelineKeepAlive&&) = delete;
        };

        struct PipelineRecord final
        {
            std::shared_ptr<const VulkanNativePipeline> native = {};
            std::vector<std::byte> shaderGroupHandles = {};
            std::vector<std::uint32_t> shaderGroupRecordSizes = {};
            std::vector<std::uint32_t> shaderGroupLocalConstantSizes = {};
        };

    public:
        struct RayTracingGroupRecord final
        {
            RayTracingShaderGroupType type = RayTracingShaderGroupType::general;
            VulkanShaderManager::NativeShaderModule general = {};
            VulkanShaderManager::NativeShaderModule closestHit = {};
            VulkanShaderManager::NativeShaderModule anyHit = {};
            VulkanShaderManager::NativeShaderModule intersection = {};
            std::optional<ShaderRecordConstantRange> shaderRecordConstants = {};
            std::uint32_t localConstantSize = 0;
            std::uint32_t recordSize = 0;
            bool hasGeneral = false;
            bool hasClosestHit = false;
            bool hasAnyHit = false;
            bool hasIntersection = false;
        };

        struct RayTracingLibraryRecord final
        {
            VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
            std::shared_ptr<void> layoutOwner;
            std::uint32_t maxPayloadSize = 0;
            std::uint32_t maxAttributeSize = 0;
            bool hasRayGeneration = false;
            std::vector<RayTracingGroupRecord> groups;
            std::string label;
        };

    private:
        struct CompilationRecord final
        {
            std::mutex mutex;
            std::condition_variable condition;
            PipelineCompilationState state = PipelineCompilationState::pending;
            std::optional<Result<PipelineHandle>> result = {};
            bool consumed = false;
            bool abandoned = false;
        };

        struct ResolvedCache final
        {
            std::shared_ptr<CacheRecord> record = {};
            PipelineCacheKey key{};
            PipelineCachePolicy policy = PipelineCachePolicy::populate;
            PipelineCacheEntryKind kind = PipelineCacheEntryKind::compute;
        };

        struct OwnedVertexBufferLayout final
        {
            std::uint32_t stride = 0;
            VertexStepMode stepMode = VertexStepMode::vertex;
            std::uint32_t instanceStepRate = 1;
            std::vector<VertexAttribute> attributes = {};
        };

        struct OwnedGraphicsPipelineDesc final
        {
            bool hasLayout = false;
            bool hasVertexShader = false;
            bool hasPixelShader = false;
            bool hasGeometryShader = false;
            bool hasHullShader = false;
            bool hasDomainShader = false;
            bool hasTaskShader = false;
            bool hasMeshShader = false;
            bool hasCache = false;
            std::vector<OwnedVertexBufferLayout> vertexBuffers = {};
            PrimitiveTopology topology = PrimitiveTopology::triangleList;
            std::uint32_t patchControlPoints = 0;
            StripIndexFormat stripIndexFormat = {};
            FrontFace frontFace = FrontFace::counterClockwise;
            CullMode cullMode = CullMode::back;
            FillMode fillMode = FillMode::solid;
            std::int32_t depthBias = 0;
            float depthBiasClamp = 0.0F;
            float depthBiasSlopeScale = 0.0F;
            bool dynamicDepthBiasEnabled = false;
            bool depthClipEnabled = true;
            std::vector<ColorTargetState> colorTargets = {};
            DepthStencilState depthStencil = {};
            std::uint32_t sampleCount = 1;
            std::uint32_t sampleMask = 0xFFFFFFFFU;
            bool alphaToCoverageEnabled = false;
            std::optional<LogicOperation> logicOperation = {};
            ConservativeRasterizationMode conservativeRasterization = ConservativeRasterizationMode::disabled;
            std::uint32_t clipDistanceCount = 0;
            std::uint32_t cullDistanceCount = 0;
            std::uint32_t viewMask = 0;
            std::vector<StreamOutputEntry> streamOutputEntries = {};
            std::vector<std::uint32_t> streamOutputBufferStrides = {};
            std::optional<std::uint32_t> rasterizedStream = 0;
            std::string label = {};
            PipelineCacheKey cacheKey = {};
            PipelineCachePolicy cachePolicy = PipelineCachePolicy::populate;
        };

        struct PreparedComputeCompilation final
        {
            bool hasLayout = false;
            bool hasComputeShader = false;
            bool hasCache = false;
            std::optional<VulkanNativePipelineLayout> layout = {};
            std::optional<VulkanShaderManager::NativeShaderModule> computeShader = {};
            std::shared_ptr<CacheRecord> cacheRecord = {};
            std::string label = {};
            PipelineCacheKey cacheKey = {};
            PipelineCachePolicy cachePolicy = PipelineCachePolicy::populate;
            std::uint32_t requiredSubgroupSize = 0;
            std::uint32_t workgroupInvocations = 0;
        };

        struct PreparedGraphicsCompilation final
        {
            OwnedGraphicsPipelineDesc desc = {};
            std::optional<VulkanNativePipelineLayout> layout = {};
            std::optional<VulkanShaderManager::NativeShaderModule> vertexShader = {};
            std::optional<VulkanShaderManager::NativeShaderModule> pixelShader = {};
            std::optional<VulkanShaderManager::NativeShaderModule> geometryShader = {};
            std::optional<VulkanShaderManager::NativeShaderModule> hullShader = {};
            std::optional<VulkanShaderManager::NativeShaderModule> domainShader = {};
            std::optional<VulkanShaderManager::NativeShaderModule> taskShader = {};
            std::optional<VulkanShaderManager::NativeShaderModule> meshShader = {};
            std::shared_ptr<CacheRecord> cacheRecord = {};
        };

        struct CompilationJob final
        {
            std::shared_ptr<CompilationRecord> record = {};
            std::variant<PreparedComputeCompilation, PreparedGraphicsCompilation> payload;
        };

        void startWorkers() noexcept;
        void stopWorkers() noexcept;
        void workerLoop() noexcept;
        [[nodiscard]] Result<PipelineHandle> publishPipeline(PipelineRecord record);
        [[nodiscard]] Result<PipelineCompilationHandle> enqueueCompilation(
            std::variant<PreparedComputeCompilation, PreparedGraphicsCompilation>&& payload);
        [[nodiscard]] Result<PreparedComputeCompilation> prepareComputeCompilation(
            const ComputePipelineDesc& desc) const;
        [[nodiscard]] Result<PreparedGraphicsCompilation> prepareGraphicsCompilation(
            const GraphicsPipelineDesc& desc) const;
        [[nodiscard]] Result<PipelineHandle> compileComputePipeline(PreparedComputeCompilation&& prepared);
        [[nodiscard]] Result<PipelineHandle> compileGraphicsPipeline(PreparedGraphicsCompilation&& prepared);
        [[nodiscard]] Result<ResolvedCache> resolvePreparedCache(PipelineCachePolicy policy,
                                                                 bool hasCacheHandle,
                                                                 std::shared_ptr<CacheRecord> cacheRecord,
                                                                 PipelineCacheKey key,
                                                                 PipelineCacheEntryKind kind) const;
        [[nodiscard]] Status consumeCacheEntry(ResolvedCache& cache);
        [[nodiscard]] Status storeCacheEntry(const ResolvedCache& cache);
        [[nodiscard]] Result<std::vector<std::shared_ptr<RayTracingLibraryRecord>>> resolveLinkedRayTracingLibraries(
            const RayTracingPipelineDesc& desc, const VulkanNativePipelineLayout& layout) const;

        struct LockedPipelineCache;
        struct GraphicsShaderStages;
        struct VertexInputLayout;

        [[nodiscard]] Result<std::shared_ptr<CacheRecord>> findCacheRecord(PipelineCacheHandle pipelineCache) const;
        [[nodiscard]] Result<LockedPipelineCache> lockCacheAndConsumeEntry(ResolvedCache& cache);
        [[nodiscard]] bool isolatesShaderModules(const VulkanNativePipelineLayout& layout) const noexcept;
        [[nodiscard]] Status validateRequiredSubgroupSize(const PreparedComputeCompilation& prepared) const;
        [[nodiscard]] Result<VkPipeline> createNativeComputePipeline(
            const VulkanShaderManager::NativeShaderModule& shader,
            VkPipelineLayout pipelineLayout,
            VkPipelineCache nativeCache,
            std::uint32_t requiredSubgroupSize) const;
        [[nodiscard]] std::shared_ptr<PipelineKeepAlive> makePipelineKeepAlive(
            VkPipeline pipeline, const std::shared_ptr<void>& layoutOwner) const;
        [[nodiscard]] Result<std::shared_ptr<PipelineKeepAlive>> adoptComputePipeline(
            VkPipeline pipeline,
            const std::shared_ptr<void>& layoutOwner,
            const std::shared_ptr<void>& shaderOwner) const;
        void labelPipeline(VkPipeline pipeline, const std::string& label) const;
        [[nodiscard]] Result<PipelineHandle> publishComputePipeline(
            VulkanNativePipelineLayout&& layout, const std::shared_ptr<PipelineKeepAlive>& keepAlive);
        [[nodiscard]] static OwnedGraphicsPipelineDesc copyFixedSizeState(const GraphicsPipelineDesc& desc);
        [[nodiscard]] Status resolveGraphicsShaders(const GraphicsPipelineDesc& desc,
                                                    PreparedGraphicsCompilation& prepared) const;
        [[nodiscard]] static Status copyVariableSizeState(const GraphicsPipelineDesc& desc,
                                                          OwnedGraphicsPipelineDesc& owned);
        [[nodiscard]] Status validateGraphicsPipelineState(const OwnedGraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateMeshStageUsage(const OwnedGraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateDepthAndGeometryFeatureSupport(const OwnedGraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateTessellationUsage(const OwnedGraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateRasterizationAndOutputSupport(const OwnedGraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateMultiviewStageSupport(const OwnedGraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateStageCombinationAndFeatureSupport(const OwnedGraphicsPipelineDesc& desc) const;
        [[nodiscard]] static Result<GraphicsShaderStages> takeValidatedShaderStages(
            PreparedGraphicsCompilation& prepared, const VulkanNativePipelineLayout& layout);
        [[nodiscard]] Status validatePointSizeOutput(const OwnedGraphicsPipelineDesc& desc,
                                                     const GraphicsShaderStages& shaders) const;
        [[nodiscard]] Status validateVertexBufferLayout(const OwnedVertexBufferLayout& buffer) const;
        [[nodiscard]] Result<VertexInputLayout> buildVertexInputLayout(
            const std::vector<OwnedVertexBufferLayout>& vertexBuffers) const;
        [[nodiscard]] Status validateColorTargets(const OwnedGraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateDepthStencilState(const DepthStencilState& depthStencil) const;
        [[nodiscard]] static VkPipelineRasterizationStateCreateInfo makeRasterizationState(
            const OwnedGraphicsPipelineDesc& desc);
        [[nodiscard]] Result<VkPipeline> createNativeGraphicsPipeline(const OwnedGraphicsPipelineDesc& desc,
                                                                      const GraphicsShaderStages& shaders,
                                                                      const VertexInputLayout& vertexLayout,
                                                                      VkPipelineLayout pipelineLayout,
                                                                      VkPipelineCache nativeCache) const;
        [[nodiscard]] std::shared_ptr<PipelineKeepAlive> adoptGraphicsPipeline(
            VkPipeline pipeline, const std::shared_ptr<void>& layoutOwner, const GraphicsShaderStages& shaders) const;
        [[nodiscard]] Result<PipelineHandle> publishGraphicsPipeline(
            const PreparedGraphicsCompilation& prepared,
            const std::shared_ptr<PipelineKeepAlive>& keepAlive,
            std::vector<std::uint32_t> vertexStrides,
            bool readsDrawIndex);

        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        const VulkanInstanceTable* instanceTable_ = nullptr;
        VkPhysicalDevice physical_ = VK_NULL_HANDLE;
        DeviceCapabilities capabilities_{};
        std::uint32_t maxComputeWorkgroupSubgroups_ = 0;
        std::uint32_t maxFragmentCombinedOutputResources_ = std::numeric_limits<std::uint32_t>::max();
        bool depthBiasClampSupported_ = true;
        bool maintenance5_ = false;
        VulkanMultiviewStageSupport multiviewStages_{};
        std::uint32_t minVertexInputBindingStrideAlignment_ = 1;
        bool isolateDynamicBufferShaders_ = false;
        VulkanShaderManager* shaders_ = nullptr;
        VulkanBindingManager* bindings_ = nullptr;
        std::array<std::uint8_t, VK_UUID_SIZE> pipelineCacheUuid_{};
        std::array<std::uint8_t, VK_UUID_SIZE> deviceUuid_{};
        std::array<std::uint8_t, VK_UUID_SIZE> driverUuid_{};
        VulkanFormatPropertiesCache formatCache_{};
        VulkanFormatPropertiesCache* activeFormatCache_ = &formatCache_;
        mutable std::shared_mutex mutex_{};
        detail::HandleTable<std::shared_ptr<CacheRecord>, PipelineCacheHandle> caches_{};
        detail::HandleTable<PipelineRecord, PipelineHandle> pipelines_{};
        detail::HandleTable<std::shared_ptr<CompilationRecord>, PipelineCompilationHandle> compilations_{};
        detail::HandleTable<std::shared_ptr<RayTracingLibraryRecord>, RayTracingPipelineLibraryHandle>
            rayTracingLibraries_{};
        std::mutex workerMutex_{};
        std::condition_variable workerCondition_{};
        std::deque<CompilationJob> workerQueue_{};
        std::vector<std::thread> workers_{};
        bool stopWorkers_ = false;
    };

#ifdef RACHIS_TESTING
    struct VulkanCapturedPipelineCreation final
    {
        std::vector<VkShaderModule> shaderModules;
        VkPipelineCache cache = VK_NULL_HANDLE;
    };

    void captureVulkanPipelineCreations(std::vector<VulkanCapturedPipelineCreation>* sink) noexcept;
#endif

} // namespace rachis::vulkan
