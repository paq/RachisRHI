#pragma once

#include "D3D12/D3D12Pipeline.h"

#include "Common/HandleTable.h"
#include "D3D12/D3D12Format.h"
#include "D3D12/D3D12NativeHelpers.h"
#include "D3D12/D3D12Query.h"
#include "D3D12/D3D12RootSignaturePolicy.h"
#include "D3D12/D3D12RuntimePolicies.h"
#include "D3D12/D3D12TestHooks.h"
#include "D3D12/D3D12VertexInput.h"
#include "Internal/VertexInputValidation.h"

#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace rachis::d3d12
{

    template<typename Function>
    class PipelineScopeRollback final
    {
    public:
        explicit PipelineScopeRollback(Function function) noexcept(std::is_nothrow_move_constructible_v<Function>)
            : function_(std::move(function))
        {
            static_assert(std::is_nothrow_invocable_v<Function&>);
        }

        PipelineScopeRollback(const PipelineScopeRollback&) = delete;
        PipelineScopeRollback& operator=(const PipelineScopeRollback&) = delete;
        PipelineScopeRollback(PipelineScopeRollback&&) = delete;
        PipelineScopeRollback& operator=(PipelineScopeRollback&&) = delete;
        ~PipelineScopeRollback() noexcept
        {
            if (active_)
            {
                std::invoke(function_);
            }
        }

        void release() noexcept
        {
            active_ = false;
        }

    private:
        Function function_;
        bool active_ = true;
    };

    using ShaderBytecode = std::vector<std::byte>;
    using SpecializationSet = std::vector<SpecializationConstant>;

    struct ShaderLibraryEntryRecord final
    {
        ShaderStage stage = ShaderStage::none;
        ShaderBytecode bytecode;
        std::string entryPoint;
        SpecializationSet specializationConstants;
    };

    struct ShaderLibraryRecord final
    {
        std::vector<ShaderLibraryEntryRecord> entries;
        std::string label;
    };

    struct ShaderInterfaceRecord final
    {
        std::vector<ShaderBindingRequirement> bindings;
        std::vector<ShaderPushConstantRequirement> pushConstants;
        std::vector<std::uint32_t> vertexInputLocations;
        std::optional<ShaderWorkgroupSize> workgroupSize;
        std::optional<ShaderRecordConstantRange> shaderRecordConstants;
        bool drawIndex = false;
    };

    struct ShaderModuleRecord final
    {
        ShaderStage stage = ShaderStage::none;
        ShaderBytecode directBytecode;
        const ShaderBytecode* bytecode = nullptr;
        std::shared_ptr<ShaderLibraryRecord> library;
        std::string entryPoint;
        std::string label;
        ShaderInterfaceRecord shaderInterface;
        SpecializationSet specializationConstants;
    };

    enum class DrawIndexCommandKind : std::uint32_t
    {
        draw,
        drawIndexed,
        dispatchMesh,
    };

    struct PipelineLayoutRecord final
    {
        Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
        std::vector<NativePipelineLayoutSlot> slots;
        std::vector<PushConstantRange> pushConstantRanges;
        std::uint32_t pushConstantRootParameter = invalidRootParameter;
        std::uint32_t pushConstantSize = 0;
        std::uint32_t drawIndexRootParameter = invalidRootParameter;
        std::mutex drawIndexSignatureMutex;
        std::unordered_map<std::uint64_t, Microsoft::WRL::ComPtr<ID3D12CommandSignature>> drawIndexSignatures;
        std::string label;
    };

    enum class PipelineCacheEntryKind : std::uint32_t
    {
        compute,
        graphics,
        mesh,
    };

    struct PipelineCacheEntry final
    {
        PipelineCacheKey key;
        PipelineCacheEntryKind kind = PipelineCacheEntryKind::compute;
        std::uint32_t reserved = 0;

        friend constexpr auto operator<=>(const PipelineCacheEntry&, const PipelineCacheEntry&) noexcept = default;
    };

    struct PipelineCacheRecord final
    {
        std::vector<std::byte> nativeData;
        Microsoft::WRL::ComPtr<ID3D12PipelineLibrary> library;
        Microsoft::WRL::ComPtr<ID3D12PipelineLibrary1> library1;
        std::set<PipelineCacheEntry> entries;
        std::string label;
        std::mutex mutex;
    };

    struct ComputePipelineRecord final
    {
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pipelineState;
        std::shared_ptr<PipelineLayoutRecord> layout;
        std::shared_ptr<ShaderModuleRecord> shader;
        std::string label;
    };

    struct GraphicsPipelineRecord final
    {
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pipelineState;
        std::shared_ptr<PipelineLayoutRecord> layout;
        std::shared_ptr<ShaderModuleRecord> vertexShader;
        std::shared_ptr<ShaderModuleRecord> pixelShader;
        std::shared_ptr<ShaderModuleRecord> geometryShader;
        std::shared_ptr<ShaderModuleRecord> hullShader;
        std::shared_ptr<ShaderModuleRecord> domainShader;
        std::shared_ptr<ShaderModuleRecord> taskShader;
        std::shared_ptr<ShaderModuleRecord> meshShader;
        std::vector<std::uint32_t> vertexStrides;
        std::vector<std::uint32_t> vertexAlignments;
        std::vector<std::uint32_t> streamOutputStrides;
        std::vector<Format> colorFormats;
        Format depthStencilFormat = Format::unknown;
        D3D12_PRIMITIVE_TOPOLOGY topology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
        StripIndexFormat stripIndexFormat;
        std::uint32_t sampleCount = 1;
        std::uint32_t viewMask = 0;
        bool usesBlendConstants = false;
        bool rasterizationEnabled = true;
        bool dynamicDepthBias = false;
        bool depthBoundsTest = false;
        bool stencilEnabled = false;
        bool readsDrawIndex = false;
        std::string label;
    };

    struct ResolvedRayTracingGroup final
    {
        RayTracingShaderGroupType type = RayTracingShaderGroupType::general;
        std::shared_ptr<ShaderModuleRecord> general;
        std::shared_ptr<ShaderModuleRecord> closestHit;
        std::shared_ptr<ShaderModuleRecord> anyHit;
        std::shared_ptr<ShaderModuleRecord> intersection;
        std::optional<ShaderRecordConstantRange> shaderRecordConstants;
    };

    struct RayTracingPipelineLibraryRecord final
    {
        Microsoft::WRL::ComPtr<ID3D12StateObject> stateObject;
        std::shared_ptr<PipelineLayoutRecord> layout;
        std::vector<std::shared_ptr<ShaderModuleRecord>> shaders;
        std::vector<std::wstring> groupNames;
        std::vector<std::uint32_t> groupLocalConstantSizes;
        std::vector<std::uint32_t> groupRecordSizes;
        std::vector<Microsoft::WRL::ComPtr<ID3D12RootSignature>> localRootSignatures;
        std::uint32_t maxPayloadSize = 0;
        std::uint32_t maxAttributeSize = 0;
        bool hasRayGeneration = false;
        std::string label;
    };

    struct RayTracingPipelineRecord final
    {
        Microsoft::WRL::ComPtr<ID3D12StateObject> stateObject;
        Microsoft::WRL::ComPtr<ID3D12StateObjectProperties> properties;
        std::shared_ptr<PipelineLayoutRecord> layout;
        std::vector<std::array<std::byte, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES>> groupHandles;
        std::vector<std::uint32_t> groupLocalConstantSizes;
        std::vector<std::uint32_t> groupRecordSizes;
        std::vector<std::shared_ptr<RayTracingPipelineLibraryRecord>> collections;
        std::string label;
    };

    using PipelineRecord = std::variant<std::shared_ptr<ComputePipelineRecord>,
                                        std::shared_ptr<GraphicsPipelineRecord>,
                                        std::shared_ptr<RayTracingPipelineRecord>>;

    struct OwnedComputePipelineDesc final
    {
        explicit OwnedComputePipelineDesc(const ComputePipelineDesc& source) : desc(source), label(source.label)
        {
        }

        [[nodiscard]] ComputePipelineDesc view() const
        {
            auto result = desc;
            result.label = label;
            return result;
        }

        ComputePipelineDesc desc;
        std::string label;
    };

    struct OwnedGraphicsPipelineDesc final
    {
        explicit OwnedGraphicsPipelineDesc(const GraphicsPipelineDesc& source)
            : desc(source), colorTargets(source.colorTargets.begin(), source.colorTargets.end()),
              streamOutputEntries(source.streamOutputEntries.begin(), source.streamOutputEntries.end()),
              streamOutputStrides(source.streamOutputBufferStrides.begin(), source.streamOutputBufferStrides.end()),
              label(source.label)
        {
            attributes.reserve(source.vertexBuffers.size());
            vertexBuffers.reserve(source.vertexBuffers.size());
            for (const auto& sourceBuffer : source.vertexBuffers)
            {
                attributes.emplace_back(sourceBuffer.attributes.begin(), sourceBuffer.attributes.end());
            }
            for (std::size_t index = 0; index < source.vertexBuffers.size(); ++index)
            {
                const auto& sourceBuffer = source.vertexBuffers[index];
                vertexBuffers.push_back({
                    .stride = sourceBuffer.stride,
                    .stepMode = sourceBuffer.stepMode,
                    .instanceStepRate = sourceBuffer.instanceStepRate,
                    .attributes = attributes[index],
                });
            }
        }

        [[nodiscard]] GraphicsPipelineDesc view() const
        {
            auto result = desc;
            result.vertexBuffers = vertexBuffers;
            result.colorTargets = colorTargets;
            result.streamOutputEntries = streamOutputEntries;
            result.streamOutputBufferStrides = streamOutputStrides;
            result.label = label;
            return result;
        }

        GraphicsPipelineDesc desc;
        std::vector<std::vector<VertexAttribute>> attributes;
        std::vector<VertexBufferLayout> vertexBuffers;
        std::vector<ColorTargetState> colorTargets;
        std::vector<StreamOutputEntry> streamOutputEntries;
        std::vector<std::uint32_t> streamOutputStrides;
        std::string label;
    };

    struct RetainedPipelineDependencies final
    {
        PipelineLayoutHandle layout;
        std::vector<ShaderModuleHandle> shaders;
        PipelineCacheHandle cache;
    };

    struct PipelineCompilationRecord final
    {
        std::mutex mutex;
        std::condition_variable condition;
        PipelineCompilationState state = PipelineCompilationState::pending;
        std::optional<Result<PipelineHandle>> result;
        bool consumed = false;
        bool abandoned = false;
    };

    struct PipelineCompilationJob final
    {
        std::shared_ptr<PipelineCompilationRecord> record;
        RetainedPipelineDependencies dependencies;
        std::variant<OwnedComputePipelineDesc, OwnedGraphicsPipelineDesc> desc;
    };

    static_assert(std::is_nothrow_move_assignable_v<RetainedPipelineDependencies>);

    struct BindingSetBatchInternalOwner final
    {
        std::array<BindingPoolCommandRetention, maxBindingSets> poolRetentions{};
        std::array<std::shared_ptr<GpuDescriptorLease>, maxBindingSets> resourceDescriptors{};
        std::array<std::shared_ptr<GpuDescriptorLease>, maxBindingSets> samplerDescriptors{};
        SmallVector<std::shared_ptr<void>, 4> sparseMappingOwners{};
        SmallVector<AccelerationStructureExecutionLease, 4> accelerationStructureLeases{};
        std::shared_ptr<DynamicDescriptorArena> dynamicDescriptorArena{};
        std::size_t setCount = 0;
    };

    struct BindingSetBatchOwner final
    {
        std::array<NativeBindingSet, maxBindingSets> sets{};
        std::size_t setCount = 0;
        std::shared_ptr<void> internalOwner;
        SmallVector<std::shared_ptr<ResidencyRecord>, 16> mergedResidencies{};
        bool usesMergedResidencies = false;
        SmallVector<std::shared_ptr<SwapchainImageLeaseState>, 4> swapchainImageLeases{};
        SmallVector<TimelinePoint, 4> sparseMappingDependencies{};

        [[nodiscard]] std::span<const NativeBindingSet> activeSets() const noexcept
        {
            return std::span<const NativeBindingSet>{sets.data(), setCount};
        }

        [[nodiscard]] std::span<const std::shared_ptr<ResidencyRecord>> residencySpan() const noexcept
        {
            if (usesMergedResidencies)
            {
                return mergedResidencies.span();
            }
            return setCount != 0 ? sets[0].staticResidencies : std::span<const std::shared_ptr<ResidencyRecord>>{};
        }
    };

    namespace pipeline_detail
    {

        inline void setDebugName(ID3D12Object& object, std::string_view label) noexcept
        {
            if (!label.empty())
            {
                static_cast<void>(
                    object.SetPrivateData(WKPDID_D3DDebugObjectName, static_cast<UINT>(label.size()), label.data()));
            }
        }

        [[nodiscard]] constexpr bool isContiguousViewMask(std::uint32_t mask) noexcept
        {
            if (mask == 0)
            {
                return true;
            }
            const auto normalized = mask >> std::countr_zero(mask);
            return (normalized & (normalized + 1U)) == 0;
        }

        [[nodiscard]] constexpr std::uint32_t multiviewLayerCount(std::uint32_t mask) noexcept
        {
            return std::bit_width(mask);
        }

        [[nodiscard]] constexpr bool isColorTargetFormat(Format value) noexcept
        {
            return format::isColorTarget(value);
        }

        [[nodiscard]] constexpr bool isValidIndexFormat(IndexFormat format) noexcept
        {
            return format == IndexFormat::uint16 || format == IndexFormat::uint32;
        }

        [[nodiscard]] constexpr std::uint32_t mipExtent(std::uint32_t extent, std::uint32_t mipLevel) noexcept
        {
            return (std::max)(1U, extent >> mipLevel);
        }

        [[nodiscard]] constexpr std::optional<ShaderResourceType> shaderResourceType(BindingType type) noexcept
        {
            switch (type)
            {
            case BindingType::uniformBuffer:
            case BindingType::dynamicUniformBuffer:
                return ShaderResourceType::uniformBuffer;
            case BindingType::readOnlyStorageBuffer:
            case BindingType::dynamicReadOnlyStorageBuffer:
                return ShaderResourceType::readOnlyStorageBuffer;
            case BindingType::storageBuffer:
            case BindingType::dynamicStorageBuffer:
                return ShaderResourceType::storageBuffer;
            case BindingType::sampledTexture:
                return ShaderResourceType::sampledTexture;
            case BindingType::storageTexture:
                return ShaderResourceType::storageTexture;
            case BindingType::sampler:
                return ShaderResourceType::sampler;
            case BindingType::accelerationStructure:
                return ShaderResourceType::accelerationStructure;
            case BindingType::minMipFeedback:
                return ShaderResourceType::minMipFeedback;
            case BindingType::mipRegionUsedFeedback:
                return ShaderResourceType::mipRegionUsedFeedback;
            }
            return std::nullopt;
        }

        [[nodiscard]] Status validateShaderInterface(const ShaderModuleRecord& shader,
                                                     const PipelineLayoutRecord& layout);

        [[nodiscard]] Result<std::shared_ptr<BindingSetBatchOwner>> prepareBindingSetBatch(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            BindingManager& bindings,
            ResourceManager& resources,
            std::span<const NativePipelineLayoutSlot> slots,
            std::uint32_t firstSet,
            std::span<const BindingSetHandle> bindingSets,
            std::span<const std::uint32_t> dynamicOffsets);

        [[nodiscard]] Status retainBindingSetBatchDependencies(QueueManager& queues,
                                                               detail::BackendCommandBuffer& commandBuffer,
                                                               const std::shared_ptr<BindingSetBatchOwner>& prepared);

        inline void applyComputeRootBindings(ID3D12GraphicsCommandList& commandList,
                                             const NativePipelineLayoutSlot& slot,
                                             const NativeBindingSet& nativeSet) noexcept
        {
            for (std::uint32_t index = 0; index < nativeSet.rootBufferCount; ++index)
            {
                const auto& rootBuffer = nativeSet.rootBuffers[index];
                switch (rootBuffer.kind)
                {
                case RootDescriptorKind::cbv:
                    commandList.SetComputeRootConstantBufferView(rootBuffer.rootParameter,
                                                                 rootBuffer.gpuVirtualAddress);
                    break;
                case RootDescriptorKind::srv:
                    commandList.SetComputeRootShaderResourceView(rootBuffer.rootParameter,
                                                                 rootBuffer.gpuVirtualAddress);
                    break;
                case RootDescriptorKind::uav:
                    commandList.SetComputeRootUnorderedAccessView(rootBuffer.rootParameter,
                                                                  rootBuffer.gpuVirtualAddress);
                    break;
                }
            }
            if (slot.resourceRootParameter != invalidRootParameter)
            {
                commandList.SetComputeRootDescriptorTable(slot.resourceRootParameter, nativeSet.resources);
            }
            if (slot.samplerRootParameter != invalidRootParameter)
            {
                commandList.SetComputeRootDescriptorTable(slot.samplerRootParameter, nativeSet.samplers);
            }
            if (slot.dynamicRootParameter != invalidRootParameter)
            {
                commandList.SetComputeRootDescriptorTable(slot.dynamicRootParameter, nativeSet.dynamicResources);
            }
        }

        inline void applyGraphicsRootBindings(ID3D12GraphicsCommandList& commandList,
                                              const NativePipelineLayoutSlot& slot,
                                              const NativeBindingSet& nativeSet) noexcept
        {
            for (std::uint32_t index = 0; index < nativeSet.rootBufferCount; ++index)
            {
                const auto& rootBuffer = nativeSet.rootBuffers[index];
                switch (rootBuffer.kind)
                {
                case RootDescriptorKind::cbv:
                    commandList.SetGraphicsRootConstantBufferView(rootBuffer.rootParameter,
                                                                  rootBuffer.gpuVirtualAddress);
                    break;
                case RootDescriptorKind::srv:
                    commandList.SetGraphicsRootShaderResourceView(rootBuffer.rootParameter,
                                                                  rootBuffer.gpuVirtualAddress);
                    break;
                case RootDescriptorKind::uav:
                    commandList.SetGraphicsRootUnorderedAccessView(rootBuffer.rootParameter,
                                                                   rootBuffer.gpuVirtualAddress);
                    break;
                }
            }
            if (slot.resourceRootParameter != invalidRootParameter)
            {
                commandList.SetGraphicsRootDescriptorTable(slot.resourceRootParameter, nativeSet.resources);
            }
            if (slot.samplerRootParameter != invalidRootParameter)
            {
                commandList.SetGraphicsRootDescriptorTable(slot.samplerRootParameter, nativeSet.samplers);
            }
            if (slot.dynamicRootParameter != invalidRootParameter)
            {
                commandList.SetGraphicsRootDescriptorTable(slot.dynamicRootParameter, nativeSet.dynamicResources);
            }
        }

    } // namespace pipeline_detail

    struct PipelineManager::Impl final
    {
        ID3D12Device* device = nullptr;
        std::uint32_t maxComputeWorkgroupInvocations = 0;
        std::array<std::uint32_t, 3> maxComputeWorkgroupSize{};
        bool geometryShaders = false;
        std::uint32_t maxPatchControlPoints = 0;
        bool meshShaders = false;
        bool taskShaders = false;
        MeshShaderCapabilities meshShaderCapabilities;
        bool dualSourceBlending = false;
        bool logicOperations = false;
        bool conservativeRasterization = false;
        bool triangleFans = false;
        bool dynamicDepthBias = false;
        bool depthBoundsTest = false;
        bool variableRateShading = false;
        bool additionalShadingRates = false;
        bool variableRateShadingTier2 = false;
        std::uint32_t maxClipDistances = 0;
        std::uint32_t maxCullDistances = 0;
        std::uint32_t maxCombinedClipAndCullDistances = 0;
        std::uint32_t maxFramebufferLayers = 1;
        std::uint32_t maxMultiviewViewCount = 0;
        bool rayTracing = false;
        RayTracingCapabilities rayTracingCapabilities;
        Microsoft::WRL::ComPtr<ID3D12Device1> pipelineLibraryDevice;
        Microsoft::WRL::ComPtr<ID3D12Device2> pipelineStateStreamDevice;
        Microsoft::WRL::ComPtr<ID3D12Device5> rayTracingDevice;
        std::mutex mutex;
        detail::HandleTable<std::shared_ptr<ShaderLibraryRecord>, ShaderLibraryHandle> shaderLibraries;
        detail::HandleTable<std::shared_ptr<ShaderModuleRecord>, ShaderModuleHandle> shaderModules;
        detail::HandleTable<std::shared_ptr<PipelineLayoutRecord>, PipelineLayoutHandle> pipelineLayouts;
        detail::HandleTable<std::shared_ptr<PipelineCacheRecord>, PipelineCacheHandle> pipelineCaches;
        detail::HandleTable<std::shared_ptr<PipelineCompilationRecord>, PipelineCompilationHandle> pipelineCompilations;
        detail::HandleTable<std::shared_ptr<RayTracingPipelineLibraryRecord>, RayTracingPipelineLibraryHandle>
            rayTracingPipelineLibraries;
        detail::HandleTable<PipelineRecord, PipelineHandle> pipelines;
        std::uint64_t nextRayTracingExportNamespace = 1;
        std::unordered_map<std::uint32_t, Microsoft::WRL::ComPtr<ID3D12CommandSignature>> dispatchIndirectSignatures;
        Microsoft::WRL::ComPtr<ID3D12CommandSignature> rayTracingIndirectSignature;
        std::unordered_map<std::uint64_t, Microsoft::WRL::ComPtr<ID3D12CommandSignature>> indirectSignatures;
        std::unordered_map<std::uint32_t, Microsoft::WRL::ComPtr<ID3D12CommandSignature>> meshIndirectSignatures;
        PipelineManager* owner = nullptr;
        std::mutex compilationQueueMutex;
        std::condition_variable compilationQueueCondition;
        std::deque<std::shared_ptr<PipelineCompilationJob>> compilationJobs;
        std::vector<std::thread> compilationWorkers;
        bool stopCompilation = false;

        [[nodiscard]] Status startCompilationWorkers();
        void stopCompilationWorkers() noexcept;
        [[nodiscard]] Status enqueueCompilation(std::shared_ptr<PipelineCompilationJob> job);
        void compilationWorker() noexcept;
        void executeCompilation(PipelineCompilationJob& job) noexcept;
        void releaseDependenciesLocked(const RetainedPipelineDependencies& dependencies) noexcept;
        void releaseDependencies(const RetainedPipelineDependencies& dependencies) noexcept;
        [[nodiscard]] Result<RetainedPipelineDependencies> retainDependencies(OwnedComputePipelineDesc& owned);
        [[nodiscard]] Result<RetainedPipelineDependencies> retainDependencies(OwnedGraphicsPipelineDesc& owned);
        [[nodiscard]] Result<Microsoft::WRL::ComPtr<ID3D12CommandSignature>> indirectDispatchSignature(
            std::uint32_t stride);
        [[nodiscard]] Result<Microsoft::WRL::ComPtr<ID3D12CommandSignature>> indirectRayTracingDispatchSignature();
        [[nodiscard]] Result<Microsoft::WRL::ComPtr<ID3D12CommandSignature>> indirectSignature(bool indexed,
                                                                                               std::uint32_t stride);
        [[nodiscard]] Result<Microsoft::WRL::ComPtr<ID3D12CommandSignature>> indirectMeshSignature(
            std::uint32_t stride);
        [[nodiscard]] Result<Microsoft::WRL::ComPtr<ID3D12CommandSignature>> drawIndexSignature(
            PipelineLayoutRecord& layout, DrawIndexCommandKind kind, std::uint32_t stride);
        [[nodiscard]] Result<std::shared_ptr<PipelineCacheRecord>> pipelineCache(PipelineCacheHandle cache,
                                                                                 PipelineCacheKey key);
        [[nodiscard]] Status validateGraphicsFixedFunctionState(const GraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateGraphicsFeatureSupport(const GraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateTessellationAndTopologySupport(const GraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateGraphicsPipelineDesc(const GraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validatePrimitiveRestartAndConservativeRasterization(
            const GraphicsPipelineDesc& desc) const;
        [[nodiscard]] Status validateMultiviewSupport(const GraphicsPipelineDesc& desc) const;
        [[nodiscard]] Result<Microsoft::WRL::ComPtr<ID3D12PipelineState>> createGraphicsPipelineState(
            const GraphicsPipelineDesc& desc,
            const D3D12_GRAPHICS_PIPELINE_STATE_DESC& nativeDesc,
            PipelineCacheRecord* cache,
            const GraphicsPipelineRecord& record) const;
    };

} // namespace rachis::d3d12
