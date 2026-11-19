#include "D3D12/D3D12DxilSignature.h"
#include "D3D12/D3D12PipelineInternal.h"

namespace rachis::d3d12
{
    using Microsoft::WRL::ComPtr;
    using namespace pipeline_detail;

    namespace
    {

        template<typename Function>
        class ScopeRollback final
        {
        public:
            explicit ScopeRollback(Function function) noexcept(std::is_nothrow_move_constructible_v<Function>)
                : function_(std::move(function))
            {
                static_assert(std::is_nothrow_invocable_v<Function&>);
            }

            ScopeRollback(const ScopeRollback&) = delete;
            ScopeRollback& operator=(const ScopeRollback&) = delete;
            ScopeRollback(ScopeRollback&&) = delete;
            ScopeRollback& operator=(ScopeRollback&&) = delete;
            ~ScopeRollback() noexcept
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

#ifdef _MSC_VER
#pragma warning(push)

#pragma warning(disable : 4324)
#endif

        template<typename T, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE subobjectType>
        struct alignas(void*) PipelineStateStreamSubobject final
        {
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type{subobjectType};
            T value{};
        };

        struct GraphicsPipelineStateStream final
        {
            PipelineStateStreamSubobject<ID3D12RootSignature*, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE>
                rootSignature;
            PipelineStateStreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS> vertexShader;
            PipelineStateStreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS> pixelShader;
            PipelineStateStreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS> geometryShader;
            PipelineStateStreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS> hullShader;
            PipelineStateStreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS> domainShader;
            PipelineStateStreamSubobject<D3D12_STREAM_OUTPUT_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT>
                streamOutput;
            PipelineStateStreamSubobject<D3D12_BLEND_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND> blendState;
            PipelineStateStreamSubobject<UINT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK> sampleMask;
            PipelineStateStreamSubobject<D3D12_RASTERIZER_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER>
                rasterizerState;
            PipelineStateStreamSubobject<D3D12_DEPTH_STENCIL_DESC1, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1>
                depthStencilState;
            PipelineStateStreamSubobject<D3D12_INPUT_LAYOUT_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT>
                inputLayout;
            PipelineStateStreamSubobject<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
                                         D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE>
                stripCutValue;
            PipelineStateStreamSubobject<D3D12_PRIMITIVE_TOPOLOGY_TYPE,
                                         D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY>
                primitiveTopologyType;
            PipelineStateStreamSubobject<D3D12_RT_FORMAT_ARRAY,
                                         D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS>
                renderTargetFormats;
            PipelineStateStreamSubobject<DXGI_FORMAT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT>
                depthStencilFormat;
            PipelineStateStreamSubobject<DXGI_SAMPLE_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC> sampleDesc;
            PipelineStateStreamSubobject<UINT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK> nodeMask;
            PipelineStateStreamSubobject<D3D12_CACHED_PIPELINE_STATE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO>
                cachedPipelineState;
            PipelineStateStreamSubobject<D3D12_PIPELINE_STATE_FLAGS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS> flags;
            PipelineStateStreamSubobject<D3D12_VIEW_INSTANCING_DESC,
                                         D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING>
                viewInstancing;
        };

        struct MeshPipelineStateStream final
        {
            PipelineStateStreamSubobject<ID3D12RootSignature*, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE>
                rootSignature;
            PipelineStateStreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS> taskShader;
            PipelineStateStreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS> meshShader;
            PipelineStateStreamSubobject<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS> pixelShader;
            PipelineStateStreamSubobject<D3D12_BLEND_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND> blendState;
            PipelineStateStreamSubobject<UINT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK> sampleMask;
            PipelineStateStreamSubobject<D3D12_RASTERIZER_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER>
                rasterizerState;
            PipelineStateStreamSubobject<D3D12_DEPTH_STENCIL_DESC1, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1>
                depthStencilState;
            PipelineStateStreamSubobject<D3D12_PRIMITIVE_TOPOLOGY_TYPE,
                                         D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY>
                primitiveTopologyType;
            PipelineStateStreamSubobject<D3D12_RT_FORMAT_ARRAY,
                                         D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS>
                renderTargetFormats;
            PipelineStateStreamSubobject<DXGI_FORMAT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT>
                depthStencilFormat;
            PipelineStateStreamSubobject<DXGI_SAMPLE_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC> sampleDesc;
            PipelineStateStreamSubobject<UINT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK> nodeMask;
            PipelineStateStreamSubobject<D3D12_CACHED_PIPELINE_STATE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO>
                cachedPipelineState;
            PipelineStateStreamSubobject<D3D12_PIPELINE_STATE_FLAGS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS> flags;
            PipelineStateStreamSubobject<D3D12_VIEW_INSTANCING_DESC,
                                         D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING>
                viewInstancing;
        };

        static_assert(std::is_standard_layout_v<MeshPipelineStateStream>);

#ifdef _MSC_VER
#pragma warning(pop)
#endif

        using ShaderBytecode = std::vector<std::byte>;
        using SpecializationSet = std::vector<SpecializationConstant>;
        using PipelineLayoutTable = detail::HandleTable<std::shared_ptr<PipelineLayoutRecord>, PipelineLayoutHandle>;
        using ShaderModuleTable = detail::HandleTable<std::shared_ptr<ShaderModuleRecord>, ShaderModuleHandle>;
        using PipelineCacheTable = detail::HandleTable<std::shared_ptr<PipelineCacheRecord>, PipelineCacheHandle>;

        [[nodiscard]] Result<SpecializationSet> buildSpecializationSet(
            std::span<const SpecializationConstant> constants)
        {
            if (constants.size() > maxSpecializationConstants)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A shader specialization set exceeds the portable constant-count limit.",
                });
            }

            SpecializationSet result(constants.begin(), constants.end());
            for (auto& constant : result)
            {
                switch (constant.format)
                {
                case SpecializationConstantFormat::boolean:
                    constant.valueBits = constant.valueBits != 0 ? 1U : 0U;
                    break;
                case SpecializationConstantFormat::float32:
                case SpecializationConstantFormat::signedInt32:
                case SpecializationConstantFormat::unsignedInt32:
                    break;
                default:
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A shader specialization constant has an invalid format.",
                    });
                }
            }
            std::ranges::sort(result, {}, &SpecializationConstant::id);
            if (std::ranges::adjacent_find(result, {}, &SpecializationConstant::id) != result.end())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader specialization constant IDs must be unique within one value set.",
                });
            }
            return result;
        }

        [[nodiscard]] constexpr D3D12_DEPTH_STENCIL_DESC1 depthStencilState(const D3D12_DEPTH_STENCIL_DESC& desc,
                                                                            bool depthBoundsTest) noexcept
        {
            return {
                .DepthEnable = desc.DepthEnable,
                .DepthWriteMask = desc.DepthWriteMask,
                .DepthFunc = desc.DepthFunc,
                .StencilEnable = desc.StencilEnable,
                .StencilReadMask = desc.StencilReadMask,
                .StencilWriteMask = desc.StencilWriteMask,
                .FrontFace = desc.FrontFace,
                .BackFace = desc.BackFace,
                .DepthBoundsTestEnable = depthBoundsTest ? TRUE : FALSE,
            };
        }

        [[nodiscard]] GraphicsPipelineStateStream graphicsPipelineStateStream(
            const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
            bool depthBoundsTest,
            const D3D12_VIEW_INSTANCING_DESC& viewInstancing)
        {
            D3D12_RT_FORMAT_ARRAY renderTargetFormats{};
            renderTargetFormats.NumRenderTargets = desc.NumRenderTargets;
            std::ranges::copy(desc.RTVFormats, renderTargetFormats.RTFormats);

            GraphicsPipelineStateStream stream{};
            stream.rootSignature.value = desc.pRootSignature;
            stream.vertexShader.value = desc.VS;
            stream.pixelShader.value = desc.PS;
            stream.geometryShader.value = desc.GS;
            stream.hullShader.value = desc.HS;
            stream.domainShader.value = desc.DS;
            stream.streamOutput.value = desc.StreamOutput;
            stream.blendState.value = desc.BlendState;
            stream.sampleMask.value = desc.SampleMask;
            stream.rasterizerState.value = desc.RasterizerState;
            stream.depthStencilState.value = depthStencilState(desc.DepthStencilState, depthBoundsTest);
            stream.inputLayout.value = desc.InputLayout;
            stream.stripCutValue.value = desc.IBStripCutValue;
            stream.primitiveTopologyType.value = desc.PrimitiveTopologyType;
            stream.renderTargetFormats.value = renderTargetFormats;
            stream.depthStencilFormat.value = desc.DSVFormat;
            stream.sampleDesc.value = desc.SampleDesc;
            stream.nodeMask.value = desc.NodeMask;
            stream.cachedPipelineState.value = desc.CachedPSO;
            stream.flags.value = desc.Flags;
            stream.viewInstancing.value = viewInstancing;
            return stream;
        }

        [[nodiscard]] MeshPipelineStateStream meshPipelineStateStream(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
                                                                      D3D12_SHADER_BYTECODE taskShader,
                                                                      D3D12_SHADER_BYTECODE meshShader,
                                                                      bool depthBoundsTest,
                                                                      const D3D12_VIEW_INSTANCING_DESC& viewInstancing)
        {
            D3D12_RT_FORMAT_ARRAY renderTargetFormats{};
            renderTargetFormats.NumRenderTargets = desc.NumRenderTargets;
            std::ranges::copy(desc.RTVFormats, renderTargetFormats.RTFormats);

            MeshPipelineStateStream stream{};
            stream.rootSignature.value = desc.pRootSignature;
            stream.taskShader.value = taskShader;
            stream.meshShader.value = meshShader;
            stream.pixelShader.value = desc.PS;
            stream.blendState.value = desc.BlendState;
            stream.sampleMask.value = desc.SampleMask;
            stream.rasterizerState.value = desc.RasterizerState;
            stream.depthStencilState.value = depthStencilState(desc.DepthStencilState, depthBoundsTest);
            stream.primitiveTopologyType.value = desc.PrimitiveTopologyType;
            stream.renderTargetFormats.value = renderTargetFormats;
            stream.depthStencilFormat.value = desc.DSVFormat;
            stream.sampleDesc.value = desc.SampleDesc;
            stream.nodeMask.value = desc.NodeMask;
            stream.cachedPipelineState.value = desc.CachedPSO;
            stream.flags.value = desc.Flags;
            stream.viewInstancing.value = viewInstancing;
            return stream;
        }

        [[nodiscard]] constexpr bool isSingleShaderStage(ShaderStage stage) noexcept
        {
            return stage == ShaderStage::vertex || stage == ShaderStage::pixel || stage == ShaderStage::compute ||
                   stage == ShaderStage::geometry || stage == ShaderStage::hull || stage == ShaderStage::domain ||
                   stage == ShaderStage::task || stage == ShaderStage::mesh || stage == ShaderStage::rayGeneration ||
                   stage == ShaderStage::anyHit || stage == ShaderStage::closestHit || stage == ShaderStage::miss ||
                   stage == ShaderStage::intersection || stage == ShaderStage::callable;
        }

        [[nodiscard]] constexpr bool isValidVisibility(ShaderStage visibility) noexcept
        {
            const auto value = static_cast<std::uint16_t>(visibility);
            const auto allowed = static_cast<std::uint16_t>(ShaderStage::all);
            return value != 0 && (value & ~allowed) == 0;
        }

        [[nodiscard]] constexpr bool rangesOverlap(const PushConstantRange& left,
                                                   const PushConstantRange& right) noexcept
        {
            const auto leftEnd = static_cast<std::uint64_t>(left.offset) + left.size;
            const auto rightEnd = static_cast<std::uint64_t>(right.offset) + right.size;
            return left.offset < rightEnd && right.offset < leftEnd;
        }

        [[nodiscard]] Result<std::uint32_t> validatePushConstantRanges(std::span<const PushConstantRange> ranges)
        {
            std::uint32_t byteSize = 0;
            for (std::size_t index = 0; index < ranges.size(); ++index)
            {
                const auto& range = ranges[index];
                if (!isValidVisibility(range.visibility) || range.size == 0 ||
                    range.offset % sizeof(std::uint32_t) != 0 || range.size % sizeof(std::uint32_t) != 0 ||
                    range.offset > maxPushConstantBytes || range.size > maxPushConstantBytes - range.offset)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Push-constant ranges require valid visibility and a non-empty, 4-byte aligned range within "
                        "128 bytes.",
                    });
                }
                for (std::size_t previous = 0; previous < index; ++previous)
                {
                    if (rangesOverlap(range, ranges[previous]) && hasAny(range.visibility, ranges[previous].visibility))
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "Overlapping push-constant ranges cannot be visible to the same shader stage.",
                        });
                    }
                }
                byteSize = (std::max)(byteSize, range.offset + range.size);
            }
            return byteSize;
        }

        [[nodiscard]] constexpr bool isValidShaderResourceType(ShaderResourceType type) noexcept
        {
            return type == ShaderResourceType::uniformBuffer || type == ShaderResourceType::readOnlyStorageBuffer ||
                   type == ShaderResourceType::storageBuffer || type == ShaderResourceType::sampledTexture ||
                   type == ShaderResourceType::storageTexture || type == ShaderResourceType::sampler ||
                   type == ShaderResourceType::accelerationStructure || type == ShaderResourceType::minMipFeedback ||
                   type == ShaderResourceType::mipRegionUsedFeedback;
        }

        [[nodiscard]] constexpr std::size_t shaderResourceClass(ShaderResourceType type) noexcept
        {
            switch (type)
            {
            case ShaderResourceType::uniformBuffer:
                return 0;
            case ShaderResourceType::readOnlyStorageBuffer:
            case ShaderResourceType::sampledTexture:
            case ShaderResourceType::accelerationStructure:
                return 1;
            case ShaderResourceType::storageBuffer:
            case ShaderResourceType::storageTexture:
            case ShaderResourceType::minMipFeedback:
            case ShaderResourceType::mipRegionUsedFeedback:
                return 2;
            default:
                return 3;
            }
        }

        [[nodiscard]] constexpr bool bindingRequirementsOverlap(const ShaderBindingRequirement& left,
                                                                const ShaderBindingRequirement& right) noexcept
        {
            if (left.set != right.set || shaderResourceClass(left.type) != shaderResourceClass(right.type))
            {
                return false;
            }
            const auto leftEnd = left.count == unboundedBindingCount
                                     ? std::numeric_limits<std::uint64_t>::max()
                                     : static_cast<std::uint64_t>(left.binding) + left.count;
            const auto rightEnd = right.count == unboundedBindingCount
                                      ? std::numeric_limits<std::uint64_t>::max()
                                      : static_cast<std::uint64_t>(right.binding) + right.count;
            return left.binding < rightEnd && right.binding < leftEnd;
        }

        [[nodiscard]] constexpr bool pushConstantRequirementsOverlap(
            const ShaderPushConstantRequirement& left, const ShaderPushConstantRequirement& right) noexcept
        {
            const auto leftEnd = static_cast<std::uint64_t>(left.offset) + left.size;
            const auto rightEnd = static_cast<std::uint64_t>(right.offset) + right.size;
            return left.offset < rightEnd && right.offset < leftEnd;
        }

        [[nodiscard]] constexpr bool fitsInvocationLimit(const ShaderWorkgroupSize& size, std::uint32_t limit) noexcept
        {
            if (size.x == 0 || size.y == 0 || size.z == 0 || limit == 0 || size.x > limit || size.y > limit / size.x)
            {
                return false;
            }
            const auto xy = size.x * size.y;
            return size.z <= limit / xy;
        }

        [[nodiscard]] constexpr bool isValidBindingRequirement(const ShaderBindingRequirement& binding) noexcept
        {
            const bool registerRangeOverflows =
                binding.count != unboundedBindingCount &&
                binding.binding > std::numeric_limits<std::uint32_t>::max() - (binding.count - 1);
            return binding.set < maxBindingSets && isValidShaderResourceType(binding.type) && binding.count != 0 &&
                   !registerRangeOverflows;
        }

        [[nodiscard]] Status validateInterfaceBindings(std::span<const ShaderBindingRequirement> bindings)
        {
            for (std::size_t index = 0; index < bindings.size(); ++index)
            {
                const auto& binding = bindings[index];
                if (!isValidBindingRequirement(binding))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Shader-interface bindings require a valid set, resource type, and non-empty register range.",
                    });
                }
                const bool conflictsWithEarlierBinding =
                    std::ranges::any_of(bindings.first(index), [&binding](const ShaderBindingRequirement& existing) {
                        return (existing.set == binding.set && existing.binding == binding.binding) ||
                               bindingRequirementsOverlap(existing, binding);
                    });
                if (conflictsWithEarlierBinding)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Shader-interface bindings must be unique and non-overlapping within each register class.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] constexpr bool isValidPushConstantRequirement(const ShaderPushConstantRequirement& range) noexcept
        {
            return range.size != 0 && range.offset % sizeof(std::uint32_t) == 0 &&
                   range.size % sizeof(std::uint32_t) == 0 && range.offset <= maxPushConstantBytes &&
                   range.size <= maxPushConstantBytes - range.offset;
        }

        [[nodiscard]] Status validateInterfacePushConstants(std::span<const ShaderPushConstantRequirement> ranges)
        {
            for (std::size_t index = 0; index < ranges.size(); ++index)
            {
                const auto& range = ranges[index];
                if (!isValidPushConstantRequirement(range))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Shader-interface push constants require non-empty, 4-byte aligned ranges within 128 bytes.",
                    });
                }
                const bool overlapsEarlierRange =
                    std::ranges::any_of(ranges.first(index), [&range](const ShaderPushConstantRequirement& existing) {
                        return pushConstantRequirementsOverlap(range, existing);
                    });
                if (overlapsEarlierRange)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Shader-interface push-constant ranges cannot overlap.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Result<std::vector<std::uint32_t>> validateAndSortVertexInputLocations(
            ShaderStage stage, std::span<const std::uint32_t> locations)
        {
            std::vector<std::uint32_t> sorted(locations.begin(), locations.end());
            std::ranges::sort(sorted);
            if ((stage != ShaderStage::vertex && !sorted.empty()) ||
                std::ranges::any_of(sorted, [](std::uint32_t location) { return location >= maxVertexAttributes; }) ||
                std::ranges::adjacent_find(sorted) != sorted.end())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Only vertex interfaces may declare unique portable input locations within the device limit.",
                });
            }
            return sorted;
        }

        [[nodiscard]] Status validateWorkgroupSize(ShaderStage stage,
                                                   const std::optional<ShaderWorkgroupSize>& workgroupSize,
                                                   std::uint32_t maxComputeWorkgroupInvocations,
                                                   const std::array<std::uint32_t, 3>& maxComputeWorkgroupSize,
                                                   const MeshShaderCapabilities& meshShaderCapabilities)
        {
            const bool requiresWorkgroup =
                stage == ShaderStage::compute || stage == ShaderStage::task || stage == ShaderStage::mesh;
            if (workgroupSize.has_value() != requiresWorkgroup)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Compute, task, and mesh interfaces require a workgroup size; other shader stages must omit it.",
                });
            }
            if (!workgroupSize.has_value())
            {
                return {};
            }
            const auto& size = *workgroupSize;
            if (size.x == 0 || size.y == 0 || size.z == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader-interface workgroup dimensions must be non-zero.",
                });
            }
            bool supported = false;
            if (stage == ShaderStage::compute)
            {
                supported = size.x <= maxComputeWorkgroupSize[0] && size.y <= maxComputeWorkgroupSize[1] &&
                            size.z <= maxComputeWorkgroupSize[2] &&
                            fitsInvocationLimit(size, maxComputeWorkgroupInvocations);
            }
            else if (stage == ShaderStage::task)
            {
                supported = fitsInvocationLimit(size, meshShaderCapabilities.maxTaskWorkgroupInvocations);
            }
            else
            {
                supported = fitsInvocationLimit(size, meshShaderCapabilities.maxMeshWorkgroupInvocations);
            }
            if (!supported)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The shader-interface workgroup size exceeds the device's stage limits.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateShaderRecordConstants(ShaderStage stage,
                                                           const std::optional<ShaderRecordConstantRange>& constants)
        {
            if (!constants.has_value())
            {
                return {};
            }
            if (!hasAny(stage, ShaderStage::allRayTracing) || constants->size == 0 ||
                constants->size % sizeof(std::uint32_t) != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader-record constants require a ray-tracing stage and a non-empty, 4-byte aligned range.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<ShaderInterfaceRecord> buildShaderInterface(
            ShaderStage stage,
            const ShaderInterfaceDesc& desc,
            std::uint32_t maxComputeWorkgroupInvocations,
            const std::array<std::uint32_t, 3>& maxComputeWorkgroupSize,
            const MeshShaderCapabilities& meshShaderCapabilities)
        {
            auto bindingValidation = validateInterfaceBindings(desc.bindings);
            if (!bindingValidation)
            {
                return std::unexpected(std::move(bindingValidation.error()));
            }
            auto pushConstantValidation = validateInterfacePushConstants(desc.pushConstants);
            if (!pushConstantValidation)
            {
                return std::unexpected(std::move(pushConstantValidation.error()));
            }
            auto vertexInputLocations = validateAndSortVertexInputLocations(stage, desc.vertexInputLocations);
            if (!vertexInputLocations)
            {
                return std::unexpected(std::move(vertexInputLocations.error()));
            }
            auto workgroupValidation = validateWorkgroupSize(stage,
                                                             desc.workgroupSize,
                                                             maxComputeWorkgroupInvocations,
                                                             maxComputeWorkgroupSize,
                                                             meshShaderCapabilities);
            if (!workgroupValidation)
            {
                return std::unexpected(std::move(workgroupValidation.error()));
            }
            auto recordConstantValidation = validateShaderRecordConstants(stage, desc.shaderRecordConstants);
            if (!recordConstantValidation)
            {
                return std::unexpected(std::move(recordConstantValidation.error()));
            }

            ShaderInterfaceRecord record;
            record.bindings.assign(desc.bindings.begin(), desc.bindings.end());
            record.pushConstants.assign(desc.pushConstants.begin(), desc.pushConstants.end());
            record.vertexInputLocations = std::move(*vertexInputLocations);
            record.workgroupSize = desc.workgroupSize;
            record.shaderRecordConstants = desc.shaderRecordConstants;
            record.drawIndex = desc.drawIndex;
            return record;
        }

        [[nodiscard]] Status validateDrawIndexBytecode(std::span<const std::byte> bytecode, bool declared)
        {
            const auto binds = dxil::bindsConstantBuffer(bytecode, drawIndexRegister, drawIndexRegisterSpace);
            if (!binds || !binds->has_value() || **binds == declared)
            {
                return {};
            }
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                declared ? "The shader interface declares the draw index, but the DXIL binds no constant buffer at "
                           "register b0, space5."
                         : "The DXIL binds the draw-index constant buffer at register b0, space5, but the shader "
                           "interface does not declare the draw index.",
            });
        }

        [[nodiscard]] Status validateVertexInterface(const ShaderModuleRecord& shader,
                                                     std::span<const D3D12_INPUT_ELEMENT_DESC> inputElements)
        {
            std::vector<std::uint32_t> declaredLocations;
            declaredLocations.reserve(inputElements.size());
            std::ranges::transform(inputElements, std::back_inserter(declaredLocations), [](const auto& element) {
                return static_cast<std::uint32_t>(element.SemanticIndex);
            });
            std::ranges::sort(declaredLocations);
            if (declaredLocations != shader.shaderInterface.vertexInputLocations)
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "The graphics pipeline's vertex locations do not exactly match the shader interface.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr D3D12_SHADER_VISIBILITY shaderVisibility(ShaderStage combined) noexcept
        {
            if (combined == ShaderStage::vertex)
            {
                return D3D12_SHADER_VISIBILITY_VERTEX;
            }
            if (combined == ShaderStage::pixel)
            {
                return D3D12_SHADER_VISIBILITY_PIXEL;
            }
            if (combined == ShaderStage::geometry)
            {
                return D3D12_SHADER_VISIBILITY_GEOMETRY;
            }
            if (combined == ShaderStage::hull)
            {
                return D3D12_SHADER_VISIBILITY_HULL;
            }
            if (combined == ShaderStage::domain)
            {
                return D3D12_SHADER_VISIBILITY_DOMAIN;
            }
            if (combined == ShaderStage::task)
            {
                return D3D12_SHADER_VISIBILITY_AMPLIFICATION;
            }
            if (combined == ShaderStage::mesh)
            {
                return D3D12_SHADER_VISIBILITY_MESH;
            }
            return D3D12_SHADER_VISIBILITY_ALL;
        }

        [[nodiscard]] constexpr D3D12_ROOT_PARAMETER_TYPE nativeRootDescriptorParameterType(
            RootDescriptorKind kind) noexcept
        {
            switch (kind)
            {
            case RootDescriptorKind::cbv:
                return D3D12_ROOT_PARAMETER_TYPE_CBV;
            case RootDescriptorKind::srv:
                return D3D12_ROOT_PARAMETER_TYPE_SRV;
            case RootDescriptorKind::uav:
                return D3D12_ROOT_PARAMETER_TYPE_UAV;
            }
            return D3D12_ROOT_PARAMETER_TYPE_CBV;
        }

        [[nodiscard]] constexpr D3D12_DESCRIPTOR_RANGE_FLAGS resourceRangeFlags() noexcept
        {
            return D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS |
                   D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        }

        [[nodiscard]] constexpr D3D12_DESCRIPTOR_RANGE_FLAGS resourceRangeFlags(
            const BindingLayoutEntry& entry) noexcept
        {
            if (hasAny(entry.flags, BindingFlag::updateAfterBind))
            {
                return D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
            }
            return resourceRangeFlags();
        }

        [[nodiscard]] constexpr DXGI_FORMAT nativeFormat(Format value) noexcept
        {
            return format::view(value);
        }

        [[nodiscard]] constexpr D3D12_PRIMITIVE_TOPOLOGY_TYPE nativeTopologyType(PrimitiveTopology topology) noexcept
        {
            switch (topology)
            {
            case PrimitiveTopology::pointList:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
            case PrimitiveTopology::lineList:
            case PrimitiveTopology::lineStrip:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            case PrimitiveTopology::triangleList:
            case PrimitiveTopology::triangleStrip:
            case PrimitiveTopology::triangleFan:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            case PrimitiveTopology::patchList:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
            }
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
        }

        [[nodiscard]] constexpr D3D12_PRIMITIVE_TOPOLOGY nativeTopology(PrimitiveTopology topology,
                                                                        std::uint32_t patchControlPoints) noexcept
        {
            switch (topology)
            {
            case PrimitiveTopology::pointList:
                return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
            case PrimitiveTopology::lineList:
                return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
            case PrimitiveTopology::lineStrip:
                return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
            case PrimitiveTopology::triangleList:
                return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            case PrimitiveTopology::triangleStrip:
                return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
            case PrimitiveTopology::triangleFan:
                return D3D_PRIMITIVE_TOPOLOGY_TRIANGLEFAN;
            case PrimitiveTopology::patchList:
                if (patchControlPoints >= 1 && patchControlPoints <= 32)
                {
                    return static_cast<D3D12_PRIMITIVE_TOPOLOGY>(
                        static_cast<std::uint32_t>(D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST) +
                        patchControlPoints - 1U);
                }
                return D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
            }
            return D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
        }

        [[nodiscard]] constexpr bool isStripTopology(PrimitiveTopology topology) noexcept
        {
            return topology == PrimitiveTopology::lineStrip || topology == PrimitiveTopology::triangleStrip;
        }

        [[nodiscard]] constexpr D3D12_INDEX_BUFFER_STRIP_CUT_VALUE nativeStripCutValue(
            const StripIndexFormat& format) noexcept
        {
            if (!format.has_value())
            {
                return D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
            }
            switch (*format)
            {
            case IndexFormat::uint16:
                return D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
            case IndexFormat::uint32:
                return D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
            }
            return D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
        }

        [[nodiscard]] constexpr D3D12_CULL_MODE nativeCullMode(CullMode mode) noexcept
        {
            switch (mode)
            {
            case CullMode::none:
                return D3D12_CULL_MODE_NONE;
            case CullMode::front:
                return D3D12_CULL_MODE_FRONT;
            case CullMode::back:
                return D3D12_CULL_MODE_BACK;
            }
            return D3D12_CULL_MODE_NONE;
        }

        [[nodiscard]] constexpr D3D12_FILL_MODE nativeFillMode(FillMode mode) noexcept
        {
            switch (mode)
            {
            case FillMode::solid:
                return D3D12_FILL_MODE_SOLID;
            case FillMode::wireframe:
                return D3D12_FILL_MODE_WIREFRAME;
            }
            return D3D12_FILL_MODE_SOLID;
        }

        [[nodiscard]] constexpr bool isValidFillMode(FillMode mode) noexcept
        {
            return mode == FillMode::solid || mode == FillMode::wireframe;
        }

        [[nodiscard]] constexpr bool isValidConservativeRasterizationMode(ConservativeRasterizationMode mode) noexcept
        {
            return mode == ConservativeRasterizationMode::disabled ||
                   mode == ConservativeRasterizationMode::overestimate;
        }

        [[nodiscard]] constexpr D3D12_CONSERVATIVE_RASTERIZATION_MODE nativeConservativeRasterizationMode(
            ConservativeRasterizationMode mode) noexcept
        {
            return mode == ConservativeRasterizationMode::overestimate ? D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON
                                                                       : D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
        }

        [[nodiscard]] constexpr D3D12_COMPARISON_FUNC nativeCompare(CompareOperation operation) noexcept
        {
            switch (operation)
            {
            case CompareOperation::never:
                return D3D12_COMPARISON_FUNC_NEVER;
            case CompareOperation::less:
                return D3D12_COMPARISON_FUNC_LESS;
            case CompareOperation::equal:
                return D3D12_COMPARISON_FUNC_EQUAL;
            case CompareOperation::lessEqual:
                return D3D12_COMPARISON_FUNC_LESS_EQUAL;
            case CompareOperation::greater:
                return D3D12_COMPARISON_FUNC_GREATER;
            case CompareOperation::notEqual:
                return D3D12_COMPARISON_FUNC_NOT_EQUAL;
            case CompareOperation::greaterEqual:
                return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
            case CompareOperation::always:
                return D3D12_COMPARISON_FUNC_ALWAYS;
            }
            return D3D12_COMPARISON_FUNC_ALWAYS;
        }

        [[nodiscard]] constexpr bool isValidCompare(CompareOperation operation) noexcept
        {
            return operation >= CompareOperation::never && operation <= CompareOperation::always;
        }

        [[nodiscard]] constexpr bool isValidBlendFactor(BlendFactor factor) noexcept
        {
            return factor >= BlendFactor::zero && factor <= BlendFactor::oneMinusSource1Alpha;
        }

        [[nodiscard]] constexpr bool isValidBlendOperation(BlendOperation operation) noexcept
        {
            return operation >= BlendOperation::add && operation <= BlendOperation::maximum;
        }

        [[nodiscard]] constexpr bool isValidLogicOperation(LogicOperation operation) noexcept
        {
            return operation >= LogicOperation::clear && operation <= LogicOperation::set;
        }

        [[nodiscard]] constexpr D3D12_LOGIC_OP nativeLogicOperation(LogicOperation operation) noexcept
        {
            switch (operation)
            {
            case LogicOperation::clear:
                return D3D12_LOGIC_OP_CLEAR;
            case LogicOperation::andOperation:
                return D3D12_LOGIC_OP_AND;
            case LogicOperation::andReverse:
                return D3D12_LOGIC_OP_AND_REVERSE;
            case LogicOperation::copy:
                return D3D12_LOGIC_OP_COPY;
            case LogicOperation::andInverted:
                return D3D12_LOGIC_OP_AND_INVERTED;
            case LogicOperation::noOp:
                return D3D12_LOGIC_OP_NOOP;
            case LogicOperation::xorOperation:
                return D3D12_LOGIC_OP_XOR;
            case LogicOperation::orOperation:
                return D3D12_LOGIC_OP_OR;
            case LogicOperation::nor:
                return D3D12_LOGIC_OP_NOR;
            case LogicOperation::equivalent:
                return D3D12_LOGIC_OP_EQUIV;
            case LogicOperation::invert:
                return D3D12_LOGIC_OP_INVERT;
            case LogicOperation::orReverse:
                return D3D12_LOGIC_OP_OR_REVERSE;
            case LogicOperation::copyInverted:
                return D3D12_LOGIC_OP_COPY_INVERTED;
            case LogicOperation::orInverted:
                return D3D12_LOGIC_OP_OR_INVERTED;
            case LogicOperation::nand:
                return D3D12_LOGIC_OP_NAND;
            case LogicOperation::set:
                return D3D12_LOGIC_OP_SET;
            }
            return D3D12_LOGIC_OP_NOOP;
        }

        [[nodiscard]] bool supportsLogicOperation(ID3D12Device& device, Format format) noexcept
        {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{
                .Format = nativeFormat(format),
                .Support1 = {},
                .Support2 = {},
            };
            return support.Format != DXGI_FORMAT_UNKNOWN &&
                   SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) &&
                   (support.Support2 & D3D12_FORMAT_SUPPORT2_OUTPUT_MERGER_LOGIC_OP) != 0;
        }

        [[nodiscard]] constexpr D3D12_BLEND nativeColorBlendFactor(BlendFactor factor) noexcept
        {
            switch (factor)
            {
            case BlendFactor::zero:
                return D3D12_BLEND_ZERO;
            case BlendFactor::one:
                return D3D12_BLEND_ONE;
            case BlendFactor::source:
                return D3D12_BLEND_SRC_COLOR;
            case BlendFactor::oneMinusSource:
                return D3D12_BLEND_INV_SRC_COLOR;
            case BlendFactor::sourceAlpha:
                return D3D12_BLEND_SRC_ALPHA;
            case BlendFactor::oneMinusSourceAlpha:
                return D3D12_BLEND_INV_SRC_ALPHA;
            case BlendFactor::destination:
                return D3D12_BLEND_DEST_COLOR;
            case BlendFactor::oneMinusDestination:
                return D3D12_BLEND_INV_DEST_COLOR;
            case BlendFactor::destinationAlpha:
                return D3D12_BLEND_DEST_ALPHA;
            case BlendFactor::oneMinusDestinationAlpha:
                return D3D12_BLEND_INV_DEST_ALPHA;
            case BlendFactor::sourceAlphaSaturated:
                return D3D12_BLEND_SRC_ALPHA_SAT;
            case BlendFactor::constant:
                return D3D12_BLEND_BLEND_FACTOR;
            case BlendFactor::oneMinusConstant:
                return D3D12_BLEND_INV_BLEND_FACTOR;
            case BlendFactor::source1:
                return D3D12_BLEND_SRC1_COLOR;
            case BlendFactor::oneMinusSource1:
                return D3D12_BLEND_INV_SRC1_COLOR;
            case BlendFactor::source1Alpha:
                return D3D12_BLEND_SRC1_ALPHA;
            case BlendFactor::oneMinusSource1Alpha:
                return D3D12_BLEND_INV_SRC1_ALPHA;
            }
            return D3D12_BLEND_ZERO;
        }

        [[nodiscard]] constexpr D3D12_BLEND nativeAlphaBlendFactor(BlendFactor factor) noexcept
        {
            switch (factor)
            {
            case BlendFactor::zero:
                return D3D12_BLEND_ZERO;
            case BlendFactor::one:
                return D3D12_BLEND_ONE;
            case BlendFactor::source:
            case BlendFactor::sourceAlpha:
                return D3D12_BLEND_SRC_ALPHA;
            case BlendFactor::oneMinusSource:
            case BlendFactor::oneMinusSourceAlpha:
                return D3D12_BLEND_INV_SRC_ALPHA;
            case BlendFactor::destination:
            case BlendFactor::destinationAlpha:
                return D3D12_BLEND_DEST_ALPHA;
            case BlendFactor::oneMinusDestination:
            case BlendFactor::oneMinusDestinationAlpha:
                return D3D12_BLEND_INV_DEST_ALPHA;
            case BlendFactor::sourceAlphaSaturated:
                return D3D12_BLEND_ONE;
            case BlendFactor::constant:
                return D3D12_BLEND_BLEND_FACTOR;
            case BlendFactor::oneMinusConstant:
                return D3D12_BLEND_INV_BLEND_FACTOR;
            case BlendFactor::source1:
            case BlendFactor::source1Alpha:
                return D3D12_BLEND_SRC1_ALPHA;
            case BlendFactor::oneMinusSource1:
            case BlendFactor::oneMinusSource1Alpha:
                return D3D12_BLEND_INV_SRC1_ALPHA;
            }
            return D3D12_BLEND_ZERO;
        }

        [[nodiscard]] constexpr D3D12_BLEND_OP nativeBlendOperation(BlendOperation operation) noexcept
        {
            switch (operation)
            {
            case BlendOperation::add:
                return D3D12_BLEND_OP_ADD;
            case BlendOperation::subtract:
                return D3D12_BLEND_OP_SUBTRACT;
            case BlendOperation::reverseSubtract:
                return D3D12_BLEND_OP_REV_SUBTRACT;
            case BlendOperation::minimum:
                return D3D12_BLEND_OP_MIN;
            case BlendOperation::maximum:
                return D3D12_BLEND_OP_MAX;
            }
            return D3D12_BLEND_OP_ADD;
        }

        [[nodiscard]] constexpr bool usesBlendConstant(BlendFactor factor) noexcept
        {
            return factor == BlendFactor::constant || factor == BlendFactor::oneMinusConstant;
        }

        [[nodiscard]] constexpr bool usesDualSource(BlendFactor factor) noexcept
        {
            return factor >= BlendFactor::source1 && factor <= BlendFactor::oneMinusSource1Alpha;
        }

        [[nodiscard]] constexpr bool isValidStencilOperation(StencilOperation operation) noexcept
        {
            return operation >= StencilOperation::keep && operation <= StencilOperation::decrementWrap;
        }

        [[nodiscard]] constexpr D3D12_STENCIL_OP nativeStencilOperation(StencilOperation operation) noexcept
        {
            switch (operation)
            {
            case StencilOperation::keep:
                return D3D12_STENCIL_OP_KEEP;
            case StencilOperation::zero:
                return D3D12_STENCIL_OP_ZERO;
            case StencilOperation::replace:
                return D3D12_STENCIL_OP_REPLACE;
            case StencilOperation::incrementClamp:
                return D3D12_STENCIL_OP_INCR_SAT;
            case StencilOperation::decrementClamp:
                return D3D12_STENCIL_OP_DECR_SAT;
            case StencilOperation::invert:
                return D3D12_STENCIL_OP_INVERT;
            case StencilOperation::incrementWrap:
                return D3D12_STENCIL_OP_INCR;
            case StencilOperation::decrementWrap:
                return D3D12_STENCIL_OP_DECR;
            }
            return D3D12_STENCIL_OP_KEEP;
        }

        [[nodiscard]] constexpr bool isValidStencilFace(const StencilFaceState& face) noexcept
        {
            return isValidCompare(face.compare) && isValidStencilOperation(face.failOperation) &&
                   isValidStencilOperation(face.depthFailOperation) && isValidStencilOperation(face.passOperation);
        }

        [[nodiscard]] constexpr D3D12_DEPTH_STENCILOP_DESC nativeStencilFace(const StencilFaceState& face) noexcept
        {
            return D3D12_DEPTH_STENCILOP_DESC{
                nativeStencilOperation(face.failOperation),
                nativeStencilOperation(face.depthFailOperation),
                nativeStencilOperation(face.passOperation),
                nativeCompare(face.compare),
            };
        }

        [[nodiscard]] constexpr bool isValidPipelineCachePolicy(PipelineCachePolicy policy) noexcept
        {
            switch (policy)
            {
            case PipelineCachePolicy::populate:
            case PipelineCachePolicy::requireHit:
                return true;
            }
            return false;
        }

        [[nodiscard]] std::wstring pipelineCacheEntryName(PipelineCacheKey key, PipelineCacheEntryKind kind)
        {
            std::wostringstream name;
            const wchar_t* prefix = L"rhi-m-";
            if (kind == PipelineCacheEntryKind::compute)
            {
                prefix = L"rhi-c-";
            }
            else if (kind == PipelineCacheEntryKind::graphics)
            {
                prefix = L"rhi-g-";
            }
            name << prefix << std::hex << std::setfill(L'0') << std::setw(16) << key.high << std::setw(16) << key.low;
            return name.str();
        }

        inline constexpr std::uint64_t pipelineCacheBlobMagic = 0x3130334443494852ULL;
        inline constexpr std::uint32_t pipelineCacheBlobVersion = 2;

        struct PipelineCacheBlobHeader final
        {
            std::uint64_t magic = pipelineCacheBlobMagic;
            std::uint64_t nativeDataSize = 0;
            std::uint32_t version = pipelineCacheBlobVersion;
            std::uint32_t entryCount = 0;
        };

        static_assert(sizeof(PipelineCacheBlobHeader) == 24);
        static_assert(sizeof(PipelineCacheEntry) == 24);

        struct DecodedPipelineCache final
        {
            std::span<const std::byte> nativeData;
            std::set<PipelineCacheEntry> entries;
        };

        [[nodiscard]] Result<DecodedPipelineCache> decodePipelineCache(std::span<const std::byte> data)
        {
            if (data.empty())
            {
                return DecodedPipelineCache{};
            }
            if (data.size() < sizeof(PipelineCacheBlobHeader))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline cache blob is smaller than the RHI header.",
                });
            }

            PipelineCacheBlobHeader header;
            std::memcpy(&header, data.data(), sizeof(header));
            if (header.magic != pipelineCacheBlobMagic || header.version != pipelineCacheBlobVersion)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline cache blob has an incompatible RHI magic or schema version.",
                });
            }

            const auto remaining = data.size() - sizeof(header);
            if (header.entryCount > remaining / sizeof(PipelineCacheEntry))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline cache blob entry table exceeds the supplied data.",
                });
            }
            const auto entriesSize = static_cast<std::size_t>(header.entryCount) * sizeof(PipelineCacheEntry);
            const auto nativeOffset = sizeof(header) + entriesSize;
            if (header.nativeDataSize != data.size() - nativeOffset)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline cache blob has an inconsistent native payload size.",
                });
            }

            DecodedPipelineCache decoded;
            for (std::uint32_t index = 0; index < header.entryCount; ++index)
            {
                PipelineCacheEntry entry;
                std::memcpy(&entry, data.data() + sizeof(header) + index * sizeof(entry), sizeof(entry));
                if (!entry.key.isValid() || entry.reserved != 0 ||
                    (entry.kind != PipelineCacheEntryKind::compute && entry.kind != PipelineCacheEntryKind::graphics &&
                     entry.kind != PipelineCacheEntryKind::mesh) ||
                    !decoded.entries.insert(entry).second)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "The pipeline cache blob contains an invalid or duplicate entry.",
                    });
                }
            }
            decoded.nativeData = data.subspan(nativeOffset);
            return decoded;
        }

        [[nodiscard]] Error pipelineLibraryError(HRESULT result, std::string_view operation)
        {
            auto code = classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result));
            if (result == E_INVALIDARG || result == D3D12_ERROR_ADAPTER_NOT_FOUND ||
                result == D3D12_ERROR_DRIVER_VERSION_MISMATCH)
            {
                code = ErrorCode::invalidArgument;
            }
            else if (result == E_NOINTERFACE || result == E_NOTIMPL)
            {
                code = ErrorCode::unsupported;
            }
            return nativeError(code, result, operation);
        }

        [[nodiscard]] constexpr ErrorCode pipelineCreationErrorCode(HRESULT result) noexcept
        {
            if (result == E_INVALIDARG)
            {
                return ErrorCode::validationError;
            }
            return classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result));
        }

        [[nodiscard]] Status validateAttachmentSampleCount(ID3D12Device& device,
                                                           Format format,
                                                           std::uint32_t sampleCount)
        {
            if (sampleCount == 1)
            {
                return {};
            }
            const auto native = nativeFormat(format);
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{
                .Format = native,
                .Support1 = {},
                .Support2 = {},
            };
            if (native == DXGI_FORMAT_UNKNOWN ||
                FAILED(device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
                (support.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET) == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "A graphics attachment format does not support multisampled rendering.",
                });
            }
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality{
                .Format = native,
                .SampleCount = sampleCount,
                .Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
                .NumQualityLevels = 0,
            };
            if (FAILED(
                    device.CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality, sizeof(quality))) ||
                quality.NumQualityLevels == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "A graphics attachment format does not support the requested sample count.",
                });
            }
            return {};
        }

        static_assert(unpromotedRootParameter == invalidRootParameter);

        [[nodiscard]] bool hasRayTracingVisibility(std::span<const NativeBindingLayout> layouts,
                                                   std::span<const PushConstantRange> pushConstantRanges) noexcept
        {
            if (std::ranges::any_of(pushConstantRanges, [](const PushConstantRange& range) {
                    return hasAny(range.visibility, ShaderStage::allRayTracing);
                }))
            {
                return true;
            }
            return std::ranges::any_of(layouts, [](const NativeBindingLayout& layout) {
                return std::ranges::any_of(layout.entries, [](const NativeBindingLayoutEntry& entry) {
                    return hasAny(entry.entry.visibility, ShaderStage::allRayTracing);
                });
            });
        }

        [[nodiscard]] RootSignaturePlan planLayoutRootSignature(std::span<const NativeBindingLayout> layouts,
                                                                std::uint32_t pushConstantSize,
                                                                bool drawIndex)
        {
            std::array<std::vector<RootSignatureSlotEntryInput>, maxBindingSets> planEntries;
            std::array<RootSignatureSlotInput, maxBindingSets> planSlots{};
            for (std::size_t setIndex = 0; setIndex < layouts.size(); ++setIndex)
            {
                planEntries[setIndex].reserve(layouts[setIndex].entries.size());
                for (const auto& entry : layouts[setIndex].entries)
                {
                    planEntries[setIndex].push_back(RootSignatureSlotEntryInput{
                        .entry = entry.entry,
                        .descriptorOffset = entry.descriptorOffset,
                    });
                }
                planSlots[setIndex] = RootSignatureSlotInput{.entries = planEntries[setIndex]};
            }
            return planRootSignature(
                std::span<const RootSignatureSlotInput>{planSlots.data(), layouts.size()}, pushConstantSize, drawIndex);
        }

        struct DescriptorTable final
        {
            std::vector<D3D12_DESCRIPTOR_RANGE1> ranges;
            ShaderStage visibility = ShaderStage::none;

            void add(const NativeBindingLayoutEntry& entry,
                     std::size_t setIndex,
                     std::uint32_t offset,
                     D3D12_DESCRIPTOR_RANGE_FLAGS flags)
            {
                ranges.push_back(D3D12_DESCRIPTOR_RANGE1{
                    .RangeType = entry.rangeType,
                    .NumDescriptors = entry.entry.count,
                    .BaseShaderRegister = entry.entry.binding,
                    .RegisterSpace = static_cast<UINT>(setIndex),
                    .Flags = flags,
                    .OffsetInDescriptorsFromTableStart = offset,
                });
                visibility = visibility | entry.entry.visibility;
            }
        };

        struct DescriptorTableRanges final
        {
            std::array<DescriptorTable, maxBindingSets> resources;
            std::array<DescriptorTable, maxBindingSets> dynamicResources;
            std::array<DescriptorTable, maxBindingSets> samplers;
        };

        void addSlotDescriptorRanges(const NativeBindingLayout& layout,
                                     const RootSignatureSlotPlan& slotPlan,
                                     std::size_t setIndex,
                                     DescriptorTableRanges& ranges)
        {
            std::uint32_t dynamicCursor = 0;
            for (std::size_t entryIndex = 0; entryIndex < layout.entries.size(); ++entryIndex)
            {
                const auto& entry = layout.entries[entryIndex];
                const bool isDynamic = isDynamicBufferBindingType(entry.entry.type);
                const auto firstDynamicIndex = dynamicCursor;
                if (isDynamic)
                {
                    dynamicCursor += entry.entry.count;
                }
                if (entryIndex < slotPlan.entryPromoted.size() && slotPlan.entryPromoted[entryIndex] != 0)
                {
                    continue;
                }
                if (entry.entry.type == BindingType::sampler)
                {
                    ranges.samplers[setIndex].add(
                        entry, setIndex, entry.descriptorOffset, D3D12_DESCRIPTOR_RANGE_FLAG_NONE);
                }
                else if (!isDynamic)
                {
                    ranges.resources[setIndex].add(
                        entry, setIndex, entry.descriptorOffset, resourceRangeFlags(entry.entry));
                }
                else
                {
                    auto& table = slotPlan.resourceTableIsPerBind() ? ranges.resources[setIndex]
                                                                    : ranges.dynamicResources[setIndex];
                    table.add(entry, setIndex, slotPlan.dynamicTableOffsets[firstDynamicIndex], resourceRangeFlags());
                }
            }
        }

        [[nodiscard]] DescriptorTableRanges buildDescriptorTableRanges(std::span<const NativeBindingLayout> layouts,
                                                                       const RootSignaturePlan& rootPlan)
        {
            DescriptorTableRanges ranges;
            for (std::size_t setIndex = 0; setIndex < layouts.size(); ++setIndex)
            {
                addSlotDescriptorRanges(layouts[setIndex], rootPlan.slots[setIndex], setIndex, ranges);
            }
            return ranges;
        }

        [[nodiscard]] D3D12_ROOT_PARAMETER1 nativePushConstantParameter(std::uint32_t pushConstantSize) noexcept
        {
            return {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                .Constants =
                    {
                        .ShaderRegister = pushConstantRegister,
                        .RegisterSpace = pushConstantRegisterSpace,
                        .Num32BitValues = static_cast<UINT>(pushConstantSize / sizeof(std::uint32_t)),
                    },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
            };
        }

        [[nodiscard]] constexpr D3D12_ROOT_PARAMETER1 nativeDrawIndexParameter() noexcept
        {
            return {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                .Constants =
                    {
                        .ShaderRegister = drawIndexRegister,
                        .RegisterSpace = drawIndexRegisterSpace,
                        .Num32BitValues = 1,
                    },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
            };
        }

        [[nodiscard]] D3D12_ROOT_PARAMETER1 nativeRootDescriptorParameter(const PromotedDynamicRootDescriptor& promoted,
                                                                          bool hasRayTracingStage) noexcept
        {
            return {
                .ParameterType = nativeRootDescriptorParameterType(promoted.kind),
                .Descriptor =
                    {
                        .ShaderRegister = promoted.shaderRegister,
                        .RegisterSpace = promoted.setIndex,
                        .Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                    },
                .ShaderVisibility =
                    hasRayTracingStage ? D3D12_SHADER_VISIBILITY_ALL : shaderVisibility(promoted.visibility),
            };
        }

        [[nodiscard]] D3D12_ROOT_PARAMETER1 nativeDescriptorTableParameter(const DescriptorTable& table,
                                                                           bool hasRayTracingStage) noexcept
        {
            return {
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                .DescriptorTable =
                    {
                        .NumDescriptorRanges = static_cast<UINT>(table.ranges.size()),
                        .pDescriptorRanges = table.ranges.data(),
                    },
                .ShaderVisibility =
                    hasRayTracingStage ? D3D12_SHADER_VISIBILITY_ALL : shaderVisibility(table.visibility),
            };
        }

        [[nodiscard]] NativePipelineLayoutSlot makeLayoutSlot(const NativeBindingLayout& layout,
                                                              const RootSignatureSlotPlan& slotPlan)
        {
            NativePipelineLayoutSlot slot{
                .bindingLayoutOwner = layout.owner,
                .rootPlan = slotPlan,
                .dynamicBufferCount = layout.dynamicBufferCount,
                .requiresBindingSet = layout.resourceDescriptorCount != 0 || layout.samplerDescriptorCount != 0 ||
                                      layout.hasVariableDescriptorCount,
            };
            slot.entries.reserve(layout.entries.size());
            std::ranges::transform(layout.entries,
                                   std::back_inserter(slot.entries),
                                   [](const NativeBindingLayoutEntry& entry) { return entry.entry; });
            return slot;
        }

        struct RootParameterLayout final
        {
            std::vector<D3D12_ROOT_PARAMETER1> parameters;
            std::vector<NativePipelineLayoutSlot> slots;
            std::uint32_t pushConstantRootParameter = invalidRootParameter;
            std::uint32_t drawIndexRootParameter = invalidRootParameter;
        };

        [[nodiscard]] RootParameterLayout buildRootParameters(std::span<const NativeBindingLayout> layouts,
                                                              const RootSignaturePlan& rootPlan,
                                                              const DescriptorTableRanges& tableRanges,
                                                              std::uint32_t pushConstantSize,
                                                              bool hasRayTracingStage)
        {
            RootParameterLayout result;
            result.slots.reserve(layouts.size());
            result.parameters.reserve(rootPlan.totalRootParameters);
            if (pushConstantSize != 0)
            {
                result.pushConstantRootParameter = static_cast<std::uint32_t>(result.parameters.size());
                result.parameters.push_back(nativePushConstantParameter(pushConstantSize));
            }
            if (rootPlan.drawIndexDwords != 0)
            {
                result.drawIndexRootParameter = static_cast<std::uint32_t>(result.parameters.size());
                result.parameters.push_back(nativeDrawIndexParameter());
            }
            for (const auto& promoted : rootPlan.promotedRootDescriptors)
            {
                result.parameters.push_back(nativeRootDescriptorParameter(promoted, hasRayTracingStage));
            }
            for (std::size_t setIndex = 0; setIndex < layouts.size(); ++setIndex)
            {
                const auto& layout = layouts[setIndex];
                const auto& slotPlan = rootPlan.slots[setIndex];
                auto slot = makeLayoutSlot(layout, slotPlan);
                if (!tableRanges.resources[setIndex].ranges.empty())
                {
                    slot.resourceRootParameter = static_cast<std::uint32_t>(result.parameters.size());
                    result.parameters.push_back(
                        nativeDescriptorTableParameter(tableRanges.resources[setIndex], hasRayTracingStage));
                }
                if (!tableRanges.samplers[setIndex].ranges.empty())
                {
                    slot.samplerRootParameter = static_cast<std::uint32_t>(result.parameters.size());
                    result.parameters.push_back(
                        nativeDescriptorTableParameter(tableRanges.samplers[setIndex], hasRayTracingStage));
                }
                if (!tableRanges.dynamicResources[setIndex].ranges.empty())
                {
                    slot.dynamicRootParameter = static_cast<std::uint32_t>(result.parameters.size());
                    result.parameters.push_back(
                        nativeDescriptorTableParameter(tableRanges.dynamicResources[setIndex], hasRayTracingStage));
                }
                result.slots.push_back(std::move(slot));
            }
            return result;
        }

        [[nodiscard]] Result<ComPtr<ID3DBlob>> serializeRootSignature(
            std::span<const D3D12_ROOT_PARAMETER1> rootParameters)
        {
            const D3D12_VERSIONED_ROOT_SIGNATURE_DESC nativeDesc{
                .Version = D3D_ROOT_SIGNATURE_VERSION_1_1,
                .Desc_1_1 =
                    {
                        .NumParameters = static_cast<UINT>(rootParameters.size()),
                        .pParameters = rootParameters.data(),
                        .NumStaticSamplers = 0,
                        .pStaticSamplers = nullptr,
                        .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                                 D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT,
                    },
            };
            ComPtr<ID3DBlob> serialized;
            ComPtr<ID3DBlob> diagnostics;
            const HRESULT serializeResult =
                D3D12SerializeVersionedRootSignature(&nativeDesc, &serialized, &diagnostics);
            if (FAILED(serializeResult))
            {
                std::string message = "D3D12SerializeVersionedRootSignature failed.";
                if (diagnostics != nullptr && diagnostics->GetBufferPointer() != nullptr)
                {
                    message.append(" ");
                    message.append(static_cast<const char*>(diagnostics->GetBufferPointer()),
                                   diagnostics->GetBufferSize());
                }
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    static_cast<std::int64_t>(serializeResult),
                    std::move(message),
                });
            }
            return serialized;
        }

        [[nodiscard]] Result<std::shared_ptr<PipelineLayoutRecord>> buildPipelineLayout(
            ID3D12Device& device, std::vector<NativeBindingLayout> layouts, const PipelineLayoutDesc& desc)
        {
            auto pushConstantSize = validatePushConstantRanges(desc.pushConstantRanges);
            if (!pushConstantSize)
            {
                return std::unexpected(std::move(pushConstantSize.error()));
            }
            const bool hasRayTracingStage = hasRayTracingVisibility(layouts, desc.pushConstantRanges);
            const auto rootPlan = planLayoutRootSignature(layouts, *pushConstantSize, desc.drawIndex);
            const auto tableRanges = buildDescriptorTableRanges(layouts, rootPlan);

            auto record = std::make_shared<PipelineLayoutRecord>();
            record->label = desc.label;
            record->pushConstantRanges.assign(desc.pushConstantRanges.begin(), desc.pushConstantRanges.end());
            record->pushConstantSize = *pushConstantSize;
            auto rootParameters =
                buildRootParameters(layouts, rootPlan, tableRanges, *pushConstantSize, hasRayTracingStage);
            record->pushConstantRootParameter = rootParameters.pushConstantRootParameter;
            record->drawIndexRootParameter = rootParameters.drawIndexRootParameter;
            record->slots = std::move(rootParameters.slots);

            auto serialized = serializeRootSignature(rootParameters.parameters);
            if (!serialized)
            {
                return std::unexpected(std::move(serialized.error()));
            }
            const HRESULT createResult = device.CreateRootSignature(0,
                                                                    (*serialized)->GetBufferPointer(),
                                                                    (*serialized)->GetBufferSize(),
                                                                    IID_PPV_ARGS(&record->rootSignature));
            if (FAILED(createResult))
            {
                return std::unexpected(
                    nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(createResult)),
                                createResult,
                                "ID3D12Device::CreateRootSignature"));
            }
            setDebugName(*record->rootSignature.Get(), desc.label);
            return record;
        }
    } // namespace

    namespace pipeline_detail
    {

        Status validateShaderInterface(const ShaderModuleRecord& shader, const PipelineLayoutRecord& layout)
        {
            const auto& shaderInterface = shader.shaderInterface;
            for (const auto& binding : shaderInterface.bindings)
            {
                if (binding.set >= layout.slots.size())
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "A shader-interface binding references a set absent from the pipeline layout.",
                    });
                }
                const auto& entries = layout.slots[binding.set].entries;
                const auto found = std::ranges::find(entries, binding.binding, &BindingLayoutEntry::binding);
                const auto foundType =
                    found == entries.end() ? std::optional<ShaderResourceType>{} : shaderResourceType(found->type);
                if (found == entries.end() || !foundType.has_value() || *foundType != binding.type ||
                    found->count != binding.count || !hasAny(found->visibility, shader.stage))
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "A pipeline-layout binding does not match the shader interface's type, count, or visibility.",
                    });
                }
            }

            std::array<bool, maxPushConstantBytes / sizeof(std::uint32_t)> requiredWords{};
            std::array<bool, maxPushConstantBytes / sizeof(std::uint32_t)> layoutWords{};
            for (const auto& range : shaderInterface.pushConstants)
            {
                const auto first = range.offset / sizeof(std::uint32_t);
                const auto count = range.size / sizeof(std::uint32_t);
                std::ranges::fill(std::span(requiredWords).subspan(first, count), true);
            }
            for (const auto& range : layout.pushConstantRanges)
            {
                if (!hasAny(range.visibility, shader.stage))
                {
                    continue;
                }
                const auto first = range.offset / sizeof(std::uint32_t);
                const auto count = range.size / sizeof(std::uint32_t);
                std::ranges::fill(std::span(layoutWords).subspan(first, count), true);
            }
            if (requiredWords != layoutWords)
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "The pipeline layout's stage-visible push constants do not exactly match the shader interface.",
                });
            }
            if (shaderInterface.drawIndex && layout.drawIndexRootParameter == invalidRootParameter)
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "The shader reads the draw index, but the pipeline layout does not reserve it.",
                });
            }
            return {};
        }

    } // namespace pipeline_detail

    Status PipelineManager::Impl::startCompilationWorkers()
    {
        constexpr std::size_t workerCount = 2;
        try
        {
            compilationWorkers.reserve(workerCount);
            for (std::size_t index = 0; index < workerCount; ++index)
            {
                compilationWorkers.emplace_back([this]() { compilationWorker(); });
            }
        }
        catch (const std::bad_alloc&)
        {
            stopCompilationWorkers();
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Could not allocate the asynchronous pipeline-compilation workers.",
            });
        }
        catch (const std::system_error& error)
        {
            stopCompilationWorkers();
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                error.code().value(),
                "Could not start the asynchronous pipeline-compilation workers.",
            });
        }
        return {};
    }

    void PipelineManager::Impl::stopCompilationWorkers() noexcept
    {
        {
            const std::scoped_lock lock(compilationQueueMutex);
            stopCompilation = true;
        }
        compilationQueueCondition.notify_all();
        for (auto& worker : compilationWorkers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        compilationWorkers.clear();
    }

    Status PipelineManager::Impl::enqueueCompilation(std::shared_ptr<PipelineCompilationJob> job)
    {
        try
        {
            const std::scoped_lock lock(compilationQueueMutex);
            if (stopCompilation)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The asynchronous pipeline compiler is stopping.",
                });
            }
            compilationJobs.push_back(std::move(job));
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Could not allocate asynchronous pipeline-compilation queue storage.",
            });
        }
        catch (const std::system_error& error)
        {
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                error.code().value(),
                "Could not lock the asynchronous pipeline-compilation queue.",
            });
        }
        compilationQueueCondition.notify_one();
        return {};
    }

    void PipelineManager::Impl::compilationWorker() noexcept
    {
        for (;;)
        {
            std::shared_ptr<PipelineCompilationJob> job;
            {
                std::unique_lock lock(compilationQueueMutex);
                compilationQueueCondition.wait(lock, [this]() { return stopCompilation || !compilationJobs.empty(); });
                if (compilationJobs.empty())
                {
                    return;
                }
                job = std::move(compilationJobs.front());
                compilationJobs.pop_front();
            }
            executeCompilation(*job);
        }
    }

    void PipelineManager::Impl::executeCompilation(PipelineCompilationJob& job) noexcept
    {
        Result<PipelineHandle> result = std::unexpected(Error{
            ErrorCode::nativeFailure,
            0,
            "Pipeline compilation did not produce a result.",
        });
        try
        {
            result = std::visit(
                [this](const auto& owned) -> Result<PipelineHandle> {
                    using Desc = std::remove_cvref_t<decltype(owned)>;
                    if constexpr (std::is_same_v<Desc, OwnedComputePipelineDesc>)
                    {
                        return owner->createComputePipeline(owned.view());
                    }
                    else
                    {
                        return owner->createGraphicsPipeline(owned.view());
                    }
                },
                job.desc);
        }
        catch (const std::bad_alloc&)
        {
            result = std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Asynchronous pipeline compilation exhausted host memory.",
            });
        }
        catch (...)
        {
            result = std::unexpected(Error{
                ErrorCode::nativeFailure,
                0,
                "Asynchronous pipeline compilation raised an unexpected exception.",
            });
        }

        releaseDependencies(job.dependencies);

        PipelineHandle abandonedPipeline;
        {
            const std::scoped_lock lock(job.record->mutex);
            if (job.record->abandoned)
            {
                if (result)
                {
                    abandonedPipeline = *result;
                }
            }
            else
            {
                job.record->state = result ? PipelineCompilationState::succeeded : PipelineCompilationState::failed;
                job.record->result.emplace(std::move(result));
            }
        }
        job.record->condition.notify_all();
        if (abandonedPipeline.isValid())
        {
            if (!owner->destroyPipeline(abandonedPipeline))
            {
                std::terminate();
            }
        }
    }

    void PipelineManager::Impl::releaseDependenciesLocked(const RetainedPipelineDependencies& dependencies) noexcept
    {
        if (dependencies.layout.isValid())
        {
            static_cast<void>(pipelineLayouts.erase(dependencies.layout));
        }
        for (const auto shader : dependencies.shaders)
        {
            static_cast<void>(shaderModules.erase(shader));
        }
        if (dependencies.cache.isValid())
        {
            static_cast<void>(pipelineCaches.erase(dependencies.cache));
        }
    }

    void PipelineManager::Impl::releaseDependencies(const RetainedPipelineDependencies& dependencies) noexcept
    {
        const std::scoped_lock lock(mutex);
        releaseDependenciesLocked(dependencies);
    }

    namespace
    {

        template<typename Record, typename Handle>
        [[nodiscard]] Result<std::shared_ptr<Record>> findOptionalRecord(
            const detail::HandleTable<std::shared_ptr<Record>, Handle>& table,
            Handle handle,
            std::string_view staleMessage)
        {
            if (!handle.isValid())
            {
                return std::shared_ptr<Record>{};
            }
            const auto* found = table.find(handle);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    std::string(staleMessage),
                });
            }
            return *found;
        }

    } // namespace

    Result<RetainedPipelineDependencies> PipelineManager::Impl::retainDependencies(OwnedComputePipelineDesc& owned)
    {
        RetainedPipelineDependencies retained;
        retained.shaders.reserve(1);
        const std::scoped_lock lock(mutex);

        auto layout = findOptionalRecord(
            pipelineLayouts, owned.desc.layout, "Asynchronous compute compilation references a stale pipeline layout.");
        if (!layout)
        {
            return std::unexpected(std::move(layout.error()));
        }
        auto shader = findOptionalRecord(shaderModules,
                                         owned.desc.computeShader,
                                         "Asynchronous compute compilation references a stale shader module.");
        if (!shader)
        {
            return std::unexpected(std::move(shader.error()));
        }
        auto cache = findOptionalRecord(
            pipelineCaches, owned.desc.cache, "Asynchronous compute compilation references a stale pipeline cache.");
        if (!cache)
        {
            return std::unexpected(std::move(cache.error()));
        }

        if (*layout != nullptr)
        {
            auto inserted = pipelineLayouts.emplace(*layout);
            if (!inserted)
            {
                return std::unexpected(std::move(inserted.error()));
            }
            retained.layout = *inserted;
            owned.desc.layout = retained.layout;
        }
        if (*shader != nullptr)
        {
            auto inserted = shaderModules.emplace(*shader);
            if (!inserted)
            {
                releaseDependenciesLocked(retained);
                return std::unexpected(std::move(inserted.error()));
            }
            retained.shaders.push_back(*inserted);
            owned.desc.computeShader = retained.shaders.back();
        }
        if (*cache != nullptr)
        {
            auto inserted = pipelineCaches.emplace(*cache);
            if (!inserted)
            {
                releaseDependenciesLocked(retained);
                return std::unexpected(std::move(inserted.error()));
            }
            retained.cache = *inserted;
            owned.desc.cache = retained.cache;
        }
        return retained;
    }

    namespace
    {

        inline constexpr std::array graphicsShaderHandleMembers{
            &GraphicsPipelineDesc::vertexShader,
            &GraphicsPipelineDesc::pixelShader,
            &GraphicsPipelineDesc::geometryShader,
            &GraphicsPipelineDesc::hullShader,
            &GraphicsPipelineDesc::domainShader,
            &GraphicsPipelineDesc::taskShader,
            &GraphicsPipelineDesc::meshShader,
        };

        struct LiveGraphicsPipelineDependencies final
        {
            std::shared_ptr<PipelineLayoutRecord> layout;
            std::shared_ptr<PipelineCacheRecord> cache;
            std::array<std::shared_ptr<ShaderModuleRecord>, graphicsShaderHandleMembers.size()> shaders;
        };

        [[nodiscard]] Result<LiveGraphicsPipelineDependencies> findGraphicsPipelineDependencies(
            const PipelineLayoutTable& pipelineLayouts,
            const PipelineCacheTable& pipelineCaches,
            const ShaderModuleTable& shaderModules,
            const GraphicsPipelineDesc& desc)
        {
            auto layout = findOptionalRecord(
                pipelineLayouts, desc.layout, "Asynchronous graphics compilation references a stale pipeline layout.");
            if (!layout)
            {
                return std::unexpected(std::move(layout.error()));
            }
            auto cache = findOptionalRecord(
                pipelineCaches, desc.cache, "Asynchronous graphics compilation references a stale pipeline cache.");
            if (!cache)
            {
                return std::unexpected(std::move(cache.error()));
            }
            LiveGraphicsPipelineDependencies live{
                .layout = std::move(*layout),
                .cache = std::move(*cache),
                .shaders = {},
            };
            for (std::size_t index = 0; index < graphicsShaderHandleMembers.size(); ++index)
            {
                auto shader = findOptionalRecord(shaderModules,
                                                 desc.*graphicsShaderHandleMembers[index],
                                                 "Asynchronous graphics compilation references a stale shader module.");
                if (!shader)
                {
                    return std::unexpected(std::move(shader.error()));
                }
                live.shaders[index] = std::move(*shader);
            }
            return live;
        }

    } // namespace

    Result<RetainedPipelineDependencies> PipelineManager::Impl::retainDependencies(OwnedGraphicsPipelineDesc& owned)
    {
        RetainedPipelineDependencies retained;
        retained.shaders.reserve(graphicsShaderHandleMembers.size());
        const std::scoped_lock lock(mutex);

        auto live = findGraphicsPipelineDependencies(pipelineLayouts, pipelineCaches, shaderModules, owned.desc);
        if (!live)
        {
            return std::unexpected(std::move(live.error()));
        }

        if (live->layout != nullptr)
        {
            auto inserted = pipelineLayouts.emplace(live->layout);
            if (!inserted)
            {
                return std::unexpected(std::move(inserted.error()));
            }
            retained.layout = *inserted;
            owned.desc.layout = retained.layout;
        }
        for (std::size_t index = 0; index < graphicsShaderHandleMembers.size(); ++index)
        {
            if (live->shaders[index] == nullptr)
            {
                continue;
            }
            auto inserted = shaderModules.emplace(live->shaders[index]);
            if (!inserted)
            {
                releaseDependenciesLocked(retained);
                return std::unexpected(std::move(inserted.error()));
            }
            retained.shaders.push_back(*inserted);
            owned.desc.*graphicsShaderHandleMembers[index] = retained.shaders.back();
        }
        if (live->cache != nullptr)
        {
            auto inserted = pipelineCaches.emplace(live->cache);
            if (!inserted)
            {
                releaseDependenciesLocked(retained);
                return std::unexpected(std::move(inserted.error()));
            }
            retained.cache = *inserted;
            owned.desc.cache = retained.cache;
        }
        return retained;
    }

    Result<ComPtr<ID3D12CommandSignature>> PipelineManager::Impl::indirectDispatchSignature(std::uint32_t stride)
    {
        const std::scoped_lock lock(mutex);
        if (const auto found = dispatchIndirectSignatures.find(stride); found != dispatchIndirectSignatures.end())
        {
            return found->second;
        }

        const D3D12_INDIRECT_ARGUMENT_DESC argument{
            .Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            .VertexBuffer = {},
        };
        const D3D12_COMMAND_SIGNATURE_DESC desc{
            .ByteStride = stride,
            .NumArgumentDescs = 1,
            .pArgumentDescs = &argument,
            .NodeMask = 0,
        };
        ComPtr<ID3D12CommandSignature> signature;
        const HRESULT result = device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&signature));
        if (FAILED(result))
        {
            return std::unexpected(
                nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                            result,
                            "ID3D12Device::CreateCommandSignature(dispatch)"));
        }
        try
        {
            dispatchIndirectSignatures.emplace(stride, signature);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to cache the indirect dispatch command signature.",
            });
        }
        return signature;
    }

    Result<ComPtr<ID3D12CommandSignature>> PipelineManager::Impl::indirectRayTracingDispatchSignature()
    {
        const std::scoped_lock lock(mutex);
        if (rayTracingIndirectSignature != nullptr)
        {
            return rayTracingIndirectSignature;
        }
        if (!rayTracing || rayTracingCapabilities.tier < 11)
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Indirect ray dispatch requires DXR tier 1.1 support."});
        }

        const D3D12_INDIRECT_ARGUMENT_DESC argument{
            .Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_RAYS,
            .VertexBuffer = {},
        };
        const D3D12_COMMAND_SIGNATURE_DESC desc{
            .ByteStride = sizeof(D3D12_DISPATCH_RAYS_DESC),
            .NumArgumentDescs = 1,
            .pArgumentDescs = &argument,
            .NodeMask = 0,
        };
        const HRESULT result =
            device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&rayTracingIndirectSignature));
        if (FAILED(result))
        {
            return std::unexpected(
                nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                            result,
                            "ID3D12Device::CreateCommandSignature(indirect ray dispatch)"));
        }
        return rayTracingIndirectSignature;
    }

    Result<ComPtr<ID3D12CommandSignature>> PipelineManager::Impl::indirectSignature(bool indexed, std::uint32_t stride)
    {
        const auto key = (static_cast<std::uint64_t>(stride) << 1U) | static_cast<std::uint64_t>(indexed);
        const std::scoped_lock lock(mutex);
        if (const auto found = indirectSignatures.find(key); found != indirectSignatures.end())
        {
            return found->second;
        }

        const D3D12_INDIRECT_ARGUMENT_DESC argument{
            .Type = indexed ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
            .VertexBuffer = {},
        };
        const D3D12_COMMAND_SIGNATURE_DESC desc{
            .ByteStride = stride,
            .NumArgumentDescs = 1,
            .pArgumentDescs = &argument,
            .NodeMask = 0,
        };
        ComPtr<ID3D12CommandSignature> signature;
        const HRESULT result = device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&signature));
        if (FAILED(result))
        {
            return std::unexpected(
                nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                            result,
                            "ID3D12Device::CreateCommandSignature"));
        }
        try
        {
            indirectSignatures.emplace(key, signature);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to cache the indirect draw command signature.",
            });
        }
        return signature;
    }

    Result<ComPtr<ID3D12CommandSignature>> PipelineManager::Impl::indirectMeshSignature(std::uint32_t stride)
    {
        const std::scoped_lock lock(mutex);
        if (const auto found = meshIndirectSignatures.find(stride); found != meshIndirectSignatures.end())
        {
            return found->second;
        }

        const D3D12_INDIRECT_ARGUMENT_DESC argument{
            .Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH,
            .VertexBuffer = {},
        };
        const D3D12_COMMAND_SIGNATURE_DESC desc{
            .ByteStride = stride,
            .NumArgumentDescs = 1,
            .pArgumentDescs = &argument,
            .NodeMask = 0,
        };
        ComPtr<ID3D12CommandSignature> signature;
        const HRESULT result = device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&signature));
        if (FAILED(result))
        {
            return std::unexpected(
                nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                            result,
                            "ID3D12Device::CreateCommandSignature(mesh dispatch)"));
        }
        try
        {
            meshIndirectSignatures.emplace(stride, signature);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to cache the indirect mesh-dispatch command signature.",
            });
        }
        return signature;
    }

    Result<ComPtr<ID3D12CommandSignature>> PipelineManager::Impl::drawIndexSignature(PipelineLayoutRecord& layout,
                                                                                     DrawIndexCommandKind kind,
                                                                                     std::uint32_t stride)
    {
        const auto key = (static_cast<std::uint64_t>(stride) << 2U) | static_cast<std::uint64_t>(kind);
        const std::scoped_lock lock(layout.drawIndexSignatureMutex);
        if (const auto found = layout.drawIndexSignatures.find(key); found != layout.drawIndexSignatures.end())
        {
            return found->second;
        }

        const D3D12_INDIRECT_ARGUMENT_TYPE commandType =
            kind == DrawIndexCommandKind::draw          ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW
            : kind == DrawIndexCommandKind::drawIndexed ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED
                                                        : D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        std::array<D3D12_INDIRECT_ARGUMENT_DESC, 2> arguments{};
        arguments[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        arguments[0].Constant = {
            .RootParameterIndex = layout.drawIndexRootParameter,
            .DestOffsetIn32BitValues = 0,
            .Num32BitValuesToSet = 1,
        };
        arguments[1].Type = commandType;
        const D3D12_COMMAND_SIGNATURE_DESC desc{
            .ByteStride = stride,
            .NumArgumentDescs = static_cast<UINT>(arguments.size()),
            .pArgumentDescs = arguments.data(),
            .NodeMask = 0,
        };
        ComPtr<ID3D12CommandSignature> signature;
        const HRESULT result =
            device->CreateCommandSignature(&desc, layout.rootSignature.Get(), IID_PPV_ARGS(&signature));
        if (FAILED(result))
        {
            return std::unexpected(
                nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                            result,
                            "ID3D12Device::CreateCommandSignature(draw index)"));
        }
        try
        {
            layout.drawIndexSignatures.emplace(key, signature);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to cache the draw-index command signature.",
            });
        }
        return signature;
    }

    Result<std::shared_ptr<PipelineCacheRecord>> PipelineManager::Impl::pipelineCache(PipelineCacheHandle cache,
                                                                                      PipelineCacheKey key)
    {
        if (!cache.isValid() && !key.isValid())
        {
            return std::shared_ptr<PipelineCacheRecord>{};
        }
        if (!cache.isValid() || !key.isValid())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Pipeline caching requires both a live cache handle and a non-zero cache key.",
            });
        }

        const std::scoped_lock lock(mutex);
        const auto* found = pipelineCaches.find(cache);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline cache handle is invalid or stale.",
            });
        }
        return *found;
    }

    PipelineManager::PipelineManager() : impl_(std::make_unique<Impl>())
    {
        impl_->owner = this;
    }

    PipelineManager::~PipelineManager()
    {
        impl_->stopCompilationWorkers();
    }

    Status PipelineManager::initialize(ID3D12Device& device,
                                       std::uint32_t maxComputeWorkgroupInvocations,
                                       const std::array<std::uint32_t, 3>& maxComputeWorkgroupSize,
                                       bool geometryShaders,
                                       std::uint32_t maxPatchControlPoints,
                                       bool meshShaders,
                                       bool taskShaders,
                                       const MeshShaderCapabilities& meshShaderCapabilities,
                                       bool dualSourceBlending,
                                       bool logicOperations,
                                       bool conservativeRasterization,
                                       bool triangleFans,
                                       bool dynamicDepthBias,
                                       bool depthBoundsTest,
                                       bool variableRateShading,
                                       bool additionalShadingRates,
                                       bool variableRateShadingTier2,
                                       std::uint32_t maxClipDistances,
                                       std::uint32_t maxCullDistances,
                                       std::uint32_t maxCombinedClipAndCullDistances,
                                       std::uint32_t maxFramebufferLayers,
                                       std::uint32_t maxMultiviewViewCount,
                                       bool rayTracing,
                                       const RayTracingCapabilities& rayTracingCapabilities,
                                       const SubgroupCapabilities& subgroup)
    {
        impl_->device = &device;
        impl_->maxComputeWorkgroupInvocations = maxComputeWorkgroupInvocations;
        impl_->maxComputeWorkgroupSize = maxComputeWorkgroupSize;
        impl_->geometryShaders = geometryShaders;
        impl_->maxPatchControlPoints = maxPatchControlPoints;
        impl_->meshShaders = meshShaders;
        impl_->taskShaders = taskShaders;
        impl_->meshShaderCapabilities = meshShaderCapabilities;
        impl_->dualSourceBlending = dualSourceBlending;
        impl_->logicOperations = logicOperations;
        impl_->conservativeRasterization = conservativeRasterization;
        impl_->triangleFans = triangleFans;
        impl_->dynamicDepthBias = dynamicDepthBias;
        impl_->depthBoundsTest = depthBoundsTest;
        impl_->variableRateShading = variableRateShading;
        impl_->additionalShadingRates = additionalShadingRates;
        impl_->variableRateShadingTier2 = variableRateShadingTier2;
        impl_->maxClipDistances = maxClipDistances;
        impl_->maxCullDistances = maxCullDistances;
        impl_->maxCombinedClipAndCullDistances = maxCombinedClipAndCullDistances;
        impl_->maxFramebufferLayers = maxFramebufferLayers;
        impl_->maxMultiviewViewCount = maxMultiviewViewCount;
        impl_->rayTracing = rayTracing;
        impl_->rayTracingCapabilities = rayTracingCapabilities;
        impl_->subgroup = subgroup;
        if (rayTracing && FAILED(device.QueryInterface(IID_PPV_ARGS(&impl_->rayTracingDevice))))
        {
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                0,
                "A device reporting ray tracing did not expose ID3D12Device5.",
            });
        }
        static_cast<void>(device.QueryInterface(IID_PPV_ARGS(&impl_->pipelineLibraryDevice)));
        static_cast<void>(device.QueryInterface(IID_PPV_ARGS(&impl_->pipelineStateStreamDevice)));
        return impl_->startCompilationWorkers();
    }

    Result<ShaderLibraryHandle> PipelineManager::createShaderLibrary(const ShaderLibraryDesc& desc)
    {
        if (desc.entries.empty())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shader library requires at least one staged bytecode entry.",
            });
        }

        auto record = std::make_shared<ShaderLibraryRecord>();
        record->entries.reserve(desc.entries.size());
        record->label = desc.label;
        for (const auto& entry : desc.entries)
        {
            if (!isSingleShaderStage(entry.stage) || entry.bytecode.empty() || entry.entryPoint.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Every shader-library entry requires exactly one stage, non-empty bytecode, and a non-empty name.",
                });
            }
            auto specializationConstants = buildSpecializationSet(entry.specializationConstants);
            if (!specializationConstants)
            {
                return std::unexpected(std::move(specializationConstants.error()));
            }
            if (std::ranges::any_of(record->entries, [&](const ShaderLibraryEntryRecord& existing) {
                    return existing.stage == entry.stage && existing.entryPoint == entry.entryPoint &&
                           existing.specializationConstants == *specializationConstants;
                }))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shader-library entries must be unique by stage, name, and specialization values.",
                });
            }
            record->entries.push_back({
                .stage = entry.stage,
                .bytecode = ShaderBytecode(entry.bytecode.begin(), entry.bytecode.end()),
                .entryPoint = std::string(entry.entryPoint),
                .specializationConstants = std::move(*specializationConstants),
            });
        }

        const std::scoped_lock lock(impl_->mutex);
        return impl_->shaderLibraries.emplace(std::move(record));
    }

    Status PipelineManager::destroyShaderLibrary(ShaderLibraryHandle shaderLibrary)
    {
        const std::scoped_lock lock(impl_->mutex);
        if (!impl_->shaderLibraries.erase(shaderLibrary))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader library handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<ShaderModuleHandle> PipelineManager::createShaderModule(const ShaderModuleDesc& desc)
    {
        const bool hasDirectBytecode = !desc.bytecode.empty();
        const bool hasLibrary = desc.library.isValid();
        if (!isSingleShaderStage(desc.stage) || desc.entryPoint.empty() || hasDirectBytecode == hasLibrary)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shader module requires exactly one stage and one bytecode source: direct bytes or a live library.",
            });
        }
        if (hasDirectBytecode && !desc.specializationConstants.empty())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Specialization values require a shader library containing precompiled backend variants.",
            });
        }
        auto specializationConstants = buildSpecializationSet(desc.specializationConstants);
        if (!specializationConstants)
        {
            return std::unexpected(std::move(specializationConstants.error()));
        }

        auto shaderInterface = buildShaderInterface(desc.stage,
                                                    desc.shaderInterface,
                                                    impl_->maxComputeWorkgroupInvocations,
                                                    impl_->maxComputeWorkgroupSize,
                                                    impl_->meshShaderCapabilities);
        if (!shaderInterface)
        {
            return std::unexpected(std::move(shaderInterface.error()));
        }

        auto record = std::make_shared<ShaderModuleRecord>();
        record->stage = desc.stage;
        record->entryPoint = desc.entryPoint;
        record->label = desc.label;
        record->shaderInterface = std::move(*shaderInterface);
        record->specializationConstants = std::move(*specializationConstants);

        if (!hasLibrary)
        {
            if (auto drawIndex = validateDrawIndexBytecode(desc.bytecode, record->shaderInterface.drawIndex);
                !drawIndex)
            {
                return std::unexpected(std::move(drawIndex.error()));
            }
            record->directBytecode.assign(desc.bytecode.begin(), desc.bytecode.end());
            record->bytecode = &record->directBytecode;
            const std::scoped_lock lock(impl_->mutex);
            return impl_->shaderModules.emplace(std::move(record));
        }

        const std::scoped_lock lock(impl_->mutex);
        const auto* foundLibrary = impl_->shaderLibraries.find(desc.library);
        if (foundLibrary == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shader module references an invalid or stale shader library.",
            });
        }
        const auto foundEntry =
            std::ranges::find_if((*foundLibrary)->entries, [&](const ShaderLibraryEntryRecord& entry) {
                return entry.stage == desc.stage && entry.entryPoint == desc.entryPoint &&
                       entry.specializationConstants == record->specializationConstants;
            });
        if (foundEntry == (*foundLibrary)->entries.end())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader library does not contain the requested stage, entry point, and specialization values.",
            });
        }
        if (auto drawIndex = validateDrawIndexBytecode(foundEntry->bytecode, record->shaderInterface.drawIndex);
            !drawIndex)
        {
            return std::unexpected(std::move(drawIndex.error()));
        }
        record->bytecode = &foundEntry->bytecode;
        record->library = *foundLibrary;
        return impl_->shaderModules.emplace(std::move(record));
    }

    Result<ShaderInterfaceSnapshot> PipelineManager::shaderModuleInterface(ShaderModuleHandle shaderModule) const
    {
        const std::scoped_lock lock(impl_->mutex);
        const auto* found = impl_->shaderModules.find(shaderModule);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader module handle is invalid or stale.",
            });
        }
        const auto& module = **found;
        const auto& shaderInterface = module.shaderInterface;
        ShaderInterfaceSnapshot snapshot{
            .stage = module.stage,
            .bindings = shaderInterface.bindings,
            .pushConstants = shaderInterface.pushConstants,
            .vertexInputLocations = shaderInterface.vertexInputLocations,
            .workgroupSize = shaderInterface.workgroupSize,
            .shaderRecordConstants = shaderInterface.shaderRecordConstants,
            .drawIndex = shaderInterface.drawIndex,
        };
        return snapshot;
    }

    Status PipelineManager::destroyShaderModule(ShaderModuleHandle shaderModule)
    {
        const std::scoped_lock lock(impl_->mutex);
        if (!impl_->shaderModules.erase(shaderModule))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader module handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<PipelineLayoutHandle> PipelineManager::createPipelineLayout(BindingManager& bindings,
                                                                       const PipelineLayoutDesc& desc)
    {
        if (desc.bindingLayouts.size() > maxBindingSets)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "A pipeline layout exceeds the portable four-set binding limit.",
            });
        }

        std::vector<NativeBindingLayout> layouts;
        layouts.reserve(desc.bindingLayouts.size());
        for (const auto handle : desc.bindingLayouts)
        {
            auto layout = bindings.nativeBindingLayout(handle);
            if (!layout)
            {
                return std::unexpected(std::move(layout.error()));
            }
            layouts.push_back(std::move(*layout));
        }
        auto record = buildPipelineLayout(*impl_->device, std::move(layouts), desc);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        const std::scoped_lock lock(impl_->mutex);
        return impl_->pipelineLayouts.emplace(std::move(*record));
    }

    Status PipelineManager::destroyPipelineLayout(PipelineLayoutHandle pipelineLayout)
    {
        const std::scoped_lock lock(impl_->mutex);
        if (!impl_->pipelineLayouts.erase(pipelineLayout))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline layout handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<PipelineCacheHandle> PipelineManager::createPipelineCache(const PipelineCacheDesc& desc)
    {
        if (impl_->pipelineLibraryDevice == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "D3D12 pipeline libraries are unavailable on this device.",
            });
        }

        auto decoded = decodePipelineCache(desc.initialData);
        if (!decoded)
        {
            return std::unexpected(std::move(decoded.error()));
        }
        auto record = std::make_shared<PipelineCacheRecord>();
        record->label = desc.label;
        record->entries = std::move(decoded->entries);
        record->nativeData.assign(decoded->nativeData.begin(), decoded->nativeData.end());
        const void* initialData = record->nativeData.empty() ? nullptr : record->nativeData.data();
        const HRESULT result = impl_->pipelineLibraryDevice->CreatePipelineLibrary(
            initialData, record->nativeData.size(), IID_PPV_ARGS(&record->library));
        if (FAILED(result))
        {
            return std::unexpected(pipelineLibraryError(result, "ID3D12Device1::CreatePipelineLibrary"));
        }
        if (!test::nextPipelineCacheWithoutLibrary1().exchange(false, std::memory_order_relaxed))
        {
            static_cast<void>(record->library.As(&record->library1));
        }
        setDebugName(*record->library.Get(), record->label);

        const std::scoped_lock lock(impl_->mutex);
        return impl_->pipelineCaches.emplace(std::move(record));
    }

    Status PipelineManager::destroyPipelineCache(PipelineCacheHandle pipelineCache)
    {
        const std::scoped_lock lock(impl_->mutex);
        if (!impl_->pipelineCaches.erase(pipelineCache))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline cache handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<std::vector<std::byte>> PipelineManager::pipelineCacheData(PipelineCacheHandle pipelineCache)
    {
        std::shared_ptr<PipelineCacheRecord> cache;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* found = impl_->pipelineCaches.find(pipelineCache);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline cache handle is invalid or stale.",
                });
            }
            cache = *found;
        }

        const std::scoped_lock lock(cache->mutex);
        if (cache->entries.size() > std::numeric_limits<std::uint32_t>::max())
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The pipeline cache contains too many entries for the RHI blob schema.",
            });
        }
        const auto nativeSize = cache->library->GetSerializedSize();
        const auto entriesSize = cache->entries.size() * sizeof(PipelineCacheEntry);
        const auto metadataSize = sizeof(PipelineCacheBlobHeader) + entriesSize;
        if (nativeSize > std::numeric_limits<std::size_t>::max() - metadataSize)
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The serialized pipeline cache exceeds the addressable blob size.",
            });
        }
        std::vector<std::byte> data;
        try
        {
            data.resize(metadataSize + nativeSize);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate storage for serialized pipeline cache data.",
            });
        }

        const PipelineCacheBlobHeader header{
            .nativeDataSize = nativeSize,
            .entryCount = static_cast<std::uint32_t>(cache->entries.size()),
        };
        std::memcpy(data.data(), &header, sizeof(header));
        auto entryOffset = sizeof(header);
        for (const auto& entry : cache->entries)
        {
            std::memcpy(data.data() + entryOffset, &entry, sizeof(entry));
            entryOffset += sizeof(entry);
        }
        if (nativeSize == 0)
        {
            return data;
        }
        const HRESULT result = cache->library->Serialize(data.data() + metadataSize, nativeSize);
        if (FAILED(result))
        {
            return std::unexpected(pipelineLibraryError(result, "ID3D12PipelineLibrary::Serialize"));
        }
        return data;
    }

    namespace
    {

        struct ResolvedComputePipelineRecords final
        {
            std::shared_ptr<PipelineLayoutRecord> layout;
            std::shared_ptr<ShaderModuleRecord> shader;
        };

        [[nodiscard]] Result<ResolvedComputePipelineRecords> resolveComputePipelineRecords(
            const PipelineLayoutTable& pipelineLayouts,
            const ShaderModuleTable& shaderModules,
            const ComputePipelineDesc& desc)
        {
            const auto* foundLayout = pipelineLayouts.find(desc.layout);
            const auto* foundShader = shaderModules.find(desc.computeShader);
            if (foundLayout == nullptr || foundShader == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A compute pipeline references an invalid or stale layout or shader module.",
                });
            }
            if ((*foundShader)->stage != ShaderStage::compute)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A compute pipeline requires a compute shader module.",
                });
            }
            return ResolvedComputePipelineRecords{
                .layout = *foundLayout,
                .shader = *foundShader,
            };
        }

        [[nodiscard]] D3D12_COMPUTE_PIPELINE_STATE_DESC nativeComputePipelineDesc(
            const ComputePipelineRecord& record) noexcept
        {
            return {
                .pRootSignature = record.layout->rootSignature.Get(),
                .CS =
                    {
                        .pShaderBytecode = record.shader->bytecode->data(),
                        .BytecodeLength = record.shader->bytecode->size(),
                    },
                .NodeMask = 0,
                .CachedPSO = {},
                .Flags = D3D12_PIPELINE_STATE_FLAG_NONE,
            };
        }

        [[nodiscard]] Status validateComputeWaveSize(const ShaderModuleRecord& shader,
                                                     std::uint32_t requiredSubgroupSize,
                                                     const SubgroupCapabilities& subgroup)
        {
            auto declared = dxil::readExpectedWaveLaneCounts(*shader.bytecode);
            if (!declared)
            {
                if (requiredSubgroupSize == 0)
                {
                    return {};
                }
                return std::unexpected(std::move(declared.error()));
            }
            if (requiredSubgroupSize != 0 &&
                (!declared->has_value() || (*declared)->minimum != requiredSubgroupSize ||
                 (*declared)->maximum != requiredSubgroupSize))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "On Direct3D 12, a compute pipeline that requires a subgroup size needs a shader that declares "
                    "[WaveSize] with that size.",
                });
            }
            if (declared->has_value() &&
                ((*declared)->minimum < subgroup.minSize || (*declared)->maximum > subgroup.maxSize))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The compute shader's [WaveSize] is outside this device's subgroup size range.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<ComPtr<ID3D12PipelineState>> createNativeComputePipeline(
            ID3D12Device& device, const D3D12_COMPUTE_PIPELINE_STATE_DESC& nativeDesc)
        {
            ComPtr<ID3D12PipelineState> pipelineState;
            const HRESULT result = device.CreateComputePipelineState(&nativeDesc, IID_PPV_ARGS(&pipelineState));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(pipelineCreationErrorCode(result), result, "ID3D12Device::CreateComputePipelineState"));
            }
            return pipelineState;
        }

        struct PipelineCacheSlot final
        {
            std::wstring name;
            PipelineCacheEntry entry;
        };

        [[nodiscard]] PipelineCacheSlot pipelineCacheSlot(PipelineCacheKey key, PipelineCacheEntryKind kind)
        {
            return {
                .name = pipelineCacheEntryName(key, kind),
                .entry = {.key = key, .kind = kind},
            };
        }

        [[nodiscard]] std::set<PipelineCacheEntry>::iterator recordPipelineCacheEntry(PipelineCacheRecord& cache,
                                                                                      const PipelineCacheEntry& entry)
        {
            if (test::nextPipelineCacheEntryRecordFailure().exchange(false, std::memory_order_relaxed))
            {
                throw std::bad_alloc{};
            }
            return cache.entries.insert(entry).first;
        }

        [[nodiscard]] Result<ComPtr<ID3D12PipelineState>> loadCachedComputePipeline(
            PipelineCacheRecord& cache,
            const std::wstring& cacheName,
            const D3D12_COMPUTE_PIPELINE_STATE_DESC& nativeDesc)
        {
            ComPtr<ID3D12PipelineState> pipelineState;
            const HRESULT loadResult =
                cache.library->LoadComputePipeline(cacheName.c_str(), &nativeDesc, IID_PPV_ARGS(&pipelineState));
            if (loadResult == E_INVALIDARG)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    static_cast<std::int64_t>(loadResult),
                    "The compute pipeline cache key identifies an incompatible pipeline.",
                });
            }
            if (FAILED(loadResult))
            {
                return std::unexpected(pipelineLibraryError(loadResult, "ID3D12PipelineLibrary::LoadComputePipeline"));
            }
            return pipelineState;
        }

        [[nodiscard]] Status storeComputePipelineInCache(PipelineCacheRecord& cache,
                                                         const PipelineCacheSlot& slot,
                                                         ID3D12PipelineState* pipelineState)
        {
            std::set<PipelineCacheEntry>::iterator recorded;
            try
            {
                recorded = recordPipelineCacheEntry(cache, slot.entry);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to record the compute pipeline cache entry.",
                });
            }
            const HRESULT storeResult = cache.library->StorePipeline(slot.name.c_str(), pipelineState);
            if (FAILED(storeResult))
            {
                cache.entries.erase(recorded);
            }
            if (storeResult == E_INVALIDARG)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    static_cast<std::int64_t>(storeResult),
                    "The compute pipeline cache metadata is inconsistent with its native library.",
                });
            }
            if (FAILED(storeResult))
            {
                return std::unexpected(
                    pipelineLibraryError(storeResult, "ID3D12PipelineLibrary::StorePipeline(compute)"));
            }
            return {};
        }

        [[nodiscard]] Result<ComPtr<ID3D12PipelineState>> storeOrAdoptCachedComputePipeline(
            PipelineCacheRecord& cache,
            const PipelineCacheSlot& slot,
            const D3D12_COMPUTE_PIPELINE_STATE_DESC& nativeDesc,
            ComPtr<ID3D12PipelineState> pipelineState)
        {
            const std::scoped_lock cacheLock(cache.mutex);
            if (!cache.entries.contains(slot.entry))
            {
                auto stored = storeComputePipelineInCache(cache, slot, pipelineState.Get());
                if (!stored)
                {
                    return std::unexpected(std::move(stored.error()));
                }
                return pipelineState;
            }
            auto cachedState = loadCachedComputePipeline(cache, slot.name, nativeDesc);
            if (!cachedState)
            {
                return std::unexpected(std::move(cachedState.error()));
            }
            pipelineState = std::move(*cachedState);
            return pipelineState;
        }

        [[nodiscard]] Result<ComPtr<ID3D12PipelineState>> createCachedComputePipeline(
            ID3D12Device& device,
            PipelineCacheRecord& cache,
            const ComputePipelineDesc& desc,
            const D3D12_COMPUTE_PIPELINE_STATE_DESC& nativeDesc)
        {
            PipelineCacheSlot slot;
            try
            {
                slot = pipelineCacheSlot(desc.cacheKey, PipelineCacheEntryKind::compute);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate the compute pipeline cache entry name.",
                });
            }
            {
                const std::scoped_lock cacheLock(cache.mutex);
                if (cache.entries.contains(slot.entry))
                {
                    return loadCachedComputePipeline(cache, slot.name, nativeDesc);
                }
                if (desc.cachePolicy == PipelineCachePolicy::requireHit)
                {
                    return std::unexpected(Error{
                        ErrorCode::pipelineCacheMiss,
                        0,
                        "The compute pipeline is absent from the requested pipeline cache.",
                    });
                }
            }
            auto created = createNativeComputePipeline(device, nativeDesc);
            if (!created)
            {
                return std::unexpected(std::move(created.error()));
            }
            return storeOrAdoptCachedComputePipeline(cache, slot, nativeDesc, std::move(*created));
        }

    } // namespace

    Result<PipelineHandle> PipelineManager::createComputePipeline(const ComputePipelineDesc& desc)
    {
        if (!isValidPipelineCachePolicy(desc.cachePolicy))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A compute pipeline contains an invalid pipeline-cache policy.",
            });
        }
        auto resolved = [&]() {
            const std::scoped_lock lock(impl_->mutex);
            return resolveComputePipelineRecords(impl_->pipelineLayouts, impl_->shaderModules, desc);
        }();
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        auto interfaceValidation = validateShaderInterface(*resolved->shader, *resolved->layout);
        if (!interfaceValidation)
        {
            return std::unexpected(std::move(interfaceValidation.error()));
        }
        if (auto waveSize = validateComputeWaveSize(*resolved->shader, desc.requiredSubgroupSize, impl_->subgroup);
            !waveSize)
        {
            return std::unexpected(std::move(waveSize.error()));
        }
        auto cache = impl_->pipelineCache(desc.cache, desc.cacheKey);
        if (!cache)
        {
            return std::unexpected(std::move(cache.error()));
        }
        if (desc.cachePolicy == PipelineCachePolicy::requireHit && *cache == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A cache-only compute pipeline requires a live cache and non-zero key.",
            });
        }

        auto record = std::make_shared<ComputePipelineRecord>();
        record->layout = std::move(resolved->layout);
        record->shader = std::move(resolved->shader);
        record->label = desc.label;
        const auto nativeDesc = nativeComputePipelineDesc(*record);
        auto pipelineState = *cache != nullptr ? createCachedComputePipeline(*impl_->device, **cache, desc, nativeDesc)
                                               : createNativeComputePipeline(*impl_->device, nativeDesc);
        if (!pipelineState)
        {
            return std::unexpected(std::move(pipelineState.error()));
        }
        record->pipelineState = std::move(*pipelineState);
        setDebugName(*record->pipelineState.Get(), desc.label);

        const std::scoped_lock lock(impl_->mutex);
        return impl_->pipelines.emplace(PipelineRecord{std::move(record)});
    }

    namespace
    {

        [[nodiscard]] constexpr bool isValidSampleCount(std::uint32_t sampleCount) noexcept
        {
            return sampleCount == 1 || sampleCount == 2 || sampleCount == 4 || sampleCount == 8;
        }

        [[nodiscard]] bool hasValidAttachmentsAndVertexBufferCount(const GraphicsPipelineDesc& desc) noexcept
        {
            const bool captureOnly = !desc.streamOutputEntries.empty() && !desc.rasterizedStream.has_value();
            const bool hasAttachment = !desc.colorTargets.empty() || desc.depthStencil.format != Format::unknown;
            return desc.colorTargets.size() <= maxColorAttachments && desc.vertexBuffers.size() <= maxVertexBuffers &&
                   (captureOnly || hasAttachment);
        }

        [[nodiscard]] bool hasValidMultisampleState(const GraphicsPipelineDesc& desc) noexcept
        {
            if (!isValidSampleCount(desc.sampleCount))
            {
                return false;
            }
            const auto activeSampleMask = (1U << desc.sampleCount) - 1U;
            return (desc.sampleMask & activeSampleMask) != 0 && !(desc.alphaToCoverageEnabled && desc.sampleCount == 1);
        }

        [[nodiscard]] bool hasValidDepthBias(const GraphicsPipelineDesc& desc) noexcept
        {
            const bool hasStaticDepthBias =
                desc.depthBias != 0 || desc.depthBiasClamp != 0.0F || desc.depthBiasSlopeScale != 0.0F;
            return std::isfinite(desc.depthBiasClamp) && std::isfinite(desc.depthBiasSlopeScale) &&
                   !(desc.dynamicDepthBiasEnabled && hasStaticDepthBias);
        }

        [[nodiscard]] Status validateMeshShaderStages(const GraphicsPipelineDesc& desc)
        {
            const bool hasTaskShader = desc.taskShader.isValid();
            const bool usesMeshPipeline = desc.meshShader.isValid();
            const bool usesConventionalInputState = desc.vertexShader.isValid() || desc.geometryShader.isValid() ||
                                                    desc.hullShader.isValid() || desc.domainShader.isValid() ||
                                                    !desc.vertexBuffers.empty() ||
                                                    desc.topology != PrimitiveTopology::triangleList ||
                                                    desc.patchControlPoints != 0 || desc.stripIndexFormat.has_value();
            if ((hasTaskShader && !usesMeshPipeline) || (usesMeshPipeline && usesConventionalInputState))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Mesh graphics requires a mesh module, an optional task module, and no conventional input stages "
                    "or state.",
                });
            }
            return {};
        }

        struct ResolvedGraphicsPipelineRecords final
        {
            std::shared_ptr<PipelineLayoutRecord> layout;
            std::shared_ptr<ShaderModuleRecord> vertexShader;
            std::shared_ptr<ShaderModuleRecord> pixelShader;
            std::shared_ptr<ShaderModuleRecord> geometryShader;
            std::shared_ptr<ShaderModuleRecord> hullShader;
            std::shared_ptr<ShaderModuleRecord> domainShader;
            std::shared_ptr<ShaderModuleRecord> taskShader;
            std::shared_ptr<ShaderModuleRecord> meshShader;
        };

        struct ShaderModuleLookup final
        {
            bool requested = false;
            const std::shared_ptr<ShaderModuleRecord>* found = nullptr;

            [[nodiscard]] bool isStale() const noexcept
            {
                return requested && found == nullptr;
            }

            [[nodiscard]] bool hasStageOtherThan(ShaderStage stage) const noexcept
            {
                return found != nullptr && (*found)->stage != stage;
            }

            [[nodiscard]] std::shared_ptr<ShaderModuleRecord> record() const
            {
                return found != nullptr ? *found : nullptr;
            }
        };

        [[nodiscard]] ShaderModuleLookup findRequestedShaderModule(const ShaderModuleTable& shaderModules,
                                                                   ShaderModuleHandle handle,
                                                                   bool requested) noexcept
        {
            return {.requested = requested, .found = requested ? shaderModules.find(handle) : nullptr};
        }

        [[nodiscard]] Result<ResolvedGraphicsPipelineRecords> resolveGraphicsPipelineRecords(
            const PipelineLayoutTable& pipelineLayouts,
            const ShaderModuleTable& shaderModules,
            const GraphicsPipelineDesc& desc)
        {
            const bool usesMeshPipeline = desc.meshShader.isValid();
            const auto* foundLayout = pipelineLayouts.find(desc.layout);
            const auto vertexShader = findRequestedShaderModule(shaderModules, desc.vertexShader, !usesMeshPipeline);
            const auto pixelShader =
                findRequestedShaderModule(shaderModules, desc.pixelShader, desc.pixelShader.isValid());
            const auto geometryShader =
                findRequestedShaderModule(shaderModules, desc.geometryShader, desc.geometryShader.isValid());
            const auto hullShader =
                findRequestedShaderModule(shaderModules, desc.hullShader, desc.hullShader.isValid());
            const auto domainShader =
                findRequestedShaderModule(shaderModules, desc.domainShader, desc.domainShader.isValid());
            const auto taskShader =
                findRequestedShaderModule(shaderModules, desc.taskShader, desc.taskShader.isValid());
            const auto meshShader = findRequestedShaderModule(shaderModules, desc.meshShader, usesMeshPipeline);
            const bool rasterizesWithoutPixelShader = desc.rasterizedStream.has_value() && pixelShader.found == nullptr;
            if (foundLayout == nullptr || vertexShader.isStale() || rasterizesWithoutPixelShader ||
                pixelShader.isStale() || geometryShader.isStale() || hullShader.isStale() || domainShader.isStale() ||
                taskShader.isStale() || meshShader.isStale())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics pipeline references an invalid or stale layout or shader module.",
                });
            }
            if (vertexShader.hasStageOtherThan(ShaderStage::vertex) ||
                pixelShader.hasStageOtherThan(ShaderStage::pixel) ||
                geometryShader.hasStageOtherThan(ShaderStage::geometry) ||
                hullShader.hasStageOtherThan(ShaderStage::hull) ||
                domainShader.hasStageOtherThan(ShaderStage::domain) ||
                taskShader.hasStageOtherThan(ShaderStage::task) || meshShader.hasStageOtherThan(ShaderStage::mesh))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics pipeline requires correctly staged conventional or task/mesh shader modules.",
                });
            }
            return ResolvedGraphicsPipelineRecords{
                .layout = *foundLayout,
                .vertexShader = vertexShader.record(),
                .pixelShader = pixelShader.record(),
                .geometryShader = geometryShader.record(),
                .hullShader = hullShader.record(),
                .domainShader = domainShader.record(),
                .taskShader = taskShader.record(),
                .meshShader = meshShader.record(),
            };
        }

        [[nodiscard]] Status validateShaderInterfaces(const ResolvedGraphicsPipelineRecords& records)
        {
            const std::array shaders{
                records.vertexShader.get(),
                records.pixelShader.get(),
                records.geometryShader.get(),
                records.hullShader.get(),
                records.domainShader.get(),
                records.taskShader.get(),
                records.meshShader.get(),
            };
            for (const auto* shader : shaders)
            {
                if (shader == nullptr)
                {
                    continue;
                }
                auto interfaceValidation = validateShaderInterface(*shader, *records.layout);
                if (!interfaceValidation)
                {
                    return std::unexpected(std::move(interfaceValidation.error()));
                }
            }
            return {};
        }

        [[nodiscard]] Result<std::optional<std::vector<detail::StageInterfaceVariable>>> readSignature(
            const ShaderModuleRecord& shader, dxil::SignaturePart part, std::optional<std::uint32_t> stream)
        {
            if (shader.bytecode == nullptr)
            {
                return std::optional<std::vector<detail::StageInterfaceVariable>>{};
            }
            auto signature = dxil::readStageSignature(*shader.bytecode, part);
            if (!signature)
            {
                return std::unexpected(std::move(signature.error()));
            }
            if (!signature->has_value())
            {
                return std::optional<std::vector<detail::StageInterfaceVariable>>{};
            }
            std::vector<detail::StageInterfaceVariable> variables;
            for (const auto& element : **signature)
            {
                if (!stream.has_value() || element.stream == *stream)
                {
                    variables.push_back(element.variable);
                }
            }
            return std::optional{std::move(variables)};
        }

        [[nodiscard]] Status validateGraphicsStageLinkage(const ResolvedGraphicsPipelineRecords& records,
                                                          std::optional<std::uint32_t> rasterizedStream)
        {
            const ShaderModuleRecord* producer = records.meshShader       ? records.meshShader.get()
                                                 : records.geometryShader ? records.geometryShader.get()
                                                 : records.domainShader   ? records.domainShader.get()
                                                                          : records.vertexShader.get();
            if (records.pixelShader == nullptr || producer == nullptr || !rasterizedStream.has_value())
            {
                return {};
            }
            if (producer == records.meshShader.get())
            {
                auto primitives = readSignature(*producer, dxil::SignaturePart::primitiveOutput, std::nullopt);
                if (!primitives)
                {
                    return std::unexpected(std::move(primitives.error()));
                }
                if (primitives->has_value() && !(*primitives)->empty())
                {
                    return {};
                }
            }
            const auto stream = producer == records.geometryShader.get() ? rasterizedStream : std::optional{0U};
            auto outputs = readSignature(*producer, dxil::SignaturePart::output, stream);
            if (!outputs)
            {
                return std::unexpected(std::move(outputs.error()));
            }
            auto inputs = readSignature(*records.pixelShader, dxil::SignaturePart::input, std::nullopt);
            if (!inputs)
            {
                return std::unexpected(std::move(inputs.error()));
            }
            if (!outputs->has_value() || !inputs->has_value())
            {
                return {};
            }
            return detail::validateStageLinkage(producer->stage, **outputs, **inputs);
        }

        struct StreamOutputLayout final
        {
            std::vector<D3D12_SO_DECLARATION_ENTRY> declarations;
            std::vector<UINT> bufferStrides;
        };

        struct StreamOutputUsage final
        {
            std::array<std::optional<std::uint32_t>, maxStreamOutputBuffers> slotStreams{};
            std::array<bool, maxStreamOutputStreams> usedStreams{};
            std::array<std::array<std::uint8_t, maxVertexAttributes>, maxStreamOutputStreams> usedComponents{};
            std::array<std::uint32_t, maxStreamOutputBuffers> packedBytes{};
            std::uint32_t totalComponents = 0;
        };

        [[nodiscard]] Result<std::vector<UINT>> nativeStreamOutputStrides(std::span<const std::uint32_t> strides)
        {
            std::vector<UINT> nativeStrides;
            nativeStrides.reserve(strides.size());
            for (const auto stride : strides)
            {
                if (stride == 0 || stride > maxStreamOutputStride || stride % sizeof(std::uint32_t) != 0)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Stream-output strides must be non-zero, DWORD-aligned, and within the portable limit.",
                    });
                }
                nativeStrides.push_back(stride);
            }
            return nativeStrides;
        }

        [[nodiscard]] Status validateAndRecordStreamOutputEntry(StreamOutputUsage& usage,
                                                                const StreamOutputEntry& entry,
                                                                std::size_t outputSlotCount)
        {
            if (entry.stream >= maxStreamOutputStreams || entry.shaderLocation >= maxVertexAttributes ||
                entry.outputSlot >= outputSlotCount || entry.componentCount == 0 || entry.componentCount > 4 ||
                entry.startComponent >= 4 || entry.componentCount > 4 - entry.startComponent)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A stream-output declaration contains an invalid stream, location, component range, or slot.",
                });
            }
            const auto componentMask =
                static_cast<std::uint8_t>(((1U << entry.componentCount) - 1U) << entry.startComponent);
            auto& slotStream = usage.slotStreams[entry.outputSlot];
            if ((usage.usedComponents[entry.stream][entry.shaderLocation] & componentMask) != 0 ||
                (slotStream.has_value() && *slotStream != entry.stream))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Stream-output declarations may not overlap components or mix streams in one output slot.",
                });
            }
            usage.usedComponents[entry.stream][entry.shaderLocation] |= componentMask;
            slotStream = entry.stream;
            usage.usedStreams[entry.stream] = true;
            usage.totalComponents += entry.componentCount;
            usage.packedBytes[entry.outputSlot] += entry.componentCount * sizeof(std::uint32_t);
            return {};
        }

        [[nodiscard]] constexpr D3D12_SO_DECLARATION_ENTRY nativeStreamOutputDeclaration(
            const StreamOutputEntry& entry) noexcept
        {
            return {
                .Stream = entry.stream,
                .SemanticName = "TEXCOORD",
                .SemanticIndex = entry.shaderLocation,
                .StartComponent = static_cast<BYTE>(entry.startComponent),
                .ComponentCount = static_cast<BYTE>(entry.componentCount),
                .OutputSlot = static_cast<BYTE>(entry.outputSlot),
            };
        }

        [[nodiscard]] Status validateStreamOutputUsage(const StreamOutputUsage& usage, const GraphicsPipelineDesc& desc)
        {
            const bool usesStreamOutput = !desc.streamOutputEntries.empty();
            const bool exceedsComponentBudget = usage.totalComponents > D3D12_SO_OUTPUT_COMPONENT_COUNT;
            const bool rasterizesUndeclaredStream =
                desc.rasterizedStream.has_value() &&
                (*desc.rasterizedStream >= maxStreamOutputStreams || !usage.usedStreams[*desc.rasterizedStream]);
            if (usesStreamOutput && (exceedsComponentBudget || rasterizesUndeclaredStream))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Stream output exceeds the component budget or selects an undeclared rasterized stream.",
                });
            }
            for (std::size_t slot = 0; slot < desc.streamOutputBufferStrides.size(); ++slot)
            {
                if (!usage.slotStreams[slot].has_value() ||
                    usage.packedBytes[slot] > desc.streamOutputBufferStrides[slot])
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Every stream-output slot must be declared and its packed components must fit its stride.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Result<StreamOutputLayout> buildStreamOutputLayout(const GraphicsPipelineDesc& desc)
        {
            const bool usesStreamOutput = !desc.streamOutputEntries.empty();
            const bool usesMeshPipeline = desc.meshShader.isValid();
            if (usesStreamOutput != !desc.streamOutputBufferStrides.empty() || (usesStreamOutput && usesMeshPipeline) ||
                desc.streamOutputBufferStrides.size() > maxStreamOutputBuffers ||
                desc.streamOutputEntries.size() > D3D12_SO_OUTPUT_COMPONENT_COUNT ||
                (!usesStreamOutput && (!desc.rasterizedStream.has_value() || *desc.rasterizedStream != 0)))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Stream output requires paired declarations and strides on a conventional graphics pipeline.",
                });
            }
            auto bufferStrides = nativeStreamOutputStrides(desc.streamOutputBufferStrides);
            if (!bufferStrides)
            {
                return std::unexpected(std::move(bufferStrides.error()));
            }

            StreamOutputLayout layout{.declarations = {}, .bufferStrides = std::move(*bufferStrides)};
            StreamOutputUsage usage;
            layout.declarations.reserve(desc.streamOutputEntries.size());
            for (const auto& entry : desc.streamOutputEntries)
            {
                auto recorded = validateAndRecordStreamOutputEntry(usage, entry, desc.streamOutputBufferStrides.size());
                if (!recorded)
                {
                    return std::unexpected(std::move(recorded.error()));
                }
                layout.declarations.push_back(nativeStreamOutputDeclaration(entry));
            }
            auto usageValidation = validateStreamOutputUsage(usage, desc);
            if (!usageValidation)
            {
                return std::unexpected(std::move(usageValidation.error()));
            }
            if (usesStreamOutput && !desc.geometryShader.isValid() &&
                std::ranges::any_of(desc.streamOutputEntries,
                                    [](const StreamOutputEntry& entry) { return entry.stream != 0; }))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Multiple stream-output streams require a geometry shader.",
                });
            }
            return layout;
        }

        struct VertexInputLayout final
        {
            std::vector<D3D12_INPUT_ELEMENT_DESC> elements;
            std::vector<std::uint32_t> strides;
        };

        [[nodiscard]] constexpr bool isValidVertexBufferLayout(const VertexBufferLayout& buffer) noexcept
        {
            const bool validStepMode =
                buffer.stepMode == VertexStepMode::vertex || buffer.stepMode == VertexStepMode::instance;
            const bool validStepRate = (buffer.stepMode == VertexStepMode::vertex && buffer.instanceStepRate == 1) ||
                                       (buffer.stepMode == VertexStepMode::instance && buffer.instanceStepRate != 0);
            return buffer.stride != 0 && buffer.stride <= D3D12_SO_BUFFER_MAX_STRIDE_IN_BYTES &&
                   !buffer.attributes.empty() && validStepMode && validStepRate;
        }

        [[nodiscard]] bool supportsVertexBufferFormat(ID3D12Device& device, VertexFormat format) noexcept
        {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{
                .Format = nativeVertexFormat(format),
                .Support1 = {},
                .Support2 = {},
            };
            return support.Format != DXGI_FORMAT_UNKNOWN &&
                   SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) &&
                   (support.Support1 & D3D12_FORMAT_SUPPORT1_IA_VERTEX_BUFFER) != 0;
        }

        [[nodiscard]] Status validateVertexAttribute(ID3D12Device& device,
                                                     const VertexBufferLayout& buffer,
                                                     const VertexAttribute& attribute,
                                                     const std::vector<bool>& usedLocations)
        {
            const auto attributeSize = detail::vertexFormatSize(attribute.format);
            const bool supportedFormat = supportsVertexBufferFormat(device, attribute.format);
            if (attribute.shaderLocation >= maxVertexAttributes || usedLocations[attribute.shaderLocation] ||
                attributeSize == 0 || attribute.offset > buffer.stride ||
                attributeSize > buffer.stride - attribute.offset)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Vertex attributes require unique portable locations and must fit within their stride.",
                });
            }
            if (!supportedFormat)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The device does not support the requested vertex attribute format.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr D3D12_INPUT_ELEMENT_DESC nativeInputElement(UINT inputSlot,
                                                                            const VertexBufferLayout& buffer,
                                                                            const VertexAttribute& attribute) noexcept
        {
            const bool perInstance = buffer.stepMode == VertexStepMode::instance;
            return {
                .SemanticName = "TEXCOORD",
                .SemanticIndex = attribute.shaderLocation,
                .Format = nativeVertexFormat(attribute.format),
                .InputSlot = inputSlot,
                .AlignedByteOffset = attribute.offset,
                .InputSlotClass = perInstance ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                                              : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                .InstanceDataStepRate = perInstance ? buffer.instanceStepRate : 0U,
            };
        }

        [[nodiscard]] Result<VertexInputLayout> buildVertexInputLayout(
            ID3D12Device& device,
            std::span<const VertexBufferLayout> vertexBuffers,
            const ShaderModuleRecord* vertexShader)
        {
            VertexInputLayout layout;
            std::vector<bool> usedLocations(maxVertexAttributes, false);
            layout.strides.reserve(vertexBuffers.size());
            for (std::size_t slot = 0; slot < vertexBuffers.size(); ++slot)
            {
                const auto& buffer = vertexBuffers[slot];
                if (!isValidVertexBufferLayout(buffer))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Vertex buffer layouts require attributes, a valid stride, step mode, and instance step rate.",
                    });
                }
                layout.strides.push_back(buffer.stride);
                for (const auto& attribute : buffer.attributes)
                {
                    auto attributeValidation = validateVertexAttribute(device, buffer, attribute, usedLocations);
                    if (!attributeValidation)
                    {
                        return std::unexpected(std::move(attributeValidation.error()));
                    }
                    usedLocations[attribute.shaderLocation] = true;
                    layout.elements.push_back(nativeInputElement(static_cast<UINT>(slot), buffer, attribute));
                }
            }
            if (layout.elements.size() > maxVertexAttributes)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The graphics pipeline exceeds the portable vertex attribute limit.",
                });
            }
            if (vertexShader != nullptr)
            {
                auto vertexInterfaceValidation = validateVertexInterface(*vertexShader, layout.elements);
                if (!vertexInterfaceValidation)
                {
                    return std::unexpected(std::move(vertexInterfaceValidation.error()));
                }
            }
            return layout;
        }

        [[nodiscard]] constexpr bool blendUsesConstants(const BlendState& blend) noexcept
        {
            return blend.enabled &&
                   (usesBlendConstant(blend.color.sourceFactor) || usesBlendConstant(blend.color.destinationFactor) ||
                    usesBlendConstant(blend.alpha.sourceFactor) || usesBlendConstant(blend.alpha.destinationFactor));
        }

        [[nodiscard]] constexpr bool blendUsesDualSource(const BlendState& blend) noexcept
        {
            return blend.enabled &&
                   (usesDualSource(blend.color.sourceFactor) || usesDualSource(blend.color.destinationFactor) ||
                    usesDualSource(blend.alpha.sourceFactor) || usesDualSource(blend.alpha.destinationFactor));
        }

        [[nodiscard]] std::shared_ptr<GraphicsPipelineRecord> makeGraphicsPipelineRecord(
            const GraphicsPipelineDesc& desc,
            ResolvedGraphicsPipelineRecords resolved,
            std::vector<std::uint32_t> vertexStrides)
        {
            auto record = std::make_shared<GraphicsPipelineRecord>();
            record->layout = std::move(resolved.layout);
            record->vertexShader = std::move(resolved.vertexShader);
            record->pixelShader = std::move(resolved.pixelShader);
            record->geometryShader = std::move(resolved.geometryShader);
            record->hullShader = std::move(resolved.hullShader);
            record->domainShader = std::move(resolved.domainShader);
            record->taskShader = std::move(resolved.taskShader);
            record->meshShader = std::move(resolved.meshShader);
            record->vertexStrides = std::move(vertexStrides);
            std::ranges::transform(
                desc.vertexBuffers, std::back_inserter(record->vertexAlignments), [](const VertexBufferLayout& buffer) {
                    return detail::vertexAttributesAlignment(buffer.attributes);
                });
            record->streamOutputStrides.assign(desc.streamOutputBufferStrides.begin(),
                                               desc.streamOutputBufferStrides.end());
            std::ranges::transform(
                desc.colorTargets, std::back_inserter(record->colorFormats), &ColorTargetState::format);
            record->depthStencilFormat = desc.depthStencil.format;
            record->topology = desc.meshShader.isValid() ? D3D_PRIMITIVE_TOPOLOGY_UNDEFINED
                                                         : nativeTopology(desc.topology, desc.patchControlPoints);
            record->stripIndexFormat = desc.stripIndexFormat;
            record->sampleCount = desc.sampleCount;
            record->viewMask = desc.viewMask;
            record->usesBlendConstants = std::ranges::any_of(
                desc.colorTargets, [](const ColorTargetState& target) { return blendUsesConstants(target.blend); });
            record->rasterizationEnabled = desc.rasterizedStream.has_value();
            record->dynamicDepthBias = desc.dynamicDepthBiasEnabled;
            record->depthBoundsTest = desc.depthStencil.depthBoundsTestEnabled;
            record->stencilEnabled = desc.depthStencil.stencilTestEnabled;
            record->readsDrawIndex = std::ranges::any_of(
                std::array{record->vertexShader.get(), record->taskShader.get(), record->meshShader.get()},
                [](const ShaderModuleRecord* shader) {
                    return shader != nullptr && shader->shaderInterface.drawIndex;
                });
            record->label = desc.label;
            return record;
        }

        struct BlendFeatureSupport final
        {
            bool logicOperations = false;
            bool dualSourceBlending = false;
        };

        [[nodiscard]] Status validateColorTargetState(ID3D12Device& device,
                                                      const ColorTargetState& target,
                                                      std::uint32_t sampleCount)
        {
            const auto mask = static_cast<std::uint8_t>(target.writeMask);
            if (!isColorTargetFormat(target.format) || (mask & ~static_cast<std::uint8_t>(ColorWriteMask::all)) != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics color target has an unsupported format or write mask.",
                });
            }
            auto sampleCountSupported = validateAttachmentSampleCount(device, target.format, sampleCount);
            if (!sampleCountSupported)
            {
                return std::unexpected(std::move(sampleCountSupported.error()));
            }
            const auto& colorBlend = target.blend.color;
            const auto& alphaBlend = target.blend.alpha;
            if (!isValidBlendFactor(colorBlend.sourceFactor) || !isValidBlendFactor(colorBlend.destinationFactor) ||
                !isValidBlendOperation(colorBlend.operation) || !isValidBlendFactor(alphaBlend.sourceFactor) ||
                !isValidBlendFactor(alphaBlend.destinationFactor) || !isValidBlendOperation(alphaBlend.operation))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics color target contains an invalid blend factor or operation.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateLogicOperationTarget(ID3D12Device& device,
                                                          std::span<const ColorTargetState> targets,
                                                          std::size_t index)
        {
            const auto& target = targets[index];
            if (target.blend.enabled)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Logic operations and color blending cannot be enabled together.",
                });
            }
            if (!supportsLogicOperation(device, target.format))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "A color-target format does not support output-merger logic operations.",
                });
            }
            if (index != 0 && targets[index - 1].writeMask != target.writeMask)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Logic operations require identical write masks for every color target.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateDualSourceBlendOperations(const BlendState& blend)
        {
            if (blendUsesDualSource(blend) && (blend.color.operation > BlendOperation::reverseSubtract ||
                                               blend.alpha.operation > BlendOperation::reverseSubtract))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Dual-source blending supports only add, subtract, and reverse-subtract operations.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateColorTarget(ID3D12Device& device,
                                                 const GraphicsPipelineDesc& desc,
                                                 std::size_t index)
        {
            const auto& target = desc.colorTargets[index];
            auto stateValidation = validateColorTargetState(device, target, desc.sampleCount);
            if (!stateValidation)
            {
                return std::unexpected(std::move(stateValidation.error()));
            }
            if (desc.logicOperation.has_value())
            {
                auto logicValidation = validateLogicOperationTarget(device, desc.colorTargets, index);
                if (!logicValidation)
                {
                    return std::unexpected(std::move(logicValidation.error()));
                }
            }
            return validateDualSourceBlendOperations(target.blend);
        }

        [[nodiscard]] constexpr D3D12_RENDER_TARGET_BLEND_DESC nativeRenderTargetBlend(
            const ColorTargetState& target, const std::optional<LogicOperation>& logicOperation) noexcept
        {
            return {
                .BlendEnable = target.blend.enabled ? TRUE : FALSE,
                .LogicOpEnable = logicOperation.has_value() ? TRUE : FALSE,
                .SrcBlend = nativeColorBlendFactor(target.blend.color.sourceFactor),
                .DestBlend = nativeColorBlendFactor(target.blend.color.destinationFactor),
                .BlendOp = nativeBlendOperation(target.blend.color.operation),
                .SrcBlendAlpha = nativeAlphaBlendFactor(target.blend.alpha.sourceFactor),
                .DestBlendAlpha = nativeAlphaBlendFactor(target.blend.alpha.destinationFactor),
                .BlendOpAlpha = nativeBlendOperation(target.blend.alpha.operation),
                .LogicOp = logicOperation.has_value() ? nativeLogicOperation(*logicOperation) : D3D12_LOGIC_OP_NOOP,
                .RenderTargetWriteMask = static_cast<std::uint8_t>(target.writeMask),
            };
        }

        [[nodiscard]] Result<D3D12_BLEND_DESC> buildBlendState(ID3D12Device& device,
                                                               const BlendFeatureSupport& support,
                                                               const GraphicsPipelineDesc& desc)
        {
            const bool usesLogicOperation = desc.logicOperation.has_value();
            if (usesLogicOperation && !support.logicOperations)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The device does not support output-merger logic operations.",
                });
            }
            if (usesLogicOperation && desc.colorTargets.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A logic operation requires at least one color target.",
                });
            }
            D3D12_BLEND_DESC blendState{};
            blendState.AlphaToCoverageEnable = desc.alphaToCoverageEnabled ? TRUE : FALSE;
            for (std::size_t index = 0; index < desc.colorTargets.size(); ++index)
            {
                auto targetValidation = validateColorTarget(device, desc, index);
                if (!targetValidation)
                {
                    return std::unexpected(std::move(targetValidation.error()));
                }
                blendState.RenderTarget[index] = nativeRenderTargetBlend(desc.colorTargets[index], desc.logicOperation);
            }
            const bool usesDualSourceBlending = std::ranges::any_of(
                desc.colorTargets, [](const ColorTargetState& target) { return blendUsesDualSource(target.blend); });
            if (usesDualSourceBlending && desc.colorTargets.size() != 1)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Dual-source blending is supported only for a graphics pipeline with exactly one color target.",
                });
            }
            if (usesDualSourceBlending && !support.dualSourceBlending)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The device does not support dual-source blending.",
                });
            }
            blendState.IndependentBlendEnable = usesDualSourceBlending || usesLogicOperation ? FALSE : TRUE;
            return blendState;
        }

        [[nodiscard]] Status validateDepthStencilState(ID3D12Device& device,
                                                       const DepthStencilState& depthStencil,
                                                       std::uint32_t sampleCount)
        {
            if (depthStencil.format != Format::unknown && depthStencil.format != Format::d32Float &&
                depthStencil.format != Format::d24UnormS8Uint)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics depth-stencil target requires a depth format.",
                });
            }
            if (depthStencil.format != Format::unknown)
            {
                auto sampleCountSupported = validateAttachmentSampleCount(device, depthStencil.format, sampleCount);
                if (!sampleCountSupported)
                {
                    return std::unexpected(std::move(sampleCountSupported.error()));
                }
            }
            if ((depthStencil.depthWriteEnabled && !depthStencil.depthTestEnabled) ||
                !isValidCompare(depthStencil.depthCompare) || !isValidStencilFace(depthStencil.frontFace) ||
                !isValidStencilFace(depthStencil.backFace))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The depth-stencil state contains an invalid compare or operation, or enables writes without depth "
                    "testing.",
                });
            }
            if (depthStencil.stencilTestEnabled && depthStencil.format != Format::d24UnormS8Uint)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Stencil testing requires the portable d24UnormS8Uint depth-stencil format.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr D3D12_DEPTH_STENCIL_DESC nativeDepthStencilState(
            const DepthStencilState& depthStencil) noexcept
        {
            return {
                .DepthEnable = depthStencil.depthTestEnabled ? TRUE : FALSE,
                .DepthWriteMask =
                    depthStencil.depthWriteEnabled ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO,
                .DepthFunc = nativeCompare(depthStencil.depthCompare),
                .StencilEnable = depthStencil.stencilTestEnabled ? TRUE : FALSE,
                .StencilReadMask = depthStencil.stencilReadMask,
                .StencilWriteMask = depthStencil.stencilWriteMask,
                .FrontFace = nativeStencilFace(depthStencil.frontFace),
                .BackFace = nativeStencilFace(depthStencil.backFace),
            };
        }

        struct OutputMergerState final
        {
            D3D12_BLEND_DESC blend{};
            D3D12_DEPTH_STENCIL_DESC depthStencil{};
        };

        [[nodiscard]] Result<OutputMergerState> buildOutputMergerState(ID3D12Device& device,
                                                                       const BlendFeatureSupport& support,
                                                                       const GraphicsPipelineDesc& desc)
        {
            auto blend = buildBlendState(device, support, desc);
            if (!blend)
            {
                return std::unexpected(std::move(blend.error()));
            }
            auto depthStencilValidation = validateDepthStencilState(device, desc.depthStencil, desc.sampleCount);
            if (!depthStencilValidation)
            {
                return std::unexpected(std::move(depthStencilValidation.error()));
            }
            return OutputMergerState{
                .blend = *blend,
                .depthStencil = nativeDepthStencilState(desc.depthStencil),
            };
        }

        [[nodiscard]] D3D12_SHADER_BYTECODE nativeBytecode(const std::shared_ptr<ShaderModuleRecord>& shader) noexcept
        {
            return shader == nullptr ? D3D12_SHADER_BYTECODE{}
                                     : D3D12_SHADER_BYTECODE{shader->bytecode->data(), shader->bytecode->size()};
        }

        [[nodiscard]] constexpr D3D12_RASTERIZER_DESC nativeRasterizerState(const GraphicsPipelineDesc& desc) noexcept
        {
            return {
                .FillMode = nativeFillMode(desc.fillMode),
                .CullMode = nativeCullMode(desc.cullMode),
                .FrontCounterClockwise = desc.frontFace == FrontFace::counterClockwise ? TRUE : FALSE,
                .DepthBias = desc.depthBias,
                .DepthBiasClamp = desc.depthBiasClamp,
                .SlopeScaledDepthBias = desc.depthBiasSlopeScale,
                .DepthClipEnable = desc.depthClipEnabled ? TRUE : FALSE,
                .MultisampleEnable = desc.sampleCount == 1 ? FALSE : TRUE,
                .AntialiasedLineEnable = FALSE,
                .ForcedSampleCount = 0,
                .ConservativeRaster = nativeConservativeRasterizationMode(desc.conservativeRasterization),
            };
        }

        [[nodiscard]] D3D12_GRAPHICS_PIPELINE_STATE_DESC nativeGraphicsPipelineDesc(
            const GraphicsPipelineDesc& desc,
            const GraphicsPipelineRecord& record,
            const StreamOutputLayout& streamOutput,
            std::span<const D3D12_INPUT_ELEMENT_DESC> inputElements,
            const OutputMergerState& outputMerger)
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC nativeDesc{};
            nativeDesc.pRootSignature = record.layout->rootSignature.Get();
            nativeDesc.VS = nativeBytecode(record.vertexShader);
            nativeDesc.PS = nativeBytecode(record.pixelShader);
            nativeDesc.GS = nativeBytecode(record.geometryShader);
            nativeDesc.HS = nativeBytecode(record.hullShader);
            nativeDesc.DS = nativeBytecode(record.domainShader);
            if (!desc.streamOutputEntries.empty())
            {
                nativeDesc.StreamOutput = {
                    .pSODeclaration = streamOutput.declarations.data(),
                    .NumEntries = static_cast<UINT>(streamOutput.declarations.size()),
                    .pBufferStrides = streamOutput.bufferStrides.data(),
                    .NumStrides = static_cast<UINT>(streamOutput.bufferStrides.size()),
                    .RasterizedStream = desc.rasterizedStream.value_or(D3D12_SO_NO_RASTERIZED_STREAM),
                };
            }
            nativeDesc.BlendState = outputMerger.blend;
            nativeDesc.SampleMask = desc.sampleMask;
            nativeDesc.RasterizerState = nativeRasterizerState(desc);
            nativeDesc.DepthStencilState = outputMerger.depthStencil;
            nativeDesc.InputLayout = {inputElements.data(), static_cast<UINT>(inputElements.size())};
            nativeDesc.IBStripCutValue = nativeStripCutValue(desc.stripIndexFormat);
            nativeDesc.PrimitiveTopologyType = nativeTopologyType(desc.topology);
            nativeDesc.NumRenderTargets = static_cast<UINT>(desc.colorTargets.size());
            for (std::size_t index = 0; index < desc.colorTargets.size(); ++index)
            {
                nativeDesc.RTVFormats[index] = nativeFormat(desc.colorTargets[index].format);
            }
            nativeDesc.DSVFormat = nativeFormat(desc.depthStencil.format);
            nativeDesc.SampleDesc = {desc.sampleCount, 0};
            nativeDesc.NodeMask = 0;
            nativeDesc.CachedPSO = {};
            nativeDesc.Flags = desc.dynamicDepthBiasEnabled ? D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS
                                                            : D3D12_PIPELINE_STATE_FLAG_NONE;
            return nativeDesc;
        }

        [[nodiscard]] std::array<D3D12_VIEW_INSTANCE_LOCATION, D3D12_MAX_VIEW_INSTANCE_COUNT> viewInstanceLocations(
            std::uint32_t viewMask) noexcept
        {
            std::array<D3D12_VIEW_INSTANCE_LOCATION, D3D12_MAX_VIEW_INSTANCE_COUNT> locations{};
            const auto viewInstanceCount = multiviewLayerCount(viewMask);
            for (std::uint32_t index = 0; index < viewInstanceCount; ++index)
            {
                locations[index] = {
                    .ViewportArrayIndex = 0,
                    .RenderTargetArrayIndex = index,
                };
            }
            return locations;
        }

        struct NativeGraphicsPipelineRequest final
        {
            ID3D12Device* device = nullptr;
            ID3D12Device2* stateStreamDevice = nullptr;
            const D3D12_GRAPHICS_PIPELINE_STATE_DESC* desc = nullptr;
            const D3D12_PIPELINE_STATE_STREAM_DESC* stateStreamDesc = nullptr;

            [[nodiscard]] bool usesStateStream() const noexcept
            {
                return stateStreamDesc != nullptr;
            }
        };

        [[nodiscard]] Error graphicsPipelineCreationError(HRESULT result, const GraphicsPipelineDesc& desc)
        {
            const char* operation = "ID3D12Device::CreateGraphicsPipelineState";
            if (desc.meshShader.isValid())
            {
                operation = "ID3D12Device2::CreatePipelineState(mesh)";
            }
            else if (desc.viewMask != 0)
            {
                operation = "ID3D12Device2::CreatePipelineState(multiview)";
            }
            else if (desc.depthStencil.depthBoundsTestEnabled)
            {
                operation = "ID3D12Device2::CreatePipelineState(depth bounds)";
            }
            return nativeError(pipelineCreationErrorCode(result), result, operation);
        }

        [[nodiscard]] Result<ComPtr<ID3D12PipelineState>> createNativeGraphicsPipeline(
            const NativeGraphicsPipelineRequest& request, const GraphicsPipelineDesc& desc)
        {
            ComPtr<ID3D12PipelineState> pipelineState;
            const HRESULT result =
                request.usesStateStream()
                    ? request.stateStreamDevice->CreatePipelineState(request.stateStreamDesc,
                                                                     IID_PPV_ARGS(&pipelineState))
                    : request.device->CreateGraphicsPipelineState(request.desc, IID_PPV_ARGS(&pipelineState));
            if (FAILED(result))
            {
                return std::unexpected(graphicsPipelineCreationError(result, desc));
            }
            return pipelineState;
        }

        [[nodiscard]] Status validateCachedGraphicsPipelineStream(const PipelineCacheRecord& cache,
                                                                  const NativeGraphicsPipelineRequest& request)
        {
            if (request.usesStateStream() && cache.library1 == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The pipeline cache cannot load this graphics pipeline-state stream on the device.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<ComPtr<ID3D12PipelineState>> loadCachedGraphicsPipeline(
            PipelineCacheRecord& cache, const std::wstring& cacheName, const NativeGraphicsPipelineRequest& request)
        {
            auto streamSupport = validateCachedGraphicsPipelineStream(cache, request);
            if (!streamSupport)
            {
                return std::unexpected(std::move(streamSupport.error()));
            }
            ComPtr<ID3D12PipelineState> pipelineState;
            const HRESULT loadResult = request.usesStateStream()
                                           ? cache.library1->LoadPipeline(cacheName.c_str(),
                                                                          request.stateStreamDesc,
                                                                          IID_PPV_ARGS(&pipelineState))
                                           : cache.library->LoadGraphicsPipeline(
                                                 cacheName.c_str(), request.desc, IID_PPV_ARGS(&pipelineState));
            if (loadResult == E_INVALIDARG)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    static_cast<std::int64_t>(loadResult),
                    "The graphics pipeline cache key identifies an incompatible pipeline.",
                });
            }
            if (FAILED(loadResult))
            {
                return std::unexpected(pipelineLibraryError(loadResult, "ID3D12PipelineLibrary::LoadGraphicsPipeline"));
            }
            return pipelineState;
        }

        [[nodiscard]] Status storeGraphicsPipelineInCache(PipelineCacheRecord& cache,
                                                          const PipelineCacheSlot& slot,
                                                          ID3D12PipelineState* pipelineState)
        {
            std::set<PipelineCacheEntry>::iterator recorded;
            try
            {
                recorded = recordPipelineCacheEntry(cache, slot.entry);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to record the graphics pipeline cache entry.",
                });
            }
            const HRESULT storeResult = cache.library->StorePipeline(slot.name.c_str(), pipelineState);
            if (FAILED(storeResult))
            {
                cache.entries.erase(recorded);
            }
            if (storeResult == E_INVALIDARG)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    static_cast<std::int64_t>(storeResult),
                    "The graphics pipeline cache metadata is inconsistent with its native library.",
                });
            }
            if (FAILED(storeResult))
            {
                return std::unexpected(
                    pipelineLibraryError(storeResult, "ID3D12PipelineLibrary::StorePipeline(graphics)"));
            }
            return {};
        }

        [[nodiscard]] Result<ComPtr<ID3D12PipelineState>> storeOrAdoptCachedGraphicsPipeline(
            PipelineCacheRecord& cache,
            const PipelineCacheSlot& slot,
            const NativeGraphicsPipelineRequest& request,
            ComPtr<ID3D12PipelineState> pipelineState)
        {
            const std::scoped_lock cacheLock(cache.mutex);
            if (!cache.entries.contains(slot.entry))
            {
                auto stored = storeGraphicsPipelineInCache(cache, slot, pipelineState.Get());
                if (!stored)
                {
                    return std::unexpected(std::move(stored.error()));
                }
                return pipelineState;
            }
            auto cachedState = loadCachedGraphicsPipeline(cache, slot.name, request);
            if (!cachedState)
            {
                return std::unexpected(std::move(cachedState.error()));
            }
            pipelineState = std::move(*cachedState);
            return pipelineState;
        }

        [[nodiscard]] Result<ComPtr<ID3D12PipelineState>> createCachedGraphicsPipeline(
            const NativeGraphicsPipelineRequest& request, PipelineCacheRecord& cache, const GraphicsPipelineDesc& desc)
        {
            auto streamSupport = validateCachedGraphicsPipelineStream(cache, request);
            if (!streamSupport)
            {
                return std::unexpected(std::move(streamSupport.error()));
            }
            const auto cacheKind =
                desc.meshShader.isValid() ? PipelineCacheEntryKind::mesh : PipelineCacheEntryKind::graphics;
            PipelineCacheSlot slot;
            try
            {
                slot = pipelineCacheSlot(desc.cacheKey, cacheKind);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate the graphics pipeline cache entry name.",
                });
            }
            {
                const std::scoped_lock cacheLock(cache.mutex);
                if (cache.entries.contains(slot.entry))
                {
                    return loadCachedGraphicsPipeline(cache, slot.name, request);
                }
                if (desc.cachePolicy == PipelineCachePolicy::requireHit)
                {
                    return std::unexpected(Error{
                        ErrorCode::pipelineCacheMiss,
                        0,
                        "The graphics pipeline is absent from the requested pipeline cache.",
                    });
                }
            }
            auto created = createNativeGraphicsPipeline(request, desc);
            if (!created)
            {
                return std::unexpected(std::move(created.error()));
            }
            return storeOrAdoptCachedGraphicsPipeline(cache, slot, request, std::move(*created));
        }

    } // namespace

    Status PipelineManager::Impl::validateGraphicsFixedFunctionState(const GraphicsPipelineDesc& desc) const
    {
        const bool withinClipAndCullDistanceLimits =
            desc.clipDistanceCount <= maxClipDistances && desc.cullDistanceCount <= maxCullDistances &&
            desc.clipDistanceCount + desc.cullDistanceCount <= maxCombinedClipAndCullDistances;
        const bool depthBoundsTestHasDepthTarget =
            !desc.depthStencil.depthBoundsTestEnabled || desc.depthStencil.format != Format::unknown;
        const bool validLogicOperation =
            !desc.logicOperation.has_value() || isValidLogicOperation(*desc.logicOperation);
        if (!hasValidAttachmentsAndVertexBufferCount(desc) || !hasValidMultisampleState(desc) ||
            !isValidFillMode(desc.fillMode) || !isValidConservativeRasterizationMode(desc.conservativeRasterization) ||
            !withinClipAndCullDistanceLimits || !hasValidDepthBias(desc) || !depthBoundsTestHasDepthTarget ||
            !validLogicOperation || !isContiguousViewMask(desc.viewMask) ||
            !isValidPipelineCachePolicy(desc.cachePolicy))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Graphics pipelines require valid attachment, rasterization, and multisample state.",
            });
        }
        return {};
    }

    Status PipelineManager::Impl::validateGraphicsFeatureSupport(const GraphicsPipelineDesc& desc) const
    {
        if (desc.meshShader.isValid() && !meshShaders)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support mesh shaders.",
            });
        }
        if (desc.taskShader.isValid() && !taskShaders)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support task shaders.",
            });
        }
        if (desc.dynamicDepthBiasEnabled && !dynamicDepthBias)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support dynamic depth bias.",
            });
        }
        if (desc.depthStencil.depthBoundsTestEnabled && !depthBoundsTest)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support depth-bounds testing.",
            });
        }
        if (desc.geometryShader.isValid() && !geometryShaders)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support geometry shaders.",
            });
        }
        return {};
    }

    Status PipelineManager::Impl::validateTessellationAndTopologySupport(const GraphicsPipelineDesc& desc) const
    {
        const bool hasHullShader = desc.hullShader.isValid();
        const bool hasDomainShader = desc.domainShader.isValid();
        const bool hasTessellationPair = hasHullShader && hasDomainShader;
        const bool validOrdinaryTopology = desc.topology != PrimitiveTopology::patchList && !hasHullShader &&
                                           !hasDomainShader && desc.patchControlPoints == 0;
        const bool validPatchTopology = desc.topology == PrimitiveTopology::patchList && hasTessellationPair &&
                                        desc.patchControlPoints >= 1 && desc.patchControlPoints <= 32;
        if (!validOrdinaryTopology && !validPatchTopology)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Tessellation requires paired hull and domain shaders, patch-list topology, and 1-32 control points.",
            });
        }
        if (hasTessellationPair && (maxPatchControlPoints == 0 || desc.patchControlPoints > maxPatchControlPoints))
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support the requested tessellation patch size.",
            });
        }
        if (desc.topology == PrimitiveTopology::triangleFan && !triangleFans)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support triangle-fan topology.",
            });
        }
        return {};
    }

    Status PipelineManager::Impl::validateGraphicsPipelineDesc(const GraphicsPipelineDesc& desc) const
    {
        auto fixedFunctionState = validateGraphicsFixedFunctionState(desc);
        if (!fixedFunctionState)
        {
            return std::unexpected(std::move(fixedFunctionState.error()));
        }
        auto meshStages = validateMeshShaderStages(desc);
        if (!meshStages)
        {
            return std::unexpected(std::move(meshStages.error()));
        }
        auto featureSupport = validateGraphicsFeatureSupport(desc);
        if (!featureSupport)
        {
            return std::unexpected(std::move(featureSupport.error()));
        }
        return validateTessellationAndTopologySupport(desc);
    }

    Status PipelineManager::Impl::validatePrimitiveRestartAndConservativeRasterization(
        const GraphicsPipelineDesc& desc) const
    {
        const bool usesMeshPipeline = desc.meshShader.isValid();
        const bool hasNativeTopology =
            nativeTopology(desc.topology, desc.patchControlPoints) != D3D_PRIMITIVE_TOPOLOGY_UNDEFINED &&
            nativeTopologyType(desc.topology) != D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
        const bool hasValidPrimitiveRestart =
            !desc.stripIndexFormat.has_value() ||
            (isStripTopology(desc.topology) && isValidIndexFormat(*desc.stripIndexFormat));
        if (!usesMeshPipeline && (!hasNativeTopology || !hasValidPrimitiveRestart))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Primitive restart requires a valid line or triangle strip topology and index format.",
            });
        }
        if (desc.conservativeRasterization == ConservativeRasterizationMode::overestimate)
        {
            if (!conservativeRasterization)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The device does not support conservative rasterization.",
                });
            }
            if (desc.fillMode != FillMode::solid ||
                (!usesMeshPipeline && nativeTopologyType(desc.topology) != D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Conservative rasterization requires solid triangle topology.",
                });
            }
        }
        return {};
    }

    Status PipelineManager::Impl::validateMultiviewSupport(const GraphicsPipelineDesc& desc) const
    {
        if (desc.viewMask != 0 &&
            (multiviewLayerCount(desc.viewMask) > maxMultiviewViewCount || pipelineStateStreamDevice == nullptr))
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The graphics pipeline view mask exceeds the device's multiview support.",
            });
        }
        return {};
    }

    Result<ComPtr<ID3D12PipelineState>> PipelineManager::Impl::createGraphicsPipelineState(
        const GraphicsPipelineDesc& desc,
        const D3D12_GRAPHICS_PIPELINE_STATE_DESC& nativeDesc,
        PipelineCacheRecord* cache,
        const GraphicsPipelineRecord& record) const
    {
        const auto viewLocations = viewInstanceLocations(desc.viewMask);
        const D3D12_VIEW_INSTANCING_DESC viewInstancing{
            .ViewInstanceCount = multiviewLayerCount(desc.viewMask),
            .pViewInstanceLocations = viewLocations.data(),
            .Flags = D3D12_VIEW_INSTANCING_FLAG_ENABLE_VIEW_INSTANCE_MASKING,
        };
        const bool usesMeshPipeline = desc.meshShader.isValid();
        const bool usesMultiview = desc.viewMask != 0;
        const bool usesDepthBoundsTest = desc.depthStencil.depthBoundsTestEnabled;
        auto graphicsStream = graphicsPipelineStateStream(nativeDesc, usesDepthBoundsTest, viewInstancing);
        const D3D12_PIPELINE_STATE_STREAM_DESC graphicsStreamDesc{
            .SizeInBytes =
                usesMultiview ? sizeof(graphicsStream) : offsetof(GraphicsPipelineStateStream, viewInstancing),
            .pPipelineStateSubobjectStream = &graphicsStream,
        };
        auto meshStream = meshPipelineStateStream(nativeDesc,
                                                  nativeBytecode(record.taskShader),
                                                  nativeBytecode(record.meshShader),
                                                  usesDepthBoundsTest,
                                                  viewInstancing);
        const D3D12_PIPELINE_STATE_STREAM_DESC meshStreamDesc{
            .SizeInBytes = usesMultiview ? sizeof(meshStream) : offsetof(MeshPipelineStateStream, viewInstancing),
            .pPipelineStateSubobjectStream = &meshStream,
        };
        const auto& activeStreamDesc = usesMeshPipeline ? meshStreamDesc : graphicsStreamDesc;
        const bool usesPipelineStateStream = usesMeshPipeline || usesMultiview || usesDepthBoundsTest;
        const NativeGraphicsPipelineRequest request{
            .device = device,
            .stateStreamDevice = pipelineStateStreamDevice.Get(),
            .desc = &nativeDesc,
            .stateStreamDesc = usesPipelineStateStream ? &activeStreamDesc : nullptr,
        };
        if (cache != nullptr)
        {
            return createCachedGraphicsPipeline(request, *cache, desc);
        }
        return createNativeGraphicsPipeline(request, desc);
    }

    Result<PipelineHandle> PipelineManager::createGraphicsPipeline(const GraphicsPipelineDesc& desc)
    {
        auto descValidation = impl_->validateGraphicsPipelineDesc(desc);
        if (!descValidation)
        {
            return std::unexpected(std::move(descValidation.error()));
        }
        auto resolved = [&]() {
            const std::scoped_lock lock(impl_->mutex);
            return resolveGraphicsPipelineRecords(impl_->pipelineLayouts, impl_->shaderModules, desc);
        }();
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        auto interfaceValidation = validateShaderInterfaces(*resolved);
        if (!interfaceValidation)
        {
            return std::unexpected(std::move(interfaceValidation.error()));
        }
        if (auto linkage = validateGraphicsStageLinkage(*resolved, desc.rasterizedStream); !linkage)
        {
            return std::unexpected(std::move(linkage.error()));
        }
        auto cache = impl_->pipelineCache(desc.cache, desc.cacheKey);
        if (!cache)
        {
            return std::unexpected(std::move(cache.error()));
        }
        if (desc.cachePolicy == PipelineCachePolicy::requireHit && *cache == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A cache-only graphics pipeline requires a live cache and non-zero key.",
            });
        }
        auto rasterizationValidation = impl_->validatePrimitiveRestartAndConservativeRasterization(desc);
        if (!rasterizationValidation)
        {
            return std::unexpected(std::move(rasterizationValidation.error()));
        }
        auto multiviewValidation = impl_->validateMultiviewSupport(desc);
        if (!multiviewValidation)
        {
            return std::unexpected(std::move(multiviewValidation.error()));
        }
        auto streamOutput = buildStreamOutputLayout(desc);
        if (!streamOutput)
        {
            return std::unexpected(std::move(streamOutput.error()));
        }
        auto vertexInput = buildVertexInputLayout(*impl_->device, desc.vertexBuffers, resolved->vertexShader.get());
        if (!vertexInput)
        {
            return std::unexpected(std::move(vertexInput.error()));
        }

        auto record = makeGraphicsPipelineRecord(desc, std::move(*resolved), std::move(vertexInput->strides));
        const BlendFeatureSupport blendSupport{
            .logicOperations = impl_->logicOperations,
            .dualSourceBlending = impl_->dualSourceBlending,
        };
        auto outputMerger = buildOutputMergerState(*impl_->device, blendSupport, desc);
        if (!outputMerger)
        {
            return std::unexpected(std::move(outputMerger.error()));
        }
        const auto nativeDesc =
            nativeGraphicsPipelineDesc(desc, *record, *streamOutput, vertexInput->elements, *outputMerger);
        auto pipelineState = impl_->createGraphicsPipelineState(desc, nativeDesc, cache->get(), *record);
        if (!pipelineState)
        {
            return std::unexpected(std::move(pipelineState.error()));
        }
        record->pipelineState = std::move(*pipelineState);
        setDebugName(*record->pipelineState.Get(), desc.label);

        const std::scoped_lock lock(impl_->mutex);
        return impl_->pipelines.emplace(PipelineRecord{std::move(record)});
    }

    Result<PipelineCompilationHandle> PipelineManager::beginComputePipelineCompilation(const ComputePipelineDesc& desc)
    {
        OwnedComputePipelineDesc owned(desc);
        auto record = std::make_shared<PipelineCompilationRecord>();
        auto job = std::make_shared<PipelineCompilationJob>(PipelineCompilationJob{
            .record = record,
            .dependencies = {},
            .desc = std::move(owned),
        });
        auto retained = impl_->retainDependencies(std::get<OwnedComputePipelineDesc>(job->desc));
        if (!retained)
        {
            return std::unexpected(std::move(retained.error()));
        }
        job->dependencies = std::move(*retained);
        ScopeRollback dependencyRollback([&]() noexcept { impl_->releaseDependencies(job->dependencies); });

        auto inserted = [&]() {
            const std::scoped_lock lock(impl_->mutex);
            return impl_->pipelineCompilations.emplace(record);
        }();
        if (!inserted)
        {
            return std::unexpected(std::move(inserted.error()));
        }
        const auto handle = *inserted;
        ScopeRollback handleRollback([&]() noexcept {
            const std::scoped_lock lock(impl_->mutex);
            static_cast<void>(impl_->pipelineCompilations.erase(handle));
        });
        auto queued = impl_->enqueueCompilation(job);
        if (!queued)
        {
            return std::unexpected(std::move(queued.error()));
        }
        handleRollback.release();
        dependencyRollback.release();
        return handle;
    }

    Result<PipelineCompilationHandle> PipelineManager::beginGraphicsPipelineCompilation(
        const GraphicsPipelineDesc& desc)
    {
        OwnedGraphicsPipelineDesc owned(desc);
        auto record = std::make_shared<PipelineCompilationRecord>();
        auto job = std::make_shared<PipelineCompilationJob>(PipelineCompilationJob{
            .record = record,
            .dependencies = {},
            .desc = std::move(owned),
        });
        auto retained = impl_->retainDependencies(std::get<OwnedGraphicsPipelineDesc>(job->desc));
        if (!retained)
        {
            return std::unexpected(std::move(retained.error()));
        }
        job->dependencies = std::move(*retained);
        ScopeRollback dependencyRollback([&]() noexcept { impl_->releaseDependencies(job->dependencies); });

        auto inserted = [&]() {
            const std::scoped_lock lock(impl_->mutex);
            return impl_->pipelineCompilations.emplace(record);
        }();
        if (!inserted)
        {
            return std::unexpected(std::move(inserted.error()));
        }
        const auto handle = *inserted;
        ScopeRollback handleRollback([&]() noexcept {
            const std::scoped_lock lock(impl_->mutex);
            static_cast<void>(impl_->pipelineCompilations.erase(handle));
        });
        auto queued = impl_->enqueueCompilation(job);
        if (!queued)
        {
            return std::unexpected(std::move(queued.error()));
        }
        handleRollback.release();
        dependencyRollback.release();
        return handle;
    }

    Result<PipelineCompilationState> PipelineManager::pipelineCompilationState(PipelineCompilationHandle compilation)
    {
        std::shared_ptr<PipelineCompilationRecord> record;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* found = impl_->pipelineCompilations.find(compilation);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline-compilation handle is invalid or stale.",
                });
            }
            record = *found;
        }

        const std::scoped_lock lock(record->mutex);
        if (record->consumed || record->abandoned)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The pipeline-compilation request has already been consumed or abandoned.",
            });
        }
        return record->state;
    }

    Result<PipelineHandle> PipelineManager::waitPipelineCompilation(PipelineCompilationHandle compilation)
    {
        std::shared_ptr<PipelineCompilationRecord> record;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* found = impl_->pipelineCompilations.find(compilation);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline-compilation handle is invalid or stale.",
                });
            }
            record = *found;
        }

        Result<PipelineHandle> result = std::unexpected(Error{
            ErrorCode::invalidState,
            0,
            "The pipeline-compilation request has no result.",
        });
        {
            std::unique_lock lock(record->mutex);
            record->condition.wait(
                lock, [&record]() { return record->result.has_value() || record->consumed || record->abandoned; });
            if (record->consumed || record->abandoned || !record->result.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The pipeline-compilation request was already consumed or abandoned.",
                });
            }
            result = std::move(*record->result);
            record->consumed = true;
        }
        record->condition.notify_all();
        {
            const std::scoped_lock lock(impl_->mutex);
            static_cast<void>(impl_->pipelineCompilations.erase(compilation));
        }
        return result;
    }

    Status PipelineManager::destroyPipelineCompilation(PipelineCompilationHandle compilation)
    {
        std::shared_ptr<PipelineCompilationRecord> record;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* found = impl_->pipelineCompilations.find(compilation);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline-compilation handle is invalid or stale.",
                });
            }
            record = *found;
            static_cast<void>(impl_->pipelineCompilations.erase(compilation));
        }

        PipelineHandle abandonedPipeline;
        {
            const std::scoped_lock lock(record->mutex);
            if (record->consumed)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The pipeline-compilation request has already been consumed.",
                });
            }
            record->abandoned = true;
            if (record->result.has_value() && *record->result)
            {
                abandonedPipeline = **record->result;
            }
        }
        record->condition.notify_all();
        if (abandonedPipeline.isValid())
        {
            return destroyPipeline(abandonedPipeline);
        }
        return {};
    }

    Status PipelineManager::destroyPipeline(PipelineHandle pipeline)
    {
        const std::scoped_lock lock(impl_->mutex);
        if (!impl_->pipelines.erase(pipeline))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<NativeShaderModule> PipelineManager::nativeShaderModule(ShaderModuleHandle shaderModule)
    {
        const std::scoped_lock lock(impl_->mutex);
        const auto* found = impl_->shaderModules.find(shaderModule);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader module handle is invalid or stale.",
            });
        }
        return NativeShaderModule{
            .owner = *found,
            .stage = (*found)->stage,
            .bytecode =
                {
                    .pShaderBytecode = (*found)->bytecode->data(),
                    .BytecodeLength = (*found)->bytecode->size(),
                },
            .entryPoint = (*found)->entryPoint,
        };
    }

    Result<NativePipelineLayout> PipelineManager::nativePipelineLayout(PipelineLayoutHandle pipelineLayout)
    {
        const std::scoped_lock lock(impl_->mutex);
        const auto* found = impl_->pipelineLayouts.find(pipelineLayout);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline layout handle is invalid or stale.",
            });
        }
        return NativePipelineLayout{
            .owner = *found,
            .rootSignature = (*found)->rootSignature.Get(),
            .slots = (*found)->slots,
            .pushConstantRanges = (*found)->pushConstantRanges,
            .pushConstantRootParameter = (*found)->pushConstantRootParameter,
            .pushConstantSize = (*found)->pushConstantSize,
        };
    }

} // namespace rachis::d3d12
