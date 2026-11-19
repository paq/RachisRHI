#include "Internal/VulkanPipelines.h"

#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanRuntime.h"
#include "Internal/VulkanValidationHelpers.h"
#include "Rachis/Binding.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rachis::vulkan
{
    namespace
    {

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

        [[nodiscard]] constexpr bool isValidFillMode(FillMode mode) noexcept
        {
            return mode == FillMode::solid || mode == FillMode::wireframe;
        }

        [[nodiscard]] constexpr bool isValidConservativeRasterizationMode(ConservativeRasterizationMode mode) noexcept
        {
            return mode == ConservativeRasterizationMode::disabled ||
                   mode == ConservativeRasterizationMode::overestimate;
        }

        [[nodiscard]] constexpr bool isValidCompare(CompareOperation operation) noexcept
        {
            return operation >= CompareOperation::never && operation <= CompareOperation::always;
        }

        [[nodiscard]] constexpr bool isValidStencilOperation(StencilOperation operation) noexcept
        {
            return operation >= StencilOperation::keep && operation <= StencilOperation::decrementWrap;
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

        [[nodiscard]] constexpr bool isValidStencilFace(const StencilFaceState& face) noexcept
        {
            return isValidCompare(face.compare) && isValidStencilOperation(face.failOperation) &&
                   isValidStencilOperation(face.depthFailOperation) && isValidStencilOperation(face.passOperation);
        }

        [[nodiscard]] constexpr bool usesBlendConstant(BlendFactor factor) noexcept
        {
            return factor == BlendFactor::constant || factor == BlendFactor::oneMinusConstant;
        }

        [[nodiscard]] constexpr bool usesDualSource(BlendFactor factor) noexcept
        {
            return factor >= BlendFactor::source1 && factor <= BlendFactor::oneMinusSource1Alpha;
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

        [[nodiscard]] constexpr VkLogicOp vulkanLogicOp(LogicOperation operation) noexcept
        {
            switch (operation)
            {
            case LogicOperation::clear:
                return VK_LOGIC_OP_CLEAR;
            case LogicOperation::andOperation:
                return VK_LOGIC_OP_AND;
            case LogicOperation::andReverse:
                return VK_LOGIC_OP_AND_REVERSE;
            case LogicOperation::copy:
                return VK_LOGIC_OP_COPY;
            case LogicOperation::andInverted:
                return VK_LOGIC_OP_AND_INVERTED;
            case LogicOperation::noOp:
                return VK_LOGIC_OP_NO_OP;
            case LogicOperation::xorOperation:
                return VK_LOGIC_OP_XOR;
            case LogicOperation::orOperation:
                return VK_LOGIC_OP_OR;
            case LogicOperation::nor:
                return VK_LOGIC_OP_NOR;
            case LogicOperation::equivalent:
                return VK_LOGIC_OP_EQUIVALENT;
            case LogicOperation::invert:
                return VK_LOGIC_OP_INVERT;
            case LogicOperation::orReverse:
                return VK_LOGIC_OP_OR_REVERSE;
            case LogicOperation::copyInverted:
                return VK_LOGIC_OP_COPY_INVERTED;
            case LogicOperation::orInverted:
                return VK_LOGIC_OP_OR_INVERTED;
            case LogicOperation::nand:
                return VK_LOGIC_OP_NAND;
            case LogicOperation::set:
                return VK_LOGIC_OP_SET;
            }
            return VK_LOGIC_OP_COPY;
        }

        [[nodiscard]] constexpr bool isStripTopology(PrimitiveTopology topology) noexcept
        {
            return topology == PrimitiveTopology::lineStrip || topology == PrimitiveTopology::triangleStrip;
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

        [[nodiscard]] constexpr VkPrimitiveTopology vulkanTopology(PrimitiveTopology topology) noexcept
        {
            switch (topology)
            {
            case PrimitiveTopology::pointList:
                return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            case PrimitiveTopology::lineList:
                return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            case PrimitiveTopology::lineStrip:
                return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            case PrimitiveTopology::triangleList:
                return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            case PrimitiveTopology::triangleStrip:
                return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
            case PrimitiveTopology::patchList:
                return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
            case PrimitiveTopology::triangleFan:
                return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
            }
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        }

        [[nodiscard]] constexpr VkCullModeFlags vulkanCullMode(CullMode mode) noexcept
        {
            switch (mode)
            {
            case CullMode::none:
                return VK_CULL_MODE_NONE;
            case CullMode::front:
                return VK_CULL_MODE_FRONT_BIT;
            case CullMode::back:
                return VK_CULL_MODE_BACK_BIT;
            }
            return VK_CULL_MODE_BACK_BIT;
        }

        [[nodiscard]] constexpr VkCompareOp vulkanCompare(CompareOperation operation) noexcept
        {
            switch (operation)
            {
            case CompareOperation::never:
                return VK_COMPARE_OP_NEVER;
            case CompareOperation::less:
                return VK_COMPARE_OP_LESS;
            case CompareOperation::equal:
                return VK_COMPARE_OP_EQUAL;
            case CompareOperation::lessEqual:
                return VK_COMPARE_OP_LESS_OR_EQUAL;
            case CompareOperation::greater:
                return VK_COMPARE_OP_GREATER;
            case CompareOperation::notEqual:
                return VK_COMPARE_OP_NOT_EQUAL;
            case CompareOperation::greaterEqual:
                return VK_COMPARE_OP_GREATER_OR_EQUAL;
            case CompareOperation::always:
                return VK_COMPARE_OP_ALWAYS;
            }
            return VK_COMPARE_OP_ALWAYS;
        }

        [[nodiscard]] constexpr VkStencilOp vulkanStencilOp(StencilOperation operation) noexcept
        {
            switch (operation)
            {
            case StencilOperation::keep:
                return VK_STENCIL_OP_KEEP;
            case StencilOperation::zero:
                return VK_STENCIL_OP_ZERO;
            case StencilOperation::replace:
                return VK_STENCIL_OP_REPLACE;
            case StencilOperation::incrementClamp:
                return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
            case StencilOperation::decrementClamp:
                return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
            case StencilOperation::invert:
                return VK_STENCIL_OP_INVERT;
            case StencilOperation::incrementWrap:
                return VK_STENCIL_OP_INCREMENT_AND_WRAP;
            case StencilOperation::decrementWrap:
                return VK_STENCIL_OP_DECREMENT_AND_WRAP;
            }
            return VK_STENCIL_OP_KEEP;
        }

        [[nodiscard]] constexpr VkBlendFactor vulkanBlendFactor(BlendFactor factor) noexcept
        {
            switch (factor)
            {
            case BlendFactor::zero:
                return VK_BLEND_FACTOR_ZERO;
            case BlendFactor::one:
                return VK_BLEND_FACTOR_ONE;
            case BlendFactor::source:
                return VK_BLEND_FACTOR_SRC_COLOR;
            case BlendFactor::oneMinusSource:
                return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
            case BlendFactor::sourceAlpha:
                return VK_BLEND_FACTOR_SRC_ALPHA;
            case BlendFactor::oneMinusSourceAlpha:
                return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            case BlendFactor::destination:
                return VK_BLEND_FACTOR_DST_COLOR;
            case BlendFactor::oneMinusDestination:
                return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
            case BlendFactor::destinationAlpha:
                return VK_BLEND_FACTOR_DST_ALPHA;
            case BlendFactor::oneMinusDestinationAlpha:
                return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
            case BlendFactor::sourceAlphaSaturated:
                return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
            case BlendFactor::constant:
                return VK_BLEND_FACTOR_CONSTANT_COLOR;
            case BlendFactor::oneMinusConstant:
                return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
            case BlendFactor::source1:
                return VK_BLEND_FACTOR_SRC1_COLOR;
            case BlendFactor::oneMinusSource1:
                return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
            case BlendFactor::source1Alpha:
                return VK_BLEND_FACTOR_SRC1_ALPHA;
            case BlendFactor::oneMinusSource1Alpha:
                return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
            }
            return VK_BLEND_FACTOR_ONE;
        }

        [[nodiscard]] constexpr VkBlendOp vulkanBlendOp(BlendOperation operation) noexcept
        {
            switch (operation)
            {
            case BlendOperation::add:
                return VK_BLEND_OP_ADD;
            case BlendOperation::subtract:
                return VK_BLEND_OP_SUBTRACT;
            case BlendOperation::reverseSubtract:
                return VK_BLEND_OP_REVERSE_SUBTRACT;
            case BlendOperation::minimum:
                return VK_BLEND_OP_MIN;
            case BlendOperation::maximum:
                return VK_BLEND_OP_MAX;
            }
            return VK_BLEND_OP_ADD;
        }

        [[nodiscard]] constexpr VkColorComponentFlags vulkanColorWriteMask(ColorWriteMask mask) noexcept
        {
            VkColorComponentFlags flags = 0;
            const auto bits = static_cast<std::uint8_t>(mask);
            if ((bits & static_cast<std::uint8_t>(ColorWriteMask::red)) != 0)
            {
                flags |= VK_COLOR_COMPONENT_R_BIT;
            }
            if ((bits & static_cast<std::uint8_t>(ColorWriteMask::green)) != 0)
            {
                flags |= VK_COLOR_COMPONENT_G_BIT;
            }
            if ((bits & static_cast<std::uint8_t>(ColorWriteMask::blue)) != 0)
            {
                flags |= VK_COLOR_COMPONENT_B_BIT;
            }
            if ((bits & static_cast<std::uint8_t>(ColorWriteMask::alpha)) != 0)
            {
                flags |= VK_COLOR_COMPONENT_A_BIT;
            }
            return flags;
        }

        [[nodiscard]] constexpr VkSampleCountFlagBits vulkanSampleCount(std::uint32_t sampleCount) noexcept
        {
            switch (sampleCount)
            {
            case 1:
                return VK_SAMPLE_COUNT_1_BIT;
            case 2:
                return VK_SAMPLE_COUNT_2_BIT;
            case 4:
                return VK_SAMPLE_COUNT_4_BIT;
            case 8:
                return VK_SAMPLE_COUNT_8_BIT;
            default:
                break;
            }
            return VK_SAMPLE_COUNT_1_BIT;
        }

        struct ShaderSpecialization final
        {
            std::vector<VkSpecializationMapEntry> entries;
            std::vector<std::uint32_t> values;
            VkSpecializationInfo info{};

            [[nodiscard]] const VkSpecializationInfo* pointer() const noexcept
            {
                return entries.empty() ? nullptr : &info;
            }
        };

        [[nodiscard]] ShaderSpecialization makeSpecialization(std::span<const SpecializationConstant> constants)
        {
            ShaderSpecialization specialization{};
            specialization.entries.reserve(constants.size());
            specialization.values.reserve(constants.size());
            for (std::size_t index = 0; index < constants.size(); ++index)
            {
                specialization.entries.push_back(VkSpecializationMapEntry{
                    .constantID = constants[index].id,
                    .offset = static_cast<std::uint32_t>(index * sizeof(std::uint32_t)),
                    .size = sizeof(std::uint32_t),
                });
                specialization.values.push_back(constants[index].valueBits);
            }
            specialization.info = VkSpecializationInfo{
                .mapEntryCount = static_cast<std::uint32_t>(specialization.entries.size()),
                .pMapEntries = specialization.entries.data(),
                .dataSize = specialization.values.size() * sizeof(std::uint32_t),
                .pData = specialization.values.data(),
            };
            return specialization;
        }

        [[nodiscard]] Status validateShaderInterface(const VulkanShaderManager::NativeShaderModule& shader,
                                                     const VulkanNativePipelineLayout& layout)
        {
            for (const auto& binding : shader.bindings)
            {
                if (binding.set >= layout.sets.size())
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "A shader-interface binding references a set absent from the pipeline layout.",
                    });
                }
                const auto& entries = layout.sets[binding.set].entries;
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
            if (shader.drawIndex && !layout.drawIndex)
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "The shader reads the draw index, but the pipeline layout does not reserve it.",
                });
            }
            return validation_detail::validateShaderPushConstantCoverage(
                shader.pushConstants, layout.pushConstantRanges, shader.stage);
        }

        [[nodiscard]] Status validateVertexInterface(const VulkanShaderManager::NativeShaderModule& shader,
                                                     std::span<const std::uint32_t> declaredLocations)
        {
            std::vector<std::uint32_t> declared{declaredLocations.begin(), declaredLocations.end()};
            std::ranges::sort(declared);
            if (declared != shader.vertexInputLocations)
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "The graphics pipeline's vertex locations do not exactly match the shader interface.",
                });
            }
            return {};
        }

        inline constexpr std::uint64_t pipelineCacheBlobMagic = 0x31304B5643494852ULL;
        inline constexpr std::uint32_t pipelineCacheBlobVersion = 1;

        struct PipelineCacheBlobHeader final
        {
            std::uint64_t magic = pipelineCacheBlobMagic;
            std::uint32_t version = pipelineCacheBlobVersion;
            std::uint32_t entryCount = 0;
            std::uint64_t nativeDataSize = 0;
            std::array<std::uint8_t, VK_UUID_SIZE> pipelineCacheUuid{};
            std::array<std::uint8_t, VK_UUID_SIZE> deviceUuid{};
            std::array<std::uint8_t, VK_UUID_SIZE> driverUuid{};
        };

        static_assert(sizeof(PipelineCacheBlobHeader) == 72);
        static_assert(sizeof(VulkanPipelineManager::PipelineCacheEntry) == 24);

        struct DecodedPipelineCache final
        {
            std::span<const std::byte> nativeData;
            std::set<VulkanPipelineManager::PipelineCacheEntry> entries;
        };

        [[nodiscard]] Result<DecodedPipelineCache> decodePipelineCache(
            std::span<const std::byte> data,
            std::span<const std::uint8_t, VK_UUID_SIZE> pipelineCacheUuid,
            std::span<const std::uint8_t, VK_UUID_SIZE> deviceUuid,
            std::span<const std::uint8_t, VK_UUID_SIZE> driverUuid)
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
            if (!std::equal(
                    header.pipelineCacheUuid.begin(), header.pipelineCacheUuid.end(), pipelineCacheUuid.begin()) ||
                !std::equal(header.deviceUuid.begin(), header.deviceUuid.end(), deviceUuid.begin()) ||
                !std::equal(header.driverUuid.begin(), header.driverUuid.end(), driverUuid.begin()))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline cache blob does not match this adapter, driver, or pipeline-cache UUID.",
                });
            }

            const auto remaining = data.size() - sizeof(header);
            if (header.entryCount > remaining / sizeof(VulkanPipelineManager::PipelineCacheEntry))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline cache blob entry table exceeds the supplied data.",
                });
            }
            const auto entriesSize =
                static_cast<std::size_t>(header.entryCount) * sizeof(VulkanPipelineManager::PipelineCacheEntry);
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
                VulkanPipelineManager::PipelineCacheEntry entry;
                std::memcpy(&entry, data.data() + sizeof(header) + index * sizeof(entry), sizeof(entry));
                if (!entry.key.isValid() || entry.reserved != 0 ||
                    (entry.kind != VulkanPipelineManager::PipelineCacheEntryKind::compute &&
                     entry.kind != VulkanPipelineManager::PipelineCacheEntryKind::graphics) ||
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

    } // namespace

    VulkanPipelineManager::CacheKeepAlive::~CacheKeepAlive()
    {
        if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroyPipelineCache != nullptr &&
            cache != VK_NULL_HANDLE)
        {
            table->vkDestroyPipelineCache(device, cache, nullptr);
        }
    }

    VulkanPipelineManager::PipelineKeepAlive::~PipelineKeepAlive()
    {
        if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroyPipeline != nullptr &&
            pipeline != VK_NULL_HANDLE)
        {
            table->vkDestroyPipeline(device, pipeline, nullptr);
        }
    }

    VulkanPipelineManager::~VulkanPipelineManager()
    {
        shutdown();
    }

    void VulkanPipelineManager::initialize(VkDevice device,
                                           const VolkDeviceTable* table,
                                           VkPhysicalDevice physical,
                                           const DeviceCapabilities& capabilities,
                                           VulkanShaderManager* shaders,
                                           VulkanBindingManager* bindings,
                                           const VulkanInstanceTable* instanceTable,
                                           bool depthBiasClamp,
                                           VulkanFormatPropertiesCache* sharedFormatCache,
                                           bool maintenance5,
                                           VulkanMultiviewStageSupport multiviewStages,
                                           std::uint32_t minVertexInputBindingStrideAlignment,
                                           bool isolateDynamicBufferShaders) noexcept
    {
        stopWorkers();
        device_ = device;
        table_ = table;
        instanceTable_ = instanceTable;
        physical_ = physical;
        capabilities_ = capabilities;
        depthBiasClampSupported_ = depthBiasClamp;
        maintenance5_ = maintenance5;
        multiviewStages_ = multiviewStages;
        minVertexInputBindingStrideAlignment_ = (std::max)(minVertexInputBindingStrideAlignment, 1U);
        isolateDynamicBufferShaders_ = isolateDynamicBufferShaders;
        shaders_ = shaders;
        bindings_ = bindings;
        formatCache_.reset();
        activeFormatCache_ = sharedFormatCache != nullptr ? sharedFormatCache : &formatCache_;
        maxComputeWorkgroupSubgroups_ = 0;
        if (physical != VK_NULL_HANDLE && instanceTable != nullptr)
        {
            VkPhysicalDeviceVulkan13Properties v13{};
            v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
            VkPhysicalDeviceIDProperties id{};
            id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            id.pNext = &v13;

            VkPhysicalDeviceProperties2 properties2{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                .pNext = &id,
                .properties = {},
            };
            if (instanceTable->vkGetPhysicalDeviceProperties2 != nullptr)
            {
                instanceTable->vkGetPhysicalDeviceProperties2(physical, &properties2);
            }
            std::memcpy(pipelineCacheUuid_.data(), properties2.properties.pipelineCacheUUID, VK_UUID_SIZE);
            std::memcpy(deviceUuid_.data(), id.deviceUUID, VK_UUID_SIZE);
            std::memcpy(driverUuid_.data(), id.driverUUID, VK_UUID_SIZE);
            maxComputeWorkgroupSubgroups_ = v13.maxComputeWorkgroupSubgroups;
        }
        if (device_ != VK_NULL_HANDLE && table_ != nullptr)
        {
            startWorkers();
        }
    }

    void VulkanPipelineManager::shutdown() noexcept
    {
        stopWorkers();
        detail::HandleTable<std::shared_ptr<CompilationRecord>, PipelineCompilationHandle> compilations{};
        detail::HandleTable<PipelineRecord, PipelineHandle> pipelines{};
        detail::HandleTable<std::shared_ptr<RayTracingLibraryRecord>, RayTracingPipelineLibraryHandle>
            rayTracingLibraries{};
        detail::HandleTable<std::shared_ptr<CacheRecord>, PipelineCacheHandle> caches{};
        {
            const std::scoped_lock lock(mutex_);
            std::swap(compilations, compilations_);
            std::swap(pipelines, pipelines_);
            std::swap(rayTracingLibraries, rayTracingLibraries_);
            std::swap(caches, caches_);
        }
        formatCache_.reset();
        activeFormatCache_ = &formatCache_;
        shaders_ = nullptr;
        bindings_ = nullptr;
        maintenance5_ = false;
        isolateDynamicBufferShaders_ = false;
        physical_ = VK_NULL_HANDLE;
        instanceTable_ = nullptr;
        table_ = nullptr;
        device_ = VK_NULL_HANDLE;
        pipelineCacheUuid_ = {};
        deviceUuid_ = {};
        driverUuid_ = {};
    }

    void VulkanPipelineManager::startWorkers() noexcept
    {
        try
        {
            {
                const std::scoped_lock lock(workerMutex_);
                stopWorkers_ = false;
            }
            const auto hardwareThreads = std::thread::hardware_concurrency();
            const std::size_t workerCount =
                std::clamp<std::size_t>(hardwareThreads == 0U ? 2U : hardwareThreads, 2U, 4U);
            workers_.reserve(workerCount);
            for (std::size_t index = 0; index < workerCount; ++index)
            {
                workers_.emplace_back(&VulkanPipelineManager::workerLoop, this);
            }
        }
        catch (...)
        {
        }
    }

    void VulkanPipelineManager::stopWorkers() noexcept
    {
        std::vector<std::thread> workers;
        std::deque<CompilationJob> cancelledJobs;
        {
            const std::scoped_lock lock(workerMutex_);
            stopWorkers_ = true;
            cancelledJobs = std::move(workerQueue_);
            workers = std::move(workers_);
        }
        workerCondition_.notify_all();
        for (auto& job : cancelledJobs)
        {
            if (job.record != nullptr)
            {
                {
                    const std::scoped_lock recordLock(job.record->mutex);
                    job.record->abandoned = true;
                }
                job.record->condition.notify_all();
            }
        }
        for (auto& worker : workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    void VulkanPipelineManager::workerLoop() noexcept
    {
        for (;;)
        {
            std::optional<CompilationJob> job;
            {
                std::unique_lock lock(workerMutex_);
                workerCondition_.wait(lock, [this]() { return stopWorkers_ || !workerQueue_.empty(); });
                if (stopWorkers_)
                {
                    return;
                }
                job = std::move(workerQueue_.front());
                workerQueue_.pop_front();
            }

            if (!job.has_value() || job->record == nullptr)
            {
                continue;
            }

            bool alreadyAbandoned = false;
            {
                const std::scoped_lock recordLock(job->record->mutex);
                alreadyAbandoned = job->record->abandoned;
            }
            if (alreadyAbandoned)
            {
                job->record->condition.notify_all();
                continue;
            }

            Result<PipelineHandle> compiled = std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The pipeline compilation job had no payload.",
            });
            try
            {
                if (auto* compute = std::get_if<PreparedComputeCompilation>(&job->payload))
                {
                    compiled = compileComputePipeline(std::move(*compute));
                }
                else if (auto* graphics = std::get_if<PreparedGraphicsCompilation>(&job->payload))
                {
                    compiled = compileGraphicsPipeline(std::move(*graphics));
                }
            }
            catch (const std::bad_alloc&)
            {
                compiled = std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate storage during asynchronous pipeline compilation.",
                });
            }

            std::optional<PipelineHandle> abandonedPipeline;
            {
                const std::scoped_lock recordLock(job->record->mutex);
                if (job->record->abandoned)
                {
                    if (compiled.has_value())
                    {
                        abandonedPipeline = *compiled;
                    }
                }
                else
                {
                    job->record->state =
                        compiled.has_value() ? PipelineCompilationState::succeeded : PipelineCompilationState::failed;
                    job->record->result = std::move(compiled);
                }
            }
            job->record->condition.notify_all();
            if (abandonedPipeline.has_value())
            {
                static_cast<void>(destroyPipeline(*abandonedPipeline));
            }
        }
    }

    Result<VulkanPipelineManager::ResolvedCache> VulkanPipelineManager::resolvePreparedCache(
        PipelineCachePolicy policy,
        bool hasCacheHandle,
        std::shared_ptr<CacheRecord> cacheRecord,
        PipelineCacheKey key,
        PipelineCacheEntryKind kind) const
    {
        if (!isValidPipelineCachePolicy(policy))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A pipeline contains an invalid pipeline-cache policy.",
            });
        }
        if (!hasCacheHandle && !key.isValid())
        {
            if (policy == PipelineCachePolicy::requireHit)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A cache-only pipeline requires a live cache and non-zero key.",
                });
            }
            return ResolvedCache{.policy = policy, .kind = kind};
        }
        if (!hasCacheHandle || !key.isValid())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Pipeline caching requires both a live cache handle and a non-zero cache key.",
            });
        }
        if (cacheRecord == nullptr || cacheRecord->keepAlive == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline cache handle is invalid or stale.",
            });
        }
        return ResolvedCache{
            .record = std::move(cacheRecord),
            .key = key,
            .policy = policy,
            .kind = kind,
        };
    }

    Status VulkanPipelineManager::consumeCacheEntry(ResolvedCache& cache)
    {
        if (cache.record == nullptr)
        {
            return {};
        }
        const PipelineCacheEntry entry{.key = cache.key, .kind = cache.kind};
        if (!cache.record->entries.contains(entry) && cache.policy == PipelineCachePolicy::requireHit)
        {
            return std::unexpected(Error{
                ErrorCode::pipelineCacheMiss,
                0,
                "The pipeline is absent from the requested pipeline cache.",
            });
        }
        return {};
    }

    Status VulkanPipelineManager::storeCacheEntry(const ResolvedCache& cache)
    {
        if (cache.record == nullptr || cache.policy != PipelineCachePolicy::populate)
        {
            return {};
        }
        cache.record->entries.insert(PipelineCacheEntry{.key = cache.key, .kind = cache.kind});
        return {};
    }

    Result<PipelineCacheHandle> VulkanPipelineManager::createPipelineCache(const PipelineCacheDesc& desc)
    {
        if (table_ == nullptr || device_ == VK_NULL_HANDLE || table_->vkCreatePipelineCache == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The Vulkan pipeline manager is not initialized.",
            });
        }
        auto decoded = decodePipelineCache(desc.initialData, pipelineCacheUuid_, deviceUuid_, driverUuid_);
        if (!decoded)
        {
            return std::unexpected(std::move(decoded.error()));
        }

        auto keepAlive = std::make_shared<CacheKeepAlive>();
        keepAlive->device = device_;
        keepAlive->table = table_;
        const auto createInfo = VkPipelineCacheCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = {},
            .flags = {},
            .initialDataSize = decoded->nativeData.size(),
            .pInitialData = decoded->nativeData.empty() ? nullptr : decoded->nativeData.data(),
        };
        const auto created = table_->vkCreatePipelineCache(device_, &createInfo, nullptr, &keepAlive->cache);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreatePipelineCache"));
        }
        if (!desc.label.empty())
        {
            const std::string label(desc.label);
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_PIPELINE_CACHE,
                                reinterpret_cast<std::uint64_t>(keepAlive->cache),
                                label.c_str());
        }

        auto record = std::make_shared<CacheRecord>();
        record->keepAlive = std::move(keepAlive);
        record->entries = std::move(decoded->entries);
        record->label = desc.label;
        const std::scoped_lock lock(mutex_);
        return caches_.emplace(std::move(record));
    }

    Status VulkanPipelineManager::destroyPipelineCache(PipelineCacheHandle pipelineCache)
    {
        const std::scoped_lock lock(mutex_);
        if (!caches_.erase(pipelineCache))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline cache handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<std::vector<std::byte>> VulkanPipelineManager::pipelineCacheData(PipelineCacheHandle pipelineCache)
    {
        std::shared_ptr<CacheRecord> cache;
        {
            const std::scoped_lock lock(mutex_);
            const auto* found = caches_.find(pipelineCache);
            if (found == nullptr || *found == nullptr || (*found)->keepAlive == nullptr)
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
        std::size_t nativeSize = 0;
        const auto queried = table_->vkGetPipelineCacheData(device_, cache->keepAlive->cache, &nativeSize, nullptr);
        if (queried != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(ErrorCode::nativeFailure, queried, "vkGetPipelineCacheData"));
        }
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
            .entryCount = static_cast<std::uint32_t>(cache->entries.size()),
            .nativeDataSize = nativeSize,
            .pipelineCacheUuid = pipelineCacheUuid_,
            .deviceUuid = deviceUuid_,
            .driverUuid = driverUuid_,
        };
        std::memcpy(data.data(), &header, sizeof(header));
        auto entryOffset = sizeof(header);
        for (const auto& entry : cache->entries)
        {
            std::memcpy(data.data() + entryOffset, &entry, sizeof(entry));
            entryOffset += sizeof(entry);
        }
        if (nativeSize != 0)
        {
            const auto serialized = table_->vkGetPipelineCacheData(
                device_, cache->keepAlive->cache, &nativeSize, data.data() + metadataSize);
            if (serialized != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(ErrorCode::nativeFailure, serialized, "vkGetPipelineCacheData"));
            }
            if (nativeSize != header.nativeDataSize)
            {
                data.resize(metadataSize + nativeSize);
                auto* rewritten = reinterpret_cast<PipelineCacheBlobHeader*>(data.data());
                rewritten->nativeDataSize = nativeSize;
            }
        }
        return data;
    }

    Result<PipelineHandle> VulkanPipelineManager::publishPipeline(PipelineRecord record)
    {
        const std::scoped_lock lock(mutex_);
        return pipelines_.emplace(std::move(record));
    }

    Result<PipelineCompilationHandle> VulkanPipelineManager::enqueueCompilation(
        std::variant<PreparedComputeCompilation, PreparedGraphicsCompilation>&& payload)
    {
        if (table_ == nullptr || device_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The Vulkan pipeline manager is not initialized.",
            });
        }

        std::shared_ptr<CompilationRecord> record;
        try
        {
            record = std::make_shared<CompilationRecord>();
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate asynchronous pipeline compilation state.",
            });
        }

        Result<PipelineCompilationHandle> handle = std::unexpected(Error{
            ErrorCode::outOfMemory,
            0,
            "Failed to allocate asynchronous pipeline compilation handle.",
        });
        {
            const std::scoped_lock lock(mutex_);
            handle = compilations_.emplace(record);
        }
        if (!handle.has_value())
        {
            return handle;
        }

        try
        {
            const std::scoped_lock workerLock(workerMutex_);
            if (stopWorkers_ || workers_.empty())
            {
                const std::scoped_lock lock(mutex_);
                static_cast<void>(compilations_.erase(*handle));
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The Vulkan asynchronous pipeline worker pool is unavailable.",
                });
            }
            workerQueue_.push_back(CompilationJob{
                .record = std::move(record),
                .payload = std::move(payload),
            });
        }
        catch (const std::bad_alloc&)
        {
            const std::scoped_lock lock(mutex_);
            static_cast<void>(compilations_.erase(*handle));
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to enqueue asynchronous pipeline compilation job.",
            });
        }
        workerCondition_.notify_one();
        return handle;
    }

    Result<std::shared_ptr<VulkanPipelineManager::CacheRecord>> VulkanPipelineManager::findCacheRecord(
        PipelineCacheHandle pipelineCache) const
    {
        const std::shared_lock lock(mutex_);
        const auto* found = caches_.find(pipelineCache);
        if (found == nullptr || *found == nullptr || (*found)->keepAlive == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline cache handle is invalid or stale.",
            });
        }
        return *found;
    }

    Result<VulkanPipelineManager::PreparedComputeCompilation> VulkanPipelineManager::prepareComputeCompilation(
        const ComputePipelineDesc& desc) const
    {
        PreparedComputeCompilation prepared{
            .hasLayout = desc.layout.isValid(),
            .hasComputeShader = desc.computeShader.isValid(),
            .hasCache = desc.cache.isValid(),
            .cacheKey = desc.cacheKey,
            .cachePolicy = desc.cachePolicy,
            .requiredSubgroupSize = desc.requiredSubgroupSize,
        };
        if (desc.requiredSubgroupSize != 0 && shaders_ != nullptr && desc.computeShader.isValid())
        {
            if (const auto shaderInterface = shaders_->shaderModuleInterface(desc.computeShader);
                shaderInterface && shaderInterface->workgroupSize.has_value())
            {
                const auto& size = *shaderInterface->workgroupSize;
                prepared.workgroupInvocations = size.x * size.y * size.z;
            }
        }
        if (desc.cache.isValid())
        {
            auto cacheRecord = findCacheRecord(desc.cache);
            if (!cacheRecord)
            {
                return std::unexpected(std::move(cacheRecord.error()));
            }
            prepared.cacheRecord = std::move(*cacheRecord);
        }
        if (desc.layout.isValid())
        {
            if (bindings_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The Vulkan pipeline manager is not initialized.",
                });
            }
            auto layout = bindings_->nativePipelineLayout(desc.layout);
            if (!layout)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A compute pipeline references an invalid or stale layout or shader module.",
                });
            }
            prepared.layout = std::move(*layout);
        }
        if (desc.computeShader.isValid())
        {
            if (shaders_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The Vulkan pipeline manager is not initialized.",
                });
            }
            auto shader = shaders_->nativeShaderModule(desc.computeShader);
            if (!shader)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A compute pipeline references an invalid or stale layout or shader module.",
                });
            }
            prepared.computeShader = std::move(*shader);
        }
        try
        {
            prepared.label = std::string(desc.label);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate pipeline label storage.",
            });
        }
        return prepared;
    }

    namespace
    {

        [[nodiscard]] Status validateComputeShader(const VulkanShaderManager::NativeShaderModule& shader,
                                                   const VulkanNativePipelineLayout& layout)
        {
            if (shader.stage != ShaderStage::compute || shader.module == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A compute pipeline requires a compute shader module.",
                });
            }
            return validateShaderInterface(shader, layout);
        }

        [[nodiscard]] VkPipelineShaderStageCreateInfo makeShaderStageCreateInfo(
            VkShaderStageFlagBits stage,
            const VulkanShaderManager::NativeShaderModule& shader,
            const ShaderSpecialization& specialization)
        {
            return VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .stage = stage,
                .module = shader.module,
                .pName = shader.entryPoint.c_str(),
                .pSpecializationInfo = specialization.pointer(),
            };
        }

        struct PipelineSetLayouts final
        {
            std::vector<std::uint8_t> required;
            std::vector<std::uint32_t> dynamicCounts;
            std::vector<VkDescriptorSetLayout> layouts;
        };

        [[nodiscard]] PipelineSetLayouts collectSetLayouts(const VulkanNativePipelineLayout& layout)
        {
            PipelineSetLayouts sets;
            sets.required.reserve(layout.sets.size());
            sets.dynamicCounts.reserve(layout.sets.size());
            sets.layouts.reserve(layout.sets.size());
            for (const auto& set : layout.sets)
            {
                sets.required.push_back(set.entries.empty() ? std::uint8_t{0} : std::uint8_t{1});
                sets.dynamicCounts.push_back(set.dynamicBufferCount);
                sets.layouts.push_back(set.setLayout);
            }
            return sets;
        }

        constexpr std::size_t maxIsolatedShaderStages = 7;

        class IsolatedShaderModules final
        {
        public:
            IsolatedShaderModules(VkDevice device, const VolkDeviceTable* table) noexcept
                : device_(device), table_(table)
            {
            }

            ~IsolatedShaderModules()
            {
                for (std::size_t index = 0; index < count_; ++index)
                {
                    table_->vkDestroyShaderModule(device_, modules_[index], nullptr);
                }
            }

            IsolatedShaderModules(const IsolatedShaderModules&) = delete;
            IsolatedShaderModules& operator=(const IsolatedShaderModules&) = delete;
            IsolatedShaderModules(IsolatedShaderModules&&) = delete;
            IsolatedShaderModules& operator=(IsolatedShaderModules&&) = delete;

            [[nodiscard]] Status replace(VulkanShaderManager::NativeShaderModule& shader)
            {
                if (shader.spirv.empty() || count_ == modules_.size())
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "The shader module does not retain the SPIR-V for a pipeline-private shader module.",
                    });
                }
                const VkShaderModuleCreateInfo createInfo{
                    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    .pNext = {},
                    .flags = {},
                    .codeSize = shader.spirv.size_bytes(),
                    .pCode = shader.spirv.data(),
                };
                VkShaderModule module = VK_NULL_HANDLE;
                const auto created = table_->vkCreateShaderModule(device_, &createInfo, nullptr, &module);
                if (created != VK_SUCCESS)
                {
                    return std::unexpected(
                        makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateShaderModule"));
                }
                modules_[count_++] = module;
                shader.module = module;
                return {};
            }

        private:
            VkDevice device_ = VK_NULL_HANDLE;
            const VolkDeviceTable* table_ = nullptr;
            std::array<VkShaderModule, maxIsolatedShaderStages> modules_{};
            std::size_t count_ = 0;
        };

#ifdef RACHIS_TESTING
        std::mutex capturedPipelineMutex;
        std::vector<VulkanCapturedPipelineCreation>* capturedPipelines = nullptr;

        void capturePipelineCreation(std::span<const VkPipelineShaderStageCreateInfo> stages, VkPipelineCache cache)
        {
            const std::scoped_lock lock(capturedPipelineMutex);
            if (capturedPipelines == nullptr)
            {
                return;
            }
            VulkanCapturedPipelineCreation captured{.shaderModules = {}, .cache = cache};
            for (const auto& stage : stages)
            {
                captured.shaderModules.push_back(stage.module);
            }
            capturedPipelines->push_back(std::move(captured));
        }
#endif

    } // namespace

#ifdef RACHIS_TESTING
    void captureVulkanPipelineCreations(std::vector<VulkanCapturedPipelineCreation>* sink) noexcept
    {
        const std::scoped_lock lock(capturedPipelineMutex);
        capturedPipelines = sink;
    }
#endif

    bool VulkanPipelineManager::isolatesShaderModules(const VulkanNativePipelineLayout& layout) const noexcept
    {
        return isolateDynamicBufferShaders_ &&
               std::ranges::any_of(layout.sets, [](const auto& set) { return set.dynamicBufferCount != 0; });
    }

    struct VulkanPipelineManager::LockedPipelineCache final
    {
        std::unique_lock<std::mutex> lock;
        VkPipelineCache native = VK_NULL_HANDLE;
    };

    Result<VulkanPipelineManager::LockedPipelineCache> VulkanPipelineManager::lockCacheAndConsumeEntry(
        ResolvedCache& cache)
    {
        LockedPipelineCache locked;
        if (cache.record == nullptr)
        {
            return locked;
        }
        locked.lock = std::unique_lock(cache.record->mutex);
        if (auto consumed = consumeCacheEntry(cache); !consumed)
        {
            return std::unexpected(std::move(consumed.error()));
        }
        locked.native = cache.record->keepAlive->cache;
        return locked;
    }

    Status VulkanPipelineManager::validateRequiredSubgroupSize(const PreparedComputeCompilation& prepared) const
    {
        const auto size = prepared.requiredSubgroupSize;
        if (size == 0)
        {
            return {};
        }
        if (!hasAny(capabilities_.subgroup.requiredSizeStages, ShaderStage::compute) ||
            size < capabilities_.subgroup.minSize || size > capabilities_.subgroup.maxSize)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "This Vulkan device cannot run compute pipelines with the required subgroup size.",
            });
        }
        if (maxComputeWorkgroupSubgroups_ != 0 &&
            prepared.workgroupInvocations > static_cast<std::uint64_t>(maxComputeWorkgroupSubgroups_) * size)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The compute shader's workgroup needs more subgroups of the required size than this Vulkan device "
                "allows in one workgroup.",
            });
        }
        return {};
    }

    Result<VkPipeline> VulkanPipelineManager::createNativeComputePipeline(
        const VulkanShaderManager::NativeShaderModule& shader,
        VkPipelineLayout pipelineLayout,
        VkPipelineCache nativeCache,
        std::uint32_t requiredSubgroupSize) const
    {
        const auto specialization = makeSpecialization(shader.specializationConstants);
        const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo requiredSize{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
            .pNext = {},
            .requiredSubgroupSize = requiredSubgroupSize,
        };
        auto stage = makeShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, shader, specialization);
        if (requiredSubgroupSize != 0)
        {
            stage.pNext = &requiredSize;
            stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        }
        const auto createInfo = VkComputePipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = {},
            .flags = {},
            .stage = stage,
            .layout = pipelineLayout,
            .basePipelineHandle = {},
            .basePipelineIndex = {},
        };

#ifdef RACHIS_TESTING
        capturePipelineCreation(std::span(&createInfo.stage, 1), nativeCache);
#endif
        VkPipeline pipeline = VK_NULL_HANDLE;
        const auto created = table_->vkCreateComputePipelines(device_, nativeCache, 1, &createInfo, nullptr, &pipeline);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateComputePipelines"));
        }
        return pipeline;
    }

    std::shared_ptr<VulkanPipelineManager::PipelineKeepAlive> VulkanPipelineManager::makePipelineKeepAlive(
        VkPipeline pipeline, const std::shared_ptr<void>& layoutOwner) const
    {
        auto keepAlive = std::make_shared<PipelineKeepAlive>();
        keepAlive->device = device_;
        keepAlive->table = table_;
        keepAlive->pipeline = pipeline;
        keepAlive->layoutOwner = layoutOwner;
        return keepAlive;
    }

    Result<std::shared_ptr<VulkanPipelineManager::PipelineKeepAlive>> VulkanPipelineManager::adoptComputePipeline(
        VkPipeline pipeline, const std::shared_ptr<void>& layoutOwner, const std::shared_ptr<void>& shaderOwner) const
    {
        std::shared_ptr<PipelineKeepAlive> keepAlive;
        try
        {
            keepAlive = makePipelineKeepAlive(pipeline, layoutOwner);
            keepAlive->shaderOwners.push_back(shaderOwner);
        }
        catch (const std::bad_alloc&)
        {
            if (keepAlive == nullptr)
            {
                table_->vkDestroyPipeline(device_, pipeline, nullptr);
            }
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate native compute pipeline state.",
            });
        }
        return keepAlive;
    }

    void VulkanPipelineManager::labelPipeline(VkPipeline pipeline, const std::string& label) const
    {
        if (label.empty())
        {
            return;
        }
        setVulkanObjectName(
            instanceTable_, device_, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<std::uint64_t>(pipeline), label.c_str());
    }

    Result<PipelineHandle> VulkanPipelineManager::publishComputePipeline(
        VulkanNativePipelineLayout&& layout, const std::shared_ptr<PipelineKeepAlive>& keepAlive)
    {
        try
        {
            auto sets = collectSetLayouts(layout);
            auto native = std::make_shared<VulkanNativePipeline>(VulkanNativePipeline{
                .owner = keepAlive,
                .layoutOwner = layout.owner,
                .pipeline = keepAlive->pipeline,
                .layout = layout.pipelineLayout,
                .compute = true,
                .rasterizationEnabled = false,
                .colorFormats = {},
                .vertexStrides = {},
                .vertexAlignments = {},
                .pushConstantRanges = std::move(layout.pushConstantRanges),
                .nativePushConstantRanges = std::move(layout.nativePushConstantRanges),
                .setRequired = std::move(sets.required),
                .setDynamicCounts = std::move(sets.dynamicCounts),
                .setLayouts = std::move(sets.layouts),
            });
            return publishPipeline(PipelineRecord{
                .native = std::move(native),
            });
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate immutable compute pipeline record.",
            });
        }
    }

    Result<PipelineHandle> VulkanPipelineManager::compileComputePipeline(PreparedComputeCompilation&& prepared)
    {
        auto cache = resolvePreparedCache(prepared.cachePolicy,
                                          prepared.hasCache,
                                          std::move(prepared.cacheRecord),
                                          prepared.cacheKey,
                                          PipelineCacheEntryKind::compute);
        if (!cache)
        {
            return std::unexpected(std::move(cache.error()));
        }
        if (shaders_ == nullptr || bindings_ == nullptr || table_ == nullptr || device_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The Vulkan pipeline manager is not initialized.",
            });
        }

        if (!prepared.layout.has_value() || !prepared.computeShader.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A compute pipeline references an invalid or stale layout or shader module.",
            });
        }
        auto layout = std::move(*prepared.layout);
        auto shader = std::move(*prepared.computeShader);
        if (auto valid = validateComputeShader(shader, layout); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        if (auto valid = validateRequiredSubgroupSize(prepared); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }

        IsolatedShaderModules isolated(device_, table_);
        const bool isolates = isolatesShaderModules(layout);
        if (isolates)
        {
            if (auto replaced = isolated.replace(shader); !replaced)
            {
                return std::unexpected(std::move(replaced.error()));
            }
        }

        auto lockedCache = lockCacheAndConsumeEntry(*cache);
        if (!lockedCache)
        {
            return std::unexpected(std::move(lockedCache.error()));
        }

        auto pipeline = createNativeComputePipeline(shader,
                                                    layout.pipelineLayout,
                                                    isolates ? VK_NULL_HANDLE : lockedCache->native,
                                                    prepared.requiredSubgroupSize);
        if (!pipeline)
        {
            return std::unexpected(std::move(pipeline.error()));
        }
        static_cast<void>(storeCacheEntry(*cache));

        auto keepAlive = adoptComputePipeline(*pipeline, layout.owner, shader.owner);
        if (!keepAlive)
        {
            return std::unexpected(std::move(keepAlive.error()));
        }
        labelPipeline(*pipeline, prepared.label);
        return publishComputePipeline(std::move(layout), *keepAlive);
    }

    Result<PipelineHandle> VulkanPipelineManager::createComputePipeline(const ComputePipelineDesc& desc)
    {
        auto prepared = prepareComputeCompilation(desc);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        return compileComputePipeline(std::move(*prepared));
    }

    VulkanPipelineManager::OwnedGraphicsPipelineDesc VulkanPipelineManager::copyFixedSizeState(
        const GraphicsPipelineDesc& desc)
    {
        return OwnedGraphicsPipelineDesc{
            .hasLayout = desc.layout.isValid(),
            .hasVertexShader = desc.vertexShader.isValid(),
            .hasPixelShader = desc.pixelShader.isValid(),
            .hasGeometryShader = desc.geometryShader.isValid(),
            .hasHullShader = desc.hullShader.isValid(),
            .hasDomainShader = desc.domainShader.isValid(),
            .hasTaskShader = desc.taskShader.isValid(),
            .hasMeshShader = desc.meshShader.isValid(),
            .hasCache = desc.cache.isValid(),
            .topology = desc.topology,
            .patchControlPoints = desc.patchControlPoints,
            .stripIndexFormat = desc.stripIndexFormat,
            .frontFace = desc.frontFace,
            .cullMode = desc.cullMode,
            .fillMode = desc.fillMode,
            .depthBias = desc.depthBias,
            .depthBiasClamp = desc.depthBiasClamp,
            .depthBiasSlopeScale = desc.depthBiasSlopeScale,
            .dynamicDepthBiasEnabled = desc.dynamicDepthBiasEnabled,
            .depthClipEnabled = desc.depthClipEnabled,
            .depthStencil = desc.depthStencil,
            .sampleCount = desc.sampleCount,
            .sampleMask = desc.sampleMask,
            .alphaToCoverageEnabled = desc.alphaToCoverageEnabled,
            .logicOperation = desc.logicOperation,
            .conservativeRasterization = desc.conservativeRasterization,
            .clipDistanceCount = desc.clipDistanceCount,
            .cullDistanceCount = desc.cullDistanceCount,
            .viewMask = desc.viewMask,
            .rasterizedStream = desc.rasterizedStream,
            .cacheKey = desc.cacheKey,
            .cachePolicy = desc.cachePolicy,
        };
    }

    namespace
    {

        [[nodiscard]] Result<std::optional<VulkanShaderManager::NativeShaderModule>> resolveGraphicsShader(
            const VulkanShaderManager* shaders, ShaderModuleHandle handle)
        {
            if (!handle.isValid())
            {
                return std::nullopt;
            }
            if (shaders == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The Vulkan pipeline manager is not initialized.",
                });
            }
            auto found = shaders->nativeShaderModule(handle);
            if (!found)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics pipeline references an invalid or stale layout or shader module.",
                });
            }
            return std::move(*found);
        }

    } // namespace

    Status VulkanPipelineManager::resolveGraphicsShaders(const GraphicsPipelineDesc& desc,
                                                         PreparedGraphicsCompilation& prepared) const
    {
        using ResolvedShader = std::optional<VulkanShaderManager::NativeShaderModule>;
        const std::array<std::pair<ShaderModuleHandle, ResolvedShader*>, 7> stages{{
            {desc.vertexShader, &prepared.vertexShader},
            {desc.pixelShader, &prepared.pixelShader},
            {desc.geometryShader, &prepared.geometryShader},
            {desc.hullShader, &prepared.hullShader},
            {desc.domainShader, &prepared.domainShader},
            {desc.taskShader, &prepared.taskShader},
            {desc.meshShader, &prepared.meshShader},
        }};
        for (const auto& [handle, destination] : stages)
        {
            auto resolved = resolveGraphicsShader(shaders_, handle);
            if (!resolved)
            {
                return std::unexpected(std::move(resolved.error()));
            }
            *destination = std::move(*resolved);
        }
        return {};
    }

    Status VulkanPipelineManager::copyVariableSizeState(const GraphicsPipelineDesc& desc,
                                                        OwnedGraphicsPipelineDesc& owned)
    {
        try
        {
            owned.vertexBuffers.reserve(desc.vertexBuffers.size());
            for (const auto& buffer : desc.vertexBuffers)
            {
                owned.vertexBuffers.push_back(OwnedVertexBufferLayout{
                    .stride = buffer.stride,
                    .stepMode = buffer.stepMode,
                    .instanceStepRate = buffer.instanceStepRate,
                    .attributes = std::vector<VertexAttribute>(buffer.attributes.begin(), buffer.attributes.end()),
                });
            }
            owned.colorTargets.assign(desc.colorTargets.begin(), desc.colorTargets.end());
            owned.streamOutputEntries.assign(desc.streamOutputEntries.begin(), desc.streamOutputEntries.end());
            owned.streamOutputBufferStrides.assign(desc.streamOutputBufferStrides.begin(),
                                                   desc.streamOutputBufferStrides.end());
            owned.label = std::string(desc.label);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate graphics pipeline compilation state.",
            });
        }
        return {};
    }

    Result<VulkanPipelineManager::PreparedGraphicsCompilation> VulkanPipelineManager::prepareGraphicsCompilation(
        const GraphicsPipelineDesc& desc) const
    {
        PreparedGraphicsCompilation prepared{};
        prepared.desc = copyFixedSizeState(desc);

        if (desc.cache.isValid())
        {
            auto cacheRecord = findCacheRecord(desc.cache);
            if (!cacheRecord)
            {
                return std::unexpected(std::move(cacheRecord.error()));
            }
            prepared.cacheRecord = std::move(*cacheRecord);
        }
        if (desc.layout.isValid())
        {
            if (bindings_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The Vulkan pipeline manager is not initialized.",
                });
            }
            auto layout = bindings_->nativePipelineLayout(desc.layout);
            if (!layout)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics pipeline references an invalid or stale layout or shader module.",
                });
            }
            prepared.layout = std::move(*layout);
        }
        if (auto resolved = resolveGraphicsShaders(desc, prepared); !resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        if (auto copied = copyVariableSizeState(desc, prepared.desc); !copied)
        {
            return std::unexpected(std::move(copied.error()));
        }
        return prepared;
    }

    namespace
    {

        [[nodiscard]] constexpr bool hasStaticDepthBias(std::int32_t depthBias,
                                                        float depthBiasClamp,
                                                        float depthBiasSlopeScale) noexcept
        {
            return depthBias != 0 || depthBiasClamp != 0.0F || depthBiasSlopeScale != 0.0F;
        }

    } // namespace

    Status VulkanPipelineManager::validateGraphicsPipelineState(const OwnedGraphicsPipelineDesc& desc) const
    {
        const bool withinAttachmentAndVertexBufferLimits =
            desc.colorTargets.size() <= maxColorAttachments && desc.vertexBuffers.size() <= maxVertexBuffers;
        const bool hasOutput = !desc.streamOutputEntries.empty() || !desc.colorTargets.empty() ||
                               desc.depthStencil.format != Format::unknown;
        const bool validSampleCount =
            desc.sampleCount == 1 || desc.sampleCount == 2 || desc.sampleCount == 4 || desc.sampleCount == 8;
        const auto activeSampleMask = validSampleCount ? (1U << desc.sampleCount) - 1U : 0U;
        const bool validMultisampleState = validSampleCount && (desc.sampleMask & activeSampleMask) != 0 &&
                                           !(desc.alphaToCoverageEnabled && desc.sampleCount == 1);
        const bool withinClipAndCullDistanceLimits =
            desc.clipDistanceCount <= capabilities_.limits.maxClipDistances &&
            desc.cullDistanceCount <= capabilities_.limits.maxCullDistances &&
            desc.clipDistanceCount + desc.cullDistanceCount <= capabilities_.limits.maxCombinedClipAndCullDistances;
        const bool usesStaticDepthBias =
            hasStaticDepthBias(desc.depthBias, desc.depthBiasClamp, desc.depthBiasSlopeScale);
        const bool validDepthBias = std::isfinite(desc.depthBiasClamp) && std::isfinite(desc.depthBiasSlopeScale) &&
                                    !(desc.dynamicDepthBiasEnabled && usesStaticDepthBias);
        const bool depthBoundsHaveDepthTarget =
            !desc.depthStencil.depthBoundsTestEnabled || desc.depthStencil.format != Format::unknown;
        const bool validLogicOperation =
            !desc.logicOperation.has_value() || isValidLogicOperation(*desc.logicOperation);
        if (!withinAttachmentAndVertexBufferLimits || !hasOutput || !validMultisampleState ||
            !isValidFillMode(desc.fillMode) || !isValidConservativeRasterizationMode(desc.conservativeRasterization) ||
            !withinClipAndCullDistanceLimits || !validDepthBias || !depthBoundsHaveDepthTarget ||
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

    Status VulkanPipelineManager::validateMeshStageUsage(const OwnedGraphicsPipelineDesc& desc) const
    {
        const bool usesConventionalInput = desc.hasVertexShader || desc.hasGeometryShader || desc.hasHullShader ||
                                           desc.hasDomainShader || !desc.vertexBuffers.empty() ||
                                           desc.topology != PrimitiveTopology::triangleList ||
                                           desc.patchControlPoints != 0 || desc.stripIndexFormat.has_value();
        if ((desc.hasTaskShader && !desc.hasMeshShader) || (desc.hasMeshShader && usesConventionalInput))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Mesh graphics requires a mesh module, an optional task module, and no conventional input stages or "
                "state.",
            });
        }
        if (desc.hasMeshShader && !capabilities_.meshShaders)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support mesh shaders.",
            });
        }
        if (desc.hasTaskShader && !capabilities_.taskShaders)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support task shaders.",
            });
        }
        return {};
    }

    Status VulkanPipelineManager::validateDepthAndGeometryFeatureSupport(const OwnedGraphicsPipelineDesc& desc) const
    {
        if (desc.dynamicDepthBiasEnabled && !capabilities_.dynamicDepthBias)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support dynamic depth bias.",
            });
        }
        if (desc.depthStencil.depthBoundsTestEnabled && !capabilities_.depthBoundsTest)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support depth-bounds testing.",
            });
        }
        if (desc.hasGeometryShader && !capabilities_.geometryShaders)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support geometry shaders.",
            });
        }
        return {};
    }

    Status VulkanPipelineManager::validateTessellationUsage(const OwnedGraphicsPipelineDesc& desc) const
    {
        const bool hasTessellationPair = desc.hasHullShader && desc.hasDomainShader;
        const bool validOrdinaryTopology = desc.topology != PrimitiveTopology::patchList && !desc.hasHullShader &&
                                           !desc.hasDomainShader && desc.patchControlPoints == 0;
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
        if (hasTessellationPair && !capabilities_.tessellationShaders)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support tessellation shaders.",
            });
        }
        if (hasTessellationPair && desc.patchControlPoints > capabilities_.tessellation.maxPatchControlPoints)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The requested patch control-point count exceeds the device limit.",
            });
        }
        return {};
    }

    Status VulkanPipelineManager::validateRasterizationAndOutputSupport(const OwnedGraphicsPipelineDesc& desc) const
    {
        if (desc.topology == PrimitiveTopology::triangleFan && !capabilities_.triangleFans)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support triangle-fan topology.",
            });
        }
        if (desc.fillMode == FillMode::wireframe && !capabilities_.wireframeRasterization)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support wireframe rasterization.",
            });
        }
        if (!desc.depthClipEnabled && !capabilities_.unclippedDepth)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support unclipped depth.",
            });
        }
        if (desc.viewMask != 0 && (capabilities_.maxMultiviewViewCount == 0 ||
                                   multiviewLayerCount(desc.viewMask) > capabilities_.maxMultiviewViewCount))
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support multiview rendering.",
            });
        }
        if (!desc.streamOutputEntries.empty())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support stream output.",
            });
        }
        if (desc.logicOperation.has_value() && !capabilities_.logicOperations)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support logic operations.",
            });
        }
        return {};
    }

    Status VulkanPipelineManager::validateMultiviewStageSupport(const OwnedGraphicsPipelineDesc& desc) const
    {
        if (desc.viewMask == 0)
        {
            return {};
        }
        if (desc.hasGeometryShader && !multiviewStages_.geometry)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support multiview rendering with geometry shaders.",
            });
        }
        if (desc.hasHullShader && desc.hasDomainShader && !multiviewStages_.tessellation)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support multiview rendering with tessellation shaders.",
            });
        }
        if (desc.hasMeshShader && !multiviewStages_.mesh)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support multiview rendering with mesh shaders.",
            });
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] Status validateConservativeRasterization(ConservativeRasterizationMode mode,
                                                               FillMode fillMode,
                                                               const DeviceCapabilities& capabilities)
        {
            if (mode == ConservativeRasterizationMode::disabled)
            {
                return {};
            }
            if (fillMode == FillMode::wireframe)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Conservative rasterization requires solid fill mode.",
                });
            }
            if (!capabilities.conservativeRasterization)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The device does not support conservative rasterization.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validatePrimitiveRestart(const StripIndexFormat& stripIndexFormat,
                                                      PrimitiveTopology topology,
                                                      const DeviceCapabilities& capabilities)
        {
            if (!stripIndexFormat.has_value())
            {
                return {};
            }
            const bool validIndexFormat =
                *stripIndexFormat == IndexFormat::uint16 || *stripIndexFormat == IndexFormat::uint32;
            if (!isStripTopology(topology) || !validIndexFormat)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Primitive restart requires a strip topology and a valid index format.",
                });
            }
            if (!capabilities.primitiveRestart)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The device does not support primitive restart.",
                });
            }
            return {};
        }

    } // namespace

    Status VulkanPipelineManager::validateStageCombinationAndFeatureSupport(const OwnedGraphicsPipelineDesc& desc) const
    {
        if (auto valid = validateMeshStageUsage(desc); !valid)
        {
            return valid;
        }
        if (auto supported = validateDepthAndGeometryFeatureSupport(desc); !supported)
        {
            return supported;
        }
        if (auto valid = validateTessellationUsage(desc); !valid)
        {
            return valid;
        }
        if (auto supported = validateRasterizationAndOutputSupport(desc); !supported)
        {
            return supported;
        }
        if (auto supported = validateMultiviewStageSupport(desc); !supported)
        {
            return supported;
        }
        if (auto valid =
                validateConservativeRasterization(desc.conservativeRasterization, desc.fillMode, capabilities_);
            !valid)
        {
            return valid;
        }
        if (auto valid = validatePrimitiveRestart(desc.stripIndexFormat, desc.topology, capabilities_); !valid)
        {
            return valid;
        }
        return validation_detail::validateDepthBiasClampSupport(depthBiasClampSupported_, desc.depthBiasClamp);
    }

    namespace
    {

        struct PipelineShaderStage final
        {
            VkShaderStageFlagBits stage = {};
            const VulkanShaderManager::NativeShaderModule* shader = nullptr;
        };

    } // namespace

    struct VulkanPipelineManager::GraphicsShaderStages final
    {
        std::optional<VulkanShaderManager::NativeShaderModule> vertex;
        std::optional<VulkanShaderManager::NativeShaderModule> task;
        std::optional<VulkanShaderManager::NativeShaderModule> mesh;
        std::optional<VulkanShaderManager::NativeShaderModule> pixel;
        std::optional<VulkanShaderManager::NativeShaderModule> geometry;
        std::optional<VulkanShaderManager::NativeShaderModule> hull;
        std::optional<VulkanShaderManager::NativeShaderModule> domain;

        [[nodiscard]] std::vector<PipelineShaderStage> inPipelineOrder() const;
    };

    std::vector<PipelineShaderStage> VulkanPipelineManager::GraphicsShaderStages::inPipelineOrder() const
    {
        std::vector<PipelineShaderStage> present;
        const auto appendIfPresent = [&present](VkShaderStageFlagBits stage,
                                                const std::optional<VulkanShaderManager::NativeShaderModule>& shader) {
            if (shader.has_value())
            {
                present.push_back(PipelineShaderStage{.stage = stage, .shader = &*shader});
            }
        };
        appendIfPresent(VK_SHADER_STAGE_VERTEX_BIT, vertex);
        appendIfPresent(VK_SHADER_STAGE_TASK_BIT_EXT, task);
        appendIfPresent(VK_SHADER_STAGE_MESH_BIT_EXT, mesh);
        appendIfPresent(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, hull);
        appendIfPresent(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, domain);
        appendIfPresent(VK_SHADER_STAGE_GEOMETRY_BIT, geometry);
        appendIfPresent(VK_SHADER_STAGE_FRAGMENT_BIT, pixel);
        return present;
    }

    namespace
    {

        [[nodiscard]] Status validateStagedShader(const std::optional<VulkanShaderManager::NativeShaderModule>& shader,
                                                  ShaderStage stage)
        {
            if (!shader.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics pipeline references an invalid or stale layout or shader module.",
                });
            }
            if (shader->stage != stage || shader->module == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics pipeline requires correctly staged conventional or task/mesh shader modules.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateStagedShaderForLayout(
            const std::optional<VulkanShaderManager::NativeShaderModule>& shader,
            ShaderStage stage,
            const VulkanNativePipelineLayout& layout)
        {
            if (auto staged = validateStagedShader(shader, stage); !staged)
            {
                return staged;
            }
            return validateShaderInterface(*shader, layout);
        }

        [[nodiscard]] Status validateShaderInterfaceIfPresent(
            const std::optional<VulkanShaderManager::NativeShaderModule>& shader,
            const VulkanNativePipelineLayout& layout)
        {
            if (!shader.has_value())
            {
                return {};
            }
            return validateShaderInterface(*shader, layout);
        }

        [[nodiscard]] Status validateGraphicsStageLinkage(const VulkanShaderManager::NativeShaderModule* producer,
                                                          const VulkanShaderManager::NativeShaderModule* pixel)
        {
            if (producer == nullptr || pixel == nullptr)
            {
                return {};
            }
            return detail::validateStageLinkage(producer->stage, producer->stageOutputs, pixel->stageInputs);
        }

        [[nodiscard]] Status validateTessellationShaderPair(
            const std::optional<VulkanShaderManager::NativeShaderModule>& hull,
            const std::optional<VulkanShaderManager::NativeShaderModule>& domain,
            const VulkanNativePipelineLayout& layout)
        {
            if (!hull.has_value() || !domain.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics pipeline references an invalid or stale layout or shader module.",
                });
            }
            if (hull->stage != ShaderStage::hull || hull->module == VK_NULL_HANDLE ||
                domain->stage != ShaderStage::domain || domain->module == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics pipeline requires correctly staged conventional or task/mesh shader modules.",
                });
            }
            if (auto hullInterface = validateShaderInterface(*hull, layout); !hullInterface)
            {
                return hullInterface;
            }
            return validateShaderInterface(*domain, layout);
        }

    } // namespace

    Result<VulkanPipelineManager::GraphicsShaderStages> VulkanPipelineManager::takeValidatedShaderStages(
        PreparedGraphicsCompilation& prepared, const VulkanNativePipelineLayout& layout)
    {
        const auto& desc = prepared.desc;
        GraphicsShaderStages stages;
        if (desc.hasMeshShader)
        {
            if (auto valid = validateStagedShaderForLayout(prepared.meshShader, ShaderStage::mesh, layout); !valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            stages.mesh = std::move(*prepared.meshShader);
            if (desc.hasTaskShader)
            {
                if (auto valid = validateStagedShaderForLayout(prepared.taskShader, ShaderStage::task, layout); !valid)
                {
                    return std::unexpected(std::move(valid.error()));
                }
                stages.task = std::move(*prepared.taskShader);
            }
        }
        else
        {
            if (auto valid = validateStagedShader(prepared.vertexShader, ShaderStage::vertex); !valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            stages.vertex = std::move(*prepared.vertexShader);
        }
        if (desc.hasPixelShader || desc.rasterizedStream.has_value())
        {
            if (auto valid = validateStagedShader(prepared.pixelShader, ShaderStage::pixel); !valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            stages.pixel = std::move(*prepared.pixelShader);
        }
        if (auto valid = validateShaderInterfaceIfPresent(stages.vertex, layout); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        if (auto valid = validateShaderInterfaceIfPresent(stages.pixel, layout); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        if (desc.hasGeometryShader)
        {
            if (auto valid = validateStagedShaderForLayout(prepared.geometryShader, ShaderStage::geometry, layout);
                !valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            stages.geometry = std::move(*prepared.geometryShader);
        }
        if (desc.hasHullShader && desc.hasDomainShader)
        {
            if (auto valid = validateTessellationShaderPair(prepared.hullShader, prepared.domainShader, layout); !valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            stages.hull = std::move(*prepared.hullShader);
            stages.domain = std::move(*prepared.domainShader);
        }
        return stages;
    }

    Status VulkanPipelineManager::validatePointSizeOutput(const OwnedGraphicsPipelineDesc& desc,
                                                          const GraphicsShaderStages& shaders) const
    {
        const bool vertexStageRasterizesPoints = desc.topology == PrimitiveTopology::pointList &&
                                                 shaders.vertex.has_value() && !shaders.geometry.has_value() &&
                                                 !shaders.domain.has_value();
        if (maintenance5_ || !vertexStageRasterizesPoints || shaders.vertex->writesPointSize)
        {
            return {};
        }
        return std::unexpected(Error{
            ErrorCode::unsupported,
            0,
            "Without VK_KHR_maintenance5, a point-list pipeline with no geometry or tessellation stage needs a vertex "
            "shader that writes PointSize; write 1.0 to an output declared [[vk::builtin(\"PointSize\")]].",
        });
    }

    struct VulkanPipelineManager::VertexInputLayout final
    {
        std::vector<VkVertexInputBindingDescription> bindings;
        std::vector<VkVertexInputAttributeDescription> attributes;
        std::vector<std::uint32_t> declaredLocations;
        std::vector<std::uint32_t> strides;
    };

    Status VulkanPipelineManager::validateVertexBufferLayout(const OwnedVertexBufferLayout& buffer) const
    {
        const bool validStepMode =
            buffer.stepMode == VertexStepMode::vertex || buffer.stepMode == VertexStepMode::instance;
        const bool validStepRate = (buffer.stepMode == VertexStepMode::vertex && buffer.instanceStepRate == 1) ||
                                   (buffer.stepMode == VertexStepMode::instance && buffer.instanceStepRate != 0);
        if (buffer.stride == 0 || buffer.stride > capabilities_.vertexInput.maxBufferStride ||
            buffer.attributes.empty() || !validStepMode || !validStepRate)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Vertex buffer layouts require attributes, a valid stride, step mode, and instance step rate.",
            });
        }
        if (buffer.stepMode == VertexStepMode::instance &&
            buffer.instanceStepRate > capabilities_.vertexInput.maxInstanceStepRate)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support the requested vertex instance step rate.",
            });
        }
        if (buffer.stride % minVertexInputBindingStrideAlignment_ != 0)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device requires vertex buffer strides to be a multiple of its portability-subset stride "
                "alignment.",
            });
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] Status validateVertexAttribute(const VertexAttribute& attribute,
                                                     std::uint32_t stride,
                                                     std::span<const std::uint32_t> declaredLocations,
                                                     const VertexInputCapabilities& capabilities)
        {
            const auto attributeSize = detail::vertexFormatSize(attribute.format);
            const bool locationAlreadyDeclared =
                std::ranges::find(declaredLocations, attribute.shaderLocation) != declaredLocations.end();
            const bool fitsWithinStride =
                attributeSize != 0 && attribute.offset <= stride && attributeSize <= stride - attribute.offset;
            if (attribute.shaderLocation >= maxVertexAttributes || locationAlreadyDeclared || !fitsWithinStride)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Vertex attributes require unique portable locations and must fit within their stride.",
                });
            }
            if (vulkanVertexFormat(attribute.format) == VK_FORMAT_UNDEFINED ||
                !capabilities.supports(attribute.format) || attribute.offset > capabilities.maxAttributeOffset)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The device does not support the requested vertex attribute format.",
                });
            }
            return {};
        }

    } // namespace

    Result<VulkanPipelineManager::VertexInputLayout> VulkanPipelineManager::buildVertexInputLayout(
        const std::vector<OwnedVertexBufferLayout>& vertexBuffers) const
    {
        VertexInputLayout layout;
        layout.bindings.reserve(vertexBuffers.size());
        layout.strides.reserve(vertexBuffers.size());
        for (std::size_t slot = 0; slot < vertexBuffers.size(); ++slot)
        {
            const auto& buffer = vertexBuffers[slot];
            if (auto valid = validateVertexBufferLayout(buffer); !valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            layout.bindings.push_back(VkVertexInputBindingDescription{
                .binding = static_cast<std::uint32_t>(slot),
                .stride = buffer.stride,
                .inputRate = buffer.stepMode == VertexStepMode::instance ? VK_VERTEX_INPUT_RATE_INSTANCE
                                                                         : VK_VERTEX_INPUT_RATE_VERTEX,
            });
            layout.strides.push_back(buffer.stride);
            for (const auto& attribute : buffer.attributes)
            {
                if (auto valid = validateVertexAttribute(
                        attribute, buffer.stride, layout.declaredLocations, capabilities_.vertexInput);
                    !valid)
                {
                    return std::unexpected(std::move(valid.error()));
                }
                layout.declaredLocations.push_back(attribute.shaderLocation);
                layout.attributes.push_back(VkVertexInputAttributeDescription{
                    .location = attribute.shaderLocation,
                    .binding = static_cast<std::uint32_t>(slot),
                    .format = vulkanVertexFormat(attribute.format),
                    .offset = attribute.offset,
                });
            }
        }
        if (layout.attributes.size() > maxVertexAttributes)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The graphics pipeline exceeds the portable vertex attribute limit.",
            });
        }
        return layout;
    }

    namespace
    {

        [[nodiscard]] constexpr bool isValidBlendComponent(const BlendComponent& component) noexcept
        {
            return isValidBlendFactor(component.sourceFactor) && isValidBlendFactor(component.destinationFactor) &&
                   isValidBlendOperation(component.operation);
        }

        [[nodiscard]] constexpr bool isValidColorTargetState(const ColorTargetState& target) noexcept
        {
            return target.format != Format::unknown && isValidBlendComponent(target.blend.color) &&
                   isValidBlendComponent(target.blend.alpha);
        }

        [[nodiscard]] constexpr bool usesDualSourceBlending(const BlendState& blend) noexcept
        {
            return blend.enabled &&
                   (usesDualSource(blend.color.sourceFactor) || usesDualSource(blend.color.destinationFactor) ||
                    usesDualSource(blend.alpha.sourceFactor) || usesDualSource(blend.alpha.destinationFactor));
        }

        [[nodiscard]] Status validateColorTargetSupport(const ColorTargetState& target,
                                                        FormatFeature formatFeatures,
                                                        bool usesLogicOperation)
        {
            if (!hasAny(formatFeatures, FormatFeature::colorAttachment))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The device does not support the requested color-attachment format.",
                });
            }
            if (target.blend.enabled && !hasAny(formatFeatures, FormatFeature::blendable))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The device does not support blending for the requested color-attachment format.",
                });
            }
            if (usesLogicOperation && target.blend.enabled)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Logic operations and color blending cannot be enabled together.",
                });
            }
            if (usesLogicOperation && !hasAny(formatFeatures, FormatFeature::logicOperation))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "A color-target format does not support output-merger logic operations.",
                });
            }
            if (usesDualSourceBlending(target.blend) &&
                (target.blend.color.operation > BlendOperation::reverseSubtract ||
                 target.blend.alpha.operation > BlendOperation::reverseSubtract))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Dual-source blending supports only add, subtract, and reverse-subtract operations.",
                });
            }
            return {};
        }

    } // namespace

    Status VulkanPipelineManager::validateColorTargets(const OwnedGraphicsPipelineDesc& desc) const
    {
        const bool usesLogicOperation = desc.logicOperation.has_value();
        bool anyTargetUsesDualSource = false;
        for (const auto& target : desc.colorTargets)
        {
            if (!isValidColorTargetState(target))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Graphics pipelines require valid attachment, rasterization, and multisample state.",
                });
            }
            const auto properties = activeFormatCache_->query(instanceTable_, physical_, target.format);
            if (auto supported = validateColorTargetSupport(target, properties.features, usesLogicOperation);
                !supported)
            {
                return supported;
            }
            anyTargetUsesDualSource = anyTargetUsesDualSource || usesDualSourceBlending(target.blend);
        }
        if (anyTargetUsesDualSource && desc.colorTargets.size() != 1)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Dual-source blending is supported only for a graphics pipeline with exactly one color target.",
            });
        }
        if (anyTargetUsesDualSource && !capabilities_.dualSourceBlending)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support dual-source blending.",
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
        return {};
    }

    Status VulkanPipelineManager::validateDepthStencilState(const DepthStencilState& depthStencil) const
    {
        if (depthStencil.format != Format::unknown)
        {
            if (depthStencil.format != Format::d32Float && depthStencil.format != Format::d24UnormS8Uint)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A graphics depth-stencil target requires a depth format.",
                });
            }
            const auto properties = activeFormatCache_->query(instanceTable_, physical_, depthStencil.format);
            if (!hasAny(properties.features, FormatFeature::depthStencilAttachment))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The device does not support the requested depth-stencil format.",
                });
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

    namespace
    {

        [[nodiscard]] std::vector<ShaderSpecialization> makeSpecializations(std::span<const PipelineShaderStage> stages)
        {
            std::vector<ShaderSpecialization> specializations;
            specializations.reserve(stages.size());
            for (const auto& stage : stages)
            {
                specializations.push_back(makeSpecialization(stage.shader->specializationConstants));
            }
            return specializations;
        }

        [[nodiscard]] std::vector<VkPipelineShaderStageCreateInfo> makeShaderStageCreateInfos(
            std::span<const PipelineShaderStage> stages, std::span<const ShaderSpecialization> specializations)
        {
            std::vector<VkPipelineShaderStageCreateInfo> createInfos;
            createInfos.reserve(stages.size());
            for (std::size_t index = 0; index < stages.size(); ++index)
            {
                createInfos.push_back(
                    makeShaderStageCreateInfo(stages[index].stage, *stages[index].shader, specializations[index]));
            }
            return createInfos;
        }

        [[nodiscard]] VkPipelineVertexInputStateCreateInfo makeVertexInputState(
            std::span<const VkVertexInputBindingDescription> bindings,
            std::span<const VkVertexInputAttributeDescription> attributes)
        {
            return VkPipelineVertexInputStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .vertexBindingDescriptionCount = static_cast<std::uint32_t>(bindings.size()),
                .pVertexBindingDescriptions = bindings.data(),
                .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size()),
                .pVertexAttributeDescriptions = attributes.data(),
            };
        }

        [[nodiscard]] VkPipelineInputAssemblyStateCreateInfo makeInputAssemblyState(PrimitiveTopology topology,
                                                                                    bool primitiveRestartEnabled)
        {
            return VkPipelineInputAssemblyStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .topology = vulkanTopology(topology),
                .primitiveRestartEnable = primitiveRestartEnabled ? VK_TRUE : VK_FALSE,
            };
        }

        [[nodiscard]] VkPipelineTessellationStateCreateInfo makeTessellationState(std::uint32_t patchControlPoints)
        {
            return VkPipelineTessellationStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .patchControlPoints = patchControlPoints,
            };
        }

        [[nodiscard]] VkPipelineViewportStateCreateInfo makeDynamicViewportState(bool multiViewport)
        {
            return VkPipelineViewportStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .viewportCount = multiViewport ? 0U : 1U,
                .pViewports = {},
                .scissorCount = multiViewport ? 0U : 1U,
                .pScissors = {},
            };
        }

        [[nodiscard]] VkPipelineRasterizationConservativeStateCreateInfoEXT makeOverestimatingConservativeState()
        {
            return VkPipelineRasterizationConservativeStateCreateInfoEXT{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
                .pNext = {},
                .flags = {},
                .conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT,
                .extraPrimitiveOverestimationSize = 0.0F,
            };
        }

        [[nodiscard]] VkPipelineMultisampleStateCreateInfo makeMultisampleState(std::uint32_t sampleCount,
                                                                                const VkSampleMask* sampleMask,
                                                                                bool alphaToCoverageEnabled)
        {
            return VkPipelineMultisampleStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .rasterizationSamples = vulkanSampleCount(sampleCount),
                .sampleShadingEnable = VK_FALSE,
                .minSampleShading = {},
                .pSampleMask = sampleMask,
                .alphaToCoverageEnable = alphaToCoverageEnabled ? VK_TRUE : VK_FALSE,
                .alphaToOneEnable = {},
            };
        }

        [[nodiscard]] VkStencilOpState makeStencilOpState(const StencilFaceState& face,
                                                          const DepthStencilState& depthStencil)
        {
            return VkStencilOpState{
                .failOp = vulkanStencilOp(face.failOperation),
                .passOp = vulkanStencilOp(face.passOperation),
                .depthFailOp = vulkanStencilOp(face.depthFailOperation),
                .compareOp = vulkanCompare(face.compare),
                .compareMask = depthStencil.stencilReadMask,
                .writeMask = depthStencil.stencilWriteMask,
                .reference = {},
            };
        }

        [[nodiscard]] VkPipelineDepthStencilStateCreateInfo makeDepthStencilState(const DepthStencilState& depthStencil)
        {
            return VkPipelineDepthStencilStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .depthTestEnable = depthStencil.depthTestEnabled ? VK_TRUE : VK_FALSE,
                .depthWriteEnable = depthStencil.depthWriteEnabled ? VK_TRUE : VK_FALSE,
                .depthCompareOp = vulkanCompare(depthStencil.depthCompare),
                .depthBoundsTestEnable = depthStencil.depthBoundsTestEnabled ? VK_TRUE : VK_FALSE,
                .stencilTestEnable = depthStencil.stencilTestEnabled ? VK_TRUE : VK_FALSE,
                .front = makeStencilOpState(depthStencil.frontFace, depthStencil),
                .back = makeStencilOpState(depthStencil.backFace, depthStencil),
                .minDepthBounds = {},
                .maxDepthBounds = {},
            };
        }

        [[nodiscard]] std::vector<VkPipelineColorBlendAttachmentState> makeBlendAttachmentStates(
            std::span<const ColorTargetState> targets)
        {
            std::vector<VkPipelineColorBlendAttachmentState> attachments;
            attachments.reserve(targets.size());
            for (const auto& target : targets)
            {
                attachments.push_back(VkPipelineColorBlendAttachmentState{
                    .blendEnable = target.blend.enabled ? VK_TRUE : VK_FALSE,
                    .srcColorBlendFactor = vulkanBlendFactor(target.blend.color.sourceFactor),
                    .dstColorBlendFactor = vulkanBlendFactor(target.blend.color.destinationFactor),
                    .colorBlendOp = vulkanBlendOp(target.blend.color.operation),
                    .srcAlphaBlendFactor = vulkanBlendFactor(target.blend.alpha.sourceFactor),
                    .dstAlphaBlendFactor = vulkanBlendFactor(target.blend.alpha.destinationFactor),
                    .alphaBlendOp = vulkanBlendOp(target.blend.alpha.operation),
                    .colorWriteMask = vulkanColorWriteMask(target.writeMask),
                });
            }
            return attachments;
        }

        [[nodiscard]] VkPipelineColorBlendStateCreateInfo makeColorBlendState(
            const std::optional<LogicOperation>& logicOperation,
            std::span<const VkPipelineColorBlendAttachmentState> attachments)
        {
            return VkPipelineColorBlendStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .logicOpEnable = logicOperation.has_value() ? VK_TRUE : VK_FALSE,
                .logicOp = logicOperation.has_value() ? vulkanLogicOp(*logicOperation) : VK_LOGIC_OP_COPY,
                .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .blendConstants = {},
            };
        }

        [[nodiscard]] std::vector<VkDynamicState> graphicsDynamicStates(const DeviceCapabilities& capabilities,
                                                                        bool dynamicDepthBias,
                                                                        bool depthBoundsTest)
        {
            std::vector<VkDynamicState> dynamicStates{
                capabilities.multiViewport ? VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT : VK_DYNAMIC_STATE_VIEWPORT,
                capabilities.multiViewport ? VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT : VK_DYNAMIC_STATE_SCISSOR,
                VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                VK_DYNAMIC_STATE_STENCIL_REFERENCE,
            };
            if (dynamicDepthBias)
            {
                dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
            }
            if (depthBoundsTest)
            {
                dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
            }
            if (capabilities.variableRateShading)
            {
                dynamicStates.push_back(VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR);
            }
            return dynamicStates;
        }

        [[nodiscard]] VkPipelineDynamicStateCreateInfo makeDynamicState(std::span<const VkDynamicState> dynamicStates)
        {
            return VkPipelineDynamicStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
                .pDynamicStates = dynamicStates.data(),
            };
        }

        [[nodiscard]] std::vector<VkFormat> vulkanColorFormats(std::span<const ColorTargetState> targets)
        {
            std::vector<VkFormat> formats;
            formats.reserve(targets.size());
            for (const auto& target : targets)
            {
                formats.push_back(vulkanFormat(target.format));
            }
            return formats;
        }

        [[nodiscard]] VkPipelineRenderingCreateInfo makeRenderingInfo(std::uint32_t viewMask,
                                                                      std::span<const VkFormat> colorFormats,
                                                                      Format depthStencilFormat)
        {
            const auto depthFormat =
                depthStencilFormat == Format::unknown ? VK_FORMAT_UNDEFINED : vulkanFormat(depthStencilFormat);
            const auto stencilFormat = depthStencilFormat == Format::d24UnormS8Uint ? depthFormat : VK_FORMAT_UNDEFINED;
            return VkPipelineRenderingCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                .pNext = {},
                .viewMask = viewMask,
                .colorAttachmentCount = static_cast<std::uint32_t>(colorFormats.size()),
                .pColorAttachmentFormats = colorFormats.data(),
                .depthAttachmentFormat = depthFormat,
                .stencilAttachmentFormat = stencilFormat,
            };
        }

    } // namespace

    VkPipelineRasterizationStateCreateInfo VulkanPipelineManager::makeRasterizationState(
        const OwnedGraphicsPipelineDesc& desc)
    {
        const bool usesStaticDepthBias =
            hasStaticDepthBias(desc.depthBias, desc.depthBiasClamp, desc.depthBiasSlopeScale);
        return VkPipelineRasterizationStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = {},
            .flags = {},
            .depthClampEnable = desc.depthClipEnabled ? VK_FALSE : VK_TRUE,
            .rasterizerDiscardEnable = desc.rasterizedStream.has_value() ? VK_FALSE : VK_TRUE,
            .polygonMode = desc.fillMode == FillMode::wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL,
            .cullMode = vulkanCullMode(desc.cullMode),
            .frontFace =
                desc.frontFace == FrontFace::clockwise ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = (desc.dynamicDepthBiasEnabled || usesStaticDepthBias) ? VK_TRUE : VK_FALSE,
            .depthBiasConstantFactor = static_cast<float>(desc.depthBias),
            .depthBiasClamp = desc.depthBiasClamp,
            .depthBiasSlopeFactor = desc.depthBiasSlopeScale,
            .lineWidth = 1.0F,
        };
    }

    Result<VkPipeline> VulkanPipelineManager::createNativeGraphicsPipeline(const OwnedGraphicsPipelineDesc& desc,
                                                                           const GraphicsShaderStages& shaders,
                                                                           const VertexInputLayout& vertexLayout,
                                                                           VkPipelineLayout pipelineLayout,
                                                                           VkPipelineCache nativeCache) const
    {
        const auto shaderStages = shaders.inPipelineOrder();
        const auto specializations = makeSpecializations(shaderStages);
        const auto stages = makeShaderStageCreateInfos(shaderStages, specializations);
        const auto vertexInput = makeVertexInputState(vertexLayout.bindings, vertexLayout.attributes);
        const auto inputAssembly = makeInputAssemblyState(desc.topology, desc.stripIndexFormat.has_value());
        const auto tessellation = makeTessellationState(desc.patchControlPoints);
        const bool usesTessellation = desc.hasHullShader && desc.hasDomainShader;
        const auto viewport = makeDynamicViewportState(capabilities_.multiViewport);
        auto rasterization = makeRasterizationState(desc);
        const auto conservative = makeOverestimatingConservativeState();
        if (desc.conservativeRasterization != ConservativeRasterizationMode::disabled)
        {
            rasterization.pNext = &conservative;
        }
        VkSampleMask const sampleMask = desc.sampleMask;
        const auto multisample = makeMultisampleState(desc.sampleCount, &sampleMask, desc.alphaToCoverageEnabled);
        const auto depthStencil = makeDepthStencilState(desc.depthStencil);
        const auto blendAttachments = makeBlendAttachmentStates(desc.colorTargets);
        const auto colorBlend = makeColorBlendState(desc.logicOperation, blendAttachments);
        const auto dynamicStates = graphicsDynamicStates(
            capabilities_, desc.dynamicDepthBiasEnabled, desc.depthStencil.depthBoundsTestEnabled);
        const auto dynamic = makeDynamicState(dynamicStates);
        const auto colorFormats = vulkanColorFormats(desc.colorTargets);
        const auto rendering = makeRenderingInfo(desc.viewMask, colorFormats, desc.depthStencil.format);
        const auto createInfo = VkGraphicsPipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering,
            .flags = capabilities_.variableRateShadingTier2
                         ? VkPipelineCreateFlags{VK_PIPELINE_CREATE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR}
                         : VkPipelineCreateFlags{},
            .stageCount = static_cast<std::uint32_t>(stages.size()),
            .pStages = stages.data(),
            .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pTessellationState = usesTessellation ? &tessellation : nullptr,
            .pViewportState = &viewport,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlend,
            .pDynamicState = &dynamic,
            .layout = pipelineLayout,
            .renderPass = {},
            .subpass = {},
            .basePipelineHandle = {},
            .basePipelineIndex = {},
        };

#ifdef RACHIS_TESTING
        capturePipelineCreation(stages, nativeCache);
#endif
        VkPipeline pipeline = VK_NULL_HANDLE;
        const auto created =
            table_->vkCreateGraphicsPipelines(device_, nativeCache, 1, &createInfo, nullptr, &pipeline);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateGraphicsPipelines"));
        }
        return pipeline;
    }

    std::shared_ptr<VulkanPipelineManager::PipelineKeepAlive> VulkanPipelineManager::adoptGraphicsPipeline(
        VkPipeline pipeline, const std::shared_ptr<void>& layoutOwner, const GraphicsShaderStages& shaders) const
    {
        auto keepAlive = makePipelineKeepAlive(pipeline, layoutOwner);
        for (const auto& stage : shaders.inPipelineOrder())
        {
            keepAlive->shaderOwners.push_back(stage.shader->owner);
        }
        return keepAlive;
    }

    namespace
    {

        [[nodiscard]] std::vector<Format> colorTargetFormats(std::span<const ColorTargetState> targets)
        {
            std::vector<Format> formats;
            formats.reserve(targets.size());
            for (const auto& target : targets)
            {
                formats.push_back(target.format);
            }
            return formats;
        }

        [[nodiscard]] constexpr bool usesBlendConstants(const BlendState& blend) noexcept
        {
            return blend.enabled &&
                   (usesBlendConstant(blend.color.sourceFactor) || usesBlendConstant(blend.color.destinationFactor) ||
                    usesBlendConstant(blend.alpha.sourceFactor) || usesBlendConstant(blend.alpha.destinationFactor));
        }

        [[nodiscard]] bool anyTargetUsesBlendConstants(std::span<const ColorTargetState> targets)
        {
            return std::ranges::any_of(targets,
                                       [](const ColorTargetState& target) { return usesBlendConstants(target.blend); });
        }

    } // namespace

    Result<PipelineHandle> VulkanPipelineManager::publishGraphicsPipeline(
        const PreparedGraphicsCompilation& prepared,
        const std::shared_ptr<PipelineKeepAlive>& keepAlive,
        std::vector<std::uint32_t> vertexStrides,
        bool readsDrawIndex)
    {
        const auto& desc = prepared.desc;
        const auto& layout = *prepared.layout;
        try
        {
            auto publishedColorFormats = colorTargetFormats(desc.colorTargets);
            std::vector<std::uint32_t> vertexAlignments;
            vertexAlignments.reserve(desc.vertexBuffers.size());
            for (const auto& buffer : desc.vertexBuffers)
            {
                vertexAlignments.push_back(detail::vertexAttributesAlignment(buffer.attributes));
            }
            auto sets = collectSetLayouts(layout);
            auto native = std::make_shared<VulkanNativePipeline>(VulkanNativePipeline{
                .owner = keepAlive,
                .layoutOwner = layout.owner,
                .pipeline = keepAlive->pipeline,
                .layout = layout.pipelineLayout,
                .compute = false,
                .mesh = desc.hasMeshShader,
                .rasterizationEnabled = desc.rasterizedStream.has_value(),
                .stencilEnabled = desc.depthStencil.stencilTestEnabled,
                .usesBlendConstants = anyTargetUsesBlendConstants(desc.colorTargets),
                .dynamicDepthBias = desc.dynamicDepthBiasEnabled,
                .depthBoundsTest = desc.depthStencil.depthBoundsTestEnabled,
                .readsDrawIndex = readsDrawIndex,
                .sampleCount = desc.sampleCount,
                .viewMask = desc.viewMask,
                .depthStencilFormat = desc.depthStencil.format,
                .colorFormats = std::move(publishedColorFormats),
                .vertexStrides = std::move(vertexStrides),
                .vertexAlignments = std::move(vertexAlignments),
                .pushConstantRanges = layout.pushConstantRanges,
                .nativePushConstantRanges = layout.nativePushConstantRanges,
                .setRequired = std::move(sets.required),
                .setDynamicCounts = std::move(sets.dynamicCounts),
                .setLayouts = std::move(sets.layouts),
            });
            return publishPipeline(PipelineRecord{
                .native = std::move(native),
            });
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate immutable graphics pipeline record.",
            });
        }
    }

    Result<PipelineHandle> VulkanPipelineManager::compileGraphicsPipeline(PreparedGraphicsCompilation&& prepared)
    {
        const auto& desc = prepared.desc;
        if (auto valid = validateGraphicsPipelineState(desc); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        auto cache = resolvePreparedCache(desc.cachePolicy,
                                          desc.hasCache,
                                          std::move(prepared.cacheRecord),
                                          desc.cacheKey,
                                          PipelineCacheEntryKind::graphics);
        if (!cache)
        {
            return std::unexpected(std::move(cache.error()));
        }
        if (auto supported = validateStageCombinationAndFeatureSupport(desc); !supported)
        {
            return std::unexpected(std::move(supported.error()));
        }
        if (shaders_ == nullptr || bindings_ == nullptr || table_ == nullptr || device_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The Vulkan pipeline manager is not initialized.",
            });
        }
        if (!prepared.layout.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A graphics pipeline references an invalid or stale layout or shader module.",
            });
        }
        const auto& layout = *prepared.layout;

        auto shaders = takeValidatedShaderStages(prepared, layout);
        if (!shaders)
        {
            return std::unexpected(std::move(shaders.error()));
        }
        if (shaders->task.has_value() && shaders->mesh.has_value() && shaders->mesh->drawIndex)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "On Vulkan, only the task shader of a task and mesh pipeline can read the draw index. Pass it to the "
                "mesh shader in the task payload.",
            });
        }
        const auto* lastPreRasterizationStage = shaders->geometry.has_value() ? &*shaders->geometry
                                                : shaders->domain.has_value() ? &*shaders->domain
                                                : shaders->mesh.has_value()   ? &*shaders->mesh
                                                : shaders->vertex.has_value() ? &*shaders->vertex
                                                                              : nullptr;
        const auto* pixelStage = shaders->pixel.has_value() ? &*shaders->pixel : nullptr;
        if (auto linked = validateGraphicsStageLinkage(lastPreRasterizationStage, pixelStage); !linked)
        {
            return std::unexpected(std::move(linked.error()));
        }
        auto vertexLayout = buildVertexInputLayout(desc.vertexBuffers);
        if (!vertexLayout)
        {
            return std::unexpected(std::move(vertexLayout.error()));
        }
        if (shaders->vertex.has_value())
        {
            if (auto vertexInterface = validateVertexInterface(*shaders->vertex, vertexLayout->declaredLocations);
                !vertexInterface)
            {
                return std::unexpected(std::move(vertexInterface.error()));
            }
        }
        if (auto pointSize = validatePointSizeOutput(desc, *shaders); !pointSize)
        {
            return std::unexpected(std::move(pointSize.error()));
        }

        IsolatedShaderModules isolated(device_, table_);
        const bool isolates = isolatesShaderModules(layout);
        if (isolates)
        {
            for (auto* stage : {&shaders->vertex,
                                &shaders->task,
                                &shaders->mesh,
                                &shaders->pixel,
                                &shaders->geometry,
                                &shaders->hull,
                                &shaders->domain})
            {
                if (!stage->has_value())
                {
                    continue;
                }
                if (auto replaced = isolated.replace(**stage); !replaced)
                {
                    return std::unexpected(std::move(replaced.error()));
                }
            }
        }

        auto lockedCache = lockCacheAndConsumeEntry(*cache);
        if (!lockedCache)
        {
            return std::unexpected(std::move(lockedCache.error()));
        }
        if (auto valid = validateColorTargets(desc); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        if (auto valid = validateDepthStencilState(desc.depthStencil); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }

        auto pipeline = createNativeGraphicsPipeline(
            desc, *shaders, *vertexLayout, layout.pipelineLayout, isolates ? VK_NULL_HANDLE : lockedCache->native);
        if (!pipeline)
        {
            return std::unexpected(std::move(pipeline.error()));
        }
        static_cast<void>(storeCacheEntry(*cache));

        const auto keepAlive = adoptGraphicsPipeline(*pipeline, layout.owner, *shaders);
        labelPipeline(*pipeline, desc.label);
        const bool readsDrawIndex =
            std::ranges::any_of(std::array{&shaders->vertex, &shaders->task, &shaders->mesh},
                                [](const std::optional<VulkanShaderManager::NativeShaderModule>* stage) {
                                    return stage->has_value() && (*stage)->drawIndex;
                                });
        return publishGraphicsPipeline(prepared, keepAlive, std::move(vertexLayout->strides), readsDrawIndex);
    }

    Result<PipelineHandle> VulkanPipelineManager::createGraphicsPipeline(const GraphicsPipelineDesc& desc)
    {
        auto prepared = prepareGraphicsCompilation(desc);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        return compileGraphicsPipeline(std::move(*prepared));
    }

    Result<PipelineCompilationHandle> VulkanPipelineManager::beginComputePipelineCompilation(
        const ComputePipelineDesc& desc)
    {
        auto prepared = prepareComputeCompilation(desc);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        return enqueueCompilation(std::move(*prepared));
    }

    Result<PipelineCompilationHandle> VulkanPipelineManager::beginGraphicsPipelineCompilation(
        const GraphicsPipelineDesc& desc)
    {
        auto prepared = prepareGraphicsCompilation(desc);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        return enqueueCompilation(std::move(*prepared));
    }

    Result<PipelineCompilationState> VulkanPipelineManager::pipelineCompilationState(
        PipelineCompilationHandle compilation)
    {
        std::shared_ptr<CompilationRecord> record;
        {
            const std::scoped_lock lock(mutex_);
            const auto* found = compilations_.find(compilation);
            if (found == nullptr || *found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline compilation handle is invalid or already consumed.",
                });
            }
            record = *found;
        }

        const std::scoped_lock recordLock(record->mutex);
        if (record->consumed || record->abandoned)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The pipeline compilation request has already been consumed or abandoned.",
            });
        }
        return record->state;
    }

    Result<PipelineHandle> VulkanPipelineManager::waitPipelineCompilation(PipelineCompilationHandle compilation)
    {
        std::shared_ptr<CompilationRecord> record;
        {
            const std::scoped_lock lock(mutex_);
            const auto* found = compilations_.find(compilation);
            if (found == nullptr || *found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline compilation handle is invalid or already consumed.",
                });
            }
            record = *found;
        }

        Result<PipelineHandle> result = std::unexpected(Error{
            ErrorCode::invalidState,
            0,
            "The pipeline compilation request has no result.",
        });
        {
            std::unique_lock recordLock(record->mutex);
            record->condition.wait(recordLock, [&record]() {
                return record->result.has_value() || record->consumed || record->abandoned;
            });
            if (record->consumed || record->abandoned || !record->result.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The pipeline compilation request was already consumed or abandoned.",
                });
            }
            result = std::move(*record->result);
            record->consumed = true;
        }
        record->condition.notify_all();
        {
            const std::scoped_lock lock(mutex_);
            static_cast<void>(compilations_.erase(compilation));
        }
        return result;
    }

    Status VulkanPipelineManager::destroyPipelineCompilation(PipelineCompilationHandle compilation)
    {
        std::shared_ptr<CompilationRecord> record;
        {
            const std::scoped_lock lock(mutex_);
            const auto* found = compilations_.find(compilation);
            if (found == nullptr || *found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline compilation handle is invalid or stale.",
                });
            }
            record = *found;
            static_cast<void>(compilations_.erase(compilation));
        }

        std::optional<PipelineHandle> abandonedPipeline;
        {
            const std::scoped_lock recordLock(record->mutex);
            if (record->consumed)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The pipeline compilation request has already been consumed.",
                });
            }
            record->abandoned = true;
            if (record->result.has_value() && record->result->has_value())
            {
                abandonedPipeline = **record->result;
            }
        }
        record->condition.notify_all();
        if (abandonedPipeline.has_value())
        {
            return destroyPipeline(*abandonedPipeline);
        }
        return {};
    }

    Status VulkanPipelineManager::destroyPipeline(PipelineHandle pipeline)
    {
        PipelineRecord retired;
        {
            const std::scoped_lock lock(mutex_);
            auto* found = pipelines_.find(pipeline);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline handle is invalid or stale.",
                });
            }
            retired = std::move(*found);
            static_cast<void>(pipelines_.erase(pipeline));
        }
        return {};
    }

    Result<std::shared_ptr<const VulkanNativePipeline>> VulkanPipelineManager::nativePipeline(
        PipelineHandle pipeline) const
    {
        const std::shared_lock lock(mutex_);
        const auto* found = pipelines_.find(pipeline);
        if (found == nullptr || found->native == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline handle is invalid or stale.",
            });
        }
        return found->native;
    }

} // namespace rachis::vulkan
