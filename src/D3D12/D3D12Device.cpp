#include "Common/MonotonicIdentitySource.h"
#include "D3D12/D3D12Binding.h"
#include "D3D12/D3D12DebugMarker.h"
#include "D3D12/D3D12Diagnostics.h"
#include "D3D12/D3D12Format.h"
#include "D3D12/D3D12Mipmap.h"
#include "D3D12/D3D12NativeHelpers.h"
#include "D3D12/D3D12Pipeline.h"
#include "D3D12/D3D12Query.h"
#include "D3D12/D3D12Queue.h"
#include "D3D12/D3D12Resource.h"
#include "D3D12/D3D12RuntimePolicies.h"
#include "D3D12/D3D12Sampler.h"
#include "D3D12/D3D12Swapchain.h"
#include "D3D12/D3D12TestHooks.h"
#include "D3D12/D3D12VertexInput.h"
#include "D3D12/D3D12View.h"
#include "Internal/D3D12Factory.h"
#include "Internal/DeviceIdentity.h"
#include "Internal/TextureFormatRules.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <Windows.h>

#ifdef DeviceCapabilities
#undef DeviceCapabilities
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace rachis::d3d12
{
    namespace
    {

        using Microsoft::WRL::ComPtr;

        static_assert(maxViewportCount == D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE);

        [[nodiscard]] DeviceFaultStatus deviceFaultStatusFromNativeReason(std::int64_t nativeReason) noexcept
        {
            switch (static_cast<HRESULT>(nativeReason))
            {
            case DXGI_ERROR_DEVICE_REMOVED:
                return DeviceFaultStatus::removed;
            case DXGI_ERROR_DEVICE_RESET:
                return DeviceFaultStatus::reset;
            case DXGI_ERROR_DEVICE_HUNG:
                return DeviceFaultStatus::hung;
            case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
                return DeviceFaultStatus::driverInternalError;
            case DXGI_ERROR_INVALID_CALL:
                return DeviceFaultStatus::invalidCall;
            case E_OUTOFMEMORY:
                return DeviceFaultStatus::outOfMemory;
            default:
                return DeviceFaultStatus::unknown;
            }
        }

        [[nodiscard]] AdapterId adapterId(LUID luid) noexcept
        {
            return AdapterId{
                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(luid.HighPart)) << 32U) |
                    static_cast<std::uint64_t>(luid.LowPart),
            };
        }

        [[nodiscard]] std::string utf8Name(const wchar_t* name)
        {
            const int length = WideCharToMultiByte(CP_UTF8, 0, name, -1, nullptr, 0, nullptr, nullptr);
            if (length <= 1)
            {
                return {};
            }

            std::string result(static_cast<std::size_t>(length), '\0');
            WideCharToMultiByte(CP_UTF8, 0, name, -1, result.data(), length, nullptr, nullptr);
            result.pop_back();
            return result;
        }

        [[nodiscard]] AdapterInfo adapterInfo(const DXGI_ADAPTER_DESC1& desc)
        {
            return AdapterInfo{
                .id = adapterId(desc.AdapterLuid),
                .name = utf8Name(desc.Description),
                .type = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 ? AdapterType::software : AdapterType::unknown,
                .vendorId = desc.VendorId,
                .deviceId = desc.DeviceId,
                .dedicatedVideoMemory = desc.DedicatedVideoMemory,
                .sharedSystemMemory = desc.SharedSystemMemory,
                .backend = Backend::d3d12,
            };
        }

        [[nodiscard]] HRESULT createNativeDevice(IDXGIAdapter1* adapter, REFIID riid, void** device) noexcept
        {
            if (const auto observer = test::nativeDeviceCreationObserver().load(std::memory_order_acquire);
                observer != nullptr)
            {
                observer();
            }
            return D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, riid, device);
        }

        [[nodiscard]] HRESULT probeRequiredFeatureLevel(IDXGIAdapter1* adapter) noexcept
        {
            return createNativeDevice(adapter, __uuidof(ID3D12Device), nullptr);
        }

        [[nodiscard]] bool isCapabilityIndependentProbeFailure(HRESULT result) noexcept
        {
            return isD3D12DeviceLossHresult(static_cast<std::int32_t>(result)) || result == E_OUTOFMEMORY;
        }

        [[nodiscard]] Result<ComPtr<IDXGIFactory6>> createFactory(bool debug)
        {
            ComPtr<IDXGIFactory6> factory;
            const auto flags = debug ? DXGI_CREATE_FACTORY_DEBUG : 0U;
            const HRESULT result = CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory));
            if (FAILED(result))
            {
                return std::unexpected(nativeError(ErrorCode::backendUnavailable, result, "CreateDXGIFactory2"));
            }
            return factory;
        }

        [[nodiscard]] constexpr bool isBasicRenderDriver(const DXGI_ADAPTER_DESC1& desc) noexcept
        {
            return desc.VendorId == 0x1414 && desc.DeviceId == 0x8c;
        }

        [[nodiscard]] Result<std::vector<ComPtr<IDXGIAdapter1>>> enumerateHardwareAdapters(
            IDXGIFactory6& factory, DXGI_GPU_PREFERENCE preference)
        {
            std::vector<ComPtr<IDXGIAdapter1>> adapters;
            for (UINT index = 0;; ++index)
            {
                ComPtr<IDXGIAdapter1> adapter;
                const HRESULT result = factory.EnumAdapterByGpuPreference(index, preference, IID_PPV_ARGS(&adapter));
                if (result == DXGI_ERROR_NOT_FOUND)
                {
                    break;
                }
                if (FAILED(result))
                {
                    return std::unexpected(
                        nativeError(ErrorCode::nativeFailure, result, "IDXGIFactory6::EnumAdapterByGpuPreference"));
                }

                DXGI_ADAPTER_DESC1 desc{};
                if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 ||
                    isBasicRenderDriver(desc))
                {
                    continue;
                }
                const HRESULT probe = probeRequiredFeatureLevel(adapter.Get());
                if (FAILED(probe) && !isCapabilityIndependentProbeFailure(probe))
                {
                    continue;
                }
                adapters.push_back(std::move(adapter));
            }
            return adapters;
        }

        [[nodiscard]] Result<ComPtr<IDXGIAdapter1>> requireWarpFeatureLevel(ComPtr<IDXGIAdapter1> adapter)
        {
            const HRESULT probe = probeRequiredFeatureLevel(adapter.Get());
            if (FAILED(probe))
            {
                if (isCapabilityIndependentProbeFailure(probe))
                {
                    return std::unexpected(nativeError(
                        ErrorCode::deviceCreationFailed, probe, "D3D12CreateDevice(WARP feature-level probe)"));
                }
                return std::unexpected(nativeError(ErrorCode::unsupported,
                                                   probe,
                                                   "The WARP adapter does not support D3D feature level 12.0: "
                                                   "D3D12CreateDevice(WARP feature-level probe)"));
            }
            return adapter;
        }

        [[nodiscard]] Result<ComPtr<IDXGIAdapter1>> warpAdapter(IDXGIFactory6& factory)
        {
            ComPtr<IDXGIAdapter1> adapter;
            const HRESULT result = factory.EnumWarpAdapter(IID_PPV_ARGS(&adapter));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(ErrorCode::adapterNotFound, result, "IDXGIFactory6::EnumWarpAdapter"));
            }
            return requireWarpFeatureLevel(std::move(adapter));
        }

        [[nodiscard]] Result<ComPtr<IDXGIAdapter1>> selectAdapter(IDXGIFactory6& factory, const DeviceDesc& desc)
        {
            if (desc.adapterPreference == AdapterPreference::warp)
            {
                auto warp = warpAdapter(factory);
                if (!warp)
                {
                    return std::unexpected(std::move(warp.error()));
                }
                if (desc.adapterId)
                {
                    DXGI_ADAPTER_DESC1 nativeDesc{};
                    (*warp)->GetDesc1(&nativeDesc);
                    if (adapterId(nativeDesc.AdapterLuid) != *desc.adapterId)
                    {
                        return std::unexpected(Error{
                            ErrorCode::adapterNotFound,
                            0,
                            "The requested adapter ID does not identify the WARP adapter.",
                        });
                    }
                }
                return std::move(*warp);
            }

            const auto preference = desc.adapterPreference == AdapterPreference::minimumPower
                                        ? DXGI_GPU_PREFERENCE_MINIMUM_POWER
                                        : DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
            auto adapters = enumerateHardwareAdapters(factory, preference);
            if (!adapters)
            {
                return std::unexpected(std::move(adapters.error()));
            }

            for (auto& adapter : *adapters)
            {
                if (!desc.adapterId)
                {
                    return std::move(adapter);
                }

                DXGI_ADAPTER_DESC1 nativeDesc{};
                if (SUCCEEDED(adapter->GetDesc1(&nativeDesc)) && adapterId(nativeDesc.AdapterLuid) == *desc.adapterId)
                {
                    return std::move(adapter);
                }
            }

            if (desc.adapterId)
            {
                ComPtr<IDXGIAdapter1> warp;
                DXGI_ADAPTER_DESC1 nativeDesc{};
                if (SUCCEEDED(factory.EnumWarpAdapter(IID_PPV_ARGS(&warp))) && SUCCEEDED(warp->GetDesc1(&nativeDesc)) &&
                    adapterId(nativeDesc.AdapterLuid) == *desc.adapterId)
                {
                    return requireWarpFeatureLevel(std::move(warp));
                }
            }

            return std::unexpected(Error{
                ErrorCode::adapterNotFound,
                0,
                desc.adapterId ? "No D3D12 adapter matches the requested adapter ID."
                               : "No D3D12 hardware adapter supporting feature level 12.0 was found.",
            });
        }

        [[nodiscard]] ApiVersion featureLevelVersion(D3D_FEATURE_LEVEL level) noexcept
        {
            switch (level)
            {
            case D3D_FEATURE_LEVEL_12_2:
                return {12, 2};
            case D3D_FEATURE_LEVEL_12_1:
                return {12, 1};
            default:
                return {12, 0};
            }
        }

        [[nodiscard]] ApiVersion shaderModelVersion(D3D_SHADER_MODEL model) noexcept
        {
            const auto value = static_cast<std::uint32_t>(model);
            return {
                static_cast<std::uint16_t>((value >> 4U) & 0xFU),
                static_cast<std::uint16_t>(value & 0xFU),
            };
        }

        using CapabilityFormats = format::Capabilities;

        [[nodiscard]] constexpr CapabilityFormats nativeCapabilityFormats(Format value) noexcept
        {
            return format::capabilities(value);
        }

        [[nodiscard]] constexpr Format alternateViewFormat(Format value) noexcept
        {
            return format::alternateView(value);
        }

        [[nodiscard]] constexpr DXGI_FORMAT mutableResourceFormat(Format value) noexcept
        {
            return format::mutableResource(value);
        }

        template<typename Interface>
        [[nodiscard]] bool supportsInterface(ID3D12Device& device) noexcept
        {
            ComPtr<Interface> queried;
            return SUCCEEDED(device.QueryInterface(IID_PPV_ARGS(&queried)));
        }

        [[nodiscard]] D3D12_FEATURE_DATA_FORMAT_SUPPORT queryFormatSupport(ID3D12Device& device,
                                                                           DXGI_FORMAT nativeFormat) noexcept
        {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{
                .Format = nativeFormat,
                .Support1 = {},
                .Support2 = {},
            };
            if (nativeFormat == DXGI_FORMAT_UNKNOWN ||
                FAILED(device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))))
            {
                return D3D12_FEATURE_DATA_FORMAT_SUPPORT{};
            }
            return support;
        }

        struct CapabilityFormatSupport
        {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT resource{};
            D3D12_FEATURE_DATA_FORMAT_SUPPORT sampled{};
            D3D12_FEATURE_DATA_FORMAT_SUPPORT storage{};
            D3D12_FEATURE_DATA_FORMAT_SUPPORT attachment{};
        };

        [[nodiscard]] std::optional<CapabilityFormatSupport> queryCapabilityFormatSupport(
            ID3D12Device& device, const CapabilityFormats& formats) noexcept
        {
            const auto resourceSupport = queryFormatSupport(device, formats.resource);
            if (resourceSupport.Format == DXGI_FORMAT_UNKNOWN)
            {
                return std::nullopt;
            }
            return CapabilityFormatSupport{
                .resource = resourceSupport,
                .sampled = queryFormatSupport(device, formats.sampled),
                .storage = queryFormatSupport(device, formats.storage),
                .attachment = queryFormatSupport(device, formats.attachment),
            };
        }

        [[nodiscard]] bool supportsViewFormatReinterpretation(
            ID3D12Device& device, Format format, const D3D12_FEATURE_DATA_FORMAT_SUPPORT& resourceSupport) noexcept
        {
            const auto alternateFormat = alternateViewFormat(format);
            const auto typelessFormat = mutableResourceFormat(format);
            if (alternateFormat == Format::unknown || typelessFormat == DXGI_FORMAT_UNKNOWN)
            {
                return false;
            }

            const auto alternateSupport = queryFormatSupport(device, nativeCapabilityFormats(alternateFormat).resource);
            const auto typelessSupport = queryFormatSupport(device, typelessFormat);
            constexpr auto textureDimensions = static_cast<std::uint32_t>(D3D12_FORMAT_SUPPORT1_TEXTURE1D) |
                                               static_cast<std::uint32_t>(D3D12_FORMAT_SUPPORT1_TEXTURE2D) |
                                               static_cast<std::uint32_t>(D3D12_FORMAT_SUPPORT1_TEXTURE3D) |
                                               static_cast<std::uint32_t>(D3D12_FORMAT_SUPPORT1_TEXTURECUBE);
            const auto commonDimensions = static_cast<std::uint32_t>(resourceSupport.Support1) &
                                          static_cast<std::uint32_t>(alternateSupport.Support1) &
                                          static_cast<std::uint32_t>(typelessSupport.Support1) & textureDimensions;
            return commonDimensions != 0;
        }

        [[nodiscard]] FormatFeature textureDimensionFeatures(const D3D12_FEATURE_DATA_FORMAT_SUPPORT& resourceSupport,
                                                             bool depthStencil) noexcept
        {
            auto features = FormatFeature::none;
            if ((resourceSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE1D) != 0)
            {
                features |= FormatFeature::texture1D;
            }
            if ((resourceSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) != 0)
            {
                features |= FormatFeature::texture2D;
            }
            if ((resourceSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE3D) != 0)
            {
                features |= FormatFeature::texture3D;
            }
            if ((resourceSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURECUBE) != 0)
            {
                features |= FormatFeature::cube;
            }
            return depthStencil ? detail::depthTextureFormatFeatures(features) : features;
        }

        [[nodiscard]] FormatFeature shaderAccessFeatures(
            const D3D12_FEATURE_DATA_FORMAT_SUPPORT& sampledSupport,
            const D3D12_FEATURE_DATA_FORMAT_SUPPORT& storageSupport) noexcept
        {
            auto features = FormatFeature::none;
            if (format::supportsSampledUsage(sampledSupport.Support1))
            {
                features |= FormatFeature::sampled;
            }
            if ((sampledSupport.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE) != 0)
            {
                features |= FormatFeature::filterable;
            }
            if ((storageSupport.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD) != 0)
            {
                features |= FormatFeature::storageRead;
            }
            if ((storageSupport.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0)
            {
                features |= FormatFeature::storageWrite;
            }
            return features;
        }

        [[nodiscard]] FormatFeature attachmentFeatures(const D3D12_FEATURE_DATA_FORMAT_SUPPORT& attachmentSupport,
                                                       bool depthStencil) noexcept
        {
            auto features = FormatFeature::none;
            if (!depthStencil && (attachmentSupport.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET) != 0)
            {
                features |= FormatFeature::colorAttachment;
            }
            if (depthStencil && (attachmentSupport.Support1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL) != 0)
            {
                features |= FormatFeature::depthStencilAttachment;
            }
            if ((attachmentSupport.Support1 & D3D12_FORMAT_SUPPORT1_BLENDABLE) != 0)
            {
                features |= FormatFeature::blendable;
            }
            if ((attachmentSupport.Support2 & D3D12_FORMAT_SUPPORT2_OUTPUT_MERGER_LOGIC_OP) != 0)
            {
                features |= FormatFeature::logicOperation;
            }
            return features;
        }

        void assignMultisampleResolveModes(ID3D12Device& device,
                                           Format format,
                                           const CapabilityFormats& formats,
                                           const CapabilityFormatSupport& support,
                                           FormatProperties& properties) noexcept
        {
            const bool hasMultisampling = supportsSampleCount(properties.sampleCounts, 2) ||
                                          supportsSampleCount(properties.sampleCounts, 4) ||
                                          supportsSampleCount(properties.sampleCounts, 8);
            if (!hasMultisampling)
            {
                return;
            }

            const bool isDepthStencilAttachment = hasAny(properties.features, FormatFeature::depthStencilAttachment);
            if (!formats.depthStencil && (support.attachment.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE) != 0)
            {
                properties.colorResolveModes = ResolveModeMask::average;
            }
            if (format == Format::d32Float && (support.sampled.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_LOAD) != 0 &&
                isDepthStencilAttachment)
            {
                properties.depthResolveModes = ResolveModeMask::minimum | ResolveModeMask::maximum;
            }
            if (format == Format::d24UnormS8Uint && isDepthStencilAttachment &&
                supportsInterface<ID3D12Device1>(device))
            {
                properties.depthResolveModes = ResolveModeMask::minimum | ResolveModeMask::maximum;
                properties.stencilResolveModes = ResolveModeMask::minimum | ResolveModeMask::maximum;
            }
            if (properties.colorResolveModes != ResolveModeMask::none ||
                properties.depthResolveModes != ResolveModeMask::none ||
                properties.stencilResolveModes != ResolveModeMask::none)
            {
                properties.features |= FormatFeature::multisampleResolve;
            }
        }

        [[nodiscard]] FormatProperties d3d12FormatProperties(ID3D12Device& device, Format format) noexcept
        {
            const auto formats = nativeCapabilityFormats(format);
            if (formats.resource == DXGI_FORMAT_UNKNOWN)
            {
                return {};
            }
            const auto support = queryCapabilityFormatSupport(device, formats);
            if (!support)
            {
                return {};
            }

            FormatProperties result;
            if (supportsViewFormatReinterpretation(device, format, support->resource))
            {
                result.features |= FormatFeature::viewFormatReinterpretation;
            }
            result.features |= textureDimensionFeatures(support->resource, formats.depthStencil);
            if (hasAny(result.features, FormatFeature::texture2D))
            {
                result.sampleCounts = SampleCountMask::x1;
            }
            result.features |= shaderAccessFeatures(support->sampled, support->storage);
            result.features |= attachmentFeatures(support->attachment, formats.depthStencil);
            if (hasAny(result.features,
                       FormatFeature::texture1D | FormatFeature::texture2D | FormatFeature::texture3D |
                           FormatFeature::cube))
            {
                result.features |= FormatFeature::copySource | FormatFeature::copyDestination;
            }
            if (hasAny(result.features, FormatFeature::colorAttachment | FormatFeature::depthStencilAttachment))
            {
                result.sampleCounts = format::attachmentSampleCounts(device, formats.attachment);
            }
            assignMultisampleResolveModes(device, format, formats, *support, result);
            return result;
        }

        [[nodiscard]] DeviceLimits d3d12DeviceLimits(const AdapterInfo& adapter) noexcept
        {
            constexpr std::uint64_t bytesPerMegabyte = 1024ULL * 1024U;
            constexpr std::uint64_t minimumResourceBytes =
                static_cast<std::uint64_t>(D3D12_REQ_RESOURCE_SIZE_IN_MEGABYTES_EXPRESSION_A_TERM) * bytesPerMegabyte;
            constexpr std::uint64_t maximumResourceBytes = 2ULL * 1024U * 1024U * 1024U;
            const auto dedicatedMemoryQuarter = adapter.dedicatedVideoMemory / 4U;

            return DeviceLimits{
                .maxTextureDimension1D = D3D12_REQ_TEXTURE1D_U_DIMENSION,
                .maxTextureDimension2D = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
                .maxTextureDimension3D = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
                .maxTextureArrayLayers = D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION,
                .maxFramebufferLayers = 1,
                .maxTextureMipLevels = D3D12_REQ_MIP_LEVELS,
                .maxComputeWorkgroupInvocations = D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP,
                .maxComputeWorkgroupSize = {D3D12_CS_THREAD_GROUP_MAX_X,
                                            D3D12_CS_THREAD_GROUP_MAX_Y,
                                            D3D12_CS_THREAD_GROUP_MAX_Z},
                .maxComputeWorkgroupCount = {D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION,
                                             D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION,
                                             D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION},
                .maxBufferSize = std::max(minimumResourceBytes, std::min(maximumResourceBytes, dedicatedMemoryQuarter)),
                .maxUniformBufferSize = static_cast<std::uint64_t>(D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT) * 16U,
                .minUniformBufferOffsetAlignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
                .maxSamplerAnisotropy = D3D12_REQ_MAXANISOTROPY,
                .maxSamplerLodBias = D3D12_MIP_LOD_BIAS_MAX,
                .maxColorAttachments = D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT,
                .maxViewports = D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE,
                .maxVertexBuffers = maxVertexBuffers,
                .maxVertexAttributes = maxVertexAttributes,
                .maxBindingSets = maxBindingSets,
                .maxPushConstantBytes = maxPushConstantBytes,
                .maxBindingPoolResourceDescriptors = bindingPoolResourceDescriptorBudget,
                .maxBindingSamplerDescriptors = bindingSamplerDescriptorBudget,
                .maxClipDistances = D3D12_CLIP_OR_CULL_DISTANCE_COUNT,
                .maxCullDistances = D3D12_CLIP_OR_CULL_DISTANCE_COUNT,
                .maxCombinedClipAndCullDistances = D3D12_CLIP_OR_CULL_DISTANCE_COUNT,
            };
        }

        [[nodiscard]] VertexInputCapabilities d3d12VertexInputCapabilities(ID3D12Device& device) noexcept
        {
            VertexInputCapabilities capabilities{
                .maxBufferStride = D3D12_SO_BUFFER_MAX_STRIDE_IN_BYTES,
                .maxAttributeOffset = D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES - 1U,
                .maxInstanceStepRate = UINT_MAX,
            };
            for (std::uint32_t index = 0; index < vertexFormatCount; ++index)
            {
                const auto format = static_cast<VertexFormat>(index);
                D3D12_FEATURE_DATA_FORMAT_SUPPORT support{
                    .Format = nativeVertexFormat(format),
                    .Support1 = {},
                    .Support2 = {},
                };
                if (support.Format != DXGI_FORMAT_UNKNOWN &&
                    SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) &&
                    (support.Support1 & D3D12_FORMAT_SUPPORT1_IA_VERTEX_BUFFER) != 0)
                {
                    capabilities.supportedFormats |= std::uint64_t{1} << index;
                }
            }
            return capabilities;
        }

        [[nodiscard]] DeviceCapabilities guaranteedCapabilities() noexcept
        {
            DeviceCapabilities capabilities{};
            capabilities.coordinateConventions = {
                .ndcYAxis = NdcYAxisDirection::up,
                .ndcDepthRange = NdcDepthRange::zeroToOne,
                .textureOrigin = TextureOrigin::upperLeft,
            };
            capabilities.indirectDispatch = true;
            capabilities.multiDispatchIndirect = true;
            capabilities.dispatchIndirectCount = true;
            capabilities.indirectDraws = true;
            capabilities.indirectDrawBaseInstance = true;
            capabilities.multiDrawIndirect = true;
            capabilities.drawIndirectCount = true;
            capabilities.multiViewport = true;
            capabilities.occlusionQueries = true;
            capabilities.pipelineStatisticsQueries = true;
            capabilities.streamOutput = true;
            capabilities.streamOutputRasterizationDisable = true;
            capabilities.streamOutputStatisticsQueries = true;
            capabilities.conditionalExecution = true;
            capabilities.partiallyBoundBindingArrays = true;
            capabilities.persistentMapping = true;
            capabilities.mappedMemory = {
                .hostCoherent = true,
                .nonCoherentAtomSize = 1,
            };
            capabilities.sharedTextureHandles = true;
            capabilities.sharedBufferHandles = true;
            capabilities.sharedTimelineHandles = true;

            capabilities.dualSourceBlending = true;

            capabilities.fragmentPrimitiveId = true;

            capabilities.geometryShaders = true;

            capabilities.tessellationShaders = true;
            capabilities.tessellation.maxPatchControlPoints = 32;
            capabilities.wireframeRasterization = true;
            capabilities.unclippedDepth = true;
            capabilities.primitiveRestart = true;

            capabilities.samplerMinMaxReduction = true;
            capabilities.textureViewComponentSwizzle = true;
            capabilities.debugMarkers = RACHIS_ENABLE_DEBUG_MARKERS != 0;
            return capabilities;
        }

        struct DeviceInterfaceSupport
        {
            bool device1 = false;
            bool device2 = false;
            bool device8 = false;
            bool device3 = false;
        };

        [[nodiscard]] DeviceInterfaceSupport queryDeviceInterfaceSupport(ID3D12Device& device) noexcept
        {
            return DeviceInterfaceSupport{
                .device1 = supportsInterface<ID3D12Device1>(device),
                .device2 = supportsInterface<ID3D12Device2>(device),
                .device8 = supportsInterface<ID3D12Device8>(device),
                .device3 = supportsInterface<ID3D12Device3>(device),
            };
        }

        [[nodiscard]] ApiVersion queryFeatureLevel(ID3D12Device& device) noexcept
        {
            std::array requestedLevels{
                D3D_FEATURE_LEVEL_12_2,
                D3D_FEATURE_LEVEL_12_1,
                D3D_FEATURE_LEVEL_12_0,
            };
            D3D12_FEATURE_DATA_FEATURE_LEVELS levels{
                .NumFeatureLevels = static_cast<UINT>(requestedLevels.size()),
                .pFeatureLevelsRequested = requestedLevels.data(),
                .MaxSupportedFeatureLevel = D3D_FEATURE_LEVEL_12_0,
            };
            if (FAILED(device.CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &levels, sizeof(levels))))
            {
                return {12, 0};
            }
            return featureLevelVersion(levels.MaxSupportedFeatureLevel);
        }

        [[nodiscard]] ApiVersion queryShaderModel(ID3D12Device& device) noexcept
        {
            D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{D3D_SHADER_MODEL_6_6};
            if (FAILED(device.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))))
            {
                return {};
            }
            return shaderModelVersion(shaderModel.HighestShaderModel);
        }

        void detectBarycentricsAndMultiview(ID3D12Device& device,
                                            const DeviceInterfaceSupport& interfaces,
                                            DeviceCapabilities& capabilities) noexcept
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS3 options3{};
            if (FAILED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &options3, sizeof(options3))))
            {
                return;
            }

            capabilities.fragmentBarycentricCoordinates =
                options3.BarycentricsSupported != FALSE && capabilities.shaderModel >= ApiVersion{6, 1};
            capabilities.multiviewRendering = options3.ViewInstancingTier != D3D12_VIEW_INSTANCING_TIER_NOT_SUPPORTED &&
                                              capabilities.shaderModel >= ApiVersion{6, 1} && interfaces.device2;
            if (capabilities.multiviewRendering)
            {
                capabilities.maxMultiviewViewCount = D3D12_MAX_VIEW_INSTANCE_COUNT;
                capabilities.limits.maxFramebufferLayers = D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
            }
        }

        void detectShaderInt64AndSubgroupOperations(ID3D12Device& device, DeviceCapabilities& capabilities) noexcept
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
            if (FAILED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1))))
            {
                return;
            }

            const bool supportsShaderModel60 = capabilities.shaderModel >= ApiVersion{6, 0};
            capabilities.shaderInt64 = options1.Int64ShaderOps != FALSE && supportsShaderModel60;
            if (options1.WaveOps != FALSE && supportsShaderModel60 && options1.WaveLaneCountMin > 0 &&
                options1.WaveLaneCountMax >= options1.WaveLaneCountMin)
            {
                capabilities.subgroupOperations = true;
                capabilities.subgroup = {
                    .minSize = options1.WaveLaneCountMin,
                    .maxSize = options1.WaveLaneCountMax,
                };
            }
        }

        void detectDepthBoundsTest(ID3D12Device& device,
                                   const DeviceInterfaceSupport& interfaces,
                                   DeviceCapabilities& capabilities) noexcept
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS2 options2{};
            if (SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, &options2, sizeof(options2))))
            {
                capabilities.depthBoundsTest = options2.DepthBoundsTestSupported != FALSE && interfaces.device2;
            }
        }

        void detectNative16BitShaderOperations(ID3D12Device& device, DeviceCapabilities& capabilities) noexcept
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4{};
            if (FAILED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options4, sizeof(options4))))
            {
                return;
            }

            const bool supportsNative16BitOperations =
                options4.Native16BitShaderOpsSupported != FALSE && capabilities.shaderModel >= ApiVersion{6, 2};

            capabilities.shaderFloat16 = supportsNative16BitOperations;
            capabilities.shaderInt16 = supportsNative16BitOperations;
        }

        void detectRayTracing(ID3D12Device& device, DeviceCapabilities& capabilities) noexcept
        {
            const bool supportsDevice5 = supportsInterface<ID3D12Device5>(device);
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5{};
            const bool supportsRayTracingTier =
                supportsDevice5 &&
                SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5))) &&
                options5.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
            if (!supportsRayTracingTier)
            {
                return;
            }

            capabilities.rayTracing = capabilities.shaderModel >= ApiVersion{6, 3};
            capabilities.indirectRayDispatch =
                capabilities.rayTracing && options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
            capabilities.rayQuery =
                options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1 && capabilities.shaderModel >= ApiVersion{6, 5};
            capabilities.rayTracingLimits = {
                .tier = options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1 ? 11U : 10U,
                .maxGeometriesPerBottomLevel = D3D12_RAYTRACING_MAX_GEOMETRIES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE,
                .maxInstancesPerTopLevel = D3D12_RAYTRACING_MAX_INSTANCES_PER_TOP_LEVEL_ACCELERATION_STRUCTURE,
                .maxPrimitivesPerBottomLevel = D3D12_RAYTRACING_MAX_PRIMITIVES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE,
                .shaderGroupHandleSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
                .shaderGroupHandleAlignment = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT,
                .shaderGroupBaseAlignment = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT,
                .maxShaderGroupStride = D3D12_RAYTRACING_MAX_SHADER_RECORD_STRIDE,
                .maxRecursionDepth = D3D12_RAYTRACING_MAX_DECLARABLE_TRACE_RECURSION_DEPTH,
                .maxAttributeSize = D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES,
                .maxDispatchInvocations = D3D12_RAYTRACING_MAX_RAY_GENERATION_SHADER_THREADS,
            };
        }

        void detectVariableRateShading(ID3D12Device& device, DeviceCapabilities& capabilities) noexcept
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS6 options6{};
            if (FAILED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &options6, sizeof(options6))))
            {
                return;
            }

            capabilities.variableRateShading =
                options6.VariableShadingRateTier != D3D12_VARIABLE_SHADING_RATE_TIER_NOT_SUPPORTED;
            capabilities.additionalShadingRates =
                capabilities.variableRateShading && options6.AdditionalShadingRatesSupported != FALSE;
            capabilities.variableRateShadingTier2 =
                options6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_2 &&
                options6.ShadingRateImageTileSize != 0;
            capabilities.perPrimitiveShadingRateWithViewportIndexing =
                capabilities.variableRateShadingTier2 &&
                options6.PerPrimitiveShadingRateSupportedWithViewportIndexing != FALSE;
            capabilities.shadingRateImageTileSize =
                capabilities.variableRateShadingTier2 ? options6.ShadingRateImageTileSize : 0;
        }

        void detectMeshShadersAndSamplerFeedback(ID3D12Device& device,
                                                 const DeviceInterfaceSupport& interfaces,
                                                 DeviceCapabilities& capabilities) noexcept
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7{};
            if (FAILED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7))))
            {
                return;
            }

            const bool supportsMeshShaders = options7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1 &&
                                             capabilities.shaderModel >= ApiVersion{6, 5} && interfaces.device2;
            capabilities.samplerFeedback = options7.SamplerFeedbackTier >= D3D12_SAMPLER_FEEDBACK_TIER_1_0 &&
                                           capabilities.shaderModel >= ApiVersion{6, 5} && interfaces.device8;
            capabilities.meshShaders = supportsMeshShaders;
            capabilities.taskShaders = supportsMeshShaders;
            if (supportsMeshShaders)
            {
                capabilities.meshShader = {
                    .maxTaskWorkgroupInvocations = 128,
                    .maxTaskPayloadBytes = 16U * 1024U,
                    .maxMeshWorkgroupInvocations = 128,
                    .maxMeshOutputVertices = 256,
                    .maxMeshOutputPrimitives = 256,
                    .maxMeshOutputMemoryBytes = 32U * 1024U,
                    .maxWorkgroupCount = {D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION,
                                          D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION,
                                          D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION},
                    .maxTotalWorkgroups = 4'194'303,
                };
            }
        }

        void detectInt64AtomicsAndMeshPipelineStatistics(ID3D12Device& device,
                                                         DeviceCapabilities& capabilities) noexcept
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS9 options9{};
            if (FAILED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &options9, sizeof(options9))))
            {
                return;
            }

            const bool supportsShaderModel66 = capabilities.shaderModel >= ApiVersion{6, 6};
            capabilities.shaderInt64AtomicGroupShared = capabilities.shaderInt64 && supportsShaderModel66 &&
                                                        options9.AtomicInt64OnGroupSharedSupported != FALSE;
            capabilities.shaderInt64AtomicTypedResource = capabilities.shaderInt64 && supportsShaderModel66 &&
                                                          options9.AtomicInt64OnTypedResourceSupported != FALSE;
            capabilities.meshPipelineStatisticsQueries =
                capabilities.meshShaders && options9.MeshShaderPipelineStatsSupported != FALSE;
        }

        void detectBaseOptionsCapabilities(ID3D12Device& device, DeviceCapabilities& capabilities) noexcept
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
            if (FAILED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))))
            {
                return;
            }

            capabilities.logicOperations = options.OutputMergerLogicOp != FALSE;
            capabilities.rasterizerOrderedStorageAccess = options.ROVsSupported != FALSE;
            capabilities.resourceBindingTier = static_cast<std::uint32_t>(options.ResourceBindingTier);
            capabilities.resourceHeapTier = static_cast<std::uint32_t>(options.ResourceHeapTier);
            capabilities.sparseBuffers = options.TiledResourcesTier >= D3D12_TILED_RESOURCES_TIER_2;
            capabilities.sparseTextures = options.TiledResourcesTier >= D3D12_TILED_RESOURCES_TIER_2;
            if (options.VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation != FALSE)
            {
                capabilities.layeredRenderingModes = LayeredRenderingMode::vertexStage;
                capabilities.limits.maxFramebufferLayers = D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
            }
            capabilities.conservativeRasterization =
                options.ConservativeRasterizationTier != D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED;
            capabilities.variableDescriptorCountBindingArrays =
                options.ResourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_3;
            if (capabilities.variableDescriptorCountBindingArrays)
            {
                capabilities.limits.maxVariableResourceDescriptors = bindingPoolResourceDescriptorBudget;
                capabilities.limits.maxVariableBufferDescriptors = bindingPoolResourceDescriptorBudget;
                capabilities.limits.maxVariableSamplerDescriptors = bindingSamplerDescriptorBudget;
            }
        }

        void detectEnhancedBarriers(ID3D12Device& device, DeviceCapabilities& capabilities) noexcept
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12{};
            if (SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12, sizeof(options12))))
            {
                capabilities.enhancedBarriers = options12.EnhancedBarriersSupported != FALSE;
            }
        }

        void detectPipelineStateExtensions(ID3D12Device& device, DeviceCapabilities& capabilities) noexcept
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS14 options14{};
            if (SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS14, &options14, sizeof(options14))))
            {
                capabilities.independentFrontAndBackStencilReference =
                    options14.IndependentFrontAndBackStencilRefMaskSupported != FALSE;
            }

            D3D12_FEATURE_DATA_D3D12_OPTIONS15 options15{};
            if (SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS15, &options15, sizeof(options15))))
            {
                capabilities.triangleFans = options15.TriangleFanSupported != FALSE;
            }

            D3D12_FEATURE_DATA_D3D12_OPTIONS16 options16{};
            if (SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &options16, sizeof(options16))))
            {
                capabilities.dynamicDepthBias = options16.DynamicDepthBiasSupported != FALSE;
            }
        }

        struct MemoryArchitecture final
        {
            bool unified = false;
            bool cacheCoherent = false;
        };

        [[nodiscard]] std::optional<MemoryArchitecture> queryMemoryArchitecture(ID3D12Device& device) noexcept
        {
            D3D12_FEATURE_DATA_ARCHITECTURE1 architecture{
                .NodeIndex = 0,
                .TileBasedRenderer = {},
                .UMA = {},
                .CacheCoherentUMA = {},
                .IsolatedMMU = {},
            };
            if (FAILED(device.CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &architecture, sizeof(architecture))))
            {
                return std::nullopt;
            }
            return MemoryArchitecture{
                .unified = architecture.UMA != FALSE,
                .cacheCoherent = architecture.CacheCoherentUMA != FALSE,
            };
        }

        void detectUnifiedMemoryArchitecture(ID3D12Device& device, DeviceCapabilities& capabilities) noexcept
        {
            const auto architecture = queryMemoryArchitecture(device).value_or(MemoryArchitecture{});
            capabilities.unifiedMemory = architecture.unified;
            capabilities.cacheCoherentUnifiedMemory = architecture.cacheCoherent;
        }

        [[nodiscard]] AdapterType hardwareAdapterType(ID3D12Device& device) noexcept
        {
            const auto architecture = queryMemoryArchitecture(device);
            if (!architecture)
            {
                return AdapterType::unknown;
            }
            return architecture->unified ? AdapterType::integrated : AdapterType::discrete;
        }

        [[nodiscard]] DeviceCapabilities queryCapabilities(ID3D12Device& device, const AdapterInfo& adapter) noexcept
        {
            auto capabilities = guaranteedCapabilities();
            capabilities.nodeCount = device.GetNodeCount();
            capabilities.vertexInput = d3d12VertexInputCapabilities(device);
            capabilities.limits = d3d12DeviceLimits(adapter);

            const auto interfaces = queryDeviceInterfaceSupport(device);
            capabilities.pipelineCaches = interfaces.device1;
            capabilities.residencyPriorities = interfaces.device1;
            capabilities.manualResidency = interfaces.device3;

            capabilities.featureLevel = queryFeatureLevel(device);
            capabilities.shaderModel = queryShaderModel(device);
            capabilities.mipmapGeneration = capabilities.shaderModel.major >= 6;

            detectBarycentricsAndMultiview(device, interfaces, capabilities);
            detectShaderInt64AndSubgroupOperations(device, capabilities);
            detectDepthBoundsTest(device, interfaces, capabilities);
            detectNative16BitShaderOperations(device, capabilities);
            detectRayTracing(device, capabilities);
            detectVariableRateShading(device, capabilities);
            detectMeshShadersAndSamplerFeedback(device, interfaces, capabilities);
            detectInt64AtomicsAndMeshPipelineStatistics(device, capabilities);
            detectBaseOptionsCapabilities(device, capabilities);
            detectEnhancedBarriers(device, capabilities);
            detectPipelineStateExtensions(device, capabilities);
            detectUnifiedMemoryArchitecture(device, capabilities);

            return capabilities;
        }

        [[nodiscard]] DiagnosticSeverity diagnosticSeverity(D3D12_MESSAGE_SEVERITY severity) noexcept
        {
            switch (severity)
            {
            case D3D12_MESSAGE_SEVERITY_CORRUPTION:
                return DiagnosticSeverity::corruption;
            case D3D12_MESSAGE_SEVERITY_ERROR:
                return DiagnosticSeverity::error;
            case D3D12_MESSAGE_SEVERITY_WARNING:
                return DiagnosticSeverity::warning;
            default:
                return DiagnosticSeverity::information;
            }
        }

        void configureDred(bool enableDred) noexcept
        {
            ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> settings;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&settings))))
            {
                const auto enablement = selectDredEnablementPolicy(enableDred) == DredEnablementPolicy::forcedOn
                                            ? D3D12_DRED_ENABLEMENT_FORCED_ON
                                            : D3D12_DRED_ENABLEMENT_SYSTEM_CONTROLLED;
                settings->SetAutoBreadcrumbsEnablement(enablement);
                settings->SetPageFaultEnablement(enablement);
                settings->SetBreadcrumbContextEnablement(enablement);
            }
        }

        struct DebugLayerProcessState final
        {
            std::mutex mutex;
            bool debugLayerEnabled = false;
            bool gpuValidationEnabled = false;
            std::uint64_t devicesWithoutDebugLayer = 0;
            std::uint64_t devicesWithoutGpuValidation = 0;
        };

        [[nodiscard]] DebugLayerProcessState& debugLayerProcessState() noexcept
        {
            union ImmortalState {
                DebugLayerProcessState value;

                ImmortalState() noexcept : value()
                {
                }

                ~ImmortalState()
                {
                }

                ImmortalState(const ImmortalState&) = delete;
                ImmortalState& operator=(const ImmortalState&) = delete;
                ImmortalState(ImmortalState&&) = delete;
                ImmortalState& operator=(ImmortalState&&) = delete;
            };
            static ImmortalState s_state;
            return s_state.value;
        }

        class ProcessDeviceRegistration final
        {
        public:
            ProcessDeviceRegistration() noexcept = default;

            ~ProcessDeviceRegistration()
            {
                release();
            }

            ProcessDeviceRegistration(const ProcessDeviceRegistration&) = delete;
            ProcessDeviceRegistration& operator=(const ProcessDeviceRegistration&) = delete;

            ProcessDeviceRegistration(ProcessDeviceRegistration&& other) noexcept
                : state_(std::exchange(other.state_, nullptr)),
                  withoutDebugLayer_(std::exchange(other.withoutDebugLayer_, false)),
                  withoutGpuValidation_(std::exchange(other.withoutGpuValidation_, false))
            {
            }

            ProcessDeviceRegistration& operator=(ProcessDeviceRegistration&& other) noexcept
            {
                if (this != &other)
                {
                    release();
                    state_ = std::exchange(other.state_, nullptr);
                    withoutDebugLayer_ = std::exchange(other.withoutDebugLayer_, false);
                    withoutGpuValidation_ = std::exchange(other.withoutGpuValidation_, false);
                }
                return *this;
            }

            [[nodiscard]] static ProcessDeviceRegistration registerWhileLocked(DebugLayerProcessState& state,
                                                                               bool debugLayerActive,
                                                                               bool gpuValidationActive) noexcept
            {
                ProcessDeviceRegistration registration;
                if (debugLayerActive && gpuValidationActive)
                {
                    return registration;
                }
                registration.state_ = &state;
                registration.withoutDebugLayer_ = !debugLayerActive;
                registration.withoutGpuValidation_ = !gpuValidationActive;
                if (registration.withoutDebugLayer_)
                {
                    ++state.devicesWithoutDebugLayer;
                }
                if (registration.withoutGpuValidation_)
                {
                    ++state.devicesWithoutGpuValidation;
                }
                return registration;
            }

        private:
            void release() noexcept
            {
                if (state_ == nullptr)
                {
                    return;
                }
                const std::scoped_lock lock(state_->mutex);
                if (withoutDebugLayer_)
                {
                    --state_->devicesWithoutDebugLayer;
                }
                if (withoutGpuValidation_)
                {
                    --state_->devicesWithoutGpuValidation;
                }
                state_ = nullptr;
                withoutDebugLayer_ = false;
                withoutGpuValidation_ = false;
            }

            DebugLayerProcessState* state_ = nullptr;
            bool withoutDebugLayer_ = false;
            bool withoutGpuValidation_ = false;
        };

        [[nodiscard]] AdapterType probeHardwareAdapterType(IDXGIAdapter1& adapter) noexcept
        {
            ComPtr<ID3D12Device> device;
            if (FAILED(createNativeDevice(&adapter, IID_PPV_ARGS(&device))))
            {
                return AdapterType::unknown;
            }
            return hardwareAdapterType(*device.Get());
        }

        [[nodiscard]] bool hasActiveDebugLayer(ID3D12Device& device) noexcept
        {
            ComPtr<ID3D12DebugDevice> debugDevice;
            return SUCCEEDED(device.QueryInterface(IID_PPV_ARGS(&debugDevice)));
        }

        enum class ProcessValidationFeature
        {
            debugLayer,
            gpuBasedValidation,
        };

        [[nodiscard]] std::string blockedValidationReason(std::uint64_t liveDevices, ProcessValidationFeature feature)
        {
            const bool gpuBased = feature == ProcessValidationFeature::gpuBasedValidation;
            const bool single = liveDevices == 1;
            std::ostringstream reason;
            reason << liveDevices
                   << (single ? " RachisRHI D3D12 device that was" : " RachisRHI D3D12 devices that were")
                   << " created without " << (gpuBased ? "GPU-based validation " : "the debug layer ")
                   << (single ? "is" : "are") << " still alive. "
                   << (gpuBased ? "D3D12 GPU-based validation" : "The D3D12 debug layer")
                   << " applies to the whole process, and enabling it now would remove every existing D3D12 device. "
                      "Destroy "
                      "those devices first, or request "
                   << (gpuBased ? "GPU-based validation" : "validation")
                   << " before creating the first D3D12 device in the process.";
            return reason.str();
        }

        struct ValidationPlan final
        {
            bool enabled = false;
            std::uint64_t blockingDevices = 0;
            std::uint64_t gpuBlockingDevices = 0;
        };

        [[nodiscard]] Result<ValidationPlan> enableValidation(const DeviceDesc& desc, DebugLayerProcessState& state)
        {
            if (desc.validation == ValidationLevel::disabled)
            {
                return ValidationPlan{};
            }

            ComPtr<ID3D12Debug> debug;
            const HRESULT debugResult = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
            if (FAILED(debugResult))
            {
                if (desc.requireValidation)
                {
                    return std::unexpected(
                        nativeError(ErrorCode::validationError, debugResult, "D3D12GetDebugInterface"));
                }
                return ValidationPlan{};
            }

            if (!state.debugLayerEnabled && state.devicesWithoutDebugLayer > 0)
            {
                if (desc.requireValidation)
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "D3D12 validation is required but cannot be enabled: " +
                            blockedValidationReason(state.devicesWithoutDebugLayer,
                                                    ProcessValidationFeature::debugLayer),
                    });
                }
                return ValidationPlan{.enabled = false, .blockingDevices = state.devicesWithoutDebugLayer};
            }

            const bool enableGpuValidation = desc.validation == ValidationLevel::gpu && !state.gpuValidationEnabled;
            const std::uint64_t gpuBlockingDevices = enableGpuValidation ? state.devicesWithoutGpuValidation : 0;
            if (gpuBlockingDevices > 0 && desc.requireValidation)
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "D3D12 GPU-based validation is required but cannot be enabled: " +
                        blockedValidationReason(gpuBlockingDevices, ProcessValidationFeature::gpuBasedValidation),
                });
            }

            ComPtr<ID3D12Debug1> gpuDebug;
            if (enableGpuValidation && gpuBlockingDevices == 0)
            {
                const HRESULT gpuResult = debug.As(&gpuDebug);
                if (FAILED(gpuResult) && desc.requireValidation)
                {
                    return std::unexpected(nativeError(ErrorCode::validationError, gpuResult, "ID3D12Debug1 query"));
                }
            }

            if (!state.debugLayerEnabled)
            {
                debug->EnableDebugLayer();
                state.debugLayerEnabled = true;
            }
            if (gpuDebug != nullptr)
            {
                gpuDebug->SetEnableGPUBasedValidation(TRUE);
                state.gpuValidationEnabled = true;
            }

            return ValidationPlan{.enabled = true, .blockingDevices = 0, .gpuBlockingDevices = gpuBlockingDevices};
        }

    } // namespace

    class D3D12Device final : public detail::BackendDevice
    {
    public:
        D3D12Device(ProcessDeviceRegistration debugLayerRegistration,
                    ComPtr<ID3D12Device> device,
                    ComPtr<IDXGIAdapter3> budgetAdapter,
                    AdapterInfo adapter,
                    DeviceCapabilities capabilities,
                    DiagnosticCallback callback,
                    void* callbackUserData) noexcept
            : debugLayerRegistration_(std::move(debugLayerRegistration)), device_(std::move(device)),
              budgetAdapter_(std::move(budgetAdapter)), adapter_(std::move(adapter)), capabilities_(capabilities),
              callback_(callback), callbackUserData_(callbackUserData)
        {
        }

        ~D3D12Device() override
        {
            if (infoQueue_ != nullptr && callbackCookie_ != 0)
            {
                infoQueue_->UnregisterMessageCallback(callbackCookie_);
            }
        }

        D3D12Device(const D3D12Device&) = delete;
        D3D12Device& operator=(const D3D12Device&) = delete;
        D3D12Device(D3D12Device&&) = delete;
        D3D12Device& operator=(D3D12Device&&) = delete;

        [[nodiscard]] Status registerDiagnostics(bool validationEnabled)
        {
            if (!validationEnabled)
            {
                return {};
            }

            const HRESULT queryResult = device_.As(&infoQueue_);
            if (FAILED(queryResult))
            {
                return std::unexpected(nativeError(ErrorCode::validationError, queryResult, "ID3D12InfoQueue1 query"));
            }

            std::array ignoredMessages{
                D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
                D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
            };
            D3D12_INFO_QUEUE_FILTER filter{};
            filter.DenyList.NumIDs = static_cast<UINT>(ignoredMessages.size());
            filter.DenyList.pIDList = ignoredMessages.data();
            const HRESULT filterResult = infoQueue_->AddStorageFilterEntries(&filter);
            if (FAILED(filterResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::validationError, filterResult, "ID3D12InfoQueue1::AddStorageFilterEntries"));
            }

            const HRESULT registerResult = infoQueue_->RegisterMessageCallback(
                &D3D12Device::messageCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, this, &callbackCookie_);
            if (FAILED(registerResult))
            {
                return std::unexpected(nativeError(
                    ErrorCode::validationError, registerResult, "ID3D12InfoQueue1::RegisterMessageCallback"));
            }
            return {};
        }

        void publishDiagnostic(DiagnosticSeverity severity, std::string_view description) noexcept
        {
            countDiagnostic(severity);
            if (callback_ != nullptr)
            {
                callback_(DiagnosticMessage{severity, 0, description}, callbackUserData_);
            }
        }

        [[nodiscard]] std::string_view backendName() const noexcept override
        {
            return "D3D12";
        }

        [[nodiscard]] std::uint64_t deviceToken() const noexcept override
        {
            return deviceToken_;
        }

        [[nodiscard]] const AdapterInfo& adapterInfo() const noexcept override
        {
            return adapter_;
        }

        [[nodiscard]] const DeviceCapabilities& capabilities() const noexcept override
        {
            return capabilities_;
        }

        [[nodiscard]] DeviceOperationalState operationalState() const noexcept override
        {
            return operationalState_.load(std::memory_order_acquire);
        }

        [[nodiscard]] std::optional<Error> deviceLossError() const override
        {
            if (operationalState_.load(std::memory_order_acquire) != DeviceOperationalState::lost)
            {
                return std::nullopt;
            }
            const std::scoped_lock lock(deviceLossMutex_);
            return deviceLossError_ != nullptr ? std::optional<Error>{*deviceLossError_} : std::nullopt;
        }

        [[nodiscard]] Result<std::vector<MemoryBudget>> queryMemoryBudgets(std::uint32_t nodeIndex) const override
        {
            if (!capabilities_.memoryBudgets || budgetAdapter_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Memory-budget queries are unavailable on this adapter.",
                });
            }
            if (nodeIndex >= capabilities_.nodeCount)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The memory-budget node index is outside the device node range.",
                });
            }
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }

            constexpr std::array groups{
                DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
            };
            const auto allocationStatistics = nodeIndex == 0
                                                  ? resources_.memoryAllocationStatistics()
                                                  : std::array<MemoryAllocationStatistics, memorySegmentGroupCount>{};
            std::vector<MemoryBudget> budgets;
            budgets.reserve(groups.size());
            for (std::uint32_t heapIndex = 0; heapIndex < groups.size(); ++heapIndex)
            {
                DXGI_QUERY_VIDEO_MEMORY_INFO information{};
                const HRESULT result = budgetAdapter_->QueryVideoMemoryInfo(nodeIndex, groups[heapIndex], &information);
                if (FAILED(result))
                {
                    const auto code = isD3D12DeviceLossHresult(static_cast<std::int32_t>(result))
                                          ? ErrorCode::deviceLost
                                          : ErrorCode::nativeFailure;
                    return observeDeviceLoss(
                        Result<std::vector<MemoryBudget>>{
                            std::unexpected(nativeError(code, result, "IDXGIAdapter3::QueryVideoMemoryInfo")),
                        },
                        "memory-budget query");
                }
                budgets.push_back(MemoryBudget{
                    .heapIndex = heapIndex,
                    .properties = heapIndex == 0 ? MemoryHeapProperty::deviceLocal : MemoryHeapProperty::none,
                    .features = nodeIndex == 0
                                    ? MemoryBudgetFeature::reservation | MemoryBudgetFeature::allocationStatistics
                                    : MemoryBudgetFeature::reservation,
                    .blockBytes = allocationStatistics[heapIndex].blockBytes,
                    .allocationBytes = allocationStatistics[heapIndex].allocationBytes,
                    .blockCount = allocationStatistics[heapIndex].blockCount,
                    .allocationCount = allocationStatistics[heapIndex].allocationCount,
                    .budgetBytes = information.Budget,
                    .usageBytes = information.CurrentUsage,
                    .availableForReservationBytes = information.AvailableForReservation,
                    .reservationBytes = information.CurrentReservation,
                });
            }
            return budgets;
        }

        [[nodiscard]] Result<ResourceMemorySnapshot> captureResourceMemorySnapshot() const override
        {
            return resources_.captureMemorySnapshot();
        }

        [[nodiscard]] Result<DeviceFaultReport> captureDeviceFaultReport() const override
        {
            {
                const std::scoped_lock lock(deviceLossMutex_);
                if (deviceFaultReport_ != nullptr)
                {
                    return *deviceFaultReport_;
                }
            }
            return d3d12::captureDeviceFaultReport(*device_.Get());
        }

        [[nodiscard]] BackendAllocationSnapshot captureBackendAllocationSnapshot() const noexcept override
        {
            auto descriptors = resources_.descriptorAllocationStatistics();
            descriptors += views_.descriptorAllocationStatistics();
            descriptors += samplers_.descriptorAllocationStatistics();
            descriptors += bindings_.descriptorAllocationStatistics();
            return {
                .liveDescriptorCount = descriptors.liveCount,
                .reservedDescriptorCount = descriptors.reservedCount,
            };
        }

        [[nodiscard]] FormatProperties queryFormatProperties(Format format) const noexcept override
        {
            return d3d12FormatProperties(*device_.Get(), format);
        }

        [[nodiscard]] QueueProperties queueProperties(QueueType queueType) const noexcept override
        {
            return queries_.queueProperties(queueType);
        }

        [[nodiscard]] Result<QueueClockCalibration> calibrateQueueClock(QueueType queueType) const override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queries_.calibrateQueueClock(queues_, queueType), "queue clock calibration");
        }

        [[nodiscard]] DiagnosticCounts diagnosticCounts() const noexcept override
        {
            return {
                information_.load(std::memory_order_relaxed),
                warnings_.load(std::memory_order_relaxed),
                errors_.load(std::memory_order_relaxed),
                corruptions_.load(std::memory_order_relaxed),
            };
        }

        [[nodiscard]] BindingUpdateStatistics bindingUpdateStatistics() const noexcept override
        {
            return bindings_.statistics();
        }

        [[nodiscard]] Status initializeQueues()
        {
            const auto token = test::nextDeviceTokenExhaustion().exchange(false, std::memory_order_relaxed)
                                   ? std::optional<std::uint64_t>{}
                                   : detail::deviceTokenSource().acquire();
            if (!token.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "The D3D12 device identity space is exhausted and cannot be reused safely.",
                });
            }
            auto initialized = queues_.initialize(*device_.Get(), *token);
            if (!initialized)
            {
                return initialized;
            }
            deviceToken_ = *token;
            return {};
        }

        [[nodiscard]] Status initializeQueries(bool enhancedBarriers)
        {
            return queries_.initialize(
                *device_.Get(), queues_, enhancedBarriers, capabilities_.meshPipelineStatisticsQueries);
        }

        [[nodiscard]] Status initializeResources(IDXGIAdapter& adapter, bool enhancedBarriers)
        {
            auto resources = resources_.initialize(*device_.Get(),
                                                   adapter,
                                                   enhancedBarriers,
                                                   capabilities_.variableRateShadingTier2,
                                                   capabilities_.rayTracing,
                                                   capabilities_.sparseBuffers && capabilities_.sparseTextures,
                                                   capabilities_.samplerFeedback);
            if (!resources)
            {
                return resources;
            }
            auto views = views_.initialize(*device_.Get());
            if (!views)
            {
                return views;
            }
            auto samplers = samplers_.initialize(*device_.Get());
            if (!samplers)
            {
                return samplers;
            }
            auto bindings = bindings_.initialize(*device_.Get(),
                                                 capabilities_.variableDescriptorCountBindingArrays,
                                                 capabilities_.rayTracing,
                                                 capabilities_.samplerFeedback);
            if (!bindings)
            {
                return bindings;
            }
            if (capabilities_.mipmapGeneration)
            {
                auto mipmaps = mipmaps_.initialize(*device_.Get());
                if (!mipmaps)
                {
                    return mipmaps;
                }
            }
            auto pipelines = pipelines_.initialize(*device_.Get(),
                                                   capabilities_.limits.maxComputeWorkgroupInvocations,
                                                   capabilities_.limits.maxComputeWorkgroupSize,
                                                   capabilities_.geometryShaders,
                                                   capabilities_.tessellation.maxPatchControlPoints,
                                                   capabilities_.meshShaders,
                                                   capabilities_.taskShaders,
                                                   capabilities_.meshShader,
                                                   capabilities_.dualSourceBlending,
                                                   capabilities_.logicOperations,
                                                   capabilities_.conservativeRasterization,
                                                   capabilities_.triangleFans,
                                                   capabilities_.dynamicDepthBias,
                                                   capabilities_.depthBoundsTest,
                                                   capabilities_.variableRateShading,
                                                   capabilities_.additionalShadingRates,
                                                   capabilities_.variableRateShadingTier2,
                                                   capabilities_.limits.maxClipDistances,
                                                   capabilities_.limits.maxCullDistances,
                                                   capabilities_.limits.maxCombinedClipAndCullDistances,
                                                   capabilities_.limits.maxFramebufferLayers,
                                                   capabilities_.maxMultiviewViewCount,
                                                   capabilities_.rayTracing,
                                                   capabilities_.rayTracingLimits);
            if (!pipelines)
            {
                return pipelines;
            }
            return swapchains_.initialize(*device_.Get());
        }

        [[nodiscard]] Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc& desc) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.createCommandPool(desc), "command-pool creation");
        }

        [[nodiscard]] Status destroyCommandPool(CommandPoolHandle commandPool) override
        {
            return queues_.destroyCommandPool(commandPool);
        }

        [[nodiscard]] Status resetCommandPool(CommandPoolHandle commandPool) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.resetCommandPool(commandPool), "command-pool reset");
        }

        [[nodiscard]] Result<std::unique_ptr<detail::BackendCommandBuffer>> acquireCommandBuffer(
            CommandPoolHandle commandPool) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.acquire(commandPool), "command-buffer acquisition");
        }

        [[nodiscard]] Status resetCommandBuffer(detail::BackendCommandBuffer& commandBuffer) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.reset(commandBuffer), "command-buffer reset");
        }

        [[nodiscard]] Result<TimelinePoint> submit(
            std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers,
            std::span<const TimelinePoint> waits) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.submit(commandBuffers, waits), "queue submission");
        }

        [[nodiscard]] Result<TimelinePointStatus> queryTimelinePoint(TimelinePoint point) const override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.queryTimelinePoint(point), "timeline query");
        }

        [[nodiscard]] Result<TimelinePointStatus> wait(TimelinePoint point, std::chrono::nanoseconds timeout) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.wait(point, timeout), "bounded timeline wait");
        }

        [[nodiscard]] Status wait(TimelinePoint point) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.wait(point), "timeline wait");
        }

        [[nodiscard]] Result<SharedTimelinePoint> exportTimelinePoint(TimelinePoint point) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return queues_.exportTimelinePoint(point);
        }

        [[nodiscard]] Result<TimelinePoint> enqueueSharedTimelineWait(QueueType queueType,
                                                                      SharedTimelinePointView point) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.enqueueSharedTimelineWait(queueType, point), "shared timeline wait");
        }

        [[nodiscard]] Status waitIdle(QueueType queueType) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.waitIdle(queueType), "queue idle wait");
        }

        [[nodiscard]] Status waitIdle() override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(queues_.waitIdle(), "device idle wait");
        }

        [[nodiscard]] std::uint64_t completedValue(QueueType queueType) const noexcept override
        {
            return queues_.completedValue(queueType);
        }

        void collectGarbage() noexcept override
        {
            queues_.collectGarbage();
        }

        [[nodiscard]] Status beginDebugGroup(detail::BackendCommandBuffer& commandBuffer,
                                             const DebugMarkerDesc& desc) override
        {
            return debugMarkers_.beginGroup(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Status endDebugGroup(detail::BackendCommandBuffer& commandBuffer) override
        {
            return debugMarkers_.endGroup(queues_, commandBuffer);
        }

        [[nodiscard]] Status insertDebugMarker(detail::BackendCommandBuffer& commandBuffer,
                                               const DebugMarkerDesc& desc) override
        {
            return debugMarkers_.insertMarker(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Result<ResourceHeapHandle> createResourceHeap(const ResourceHeapDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.createResourceHeap(desc); }, "resource-heap creation");
        }

        [[nodiscard]] Status destroyResourceHeap(ResourceHeapHandle heap) override
        {
            return resources_.destroyResourceHeap(heap);
        }

        [[nodiscard]] Result<ResidencyState> queryResidency(ResourceHeapHandle heap) const override
        {
            return resources_.queryResidency(queues_, heap);
        }

        [[nodiscard]] Result<ResourceMemoryRequirements> queryBufferMemoryRequirements(
            const BufferDesc& desc) const override
        {
            return resources_.queryBufferMemoryRequirements(desc);
        }

        [[nodiscard]] Result<BufferHandle> createBuffer(const BufferDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.createBuffer(desc); }, "buffer creation");
        }

        [[nodiscard]] Result<BufferHandle> createPlacedBuffer(const ResourcePlacement& placement,
                                                              const BufferDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.createPlacedBuffer(placement, desc); },
                                   "placed-buffer creation");
        }

        [[nodiscard]] Result<BufferHandle> createSparseBuffer(const BufferDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.createSparseBuffer(desc); }, "sparse-buffer creation");
        }

        [[nodiscard]] Result<SparseBufferProperties> querySparseBufferProperties(BufferHandle buffer) const override
        {
            return resources_.querySparseBufferProperties(buffer);
        }

        [[nodiscard]] Result<TimelinePoint> enqueueSparseBufferMappings(
            QueueType queueType, std::span<const SparseBufferTileMapping> mappings) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(resources_.enqueueSparseBufferMappings(queues_, queueType, mappings),
                                     "sparse-buffer mapping request");
        }

        [[nodiscard]] Result<BufferHandle> createSharedBuffer(const BufferDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.createSharedBuffer(desc); }, "shared-buffer creation");
        }

        [[nodiscard]] Result<SharedHandle> exportSharedBuffer(BufferHandle buffer) override
        {
            return guardDeviceLoss([&] { return resources_.exportSharedBuffer(buffer); }, "shared-buffer export");
        }

        [[nodiscard]] Result<BufferHandle> importSharedBuffer(SharedHandleView handle, const BufferDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.importSharedBuffer(handle, desc); }, "shared-buffer import");
        }

        [[nodiscard]] Status destroyBuffer(BufferHandle buffer) override
        {
            return resources_.destroyBuffer(buffer);
        }

        [[nodiscard]] Result<ResidencyState> queryResidency(BufferHandle buffer) const override
        {
            return resources_.queryResidency(queues_, buffer);
        }

        [[nodiscard]] Result<MappedMemory> mapBuffer(BufferHandle buffer,
                                                     std::uint64_t offset,
                                                     std::uint64_t size) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(resources_.mapBuffer(buffer, offset, size), "buffer mapping");
        }

        [[nodiscard]] Status unmapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size) override
        {
            return resources_.unmapBuffer(buffer, offset, size);
        }

        [[nodiscard]] Status flushMappedMemoryRanges(BufferHandle buffer,
                                                     std::span<const MappedMemoryRange> ranges) override
        {
            return resources_.flushMappedMemoryRanges(buffer, ranges);
        }

        [[nodiscard]] Status invalidateMappedMemoryRanges(BufferHandle buffer,
                                                          std::span<const MappedMemoryRange> ranges) override
        {
            return resources_.invalidateMappedMemoryRanges(buffer, ranges);
        }

        [[nodiscard]] Status writeBuffer(BufferHandle buffer,
                                         std::uint64_t offset,
                                         std::span<const std::byte> data) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(resources_.writeBuffer(buffer, offset, data), "host buffer write");
        }

        [[nodiscard]] Status readBuffer(BufferHandle buffer,
                                        std::uint64_t offset,
                                        std::span<std::byte> destination) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(resources_.readBuffer(buffer, offset, destination), "host buffer read");
        }

        [[nodiscard]] Status copyBufferRegions(detail::BackendCommandBuffer& commandBuffer,
                                               BufferHandle source,
                                               BufferHandle destination,
                                               std::span<const BufferCopyRegion> regions) override
        {
            return resources_.copyBufferRegions(queues_, commandBuffer, source, destination, regions);
        }

        [[nodiscard]] Status bufferBarrier(detail::BackendCommandBuffer& commandBuffer,
                                           const BufferBarrier& barrier) override
        {
            return resources_.bufferBarrier(queues_, commandBuffer, barrier);
        }

        [[nodiscard]] Status samplerFeedbackBarrier(detail::BackendCommandBuffer& commandBuffer,
                                                    const SamplerFeedbackBarrier& barrier) override
        {
            return resources_.samplerFeedbackBarrier(queues_, commandBuffer, barrier);
        }

        [[nodiscard]] Status activateBufferAlias(detail::BackendCommandBuffer& commandBuffer,
                                                 const BufferAliasActivation& activation) override
        {
            return resources_.activateBufferAlias(queues_, commandBuffer, activation);
        }

        [[nodiscard]] Status resourceBarriers(detail::BackendCommandBuffer& commandBuffer,
                                              const ResourceBarrierBatch& barriers) override
        {
            return resources_.resourceBarriers(queues_, commandBuffer, barriers);
        }

        [[nodiscard]] Result<AccelerationStructureBuildSizes> queryBottomLevelAccelerationStructureBuildSizes(
            const BottomLevelAccelerationStructureBuildInput& input) const override
        {
            return resources_.queryBottomLevelAccelerationStructureBuildSizes(input);
        }

        [[nodiscard]] Result<AccelerationStructureBuildSizes> queryTopLevelAccelerationStructureBuildSizes(
            const TopLevelAccelerationStructureBuildInput& input) const override
        {
            return resources_.queryTopLevelAccelerationStructureBuildSizes(input);
        }

        [[nodiscard]] Result<AccelerationStructureHandle> createAccelerationStructure(
            const AccelerationStructureDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.createAccelerationStructure(desc); },
                                   "acceleration-structure creation");
        }

        [[nodiscard]] Status destroyAccelerationStructure(AccelerationStructureHandle accelerationStructure) override
        {
            return resources_.destroyAccelerationStructure(accelerationStructure);
        }

        [[nodiscard]] Status buildBottomLevelAccelerationStructure(
            detail::BackendCommandBuffer& commandBuffer, const BottomLevelAccelerationStructureBuildDesc& desc) override
        {
            return resources_.buildBottomLevelAccelerationStructure(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Status buildTopLevelAccelerationStructure(
            detail::BackendCommandBuffer& commandBuffer, const TopLevelAccelerationStructureBuildDesc& desc) override
        {
            return resources_.buildTopLevelAccelerationStructure(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Status updateBottomLevelAccelerationStructure(
            detail::BackendCommandBuffer& commandBuffer,
            const BottomLevelAccelerationStructureUpdateDesc& desc) override
        {
            return resources_.updateBottomLevelAccelerationStructure(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Status updateTopLevelAccelerationStructure(
            detail::BackendCommandBuffer& commandBuffer, const TopLevelAccelerationStructureUpdateDesc& desc) override
        {
            return resources_.updateTopLevelAccelerationStructure(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Status updateBottomLevelAccelerationStructureInPlace(
            detail::BackendCommandBuffer& commandBuffer,
            const BottomLevelAccelerationStructureInPlaceUpdateDesc& desc) override
        {
            return resources_.updateBottomLevelAccelerationStructureInPlace(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Status updateTopLevelAccelerationStructureInPlace(
            detail::BackendCommandBuffer& commandBuffer,
            const TopLevelAccelerationStructureInPlaceUpdateDesc& desc) override
        {
            return resources_.updateTopLevelAccelerationStructureInPlace(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Status compactAccelerationStructure(detail::BackendCommandBuffer& commandBuffer,
                                                          const AccelerationStructureCompactionDesc& desc) override
        {
            return resources_.compactAccelerationStructure(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Status serializeAccelerationStructure(detail::BackendCommandBuffer& commandBuffer,
                                                            const AccelerationStructureSerializationDesc& desc) override
        {
            return resources_.serializeAccelerationStructure(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Result<AccelerationStructureArchiveInfo> inspectAccelerationStructureArchive(
            std::span<const std::byte> archive) const override
        {
            return resources_.inspectAccelerationStructureArchive(archive);
        }

        [[nodiscard]] Status deserializeAccelerationStructure(
            detail::BackendCommandBuffer& commandBuffer, const AccelerationStructureDeserializationDesc& desc) override
        {
            return resources_.deserializeAccelerationStructure(queues_, commandBuffer, desc);
        }

        [[nodiscard]] Result<ResourceMemoryRequirements> queryTextureMemoryRequirements(
            const TextureDesc& desc) const override
        {
            return resources_.queryTextureMemoryRequirements(desc);
        }

        [[nodiscard]] Result<TextureHandle> createTexture(const TextureDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.createTexture(desc); }, "texture creation");
        }

        [[nodiscard]] Result<TextureHandle> createPlacedTexture(const ResourcePlacement& placement,
                                                                const TextureDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.createPlacedTexture(placement, desc); },
                                   "placed-texture creation");
        }

        [[nodiscard]] Result<TextureHandle> createSparseTexture(const TextureDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.createSparseTexture(desc); }, "sparse-texture creation");
        }

        [[nodiscard]] Result<SparseTextureProperties> querySparseTextureProperties(TextureHandle texture) const override
        {
            return resources_.querySparseTextureProperties(texture);
        }

        [[nodiscard]] Result<TimelinePoint> enqueueSparseTextureMappings(
            QueueType queueType, const SparseTextureMappingBatch& mappings) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(resources_.enqueueSparseTextureMappings(queues_, queueType, mappings),
                                     "sparse-texture mapping request");
        }

        [[nodiscard]] Result<SamplerFeedbackMapHandle> createMinMipFeedbackMap(
            const MinMipFeedbackMapDesc& desc) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(resources_.createMinMipFeedbackMap(desc), "create MinMip feedback map");
        }

        [[nodiscard]] Result<MinMipFeedbackMapProperties> queryMinMipFeedbackMapProperties(
            SamplerFeedbackMapHandle feedbackMap) const override
        {
            return resources_.queryMinMipFeedbackMapProperties(feedbackMap);
        }

        [[nodiscard]] Result<SamplerFeedbackMapHandle> createMipRegionUsedFeedbackMap(
            const MipRegionUsedFeedbackMapDesc& desc) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(resources_.createMipRegionUsedFeedbackMap(desc),
                                     "create MipRegionUsed feedback map");
        }

        [[nodiscard]] Result<MipRegionUsedFeedbackMapProperties> queryMipRegionUsedFeedbackMapProperties(
            SamplerFeedbackMapHandle feedbackMap) const override
        {
            return resources_.queryMipRegionUsedFeedbackMapProperties(feedbackMap);
        }

        [[nodiscard]] Status destroySamplerFeedbackMap(SamplerFeedbackMapHandle feedbackMap) override
        {
            return resources_.destroySamplerFeedbackMap(feedbackMap);
        }

        [[nodiscard]] Status resetSamplerFeedbackMap(detail::BackendCommandBuffer& commandBuffer,
                                                     SamplerFeedbackMapHandle feedbackMap) override
        {
            return resources_.resetSamplerFeedbackMap(queues_, bindings_, commandBuffer, feedbackMap);
        }

        [[nodiscard]] Status decodeMinMipFeedback(detail::BackendCommandBuffer& commandBuffer,
                                                  const MinMipFeedbackDecode& decode) override
        {
            return resources_.decodeMinMipFeedback(queues_, commandBuffer, decode);
        }

        [[nodiscard]] Status decodeMipRegionUsedFeedback(detail::BackendCommandBuffer& commandBuffer,
                                                         const MipRegionUsedFeedbackDecode& decode) override
        {
            return resources_.decodeMipRegionUsedFeedback(queues_, commandBuffer, decode);
        }

        [[nodiscard]] Status encodeMinMipFeedback(detail::BackendCommandBuffer& commandBuffer,
                                                  const MinMipFeedbackEncode& encode) override
        {
            return resources_.encodeMinMipFeedback(queues_, commandBuffer, encode);
        }

        [[nodiscard]] Status encodeMipRegionUsedFeedback(detail::BackendCommandBuffer& commandBuffer,
                                                         const MipRegionUsedFeedbackEncode& encode) override
        {
            return resources_.encodeMipRegionUsedFeedback(queues_, commandBuffer, encode);
        }

        [[nodiscard]] Result<TextureHandle> createSharedTexture(const TextureDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.createSharedTexture(desc); }, "shared-texture creation");
        }

        [[nodiscard]] Result<SharedHandle> exportSharedTexture(TextureHandle texture) override
        {
            return guardDeviceLoss([&] { return resources_.exportSharedTexture(texture); }, "shared-texture export");
        }

        [[nodiscard]] Result<TextureHandle> importSharedTexture(SharedHandleView handle,
                                                                const TextureDesc& desc) override
        {
            return guardDeviceLoss([&] { return resources_.importSharedTexture(handle, desc); },
                                   "shared-texture import");
        }

        [[nodiscard]] Status destroyTexture(TextureHandle texture) override
        {
            return resources_.destroyTexture(texture);
        }

        [[nodiscard]] Result<ResidencyState> queryResidency(TextureHandle texture) const override
        {
            return resources_.queryResidency(queues_, texture);
        }

        [[nodiscard]] Result<TextureTransferLayout> queryTextureTransferLayout(
            TextureHandle texture, const BufferTextureCopyRegion& region) const override
        {
            return resources_.queryTextureTransferLayout(texture, region);
        }

        [[nodiscard]] Result<TimelinePoint> enqueueMakeResident(QueueType queueType,
                                                                const ResidencySet& resources,
                                                                ResidencyBudgetPolicy budgetPolicy) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(resources_.enqueueMakeResident(queues_, queueType, resources, budgetPolicy),
                                     "make-resident request");
        }

        [[nodiscard]] Status evict(const ResidencySet& resources) override
        {
            return guardDeviceLoss([&] { return resources_.evict(queues_, resources); }, "eviction");
        }

        [[nodiscard]] Status copyBufferToTextureRegions(detail::BackendCommandBuffer& commandBuffer,
                                                        BufferHandle source,
                                                        TextureHandle destination,
                                                        std::span<const BufferTextureCopyRegion> regions) override
        {
            return resources_.copyBufferToTextureRegions(queues_, commandBuffer, source, destination, regions);
        }

        [[nodiscard]] Status copyTextureToBufferRegions(detail::BackendCommandBuffer& commandBuffer,
                                                        TextureHandle source,
                                                        BufferHandle destination,
                                                        std::span<const BufferTextureCopyRegion> regions) override
        {
            return resources_.copyTextureToBufferRegions(queues_, commandBuffer, source, destination, regions);
        }

        [[nodiscard]] Status copyTextureRegions(detail::BackendCommandBuffer& commandBuffer,
                                                TextureHandle source,
                                                TextureHandle destination,
                                                std::span<const TextureCopyRegion> regions) override
        {
            return resources_.copyTextureRegions(queues_, commandBuffer, source, destination, regions);
        }

        [[nodiscard]] Status resolveTexture(detail::BackendCommandBuffer& commandBuffer,
                                            const TextureResolve& resolve) override
        {
            return resources_.resolveTexture(queues_, bindings_, pipelines_, commandBuffer, resolve);
        }

        [[nodiscard]] Status generateMipmaps(detail::BackendCommandBuffer& commandBuffer,
                                             const TextureMipmapGeneration& generation) override
        {
            return mipmaps_.generate(queues_, resources_, bindings_, commandBuffer, generation);
        }

        [[nodiscard]] Status textureBarrier(detail::BackendCommandBuffer& commandBuffer,
                                            const TextureBarrier& barrier) override
        {
            return resources_.textureBarrier(queues_, commandBuffer, barrier);
        }

        [[nodiscard]] Status activateTextureAlias(detail::BackendCommandBuffer& commandBuffer,
                                                  const TextureAliasActivation& activation) override
        {
            return resources_.activateTextureAlias(queues_, commandBuffer, activation);
        }

        [[nodiscard]] Result<TextureViewHandle> createTextureView(const TextureViewDesc& desc) override
        {
            return guardDeviceLoss([&] { return views_.createTextureView(resources_, desc); }, "texture-view creation");
        }

        [[nodiscard]] Status destroyTextureView(TextureViewHandle view) override
        {
            return views_.destroyTextureView(view);
        }

        [[nodiscard]] Result<SamplerHandle> createSampler(const SamplerDesc& desc) override
        {
            return guardDeviceLoss([&] { return samplers_.createSampler(desc); }, "sampler creation");
        }

        [[nodiscard]] Status destroySampler(SamplerHandle sampler) override
        {
            return samplers_.destroySampler(sampler);
        }

        [[nodiscard]] Result<BindingLayoutHandle> createBindingLayout(const BindingLayoutDesc& desc) override
        {
            return guardDeviceLoss([&] { return bindings_.createBindingLayout(desc); }, "binding-layout creation");
        }

        [[nodiscard]] Status destroyBindingLayout(BindingLayoutHandle layout) override
        {
            return bindings_.destroyBindingLayout(layout);
        }

        [[nodiscard]] Result<BindingPoolHandle> createBindingPool(const BindingPoolDesc& desc) override
        {
            return guardDeviceLoss([&] { return bindings_.createBindingPool(desc); }, "binding-pool creation");
        }

        [[nodiscard]] Status destroyBindingPool(BindingPoolHandle pool) override
        {
            return bindings_.destroyBindingPool(pool);
        }

        [[nodiscard]] Result<BindingSetHandle> createBindingSet(const BindingSetDesc& desc) override
        {
            return guardDeviceLoss(
                [&] {
                    queues_.collectGarbage();
                    return bindings_.createBindingSet(resources_, views_, samplers_, desc);
                },
                "binding-set creation");
        }

        [[nodiscard]] Status updateBindingSet(BindingSetHandle set, std::span<const BindingWrite> writes) override
        {
            queues_.collectGarbage();
            return bindings_.updateBindingSet(resources_, views_, samplers_, set, writes);
        }

        [[nodiscard]] Status destroyBindingSet(BindingSetHandle set) override
        {
            return bindings_.destroyBindingSet(set);
        }

        [[nodiscard]] Result<ShaderLibraryHandle> createShaderLibrary(const ShaderLibraryDesc& desc) override
        {
            return guardDeviceLoss([&] { return pipelines_.createShaderLibrary(desc); }, "shader-library creation");
        }

        [[nodiscard]] Status destroyShaderLibrary(ShaderLibraryHandle shaderLibrary) override
        {
            return pipelines_.destroyShaderLibrary(shaderLibrary);
        }

        [[nodiscard]] Result<ShaderModuleHandle> createShaderModule(const ShaderModuleDesc& desc) override
        {
            return guardDeviceLoss([&] { return pipelines_.createShaderModule(desc); }, "shader-module creation");
        }

        [[nodiscard]] Result<ShaderInterfaceSnapshot> shaderModuleInterface(
            ShaderModuleHandle shaderModule) const override
        {
            return pipelines_.shaderModuleInterface(shaderModule);
        }

        [[nodiscard]] Status destroyShaderModule(ShaderModuleHandle shaderModule) override
        {
            return pipelines_.destroyShaderModule(shaderModule);
        }

        [[nodiscard]] Result<PipelineLayoutHandle> createPipelineLayout(const PipelineLayoutDesc& desc) override
        {
            return guardDeviceLoss([&] { return pipelines_.createPipelineLayout(bindings_, desc); },
                                   "pipeline-layout creation");
        }

        [[nodiscard]] Status destroyPipelineLayout(PipelineLayoutHandle pipelineLayout) override
        {
            return pipelines_.destroyPipelineLayout(pipelineLayout);
        }

        [[nodiscard]] Result<PipelineCacheHandle> createPipelineCache(const PipelineCacheDesc& desc) override
        {
            return guardDeviceLoss([&] { return pipelines_.createPipelineCache(desc); }, "pipeline-cache creation");
        }

        [[nodiscard]] Status destroyPipelineCache(PipelineCacheHandle pipelineCache) override
        {
            return pipelines_.destroyPipelineCache(pipelineCache);
        }

        [[nodiscard]] Result<std::vector<std::byte>> pipelineCacheData(PipelineCacheHandle pipelineCache) override
        {
            auto data = pipelines_.pipelineCacheData(pipelineCache);
            if (!data)
            {
                return observeDeviceLoss(std::move(data), "pipeline-cache serialization");
            }
            return data;
        }

        [[nodiscard]] Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc& desc) override
        {
            return guardDeviceLoss([&] { return pipelines_.createComputePipeline(desc); }, "compute-pipeline creation");
        }

        [[nodiscard]] Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc& desc) override
        {
            return guardDeviceLoss([&] { return pipelines_.createGraphicsPipeline(desc); },
                                   "graphics-pipeline creation");
        }

        [[nodiscard]] Result<PipelineCompilationHandle> beginComputePipelineCompilation(
            const ComputePipelineDesc& desc) override
        {
            return guardDeviceLoss([&] { return pipelines_.beginComputePipelineCompilation(desc); },
                                   "compute-pipeline compilation");
        }

        [[nodiscard]] Result<PipelineCompilationHandle> beginGraphicsPipelineCompilation(
            const GraphicsPipelineDesc& desc) override
        {
            return guardDeviceLoss([&] { return pipelines_.beginGraphicsPipelineCompilation(desc); },
                                   "graphics-pipeline compilation");
        }

        [[nodiscard]] Result<PipelineCompilationState> pipelineCompilationState(
            PipelineCompilationHandle compilation) override
        {
            return pipelines_.pipelineCompilationState(compilation);
        }

        [[nodiscard]] Result<PipelineHandle> waitPipelineCompilation(PipelineCompilationHandle compilation) override
        {
            auto compiled = pipelines_.waitPipelineCompilation(compilation);
            if (!compiled)
            {
                return observeDeviceLoss(std::move(compiled), "pipeline compilation");
            }
            return compiled;
        }

        [[nodiscard]] Status destroyPipelineCompilation(PipelineCompilationHandle compilation) override
        {
            return pipelines_.destroyPipelineCompilation(compilation);
        }

        [[nodiscard]] Result<RayTracingPipelineLibraryHandle> createRayTracingPipelineLibrary(
            const RayTracingPipelineLibraryDesc& desc) override
        {
            return guardDeviceLoss([&] { return pipelines_.createRayTracingPipelineLibrary(desc); },
                                   "ray-tracing pipeline-library creation");
        }

        [[nodiscard]] Status destroyRayTracingPipelineLibrary(RayTracingPipelineLibraryHandle library) override
        {
            return pipelines_.destroyRayTracingPipelineLibrary(library);
        }

        [[nodiscard]] Result<PipelineHandle> createRayTracingPipeline(const RayTracingPipelineDesc& desc) override
        {
            return guardDeviceLoss([&] { return pipelines_.createRayTracingPipeline(desc); },
                                   "ray-tracing pipeline creation");
        }

        [[nodiscard]] Status getRayTracingShaderGroupHandles(PipelineHandle pipeline,
                                                             std::uint32_t firstGroup,
                                                             std::uint32_t groupCount,
                                                             std::span<std::byte> destination) override
        {
            return pipelines_.getRayTracingShaderGroupHandles(pipeline, firstGroup, groupCount, destination);
        }

        [[nodiscard]] Result<std::uint32_t> rayTracingShaderGroupRecordSize(PipelineHandle pipeline,
                                                                            std::uint32_t groupIndex) override
        {
            return pipelines_.rayTracingShaderGroupRecordSize(pipeline, groupIndex);
        }

        [[nodiscard]] Status writeRayTracingShaderGroupRecord(PipelineHandle pipeline,
                                                              std::uint32_t groupIndex,
                                                              std::span<const std::byte> localConstants,
                                                              std::span<std::byte> destination) override
        {
            return pipelines_.writeRayTracingShaderGroupRecord(pipeline, groupIndex, localConstants, destination);
        }

        [[nodiscard]] Status destroyPipeline(PipelineHandle pipeline) override
        {
            return pipelines_.destroyPipeline(pipeline);
        }

        [[nodiscard]] Status bindComputePipeline(detail::BackendCommandBuffer& commandBuffer,
                                                 PipelineHandle pipeline) override
        {
            return pipelines_.bindComputePipeline(queues_, bindings_, commandBuffer, pipeline);
        }

        [[nodiscard]] Status bindComputeBindingSet(detail::BackendCommandBuffer& commandBuffer,
                                                   std::uint32_t setIndex,
                                                   BindingSetHandle bindingSet,
                                                   std::span<const std::uint32_t> dynamicOffsets) override
        {
            return pipelines_.bindComputeBindingSet(
                queues_, bindings_, resources_, commandBuffer, setIndex, bindingSet, dynamicOffsets);
        }

        [[nodiscard]] Status bindComputeBindingSets(detail::BackendCommandBuffer& commandBuffer,
                                                    std::uint32_t firstSet,
                                                    std::span<const BindingSetHandle> bindingSets,
                                                    std::span<const std::uint32_t> dynamicOffsets) override
        {
            return pipelines_.bindComputeBindingSets(
                queues_, bindings_, resources_, commandBuffer, firstSet, bindingSets, dynamicOffsets);
        }

        [[nodiscard]] Status setComputePushConstants(detail::BackendCommandBuffer& commandBuffer,
                                                     std::uint32_t offset,
                                                     std::span<const std::byte> data) override
        {
            return pipelines_.setComputePushConstants(queues_, commandBuffer, offset, data);
        }

        [[nodiscard]] Status dispatch(detail::BackendCommandBuffer& commandBuffer,
                                      std::uint32_t groupCountX,
                                      std::uint32_t groupCountY,
                                      std::uint32_t groupCountZ) override
        {
            return pipelines_.dispatch(queues_, commandBuffer, groupCountX, groupCountY, groupCountZ);
        }

        [[nodiscard]] Status dispatchIndirect(detail::BackendCommandBuffer& commandBuffer,
                                              const IndirectDispatchDesc& desc) override
        {
            return pipelines_.dispatchIndirect(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Status bindRayTracingPipeline(detail::BackendCommandBuffer& commandBuffer,
                                                    PipelineHandle pipeline) override
        {
            return pipelines_.bindRayTracingPipeline(queues_, bindings_, commandBuffer, pipeline);
        }

        [[nodiscard]] Status bindRayTracingBindingSet(detail::BackendCommandBuffer& commandBuffer,
                                                      std::uint32_t setIndex,
                                                      BindingSetHandle bindingSet,
                                                      std::span<const std::uint32_t> dynamicOffsets) override
        {
            return pipelines_.bindRayTracingBindingSet(
                queues_, bindings_, resources_, commandBuffer, setIndex, bindingSet, dynamicOffsets);
        }

        [[nodiscard]] Status bindRayTracingBindingSets(detail::BackendCommandBuffer& commandBuffer,
                                                       std::uint32_t firstSet,
                                                       std::span<const BindingSetHandle> bindingSets,
                                                       std::span<const std::uint32_t> dynamicOffsets) override
        {
            return pipelines_.bindRayTracingBindingSets(
                queues_, bindings_, resources_, commandBuffer, firstSet, bindingSets, dynamicOffsets);
        }

        [[nodiscard]] Status setRayTracingPushConstants(detail::BackendCommandBuffer& commandBuffer,
                                                        std::uint32_t offset,
                                                        std::span<const std::byte> data) override
        {
            return pipelines_.setRayTracingPushConstants(queues_, commandBuffer, offset, data);
        }

        [[nodiscard]] Status dispatchRays(detail::BackendCommandBuffer& commandBuffer,
                                          const RayTracingDispatchDesc& desc) override
        {
            return pipelines_.dispatchRays(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Status dispatchRaysIndirect(detail::BackendCommandBuffer& commandBuffer,
                                                  const RayTracingIndirectDispatchDesc& desc) override
        {
            return pipelines_.dispatchRaysIndirect(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Status beginRendering(detail::BackendCommandBuffer& commandBuffer,
                                            const RenderingDesc& desc) override
        {
            return pipelines_.beginRendering(queues_, views_, queries_, commandBuffer, desc);
        }

        [[nodiscard]] Status endRendering(detail::BackendCommandBuffer& commandBuffer) override
        {
            return pipelines_.endRendering(queues_, queries_, commandBuffer);
        }

        [[nodiscard]] Status clearColorAttachment(detail::BackendCommandBuffer& commandBuffer,
                                                  std::uint32_t attachmentIndex,
                                                  const ClearColor& clearValue,
                                                  std::span<const ScissorRect> rectangles) override
        {
            return pipelines_.clearColorAttachment(queues_, commandBuffer, attachmentIndex, clearValue, rectangles);
        }

        [[nodiscard]] Status clearDepthStencilAttachment(detail::BackendCommandBuffer& commandBuffer,
                                                         DepthStencilClearAspect aspects,
                                                         float depth,
                                                         std::uint32_t stencil,
                                                         std::span<const ScissorRect> rectangles) override
        {
            return pipelines_.clearDepthStencilAttachment(queues_, commandBuffer, aspects, depth, stencil, rectangles);
        }

        [[nodiscard]] Status bindGraphicsPipeline(detail::BackendCommandBuffer& commandBuffer,
                                                  PipelineHandle pipeline) override
        {
            return pipelines_.bindGraphicsPipeline(queues_, bindings_, commandBuffer, pipeline);
        }

        [[nodiscard]] Status bindGraphicsBindingSet(detail::BackendCommandBuffer& commandBuffer,
                                                    std::uint32_t setIndex,
                                                    BindingSetHandle bindingSet,
                                                    std::span<const std::uint32_t> dynamicOffsets) override
        {
            return pipelines_.bindGraphicsBindingSet(
                queues_, bindings_, resources_, commandBuffer, setIndex, bindingSet, dynamicOffsets);
        }

        [[nodiscard]] Status bindGraphicsBindingSets(detail::BackendCommandBuffer& commandBuffer,
                                                     std::uint32_t firstSet,
                                                     std::span<const BindingSetHandle> bindingSets,
                                                     std::span<const std::uint32_t> dynamicOffsets) override
        {
            return pipelines_.bindGraphicsBindingSets(
                queues_, bindings_, resources_, commandBuffer, firstSet, bindingSets, dynamicOffsets);
        }

        [[nodiscard]] Status setGraphicsPushConstants(detail::BackendCommandBuffer& commandBuffer,
                                                      ShaderStage visibility,
                                                      std::uint32_t offset,
                                                      std::span<const std::byte> data) override
        {
            return pipelines_.setGraphicsPushConstants(queues_, commandBuffer, visibility, offset, data);
        }

        [[nodiscard]] Status setViewports(detail::BackendCommandBuffer& commandBuffer,
                                          std::span<const Viewport> viewports) override
        {
            return pipelines_.setViewports(queues_, commandBuffer, viewports);
        }

        [[nodiscard]] Status setScissors(detail::BackendCommandBuffer& commandBuffer,
                                         std::span<const ScissorRect> scissors) override
        {
            return pipelines_.setScissors(queues_, commandBuffer, scissors);
        }

        [[nodiscard]] Status setBlendConstants(detail::BackendCommandBuffer& commandBuffer,
                                               const BlendConstants& constants) override
        {
            return pipelines_.setBlendConstants(queues_, commandBuffer, constants);
        }

        [[nodiscard]] Status setDepthBias(detail::BackendCommandBuffer& commandBuffer, const DepthBias& bias) override
        {
            return pipelines_.setDepthBias(queues_, commandBuffer, bias);
        }

        [[nodiscard]] Status setDepthBounds(detail::BackendCommandBuffer& commandBuffer,
                                            const DepthBounds& bounds) override
        {
            return pipelines_.setDepthBounds(queues_, commandBuffer, bounds);
        }

        [[nodiscard]] Status setStencilReference(detail::BackendCommandBuffer& commandBuffer,
                                                 StencilFace face,
                                                 std::uint32_t reference) override
        {
            if ((face == StencilFace::front || face == StencilFace::back) &&
                !capabilities_.independentFrontAndBackStencilReference)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Independent front/back stencil references are unavailable on this adapter.",
                });
            }
            return pipelines_.setStencilReference(queues_, commandBuffer, face, reference);
        }

        [[nodiscard]] Status setShadingRate(detail::BackendCommandBuffer& commandBuffer,
                                            ShadingRate rate,
                                            const ShadingRateCombiners& combiners) override
        {
            return pipelines_.setShadingRate(queues_, commandBuffer, rate, combiners);
        }

        [[nodiscard]] Status setShadingRateImage(detail::BackendCommandBuffer& commandBuffer,
                                                 TextureHandle image) override
        {
            return pipelines_.setShadingRateImage(queues_, resources_, commandBuffer, image);
        }

        [[nodiscard]] Status bindVertexBuffers(detail::BackendCommandBuffer& commandBuffer,
                                               std::uint32_t firstSlot,
                                               std::span<const VertexBufferBinding> bindings) override
        {
            return pipelines_.bindVertexBuffers(queues_, resources_, commandBuffer, firstSlot, bindings);
        }

        [[nodiscard]] Status bindStreamOutputBuffers(detail::BackendCommandBuffer& commandBuffer,
                                                     std::uint32_t firstSlot,
                                                     std::span<const StreamOutputBufferBinding> bindings) override
        {
            return pipelines_.bindStreamOutputBuffers(queues_, resources_, commandBuffer, firstSlot, bindings);
        }

        [[nodiscard]] Status bindIndexBuffer(detail::BackendCommandBuffer& commandBuffer,
                                             BufferHandle buffer,
                                             IndexFormat format,
                                             std::uint64_t offset) override
        {
            return pipelines_.bindIndexBuffer(queues_, resources_, commandBuffer, buffer, format, offset);
        }

        [[nodiscard]] Status draw(detail::BackendCommandBuffer& commandBuffer,
                                  std::uint32_t vertexCount,
                                  std::uint32_t instanceCount,
                                  std::uint32_t firstVertex,
                                  std::uint32_t firstInstance) override
        {
            return pipelines_.draw(queues_, commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
        }

        [[nodiscard]] Status drawIndexed(detail::BackendCommandBuffer& commandBuffer,
                                         std::uint32_t indexCount,
                                         std::uint32_t instanceCount,
                                         std::uint32_t firstIndex,
                                         std::int32_t baseVertex,
                                         std::uint32_t firstInstance) override
        {
            return pipelines_.drawIndexed(
                queues_, commandBuffer, indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
        }

        [[nodiscard]] Status dispatchMesh(detail::BackendCommandBuffer& commandBuffer,
                                          std::uint32_t groupCountX,
                                          std::uint32_t groupCountY,
                                          std::uint32_t groupCountZ) override
        {
            return pipelines_.dispatchMesh(queues_, commandBuffer, groupCountX, groupCountY, groupCountZ);
        }

        [[nodiscard]] Status dispatchMeshIndirect(detail::BackendCommandBuffer& commandBuffer,
                                                  const IndirectMeshDispatchDesc& desc) override
        {
            return pipelines_.dispatchMeshIndirect(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Status drawIndirect(detail::BackendCommandBuffer& commandBuffer,
                                          const IndirectDrawDesc& desc) override
        {
            return pipelines_.drawIndirect(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Status drawIndexedIndirect(detail::BackendCommandBuffer& commandBuffer,
                                                 const IndirectDrawDesc& desc) override
        {
            return pipelines_.drawIndexedIndirect(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Status beginConditionalExecution(detail::BackendCommandBuffer& commandBuffer,
                                                       const ConditionalExecutionDesc& desc) override
        {
            return queries_.beginConditionalExecution(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Status endConditionalExecution(detail::BackendCommandBuffer& commandBuffer) override
        {
            return queries_.endConditionalExecution(queues_, commandBuffer);
        }

        [[nodiscard]] Result<QueryPoolHandle> createTimestampQueryPool(const TimestampQueryPoolDesc& desc) override
        {
            return guardDeviceLoss([&] { return queries_.createTimestampQueryPool(desc); },
                                   "timestamp query-pool creation");
        }

        [[nodiscard]] Status destroyQueryPool(QueryPoolHandle queryPool) override
        {
            return queries_.destroyQueryPool(queryPool);
        }

        [[nodiscard]] Status resetQueryPool(detail::BackendCommandBuffer& commandBuffer,
                                            QueryPoolHandle queryPool,
                                            std::uint32_t firstQuery,
                                            std::uint32_t queryCount) override
        {
            return queries_.resetQueryPool(queues_, commandBuffer, queryPool, firstQuery, queryCount);
        }

        [[nodiscard]] Status writeTimestamp(detail::BackendCommandBuffer& commandBuffer,
                                            QueryPoolHandle queryPool,
                                            std::uint32_t queryIndex) override
        {
            return queries_.writeTimestamp(queues_, commandBuffer, queryPool, queryIndex);
        }

        [[nodiscard]] Status beginGpuProfileScope(detail::BackendCommandBuffer& commandBuffer,
                                                  const GpuProfileScopeDesc& desc) override
        {
            return queries_.beginGpuProfileScope(queues_, debugMarkers_, commandBuffer, desc);
        }

        [[nodiscard]] Status endGpuProfileScope(detail::BackendCommandBuffer& commandBuffer) override
        {
            return queries_.endGpuProfileScope(queues_, debugMarkers_, commandBuffer);
        }

        [[nodiscard]] Status resolveTimestamps(detail::BackendCommandBuffer& commandBuffer,
                                               const TimestampQueryResolveDesc& desc) override
        {
            return queries_.resolveTimestamps(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Result<QueryPoolHandle> createOcclusionQueryPool(const OcclusionQueryPoolDesc& desc) override
        {
            return guardDeviceLoss([&] { return queries_.createOcclusionQueryPool(desc); },
                                   "occlusion query-pool creation");
        }

        [[nodiscard]] Status beginOcclusionQuery(detail::BackendCommandBuffer& commandBuffer,
                                                 QueryPoolHandle queryPool,
                                                 std::uint32_t queryIndex) override
        {
            return queries_.beginOcclusionQuery(queues_, commandBuffer, queryPool, queryIndex);
        }

        [[nodiscard]] Status endOcclusionQuery(detail::BackendCommandBuffer& commandBuffer) override
        {
            return queries_.endOcclusionQuery(queues_, commandBuffer);
        }

        [[nodiscard]] Status resolveOcclusionQueries(detail::BackendCommandBuffer& commandBuffer,
                                                     const OcclusionQueryResolveDesc& desc) override
        {
            return queries_.resolveOcclusionQueries(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Result<QueryPoolHandle> createPipelineStatisticsQueryPool(
            const PipelineStatisticsQueryPoolDesc& desc) override
        {
            return guardDeviceLoss([&] { return queries_.createPipelineStatisticsQueryPool(desc); },
                                   "pipeline-statistics query-pool creation");
        }

        [[nodiscard]] Status beginPipelineStatisticsQuery(detail::BackendCommandBuffer& commandBuffer,
                                                          QueryPoolHandle queryPool,
                                                          std::uint32_t queryIndex) override
        {
            return queries_.beginPipelineStatisticsQuery(queues_, commandBuffer, queryPool, queryIndex);
        }

        [[nodiscard]] Status endPipelineStatisticsQuery(detail::BackendCommandBuffer& commandBuffer) override
        {
            return queries_.endPipelineStatisticsQuery(queues_, commandBuffer);
        }

        [[nodiscard]] Status resolvePipelineStatisticsQueries(detail::BackendCommandBuffer& commandBuffer,
                                                              const PipelineStatisticsQueryResolveDesc& desc) override
        {
            return queries_.resolvePipelineStatisticsQueries(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Result<QueryPoolHandle> createStreamOutputStatisticsQueryPool(
            const StreamOutputStatisticsQueryPoolDesc& desc) override
        {
            return guardDeviceLoss([&] { return queries_.createStreamOutputStatisticsQueryPool(desc); },
                                   "stream-output statistics query-pool creation");
        }

        [[nodiscard]] Status beginStreamOutputStatisticsQuery(detail::BackendCommandBuffer& commandBuffer,
                                                              QueryPoolHandle queryPool,
                                                              std::uint32_t queryIndex) override
        {
            return queries_.beginStreamOutputStatisticsQuery(queues_, commandBuffer, queryPool, queryIndex);
        }

        [[nodiscard]] Status endStreamOutputStatisticsQuery(detail::BackendCommandBuffer& commandBuffer,
                                                            std::uint32_t stream) override
        {
            return queries_.endStreamOutputStatisticsQuery(queues_, commandBuffer, stream);
        }

        [[nodiscard]] Status resolveStreamOutputStatisticsQueries(
            detail::BackendCommandBuffer& commandBuffer, const StreamOutputStatisticsQueryResolveDesc& desc) override
        {
            return queries_.resolveStreamOutputStatisticsQueries(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Result<QueryPoolHandle> createAccelerationStructureCompactedSizeQueryPool(
            const AccelerationStructureCompactedSizeQueryPoolDesc& desc) override
        {
            if (!capabilities_.rayTracing)
            {
                return std::unexpected(
                    Error{ErrorCode::unsupported, 0, "Compacted-size queries require ray-tracing support."});
            }
            return guardDeviceLoss([&] { return queries_.createAccelerationStructureCompactedSizeQueryPool(desc); },
                                   "compacted-size query-pool creation");
        }

        [[nodiscard]] Status writeAccelerationStructureCompactedSize(detail::BackendCommandBuffer& commandBuffer,
                                                                     QueryPoolHandle queryPool,
                                                                     std::uint32_t queryIndex,
                                                                     AccelerationStructureHandle source) override
        {
            return queries_.writeAccelerationStructureCompactedSize(
                queues_, resources_, commandBuffer, queryPool, queryIndex, source);
        }

        [[nodiscard]] Status resolveAccelerationStructureCompactedSizes(
            detail::BackendCommandBuffer& commandBuffer,
            const AccelerationStructureCompactedSizeQueryResolveDesc& desc) override
        {
            return queries_.resolveAccelerationStructureCompactedSizes(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Result<QueryPoolHandle> createAccelerationStructureSerializationInfoQueryPool(
            const AccelerationStructureSerializationInfoQueryPoolDesc& desc) override
        {
            if (!capabilities_.rayTracing)
            {
                return std::unexpected(
                    Error{ErrorCode::unsupported, 0, "Serialization-info queries require ray-tracing support."});
            }
            return guardDeviceLoss([&] { return queries_.createAccelerationStructureSerializationInfoQueryPool(desc); },
                                   "serialization-info query-pool creation");
        }

        [[nodiscard]] Status writeAccelerationStructureSerializationInfo(detail::BackendCommandBuffer& commandBuffer,
                                                                         QueryPoolHandle queryPool,
                                                                         std::uint32_t queryIndex,
                                                                         AccelerationStructureHandle source) override
        {
            return queries_.writeAccelerationStructureSerializationInfo(
                queues_, resources_, commandBuffer, queryPool, queryIndex, source);
        }

        [[nodiscard]] Status resolveAccelerationStructureSerializationInfo(
            detail::BackendCommandBuffer& commandBuffer,
            const AccelerationStructureSerializationInfoQueryResolveDesc& desc) override
        {
            return queries_.resolveAccelerationStructureSerializationInfo(queues_, resources_, commandBuffer, desc);
        }

        [[nodiscard]] Result<SurfaceHandle> createSurface(const SurfaceDesc& desc) override
        {
            return guardDeviceLoss([&] { return swapchains_.createSurface(desc); }, "surface creation");
        }

        [[nodiscard]] Result<SurfaceCapabilities> querySurfaceCapabilities(SurfaceHandle surface) const override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return swapchains_.querySurfaceCapabilities(surface);
        }

        [[nodiscard]] Status destroySurface(SurfaceHandle surface) override
        {
            return swapchains_.destroySurface(surface);
        }

        [[nodiscard]] Result<SwapchainHandle> createSwapchain(const SwapchainDesc& desc) override
        {
            return guardDeviceLoss([&] { return swapchains_.createSwapchain(queues_, resources_, views_, desc); },
                                   "swapchain creation");
        }

        [[nodiscard]] Status destroySwapchain(SwapchainHandle swapchain) override
        {
            std::optional<Error> observedLoss;
            auto destroyed = swapchains_.destroySwapchain(queues_,
                                                          resources_,
                                                          views_,
                                                          swapchain,
                                                          operationalState() == DeviceOperationalState::lost,
                                                          observedLoss);
            if (observedLoss.has_value())
            {
                static_cast<void>(
                    observeDeviceLoss(Status{std::unexpected(std::move(*observedLoss))}, "swapchain destruction"));
            }
            return destroyed;
        }

        [[nodiscard]] Status resizeSwapchain(SwapchainHandle swapchain,
                                             std::uint32_t width,
                                             std::uint32_t height) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(swapchains_.resizeSwapchain(queues_, resources_, views_, swapchain, width, height),
                                     "swapchain resize");
        }

        [[nodiscard]] Result<AcquiredSwapchainImage> acquireNextImage(SwapchainHandle swapchain) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(swapchains_.acquireNextImage(queues_, swapchain), "swapchain acquisition");
        }

        [[nodiscard]] bool hasAcquiredSwapchainImage(SwapchainHandle swapchain) const noexcept override
        {
            return swapchains_.hasAcquiredImage(swapchain);
        }

        [[nodiscard]] Status cancelSwapchainImage(SwapchainHandle swapchain) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return swapchains_.cancelSwapchainImage(swapchain);
        }

        [[nodiscard]] Result<PresentStatus> present(SwapchainHandle swapchain, TimelinePoint renderComplete) override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return observeDeviceLoss(swapchains_.present(queues_, swapchain, renderComplete), "presentation");
        }

        [[nodiscard]] Result<SwapchainPresentationInfo> querySwapchainPresentation(
            SwapchainHandle swapchain) const override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return swapchains_.queryPresentation(swapchain);
        }

        [[nodiscard]] Result<SwapchainFrameTiming> querySwapchainFrameTiming(SwapchainHandle swapchain) const override
        {
            if (auto loss = deviceLossError())
            {
                return std::unexpected(std::move(*loss));
            }
            return swapchains_.queryFrameTiming(swapchain);
        }

    private:
        template<typename Operation>
        [[nodiscard]] std::invoke_result_t<Operation&> guardDeviceLoss(Operation&& operation,
                                                                       std::string_view context) const
        {
            using OperationResult = std::invoke_result_t<Operation&>;
            if (auto loss = deviceLossError())
            {
                return OperationResult{std::unexpected(std::move(*loss))};
            }
            auto result = operation();
            if (!result)
            {
                return observeDeviceLoss(std::move(result), context);
            }
            return result;
        }

        void reclassifyRemovedDeviceFailure(Error& error) const
        {
            const bool nativeFailure =
                error.code == ErrorCode::nativeFailure || error.code == ErrorCode::outOfMemory || error.nativeCode != 0;
            if (error.code == ErrorCode::deviceLost || !nativeFailure)
            {
                return;
            }
            const HRESULT removedReason = device_->GetDeviceRemovedReason();
            if (SUCCEEDED(removedReason))
            {
                return;
            }
            std::ostringstream message;
            message << error.message << (error.message.empty() ? "" : " ")
                    << "The D3D12 device was removed: ID3D12Device::GetDeviceRemovedReason returned HRESULT 0x"
                    << std::hex << std::uppercase << static_cast<std::uint32_t>(removedReason) << '.';
            error = Error{ErrorCode::deviceLost, static_cast<std::int64_t>(removedReason), message.str()};
        }

        template<typename T>
        [[nodiscard]] Result<T> observeDeviceLoss(Result<T> result, std::string_view context) const
        {
            if (!result)
            {
                reclassifyRemovedDeviceFailure(result.error());
            }
            if (!result && result.error().code == ErrorCode::deviceLost)
            {
                bool publish = false;
                const DeviceFaultReport* reportToLog = nullptr;
                std::optional<Error> publishedError;
                {
                    const std::scoped_lock lock(deviceLossMutex_);
                    if (deviceLossError_ == nullptr)
                    {
                        Error terminalError = result.error();
                        if (terminalError.nativeCode == 0)
                        {
                            const HRESULT removedReason = device_->GetDeviceRemovedReason();
                            if (FAILED(removedReason))
                            {
                                terminalError.nativeCode = static_cast<std::int64_t>(removedReason);
                            }
                        }
                        auto captured = d3d12::captureDeviceFaultReport(*device_.Get());
                        DeviceFaultReport report = captured ? std::move(*captured)
                                                            : DeviceFaultReport{
                                                                  .status = DeviceFaultStatus::unknown,
                                                                  .nativeReason = terminalError.nativeCode,
                                                              };
                        if (report.status == DeviceFaultStatus::healthy ||
                            (report.nativeReason == 0 && terminalError.nativeCode != 0))
                        {
                            report.status = deviceFaultStatusFromNativeReason(terminalError.nativeCode);
                            report.nativeReason = terminalError.nativeCode;
                        }
                        auto errorOwner = std::make_shared<const Error>(std::move(terminalError));
                        auto reportOwner = std::make_shared<const DeviceFaultReport>(std::move(report));
                        publishedError.emplace(*errorOwner);
                        deviceLossError_ = std::move(errorOwner);
                        deviceFaultReport_ = std::move(reportOwner);
                        reportToLog = deviceFaultReport_.get();
                        operationalState_.store(DeviceOperationalState::lost, std::memory_order_release);
                        publish = true;
                        countDiagnostic(DiagnosticSeverity::error);
                    }
                    else
                    {
                        publishedError.emplace(*deviceLossError_);
                    }
                }

                if (publish)
                {
                    emitDeviceFaultReport(*reportToLog, context, callback_, callbackUserData_);
                }
                return std::unexpected(std::move(*publishedError));
            }

            auto terminalError = deviceLossError();
            if (terminalError.has_value())
            {
                return std::unexpected(std::move(*terminalError));
            }
            return result;
        }

        static void CALLBACK messageCallback(D3D12_MESSAGE_CATEGORY,
                                             D3D12_MESSAGE_SEVERITY severity,
                                             D3D12_MESSAGE_ID id,
                                             LPCSTR description,
                                             void* context) noexcept
        {
            auto& self = *static_cast<D3D12Device*>(context);
            const auto portableSeverity = diagnosticSeverity(severity);
            self.countDiagnostic(portableSeverity);

            const DiagnosticMessage message{
                portableSeverity,
                static_cast<std::uint32_t>(id),
                description != nullptr ? std::string_view(description) : std::string_view{},
            };
            if (self.callback_ != nullptr)
            {
                self.callback_(message, self.callbackUserData_);
            }
        }

        void countDiagnostic(DiagnosticSeverity severity) const noexcept
        {
            switch (severity)
            {
            case DiagnosticSeverity::corruption:
                corruptions_.fetch_add(1, std::memory_order_relaxed);
                break;
            case DiagnosticSeverity::error:
                errors_.fetch_add(1, std::memory_order_relaxed);
                break;
            case DiagnosticSeverity::warning:
                warnings_.fetch_add(1, std::memory_order_relaxed);
                break;
            case DiagnosticSeverity::information:
                information_.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }

        ProcessDeviceRegistration debugLayerRegistration_;
        ComPtr<ID3D12Device> device_;
        ComPtr<IDXGIAdapter3> budgetAdapter_;
        SwapchainManager swapchains_;
        ResourceManager resources_;
        ViewManager views_;
        SamplerManager samplers_;
        BindingManager bindings_;
        MipmapGenerator mipmaps_;
        PipelineManager pipelines_;
        QueryManager queries_;
        QueueManager queues_;
        DebugMarkerManager debugMarkers_;
        ComPtr<ID3D12InfoQueue1> infoQueue_;
        DWORD callbackCookie_ = 0;
        std::uint64_t deviceToken_ = 0;
        AdapterInfo adapter_;
        DeviceCapabilities capabilities_;
        DiagnosticCallback callback_ = nullptr;
        void* callbackUserData_ = nullptr;
        mutable std::atomic_uint64_t information_ = 0;
        mutable std::atomic_uint64_t warnings_ = 0;
        mutable std::atomic_uint64_t errors_ = 0;
        mutable std::atomic_uint64_t corruptions_ = 0;
        mutable std::mutex deviceLossMutex_;
        mutable std::shared_ptr<const Error> deviceLossError_;
        mutable std::shared_ptr<const DeviceFaultReport> deviceFaultReport_;
        mutable std::atomic<DeviceOperationalState> operationalState_ = DeviceOperationalState::active;
    };

    [[nodiscard]] Result<std::vector<AdapterInfo>> enumerateAdapters()
    {
        auto factory = createFactory(false);
        if (!factory)
        {
            return std::unexpected(std::move(factory.error()));
        }

        auto& debugLayer = debugLayerProcessState();
        const std::scoped_lock processLock(debugLayer.mutex);
        auto hardware = enumerateHardwareAdapters(*factory->Get(), DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE);
        if (!hardware)
        {
            return std::unexpected(std::move(hardware.error()));
        }

        std::vector<AdapterInfo> result;
        result.reserve(hardware->size() + 1);
        for (const auto& adapter : *hardware)
        {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc)))
            {
                auto info = adapterInfo(desc);
                info.type = probeHardwareAdapterType(*adapter.Get());
                result.push_back(std::move(info));
            }
        }

        auto warp = warpAdapter(*factory->Get());
        if (warp)
        {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED((*warp)->GetDesc1(&desc)))
            {
                result.push_back(adapterInfo(desc));
            }
        }

        return result;
    }

    test::DebugLayerProcessSnapshot test::debugLayerProcessSnapshot() noexcept
    {
        auto& state = debugLayerProcessState();
        const std::scoped_lock lock(state.mutex);
        return {
            .debugLayerEnabled = state.debugLayerEnabled,
            .gpuValidationEnabled = state.gpuValidationEnabled,
            .devicesWithoutDebugLayer = state.devicesWithoutDebugLayer,
            .devicesWithoutGpuValidation = state.devicesWithoutGpuValidation,
        };
    }

    bool test::debugLayerProcessLockAvailable() noexcept
    {
        auto& state = debugLayerProcessState();
        if (!state.mutex.try_lock())
        {
            return false;
        }
        state.mutex.unlock();
        return true;
    }

    [[nodiscard]] Result<std::unique_ptr<detail::BackendDevice>> createDevice(const DeviceDesc& desc)
    {
        auto& debugLayer = debugLayerProcessState();
        ProcessDeviceRegistration debugLayerRegistration;
        ValidationPlan validation;
        ComPtr<IDXGIAdapter1> adapter;
        DXGI_ADAPTER_DESC1 nativeAdapterInfo{};
        ComPtr<ID3D12Device> nativeDevice;
        {
            const std::scoped_lock processLock(debugLayer.mutex);
            auto plan = enableValidation(desc, debugLayer);
            if (!plan)
            {
                return std::unexpected(std::move(plan.error()));
            }
            validation = *plan;

            auto factory = createFactory(validation.enabled);
            if (!factory && validation.enabled && !desc.requireValidation)
            {
                factory = createFactory(false);
            }
            if (!factory)
            {
                return std::unexpected(std::move(factory.error()));
            }

            auto selected = selectAdapter(*factory->Get(), desc);
            if (!selected)
            {
                return std::unexpected(std::move(selected.error()));
            }
            adapter = std::move(*selected);

            const HRESULT descResult = adapter->GetDesc1(&nativeAdapterInfo);
            if (FAILED(descResult))
            {
                return std::unexpected(nativeError(ErrorCode::nativeFailure, descResult, "IDXGIAdapter1::GetDesc1"));
            }

            configureDred(desc.enableDred);
            const HRESULT deviceResult = createNativeDevice(adapter.Get(), IID_PPV_ARGS(&nativeDevice));
            if (FAILED(deviceResult))
            {
                return std::unexpected(nativeError(ErrorCode::deviceCreationFailed, deviceResult, "D3D12CreateDevice"));
            }

            const bool debugLayerActive = hasActiveDebugLayer(*nativeDevice.Get());
            if (debugLayerActive)
            {
                debugLayer.debugLayerEnabled = true;
            }
            debugLayerRegistration = ProcessDeviceRegistration::registerWhileLocked(
                debugLayer, debugLayerActive, debugLayerActive && debugLayer.gpuValidationEnabled);
        }

        ComPtr<IDXGIAdapter3> budgetAdapter;
        static_cast<void>(adapter.As(&budgetAdapter));
        auto portableAdapterInfo = adapterInfo(nativeAdapterInfo);
        if (portableAdapterInfo.type != AdapterType::software)
        {
            portableAdapterInfo.type = hardwareAdapterType(*nativeDevice.Get());
        }
        auto capabilities = queryCapabilities(*nativeDevice.Get(), portableAdapterInfo);
        if (test::forceLegacyBarriers().load(std::memory_order_relaxed))
        {
            capabilities.enhancedBarriers = false;
        }
        capabilities.memoryBudgets = budgetAdapter != nullptr;
        auto* allocation = new (std::nothrow) D3D12Device(std::move(debugLayerRegistration),
                                                          std::move(nativeDevice),
                                                          std::move(budgetAdapter),
                                                          std::move(portableAdapterInfo),
                                                          capabilities,
                                                          desc.diagnosticCallback,
                                                          desc.diagnosticUserData);
        if (allocation == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate the D3D12 device backend.",
            });
        }

        std::unique_ptr<D3D12Device> device(allocation);
        auto resources = device->initializeResources(*adapter.Get(), capabilities.enhancedBarriers);
        if (!resources)
        {
            return std::unexpected(std::move(resources.error()));
        }
        auto diagnostics = device->registerDiagnostics(validation.enabled);
        if (!diagnostics && desc.requireValidation)
        {
            return std::unexpected(std::move(diagnostics.error()));
        }

        auto queues = device->initializeQueues();
        if (!queues)
        {
            return std::unexpected(std::move(queues.error()));
        }
        auto queries = device->initializeQueries(capabilities.enhancedBarriers);
        if (!queries)
        {
            return std::unexpected(std::move(queries.error()));
        }

        if (validation.blockingDevices > 0)
        {
            const auto warning =
                "D3D12 validation was requested but is not enabled for this device: " +
                blockedValidationReason(validation.blockingDevices, ProcessValidationFeature::debugLayer);
            device->publishDiagnostic(DiagnosticSeverity::warning, warning);
        }
        if (validation.gpuBlockingDevices > 0)
        {
            const auto warning =
                "D3D12 GPU-based validation was requested but is not enabled for this device, which runs with standard "
                "validation instead: " +
                blockedValidationReason(validation.gpuBlockingDevices, ProcessValidationFeature::gpuBasedValidation);
            device->publishDiagnostic(DiagnosticSeverity::warning, warning);
        }

        return std::unique_ptr<detail::BackendDevice>(std::move(device));
    }

} // namespace rachis::d3d12
