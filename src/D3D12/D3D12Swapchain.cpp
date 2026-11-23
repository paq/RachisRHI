#include "D3D12/D3D12Swapchain.h"

#include "Common/AcquireDeadline.h"
#include "Common/HandleTable.h"
#include "Common/TimedMutex.h"
#include "D3D12/D3D12Format.h"
#include "D3D12/D3D12NativeHelpers.h"
#include "D3D12/D3D12PresentationProgress.h"
#include "D3D12/D3D12RuntimePolicies.h"
#include "D3D12/D3D12TestHooks.h"

#include <dcomp.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <Windows.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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

        static_assert(DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT ==
                      kDxgiSwapChainFlagFrameLatencyWaitableObject);
        static_assert(DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING == kDxgiSwapChainFlagAllowTearing);

        struct WaitableHandleCloser final
        {
            void operator()(void* handle) const noexcept
            {
                if (handle != nullptr)
                {
                    static_cast<void>(CloseHandle(handle));
                }
            }
        };

        using UniqueWaitableHandle = std::unique_ptr<void, WaitableHandleCloser>;

        constexpr DWORD frameLatencyWaitTimeoutMs = 1000;

        struct SurfaceRecord final
        {
            SurfaceKind kind = SurfaceKind::win32;
            HWND window = nullptr;
            std::string label;
            bool swapchainClaimed = false;
        };

        struct SwapchainBuffer final
        {
            TextureHandle texture = {};
            TextureViewHandle renderTargetView = {};
            std::shared_ptr<SwapchainImageLeaseState> lease = {};
            TimelinePoint lastUse = {};
        };

        struct SwapchainRecord final
        {
            mutable rachis::detail::TimedMutex recordMutex;
            std::shared_ptr<SurfaceRecord> surface;
            ComPtr<IDXGISwapChain4> swapchain;
            ComPtr<IDCompositionDevice> compositionDevice;
            ComPtr<IDCompositionTarget> compositionTarget;
            ComPtr<IDCompositionVisual> compositionVisual;
            UniqueWaitableHandle frameLatencyWaitableObject;
            SwapchainLatencySlotState latencySlot;
            SwapchainDesc desc;
            std::string label;
            std::vector<SwapchainBuffer> buffers;
            PresentationProgressTracker presentation;
            std::uint64_t generation = 1;
            std::uint32_t acquiredIndex = 0;
            bool acquired = false;
            bool poisoned = false;
            bool deviceLost = false;
            bool requiresRefresh = false;
            bool allowTearing = false;
            bool destroyed = false;
        };

        class PendingSurfaceClaim final
        {
        public:
            PendingSurfaceClaim(std::shared_ptr<SurfaceRecord> surface, rachis::detail::TimedMutex& mutex) noexcept
                : surface_(std::move(surface)), mutex_(&mutex)
            {
            }

            ~PendingSurfaceClaim()
            {
                if (surface_ != nullptr)
                {
                    const std::scoped_lock lock(*mutex_);
                    surface_->swapchainClaimed = false;
                }
            }

            PendingSurfaceClaim(const PendingSurfaceClaim&) = delete;
            PendingSurfaceClaim& operator=(const PendingSurfaceClaim&) = delete;
            PendingSurfaceClaim(PendingSurfaceClaim&&) = delete;
            PendingSurfaceClaim& operator=(PendingSurfaceClaim&&) = delete;

            void publish() noexcept
            {
                surface_.reset();
            }

        private:
            std::shared_ptr<SurfaceRecord> surface_;
            rachis::detail::TimedMutex* mutex_ = nullptr;
        };

        [[nodiscard]] Error invalidSwapchainError() noexcept
        {
            return Error{ErrorCode::invalidArgument, 0, "The swapchain handle is invalid or stale."};
        }

        [[nodiscard]] Error referencedSwapchainImagesError(std::int64_t nativeCode)
        {
            return Error{
                ErrorCode::invalidState,
                nativeCode,
                "Destroy views of swapchain images, and binding sets that use them, before resizing the swapchain.",
            };
        }

        [[nodiscard]] std::shared_ptr<SwapchainRecord> findSwapchainRecord(
            rachis::detail::TimedMutex& mutex,
            const detail::HandleTable<std::shared_ptr<SwapchainRecord>, SwapchainHandle>& swapchains,
            SwapchainHandle swapchain) noexcept
        {
            const std::scoped_lock lock(mutex);
            const auto* found = swapchains.find(swapchain);
            return found != nullptr ? *found : nullptr;
        }

        [[nodiscard]] std::optional<std::shared_ptr<SwapchainRecord>> findSwapchainRecord(
            rachis::detail::TimedMutex& mutex,
            const detail::HandleTable<std::shared_ptr<SwapchainRecord>, SwapchainHandle>& swapchains,
            SwapchainHandle swapchain,
            const rachis::detail::AcquireDeadline& deadline)
        {
            const auto lock = deadline.lock(mutex);
            if (!lock.owns_lock())
            {
                return std::nullopt;
            }
            const auto* found = swapchains.find(swapchain);
            return found != nullptr ? *found : nullptr;
        }

        [[nodiscard]] std::shared_ptr<SurfaceRecord> findSurfaceRecord(
            rachis::detail::TimedMutex& mutex,
            const detail::HandleTable<std::shared_ptr<SurfaceRecord>, SurfaceHandle>& surfaces,
            SurfaceHandle surface)
        {
            const std::scoped_lock lock(mutex);
            const auto* found = surfaces.find(surface);
            return found != nullptr ? *found : nullptr;
        }

        [[nodiscard]] constexpr DXGI_FORMAT nativeFormat(Format format) noexcept
        {
            switch (format)
            {
            case Format::rgba8Unorm:
            case Format::rgba8UnormSrgb:
                return DXGI_FORMAT_R8G8B8A8_UNORM;
            case Format::bgra8Unorm:
            case Format::bgra8UnormSrgb:
                return DXGI_FORMAT_B8G8R8A8_UNORM;
            default:
                return DXGI_FORMAT_UNKNOWN;
            }
        }

        [[nodiscard]] constexpr bool viewsCastBackBufferFormat(Format value) noexcept
        {
            return format::view(value) != nativeFormat(value);
        }

        [[nodiscard]] constexpr DXGI_COLOR_SPACE_TYPE nativeColorSpace(SurfaceColorSpace colorSpace) noexcept
        {
            switch (colorSpace)
            {
            case SurfaceColorSpace::srgbNonlinear:
                return DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
            case SurfaceColorSpace::extendedSrgbLinear:
                return DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
            case SurfaceColorSpace::hdr10St2084:
                return DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
            }
            return DXGI_COLOR_SPACE_CUSTOM;
        }

        [[nodiscard]] Result<SurfaceCapabilities> capabilitiesForSurface(const SurfaceRecord& surface,
                                                                         bool tearingSupported)
        {
            if (surface.window == nullptr || IsWindow(surface.window) == FALSE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The surface's native window is no longer live.",
                });
            }

            RECT client{};
            if (GetClientRect(surface.window, &client) == FALSE)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    static_cast<std::int64_t>(GetLastError()),
                    "GetClientRect failed while querying surface capabilities.",
                });
            }

            auto presentModes = PresentModeMask::fifo;
            if (tearingSupported && surface.kind == SurfaceKind::win32)
            {
                presentModes |= PresentModeMask::immediate;
            }

            SurfaceCapabilities result{
                .minImageCount = 2,
                .maxImageCount = 16,
                .minWidth = 1,
                .minHeight = 1,
                .maxWidth = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
                .maxHeight = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
                .preferredWidth = static_cast<std::uint32_t>((std::max)(client.right - client.left, 0L)),
                .preferredHeight = static_cast<std::uint32_t>((std::max)(client.bottom - client.top, 0L)),
                .usages = TextureUsage::renderTarget | TextureUsage::copySource | TextureUsage::copyDestination |
                          TextureUsage::sampled,
                .presentModes = presentModes,
            };
            result.formats = {
                SurfaceFormat{.format = Format::bgra8Unorm, .colorSpace = SurfaceColorSpace::srgbNonlinear},
                SurfaceFormat{.format = Format::rgba8Unorm, .colorSpace = SurfaceColorSpace::srgbNonlinear},

                SurfaceFormat{.format = Format::bgra8UnormSrgb, .colorSpace = SurfaceColorSpace::srgbNonlinear},
                SurfaceFormat{.format = Format::rgba8UnormSrgb, .colorSpace = SurfaceColorSpace::srgbNonlinear},
            };
            return result;
        }

        [[nodiscard]] constexpr bool validPresentMode(PresentMode mode) noexcept
        {
            return mode == PresentMode::fifo || mode == PresentMode::immediate;
        }

        [[nodiscard]] constexpr bool usageIsSubset(TextureUsage value, TextureUsage supported) noexcept
        {
            using Value = std::underlying_type_t<TextureUsage>;
            return (static_cast<Value>(value) & ~static_cast<Value>(supported)) == 0;
        }

        [[nodiscard]] bool supportsSurfaceFormat(const SurfaceCapabilities& capabilities, const SwapchainDesc& desc)
        {
            return std::ranges::find(capabilities.formats,
                                     SurfaceFormat{.format = desc.format, .colorSpace = desc.colorSpace}) !=
                   capabilities.formats.end();
        }

        [[nodiscard]] constexpr bool validSwapchainUsage(TextureUsage usage) noexcept
        {
            constexpr auto allowed = TextureUsage::renderTarget | TextureUsage::copySource |
                                     TextureUsage::copyDestination | TextureUsage::sampled;
            using Value = std::underlying_type_t<TextureUsage>;
            return hasAny(usage, TextureUsage::renderTarget) &&
                   (static_cast<Value>(usage) & ~static_cast<Value>(allowed)) == 0;
        }

        [[nodiscard]] constexpr DXGI_USAGE nativeUsage(TextureUsage usage) noexcept
        {
            auto result = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            if (hasAny(usage, TextureUsage::sampled))
            {
                result |= DXGI_USAGE_SHADER_INPUT;
            }
            return result;
        }

        [[nodiscard]] bool queryTearingSupport(IDXGIFactory7& factory) noexcept
        {
            BOOL supported = FALSE;
            return SUCCEEDED(factory.CheckFeatureSupport(
                       DXGI_FEATURE_PRESENT_ALLOW_TEARING, &supported, sizeof(supported))) &&
                   supported == TRUE;
        }

        [[nodiscard]] bool queryFullyTypedFormatCasting(ID3D12Device& device) noexcept
        {
#ifdef RACHIS_TESTING
            if (test::forceFullyTypedFormatCastingUnsupported().load(std::memory_order_relaxed))
            {
                return false;
            }
#endif
            D3D12_FEATURE_DATA_D3D12_OPTIONS3 options{};
            return SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &options, sizeof(options))) &&
                   options.CastingFullyTypedFormatSupported != FALSE;
        }

        void releaseBackBuffers(ResourceManager& resources, ViewManager& views, SwapchainRecord& record) noexcept
        {
            for (const auto& buffer : record.buffers)
            {
                if (buffer.renderTargetView.isValid())
                {
                    if (!views.destroyTextureView(buffer.renderTargetView, true))
                    {
                        std::terminate();
                    }
                }
                if (buffer.texture.isValid())
                {
                    if (!resources.destroyTexture(buffer.texture, true))
                    {
                        std::terminate();
                    }
                }
            }
            record.buffers.clear();
        }

        void releaseNativeSwapchain(SwapchainRecord& record) noexcept
        {
            if (record.compositionVisual != nullptr)
            {
                static_cast<void>(record.compositionVisual->SetContent(nullptr));
            }
            if (record.compositionTarget != nullptr)
            {
                static_cast<void>(record.compositionTarget->SetRoot(nullptr));
            }
            if (record.compositionDevice != nullptr)
            {
                static_cast<void>(record.compositionDevice->Commit());
            }
            record.compositionVisual.Reset();
            record.compositionTarget.Reset();
            record.compositionDevice.Reset();
            record.frameLatencyWaitableObject.reset();
            record.latencySlot.reset();
            const auto remainingReferences = record.swapchain.Reset();
#ifdef RACHIS_TESTING
            if (remainingReferences != 0)
            {
                test::nativeSwapchainsReleasedWhileReferenced().fetch_add(1, std::memory_order_relaxed);
            }
#else
            static_cast<void>(remainingReferences);
#endif
        }

        [[nodiscard]] HRESULT nativeBackBuffer(SwapchainRecord& record,
                                               std::uint32_t index,
                                               ComPtr<ID3D12Resource>& native)
        {
#ifdef RACHIS_TESTING
            const auto injectedResult =
                static_cast<HRESULT>(test::nextSwapchainGetBufferResult().exchange(0, std::memory_order_relaxed));
            if (injectedResult != 0)
            {
                return injectedResult;
            }
#endif
            return record.swapchain->GetBuffer(index, IID_PPV_ARGS(&native));
        }

        [[nodiscard]] Status createBackBuffers(ResourceManager& resources, ViewManager& views, SwapchainRecord& record)
        {
            record.buffers.reserve(record.desc.imageCount);
            for (std::uint32_t index = 0; index < record.desc.imageCount; ++index)
            {
                ComPtr<ID3D12Resource> native;
                const HRESULT bufferResult = nativeBackBuffer(record, index, native);
                if (FAILED(bufferResult))
                {
                    releaseBackBuffers(resources, views, record);
                    return std::unexpected(
                        nativeError(ErrorCode::nativeFailure, bufferResult, "IDXGISwapChain4::GetBuffer"));
                }

                const auto imageLabel = record.label + " image " + std::to_string(index);
                auto lease = std::make_shared<SwapchainImageLeaseState>();
                auto texture = resources.importSwapchainTexture(*native.Get(),
                                                                TextureDesc{
                                                                    .dimension = TextureDimension::texture2D,
                                                                    .width = record.desc.width,
                                                                    .height = record.desc.height,
                                                                    .depthOrArrayLayers = 1,
                                                                    .mipLevels = 1,
                                                                    .sampleCount = 1,
                                                                    .format = record.desc.format,
                                                                    .usage = record.desc.usage,
                                                                    .label = imageLabel,
                                                                },
                                                                lease);
                if (!texture)
                {
                    releaseBackBuffers(resources, views, record);
                    return std::unexpected(std::move(texture.error()));
                }
                auto view = views.createTextureView(resources,
                                                    TextureViewDesc{
                                                        .texture = *texture,
                                                        .usage = TextureViewUsage::renderTarget,
                                                        .dimension = TextureViewDimension::texture2D,
                                                        .format = record.desc.format,
                                                        .aspect = TextureAspect::color,
                                                        .firstMipLevel = 0,
                                                        .mipLevelCount = 1,
                                                        .firstArrayLayer = 0,
                                                        .arrayLayerCount = 1,
                                                        .label = imageLabel,
                                                    },
                                                    true);
                if (!view)
                {
                    if (!resources.destroyTexture(*texture, true))
                    {
                        std::terminate();
                    }
                    releaseBackBuffers(resources, views, record);
                    return std::unexpected(std::move(view.error()));
                }
                record.buffers.push_back(SwapchainBuffer{
                    .texture = *texture,
                    .renderTargetView = *view,
                    .lease = std::move(lease),
                });
            }
            return {};
        }

        [[nodiscard]] bool readPresentCount(IDXGISwapChain4& swapchain, std::uint32_t& count) noexcept
        {
#ifdef RACHIS_TESTING
            const auto injectedResult = test::nextPresentCountResult().exchange(0, std::memory_order_relaxed);
            if (FAILED(static_cast<HRESULT>(injectedResult)))
            {
                return false;
            }
#endif
            UINT native = 0;
            if (FAILED(swapchain.GetLastPresentCount(&native)))
            {
                return false;
            }
            count = native;
            return true;
        }

        [[nodiscard]] bool isDeviceRemoval(HRESULT result) noexcept
        {
            return isD3D12DeviceLossHresult(static_cast<std::int32_t>(result));
        }

        [[nodiscard]] Status waitForBackBuffer(QueueManager& queues, SwapchainRecord& record, TimelinePoint lastUse)
        {
#ifdef RACHIS_TESTING
            if (test::nextSwapchainWaitDeviceLoss().exchange(false, std::memory_order_relaxed))
            {
                record.deviceLost = true;
                return std::unexpected(Error{
                    ErrorCode::deviceLost,
                    0,
                    "The D3D12 fence reported device removal while waiting for a swapchain back buffer.",
                });
            }
#endif
            if (!lastUse.isValid())
            {
                return {};
            }
            auto waited = queues.wait(lastUse);
            if (!waited && waited.error().code == ErrorCode::deviceLost)
            {
                record.deviceLost = true;
            }
            return waited;
        }

        [[nodiscard]] Status waitForBackBuffers(QueueManager& queues, SwapchainRecord& record)
        {
            for (const auto& buffer : record.buffers)
            {
                auto waited = waitForBackBuffer(queues, record, buffer.lastUse);
                if (!waited)
                {
                    return waited;
                }
            }
            queues.collectGarbage();
            return {};
        }

        [[nodiscard]] Status validateSwapchainDescForSurface(const SurfaceCapabilities& capabilities,
                                                             const SwapchainDesc& desc,
                                                             bool fullyTypedFormatCasting)
        {
            const bool extentInRange = desc.width >= capabilities.minWidth && desc.height >= capabilities.minHeight &&
                                       desc.width <= capabilities.maxWidth && desc.height <= capabilities.maxHeight;
            const bool imageCountInRange =
                desc.imageCount >= capabilities.minImageCount && desc.imageCount <= capabilities.maxImageCount;
            if (!extentInRange || !imageCountInRange || !hasAny(desc.usage, TextureUsage::renderTarget))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A swapchain requires a supported non-empty extent and image count plus render-target usage.",
                });
            }
            if (!supportsSurfaceFormat(capabilities, desc) ||
                !supportsPresentMode(capabilities.presentModes, desc.presentMode) ||
                !usageIsSubset(desc.usage, capabilities.usages) || nativeFormat(desc.format) == DXGI_FORMAT_UNKNOWN ||
                !validSwapchainUsage(desc.usage))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The swapchain format, color space, present mode, or usage is not supported by this surface.",
                });
            }
            if (hasAny(desc.usage, TextureUsage::sampled) && viewsCastBackBufferFormat(desc.format) &&
                !fullyTypedFormatCasting)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Sampled usage on an sRGB swapchain format requires a D3D12 device that can cast fully typed "
                    "formats, because flip-model back buffers are stored as UNORM.",
                });
            }
            return {};
        }

        [[nodiscard]] bool windowHasClaimedSwapchain(
            const detail::HandleTable<std::shared_ptr<SurfaceRecord>, SurfaceHandle>& surfaces,
            const detail::HandleTable<std::shared_ptr<SwapchainRecord>, SwapchainHandle>& swapchains,
            const std::shared_ptr<SurfaceRecord>& surface)
        {
            bool claimed = false;
            const auto inspect = [&](const std::shared_ptr<SurfaceRecord>& other) {
                if (other != nullptr && other != surface && other->swapchainClaimed && other->window == surface->window)
                {
                    claimed = true;
                }
            };
            surfaces.forEach([&](SurfaceHandle, const std::shared_ptr<SurfaceRecord>& other) { inspect(other); });
            swapchains.forEach([&](SwapchainHandle, const std::shared_ptr<SwapchainRecord>& record) {
                if (record != nullptr)
                {
                    inspect(record->surface);
                }
            });
            return claimed;
        }

        [[nodiscard]] Status claimSurfaceForSwapchain(
            rachis::detail::TimedMutex& mutex,
            const detail::HandleTable<std::shared_ptr<SurfaceRecord>, SurfaceHandle>& surfaces,
            const detail::HandleTable<std::shared_ptr<SwapchainRecord>, SwapchainHandle>& swapchains,
            SurfaceHandle handle,
            const std::shared_ptr<SurfaceRecord>& surface)
        {
            const std::scoped_lock lock(mutex);
            const auto* found = surfaces.find(handle);
            if (found == nullptr || *found != surface)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The swapchain surface handle became stale before creation.",
                });
            }
            if (surface->swapchainClaimed)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A Win32 surface can own only one live or pending flip-model swapchain.",
                });
            }
            if (windowHasClaimedSwapchain(surfaces, swapchains, surface))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The native window already has a live swapchain.",
                });
            }
            surface->swapchainClaimed = true;
            return {};
        }

        [[nodiscard]] constexpr bool isWin32SurfaceKind(SurfaceKind kind) noexcept
        {
            return kind == SurfaceKind::win32 || kind == SurfaceKind::win32Composition;
        }

        [[nodiscard]] constexpr bool allowsTearing(SurfaceKind kind, PresentMode mode) noexcept
        {
            return kind == SurfaceKind::win32 && mode == PresentMode::immediate;
        }

        struct NativeSwapchainResources final
        {
            ComPtr<IDXGISwapChain4> swapchain;
            ComPtr<IDCompositionDevice> compositionDevice;
            ComPtr<IDCompositionTarget> compositionTarget;
            ComPtr<IDCompositionVisual> compositionVisual;
        };

        using DCompositionCreateDeviceFn = HRESULT(WINAPI*)(IDXGIDevice*, REFIID, void**);

        [[nodiscard]] Result<DCompositionCreateDeviceFn> loadDCompositionCreateDevice()
        {
            static const auto state = []() -> std::pair<DCompositionCreateDeviceFn, DWORD> {
                HMODULE module = LoadLibraryExW(L"dcomp.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
                if (module == nullptr)
                {
                    return {nullptr, GetLastError()};
                }
                auto* const fn =
                    reinterpret_cast<DCompositionCreateDeviceFn>(GetProcAddress(module, "DCompositionCreateDevice"));
                if (fn == nullptr)
                {
                    const DWORD error = GetLastError();
                    static_cast<void>(FreeLibrary(module));
                    return {nullptr, error};
                }
                return {fn, ERROR_SUCCESS};
            }();
            if (state.first == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::backendUnavailable,
                    static_cast<std::int64_t>(state.second),
                    "DirectComposition (dcomp.dll) is unavailable on this system.",
                });
            }
            return state.first;
        }

        [[nodiscard]] DXGI_SWAP_CHAIN_DESC1 makeNativeSwapchainDesc(SurfaceKind kind,
                                                                    const SwapchainDesc& desc) noexcept
        {
            return DXGI_SWAP_CHAIN_DESC1{
                .Width = desc.width,
                .Height = desc.height,
                .Format = nativeFormat(desc.format),
                .Stereo = FALSE,
                .SampleDesc = {.Count = 1, .Quality = 0},
                .BufferUsage = nativeUsage(desc.usage),
                .BufferCount = desc.imageCount,
                .Scaling = DXGI_SCALING_STRETCH,
                .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
                .AlphaMode = DXGI_ALPHA_MODE_IGNORE,
                .Flags = d3d12SwapchainFlags(allowsTearing(kind, desc.presentMode)),
            };
        }

        [[nodiscard]] Status applyColorSpaceAndFrameLatency(IDXGISwapChain4& swapchain, const SwapchainDesc& desc)
        {
            UINT colorSpaceSupport = 0;
            const auto nativeSwapchainColorSpace = nativeColorSpace(desc.colorSpace);
            const HRESULT colorSpaceQuery =
                swapchain.CheckColorSpaceSupport(nativeSwapchainColorSpace, &colorSpaceSupport);
            if (FAILED(colorSpaceQuery) || (colorSpaceSupport & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) == 0)
            {
                return std::unexpected(nativeError(ErrorCode::nativeFailure,
                                                   FAILED(colorSpaceQuery) ? colorSpaceQuery : E_NOTIMPL,
                                                   "IDXGISwapChain4::CheckColorSpaceSupport"));
            }
            const HRESULT colorSpaceResult = swapchain.SetColorSpace1(nativeSwapchainColorSpace);
            if (FAILED(colorSpaceResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, colorSpaceResult, "IDXGISwapChain4::SetColorSpace1"));
            }
            const HRESULT latencyResult = swapchain.SetMaximumFrameLatency(desc.imageCount);
            if (FAILED(latencyResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, latencyResult, "IDXGISwapChain4::SetMaximumFrameLatency"));
            }
            return {};
        }

        [[nodiscard]] Result<NativeSwapchainResources> createFlipModelSwapchain(IDXGIFactory7& factory,
                                                                                ID3D12CommandQueue* graphicsQueue,
                                                                                HWND window,
                                                                                const SwapchainDesc& desc)
        {
            const auto nativeDesc = makeNativeSwapchainDesc(SurfaceKind::win32, desc);
            ComPtr<IDXGISwapChain1> initialSwapchain;
            const HRESULT createResult =
                factory.CreateSwapChainForHwnd(graphicsQueue, window, &nativeDesc, nullptr, nullptr, &initialSwapchain);
            if (FAILED(createResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, createResult, "IDXGIFactory7::CreateSwapChainForHwnd"));
            }
            const HRESULT associationResult = factory.MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER);
            if (FAILED(associationResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, associationResult, "IDXGIFactory7::MakeWindowAssociation"));
            }

            ComPtr<IDXGISwapChain4> swapchain;
            const HRESULT queryResult = initialSwapchain.As(&swapchain);
            if (FAILED(queryResult))
            {
                return std::unexpected(nativeError(ErrorCode::nativeFailure, queryResult, "IDXGISwapChain4 query"));
            }
            auto configured = applyColorSpaceAndFrameLatency(*swapchain.Get(), desc);
            if (!configured)
            {
                return std::unexpected(std::move(configured.error()));
            }
            return NativeSwapchainResources{.swapchain = std::move(swapchain)};
        }

        [[nodiscard]] Result<NativeSwapchainResources> createCompositionSwapchain(IDXGIFactory7& factory,
                                                                                  ID3D12CommandQueue* graphicsQueue,
                                                                                  HWND window,
                                                                                  const SwapchainDesc& desc)
        {
            auto createDCompDevice = loadDCompositionCreateDevice();
            if (!createDCompDevice)
            {
                return std::unexpected(std::move(createDCompDevice.error()));
            }

            const auto nativeDesc = makeNativeSwapchainDesc(SurfaceKind::win32Composition, desc);
            ComPtr<IDXGISwapChain1> initialSwapchain;
            const HRESULT createResult =
                factory.CreateSwapChainForComposition(graphicsQueue, &nativeDesc, nullptr, &initialSwapchain);
            if (FAILED(createResult))
            {
                return std::unexpected(nativeError(
                    ErrorCode::nativeFailure, createResult, "IDXGIFactory7::CreateSwapChainForComposition"));
            }

            ComPtr<IDXGISwapChain4> swapchain;
            const HRESULT queryResult = initialSwapchain.As(&swapchain);
            if (FAILED(queryResult))
            {
                return std::unexpected(nativeError(ErrorCode::nativeFailure, queryResult, "IDXGISwapChain4 query"));
            }
            auto configured = applyColorSpaceAndFrameLatency(*swapchain.Get(), desc);
            if (!configured)
            {
                return std::unexpected(std::move(configured.error()));
            }

            ComPtr<IDCompositionDevice> compositionDevice;
            const HRESULT dcompDeviceResult = (*createDCompDevice)(nullptr, IID_PPV_ARGS(&compositionDevice));
            if (FAILED(dcompDeviceResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, dcompDeviceResult, "DCompositionCreateDevice"));
            }

            ComPtr<IDCompositionTarget> compositionTarget;
            const HRESULT targetResult = compositionDevice->CreateTargetForHwnd(window, TRUE, &compositionTarget);
            if (FAILED(targetResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, targetResult, "IDCompositionDevice::CreateTargetForHwnd"));
            }

            ComPtr<IDCompositionVisual> compositionVisual;
            const HRESULT visualResult = compositionDevice->CreateVisual(&compositionVisual);
            if (FAILED(visualResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, visualResult, "IDCompositionDevice::CreateVisual"));
            }

            const HRESULT contentResult = compositionVisual->SetContent(swapchain.Get());
            if (FAILED(contentResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, contentResult, "IDCompositionVisual::SetContent"));
            }

            const HRESULT rootResult = compositionTarget->SetRoot(compositionVisual.Get());
            if (FAILED(rootResult))
            {
                static_cast<void>(compositionVisual->SetContent(nullptr));
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, rootResult, "IDCompositionTarget::SetRoot"));
            }

            const HRESULT commitResult = compositionDevice->Commit();
            if (FAILED(commitResult))
            {
                static_cast<void>(compositionVisual->SetContent(nullptr));
                static_cast<void>(compositionTarget->SetRoot(nullptr));
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, commitResult, "IDCompositionDevice::Commit"));
            }

            return NativeSwapchainResources{
                .swapchain = std::move(swapchain),
                .compositionDevice = std::move(compositionDevice),
                .compositionTarget = std::move(compositionTarget),
                .compositionVisual = std::move(compositionVisual),
            };
        }

        [[nodiscard]] Result<NativeSwapchainResources> createNativeSwapchain(IDXGIFactory7& factory,
                                                                             ID3D12CommandQueue* graphicsQueue,
                                                                             const SurfaceRecord& surface,
                                                                             const SwapchainDesc& desc)
        {
            if (surface.kind == SurfaceKind::win32Composition)
            {
                return createCompositionSwapchain(factory, graphicsQueue, surface.window, desc);
            }
            return createFlipModelSwapchain(factory, graphicsQueue, surface.window, desc);
        }

        [[nodiscard]] std::shared_ptr<SwapchainRecord> makeSwapchainRecord(NativeSwapchainResources native,
                                                                           std::shared_ptr<SurfaceRecord> surface,
                                                                           const SwapchainDesc& desc)
        {
            const auto surfaceKind = surface->kind;
            auto record = std::make_shared<SwapchainRecord>();
            record->swapchain = std::move(native.swapchain);
            record->compositionDevice = std::move(native.compositionDevice);
            record->compositionTarget = std::move(native.compositionTarget);
            record->compositionVisual = std::move(native.compositionVisual);
            record->frameLatencyWaitableObject.reset(record->swapchain->GetFrameLatencyWaitableObject());
            record->surface = std::move(surface);
            record->desc = desc;
            record->label = desc.label;
            record->desc.label = record->label;
            record->allowTearing = allowsTearing(surfaceKind, desc.presentMode);
            if (std::uint32_t presentCount = 0; readPresentCount(*record->swapchain.Get(), presentCount))
            {
                record->presentation.start(record->generation, presentCount);
            }
            return record;
        }

        [[nodiscard]] Status validateResizableState(const SwapchainRecord& record)
        {
            if (record.destroyed)
            {
                return std::unexpected(invalidSwapchainError());
            }
            if (record.poisoned)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A swapchain with a failed native presentation or resize must be destroyed and recreated before "
                    "resize.",
                });
            }
            if (record.acquired)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "An acquired swapchain image must be presented or cancelled before resize.",
                });
            }
            if (record.generation == std::numeric_limits<std::uint64_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "The swapchain generation is exhausted and cannot wrap safely.",
                });
            }
            return {};
        }

        [[nodiscard]] Status rejectApplicationImageViews(const ViewManager& views, const SwapchainRecord& record)
        {
            for (const auto& buffer : record.buffers)
            {
                if (buffer.texture.isValid() && views.hasApplicationTextureViews(buffer.texture))
                {
                    return std::unexpected(referencedSwapchainImagesError(0));
                }
            }
            return {};
        }

        [[nodiscard]] HRESULT resizeNativeBuffers(SwapchainRecord& record, std::uint32_t width, std::uint32_t height)
        {
            const UINT flags = d3d12SwapchainFlags(record.allowTearing);
#ifdef RACHIS_TESTING
            const auto injectedResult =
                static_cast<HRESULT>(test::nextResizeBuffersResult().exchange(0, std::memory_order_relaxed));
            if (injectedResult != 0)
            {
                return injectedResult;
            }
#endif
            return record.swapchain->ResizeBuffers(
                record.desc.imageCount, width, height, nativeFormat(record.desc.format), flags);
        }

        void beginResizedGeneration(SwapchainRecord& record, std::uint32_t width, std::uint32_t height)
        {
            record.desc.width = width;
            record.desc.height = height;
            ++record.generation;
            if (std::uint32_t presentCount = 0; readPresentCount(*record.swapchain.Get(), presentCount))
            {
                record.presentation.beginGeneration(record.generation, presentCount);
            }
            else
            {
                record.presentation.disable();
            }
        }

        [[nodiscard]] Status validateAcquirableState(const SwapchainRecord& record)
        {
            if (record.destroyed)
            {
                return std::unexpected(invalidSwapchainError());
            }
            if (record.poisoned)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A swapchain with a failed native presentation or resize must be destroyed and recreated before "
                    "acquisition.",
                });
            }
            if (record.requiresRefresh)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A swapchain that reported surface-out-of-date must be refreshed with ResizeBuffers before "
                    "acquisition.",
                });
            }
            if (record.acquired || record.buffers.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A swapchain can expose only one acquired image at a time and must have live buffers.",
                });
            }
            return {};
        }

        [[nodiscard]] DWORD frameLatencyWaitMs(const rachis::detail::AcquireDeadline& deadline) noexcept
        {
            if (!deadline.timed())
            {
                return frameLatencyWaitTimeoutMs;
            }
            const auto left = deadline.remainingMs();
            return left == std::numeric_limits<std::uint64_t>::max()
                       ? INFINITE
                       : static_cast<DWORD>(std::min<std::uint64_t>(left, INFINITE - 1));
        }

        [[nodiscard]] Status waitForFrameLatencySlot(SwapchainRecord& record,
                                                     const rachis::detail::AcquireDeadline& deadline)
        {
            if (!record.latencySlot.needsWait(record.frameLatencyWaitableObject != nullptr))
            {
                return {};
            }
            for (;;)
            {
                const DWORD waitMs = frameLatencyWaitMs(deadline);
                const DWORD waitResult = WaitForSingleObjectEx(record.frameLatencyWaitableObject.get(), waitMs, FALSE);
                if (waitResult == WAIT_OBJECT_0)
                {
                    record.latencySlot.onWaitSucceeded();
                    return {};
                }
                if (waitResult != WAIT_TIMEOUT)
                {
                    return std::unexpected(Error{
                        ErrorCode::nativeFailure,
                        waitResult == WAIT_FAILED ? static_cast<std::int64_t>(GetLastError()) : 0,
                        "WaitForSingleObjectEx failed on the DXGI swapchain frame latency waitable object.",
                    });
                }
                if (!deadline.timed() || deadline.expired())
                {
                    return std::unexpected(
                        deadline.expiry("Timed out waiting for the DXGI swapchain frame latency waitable object.",
                                        static_cast<std::int64_t>(WAIT_TIMEOUT)));
                }
                if (waitMs == 0)
                {
                    static_cast<void>(SwitchToThread());
                }
            }
        }

        [[nodiscard]] Status waitForAndClearPreviousUse(QueueManager& queues,
                                                        SwapchainRecord& record,
                                                        SwapchainBuffer& buffer,
                                                        const rachis::detail::AcquireDeadline& deadline)
        {
            const bool hadPreviousUse = buffer.lastUse.isValid();
            if (hadPreviousUse && deadline.timed() && !deadline.unbounded())
            {
                auto polled = queues.wait(buffer.lastUse, deadline.remaining());
                if (!polled)
                {
                    if (polled.error().code == ErrorCode::deviceLost)
                    {
                        record.deviceLost = true;
                    }
                    return std::unexpected(std::move(polled.error()));
                }
                if (*polled != TimelinePointStatus::complete)
                {
                    return std::unexpected(
                        deadline.expiry("The GPU has not finished the previous frame on this back buffer yet."));
                }
            }
            auto waited = waitForBackBuffer(queues, record, buffer.lastUse);
            if (!waited)
            {
                return waited;
            }
            if (hadPreviousUse)
            {
                queues.collectGarbage();
                buffer.lastUse = {};
            }
            return {};
        }

        [[nodiscard]] Status validatePresentableState(const SwapchainRecord& record)
        {
            if (record.destroyed)
            {
                return std::unexpected(invalidSwapchainError());
            }
            if (record.poisoned)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A swapchain with a failed native presentation or resize must be destroyed and recreated.",
                });
            }
            if (!record.acquired || record.acquiredIndex >= record.buffers.size())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Presentation requires one currently acquired swapchain image.",
                });
            }
            return {};
        }

        [[nodiscard]] Status failPresentationAndPoison(SwapchainRecord& record,
                                                       SwapchainBuffer& buffer,
                                                       TimelinePoint renderComplete)
        {
            auto failedPresentation = buffer.lease->failPresentation(renderComplete);
            if (!failedPresentation)
            {
                return failedPresentation;
            }
            record.acquired = false;
            record.poisoned = true;
            buffer.lastUse = renderComplete;
            return {};
        }

        struct NativePresentAttempt final
        {
            HRESULT result = S_OK;
            bool calledNativePresent = false;
            bool hadPresentCountBefore = false;
            std::uint32_t presentCountBefore = 0;
        };

        [[nodiscard]] NativePresentAttempt presentToDxgi(SwapchainRecord& record)
        {
            std::uint32_t presentCountBefore = 0;
            const bool hadPresentCountBefore = readPresentCount(*record.swapchain.Get(), presentCountBefore);
            const UINT interval = record.desc.presentMode == PresentMode::fifo ? 1U : 0U;
            const UINT flags = record.allowTearing ? DXGI_PRESENT_ALLOW_TEARING : 0U;
#ifdef RACHIS_TESTING
            const auto injectedResult =
                static_cast<HRESULT>(test::nextPresentResult().exchange(0, std::memory_order_relaxed));
            const bool calledNativePresent = injectedResult == 0;
            const HRESULT result = calledNativePresent ? record.swapchain->Present(interval, flags) : injectedResult;
#else
            constexpr bool calledNativePresent = true;
            const HRESULT result = record.swapchain->Present(interval, flags);
#endif
            return NativePresentAttempt{
                .result = result,
                .calledNativePresent = calledNativePresent,
                .hadPresentCountBefore = hadPresentCountBefore,
                .presentCountBefore = presentCountBefore,
            };
        }

        [[nodiscard]] bool presentWasAcceptedByDxgi(const NativePresentAttempt& attempt,
                                                    bool hadPresentCountAfter,
                                                    std::uint32_t presentCountAfter) noexcept
        {
            if (!attempt.calledNativePresent)
            {
                return false;
            }
            if (attempt.hadPresentCountBefore && hadPresentCountAfter)
            {
                return presentCountAfter != attempt.presentCountBefore;
            }
            return attempt.result != DXGI_STATUS_OCCLUDED;
        }

        [[nodiscard]] std::uint64_t queryPerformanceFrequency() noexcept
        {
            LARGE_INTEGER frequency{};
            return QueryPerformanceFrequency(&frequency) != FALSE && frequency.QuadPart > 0
                       ? static_cast<std::uint64_t>(frequency.QuadPart)
                       : 0;
        }

        void observeFrameStatistics(SwapchainRecord& record) noexcept
        {
            auto& presentation = record.presentation;
            if (!presentation.supported() || record.swapchain == nullptr)
            {
                return;
            }
            DXGI_FRAME_STATISTICS statistics{};
#ifdef RACHIS_TESTING
            const auto reader = test::nativeFrameStatisticsReader().load(std::memory_order_acquire);
            const auto result = reader != nullptr ? static_cast<HRESULT>(reader(record.swapchain.Get(), &statistics))
                                                  : record.swapchain->GetFrameStatistics(&statistics);
#else
            const auto result = record.swapchain->GetFrameStatistics(&statistics);
#endif
            if (FAILED(result))
            {
                return;
            }
            presentation.observedFrame(
                statistics.PresentCount,
                statistics.PresentRefreshCount,
                statistics.SyncRefreshCount,
                statistics.SyncQPCTime.QuadPart > 0 ? static_cast<std::uint64_t>(statistics.SyncQPCTime.QuadPart) : 0);
        }

        void updatePresentationProgressAndLatencySlot(SwapchainRecord& record, const NativePresentAttempt& attempt)
        {
            std::uint32_t presentCountAfter = 0;
            const bool hadPresentCountAfter = readPresentCount(*record.swapchain.Get(), presentCountAfter);
            if (hadPresentCountAfter)
            {
                record.presentation.presented(presentCountAfter);
                observeFrameStatistics(record);
            }
            else
            {
                record.presentation.disable();
            }
            record.latencySlot.onPresent(presentWasAcceptedByDxgi(attempt, hadPresentCountAfter, presentCountAfter));
        }

        [[nodiscard]] constexpr PresentStatus presentStatusFor(HRESULT result) noexcept
        {
            if (result == DXGI_STATUS_MODE_CHANGED)
            {
                return PresentStatus::surfaceOutOfDate;
            }
            if (result == DXGI_STATUS_OCCLUDED)
            {
                return PresentStatus::occluded;
            }
            return PresentStatus::presented;
        }
    } // namespace

    struct SwapchainManager::Impl final
    {
        ComPtr<IDXGIFactory7> factory;
        bool tearingSupported = false;
        bool fullyTypedFormatCasting = false;
        rachis::detail::TimedMutex mutex;
        detail::HandleTable<std::shared_ptr<SurfaceRecord>, SurfaceHandle> surfaces;
        detail::HandleTable<std::shared_ptr<SwapchainRecord>, SwapchainHandle> swapchains;
    };

    SwapchainManager::SwapchainManager() : impl_(std::make_unique<Impl>())
    {
    }

    SwapchainManager::~SwapchainManager()
    {
        if (impl_ == nullptr)
        {
            return;
        }
        impl_->swapchains.forEach([](SwapchainHandle, const std::shared_ptr<SwapchainRecord>& record) {
            if (record != nullptr)
            {
                releaseNativeSwapchain(*record);
            }
        });
    }

    Status SwapchainManager::initialize(ID3D12Device& device)
    {
        ComPtr<IDXGIFactory7> factory;
        const HRESULT result = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
        if (FAILED(result))
        {
            return std::unexpected(nativeError(ErrorCode::backendUnavailable, result, "CreateDXGIFactory2"));
        }
        impl_->tearingSupported = queryTearingSupport(*factory.Get());
        impl_->fullyTypedFormatCasting = queryFullyTypedFormatCasting(device);
        impl_->factory = std::move(factory);
        return {};
    }

    Result<SurfaceHandle> SwapchainManager::createSurface(const SurfaceDesc& desc)
    {
        auto* const window = static_cast<HWND>(desc.windowHandle);
        if (!isWin32SurfaceKind(desc.kind) || window == nullptr || IsWindow(window) == FALSE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A Win32 surface requires a live HWND.",
            });
        }
        auto record = std::make_shared<SurfaceRecord>();
        record->kind = desc.kind;
        record->window = window;
        record->label = desc.label;

        const std::scoped_lock lock(impl_->mutex);
        return impl_->surfaces.emplace(std::move(record));
    }

    Result<SurfaceCapabilities> SwapchainManager::querySurfaceCapabilities(SurfaceHandle surface) const
    {
        const auto record = findSurfaceRecord(impl_->mutex, impl_->surfaces, surface);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The surface handle is invalid or stale.",
            });
        }
        return capabilitiesForSurface(*record, impl_->tearingSupported);
    }

    Status SwapchainManager::destroySurface(SurfaceHandle surface)
    {
        const std::scoped_lock lock(impl_->mutex);
        const auto* found = impl_->surfaces.find(surface);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The surface handle is invalid or stale.",
            });
        }
        if ((*found)->swapchainClaimed)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A surface with a live or pending swapchain cannot be destroyed.",
            });
        }
        static_cast<void>(impl_->surfaces.erase(surface));
        return {};
    }

    Result<SwapchainHandle> SwapchainManager::createSwapchain(QueueManager& queues,
                                                              ResourceManager& resources,
                                                              ViewManager& views,
                                                              const SwapchainDesc& desc)
    {
        if (!validPresentMode(desc.presentMode) || nativeColorSpace(desc.colorSpace) == DXGI_COLOR_SPACE_CUSTOM)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A swapchain present mode and color space must be valid portable enumerators.",
            });
        }

        auto surface = findSurfaceRecord(impl_->mutex, impl_->surfaces, desc.surface);
        if (surface == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The swapchain surface handle is invalid or stale.",
            });
        }
        auto capabilities = capabilitiesForSurface(*surface, impl_->tearingSupported);
        if (!capabilities)
        {
            return std::unexpected(std::move(capabilities.error()));
        }
        auto supported = validateSwapchainDescForSurface(*capabilities, desc, impl_->fullyTypedFormatCasting);
        if (!supported)
        {
            return std::unexpected(std::move(supported.error()));
        }
        auto graphicsQueue = queues.nativeQueue(QueueType::graphics);
        if (!graphicsQueue)
        {
            return std::unexpected(std::move(graphicsQueue.error()));
        }

        auto claimed =
            claimSurfaceForSwapchain(impl_->mutex, impl_->surfaces, impl_->swapchains, desc.surface, surface);
        if (!claimed)
        {
            return std::unexpected(std::move(claimed.error()));
        }
        PendingSurfaceClaim pendingClaim(surface, impl_->mutex);

        auto nativeSwapchain = createNativeSwapchain(*impl_->factory.Get(), *graphicsQueue, *surface, desc);
        if (!nativeSwapchain)
        {
            return std::unexpected(std::move(nativeSwapchain.error()));
        }
        auto record = makeSwapchainRecord(std::move(*nativeSwapchain), std::move(surface), desc);
        auto buffers = createBackBuffers(resources, views, *record);
        if (!buffers)
        {
            releaseNativeSwapchain(*record);
            return std::unexpected(std::move(buffers.error()));
        }

        Result<SwapchainHandle> handle{std::unexpect, Error{}};
        {
            const std::scoped_lock lock(impl_->mutex);
            handle = impl_->swapchains.emplace(record);
            if (handle)
            {
                pendingClaim.publish();
                return *handle;
            }
        }
        releaseBackBuffers(resources, views, *record);
        releaseNativeSwapchain(*record);
        return std::unexpected(std::move(handle.error()));
    }

    Status SwapchainManager::destroySwapchain(QueueManager& queues,
                                              ResourceManager& resources,
                                              ViewManager& views,
                                              SwapchainHandle swapchain,
                                              bool deviceLost,
                                              std::optional<Error>& observedDeviceLoss)
    {
        const auto record = findSwapchainRecord(impl_->mutex, impl_->swapchains, swapchain);
        if (record == nullptr)
        {
            return std::unexpected(invalidSwapchainError());
        }
        std::unique_lock recordLock(record->recordMutex);
        if (record->destroyed)
        {
            return std::unexpected(invalidSwapchainError());
        }
        if (!deviceLost && record->acquired)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "An acquired swapchain image must be presented or cancelled before destruction.",
            });
        }
        if (!deviceLost && !record->deviceLost)
        {
            auto waited = waitForBackBuffers(queues, *record);
            if (!waited)
            {
                if (waited.error().code != ErrorCode::deviceLost)
                {
                    return waited;
                }

                observedDeviceLoss = std::move(waited.error());
            }
        }
        releaseBackBuffers(resources, views, *record);
        releaseNativeSwapchain(*record);
        record->destroyed = true;
        recordLock.unlock();

        const std::scoped_lock lock(impl_->mutex);
        if (record->surface != nullptr)
        {
            record->surface->swapchainClaimed = false;
        }
        static_cast<void>(impl_->swapchains.erase(swapchain));
        return {};
    }

    Status SwapchainManager::resizeSwapchain(QueueManager& queues,
                                             ResourceManager& resources,
                                             ViewManager& views,
                                             SwapchainHandle swapchain,
                                             std::uint32_t width,
                                             std::uint32_t height)
    {
        if (width == 0 || height == 0 || width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Swapchain resize dimensions must be non-zero and within the portable 2D limit.",
            });
        }

        const auto recordOwner = findSwapchainRecord(impl_->mutex, impl_->swapchains, swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchainError());
        }
        auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        auto resizable = validateResizableState(record);
        if (!resizable)
        {
            return resizable;
        }
        auto unviewed = rejectApplicationImageViews(views, record);
        if (!unviewed)
        {
            return unviewed;
        }
        auto waited = waitForBackBuffers(queues, record);
        if (!waited)
        {
            return waited;
        }
        releaseBackBuffers(resources, views, record);
        const HRESULT result = resizeNativeBuffers(record, width, height);
        if (isDeviceRemoval(result))
        {
            record.poisoned = true;
            record.deviceLost = true;
            return std::unexpected(nativeError(ErrorCode::deviceLost, result, "IDXGISwapChain4::ResizeBuffers"));
        }
        if (FAILED(result))
        {
            auto recovered = createBackBuffers(resources, views, record);
            if (!recovered)
            {
                record.poisoned = true;
                return std::unexpected(std::move(recovered.error()));
            }
            if (result == DXGI_ERROR_INVALID_CALL)
            {
                return std::unexpected(referencedSwapchainImagesError(static_cast<std::int64_t>(result)));
            }

            return std::unexpected(nativeError(ErrorCode::nativeFailure, result, "IDXGISwapChain4::ResizeBuffers"));
        }
        beginResizedGeneration(record, width, height);
        auto created = createBackBuffers(resources, views, record);
        if (!created)
        {
            record.poisoned = true;
            return std::unexpected(std::move(created.error()));
        }
        record.requiresRefresh = false;
        record.latencySlot.onResize();
        return {};
    }

    Result<AcquiredSwapchainImage> SwapchainManager::acquireNextImage(QueueManager& queues,
                                                                      SwapchainHandle swapchain,
                                                                      std::optional<std::chrono::nanoseconds> timeout)
    {
        const rachis::detail::AcquireDeadline deadline(timeout, std::chrono::milliseconds(frameLatencyWaitTimeoutMs));
        const auto found = findSwapchainRecord(impl_->mutex, impl_->swapchains, swapchain, deadline);
        if (!found)
        {
            return std::unexpected(deadline.lockTimeout());
        }
        const auto& recordOwner = *found;
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchainError());
        }
        auto& record = *recordOwner;
        const auto recordLock = deadline.lock(record.recordMutex);
        if (!recordLock.owns_lock())
        {
            return std::unexpected(deadline.lockTimeout());
        }
        auto acquirable = validateAcquirableState(record);
        if (!acquirable)
        {
            return std::unexpected(std::move(acquirable.error()));
        }
        const auto expired = [&](Error error) {
            if (error.code == ErrorCode::timeout && record.surface != nullptr &&
                (record.surface->window == nullptr || IsWindowVisible(record.surface->window) == FALSE ||
                 IsIconic(record.surface->window) != FALSE))
            {
                error.code = ErrorCode::notReady;
                error.message = "The swapchain window is hidden or minimized; no image was ready in time.";
            }
            return std::unexpected(std::move(error));
        };
        auto latencySlotReady = waitForFrameLatencySlot(record, deadline);
        if (!latencySlotReady)
        {
            return expired(std::move(latencySlotReady.error()));
        }
        const auto imageIndex = record.swapchain->GetCurrentBackBufferIndex();
        if (imageIndex >= record.buffers.size())
        {
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                0,
                "DXGI returned a swapchain image index outside the live buffer set.",
            });
        }
        auto& buffer = record.buffers[imageIndex];
        auto previousUseComplete = waitForAndClearPreviousUse(queues, record, buffer, deadline);
        if (!previousUseComplete)
        {
            return expired(std::move(previousUseComplete.error()));
        }
        auto acquiredLease = buffer.lease->acquire();
        if (!acquiredLease)
        {
            return std::unexpected(std::move(acquiredLease.error()));
        }
        record.acquired = true;
        record.acquiredIndex = imageIndex;
        return AcquiredSwapchainImage{
            .texture = buffer.texture,
            .renderTargetView = buffer.renderTargetView,
            .imageIndex = imageIndex,
            .width = record.desc.width,
            .height = record.desc.height,
            .format = record.desc.format,
            .colorSpace = record.desc.colorSpace,
            .generation = record.generation,
        };
    }

    bool SwapchainManager::hasAcquiredImage(SwapchainHandle swapchain) const noexcept
    {
        if (impl_ == nullptr)
        {
            return false;
        }
        const auto record = findSwapchainRecord(impl_->mutex, impl_->swapchains, swapchain);
        if (record == nullptr)
        {
            return false;
        }
        const std::scoped_lock recordLock(record->recordMutex);
        return !record->destroyed && record->acquired;
    }

    Status SwapchainManager::cancelSwapchainImage(SwapchainHandle swapchain)
    {
        const auto record = findSwapchainRecord(impl_->mutex, impl_->swapchains, swapchain);
        if (record == nullptr)
        {
            return std::unexpected(invalidSwapchainError());
        }
        const std::scoped_lock recordLock(record->recordMutex);
        if (record->destroyed)
        {
            return std::unexpected(invalidSwapchainError());
        }
        if (!record->acquired)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, "No swapchain image is acquired."});
        }
        if (record->poisoned)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A swapchain with a failed native presentation cannot cancel an acquisition.",
            });
        }
        auto cancelled = record->buffers[record->acquiredIndex].lease->cancel();
        if (!cancelled)
        {
            return cancelled;
        }
        record->acquired = false;
        record->latencySlot.onCancel();
        return {};
    }

    Result<PresentStatus> SwapchainManager::present(QueueManager& queues,
                                                    SwapchainHandle swapchain,
                                                    TimelinePoint renderComplete)
    {
        const auto recordOwner = findSwapchainRecord(impl_->mutex, impl_->swapchains, swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchainError());
        }
        auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        auto presentable = validatePresentableState(record);
        if (!presentable)
        {
            return std::unexpected(std::move(presentable.error()));
        }
        auto& buffer = record.buffers[record.acquiredIndex];
        auto validPoint = queues.validateSubmittedPoint(renderComplete, QueueType::graphics);
        if (!validPoint)
        {
            return std::unexpected(std::move(validPoint.error()));
        }
        auto exactPoint = buffer.lease->validatePresentationPoint(renderComplete);
        if (!exactPoint)
        {
            return std::unexpected(std::move(exactPoint.error()));
        }
        auto signalReservation = queues.reserveSignal(QueueType::graphics);
        if (!signalReservation)
        {
            auto failedPresentation = failPresentationAndPoison(record, buffer, renderComplete);
            if (!failedPresentation)
            {
                return std::unexpected(std::move(failedPresentation.error()));
            }
            return std::unexpected(std::move(signalReservation.error()));
        }

        auto begunPresentation = buffer.lease->beginPresentation(renderComplete);
        if (!begunPresentation)
        {
            return std::unexpected(std::move(begunPresentation.error()));
        }
        record.acquired = false;

        const auto attempt = presentToDxgi(record);
        if (isDeviceRemoval(attempt.result))
        {
            record.poisoned = true;
            record.deviceLost = true;
            buffer.lastUse = {};
            return std::unexpected(nativeError(ErrorCode::deviceLost, attempt.result, "IDXGISwapChain4::Present"));
        }
        if (FAILED(attempt.result))
        {
            record.poisoned = true;
            buffer.lastUse = renderComplete;
            return std::unexpected(nativeError(ErrorCode::nativeFailure, attempt.result, "IDXGISwapChain4::Present"));
        }
        updatePresentationProgressAndLatencySlot(record, attempt);

        const auto status = presentStatusFor(attempt.result);
        auto presentationComplete = signalReservation->signal();
        if (!presentationComplete)
        {
            record.poisoned = true;
            record.deviceLost = true;
            buffer.lastUse = {};
            return std::unexpected(std::move(presentationComplete.error()));
        }
        buffer.lastUse = *presentationComplete;
        if (status == PresentStatus::surfaceOutOfDate)
        {
            record.requiresRefresh = true;
        }
        return status;
    }

    Result<SwapchainPresentationInfo> SwapchainManager::queryPresentation(SwapchainHandle swapchain) const
    {
        const auto recordOwner = findSwapchainRecord(impl_->mutex, impl_->swapchains, swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchainError());
        }
        const auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        if (record.destroyed)
        {
            return std::unexpected(invalidSwapchainError());
        }
        return SwapchainPresentationInfo{
            .format = record.desc.format,
            .colorSpace = record.desc.colorSpace,
            .presentMode = record.desc.presentMode,
            .usage = record.desc.usage,
            .imageCount = static_cast<std::uint32_t>(record.buffers.size()),
            .framesInFlightBound = static_cast<std::uint32_t>(record.buffers.size()),
            .width = record.desc.width,
            .height = record.desc.height,
            .hdr = record.desc.colorSpace == SurfaceColorSpace::hdr10St2084 ||
                   record.desc.colorSpace == SurfaceColorSpace::extendedSrgbLinear,
        };
    }

    Result<SwapchainFrameTiming> SwapchainManager::queryFrameTiming(SwapchainHandle swapchain) const
    {
        const auto recordOwner = findSwapchainRecord(impl_->mutex, impl_->swapchains, swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchainError());
        }
        auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        if (record.destroyed)
        {
            return std::unexpected(invalidSwapchainError());
        }
        auto& presentation = record.presentation;
        observeFrameStatistics(record);
        return SwapchainFrameTiming{
            .presentationSupported = presentation.supported(),
            .presentationGeneration = presentation.generation(),
            .submittedPresentationSequence = presentation.submittedSequence(),
            .displayedPresentationSequence = presentation.displayedSequence(),
        };
    }

    Result<SwapchainPresentTimings> SwapchainManager::queryPresentTimings(SwapchainHandle swapchain,
                                                                          std::uint64_t firstSequence,
                                                                          std::span<PresentTiming> records) const
    {
        const auto recordOwner = findSwapchainRecord(impl_->mutex, impl_->swapchains, swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchainError());
        }
        auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        if (record.destroyed)
        {
            return std::unexpected(invalidSwapchainError());
        }
        const auto frequency = queryPerformanceFrequency();
        if (!record.presentation.supported() || record.swapchain == nullptr || frequency == 0)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "This swapchain reports no DXGI frame statistics, so it has no per-present timing.",
            });
        }
        observeFrameStatistics(record);
        const auto snapshot = record.presentation.readTimings(firstSequence, records);
        const auto refreshTicks = record.presentation.refreshTicks();
        const auto intervalNs = refreshTicks == 0 ? 0 : (refreshTicks * 1'000'000'000ULL + frequency / 2) / frequency;
        return SwapchainPresentTimings{
            .cpuClock = CpuClock::queryPerformanceCounter,
            .cpuFrequency = frequency,
            .generation = record.presentation.generation(),
            .submittedSequence = snapshot.submittedSequence,
            .settledSequence = snapshot.settledSequence,
            .firstAvailableSequence = snapshot.firstAvailableSequence,
            .refreshIntervalNs = intervalNs,
            .maxRefreshIntervalNs = intervalNs,
            .recordCount = snapshot.recordCount,
        };
    }

} // namespace rachis::d3d12
