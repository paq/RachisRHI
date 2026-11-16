#include "Rachis/rachis.h"

#include "C/CAbiDeviceMutex.h"
#include "C/CAbiSlotTable.h"
#include "C/CAbiText.h"
#include "C/CAbiTokenSource.h"
#include "Common/MonotonicIdentitySource.h"
#include "Rachis/TestSupport/PresentInjection.h"

#if RACHIS_CABI_FAULT_BACKEND_METAL
#include "Metal/MetalTestHooks.h"
#include "MetalSurfaceTestHost.h"
#elif RACHIS_CABI_FAULT_BACKEND_D3D12
#include "D3D12/D3D12TestHooks.h"
#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#elif RACHIS_CABI_FAULT_BACKEND_VULKAN
#include "Internal/VulkanDevice.h"
#include "Vulkan/VulkanTestHooks.h"
#if defined(_WIN32)
#include <Windows.h>
#endif
#endif

#if RACHIS_CABI_FAULT_BACKEND_METAL || defined(_WIN32)
#define RACHIS_CABI_FAULT_HAS_SURFACE_HOST 1
#else
#define RACHIS_CABI_FAULT_HAS_SURFACE_HOST 0
#endif

#if defined(_WIN32)
#define RACHIS_CABI_PRESENT_DESKTOP_REQUIRES "[requires:InteractiveDesktop]"
#else
#define RACHIS_CABI_PRESENT_DESKTOP_REQUIRES ""
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace
{
    using Device = std::unique_ptr<rachis_device, decltype(&rachis_device_destroy)>;

    class ScopedTokenSourceOverride final
    {
    public:
        explicit ScopedTokenSourceOverride(rachis::detail::MonotonicIdentitySource* source) noexcept
        {
            rachis::cabi::detail::setTokenSourceOverrideForTesting(source);
        }

        ~ScopedTokenSourceOverride()
        {
            rachis::cabi::detail::setTokenSourceOverrideForTesting(nullptr);
        }

        ScopedTokenSourceOverride(const ScopedTokenSourceOverride&) = delete;
        ScopedTokenSourceOverride& operator=(const ScopedTokenSourceOverride&) = delete;
        ScopedTokenSourceOverride(ScopedTokenSourceOverride&&) = delete;
        ScopedTokenSourceOverride& operator=(ScopedTokenSourceOverride&&) = delete;
    };

    class ScopedCommandRecordingHook final
    {
    public:
        ScopedCommandRecordingHook(rachis::cabi::detail::CommandRecordingHook hook, void* userData) noexcept
        {
            rachis::cabi::detail::setCommandRecordingHookForTesting(hook, userData);
        }

        ~ScopedCommandRecordingHook()
        {
            rachis::cabi::detail::setCommandRecordingHookForTesting(nullptr, nullptr);
        }

        ScopedCommandRecordingHook(const ScopedCommandRecordingHook&) = delete;
        ScopedCommandRecordingHook& operator=(const ScopedCommandRecordingHook&) = delete;
        ScopedCommandRecordingHook(ScopedCommandRecordingHook&&) = delete;
        ScopedCommandRecordingHook& operator=(ScopedCommandRecordingHook&&) = delete;
    };

    class ScopedSwapchainPresentHook final
    {
    public:
        ScopedSwapchainPresentHook(rachis::cabi::detail::SwapchainPresentHook hook, void* userData) noexcept
        {
            rachis::cabi::detail::setSwapchainPresentHookForTesting(hook, userData);
        }

        ~ScopedSwapchainPresentHook()
        {
            rachis::cabi::detail::setSwapchainPresentHookForTesting(nullptr, nullptr);
        }

        ScopedSwapchainPresentHook(const ScopedSwapchainPresentHook&) = delete;
        ScopedSwapchainPresentHook& operator=(const ScopedSwapchainPresentHook&) = delete;
        ScopedSwapchainPresentHook(ScopedSwapchainPresentHook&&) = delete;
        ScopedSwapchainPresentHook& operator=(ScopedSwapchainPresentHook&&) = delete;
    };

    constexpr std::string_view injectedNativeFailure = "injected native failure";

    struct NativeCallFault final
    {
        rachis::cabi::detail::NativeCall call = rachis::cabi::detail::NativeCall::createObject;
        bool raise = false;
        std::uint32_t hits = 0;
        std::string message = std::string{injectedNativeFailure};
    };

    rachis::Status injectNativeCallFault(rachis::cabi::detail::NativeCall call, void* userData)
    {
        auto& fault = *static_cast<NativeCallFault*>(userData);
        if (call != fault.call)
        {
            return {};
        }
        ++fault.hits;
        if (fault.raise)
        {
            throw std::runtime_error(fault.message);
        }
        return std::unexpected(rachis::Error{
            .code = rachis::ErrorCode::nativeFailure,
            .nativeCode = 0,
            .message = fault.message,
        });
    }

    class ScopedNativeCallFault final
    {
    public:
        explicit ScopedNativeCallFault(NativeCallFault& fault) noexcept
        {
            rachis::cabi::detail::setNativeCallHookForTesting(&injectNativeCallFault, &fault);
        }

        ~ScopedNativeCallFault()
        {
            rachis::cabi::detail::setNativeCallHookForTesting(nullptr, nullptr);
        }

        ScopedNativeCallFault(const ScopedNativeCallFault&) = delete;
        ScopedNativeCallFault& operator=(const ScopedNativeCallFault&) = delete;
        ScopedNativeCallFault(ScopedNativeCallFault&&) = delete;
        ScopedNativeCallFault& operator=(ScopedNativeCallFault&&) = delete;
    };

#if RACHIS_CABI_FAULT_BACKEND_METAL || RACHIS_CABI_FAULT_BACKEND_D3D12
    class ScopedPresentResult final
    {
    public:
        explicit ScopedPresentResult(std::int32_t value) noexcept
        {
            rachis::test::setNextPresentResult(value);
        }

        ~ScopedPresentResult()
        {
            rachis::test::setNextPresentResult(0);
        }

        ScopedPresentResult(const ScopedPresentResult&) = delete;
        ScopedPresentResult& operator=(const ScopedPresentResult&) = delete;
        ScopedPresentResult(ScopedPresentResult&&) = delete;
        ScopedPresentResult& operator=(ScopedPresentResult&&) = delete;
    };
#endif

#if RACHIS_CABI_FAULT_BACKEND_METAL
    class ScopedMetalSubmitFailure final
    {
    public:
        ScopedMetalSubmitFailure() noexcept
        {
            rachis::metal::test::nextSubmitCloseFailure().store(true, std::memory_order_relaxed);
        }

        ~ScopedMetalSubmitFailure()
        {
            rachis::metal::test::nextSubmitCloseFailure().store(false, std::memory_order_relaxed);
        }

        ScopedMetalSubmitFailure(const ScopedMetalSubmitFailure&) = delete;
        ScopedMetalSubmitFailure& operator=(const ScopedMetalSubmitFailure&) = delete;
        ScopedMetalSubmitFailure(ScopedMetalSubmitFailure&&) = delete;
        ScopedMetalSubmitFailure& operator=(ScopedMetalSubmitFailure&&) = delete;
    };

    class ScopedMetalPresentFailure final
    {
    public:
        explicit ScopedMetalPresentFailure(rachis::metal::test::InjectedPresentFailure mode) noexcept
        {
            rachis::metal::test::nextInjectedPresentFailure().store(mode, std::memory_order_relaxed);
        }

        ~ScopedMetalPresentFailure()
        {
            rachis::metal::test::nextInjectedPresentFailure().store(rachis::metal::test::InjectedPresentFailure::none,
                                                                    std::memory_order_relaxed);
        }

        ScopedMetalPresentFailure(const ScopedMetalPresentFailure&) = delete;
        ScopedMetalPresentFailure& operator=(const ScopedMetalPresentFailure&) = delete;
        ScopedMetalPresentFailure(ScopedMetalPresentFailure&&) = delete;
        ScopedMetalPresentFailure& operator=(ScopedMetalPresentFailure&&) = delete;
    };

    struct SurfaceHost final
    {
        static constexpr std::uint32_t width = 64;
        static constexpr std::uint32_t height = 48;

        void* layer = nullptr;
        void* view = nullptr;

        SurfaceHost() : layer(rxMetalTestCreateLayer(width, height)), view(rxMetalTestCreateView(layer))
        {
        }

        ~SurfaceHost()
        {
            rxMetalTestRelease(view);
            rxMetalTestRelease(layer);
        }

        SurfaceHost(const SurfaceHost&) = delete;
        SurfaceHost& operator=(const SurfaceHost&) = delete;
        SurfaceHost(SurfaceHost&&) = delete;
        SurfaceHost& operator=(SurfaceHost&&) = delete;

        [[nodiscard]] bool valid() const noexcept
        {
            return layer != nullptr && view != nullptr;
        }

        [[nodiscard]] rachis_surface_desc surfaceDesc() const noexcept
        {
            rachis_surface_desc desc{};
            desc.struct_size = sizeof(desc);
            desc.abi_version = RACHIS_ABI_VERSION;
            desc.kind = RACHIS_SURFACE_CA_METAL_LAYER;
            desc.window_handle = layer;
            desc.display_handle = view;
            desc.label = "C ABI fault injection surface";
            return desc;
        }
    };
#elif defined(_WIN32)
    struct SurfaceHost final
    {
        static constexpr std::uint32_t width = 64;
        static constexpr std::uint32_t height = 48;

        HWND window = nullptr;

        SurfaceHost()
            : window(CreateWindowExW(0,
                                     L"STATIC",
                                     L"Rachis C ABI fault injection",
                                     WS_OVERLAPPEDWINDOW,
                                     CW_USEDEFAULT,
                                     CW_USEDEFAULT,
                                     static_cast<int>(width),
                                     static_cast<int>(height),
                                     nullptr,
                                     nullptr,
                                     GetModuleHandleW(nullptr),
                                     nullptr))
        {
            if (window != nullptr)
            {
                static_cast<void>(ShowWindow(window, SW_SHOWNOACTIVATE));
            }
        }

        ~SurfaceHost()
        {
            if (window != nullptr)
            {
                static_cast<void>(DestroyWindow(window));
            }
        }

        SurfaceHost(const SurfaceHost&) = delete;
        SurfaceHost& operator=(const SurfaceHost&) = delete;
        SurfaceHost(SurfaceHost&&) = delete;
        SurfaceHost& operator=(SurfaceHost&&) = delete;

        [[nodiscard]] bool valid() const noexcept
        {
            return window != nullptr;
        }

        [[nodiscard]] rachis_surface_desc surfaceDesc() const noexcept
        {
            rachis_surface_desc desc{};
            desc.struct_size = sizeof(desc);
            desc.abi_version = RACHIS_ABI_VERSION;
            desc.kind = RACHIS_SURFACE_WIN32;
            desc.window_handle = window;
            desc.label = "C ABI fault injection surface";
            return desc;
        }
    };
#endif

#if RACHIS_CABI_FAULT_BACKEND_VULKAN
    class ScopedVulkanSubmitFailure final
    {
    public:
        explicit ScopedVulkanSubmitFailure(std::int32_t vkResult) noexcept
        {
            rachis::vulkan::test::nextEndCommandBufferResult().store(vkResult, std::memory_order_relaxed);
        }

        ~ScopedVulkanSubmitFailure()
        {
            rachis::vulkan::test::nextEndCommandBufferResult().store(0, std::memory_order_relaxed);
        }

        ScopedVulkanSubmitFailure(const ScopedVulkanSubmitFailure&) = delete;
        ScopedVulkanSubmitFailure& operator=(const ScopedVulkanSubmitFailure&) = delete;
        ScopedVulkanSubmitFailure(ScopedVulkanSubmitFailure&&) = delete;
        ScopedVulkanSubmitFailure& operator=(ScopedVulkanSubmitFailure&&) = delete;
    };
#endif

    [[nodiscard]] rachis_error makeError() noexcept
    {
        rachis_error error{};
        error.struct_size = sizeof(error);
        return error;
    }

    void requireOk(rachis_status status, const rachis_error& error)
    {
        INFO(error.message);
        REQUIRE(status == RACHIS_STATUS_OK);
    }

    struct DiagnosticEntry final
    {
        std::uint32_t severity = 0;
        std::uint32_t nativeId = 0;
        std::string message;
    };

    struct DiagnosticRecord final
    {
        std::mutex mutex;
        std::vector<DiagnosticEntry> entries;
    };

    void recordDiagnostic(std::uint32_t severity, std::uint32_t nativeId, const char* message, void* userData)
    {
        auto& record = *static_cast<DiagnosticRecord*>(userData);
        const std::scoped_lock lock(record.mutex);
        record.entries.push_back(DiagnosticEntry{
            .severity = severity,
            .nativeId = nativeId,
            .message = message == nullptr ? std::string{} : std::string{message},
        });
    }

    [[nodiscard]] Device createDevice(rachis_error& error, std::uint32_t validation = RACHIS_VALIDATION_STANDARD)
    {
        rachis_device_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.backend = RACHIS_SMOKE_BACKEND;
        desc.adapter_preference = RACHIS_SMOKE_ADAPTER;
        desc.validation_level = validation;
        desc.application_name = "Rachis C ABI fault injection tests";
        rachis_device* device = nullptr;
        requireOk(rachis_device_create(&desc, &device, &error), error);
        REQUIRE(device != nullptr);
        return Device{device, &rachis_device_destroy};
    }

    [[nodiscard]] rachis_handle createBuffer(
        rachis_device* device, std::uint64_t size, std::uint32_t usage, std::uint32_t memoryType, rachis_error& error)
    {
        rachis_buffer_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.size = size;
        desc.usage = usage;
        desc.memory_type = memoryType;
        rachis_handle buffer = 0;
        requireOk(rachis_buffer_create(device, &desc, &buffer, &error), error);
        REQUIRE(buffer != 0);
        return buffer;
    }

    [[nodiscard]] rachis_handle createPool(rachis_device* device, std::uint32_t queueType, rachis_error& error)
    {
        rachis_command_pool_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.queue_type = queueType;
        desc.max_command_buffers = 8;
        rachis_handle pool = 0;
        requireOk(rachis_command_pool_create(device, &desc, &pool, &error), error);
        REQUIRE(pool != 0);
        return pool;
    }

    [[nodiscard]] rachis_handle createRenderTarget(rachis_device* device, rachis_error& error)
    {
        rachis_texture_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.dimension = RACHIS_TEXTURE_2D;
        desc.format = RACHIS_FORMAT_RGBA8_UNORM;
        desc.usage = RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_COPY_SOURCE;
        desc.width = 4;
        desc.height = 4;
        desc.depth_or_array_layers = 1;
        desc.mip_levels = 1;
        desc.sample_count = 1;
        desc.label = "C ABI fault injection texture";
        rachis_handle texture = 0;
        requireOk(rachis_texture_create(device, &desc, &texture, &error), error);
        REQUIRE(texture != 0);
        return texture;
    }

    [[nodiscard]] rachis_handle createRenderTargetView(rachis_device* device,
                                                       rachis_handle texture,
                                                       rachis_error& error)
    {
        rachis_texture_view_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.texture = texture;
        desc.usage = RACHIS_VIEW_USAGE_RENDER_TARGET;
        desc.dimension = RACHIS_VIEW_2D;
        desc.format = RACHIS_FORMAT_UNKNOWN;
        desc.aspect = RACHIS_ASPECT_COLOR;
        desc.mip_level_count = 1;
        desc.array_layer_count = 1;
        rachis_handle view = 0;
        requireOk(rachis_texture_view_create(device, &desc, &view, &error), error);
        REQUIRE(view != 0);
        return view;
    }

#if RACHIS_CABI_FAULT_HAS_SURFACE_HOST
    [[nodiscard]] rachis_handle createSwapchain(rachis_device* device, rachis_handle surface, rachis_error& error)
    {
        rachis_surface_capabilities capabilities{};
        capabilities.struct_size = sizeof(capabilities);
        requireOk(rachis_surface_query_capabilities(device, surface, &capabilities, &error), error);
        rachis_swapchain_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.surface = surface;
        desc.width = capabilities.preferred_width == 0 ? SurfaceHost::width : capabilities.preferred_width;
        desc.height = capabilities.preferred_height == 0 ? SurfaceHost::height : capabilities.preferred_height;
        desc.image_count =
            std::max(capabilities.min_image_count,
                     std::min(capabilities.max_image_count == 0 ? 3U : capabilities.max_image_count, 3U));
        desc.format = RACHIS_FORMAT_BGRA8_UNORM;
        desc.color_space = RACHIS_COLOR_SPACE_SRGB_NONLINEAR;
        desc.usage = RACHIS_TEXTURE_RENDER_TARGET;
        desc.present_mode = RACHIS_PRESENT_FIFO;
        rachis_handle swapchain = 0;
        requireOk(rachis_swapchain_create(device, &desc, &swapchain, &error), error);
        REQUIRE(swapchain != 0);
        return swapchain;
    }

    [[nodiscard]] rachis_swapchain_image acquireSwapchainImage(rachis_device* device,
                                                               rachis_handle swapchain,
                                                               rachis_error& error)
    {
        rachis_swapchain_image image{};
        image.struct_size = sizeof(image);
        requireOk(rachis_swapchain_acquire(device, swapchain, &image, &error), error);
        REQUIRE(image.texture != 0);
        REQUIRE(image.render_target_view != 0);
        return image;
    }

    [[nodiscard]] rachis_status queryBorrowedTextureLayout(rachis_device* device,
                                                           rachis_handle texture,
                                                           rachis_error& error)
    {
        rachis_buffer_texture_region region{};
        region.aspect = RACHIS_ASPECT_COLOR;
        region.width = 1;
        region.height = 1;
        region.depth = 1;
        rachis_texture_transfer_layout layout{};
        layout.struct_size = sizeof(layout);
        return rachis_texture_query_transfer_layout(device, texture, &region, &layout, &error);
    }
#endif

#if RACHIS_CABI_FAULT_BACKEND_METAL || RACHIS_CABI_FAULT_BACKEND_D3D12
    [[nodiscard]] rachis_status probeBorrowedViewRendering(rachis_device* device,
                                                           rachis_handle pool,
                                                           rachis_handle view,
                                                           rachis_error& error)
    {
        rachis_handle command = 0;
        requireOk(rachis_command_acquire(device, pool, &command, &error), error);
        rachis_color_attachment attachment{};
        attachment.view = view;
        attachment.load_op = RACHIS_LOAD_OP_CLEAR;
        attachment.store_op = RACHIS_STORE_OP_DISCARD;
        rachis_rendering_desc rendering{};
        rendering.struct_size = sizeof(rendering);
        rendering.abi_version = RACHIS_ABI_VERSION;
        rendering.color_attachments = &attachment;
        rendering.color_attachment_count = 1;
        const auto status = rachis_command_begin_rendering(device, command, &rendering, &error);
        if (status == RACHIS_STATUS_OK)
        {
            requireOk(rachis_command_end_rendering(device, command, &error), error);
        }
        auto discardError = makeError();
        requireOk(rachis_command_discard(device, command, &discardError), discardError);
        return status;
    }

    [[nodiscard]] rachis_handle recordAndSubmitImageFrame(rachis_device* device,
                                                          rachis_handle pool,
                                                          const rachis_swapchain_image& image,
                                                          rachis_error& error)
    {
        rachis_handle command = 0;
        requireOk(rachis_command_acquire(device, pool, &command, &error), error);

        rachis_texture_barrier toRender{};
        toRender.texture = image.texture;
        toRender.stages_before = image.initial_stages;
        toRender.access_before = image.initial_access;
        toRender.layout_before = image.initial_layout;
        toRender.stages_after = RACHIS_STAGE_RENDER_TARGET;
        toRender.access_after = RACHIS_ACCESS_RENDER_TARGET;
        toRender.layout_after = RACHIS_LAYOUT_RENDER_TARGET;
        toRender.aspect = RACHIS_ASPECT_COLOR;
        toRender.mip_level_count = RACHIS_REMAINING_SUBRESOURCES;
        toRender.array_layer_count = RACHIS_REMAINING_SUBRESOURCES;

        rachis_barrier_batch batch{};
        batch.struct_size = sizeof(batch);
        batch.abi_version = RACHIS_ABI_VERSION;
        batch.texture_barriers = &toRender;
        batch.texture_barrier_count = 1;
        requireOk(rachis_command_barriers(device, command, &batch, &error), error);

        rachis_color_attachment attachment{};
        attachment.view = image.render_target_view;
        attachment.load_op = RACHIS_LOAD_OP_CLEAR;
        attachment.store_op = RACHIS_STORE_OP_STORE;
        attachment.clear_color[0] = 0.2F;
        attachment.clear_color[1] = 0.5F;
        attachment.clear_color[2] = 0.8F;
        attachment.clear_color[3] = 1.0F;
        rachis_rendering_desc rendering{};
        rendering.struct_size = sizeof(rendering);
        rendering.abi_version = RACHIS_ABI_VERSION;
        rendering.color_attachments = &attachment;
        rendering.color_attachment_count = 1;
        requireOk(rachis_command_begin_rendering(device, command, &rendering, &error), error);
        requireOk(rachis_command_end_rendering(device, command, &error), error);

        rachis_texture_barrier toPresent{};
        toPresent.texture = image.texture;
        toPresent.stages_before = RACHIS_STAGE_RENDER_TARGET;
        toPresent.access_before = RACHIS_ACCESS_RENDER_TARGET;
        toPresent.layout_before = RACHIS_LAYOUT_RENDER_TARGET;
        toPresent.stages_after = image.required_final_stages;
        toPresent.access_after = image.required_final_access;
        toPresent.layout_after = image.required_final_layout;
        toPresent.aspect = RACHIS_ASPECT_COLOR;
        toPresent.mip_level_count = RACHIS_REMAINING_SUBRESOURCES;
        toPresent.array_layer_count = RACHIS_REMAINING_SUBRESOURCES;
        batch.texture_barriers = &toPresent;
        requireOk(rachis_command_barriers(device, command, &batch, &error), error);

        rachis_handle timeline = 0;
        requireOk(rachis_command_submit(device, command, &timeline, &error), error);
        REQUIRE(timeline != 0);
        return timeline;
    }
#endif

    enum class LossTrigger : std::uint8_t
    {
        injected,
        failedEndCommandBuffer,
        removedAdapter,
    };

#if RACHIS_CABI_FAULT_BACKEND_D3D12
    constexpr std::uint32_t lossValidation = RACHIS_VALIDATION_DISABLED;
#else
    constexpr std::uint32_t lossValidation = RACHIS_VALIDATION_STANDARD;
#endif

    [[nodiscard]] std::vector<LossTrigger> lossTriggers()
    {
#if RACHIS_CABI_FAULT_BACKEND_D3D12
        return {LossTrigger::removedAdapter};
#elif RACHIS_CABI_FAULT_BACKEND_VULKAN
        return {LossTrigger::injected, LossTrigger::failedEndCommandBuffer};
#else
        return {LossTrigger::injected};
#endif
    }

    [[nodiscard]] const char* lossTriggerName(LossTrigger trigger) noexcept
    {
        switch (trigger)
        {
        case LossTrigger::injected:
            return "injected device loss";
        case LossTrigger::failedEndCommandBuffer:
            return "vkEndCommandBuffer returns VK_ERROR_DEVICE_LOST";
        case LossTrigger::removedAdapter:
            return "ID3D12Device5::RemoveDevice on the shared WARP device";
        }
        return "unknown";
    }

#if RACHIS_CABI_FAULT_BACKEND_D3D12
    void removeSharedWarpDevice()
    {
        IDXGIFactory4* factory = nullptr;
        REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
        IDXGIAdapter* adapter = nullptr;
        const auto enumerated = factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        factory->Release();
        REQUIRE(SUCCEEDED(enumerated));
        ID3D12Device5* removable = nullptr;
        const auto created = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&removable));
        adapter->Release();
        REQUIRE(SUCCEEDED(created));
        removable->RemoveDevice();
        removable->Release();
    }
#endif

    void loseDevice(rachis_device* device, rachis_handle copyPool, LossTrigger trigger)
    {
        auto error = makeError();
        if (trigger == LossTrigger::injected)
        {
#if RACHIS_CABI_FAULT_BACKEND_METAL
            REQUIRE_FALSE(rachis::metal::injectMetalDeviceLoss(rachis::DeviceFaultStatus::removed));
#elif RACHIS_CABI_FAULT_BACKEND_VULKAN
            REQUIRE_FALSE(rachis::vulkan::injectVulkanDeviceLoss(rachis::DeviceFaultStatus::removed));
#else
            FAIL("This backend has no device-loss injection.");
#endif
        }
        else
        {
#if RACHIS_CABI_FAULT_BACKEND_D3D12
            removeSharedWarpDevice();
#endif
            rachis_handle command = 0;
            requireOk(rachis_command_acquire(device, copyPool, &command, &error), error);
            rachis_handle timeline = 0;
            rachis_status status = RACHIS_STATUS_OK;
            {
#if RACHIS_CABI_FAULT_BACKEND_VULKAN
                const ScopedVulkanSubmitFailure injectFailure(-4);
#endif
                status = rachis_command_submit(device, command, &timeline, &error);
            }
            if (status == RACHIS_STATUS_OK)
            {
                status = rachis_timeline_wait(device, timeline, &error);
                auto releaseError = makeError();
                requireOk(rachis_timeline_release(device, timeline, &releaseError), releaseError);
            }
            INFO(error.message);
            CHECK(status == RACHIS_STATUS_DEVICE_LOST);
        }
        std::uint32_t state = RACHIS_DEVICE_INVALID;
        requireOk(rachis_device_operational_state(device, &state, &error), error);
        REQUIRE(state == RACHIS_DEVICE_LOST);
    }

    [[nodiscard]] rachis_handle createSampledTexture(rachis_device* device, rachis_error& error)
    {
        rachis_texture_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.dimension = RACHIS_TEXTURE_2D;
        desc.format = RACHIS_FORMAT_RGBA8_UNORM;
        desc.usage = RACHIS_TEXTURE_SAMPLED | RACHIS_TEXTURE_COPY_DESTINATION;
        desc.width = 4;
        desc.height = 4;
        desc.depth_or_array_layers = 1;
        desc.mip_levels = 1;
        desc.sample_count = 1;
        desc.label = "C ABI device-loss texture";
        rachis_handle texture = 0;
        requireOk(rachis_texture_create(device, &desc, &texture, &error), error);
        REQUIRE(texture != 0);
        return texture;
    }

    [[nodiscard]] rachis_handle createSampledView(rachis_device* device, rachis_handle texture, rachis_error& error)
    {
        rachis_texture_view_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.texture = texture;
        desc.usage = RACHIS_VIEW_USAGE_SAMPLED;
        desc.dimension = RACHIS_VIEW_2D;
        desc.format = RACHIS_FORMAT_UNKNOWN;
        desc.aspect = RACHIS_ASPECT_COLOR;
        desc.mip_level_count = 1;
        desc.array_layer_count = 1;
        rachis_handle view = 0;
        requireOk(rachis_texture_view_create(device, &desc, &view, &error), error);
        REQUIRE(view != 0);
        return view;
    }
} // namespace

#if !RACHIS_CABI_FAULT_SKIP_HOST_ONLY
TEST_CASE("MonotonicIdentitySource and allocateCAbiToken saturate permanently without wrapping",
          "[size:small][rachis][cabi][tokens]")
{
    constexpr auto maxHandle = std::numeric_limits<rachis_handle>::max();
    constexpr auto maxTag = rachis::cabi::detail::maxCAbiTokenTag;

    rachis::detail::MonotonicIdentitySource rawSource(maxHandle - 1);
    CHECK(rawSource.acquire() == maxHandle - 1);
    CHECK(rawSource.acquire() == maxHandle);
    CHECK_FALSE(rawSource.acquire().has_value());
    CHECK_FALSE(rawSource.acquire().has_value());

    rachis::detail::MonotonicIdentitySource cSource(maxTag - 1);
    CHECK(rachis::cabi::detail::allocateCAbiToken(cSource) == maxTag - 1);
    CHECK(rachis::cabi::detail::allocateCAbiToken(cSource) == maxTag);

    CHECK(rachis::cabi::detail::allocateCAbiToken(cSource) == 0);
    CHECK(rachis::cabi::detail::allocateCAbiToken(cSource) == 0);

    rachis::detail::MonotonicIdentitySource parkedSource(maxHandle);
    CHECK(rachis::cabi::detail::allocateCAbiToken(parkedSource) == 0);
    CHECK(rachis::cabi::detail::allocateCAbiToken(parkedSource) == 0);

    rachis::detail::MonotonicIdentitySource contendedSource(maxTag - 2);
    std::mutex mutex;
    std::vector<rachis_handle> issued;
    std::size_t zeroCount = 0;
    std::array<std::thread, 4> threads;
    for (auto& worker : threads)
    {
        worker = std::thread([&] {
            for (int iteration = 0; iteration < 16; ++iteration)
            {
                const auto id = rachis::cabi::detail::allocateCAbiToken(contendedSource);
                const std::scoped_lock lock(mutex);
                if (id == 0)
                {
                    ++zeroCount;
                }
                else
                {
                    issued.push_back(id);
                }
            }
        });
    }
    for (auto& worker : threads)
    {
        worker.join();
    }
    std::sort(issued.begin(), issued.end());
    const std::vector<rachis_handle> expected{maxTag - 2, maxTag - 1, maxTag};
    CHECK(issued == expected);
    CHECK(zeroCount == 4 * 16 - expected.size());
}

TEST_CASE("copyText cuts C ABI text at a UTF-8 character boundary and always terminates it",
          "[size:small][rachis][cabi][text]")
{
    std::array<char, 8> buffer{};
    const auto copied = [&](std::string_view text) {
        buffer.fill('#');
        rachis::cabi::detail::copyText(buffer, text);
        return std::string{buffer.data()};
    };

    CHECK(copied("") == "");
    CHECK(copied("abcdefg") == "abcdefg");
    CHECK(copied("abcdefgh") == "abcdefg");
    CHECK(copied("abcde\xC3\xA9") == "abcde\xC3\xA9");
    CHECK(copied("abcdef\xC3\xA9") == "abcdef");
    CHECK(copied("abcd\xE3\x83\x86z") == "abcd\xE3\x83\x86");
    CHECK(copied("abcde\xE3\x83\x86") == "abcde");
    CHECK(copied("abcdef\xE3\x83\x86") == "abcdef");
    CHECK(copied("abc\xF0\x9F\x98\x80z") == "abc\xF0\x9F\x98\x80");
    CHECK(copied("abcd\xF0\x9F\x98\x80") == "abcd");
    CHECK(copied("abcdef\xF0\x9F\x98\x80") == "abcdef");
    CHECK(copied("abc\x80\x80\x80\x80\x80") == "abc\x80\x80\x80\x80");
    CHECK(buffer.back() == '\0');

    CHECK(rachis::cabi::detail::utf8PrefixLength("\xC3\xA9", 1) == 0);
    CHECK(rachis::cabi::detail::utf8PrefixLength("\x80\x80", 1) == 1);

    std::array<char, 1> single{'#'};
    rachis::cabi::detail::copyText(single, "abc");
    CHECK(single[0] == '\0');
    rachis::cabi::detail::copyText(std::span<char>{}, "abc");
}

TEST_CASE("CAbiSlotTable grows geometrically and preserves cross-table and recycled handle isolation",
          "[size:small][rachis][cabi][tokens]")
{
    using Table = rachis::cabi::detail::CAbiSlotTable<std::uint64_t>;
    Table table;

    CHECK(table.find(0) == nullptr);
    CHECK(table.find(RACHIS_WHOLE_SIZE) == nullptr);
    CHECK_FALSE(table.erase(0));
    CHECK_FALSE(table.erase(RACHIS_WHOLE_SIZE));

    rachis::detail::MonotonicIdentitySource source(1);
    constexpr std::size_t entryCount = 8192;
    std::vector<rachis_handle> firstWave;
    firstWave.reserve(entryCount);

    for (std::uint64_t i = 0; i < entryCount; ++i)
    {
        const auto handle = table.emplace(source, i + 100U);
        REQUIRE(handle != 0);
        REQUIRE(handle != RACHIS_WHOLE_SIZE);
        firstWave.push_back(handle);
    }

    CHECK(table.size() == entryCount);

    for (std::size_t i = 0; i < entryCount; ++i)
    {
        const auto* found = table.find(firstWave[i]);
        REQUIRE(found != nullptr);
        CHECK(*found == i + 100U);
    }

    for (std::size_t i = 0; i < entryCount; i += 2)
    {
        REQUIRE(table.erase(firstWave[i]));
        CHECK(table.find(firstWave[i]) == nullptr);
        CHECK_FALSE(table.erase(firstWave[i]));
    }
    CHECK(table.size() == entryCount / 2);

    table.clear();
    CHECK(table.size() == 0);

    std::vector<rachis_handle> secondWave;
    secondWave.reserve(entryCount);
    for (std::uint64_t i = 0; i < entryCount; ++i)
    {
        const auto recycled = table.emplace(source, i + 50000U);
        REQUIRE(recycled != 0);
        REQUIRE(recycled != RACHIS_WHOLE_SIZE);
        secondWave.push_back(recycled);
    }

    for (std::size_t i = 0; i < entryCount; ++i)
    {
        CHECK(table.find(firstWave[i]) == nullptr);
        const auto* found = table.find(secondWave[i]);
        REQUIRE(found != nullptr);
        CHECK(*found == i + 50000U);
    }

    rachis_handle destroyedDeviceHandle = 0;
    {
        Table transientDeviceTable;
        destroyedDeviceHandle = transientDeviceTable.emplace(source, 777U);
        REQUIRE(destroyedDeviceHandle != 0);
    }
    Table replacementDeviceTable;
    Table siblingTable;
    const auto replacementHandle = replacementDeviceTable.emplace(source, 888U);
    const auto siblingHandle = siblingTable.emplace(source, 999U);
    CHECK(replacementHandle != destroyedDeviceHandle);
    CHECK(siblingHandle != replacementHandle);
    CHECK(replacementDeviceTable.find(destroyedDeviceHandle) == nullptr);
    CHECK(replacementDeviceTable.find(siblingHandle) == nullptr);
    CHECK(siblingTable.find(replacementHandle) == nullptr);

    rachis::cabi::detail::CAbiSlotTable<std::uint64_t, 4> bounded;
    rachis::detail::MonotonicIdentitySource boundedSource(rachis::cabi::detail::maxCAbiTokenTag - 4);
    std::array<rachis_handle, 4> boundedHandles{};
    for (std::size_t i = 0; i < boundedHandles.size(); ++i)
    {
        boundedHandles[i] = bounded.emplace(boundedSource, i + 1U);
        REQUIRE(boundedHandles[i] != 0);
        CHECK(boundedHandles[i] != RACHIS_WHOLE_SIZE);
    }

    CHECK(bounded.emplace(boundedSource, 99U) == 0);
    REQUIRE(bounded.erase(boundedHandles[3]));
    const auto maxTagHandle = bounded.emplace(boundedSource, 42U);
    REQUIRE(maxTagHandle != 0);
    CHECK(maxTagHandle != RACHIS_WHOLE_SIZE);
    REQUIRE(bounded.find(maxTagHandle) != nullptr);
    CHECK(*bounded.find(maxTagHandle) == 42U);
}

TEST_CASE("CAbiBorrowedImageHandles suspends restores and erases in O(1) without scanning live tokens",
          "[size:small][rachis][cabi][present][tokens]")
{
    struct FakeToken final
    {
        std::uint64_t native = 0;
    };

    rachis::cabi::detail::CAbiSlotTable<FakeToken> tokens;
    rachis::detail::MonotonicIdentitySource source(1);

    constexpr std::size_t liveResourceCount = 4096;
    std::vector<rachis_handle> resources;
    resources.reserve(liveResourceCount);
    for (std::uint64_t i = 0; i < liveResourceCount; ++i)
    {
        const auto handle = tokens.emplace(source, FakeToken{.native = i + 1U});
        REQUIRE(handle != 0);
        resources.push_back(handle);
    }

    rachis::cabi::detail::CAbiBorrowedImageHandles swapchainA;
    rachis::cabi::detail::CAbiBorrowedImageHandles swapchainB;
    const auto texA = tokens.emplace(source, FakeToken{.native = 0xA001U});
    const auto viewA = tokens.emplace(source, FakeToken{.native = 0xA002U});
    const auto texB = tokens.emplace(source, FakeToken{.native = 0xB001U});
    const auto viewB = tokens.emplace(source, FakeToken{.native = 0xB002U});
    swapchainA.publish(texA, viewA);
    swapchainB.publish(texB, viewB);

    REQUIRE(tokens.size() == liveResourceCount + 4);

    const auto suspendedA = swapchainA.suspend(tokens);
    CHECK(tokens.find(texA)->native == 0);
    CHECK(tokens.find(viewA)->native == 0);
    CHECK(tokens.find(texB)->native == 0xB001U);
    CHECK(tokens.find(viewB)->native == 0xB002U);

    rachis::cabi::detail::CAbiBorrowedImageHandles::restoreSuspended(tokens, suspendedA);
    CHECK(tokens.find(texA)->native == 0xA001U);
    CHECK(tokens.find(viewA)->native == 0xA002U);

    const auto suspendedAgain = swapchainA.suspend(tokens);
    swapchainA.eraseSuspended(tokens, suspendedAgain);
    CHECK(tokens.find(texA) == nullptr);
    CHECK(tokens.find(viewA) == nullptr);
    CHECK(swapchainA.handles() == std::array<rachis_handle, 2>{0, 0});
    CHECK(tokens.find(texB)->native == 0xB001U);
    CHECK(tokens.find(viewB)->native == 0xB002U);
    CHECK(tokens.size() == liveResourceCount + 2);

    swapchainB.erase(tokens);
    CHECK(tokens.find(texB) == nullptr);
    CHECK(tokens.find(viewB) == nullptr);
    CHECK(tokens.size() == liveResourceCount);

    for (std::size_t i = 0; i < liveResourceCount; ++i)
    {
        REQUIRE(tokens.find(resources[i]) != nullptr);
        CHECK(tokens.find(resources[i])->native == i + 1U);
    }
}

namespace
{
    struct DeviceMutexRound final
    {
        bool firstReaderHeldInTime = false;
        bool sharedAdmittedBeforeWriter = false;
        bool writerQueuedInTime = false;
        bool lateReaderEnteredEarly = false;
        bool lateReaderEntered = false;
        bool lateReaderSawFirstReaderHolding = false;
        int writerOrder = 0;
        int reentryOrder = 0;
        int lateReaderOrder = 0;
    };

    [[nodiscard]] DeviceMutexRound runDeviceMutexRound(bool withLateReader)
    {
        rachis::cabi::detail::CAbiDeviceMutex mutex;
        std::mutex stateMutex;
        std::condition_variable stateCv;
        bool firstReaderHolding = false;
        bool releaseFirstReader = false;
        std::atomic<int> sequence{0};
        DeviceMutexRound round;

        std::thread firstReader([&] {
            {
                const std::shared_lock lock(mutex);
                std::unique_lock state(stateMutex);
                firstReaderHolding = true;
                stateCv.notify_all();
                stateCv.wait(state, [&] { return releaseFirstReader; });
            }
            const std::shared_lock lock(mutex);
            round.reentryOrder = sequence.fetch_add(1) + 1;
        });
        {
            std::unique_lock state(stateMutex);
            round.firstReaderHeldInTime =
                stateCv.wait_for(state, std::chrono::seconds(10), [&] { return firstReaderHolding; });
        }

        std::thread writer;
        std::thread lateReader;
        if (round.firstReaderHeldInTime)
        {
            round.sharedAdmittedBeforeWriter = mutex.try_lock_shared();
            if (round.sharedAdmittedBeforeWriter)
            {
                mutex.unlock_shared();
            }

            writer = std::thread([&] {
                const std::scoped_lock lock(mutex);
                round.writerOrder = sequence.fetch_add(1) + 1;
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (!mutex.try_lock_shared())
                {
                    round.writerQueuedInTime = true;
                    break;
                }
                mutex.unlock_shared();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            if (round.writerQueuedInTime && withLateReader)
            {
                lateReader = std::thread([&] {
                    const std::shared_lock lock(mutex);
                    round.lateReaderOrder = sequence.fetch_add(1) + 1;
                    const std::scoped_lock state(stateMutex);
                    round.lateReaderEntered = true;
                    round.lateReaderSawFirstReaderHolding = !releaseFirstReader;
                    stateCv.notify_all();
                });
                std::unique_lock state(stateMutex);
                round.lateReaderEnteredEarly =
                    stateCv.wait_for(state, std::chrono::milliseconds(100), [&] { return round.lateReaderEntered; });
            }
        }

        {
            const std::scoped_lock state(stateMutex);
            releaseFirstReader = true;
        }
        stateCv.notify_all();
        firstReader.join();
        if (writer.joinable())
        {
            writer.join();
        }
        if (lateReader.joinable())
        {
            lateReader.join();
        }
        return round;
    }
} // namespace

TEST_CASE("CAbiDeviceMutex admits a waiting exclusive caller before shared callers that arrive or return after it",
          "[size:small][rachis][cabi][concurrency]")
{
    rachis::cabi::detail::CAbiDeviceMutex mutex;
    REQUIRE(mutex.try_lock());
    CHECK_FALSE(mutex.try_lock());
    CHECK_FALSE(mutex.try_lock_shared());
    mutex.unlock();
    REQUIRE(mutex.try_lock_shared());
    REQUIRE(mutex.try_lock_shared());
    CHECK_FALSE(mutex.try_lock());
    mutex.unlock_shared();
    mutex.unlock_shared();

    constexpr int roundCount = 32;
    for (int roundIndex = 0; roundIndex < roundCount; ++roundIndex)
    {
        const bool withLateReader = roundIndex == 0;
        const auto round = runDeviceMutexRound(withLateReader);
        INFO("round " << roundIndex);
        REQUIRE(round.firstReaderHeldInTime);
        CHECK(round.sharedAdmittedBeforeWriter);
        REQUIRE(round.writerQueuedInTime);
        CHECK(round.writerOrder == 1);
        CHECK(round.reentryOrder > round.writerOrder);
        if (withLateReader)
        {
            CHECK_FALSE(round.lateReaderEnteredEarly);
            CHECK(round.lateReaderEntered);
            CHECK_FALSE(round.lateReaderSawFirstReaderHolding);
            CHECK(round.lateReaderOrder > round.writerOrder);
        }
    }
}

namespace
{
    template<typename Body>
    void runDeviceMutexThreads(std::size_t threadCount, const Body& body)
    {
        std::latch start(static_cast<std::ptrdiff_t>(threadCount));
        std::vector<std::thread> threads;
        threads.reserve(threadCount);
        for (std::size_t index = 0; index < threadCount; ++index)
        {
            threads.emplace_back([&start, &body, index] {
                start.arrive_and_wait();
                body(index);
            });
        }
        for (auto& thread : threads)
        {
            thread.join();
        }
    }
} // namespace

TEST_CASE("CAbiDeviceMutex try_lock_shared succeeds whenever no exclusive caller holds or waits for the lock",
          "[size:small][rachis][cabi][concurrency]")
{
    constexpr std::size_t threadCount = 8;
    constexpr int iterationCount = 20000;
    rachis::cabi::detail::CAbiDeviceMutex mutex;
    std::atomic<int> refusedCount{0};

    runDeviceMutexThreads(threadCount, [&](std::size_t) {
        int refused = 0;
        for (int iteration = 0; iteration < iterationCount; ++iteration)
        {
            mutex.lock_shared();
            if (mutex.try_lock_shared())
            {
                mutex.unlock_shared();
            }
            else
            {
                ++refused;
            }
            mutex.unlock_shared();
            if (mutex.try_lock_shared())
            {
                mutex.unlock_shared();
            }
            else
            {
                ++refused;
            }
        }
        refusedCount.fetch_add(refused);
    });

    CHECK(refusedCount.load() == 0);
}

TEST_CASE("CAbiDeviceMutex keeps exclusive callers alone and shared callers consistent under mixed contention",
          "[size:small][rachis][cabi][concurrency]")
{
    constexpr std::size_t exclusiveThreadCount = 2;
    constexpr std::size_t sharedThreadCount = 6;
    constexpr int iterationCount = 2000;
    rachis::cabi::detail::CAbiDeviceMutex mutex;
    std::atomic<int> exclusiveHolders{0};
    std::atomic<int> sharedHolders{0};
    std::atomic<int> violationCount{0};
    std::uint64_t first = 0;
    std::uint64_t second = 0;

    runDeviceMutexThreads(exclusiveThreadCount + sharedThreadCount, [&](std::size_t index) {
        int violations = 0;
        for (int iteration = 0; iteration < iterationCount; ++iteration)
        {
            const bool tryFirst = iteration % 2 != 0;
            if (index < exclusiveThreadCount)
            {
                if (!(tryFirst && mutex.try_lock()))
                {
                    mutex.lock();
                }
                if (exclusiveHolders.fetch_add(1) != 0 || sharedHolders.load() != 0)
                {
                    ++violations;
                }
                ++first;
                ++second;
                exclusiveHolders.fetch_sub(1);
                mutex.unlock();
            }
            else
            {
                if (!(tryFirst && mutex.try_lock_shared()))
                {
                    mutex.lock_shared();
                }
                sharedHolders.fetch_add(1);
                if (exclusiveHolders.load() != 0 || first != second)
                {
                    ++violations;
                }
                sharedHolders.fetch_sub(1);
                mutex.unlock_shared();
            }
        }
        violationCount.fetch_add(violations);
    });

    CHECK(violationCount.load() == 0);
    CHECK(first == exclusiveThreadCount * iterationCount);
    CHECK(second == first);
    REQUIRE(mutex.try_lock());
    mutex.unlock();
}

TEST_CASE("CAbiDeviceMutex admits a waiting shared caller before exclusive callers that queue after it",
          "[size:small][rachis][cabi][concurrency]")
{
    constexpr int sharedRoundCount = 32;
    rachis::cabi::detail::CAbiDeviceMutex mutex;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::atomic<bool> sharedDone{false};
    std::atomic<int> exclusiveRounds{0};

    std::vector<std::thread> exclusiveThreads;
    for (int thread = 0; thread < 2; ++thread)
    {
        exclusiveThreads.emplace_back([&] {
            while (!sharedDone.load() && std::chrono::steady_clock::now() < deadline)
            {
                const std::scoped_lock lock(mutex);
                exclusiveRounds.fetch_add(1);
                const auto holdUntil = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
                while (std::chrono::steady_clock::now() < holdUntil)
                {
                    std::this_thread::yield();
                }
            }
        });
    }
    while (exclusiveRounds.load() < 4 && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::yield();
    }

    int sharedRounds = 0;
    while (sharedRounds < sharedRoundCount && std::chrono::steady_clock::now() < deadline)
    {
        const std::shared_lock lock(mutex);
        ++sharedRounds;
    }
    sharedDone.store(true);
    for (auto& thread : exclusiveThreads)
    {
        thread.join();
    }

    CHECK(exclusiveRounds.load() >= 4);
    CHECK(sharedRounds == sharedRoundCount);
}
#endif

TEST_CASE("C ABI allows concurrent command recording across pools under shared lock while rejecting concurrent "
          "same-lease use",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][concurrency]")
{
    constexpr std::size_t parallelWorkerCount = 4;
    constexpr std::size_t totalSlots = parallelWorkerCount + 1;
    constexpr std::size_t byteCount = 16;

    auto error = makeError();
    auto device = createDevice(error);

    const std::array<std::uint8_t, byteCount> sharedExpected{
        11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176};
    const auto sharedUpload =
        createBuffer(device.get(), byteCount, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto sharedReadback =
        createBuffer(device.get(), byteCount, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    requireOk(rachis_buffer_write(device.get(), sharedUpload, 0, sharedExpected.data(), byteCount, &error), error);
    {
        const auto setupPool = createPool(device.get(), RACHIS_QUEUE_COPY, error);
        rachis_handle setupCommand = 0;
        requireOk(rachis_command_acquire(device.get(), setupPool, &setupCommand, &error), error);
        requireOk(rachis_command_copy_buffer(
                      device.get(), setupCommand, sharedUpload, sharedReadback, 0, 0, byteCount, &error),
                  error);
        rachis_handle setupTimeline = 0;
        requireOk(rachis_command_submit(device.get(), setupCommand, &setupTimeline, &error), error);
        requireOk(rachis_timeline_wait(device.get(), setupTimeline, &error), error);
        requireOk(rachis_timeline_release(device.get(), setupTimeline, &error), error);
        requireOk(rachis_command_pool_destroy(device.get(), setupPool, &error), error);
    }
    requireOk(rachis_buffer_destroy(device.get(), sharedUpload, &error), error);

    rachis_texture_desc texDesc{};
    texDesc.struct_size = sizeof(texDesc);
    texDesc.abi_version = RACHIS_ABI_VERSION;
    texDesc.dimension = RACHIS_TEXTURE_2D;
    texDesc.format = RACHIS_FORMAT_RGBA8_UNORM;
    texDesc.usage = RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_COPY_SOURCE;
    texDesc.width = 4;
    texDesc.height = 4;
    texDesc.depth_or_array_layers = 1;
    texDesc.mip_levels = 1;
    texDesc.sample_count = 1;
    texDesc.label = "C ABI concurrency texture";

    struct RecordingSlot final
    {
        rachis_handle pool = 0;
        rachis_handle command = 0;
        rachis_handle timeline = 0;
        rachis_handle upload = 0;
        rachis_handle readback = 0;
        rachis_handle texture = 0;
        rachis_handle view = 0;
        std::array<std::uint8_t, byteCount> expected{};
        std::array<std::uint8_t, byteCount> actual{};
        rachis_status status = RACHIS_STATUS_OK;
        std::string failureMessage;
    };

    std::array<RecordingSlot, totalSlots> slots{};
    for (std::size_t i = 0; i < totalSlots; ++i)
    {
        auto& slot = slots[i];
        slot.pool = createPool(device.get(), RACHIS_QUEUE_GRAPHICS, error);
        requireOk(rachis_command_acquire(device.get(), slot.pool, &slot.command, &error), error);
        slot.upload = createBuffer(device.get(), byteCount, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
        slot.readback =
            createBuffer(device.get(), byteCount, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
        requireOk(rachis_texture_create(device.get(), &texDesc, &slot.texture, &error), error);
        rachis_texture_view_desc viewDesc{};
        viewDesc.struct_size = sizeof(viewDesc);
        viewDesc.abi_version = RACHIS_ABI_VERSION;
        viewDesc.texture = slot.texture;
        viewDesc.usage = RACHIS_VIEW_USAGE_RENDER_TARGET;
        viewDesc.dimension = RACHIS_VIEW_2D;
        viewDesc.format = RACHIS_FORMAT_UNKNOWN;
        viewDesc.aspect = RACHIS_ASPECT_COLOR;
        viewDesc.mip_level_count = 1;
        viewDesc.array_layer_count = 1;
        requireOk(rachis_texture_view_create(device.get(), &viewDesc, &slot.view, &error), error);
        for (std::size_t b = 0; b < byteCount; ++b)
        {
            slot.expected[b] = static_cast<std::uint8_t>((i + 1U) * 29U + b * 3U);
        }
    }
    requireOk(rachis_buffer_write(device.get(), slots[0].upload, 0, slots[0].expected.data(), byteCount, &error),
              error);

    struct CommandHookContext final
    {
        rachis_handle pausedCommand = 0;
        std::mutex mutex;
        std::condition_variable cv;
        bool pausedEntered = false;
        bool releasePaused = false;
    } hookContext;
    hookContext.pausedCommand = slots[0].command;

    const auto commandHook = [](rachis_handle command, void* userData) noexcept {
        auto& ctx = *static_cast<CommandHookContext*>(userData);
        if (command == ctx.pausedCommand)
        {
            std::unique_lock lock(ctx.mutex);
            if (!ctx.pausedEntered)
            {
                ctx.pausedEntered = true;
                ctx.cv.notify_all();
                ctx.cv.wait(lock, [&] { return ctx.releasePaused; });
            }
        }
    };

    rachis_status sameLeaseCopyStatus = RACHIS_STATUS_OK;
    rachis_error sameLeaseCopyError = makeError();
    rachis_status sameLeaseEndStatus = RACHIS_STATUS_OK;
    rachis_error sameLeaseEndError = makeError();
    bool pausedEnteredInTime = false;

    {
        const ScopedCommandRecordingHook scopedHook(commandHook, &hookContext);

        std::thread pausedThread([&] {
            auto localError = makeError();
            slots[0].status = rachis_command_copy_buffer(
                device.get(), slots[0].command, slots[0].upload, slots[0].readback, 0, 0, byteCount, &localError);
            slots[0].failureMessage = localError.message;
        });

        {
            std::unique_lock lock(hookContext.mutex);
            pausedEnteredInTime =
                hookContext.cv.wait_for(lock, std::chrono::seconds(10), [&] { return hookContext.pausedEntered; });
        }
        if (pausedEnteredInTime)
        {
            std::thread sameLeaseMisuseThread([&] {
                sameLeaseCopyStatus = rachis_command_copy_buffer(device.get(),
                                                                 slots[0].command,
                                                                 slots[0].upload,
                                                                 slots[0].readback,
                                                                 0,
                                                                 0,
                                                                 byteCount,
                                                                 &sameLeaseCopyError);
                sameLeaseEndStatus = rachis_command_end_rendering(device.get(), slots[0].command, &sameLeaseEndError);
            });
            sameLeaseMisuseThread.join();

            std::array<std::thread, parallelWorkerCount> workers{};
            for (std::size_t w = 0; w < parallelWorkerCount; ++w)
            {
                workers[w] = std::thread([&, slotIndex = w + 1U] {
                    auto& slot = slots[slotIndex];
                    auto localError = makeError();
                    const auto checkStep = [&](rachis_status stepStatus) {
                        if (stepStatus != RACHIS_STATUS_OK && slot.status == RACHIS_STATUS_OK)
                        {
                            slot.status = stepStatus;
                            slot.failureMessage = localError.message;
                        }
                        return stepStatus == RACHIS_STATUS_OK;
                    };

                    if (!checkStep(rachis_buffer_write(
                            device.get(), slot.upload, 0, slot.expected.data(), byteCount, &localError)))
                    {
                        return;
                    }
                    std::array<std::uint8_t, byteCount> readBytes{};
                    if (!checkStep(rachis_buffer_read(
                            device.get(), sharedReadback, 0, readBytes.data(), byteCount, &localError)))
                    {
                        return;
                    }
                    if (readBytes != sharedExpected)
                    {
                        slot.status = RACHIS_STATUS_NATIVE_FAILURE;
                        slot.failureMessage = "shared readback mismatch";
                        return;
                    }

                    rachis_buffer_texture_region region{};
                    region.aspect = RACHIS_ASPECT_COLOR;
                    region.width = 4;
                    region.height = 4;
                    region.depth = 1;
                    rachis_texture_transfer_layout transfer{};
                    transfer.struct_size = sizeof(transfer);
                    if (!checkStep(rachis_texture_query_transfer_layout(
                            device.get(), slot.texture, &region, &transfer, &localError)))
                    {
                        return;
                    }

                    rachis_diagnostic_counts counts{};
                    counts.struct_size = sizeof(counts);
                    if (!checkStep(rachis_device_diagnostic_counts(device.get(), &counts, &localError)))
                    {
                        return;
                    }

                    if (!checkStep(rachis_command_copy_buffer(
                            device.get(), slot.command, slot.upload, slot.readback, 0, 0, byteCount, &localError)))
                    {
                        return;
                    }

                    rachis_texture_barrier toRender{};
                    toRender.texture = slot.texture;
                    toRender.stages_before = RACHIS_STAGE_ALL;
                    toRender.access_before = RACHIS_ACCESS_NONE;
                    toRender.layout_before = RACHIS_LAYOUT_UNDEFINED;
                    toRender.stages_after = RACHIS_STAGE_RENDER_TARGET;
                    toRender.access_after = RACHIS_ACCESS_RENDER_TARGET;
                    toRender.layout_after = RACHIS_LAYOUT_RENDER_TARGET;
                    toRender.aspect = RACHIS_ASPECT_COLOR;
                    toRender.mip_level_count = RACHIS_REMAINING_SUBRESOURCES;
                    toRender.array_layer_count = RACHIS_REMAINING_SUBRESOURCES;
                    rachis_barrier_batch batch{};
                    batch.struct_size = sizeof(batch);
                    batch.abi_version = RACHIS_ABI_VERSION;
                    batch.texture_barriers = &toRender;
                    batch.texture_barrier_count = 1;
                    if (!checkStep(rachis_command_barriers(device.get(), slot.command, &batch, &localError)))
                    {
                        return;
                    }

                    rachis_color_attachment attachment{};
                    attachment.view = slot.view;
                    attachment.load_op = RACHIS_LOAD_OP_CLEAR;
                    attachment.store_op = RACHIS_STORE_OP_STORE;
                    attachment.clear_color[0] = 0.25F;
                    attachment.clear_color[1] = 0.5F;
                    attachment.clear_color[2] = 0.75F;
                    attachment.clear_color[3] = 1.0F;
                    rachis_rendering_desc rendering{};
                    rendering.struct_size = sizeof(rendering);
                    rendering.abi_version = RACHIS_ABI_VERSION;
                    rendering.color_attachments = &attachment;
                    rendering.color_attachment_count = 1;
                    if (!checkStep(rachis_command_begin_rendering(device.get(), slot.command, &rendering, &localError)))
                    {
                        return;
                    }
                    static_cast<void>(checkStep(rachis_command_end_rendering(device.get(), slot.command, &localError)));
                });
            }
            for (auto& worker : workers)
            {
                worker.join();
            }
        }

        {
            const std::scoped_lock lock(hookContext.mutex);
            hookContext.releasePaused = true;
        }
        hookContext.cv.notify_all();
        pausedThread.join();
    }

    REQUIRE(pausedEnteredInTime);
    CHECK(sameLeaseCopyStatus == RACHIS_STATUS_INVALID_STATE);
    CHECK(sameLeaseCopyError.code == RACHIS_STATUS_INVALID_STATE);
    CHECK(sameLeaseEndStatus == RACHIS_STATUS_INVALID_STATE);
    CHECK(sameLeaseEndError.code == RACHIS_STATUS_INVALID_STATE);

    for (std::size_t i = 0; i < totalSlots; ++i)
    {
        auto& slot = slots[i];
        INFO("slot " << i << ": " << slot.failureMessage);
        REQUIRE(slot.status == RACHIS_STATUS_OK);
        requireOk(rachis_command_submit(device.get(), slot.command, &slot.timeline, &error), error);
    }
    for (auto& slot : slots)
    {
        requireOk(rachis_timeline_wait(device.get(), slot.timeline, &error), error);
        requireOk(rachis_timeline_release(device.get(), slot.timeline, &error), error);
        requireOk(rachis_buffer_read(device.get(), slot.readback, 0, slot.actual.data(), byteCount, &error), error);
        CHECK(slot.actual == slot.expected);
    }

    for (const auto& slot : slots)
    {
        requireOk(rachis_texture_view_destroy(device.get(), slot.view, &error), error);
        requireOk(rachis_texture_destroy(device.get(), slot.texture, &error), error);
        requireOk(rachis_buffer_destroy(device.get(), slot.readback, &error), error);
        requireOk(rachis_buffer_destroy(device.get(), slot.upload, &error), error);
        requireOk(rachis_command_pool_destroy(device.get(), slot.pool, &error), error);
    }
    requireOk(rachis_buffer_destroy(device.get(), sharedReadback, &error), error);
}

TEST_CASE("C ABI submit waits for recording already in progress but not for recording that starts or repeats after it",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][concurrency]")
{
    constexpr std::uint64_t byteCount = 16;

    auto error = makeError();
    auto device = createDevice(error);

    const std::array<std::uint8_t, byteCount> expected{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3};
    const auto upload = createBuffer(device.get(), byteCount, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback =
        createBuffer(device.get(), byteCount, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    requireOk(rachis_buffer_write(device.get(), upload, 0, expected.data(), byteCount, &error), error);

    const auto heldPool = createPool(device.get(), RACHIS_QUEUE_COPY, error);
    const auto submittedPool = createPool(device.get(), RACHIS_QUEUE_COPY, error);
    const auto latePool = createPool(device.get(), RACHIS_QUEUE_COPY, error);
    rachis_handle heldCommand = 0;
    rachis_handle submittedCommand = 0;
    rachis_handle lateCommand = 0;
    requireOk(rachis_command_acquire(device.get(), heldPool, &heldCommand, &error), error);
    requireOk(rachis_command_acquire(device.get(), submittedPool, &submittedCommand, &error), error);
    requireOk(rachis_command_acquire(device.get(), latePool, &lateCommand, &error), error);
    requireOk(rachis_command_copy_buffer(device.get(), submittedCommand, upload, readback, 0, 0, byteCount, &error),
              error);

    struct LockOrderContext final
    {
        rachis_handle heldCommand = 0;
        rachis_handle lateCommand = 0;
        std::mutex mutex;
        std::condition_variable cv;
        bool heldEntered = false;
        bool releaseHeld = false;
        bool heldReentered = false;
        bool reentrySawSubmitReturn = false;
        bool submitReturned = false;
        bool lateEntered = false;
        bool lateEnteredWhileHeld = false;
    } ctx;
    ctx.heldCommand = heldCommand;
    ctx.lateCommand = lateCommand;

    const auto recordingHook = [](rachis_handle command, void* userData) noexcept {
        auto& hookCtx = *static_cast<LockOrderContext*>(userData);
        std::unique_lock lock(hookCtx.mutex);
        if (command == hookCtx.heldCommand && !hookCtx.heldEntered)
        {
            hookCtx.heldEntered = true;
            hookCtx.cv.notify_all();
            hookCtx.cv.wait(lock, [&] { return hookCtx.releaseHeld; });
        }
        else if (command == hookCtx.heldCommand)
        {
            hookCtx.heldReentered = true;
            hookCtx.reentrySawSubmitReturn =
                hookCtx.cv.wait_for(lock, std::chrono::seconds(10), [&] { return hookCtx.submitReturned; });
        }
        else if (command == hookCtx.lateCommand)
        {
            hookCtx.lateEntered = true;
            hookCtx.lateEnteredWhileHeld = !hookCtx.releaseHeld;
            hookCtx.cv.notify_all();
        }
    };

    rachis_status heldStatus = RACHIS_STATUS_NATIVE_FAILURE;
    rachis_error heldError = makeError();
    rachis_status reentryStatus = RACHIS_STATUS_NATIVE_FAILURE;
    rachis_error reentryError = makeError();
    rachis_status submitStatus = RACHIS_STATUS_NATIVE_FAILURE;
    rachis_error submitError = makeError();
    rachis_handle timeline = 0;
    rachis_status lateStatus = RACHIS_STATUS_NATIVE_FAILURE;
    rachis_error lateError = makeError();
    bool heldEnteredInTime = false;
    bool submitQueuedInTime = false;
    bool lateEnteredEarly = false;

    {
        const ScopedCommandRecordingHook scopedHook(recordingHook, &ctx);

        std::thread heldThread([&] {
            heldStatus =
                rachis_command_copy_buffer(device.get(), heldCommand, upload, readback, 0, 0, byteCount, &heldError);
            reentryStatus =
                rachis_command_copy_buffer(device.get(), heldCommand, upload, readback, 0, 0, byteCount, &reentryError);
        });
        {
            std::unique_lock lock(ctx.mutex);
            heldEnteredInTime = ctx.cv.wait_for(lock, std::chrono::seconds(10), [&] { return ctx.heldEntered; });
        }

        std::thread submitThread;
        std::thread lateThread;
        if (heldEnteredInTime)
        {
            submitThread = std::thread([&] {
                submitStatus = rachis_command_submit(device.get(), submittedCommand, &timeline, &submitError);
                {
                    const std::scoped_lock lock(ctx.mutex);
                    ctx.submitReturned = true;
                }
                ctx.cv.notify_all();
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (!rachis::cabi::detail::deviceLockAdmitsSharedCallersForTesting(device.get()))
                {
                    submitQueuedInTime = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            if (submitQueuedInTime)
            {
                lateThread = std::thread([&] {
                    lateStatus = rachis_command_copy_buffer(
                        device.get(), lateCommand, upload, readback, 0, 0, byteCount, &lateError);
                });
                std::unique_lock lock(ctx.mutex);
                lateEnteredEarly =
                    ctx.cv.wait_for(lock, std::chrono::milliseconds(250), [&] { return ctx.lateEntered; });
            }
        }

        {
            const std::scoped_lock lock(ctx.mutex);
            ctx.releaseHeld = true;
        }
        ctx.cv.notify_all();
        heldThread.join();
        if (submitThread.joinable())
        {
            submitThread.join();
        }
        if (lateThread.joinable())
        {
            lateThread.join();
        }
    }

    REQUIRE(heldEnteredInTime);
    REQUIRE(submitQueuedInTime);
    CHECK_FALSE(lateEnteredEarly);
    CHECK(ctx.lateEntered);
    CHECK_FALSE(ctx.lateEnteredWhileHeld);
    CHECK(ctx.heldReentered);
    CHECK(ctx.reentrySawSubmitReturn);
    {
        INFO(heldError.message);
        CHECK(heldStatus == RACHIS_STATUS_OK);
    }
    {
        INFO(reentryError.message);
        CHECK(reentryStatus == RACHIS_STATUS_OK);
    }
    {
        INFO(submitError.message);
        REQUIRE(submitStatus == RACHIS_STATUS_OK);
    }
    {
        INFO(lateError.message);
        CHECK(lateStatus == RACHIS_STATUS_OK);
    }

    REQUIRE(timeline != 0);
    requireOk(rachis_timeline_wait(device.get(), timeline, &error), error);
    requireOk(rachis_timeline_release(device.get(), timeline, &error), error);
    std::array<std::uint8_t, byteCount> actual{};
    requireOk(rachis_buffer_read(device.get(), readback, 0, actual.data(), byteCount, &error), error);
    CHECK(actual == expected);

    requireOk(rachis_command_discard(device.get(), lateCommand, &error), error);
    requireOk(rachis_command_discard(device.get(), heldCommand, &error), error);
    requireOk(rachis_command_pool_destroy(device.get(), latePool, &error), error);
    requireOk(rachis_command_pool_destroy(device.get(), submittedPool, &error), error);
    requireOk(rachis_command_pool_destroy(device.get(), heldPool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
}

TEST_CASE("C ABI handle allocation saturates cleanly at UINT64_MAX without wraparound",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][tokens]")
{
    constexpr auto maxTag = rachis::cabi::detail::maxCAbiTokenTag;

    auto error = makeError();
    auto device = createDevice(error);
    const std::array<std::uint8_t, 4> bytes{4, 3, 2, 1};
    const auto upload =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    const auto pool = createPool(device.get(), RACHIS_QUEUE_COPY, error);
    rachis_handle preAcquired = 0;
    requireOk(rachis_command_acquire(device.get(), pool, &preAcquired, &error), error);
    requireOk(rachis_command_copy_buffer(device.get(), preAcquired, upload, readback, 0, 0, bytes.size(), &error),
              error);

    rachis_handle penultimateBuffer = 0;
    rachis_handle finalBuffer = 0;
    {
        rachis::detail::MonotonicIdentitySource nearExhaustion(maxTag - 1);
        const ScopedTokenSourceOverride tokenOverride(&nearExhaustion);

        rachis_buffer_desc bufferDesc{};
        bufferDesc.struct_size = sizeof(bufferDesc);
        bufferDesc.abi_version = RACHIS_ABI_VERSION;
        bufferDesc.size = 16;
        bufferDesc.usage = RACHIS_BUFFER_COPY_SOURCE;
        bufferDesc.memory_type = RACHIS_MEMORY_UPLOAD;

        requireOk(rachis_buffer_create(device.get(), &bufferDesc, &penultimateBuffer, &error), error);
        CHECK(penultimateBuffer != 0);
        CHECK(penultimateBuffer != RACHIS_WHOLE_SIZE);

        requireOk(rachis_buffer_create(device.get(), &bufferDesc, &finalBuffer, &error), error);
        CHECK(finalBuffer != 0);
        CHECK(finalBuffer != RACHIS_WHOLE_SIZE);

        rachis_handle exhaustedBuffer = 99;
        CHECK(rachis_buffer_create(device.get(), &bufferDesc, &exhaustedBuffer, &error) ==
              RACHIS_STATUS_RESOURCE_EXHAUSTED);
        CHECK(exhaustedBuffer == 0);
        CHECK(error.code == RACHIS_STATUS_RESOURCE_EXHAUSTED);

        exhaustedBuffer = 99;
        CHECK(rachis_buffer_create(device.get(), &bufferDesc, &exhaustedBuffer, &error) ==
              RACHIS_STATUS_RESOURCE_EXHAUSTED);
        CHECK(exhaustedBuffer == 0);

        rachis_handle exhaustedCommand = 99;
        CHECK(rachis_command_acquire(device.get(), pool, &exhaustedCommand, &error) ==
              RACHIS_STATUS_RESOURCE_EXHAUSTED);
        CHECK(exhaustedCommand == 0);

        rachis_handle exhaustedTimeline = 99;
        CHECK(rachis_command_submit(device.get(), preAcquired, &exhaustedTimeline, &error) ==
              RACHIS_STATUS_RESOURCE_EXHAUSTED);
        CHECK(exhaustedTimeline == 0);
        requireOk(rachis_buffer_write(device.get(), upload, 0, bytes.data(), bytes.size(), &error), error);
        requireOk(rachis_buffer_destroy(device.get(), finalBuffer, &error), error);
        requireOk(rachis_buffer_destroy(device.get(), penultimateBuffer, &error), error);

        exhaustedBuffer = 99;
        CHECK(rachis_buffer_create(device.get(), &bufferDesc, &exhaustedBuffer, &error) ==
              RACHIS_STATUS_RESOURCE_EXHAUSTED);
        CHECK(exhaustedBuffer == 0);
    }

    rachis_handle timeline = 0;
    requireOk(rachis_command_submit(device.get(), preAcquired, &timeline, &error), error);
    requireOk(rachis_timeline_wait(device.get(), timeline, &error), error);
    requireOk(rachis_timeline_release(device.get(), timeline, &error), error);

    std::array<std::uint8_t, 4> copied{};
    requireOk(rachis_buffer_read(device.get(), readback, 0, copied.data(), copied.size(), &error), error);
    CHECK(copied == bytes);

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
}

TEST_CASE("C ABI returns the reserved handle when buffer, command pool, or surface creation fails or throws",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][tokens][fault]")
{
    auto error = makeError();
    auto device = createDevice(error);
    const auto baseline = rachis::cabi::detail::liveTokenCountForTesting(device.get());

    rachis_buffer_desc bufferDesc{};
    bufferDesc.struct_size = sizeof(bufferDesc);
    bufferDesc.abi_version = RACHIS_ABI_VERSION;
    bufferDesc.size = 16;
    bufferDesc.usage = RACHIS_BUFFER_COPY_SOURCE;
    bufferDesc.memory_type = RACHIS_MEMORY_UPLOAD;

    rachis_command_pool_desc poolDesc{};
    poolDesc.struct_size = sizeof(poolDesc);
    poolDesc.abi_version = RACHIS_ABI_VERSION;
    poolDesc.queue_type = RACHIS_QUEUE_COPY;
    poolDesc.max_command_buffers = 8;

    rachis_surface_desc surfaceDesc{};
    surfaceDesc.struct_size = sizeof(surfaceDesc);
    surfaceDesc.abi_version = RACHIS_ABI_VERSION;
    surfaceDesc.kind = RACHIS_SURFACE_WIN32;

    for (const bool raise : {false, true})
    {
        INFO("raise " << raise);
        NativeCallFault fault{.call = rachis::cabi::detail::NativeCall::createObject, .raise = raise};
        {
            const ScopedNativeCallFault inject(fault);

            rachis_handle buffer = 99;
            error = makeError();
            CHECK(rachis_buffer_create(device.get(), &bufferDesc, &buffer, &error) == RACHIS_STATUS_NATIVE_FAILURE);
            CHECK(buffer == 0);
            CHECK(std::string_view{error.message} == injectedNativeFailure);
            CHECK(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline);

            rachis_handle pool = 99;
            error = makeError();
            CHECK(rachis_command_pool_create(device.get(), &poolDesc, &pool, &error) == RACHIS_STATUS_NATIVE_FAILURE);
            CHECK(pool == 0);
            CHECK(std::string_view{error.message} == injectedNativeFailure);
            CHECK(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline);

            rachis_handle surface = 99;
            error = makeError();
            CHECK(rachis_surface_create(device.get(), &surfaceDesc, &surface, &error) == RACHIS_STATUS_NATIVE_FAILURE);
            CHECK(surface == 0);
            CHECK(std::string_view{error.message} == injectedNativeFailure);
            CHECK(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline);
        }
        CHECK(fault.hits == 3);
    }

    error = makeError();
    const auto buffer = createBuffer(device.get(), 16, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto pool = createPool(device.get(), RACHIS_QUEUE_COPY, error);
    CHECK(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline + 2);
    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), buffer, &error), error);
    CHECK(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline);
}

TEST_CASE("C ABI passes whole diagnostic messages and cuts error messages at a UTF-8 character boundary",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][diagnostics][fault]")
{
    DiagnosticRecord record;
    rachis_device_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RACHIS_ABI_VERSION;
    desc.backend = RACHIS_SMOKE_BACKEND;
    desc.adapter_preference = RACHIS_SMOKE_ADAPTER;
    desc.validation_level = RACHIS_VALIDATION_DISABLED;
    desc.application_name = "Rachis C ABI diagnostic text";
    desc.diagnostic_callback = &recordDiagnostic;
    desc.diagnostic_user_data = &record;
    auto error = makeError();
    rachis_device* native = nullptr;
    requireOk(rachis_device_create(&desc, &native, &error), error);
    REQUIRE(native != nullptr);
    const Device device{native, &rachis_device_destroy};

    std::string longMessage;
    while (longMessage.size() < 3 * RACHIS_ERROR_MESSAGE_CAPACITY)
    {
        longMessage.append("VUID-\xE3\x83\x86\xE3\x82\xB9\xE3\x83\x88 ");
    }
    for (const auto& description : {std::string{"short diagnostic"}, longMessage})
    {
        CAPTURE(description.size());
        rachis::cabi::detail::emitDiagnosticForTesting(device.get(),
                                                       rachis::DiagnosticMessage{
                                                           .severity = rachis::DiagnosticSeverity::warning,
                                                           .nativeId = 42,
                                                           .description = description,
                                                       });
        const std::scoped_lock lock(record.mutex);
        REQUIRE_FALSE(record.entries.empty());
        CHECK(record.entries.back().severity == RACHIS_DIAGNOSTIC_WARNING);
        CHECK(record.entries.back().nativeId == 42);
        CHECK(record.entries.back().message == description);
    }

    constexpr std::size_t asciiPrefix = RACHIS_ERROR_MESSAGE_CAPACITY - 2;
    const auto splitMessage = std::string(asciiPrefix, 'a') + "\xC3\xA9" + "z";
    rachis_buffer_desc bufferDesc{};
    bufferDesc.struct_size = sizeof(bufferDesc);
    bufferDesc.abi_version = RACHIS_ABI_VERSION;
    bufferDesc.size = 16;
    bufferDesc.usage = RACHIS_BUFFER_COPY_SOURCE;
    bufferDesc.memory_type = RACHIS_MEMORY_UPLOAD;
    for (const bool raise : {false, true})
    {
        INFO("raise " << raise);
        NativeCallFault fault{
            .call = rachis::cabi::detail::NativeCall::createObject, .raise = raise, .message = splitMessage};
        const ScopedNativeCallFault inject(fault);
        rachis_handle buffer = 0;
        error = makeError();
        CHECK(rachis_buffer_create(device.get(), &bufferDesc, &buffer, &error) == RACHIS_STATUS_NATIVE_FAILURE);
        CHECK(buffer == 0);
        CHECK(std::string_view{error.message} == std::string_view{splitMessage}.substr(0, asciiPrefix));
        CHECK(fault.hits == 1);
    }
}

TEST_CASE("C ABI keeps the last view or lease of a destroyed texture for retry when the texture destroy fails",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][tokens][fault]")
{
    auto error = makeError();
    auto device = createDevice(error);
    const auto pool = createPool(device.get(), RACHIS_QUEUE_GRAPHICS, error);
    const auto baseline = rachis::cabi::detail::liveTokenCountForTesting(device.get());

    for (const bool raise : {false, true})
    {
        INFO("raise " << raise);
        error = makeError();
        const auto texture = createRenderTarget(device.get(), error);
        const auto view = createRenderTargetView(device.get(), texture, error);
        requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
        REQUIRE(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline + 2);

        NativeCallFault viewFault{.call = rachis::cabi::detail::NativeCall::destroyRetiredTexture, .raise = raise};
        {
            const ScopedNativeCallFault inject(viewFault);
            CHECK(rachis_texture_view_destroy(device.get(), view, &error) == RACHIS_STATUS_NATIVE_FAILURE);
            CHECK(std::string_view{error.message} == injectedNativeFailure);
        }
        CHECK(viewFault.hits == 1);
        CHECK(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline + 2);

        error = makeError();
        rachis_handle command = 0;
        requireOk(rachis_command_acquire(device.get(), pool, &command, &error), error);
        rachis_color_attachment attachment{};
        attachment.view = view;
        attachment.load_op = RACHIS_LOAD_OP_CLEAR;
        attachment.store_op = RACHIS_STORE_OP_STORE;
        rachis_rendering_desc rendering{};
        rendering.struct_size = sizeof(rendering);
        rendering.abi_version = RACHIS_ABI_VERSION;
        rendering.color_attachments = &attachment;
        rendering.color_attachment_count = 1;
        CHECK(rachis_command_begin_rendering(device.get(), command, &rendering, &error) ==
              RACHIS_STATUS_INVALID_ARGUMENT);
        error = makeError();
        requireOk(rachis_command_discard(device.get(), command, &error), error);

        requireOk(rachis_texture_view_destroy(device.get(), view, &error), error);
        CHECK(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline);
        CHECK(rachis_texture_view_destroy(device.get(), view, &error) == RACHIS_STATUS_INVALID_ARGUMENT);

        error = makeError();
        const auto leased = createRenderTarget(device.get(), error);
        rachis_handle lease = 0;
        requireOk(rachis_texture_lease_acquire(device.get(), leased, &lease, &error), error);
        REQUIRE(lease != 0);
        requireOk(rachis_texture_destroy(device.get(), leased, &error), error);
        REQUIRE(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline + 2);

        NativeCallFault leaseFault{.call = rachis::cabi::detail::NativeCall::destroyRetiredTexture, .raise = raise};
        {
            const ScopedNativeCallFault inject(leaseFault);
            CHECK(rachis_texture_lease_release(device.get(), lease, &error) == RACHIS_STATUS_NATIVE_FAILURE);
            CHECK(std::string_view{error.message} == injectedNativeFailure);
        }
        CHECK(leaseFault.hits == 1);
        CHECK(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline + 2);

        error = makeError();
        rachis_buffer_texture_region region{};
        region.aspect = RACHIS_ASPECT_COLOR;
        region.width = 4;
        region.height = 4;
        region.depth = 1;
        rachis_texture_transfer_layout layout{};
        layout.struct_size = sizeof(layout);
        requireOk(rachis_texture_query_transfer_layout(device.get(), lease, &region, &layout, &error), error);

        requireOk(rachis_texture_lease_release(device.get(), lease, &error), error);
        CHECK(rachis::cabi::detail::liveTokenCountForTesting(device.get()) == baseline);
        CHECK(rachis_texture_lease_release(device.get(), lease, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    }

    error = makeError();
    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
}

#if RACHIS_CABI_FAULT_BACKEND_METAL || RACHIS_CABI_FAULT_BACKEND_VULKAN
TEST_CASE("C ABI retires command handle when rachis_command_submit fails after consuming recording",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][submit][fault]")
{
    auto error = makeError();
    auto device = createDevice(error);
    const std::array<std::uint8_t, 4> bytes{7, 6, 5, 4};
    const auto upload =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    requireOk(rachis_buffer_write(device.get(), upload, 0, bytes.data(), bytes.size(), &error), error);
    const auto pool = createPool(device.get(), RACHIS_QUEUE_COPY, error);

    rachis_handle command = 0;
    requireOk(rachis_command_acquire(device.get(), pool, &command, &error), error);
    requireOk(rachis_command_copy_buffer(device.get(), command, upload, readback, 0, 0, bytes.size(), &error), error);

    rachis_handle timeline = 99;
    {
#if RACHIS_CABI_FAULT_BACKEND_METAL
        const ScopedMetalSubmitFailure injectFailure;
#elif RACHIS_CABI_FAULT_BACKEND_VULKAN
        const ScopedVulkanSubmitFailure injectFailure(-1);
#endif
        const auto submitStatus = rachis_command_submit(device.get(), command, &timeline, &error);
        CHECK(submitStatus != RACHIS_STATUS_OK);
        CHECK(timeline == 0);
    }

    CHECK(rachis_command_copy_buffer(device.get(), command, upload, readback, 0, 0, bytes.size(), &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);

    rachis_global_barrier global{};
    global.stages_before = RACHIS_STAGE_COPY;
    global.access_before = RACHIS_ACCESS_COPY_DESTINATION;
    global.stages_after = RACHIS_STAGE_COPY;
    global.access_after = RACHIS_ACCESS_COPY_SOURCE;
    rachis_barrier_batch batch{};
    batch.struct_size = sizeof(batch);
    batch.abi_version = RACHIS_ABI_VERSION;
    batch.global_barriers = &global;
    batch.global_barrier_count = 1;
    CHECK(rachis_command_barriers(device.get(), command, &batch, &error) == RACHIS_STATUS_INVALID_ARGUMENT);

    rachis_handle retryTimeline = 99;
    CHECK(rachis_command_submit(device.get(), command, &retryTimeline, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(retryTimeline == 0);
    CHECK(rachis_command_discard(device.get(), command, &error) == RACHIS_STATUS_INVALID_ARGUMENT);

    rachis_handle freshCommand = 0;
    requireOk(rachis_command_acquire(device.get(), pool, &freshCommand, &error), error);
    requireOk(rachis_command_copy_buffer(device.get(), freshCommand, upload, readback, 0, 0, bytes.size(), &error),
              error);
    rachis_handle freshTimeline = 0;
    requireOk(rachis_command_submit(device.get(), freshCommand, &freshTimeline, &error), error);
    requireOk(rachis_timeline_wait(device.get(), freshTimeline, &error), error);
    requireOk(rachis_timeline_release(device.get(), freshTimeline, &error), error);

    std::array<std::uint8_t, 4> copied{};
    requireOk(rachis_buffer_read(device.get(), readback, 0, copied.data(), copied.size(), &error), error);
    CHECK(copied == bytes);

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
}

TEST_CASE("C ABI host buffer writes, reads and mappings report device loss before handle and range errors",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][loss][fault]")
{
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_DISABLED);
    const std::array<std::uint8_t, 4> bytes{9, 8, 7, 6};
    const auto upload =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    const auto stale = createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    requireOk(rachis_buffer_destroy(device.get(), stale, &error), error);
    std::array<std::uint8_t, 4> copied{};
    requireOk(rachis_buffer_write(device.get(), upload, 0, bytes.data(), bytes.size(), &error), error);
    requireOk(rachis_buffer_read(device.get(), readback, 0, copied.data(), copied.size(), &error), error);
    const auto mappedUpload =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto mappedReadback =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    rachis_mapped_memory mapped{};
    mapped.struct_size = sizeof(mapped);
    requireOk(rachis_buffer_map(device.get(), mappedUpload, 0, RACHIS_WHOLE_SIZE, &mapped, &error), error);
    REQUIRE(mapped.data != nullptr);
    rachis_mapped_memory mappedRead{};
    mappedRead.struct_size = sizeof(mappedRead);
    requireOk(rachis_buffer_map(device.get(), mappedReadback, 0, RACHIS_WHOLE_SIZE, &mappedRead, &error), error);

#if RACHIS_CABI_FAULT_BACKEND_METAL
    const auto injected = rachis::metal::injectMetalDeviceLoss();
#elif RACHIS_CABI_FAULT_BACKEND_VULKAN
    const auto injected = rachis::vulkan::injectVulkanDeviceLoss();
#endif
    REQUIRE_FALSE(injected);
    CHECK(injected.error().code == rachis::ErrorCode::deviceLost);
    std::uint32_t operationalState = RACHIS_DEVICE_INVALID;
    requireOk(rachis_device_operational_state(device.get(), &operationalState, &error), error);
    REQUIRE(operationalState == RACHIS_DEVICE_LOST);

    error = makeError();
    CHECK(rachis_buffer_write(device.get(), upload, 0, bytes.data(), bytes.size(), &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(error.code == RACHIS_STATUS_DEVICE_LOST);
    CHECK(std::string{error.message} == injected.error().message);
    error = makeError();
    CHECK(rachis_buffer_read(device.get(), readback, 0, copied.data(), copied.size(), &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(error.code == RACHIS_STATUS_DEVICE_LOST);
    CHECK(std::string{error.message} == injected.error().message);
    CHECK(rachis_buffer_write(device.get(), stale, 0, bytes.data(), bytes.size(), &error) == RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_read(device.get(), stale, 0, copied.data(), copied.size(), &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_write(device.get(), upload, bytes.size(), bytes.data(), bytes.size(), &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_read(device.get(), readback, bytes.size(), copied.data(), copied.size(), &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_read(device.get(), readback, 0, nullptr, copied.size(), &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);

    rachis_mapped_memory rejected{};
    rejected.struct_size = sizeof(rejected);
    error = makeError();
    CHECK(rachis_buffer_map(device.get(), upload, 0, RACHIS_WHOLE_SIZE, &rejected, &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(error.code == RACHIS_STATUS_DEVICE_LOST);
    CHECK(std::string{error.message} == injected.error().message);
    CHECK(rejected.data == nullptr);
    CHECK(rachis_buffer_map(device.get(), stale, 0, RACHIS_WHOLE_SIZE, &rejected, &error) == RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_map(device.get(), upload, bytes.size(), 1, &rejected, &error) == RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_map(device.get(), upload, 0, RACHIS_WHOLE_SIZE, nullptr, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);

    const rachis_mapped_range wholeRange{0, RACHIS_WHOLE_SIZE};
    const rachis_mapped_range outsideRange{bytes.size(), 1};
    error = makeError();
    CHECK(rachis_buffer_flush_mapped_ranges(device.get(), mappedUpload, &wholeRange, 1, &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(error.code == RACHIS_STATUS_DEVICE_LOST);
    CHECK(std::string{error.message} == injected.error().message);
    error = makeError();
    CHECK(rachis_buffer_invalidate_mapped_ranges(device.get(), mappedReadback, &wholeRange, 1, &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(error.code == RACHIS_STATUS_DEVICE_LOST);
    CHECK(std::string{error.message} == injected.error().message);
    CHECK(rachis_buffer_flush_mapped_ranges(device.get(), stale, &wholeRange, 1, &error) == RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_invalidate_mapped_ranges(device.get(), stale, &wholeRange, 1, &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_flush_mapped_ranges(device.get(), mappedUpload, &outsideRange, 1, &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_invalidate_mapped_ranges(device.get(), mappedReadback, &outsideRange, 1, &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_flush_mapped_ranges(device.get(), mappedUpload, nullptr, 1, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);

    std::memcpy(mapped.data, bytes.data(), bytes.size());
    CHECK(std::memcmp(mapped.data, bytes.data(), bytes.size()) == 0);
    CHECK(rachis_buffer_destroy(device.get(), mappedUpload, &error) == RACHIS_STATUS_INVALID_STATE);
    requireOk(rachis_buffer_unmap(device.get(), mappedUpload, &error), error);
    requireOk(rachis_buffer_unmap(device.get(), mappedReadback, &error), error);
    CHECK(rachis_buffer_unmap(device.get(), stale, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_buffer_destroy(device.get(), mappedUpload, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), mappedReadback, &error), error);
}
#endif

#if RACHIS_CABI_FAULT_BACKEND_VULKAN && defined(RACHIS_CABI_COMPUTE_SHADER_PATH)
TEST_CASE("C ABI reports Vulkan compute pipeline out-of-memory as RACHIS_STATUS_OUT_OF_MEMORY",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][pipeline][fault]")
{
    constexpr std::int32_t vulkanOutOfHostMemory = -1;
    constexpr std::int32_t vulkanOutOfDeviceMemory = -2;
    std::ifstream input{RACHIS_CABI_COMPUTE_SHADER_PATH, std::ios::binary};
    const std::vector<char> bytecode{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    REQUIRE_FALSE(bytecode.empty());
    auto error = makeError();
    auto device = createDevice(error);

    const rachis_shader_binding_requirement requirement{0, 0, RACHIS_BINDING_STORAGE_BUFFER, 1};
    rachis_shader_module_desc shaderDesc{};
    shaderDesc.struct_size = sizeof(shaderDesc);
    shaderDesc.abi_version = RACHIS_ABI_VERSION;
    shaderDesc.stage = RACHIS_SHADER_COMPUTE;
    shaderDesc.bytecode = bytecode.data();
    shaderDesc.bytecode_size = bytecode.size();
    shaderDesc.entry_point = "main";
    shaderDesc.bindings = &requirement;
    shaderDesc.binding_count = 1;
    shaderDesc.workgroup_x = 64;
    shaderDesc.workgroup_y = 1;
    shaderDesc.workgroup_z = 1;
    rachis_handle shader = 0;
    requireOk(rachis_shader_module_create(device.get(), &shaderDesc, &shader, &error), error);

    const rachis_binding_layout_entry entry{0, RACHIS_BINDING_STORAGE_BUFFER, RACHIS_SHADER_COMPUTE, 1};
    rachis_binding_layout_desc bindingLayoutDesc{};
    bindingLayoutDesc.struct_size = sizeof(bindingLayoutDesc);
    bindingLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    bindingLayoutDesc.entries = &entry;
    bindingLayoutDesc.entry_count = 1;
    rachis_handle bindingLayout = 0;
    requireOk(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &bindingLayout, &error), error);

    rachis_pipeline_layout_desc pipelineLayoutDesc{};
    pipelineLayoutDesc.struct_size = sizeof(pipelineLayoutDesc);
    pipelineLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineLayoutDesc.binding_layouts = &bindingLayout;
    pipelineLayoutDesc.binding_layout_count = 1;
    rachis_handle pipelineLayout = 0;
    requireOk(rachis_pipeline_layout_create(device.get(), &pipelineLayoutDesc, &pipelineLayout, &error), error);

    rachis_compute_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.shader = shader;

    for (const auto vkResult : std::array{vulkanOutOfDeviceMemory, vulkanOutOfHostMemory})
    {
        CAPTURE(vkResult);
        REQUIRE(rachis::vulkan::test::injectCreateResult(vkResult));
        rachis_handle pipeline = 99;
        auto failure = makeError();
        const auto status = rachis_compute_pipeline_create(device.get(), &pipelineDesc, &pipeline, &failure);
        REQUIRE(rachis::vulkan::test::injectCreateResult(0));
        CHECK(status == RACHIS_STATUS_OUT_OF_MEMORY);
        CHECK(failure.code == RACHIS_STATUS_OUT_OF_MEMORY);
        CHECK(failure.native_code == vkResult);
        CHECK(pipeline == 0);
    }

    std::uint32_t operationalState = 0;
    requireOk(rachis_device_operational_state(device.get(), &operationalState, &error), error);
    CHECK(operationalState == RACHIS_DEVICE_ACTIVE);

    rachis_handle pipeline = 0;
    requireOk(rachis_compute_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error), error);
    CHECK(pipeline != 0);

    requireOk(rachis_pipeline_destroy(device.get(), pipeline, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_binding_layout_destroy(device.get(), bindingLayout, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), shader, &error), error);
}
#endif

#if RACHIS_CABI_FAULT_BACKEND_METAL
TEST_CASE("C ABI keeps Metal allocation telemetry bounded when resources churn without garbage collection",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][budget]")
{
    constexpr std::uint64_t bufferSize = 256;
    constexpr std::uint32_t bufferCount = 10'000;
    constexpr std::uint32_t textureCount = 1'000;
    constexpr std::size_t maxTelemetryEntries = 256;

    auto error = makeError();
    auto device = createDevice(error);
    const auto live =
        createBuffer(device.get(), bufferSize, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_DEVICE_LOCAL, error);

    std::size_t peakCount = 0;
    const auto observe = [&peakCount] {
        const auto count = rachis::metal::boundAllocationTelemetryCount();
        REQUIRE(count.has_value());
        peakCount = std::max(peakCount, *count);
    };
    for (std::uint32_t index = 0; index < bufferCount; ++index)
    {
        const auto buffer =
            createBuffer(device.get(), bufferSize, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_DEVICE_LOCAL, error);
        requireOk(rachis_buffer_destroy(device.get(), buffer, &error), error);
        observe();
    }

    rachis_texture_desc textureDesc{};
    textureDesc.struct_size = sizeof(textureDesc);
    textureDesc.abi_version = RACHIS_ABI_VERSION;
    textureDesc.dimension = RACHIS_TEXTURE_2D;
    textureDesc.format = RACHIS_FORMAT_RGBA8_UNORM;
    textureDesc.usage = RACHIS_TEXTURE_SAMPLED;
    textureDesc.width = 4;
    textureDesc.height = 4;
    textureDesc.depth_or_array_layers = 1;
    textureDesc.mip_levels = 1;
    textureDesc.sample_count = 1;
    for (std::uint32_t index = 0; index < textureCount; ++index)
    {
        rachis_handle texture = 0;
        requireOk(rachis_texture_create(device.get(), &textureDesc, &texture, &error), error);
        REQUIRE(texture != 0);
        requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
        observe();
    }
    CHECK(peakCount < maxTelemetryEntries);

    requireOk(rachis_buffer_destroy(device.get(), live, &error), error);
}
#endif

#if RACHIS_CABI_FAULT_BACKEND_VULKAN
TEST_CASE("C ABI one-entry command pools reacquire after a transient rachis_command_submit failure",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][submit][fault]")
{
    auto error = makeError();
    auto device = createDevice(error);
    const auto upload = createBuffer(device.get(), 4, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback = createBuffer(device.get(), 4, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);

    rachis_command_pool_desc poolDesc{};
    poolDesc.struct_size = sizeof(poolDesc);
    poolDesc.abi_version = RACHIS_ABI_VERSION;
    poolDesc.queue_type = RACHIS_QUEUE_GRAPHICS;
    poolDesc.max_command_buffers = 1;
    rachis_handle pool = 0;
    requireOk(rachis_command_pool_create(device.get(), &poolDesc, &pool, &error), error);
    REQUIRE(pool != 0);

    for (std::uint8_t round = 0; round < 2; ++round)
    {
        CAPTURE(round);
        const std::array<std::uint8_t, 4> bytes{
            static_cast<std::uint8_t>(round + 1U), 9, 8, static_cast<std::uint8_t>(round + 7U)};
        requireOk(rachis_buffer_write(device.get(), upload, 0, bytes.data(), bytes.size(), &error), error);

        rachis_handle failing = 0;
        requireOk(rachis_command_acquire(device.get(), pool, &failing, &error), error);
        requireOk(rachis_command_copy_buffer(device.get(), failing, upload, readback, 0, 0, bytes.size(), &error),
                  error);
        rachis_handle failedTimeline = 99;
        {
            const ScopedVulkanSubmitFailure injectFailure(-1);
            CHECK(rachis_command_submit(device.get(), failing, &failedTimeline, &error) == RACHIS_STATUS_OUT_OF_MEMORY);
        }
        CHECK(failedTimeline == 0);
        std::uint32_t state = RACHIS_DEVICE_INVALID;
        requireOk(rachis_device_operational_state(device.get(), &state, &error), error);
        REQUIRE(state == RACHIS_DEVICE_ACTIVE);

        rachis_handle command = 0;
        requireOk(rachis_command_acquire(device.get(), pool, &command, &error), error);
        requireOk(rachis_command_copy_buffer(device.get(), command, upload, readback, 0, 0, bytes.size(), &error),
                  error);
        rachis_handle timeline = 0;
        requireOk(rachis_command_submit(device.get(), command, &timeline, &error), error);
        requireOk(rachis_timeline_wait(device.get(), timeline, &error), error);
        requireOk(rachis_timeline_release(device.get(), timeline, &error), error);

        std::array<std::uint8_t, 4> copied{};
        requireOk(rachis_buffer_read(device.get(), readback, 0, copied.data(), copied.size(), &error), error);
        CHECK(copied == bytes);
    }

    rachis_diagnostic_counts counts{};
    counts.struct_size = sizeof(counts);
    requireOk(rachis_device_diagnostic_counts(device.get(), &counts, &error), error);
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
}
#endif

TEST_CASE("C ABI destroy and release calls succeed after device loss and retire their handles",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][loss][fault]")
{
    const auto trigger = GENERATE(Catch::Generators::from_range(lossTriggers()));
    INFO("loss trigger: " << lossTriggerName(trigger));
    auto error = makeError();
    auto device = createDevice(error, lossValidation);

    const auto storage = createBuffer(device.get(), 256, RACHIS_BUFFER_STORAGE, RACHIS_MEMORY_DEVICE_LOCAL, error);
    const auto upload = createBuffer(device.get(), 256, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto mappedUpload = createBuffer(device.get(), 256, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    rachis_mapped_memory mapped{};
    mapped.struct_size = sizeof(mapped);
    requireOk(rachis_buffer_map(device.get(), mappedUpload, 64, 128, &mapped, &error), error);
    REQUIRE(mapped.data != nullptr);
    REQUIRE(mapped.size == 128);
    const auto texture = createSampledTexture(device.get(), error);
    const auto viewedTexture = createSampledTexture(device.get(), error);
    const auto view = createSampledView(device.get(), viewedTexture, error);
    requireOk(rachis_texture_destroy(device.get(), viewedTexture, &error), error);
    const auto leasedTexture = createSampledTexture(device.get(), error);
    rachis_handle lease = 0;
    requireOk(rachis_texture_lease_acquire(device.get(), leasedTexture, &lease, &error), error);
    requireOk(rachis_texture_destroy(device.get(), leasedTexture, &error), error);

    rachis_sampler_desc samplerDesc{};
    samplerDesc.struct_size = sizeof(samplerDesc);
    samplerDesc.abi_version = RACHIS_ABI_VERSION;
    rachis_handle sampler = 0;
    requireOk(rachis_sampler_create(device.get(), &samplerDesc, &sampler, &error), error);

    const rachis_binding_layout_entry entry{0, RACHIS_BINDING_STORAGE_BUFFER, RACHIS_SHADER_COMPUTE, 1};
    rachis_binding_layout_desc layoutDesc{};
    layoutDesc.struct_size = sizeof(layoutDesc);
    layoutDesc.abi_version = RACHIS_ABI_VERSION;
    layoutDesc.entries = &entry;
    layoutDesc.entry_count = 1;
    rachis_handle bindingLayout = 0;
    requireOk(rachis_binding_layout_create(device.get(), &layoutDesc, &bindingLayout, &error), error);
    const rachis_binding_pool_size poolSize{RACHIS_BINDING_STORAGE_BUFFER, 1};
    rachis_binding_pool_desc bindingPoolDesc{};
    bindingPoolDesc.struct_size = sizeof(bindingPoolDesc);
    bindingPoolDesc.abi_version = RACHIS_ABI_VERSION;
    bindingPoolDesc.max_binding_sets = 1;
    bindingPoolDesc.sizes = &poolSize;
    bindingPoolDesc.size_count = 1;
    rachis_handle bindingPool = 0;
    requireOk(rachis_binding_pool_create(device.get(), &bindingPoolDesc, &bindingPool, &error), error);
    rachis_buffer_binding write{};
    write.buffer = storage;
    write.size = 256;
    write.structure_byte_stride = sizeof(std::uint32_t);
    rachis_binding_set_desc setDesc{};
    setDesc.struct_size = sizeof(setDesc);
    setDesc.abi_version = RACHIS_ABI_VERSION;
    setDesc.layout = bindingLayout;
    setDesc.pool = bindingPool;
    setDesc.writes = &write;
    setDesc.write_count = 1;
    rachis_handle bindingSet = 0;
    requireOk(rachis_binding_set_create(device.get(), &setDesc, &bindingSet, &error), error);
    rachis_pipeline_layout_desc pipelineLayoutDesc{};
    pipelineLayoutDesc.struct_size = sizeof(pipelineLayoutDesc);
    pipelineLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineLayoutDesc.binding_layouts = &bindingLayout;
    pipelineLayoutDesc.binding_layout_count = 1;
    rachis_handle pipelineLayout = 0;
    requireOk(rachis_pipeline_layout_create(device.get(), &pipelineLayoutDesc, &pipelineLayout, &error), error);

#ifdef RACHIS_CABI_COMPUTE_SHADER_PATH
    std::ifstream shaderFile{RACHIS_CABI_COMPUTE_SHADER_PATH, std::ios::binary};
    const std::vector<char> bytecode{std::istreambuf_iterator<char>{shaderFile}, std::istreambuf_iterator<char>{}};
    REQUIRE_FALSE(bytecode.empty());
    const rachis_shader_binding_requirement requirement{0, 0, RACHIS_BINDING_STORAGE_BUFFER, 1};
    rachis_shader_module_desc shaderDesc{};
    shaderDesc.struct_size = sizeof(shaderDesc);
    shaderDesc.abi_version = RACHIS_ABI_VERSION;
    shaderDesc.stage = RACHIS_SHADER_COMPUTE;
    shaderDesc.bytecode = bytecode.data();
    shaderDesc.bytecode_size = bytecode.size();
    shaderDesc.entry_point = "main";
    shaderDesc.bindings = &requirement;
    shaderDesc.binding_count = 1;
    shaderDesc.workgroup_x = 64;
    shaderDesc.workgroup_y = 1;
    shaderDesc.workgroup_z = 1;
    rachis_handle shader = 0;
    requireOk(rachis_shader_module_create(device.get(), &shaderDesc, &shader, &error), error);
    rachis_compute_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.shader = shader;
    rachis_handle pipeline = 0;
    requireOk(rachis_compute_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error), error);
#endif

    const auto graphicsPool = createPool(device.get(), RACHIS_QUEUE_GRAPHICS, error);
    const auto copyPool = createPool(device.get(), RACHIS_QUEUE_COPY, error);
    rachis_timestamp_pool_desc timestampDesc{};
    timestampDesc.struct_size = sizeof(timestampDesc);
    timestampDesc.abi_version = RACHIS_ABI_VERSION;
    timestampDesc.queue_type = RACHIS_QUEUE_GRAPHICS;
    timestampDesc.query_count = 2;
    rachis_handle timestamps = 0;
    requireOk(rachis_timestamp_pool_create(device.get(), &timestampDesc, &timestamps, &error), error);

    rachis_handle submitted = 0;
    requireOk(rachis_command_acquire(device.get(), copyPool, &submitted, &error), error);
    rachis_handle timeline = 0;
    requireOk(rachis_command_submit(device.get(), submitted, &timeline, &error), error);
    requireOk(rachis_timeline_wait(device.get(), timeline, &error), error);
    rachis_handle unsubmitted = 0;
    requireOk(rachis_command_acquire(device.get(), graphicsPool, &unsubmitted, &error), error);

    loseDevice(device.get(), copyPool, trigger);

    rachis_buffer_desc rejectedDesc{};
    rejectedDesc.struct_size = sizeof(rejectedDesc);
    rejectedDesc.abi_version = RACHIS_ABI_VERSION;
    rejectedDesc.size = 16;
    rejectedDesc.usage = RACHIS_BUFFER_COPY_SOURCE;
    rejectedDesc.memory_type = RACHIS_MEMORY_UPLOAD;
    rachis_handle rejected = 0;
    CHECK(rachis_buffer_create(device.get(), &rejectedDesc, &rejected, &error) == RACHIS_STATUS_DEVICE_LOST);
    CHECK(rejected == 0);

    const auto released = [&error](const char* call, rachis_status status) {
        INFO(call << ": " << error.message);
        CHECK(status == RACHIS_STATUS_OK);
    };
    released("rachis_command_discard", rachis_command_discard(device.get(), unsubmitted, &error));
    released("rachis_timeline_release", rachis_timeline_release(device.get(), timeline, &error));
#ifdef RACHIS_CABI_COMPUTE_SHADER_PATH
    released("rachis_pipeline_destroy", rachis_pipeline_destroy(device.get(), pipeline, &error));
    released("rachis_shader_module_destroy", rachis_shader_module_destroy(device.get(), shader, &error));
#endif
    released("rachis_binding_set_destroy", rachis_binding_set_destroy(device.get(), bindingSet, &error));
    released("rachis_binding_pool_destroy", rachis_binding_pool_destroy(device.get(), bindingPool, &error));
    released("rachis_pipeline_layout_destroy", rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error));
    released("rachis_binding_layout_destroy", rachis_binding_layout_destroy(device.get(), bindingLayout, &error));
    released("rachis_texture_view_destroy", rachis_texture_view_destroy(device.get(), view, &error));
    released("rachis_texture_destroy", rachis_texture_destroy(device.get(), texture, &error));
    released("rachis_texture_lease_release", rachis_texture_lease_release(device.get(), lease, &error));
    released("rachis_sampler_destroy", rachis_sampler_destroy(device.get(), sampler, &error));
    released("rachis_query_pool_destroy", rachis_query_pool_destroy(device.get(), timestamps, &error));
    released("rachis_command_pool_destroy(graphics)", rachis_command_pool_destroy(device.get(), graphicsPool, &error));
    released("rachis_command_pool_destroy(copy)", rachis_command_pool_destroy(device.get(), copyPool, &error));
    released("rachis_buffer_destroy(upload)", rachis_buffer_destroy(device.get(), upload, &error));
    std::memset(mapped.data, 0x5A, static_cast<std::size_t>(mapped.size));
    CHECK(static_cast<const std::uint8_t*>(mapped.data)[mapped.size - 1] == 0x5A);
    CHECK(rachis_buffer_destroy(device.get(), mappedUpload, &error) == RACHIS_STATUS_INVALID_STATE);
    released("rachis_buffer_unmap", rachis_buffer_unmap(device.get(), mappedUpload, &error));
    released("rachis_buffer_destroy(mapped upload)", rachis_buffer_destroy(device.get(), mappedUpload, &error));
    released("rachis_buffer_destroy(storage)", rachis_buffer_destroy(device.get(), storage, &error));

    CHECK(rachis_buffer_destroy(device.get(), storage, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_view_destroy(device.get(), view, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_lease_release(device.get(), lease, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_binding_set_destroy(device.get(), bindingSet, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_pool_destroy(device.get(), copyPool, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
}

#if RACHIS_CABI_FAULT_HAS_SURFACE_HOST
TEST_CASE("C ABI destroys a swapchain with an acquired image and its surface after device loss",
          "[size:large]" RACHIS_CABI_PRESENT_DESKTOP_REQUIRES RACHIS_CABI_DEVICE_REQUIRES
          "[rachis][cabi][present][loss][fault]")
{
    const auto trigger = GENERATE(Catch::Generators::from_range(lossTriggers()));
    INFO("loss trigger: " << lossTriggerName(trigger));
    const SurfaceHost host;
    REQUIRE(host.valid());
    auto error = makeError();
    auto device = createDevice(error, lossValidation);
    const auto surfaceDesc = host.surfaceDesc();
    rachis_handle surface = 0;
    requireOk(rachis_surface_create(device.get(), &surfaceDesc, &surface, &error), error);
    const auto swapchain = createSwapchain(device.get(), surface, error);
    const auto copyPool = createPool(device.get(), RACHIS_QUEUE_COPY, error);
    const auto image = acquireSwapchainImage(device.get(), swapchain, error);

    loseDevice(device.get(), copyPool, trigger);

    const auto released = [&error](const char* call, rachis_status status) {
        INFO(call << ": " << error.message);
        CHECK(status == RACHIS_STATUS_OK);
    };
    released("rachis_swapchain_destroy", rachis_swapchain_destroy(device.get(), swapchain, &error));
    CHECK(queryBorrowedTextureLayout(device.get(), image.texture, error) == RACHIS_STATUS_INVALID_ARGUMENT);
    released("rachis_surface_destroy", rachis_surface_destroy(device.get(), surface, &error));
    released("rachis_command_pool_destroy", rachis_command_pool_destroy(device.get(), copyPool, &error));
    CHECK(rachis_swapchain_destroy(device.get(), swapchain, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_surface_destroy(device.get(), surface, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
}
#endif

#if RACHIS_CABI_FAULT_BACKEND_METAL || RACHIS_CABI_FAULT_BACKEND_D3D12
TEST_CASE("C ABI keeps a surface while its swapchain is alive and zeroes unused capability formats",
          "[size:large]" RACHIS_CABI_PRESENT_DESKTOP_REQUIRES RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][swapchain]")
{
    const SurfaceHost host;
    REQUIRE(host.valid());
    auto error = makeError();
    auto device = createDevice(error);
    const auto surfaceDesc = host.surfaceDesc();
    rachis_handle surface = 0;
    requireOk(rachis_surface_create(device.get(), &surfaceDesc, &surface, &error), error);

    rachis_surface_capabilities capabilities;
    std::memset(&capabilities, 0xA5, sizeof(capabilities));
    capabilities.struct_size = sizeof(capabilities);
    requireOk(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error), error);
    REQUIRE(capabilities.format_count > 0);
    REQUIRE(capabilities.format_count <= RACHIS_MAX_SURFACE_FORMATS);
    for (auto index = capabilities.format_count; index < RACHIS_MAX_SURFACE_FORMATS; ++index)
    {
        CAPTURE(index);
        CHECK(capabilities.formats[index].format == RACHIS_FORMAT_UNKNOWN);
        CHECK(capabilities.formats[index].color_space == 0U);
    }

    const auto swapchain = createSwapchain(device.get(), surface, error);
    error = makeError();
    CHECK(rachis_surface_destroy(device.get(), surface, &error) == RACHIS_STATUS_INVALID_STATE);
    CHECK(error.code == RACHIS_STATUS_INVALID_STATE);
    requireOk(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error), error);

    requireOk(rachis_swapchain_destroy(device.get(), swapchain, &error), error);
    requireOk(rachis_surface_destroy(device.get(), surface, &error), error);
    CHECK(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
}

TEST_CASE("C ABI reconciles borrowed swapchain handles across pre-consume and post-consume present failures",
          "[size:large]" RACHIS_CABI_PRESENT_DESKTOP_REQUIRES RACHIS_CABI_DEVICE_REQUIRES
          "[rachis][cabi][present][fault]")
{
    {
        const SurfaceHost host;
        const SurfaceHost siblingHost;
        REQUIRE(host.valid());
        REQUIRE(siblingHost.valid());
        auto error = makeError();
        auto device = createDevice(error);
        const auto surfaceDesc = host.surfaceDesc();
        const auto siblingSurfaceDesc = siblingHost.surfaceDesc();
        rachis_handle surface = 0;
        rachis_handle siblingSurface = 0;
        requireOk(rachis_surface_create(device.get(), &surfaceDesc, &surface, &error), error);
        requireOk(rachis_surface_create(device.get(), &siblingSurfaceDesc, &siblingSurface, &error), error);
        auto swapchain = createSwapchain(device.get(), surface, error);
        const auto siblingSwapchain = createSwapchain(device.get(), siblingSurface, error);
        const auto graphicsPool = createPool(device.get(), RACHIS_QUEUE_GRAPHICS, error);
        const auto copyPool = createPool(device.get(), RACHIS_QUEUE_COPY, error);

        std::vector<rachis_handle> liveBuffers;
        liveBuffers.reserve(64);
        for (int i = 0; i < 64; ++i)
        {
            liveBuffers.push_back(
                createBuffer(device.get(), 16, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error));
        }

        {
            const auto siblingImage = acquireSwapchainImage(device.get(), siblingSwapchain, error);
            const auto image = acquireSwapchainImage(device.get(), swapchain, error);
            rachis_handle copyCommand = 0;
            requireOk(rachis_command_acquire(device.get(), copyPool, &copyCommand, &error), error);
            rachis_handle copyTimeline = 0;
            requireOk(rachis_command_submit(device.get(), copyCommand, &copyTimeline, &error), error);

            std::uint32_t presentStatus = RACHIS_PRESENT_STATUS_PRESENTED;
            {
#if RACHIS_CABI_FAULT_BACKEND_METAL
                const ScopedMetalPresentFailure injectPreConsume(
                    rachis::metal::test::InjectedPresentFailure::beforeConsumeValidationError);
                CHECK(rachis_swapchain_present(device.get(), swapchain, copyTimeline, &presentStatus, &error) ==
                      RACHIS_STATUS_VALIDATION_ERROR);
#elif RACHIS_CABI_FAULT_BACKEND_D3D12
                CHECK(rachis_swapchain_present(device.get(), swapchain, copyTimeline, &presentStatus, &error) ==
                      RACHIS_STATUS_INVALID_ARGUMENT);
#endif
            }

            requireOk(queryBorrowedTextureLayout(device.get(), image.texture, error), error);
            requireOk(probeBorrowedViewRendering(device.get(), graphicsPool, image.render_target_view, error), error);

            requireOk(rachis_swapchain_cancel(device.get(), swapchain, &error), error);
            CHECK(queryBorrowedTextureLayout(device.get(), image.texture, error) == RACHIS_STATUS_INVALID_ARGUMENT);
            CHECK(probeBorrowedViewRendering(device.get(), graphicsPool, image.render_target_view, error) ==
                  RACHIS_STATUS_INVALID_ARGUMENT);
            requireOk(queryBorrowedTextureLayout(device.get(), siblingImage.texture, error), error);
            requireOk(probeBorrowedViewRendering(device.get(), graphicsPool, siblingImage.render_target_view, error),
                      error);
            requireOk(rachis_swapchain_cancel(device.get(), siblingSwapchain, &error), error);
            CHECK(queryBorrowedTextureLayout(device.get(), siblingImage.texture, error) ==
                  RACHIS_STATUS_INVALID_ARGUMENT);

            requireOk(rachis_timeline_wait(device.get(), copyTimeline, &error), error);
            requireOk(rachis_timeline_release(device.get(), copyTimeline, &error), error);
        }

#if RACHIS_CABI_FAULT_BACKEND_METAL
        {
            const auto image = acquireSwapchainImage(device.get(), swapchain, error);
            const auto renderComplete = recordAndSubmitImageFrame(device.get(), graphicsPool, image, error);

            std::uint32_t presentStatus = RACHIS_PRESENT_STATUS_SURFACE_OUT_OF_DATE;
            {
                const ScopedMetalPresentFailure injectPreConsume(
                    rachis::metal::test::InjectedPresentFailure::beforeConsumeValidationError);
                CHECK(rachis_swapchain_present(device.get(), swapchain, renderComplete, &presentStatus, &error) ==
                      RACHIS_STATUS_VALIDATION_ERROR);
            }
            requireOk(queryBorrowedTextureLayout(device.get(), image.texture, error), error);

            requireOk(rachis_swapchain_present(device.get(), swapchain, renderComplete, &presentStatus, &error), error);
            CHECK(presentStatus == RACHIS_PRESENT_STATUS_PRESENTED);
            CHECK(queryBorrowedTextureLayout(device.get(), image.texture, error) == RACHIS_STATUS_INVALID_ARGUMENT);
            CHECK(probeBorrowedViewRendering(device.get(), graphicsPool, image.render_target_view, error) ==
                  RACHIS_STATUS_INVALID_ARGUMENT);

            requireOk(rachis_timeline_wait(device.get(), renderComplete, &error), error);
            requireOk(rachis_timeline_release(device.get(), renderComplete, &error), error);
        }
#endif

        {
            const auto image = acquireSwapchainImage(device.get(), swapchain, error);
            const auto renderComplete = recordAndSubmitImageFrame(device.get(), graphicsPool, image, error);

            std::uint32_t presentStatus = RACHIS_PRESENT_STATUS_PRESENTED;
            {
#if RACHIS_CABI_FAULT_BACKEND_METAL
                const ScopedMetalPresentFailure injectPostConsume(
                    rachis::metal::test::InjectedPresentFailure::afterConsumeNativeFailure);
#elif RACHIS_CABI_FAULT_BACKEND_D3D12
                const ScopedPresentResult injectPostConsume(static_cast<std::int32_t>(E_FAIL));
#endif
                CHECK(rachis_swapchain_present(device.get(), swapchain, renderComplete, &presentStatus, &error) ==
                      RACHIS_STATUS_NATIVE_FAILURE);
            }

            CHECK(queryBorrowedTextureLayout(device.get(), image.texture, error) == RACHIS_STATUS_INVALID_ARGUMENT);
            CHECK(probeBorrowedViewRendering(device.get(), graphicsPool, image.render_target_view, error) ==
                  RACHIS_STATUS_INVALID_ARGUMENT);
            CHECK(rachis_swapchain_cancel(device.get(), swapchain, &error) == RACHIS_STATUS_INVALID_STATE);

            requireOk(rachis_timeline_wait(device.get(), renderComplete, &error), error);
            requireOk(rachis_timeline_release(device.get(), renderComplete, &error), error);

#if RACHIS_CABI_FAULT_BACKEND_D3D12
            requireOk(rachis_swapchain_destroy(device.get(), swapchain, &error), error);
            swapchain = createSwapchain(device.get(), surface, error);
#endif
            const auto nextImage = acquireSwapchainImage(device.get(), swapchain, error);
            requireOk(queryBorrowedTextureLayout(device.get(), nextImage.texture, error), error);
            requireOk(rachis_swapchain_cancel(device.get(), swapchain, &error), error);
        }

        for (const auto buffer : liveBuffers)
        {
            requireOk(rachis_buffer_destroy(device.get(), buffer, &error), error);
        }
        requireOk(rachis_command_pool_destroy(device.get(), copyPool, &error), error);
        requireOk(rachis_command_pool_destroy(device.get(), graphicsPool, &error), error);
        requireOk(rachis_swapchain_destroy(device.get(), siblingSwapchain, &error), error);
        requireOk(rachis_swapchain_destroy(device.get(), swapchain, &error), error);
        requireOk(rachis_surface_destroy(device.get(), siblingSurface, &error), error);
        requireOk(rachis_surface_destroy(device.get(), surface, &error), error);
    }

    {
        const SurfaceHost host;
        REQUIRE(host.valid());
        auto error = makeError();
        auto device = createDevice(error, RACHIS_VALIDATION_DISABLED);
        const auto surfaceDesc = host.surfaceDesc();
        rachis_handle surface = 0;
        requireOk(rachis_surface_create(device.get(), &surfaceDesc, &surface, &error), error);
        const auto swapchain = createSwapchain(device.get(), surface, error);
        const auto graphicsPool = createPool(device.get(), RACHIS_QUEUE_GRAPHICS, error);

        const auto image = acquireSwapchainImage(device.get(), swapchain, error);
        const auto renderComplete = recordAndSubmitImageFrame(device.get(), graphicsPool, image, error);

        std::uint32_t presentStatus = RACHIS_PRESENT_STATUS_PRESENTED;
        {
#if RACHIS_CABI_FAULT_BACKEND_METAL
            const ScopedMetalPresentFailure injectPostConsumeLoss(
                rachis::metal::test::InjectedPresentFailure::afterConsumeDeviceLost);
#elif RACHIS_CABI_FAULT_BACKEND_D3D12
            const ScopedPresentResult injectPostConsumeLoss(static_cast<std::int32_t>(0x887A0005U));
#endif
            CHECK(rachis_swapchain_present(device.get(), swapchain, renderComplete, &presentStatus, &error) ==
                  RACHIS_STATUS_DEVICE_LOST);
        }

        std::uint32_t operationalState = RACHIS_DEVICE_INVALID;
        requireOk(rachis_device_operational_state(device.get(), &operationalState, &error), error);
        CHECK(operationalState == RACHIS_DEVICE_LOST);

        CHECK(queryBorrowedTextureLayout(device.get(), image.texture, error) == RACHIS_STATUS_INVALID_ARGUMENT);
#if RACHIS_CABI_FAULT_BACKEND_D3D12
        CHECK(rachis_swapchain_cancel(device.get(), swapchain, &error) == RACHIS_STATUS_DEVICE_LOST);
#else
        CHECK(rachis_swapchain_cancel(device.get(), swapchain, &error) == RACHIS_STATUS_INVALID_STATE);
#endif

        requireOk(rachis_timeline_release(device.get(), renderComplete, &error), error);
        requireOk(rachis_command_pool_destroy(device.get(), graphicsPool, &error), error);
        requireOk(rachis_swapchain_destroy(device.get(), swapchain, &error), error);
        requireOk(rachis_surface_destroy(device.get(), surface, &error), error);
    }

#if RACHIS_CABI_FAULT_BACKEND_METAL
    {
        const SurfaceHost host;
        REQUIRE(host.valid());
        auto error = makeError();
        auto device = createDevice(error, RACHIS_VALIDATION_DISABLED);
        const auto surfaceDesc = host.surfaceDesc();
        rachis_handle surface = 0;
        requireOk(rachis_surface_create(device.get(), &surfaceDesc, &surface, &error), error);
        const auto swapchain = createSwapchain(device.get(), surface, error);
        const auto graphicsPool = createPool(device.get(), RACHIS_QUEUE_GRAPHICS, error);

        const auto image = acquireSwapchainImage(device.get(), swapchain, error);
        rachis_handle command = 0;
        requireOk(rachis_command_acquire(device.get(), graphicsPool, &command, &error), error);
        rachis_handle timeline = 0;
        requireOk(rachis_command_submit(device.get(), command, &timeline, &error), error);

        std::uint32_t presentStatus = RACHIS_PRESENT_STATUS_PRESENTED;
        {
            const ScopedPresentResult injectPreConsumeLoss(static_cast<std::int32_t>(0x887A0005U));
            CHECK(rachis_swapchain_present(device.get(), swapchain, timeline, &presentStatus, &error) ==
                  RACHIS_STATUS_DEVICE_LOST);
        }

        std::uint32_t operationalState = RACHIS_DEVICE_INVALID;
        requireOk(rachis_device_operational_state(device.get(), &operationalState, &error), error);
        CHECK(operationalState == RACHIS_DEVICE_LOST);

        requireOk(queryBorrowedTextureLayout(device.get(), image.texture, error), error);
        requireOk(rachis_swapchain_cancel(device.get(), swapchain, &error), error);
        CHECK(queryBorrowedTextureLayout(device.get(), image.texture, error) == RACHIS_STATUS_INVALID_ARGUMENT);

        requireOk(rachis_timeline_release(device.get(), timeline, &error), error);
        requireOk(rachis_command_pool_destroy(device.get(), graphicsPool, &error), error);
        requireOk(rachis_swapchain_destroy(device.get(), swapchain, &error), error);
        requireOk(rachis_surface_destroy(device.get(), surface, &error), error);
    }
#endif
}

TEST_CASE("C ABI query_presentation_progress and sibling swapchains do not stall device calls during present",
          "[size:large]" RACHIS_CABI_PRESENT_DESKTOP_REQUIRES RACHIS_CABI_DEVICE_REQUIRES
          "[rachis][cabi][swapchain][concurrency]")
{
    const SurfaceHost hostA;
    const SurfaceHost hostB;
    REQUIRE(hostA.valid());
    REQUIRE(hostB.valid());

    auto error = makeError();
    auto device = createDevice(error);
    const auto surfaceDescA = hostA.surfaceDesc();
    const auto surfaceDescB = hostB.surfaceDesc();
    rachis_handle surfaceA = 0;
    rachis_handle surfaceB = 0;
    requireOk(rachis_surface_create(device.get(), &surfaceDescA, &surfaceA, &error), error);
    requireOk(rachis_surface_create(device.get(), &surfaceDescB, &surfaceB, &error), error);
    const auto swapchainA = createSwapchain(device.get(), surfaceA, error);
    const auto swapchainB = createSwapchain(device.get(), surfaceB, error);
    const auto graphicsPoolA = createPool(device.get(), RACHIS_QUEUE_GRAPHICS, error);
    const auto graphicsPoolB = createPool(device.get(), RACHIS_QUEUE_GRAPHICS, error);
    const auto copyPool = createPool(device.get(), RACHIS_QUEUE_COPY, error);

    const auto imageA = acquireSwapchainImage(device.get(), swapchainA, error);
    const auto renderCompleteA = recordAndSubmitImageFrame(device.get(), graphicsPoolA, imageA, error);

    struct PresentPauseContext final
    {
        rachis_handle targetSwapchain = 0;
        std::mutex mutex;
        std::condition_variable cv;
        bool entered = false;
        bool release = false;
        std::atomic<std::uint32_t> siblingPresentHooks{0};
    } pauseCtx;
    pauseCtx.targetSwapchain = swapchainA;

    const auto presentHook = [](rachis_handle swapchain, void* userData) noexcept {
        auto& ctx = *static_cast<PresentPauseContext*>(userData);
        if (swapchain == ctx.targetSwapchain)
        {
            std::unique_lock lock(ctx.mutex);
            if (!ctx.entered)
            {
                ctx.entered = true;
                ctx.cv.notify_all();
                ctx.cv.wait(lock, [&] { return ctx.release; });
            }
            return;
        }
        ctx.siblingPresentHooks.fetch_add(1, std::memory_order_relaxed);
    };

    rachis_status presentStatusCodeA = RACHIS_STATUS_NATIVE_FAILURE;
    std::uint32_t presentStatusA = RACHIS_PRESENT_STATUS_SURFACE_OUT_OF_DATE;
    rachis_error presentErrorA = makeError();

    rachis_status progressStatusA = RACHIS_STATUS_NATIVE_FAILURE;
    rachis_swapchain_presentation_progress progressA{};
    progressA.struct_size = sizeof(progressA);
    rachis_error progressErrorA = makeError();
    std::atomic<bool> progressDispatchedA{false};
    std::atomic<bool> progressCompletedA{false};

    rachis_status siblingStatus = RACHIS_STATUS_OK;
    std::string siblingFailure;
    std::uint32_t presentStatusB = RACHIS_PRESENT_STATUS_SURFACE_OUT_OF_DATE;
    rachis_swapchain_presentation_progress progressBeforeB{};
    rachis_swapchain_presentation_progress progressAfterB{};

    rachis_status deviceWorkerStatus = RACHIS_STATUS_OK;
    std::string deviceWorkerFailure;
    const std::array<std::uint8_t, 8> expectedCopyBytes{13, 21, 34, 55, 89, 144, 233, 42};
    std::array<std::uint8_t, 8> actualCopyBytes{};

    bool presentEnteredInTime = false;
    bool progressDispatchedInTime = false;
    bool remainedGatedDuringPresent = false;

    {
        const ScopedSwapchainPresentHook scopedHook(presentHook, &pauseCtx);

        std::thread presentThreadA([&] {
            presentStatusCodeA =
                rachis_swapchain_present(device.get(), swapchainA, renderCompleteA, &presentStatusA, &presentErrorA);
        });

        {
            std::unique_lock lock(pauseCtx.mutex);
            presentEnteredInTime =
                pauseCtx.cv.wait_for(lock, std::chrono::seconds(10), [&] { return pauseCtx.entered; });
        }

        std::mutex dispatchMutex;
        std::condition_variable dispatchCv;
        std::thread progressThreadA;
        if (presentEnteredInTime)
        {
            progressThreadA = std::thread([&] {
                {
                    const std::scoped_lock lock(dispatchMutex);
                    progressDispatchedA.store(true, std::memory_order_release);
                }
                dispatchCv.notify_all();
                progressStatusA =
                    rachis_swapchain_query_presentation_progress(device.get(), swapchainA, &progressA, &progressErrorA);
                progressCompletedA.store(true, std::memory_order_release);
            });
            {
                std::unique_lock lock(dispatchMutex);
                progressDispatchedInTime = dispatchCv.wait_for(
                    lock, std::chrono::seconds(5), [&] { return progressDispatchedA.load(std::memory_order_acquire); });
            }

            std::thread siblingThread([&] {
                auto localError = makeError();
                const auto checkStep = [&](rachis_status step) {
                    if (step != RACHIS_STATUS_OK && siblingStatus == RACHIS_STATUS_OK)
                    {
                        siblingStatus = step;
                        siblingFailure = localError.message;
                    }
                    return step == RACHIS_STATUS_OK;
                };

                rachis_surface_capabilities capsB{};
                capsB.struct_size = sizeof(capsB);
                if (!checkStep(rachis_surface_query_capabilities(device.get(), surfaceB, &capsB, &localError)))
                {
                    return;
                }

                progressBeforeB.struct_size = sizeof(progressBeforeB);
                if (!checkStep(rachis_swapchain_query_presentation_progress(
                        device.get(), swapchainB, &progressBeforeB, &localError)))
                {
                    return;
                }

                rachis_swapchain_image cancelledImageB{};
                cancelledImageB.struct_size = sizeof(cancelledImageB);
                if (!checkStep(rachis_swapchain_acquire(device.get(), swapchainB, &cancelledImageB, &localError)))
                {
                    return;
                }
                if (!checkStep(rachis_swapchain_cancel(device.get(), swapchainB, &localError)))
                {
                    return;
                }

                rachis_swapchain_image imageB{};
                imageB.struct_size = sizeof(imageB);
                if (!checkStep(rachis_swapchain_acquire(device.get(), swapchainB, &imageB, &localError)))
                {
                    return;
                }

                rachis_handle commandB = 0;
                if (!checkStep(rachis_command_acquire(device.get(), graphicsPoolB, &commandB, &localError)))
                {
                    return;
                }

                rachis_texture_barrier toRender{};
                toRender.texture = imageB.texture;
                toRender.stages_before = imageB.initial_stages;
                toRender.access_before = imageB.initial_access;
                toRender.layout_before = imageB.initial_layout;
                toRender.stages_after = RACHIS_STAGE_RENDER_TARGET;
                toRender.access_after = RACHIS_ACCESS_RENDER_TARGET;
                toRender.layout_after = RACHIS_LAYOUT_RENDER_TARGET;
                toRender.aspect = RACHIS_ASPECT_COLOR;
                toRender.mip_level_count = RACHIS_REMAINING_SUBRESOURCES;
                toRender.array_layer_count = RACHIS_REMAINING_SUBRESOURCES;
                rachis_barrier_batch batch{};
                batch.struct_size = sizeof(batch);
                batch.abi_version = RACHIS_ABI_VERSION;
                batch.texture_barriers = &toRender;
                batch.texture_barrier_count = 1;
                if (!checkStep(rachis_command_barriers(device.get(), commandB, &batch, &localError)))
                {
                    return;
                }

                rachis_color_attachment attachment{};
                attachment.view = imageB.render_target_view;
                attachment.load_op = RACHIS_LOAD_OP_CLEAR;
                attachment.store_op = RACHIS_STORE_OP_STORE;
                attachment.clear_color[1] = 0.6F;
                attachment.clear_color[3] = 1.0F;
                rachis_rendering_desc rendering{};
                rendering.struct_size = sizeof(rendering);
                rendering.abi_version = RACHIS_ABI_VERSION;
                rendering.color_attachments = &attachment;
                rendering.color_attachment_count = 1;
                if (!checkStep(rachis_command_begin_rendering(device.get(), commandB, &rendering, &localError)))
                {
                    return;
                }
                if (!checkStep(rachis_command_end_rendering(device.get(), commandB, &localError)))
                {
                    return;
                }

                rachis_texture_barrier toPresent{};
                toPresent.texture = imageB.texture;
                toPresent.stages_before = RACHIS_STAGE_RENDER_TARGET;
                toPresent.access_before = RACHIS_ACCESS_RENDER_TARGET;
                toPresent.layout_before = RACHIS_LAYOUT_RENDER_TARGET;
                toPresent.stages_after = imageB.required_final_stages;
                toPresent.access_after = imageB.required_final_access;
                toPresent.layout_after = imageB.required_final_layout;
                toPresent.aspect = RACHIS_ASPECT_COLOR;
                toPresent.mip_level_count = RACHIS_REMAINING_SUBRESOURCES;
                toPresent.array_layer_count = RACHIS_REMAINING_SUBRESOURCES;
                batch.texture_barriers = &toPresent;
                if (!checkStep(rachis_command_barriers(device.get(), commandB, &batch, &localError)))
                {
                    return;
                }

                rachis_handle renderCompleteB = 0;
                if (!checkStep(rachis_command_submit(device.get(), commandB, &renderCompleteB, &localError)))
                {
                    return;
                }
                if (!checkStep(rachis_swapchain_present(
                        device.get(), swapchainB, renderCompleteB, &presentStatusB, &localError)))
                {
                    static_cast<void>(rachis_timeline_release(device.get(), renderCompleteB, &localError));
                    return;
                }
                if (!checkStep(rachis_timeline_wait(device.get(), renderCompleteB, &localError)))
                {
                    static_cast<void>(rachis_timeline_release(device.get(), renderCompleteB, &localError));
                    return;
                }
                if (!checkStep(rachis_timeline_release(device.get(), renderCompleteB, &localError)))
                {
                    return;
                }

                progressAfterB.struct_size = sizeof(progressAfterB);
                static_cast<void>(checkStep(rachis_swapchain_query_presentation_progress(
                    device.get(), swapchainB, &progressAfterB, &localError)));
            });

            std::thread deviceWorker([&] {
                auto localError = makeError();
                const auto checkStep = [&](rachis_status step) {
                    if (step != RACHIS_STATUS_OK && deviceWorkerStatus == RACHIS_STATUS_OK)
                    {
                        deviceWorkerStatus = step;
                        deviceWorkerFailure = localError.message;
                    }
                    return step == RACHIS_STATUS_OK;
                };

                rachis_buffer_desc uploadDesc{};
                uploadDesc.struct_size = sizeof(uploadDesc);
                uploadDesc.abi_version = RACHIS_ABI_VERSION;
                uploadDesc.size = expectedCopyBytes.size();
                uploadDesc.usage = RACHIS_BUFFER_COPY_SOURCE;
                uploadDesc.memory_type = RACHIS_MEMORY_UPLOAD;
                rachis_handle upload = 0;
                if (!checkStep(rachis_buffer_create(device.get(), &uploadDesc, &upload, &localError)))
                {
                    return;
                }

                rachis_buffer_desc readbackDesc = uploadDesc;
                readbackDesc.usage = RACHIS_BUFFER_COPY_DESTINATION;
                readbackDesc.memory_type = RACHIS_MEMORY_READBACK;
                rachis_handle readback = 0;
                if (!checkStep(rachis_buffer_create(device.get(), &readbackDesc, &readback, &localError)))
                {
                    static_cast<void>(rachis_buffer_destroy(device.get(), upload, &localError));
                    return;
                }

                if (checkStep(rachis_buffer_write(
                        device.get(), upload, 0, expectedCopyBytes.data(), expectedCopyBytes.size(), &localError)))
                {
                    rachis_handle cmd = 0;
                    if (checkStep(rachis_command_acquire(device.get(), copyPool, &cmd, &localError)))
                    {
                        if (checkStep(rachis_command_copy_buffer(
                                device.get(), cmd, upload, readback, 0, 0, expectedCopyBytes.size(), &localError)))
                        {
                            rachis_handle timeline = 0;
                            if (checkStep(rachis_command_submit(device.get(), cmd, &timeline, &localError)))
                            {
                                static_cast<void>(checkStep(rachis_timeline_wait(device.get(), timeline, &localError)));
                                static_cast<void>(
                                    checkStep(rachis_timeline_release(device.get(), timeline, &localError)));
                                static_cast<void>(checkStep(rachis_buffer_read(device.get(),
                                                                               readback,
                                                                               0,
                                                                               actualCopyBytes.data(),
                                                                               actualCopyBytes.size(),
                                                                               &localError)));
                            }
                        }
                        else
                        {
                            static_cast<void>(rachis_command_discard(device.get(), cmd, &localError));
                        }
                    }
                }

                static_cast<void>(checkStep(rachis_buffer_destroy(device.get(), readback, &localError)));
                static_cast<void>(checkStep(rachis_buffer_destroy(device.get(), upload, &localError)));
            });

            siblingThread.join();
            deviceWorker.join();

            remainedGatedDuringPresent = !progressCompletedA.load(std::memory_order_acquire);
        }

        {
            const std::scoped_lock lock(pauseCtx.mutex);
            pauseCtx.release = true;
        }
        pauseCtx.cv.notify_all();
        presentThreadA.join();
        if (progressThreadA.joinable())
        {
            progressThreadA.join();
        }
    }

    REQUIRE(presentEnteredInTime);
    CHECK(progressDispatchedInTime);
    CHECK(remainedGatedDuringPresent);
    INFO(presentErrorA.message);
    CHECK(presentStatusCodeA == RACHIS_STATUS_OK);
    CHECK(presentStatusA == RACHIS_PRESENT_STATUS_PRESENTED);
    INFO(progressErrorA.message);
    CHECK(progressStatusA == RACHIS_STATUS_OK);
    INFO(siblingFailure);
    CHECK(siblingStatus == RACHIS_STATUS_OK);
    CHECK(presentStatusB == RACHIS_PRESENT_STATUS_PRESENTED);
    CHECK(pauseCtx.siblingPresentHooks.load(std::memory_order_relaxed) == 1U);
    INFO(deviceWorkerFailure);
    CHECK(deviceWorkerStatus == RACHIS_STATUS_OK);
    CHECK(actualCopyBytes == expectedCopyBytes);

    requireOk(rachis_timeline_wait(device.get(), renderCompleteA, &error), error);
    requireOk(rachis_timeline_release(device.get(), renderCompleteA, &error), error);
    requireOk(rachis_command_pool_destroy(device.get(), copyPool, &error), error);
    requireOk(rachis_command_pool_destroy(device.get(), graphicsPoolB, &error), error);
    requireOk(rachis_command_pool_destroy(device.get(), graphicsPoolA, &error), error);
    requireOk(rachis_swapchain_destroy(device.get(), swapchainB, &error), error);
    requireOk(rachis_swapchain_destroy(device.get(), swapchainA, &error), error);
    requireOk(rachis_surface_destroy(device.get(), surfaceB, &error), error);
    requireOk(rachis_surface_destroy(device.get(), surfaceA, &error), error);
}
#endif
