#pragma once

#include "D3D12/D3D12Queue.h"
#include "D3D12/D3D12Resource.h"
#include "D3D12/D3D12View.h"
#include "Rachis/Presentation.h"

#include <d3d12.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace rachis::d3d12
{

    class SwapchainManager final
    {
    public:
        SwapchainManager();
        ~SwapchainManager();

        SwapchainManager(const SwapchainManager&) = delete;
        SwapchainManager& operator=(const SwapchainManager&) = delete;
        SwapchainManager(SwapchainManager&&) = delete;
        SwapchainManager& operator=(SwapchainManager&&) = delete;

        [[nodiscard]] Status initialize(ID3D12Device& device);
        [[nodiscard]] Result<SurfaceHandle> createSurface(const SurfaceDesc& desc);
        [[nodiscard]] Result<SurfaceCapabilities> querySurfaceCapabilities(SurfaceHandle surface) const;
        [[nodiscard]] Status destroySurface(SurfaceHandle surface);
        [[nodiscard]] Result<SwapchainHandle> createSwapchain(QueueManager& queues,
                                                              ResourceManager& resources,
                                                              ViewManager& views,
                                                              const SwapchainDesc& desc);
        [[nodiscard]] Status destroySwapchain(QueueManager& queues,
                                              ResourceManager& resources,
                                              ViewManager& views,
                                              SwapchainHandle swapchain,
                                              bool deviceLost,
                                              std::optional<Error>& observedDeviceLoss);
        [[nodiscard]] Status resizeSwapchain(QueueManager& queues,
                                             ResourceManager& resources,
                                             ViewManager& views,
                                             SwapchainHandle swapchain,
                                             std::uint32_t width,
                                             std::uint32_t height);
        [[nodiscard]] Result<AcquiredSwapchainImage> acquireNextImage(QueueManager& queues, SwapchainHandle swapchain);
        [[nodiscard]] bool hasAcquiredImage(SwapchainHandle swapchain) const noexcept;
        [[nodiscard]] Status cancelSwapchainImage(SwapchainHandle swapchain);
        [[nodiscard]] Result<PresentStatus> present(QueueManager& queues,
                                                    SwapchainHandle swapchain,
                                                    TimelinePoint renderComplete);
        [[nodiscard]] Result<SwapchainPresentationInfo> queryPresentation(SwapchainHandle swapchain) const;
        [[nodiscard]] Result<SwapchainFrameTiming> queryFrameTiming(SwapchainHandle swapchain) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rachis::d3d12
