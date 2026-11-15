#pragma once

#include "D3D12/D3D12Descriptor.h"
#include "D3D12/D3D12Resource.h"
#include "Rachis/Resource.h"

#include <d3d12.h>

#include <memory>

namespace rachis::d3d12
{

    struct NativeTextureView final
    {
        std::shared_ptr<void> owner;
        TextureViewSource texture;
        D3D12_CPU_DESCRIPTOR_HANDLE descriptor{};
        TextureViewDesc viewDesc;
        TextureViewUsage usage = TextureViewUsage::sampled;
    };

    class ViewManager final
    {
    public:
        ViewManager();
        ~ViewManager();

        ViewManager(const ViewManager&) = delete;
        ViewManager& operator=(const ViewManager&) = delete;
        ViewManager(ViewManager&&) = delete;
        ViewManager& operator=(ViewManager&&) = delete;

        [[nodiscard]] Status initialize(ID3D12Device& device);
        [[nodiscard]] DescriptorAllocationStatistics descriptorAllocationStatistics() const noexcept;
        [[nodiscard]] Result<TextureViewHandle> createTextureView(ResourceManager& resources,
                                                                  const TextureViewDesc& desc,
                                                                  bool externallyOwned = false);
        [[nodiscard]] Status destroyTextureView(TextureViewHandle view, bool externallyOwned = false);
        [[nodiscard]] bool hasApplicationTextureViews(TextureHandle texture) const;
        [[nodiscard]] Result<NativeTextureView> nativeTextureView(TextureViewHandle view);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rachis::d3d12
