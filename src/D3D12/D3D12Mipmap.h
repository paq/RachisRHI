#pragma once

#include "D3D12/D3D12Binding.h"
#include "D3D12/D3D12Queue.h"
#include "D3D12/D3D12Resource.h"
#include "Rachis/Resource.h"

#include <d3d12.h>

#include <memory>

namespace rachis::d3d12
{

    class MipmapGenerator final
    {
    public:
        MipmapGenerator();
        ~MipmapGenerator();

        MipmapGenerator(const MipmapGenerator&) = delete;
        MipmapGenerator& operator=(const MipmapGenerator&) = delete;
        MipmapGenerator(MipmapGenerator&&) = delete;
        MipmapGenerator& operator=(MipmapGenerator&&) = delete;

        [[nodiscard]] Status initialize(ID3D12Device& device);
        [[nodiscard]] Status generate(QueueManager& queues,
                                      ResourceManager& resources,
                                      BindingManager& bindings,
                                      detail::BackendCommandBuffer& commandBuffer,
                                      const TextureMipmapGeneration& generation);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rachis::d3d12
