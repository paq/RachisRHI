#pragma once

#include "Rachis/Command.h"
#include "Rachis/Error.h"

#include <d3d12.h>

namespace rachis::detail
{
    class BackendCommandBuffer;
    struct DebugCommandState;
} // namespace rachis::detail

namespace rachis::d3d12
{

    class QueueManager;

    class DebugMarkerManager final
    {
    public:
        [[nodiscard]] Status beginGroup(QueueManager& queues,
                                        detail::BackendCommandBuffer& commandBuffer,
                                        const DebugMarkerDesc& desc) const;
        [[nodiscard]] Status endGroup(QueueManager& queues, detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Status insertMarker(QueueManager& queues,
                                          detail::BackendCommandBuffer& commandBuffer,
                                          const DebugMarkerDesc& desc) const;
        [[nodiscard]] Status validateProfileBegin(detail::BackendCommandBuffer& commandBuffer,
                                                  const DebugMarkerDesc& desc) const;
        void commitProfileBegin(ID3D12GraphicsCommandList& commandList,
                                detail::BackendCommandBuffer& commandBuffer,
                                const DebugMarkerDesc& desc) const noexcept;
        [[nodiscard]] Status validateProfileEnd(detail::BackendCommandBuffer& commandBuffer) const;
        void commitProfileEnd(ID3D12GraphicsCommandList& commandList,
                              detail::BackendCommandBuffer& commandBuffer) const noexcept;
    };

    void closeAbandonedDebugGroups(ID3D12GraphicsCommandList& commandList, detail::DebugCommandState& state) noexcept;

} // namespace rachis::d3d12
