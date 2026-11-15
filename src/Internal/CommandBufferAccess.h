#pragma once

#include "Internal/BackendCommandBuffer.h"
#include "Rachis/Command.h"

namespace rachis::detail
{

    struct CommandBufferAccess final
    {
        [[nodiscard]] static CommandBuffer make(std::unique_ptr<BackendCommandBuffer> backend) noexcept
        {
            return CommandBuffer(std::move(backend));
        }

        [[nodiscard]] static BackendCommandBuffer* get(CommandBuffer& commandBuffer) noexcept
        {
            return commandBuffer.backend_.get();
        }

        [[nodiscard]] static const BackendCommandBuffer* get(const CommandBuffer& commandBuffer) noexcept
        {
            return commandBuffer.backend_.get();
        }
    };

} // namespace rachis::detail
