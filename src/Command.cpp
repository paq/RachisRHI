#include "Rachis/Command.h"

#include "Internal/BackendCommandBuffer.h"

#include <memory>
#include <utility>

namespace rachis
{

    CommandBuffer::CommandBuffer() noexcept = default;

    CommandBuffer::CommandBuffer(std::unique_ptr<detail::BackendCommandBuffer> backend) noexcept
        : backend_(std::move(backend))
    {
    }

    CommandBuffer::~CommandBuffer() = default;
    CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept = default;
    CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept = default;

    bool CommandBuffer::isValid() const noexcept
    {
        return backend_ != nullptr;
    }

    CommandBuffer::operator bool() const noexcept
    {
        return isValid();
    }

    QueueType CommandBuffer::queueType() const noexcept
    {
        return backend_ != nullptr ? backend_->queueType() : QueueType::graphics;
    }

} // namespace rachis
