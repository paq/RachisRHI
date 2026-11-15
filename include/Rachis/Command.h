#pragma once

#include "Rachis/Handle.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace rachis
{

    enum class QueueType : std::uint8_t
    {
        graphics,
        compute,
        copy,
    };

    enum class CommandPoolFlag : std::uint32_t
    {
        none = 0,

        allowIndividualReset = 1U << 0U,

        unretainedResources = 1U << 1U,
    };

    [[nodiscard]] constexpr CommandPoolFlag operator|(CommandPoolFlag left, CommandPoolFlag right) noexcept
    {
        return static_cast<CommandPoolFlag>(static_cast<std::uint32_t>(left) | static_cast<std::uint32_t>(right));
    }

    [[nodiscard]] constexpr bool hasAny(CommandPoolFlag value, CommandPoolFlag flags) noexcept
    {
        return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(flags)) != 0;
    }

    struct CommandPoolDesc final
    {
        QueueType queueType = QueueType::graphics;
        std::uint32_t maxCommandBuffers = 0;
        std::string_view label = {};
        CommandPoolFlag flags = CommandPoolFlag::none;
    };

    inline constexpr std::uint32_t maxDebugLabelBytes = 1024;
    inline constexpr std::uint32_t maxDebugGroupDepth = 64;

    struct DebugColor final
    {
        float red = 0.0F;
        float green = 0.0F;
        float blue = 0.0F;
        float alpha = 1.0F;
    };

    struct DebugMarkerDesc final
    {
        std::string_view label = {};
        std::optional<DebugColor> color = {};
    };

    namespace detail
    {
        class BackendCommandBuffer;
        struct CommandBufferAccess;
        struct TimelinePointAccess;
    } // namespace detail

    class TimelinePoint final
    {
    public:
        constexpr TimelinePoint() noexcept = default;

        [[nodiscard]] constexpr bool isValid() const noexcept
        {
            return deviceToken_ != 0 && value_ != 0;
        }

        [[nodiscard]] constexpr QueueType queueType() const noexcept
        {
            return queueType_;
        }

        [[nodiscard]] constexpr std::uint64_t value() const noexcept
        {
            return value_;
        }

        friend constexpr bool operator==(const TimelinePoint&, const TimelinePoint&) noexcept = default;

    private:
        constexpr TimelinePoint(QueueType queueType, std::uint64_t value, std::uint64_t deviceToken) noexcept
            : queueType_(queueType), value_(value), deviceToken_(deviceToken)
        {
        }

        QueueType queueType_ = QueueType::graphics;
        std::uint64_t value_ = 0;
        std::uint64_t deviceToken_ = 0;

        friend struct detail::TimelinePointAccess;
    };

    enum class TimelinePointStatus : std::uint8_t
    {
        pending,
        complete,
    };

    class CommandBuffer final
    {
    public:
        CommandBuffer() noexcept;
        ~CommandBuffer();

        CommandBuffer(CommandBuffer&& other) noexcept;
        CommandBuffer& operator=(CommandBuffer&& other) noexcept;

        CommandBuffer(const CommandBuffer&) = delete;
        CommandBuffer& operator=(const CommandBuffer&) = delete;

        [[nodiscard]] bool isValid() const noexcept;
        [[nodiscard]] explicit operator bool() const noexcept;
        [[nodiscard]] QueueType queueType() const noexcept;

    private:
        explicit CommandBuffer(std::unique_ptr<detail::BackendCommandBuffer> backend) noexcept;

        std::unique_ptr<detail::BackendCommandBuffer> backend_;

        friend class Device;
        friend struct detail::CommandBufferAccess;
    };

} // namespace rachis
