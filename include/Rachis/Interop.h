#pragma once

#include <cstdint>
#include <limits>

namespace rachis
{

    enum class SharedHandleType : std::uint8_t
    {
        unknown,
        windowsNt,
    };

    struct SharedHandleView final
    {
        SharedHandleType type = SharedHandleType::unknown;
        std::uintptr_t value = 0;

        [[nodiscard]] constexpr bool isValid() const noexcept
        {
            switch (type)
            {
            case SharedHandleType::windowsNt:
                return value != 0 && value != std::numeric_limits<std::uintptr_t>::max();
            case SharedHandleType::unknown:
                return false;
            }
            return false;
        }
    };

    class SharedHandle final
    {
    public:
        SharedHandle() noexcept = default;
        ~SharedHandle();

        SharedHandle(SharedHandle&& other) noexcept;
        SharedHandle& operator=(SharedHandle&& other) noexcept;

        SharedHandle(const SharedHandle&) = delete;
        SharedHandle& operator=(const SharedHandle&) = delete;

        [[nodiscard]] static SharedHandle adopt(SharedHandleView handle) noexcept;

        [[nodiscard]] bool isValid() const noexcept
        {
            return view_.isValid();
        }
        [[nodiscard]] explicit operator bool() const noexcept
        {
            return isValid();
        }
        [[nodiscard]] SharedHandleView view() const noexcept
        {
            return view_;
        }
        [[nodiscard]] SharedHandleView release() noexcept;
        void reset() noexcept;

    private:
        explicit SharedHandle(SharedHandleView handle) noexcept : view_(handle)
        {
        }

        SharedHandleView view_;
    };

    struct SharedTimelinePointView final
    {
        SharedHandleView handle = {};
        std::uint64_t value = 0;

        [[nodiscard]] constexpr bool isValid() const noexcept
        {
            return handle.isValid() && value != 0;
        }
    };

    struct SharedTimelinePoint final
    {
        SharedHandle handle = {};
        std::uint64_t value = 0;

        [[nodiscard]] bool isValid() const noexcept
        {
            return handle.isValid() && value != 0;
        }
        [[nodiscard]] SharedTimelinePointView view() const noexcept
        {
            return {handle.view(), value};
        }
    };

} // namespace rachis
