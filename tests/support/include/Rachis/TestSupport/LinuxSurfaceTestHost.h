#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace rachis::test
{

    struct WindowExtent final
    {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    class WaylandTestWindow final
    {
        struct State;

    public:
        [[nodiscard]] static bool compositorReachable();
        [[nodiscard]] static std::unique_ptr<WaylandTestWindow> open(const char* title, std::string& error);

        explicit WaylandTestWindow(std::unique_ptr<State> state) noexcept;
        ~WaylandTestWindow();
        WaylandTestWindow(const WaylandTestWindow&) = delete;
        WaylandTestWindow& operator=(const WaylandTestWindow&) = delete;
        WaylandTestWindow(WaylandTestWindow&&) = delete;
        WaylandTestWindow& operator=(WaylandTestWindow&&) = delete;

        [[nodiscard]] void* display() const noexcept;
        [[nodiscard]] void* surface() const noexcept;

        [[nodiscard]] bool roundtrip();
        [[nodiscard]] int connectionError() const noexcept;

    private:
        std::unique_ptr<State> state_;
    };

    class XcbTestWindow final
    {
        struct State;

    public:
        [[nodiscard]] static std::unique_ptr<XcbTestWindow> open(const char* title,
                                                                 WindowExtent extent,
                                                                 std::string& error);

        explicit XcbTestWindow(std::unique_ptr<State> state) noexcept;
        ~XcbTestWindow();
        XcbTestWindow(const XcbTestWindow&) = delete;
        XcbTestWindow& operator=(const XcbTestWindow&) = delete;
        XcbTestWindow(XcbTestWindow&&) = delete;
        XcbTestWindow& operator=(XcbTestWindow&&) = delete;

        [[nodiscard]] void* connection() const noexcept;
        [[nodiscard]] void* window() const noexcept;

        [[nodiscard]] WindowExtent resize(WindowExtent extent);
        [[nodiscard]] WindowExtent geometry() const;
        [[nodiscard]] int connectionError() const noexcept;

    private:
        std::unique_ptr<State> state_;
    };

} // namespace rachis::test
