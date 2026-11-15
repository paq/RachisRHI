#include "LinuxSurfaceTestHost.h"

#include "xdg-shell-client-protocol.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <poll.h>
#include <string>
#include <string_view>
#include <wayland-client.h>
#include <xcb/xcb.h>

namespace rachis::test
{

    namespace
    {

        [[nodiscard]] std::string environmentValue(const char* name)
        {
            const char* value = std::getenv(name);
            return value == nullptr ? std::string("<unset>") : std::string(value);
        }

        struct FreeDeleter final
        {
            void operator()(void* pointer) const noexcept
            {
                std::free(pointer);
            }
        };

        template<typename T>
        using XcbReply = std::unique_ptr<T, FreeDeleter>;

        template<typename Predicate>
        [[nodiscard]] bool waitForXcbEvent(xcb_connection_t* connection,
                                           std::chrono::milliseconds timeout,
                                           const Predicate& predicate)
        {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (true)
            {
                while (const XcbReply<xcb_generic_event_t> event{xcb_poll_for_event(connection)})
                {
                    if (predicate(*event))
                    {
                        return true;
                    }
                }
                if (xcb_connection_has_error(connection) != 0)
                {
                    return false;
                }
                const auto remaining =
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
                if (remaining.count() <= 0)
                {
                    return false;
                }
                pollfd descriptor{
                    .fd = xcb_get_file_descriptor(connection),
                    .events = POLLIN,
                    .revents = 0,
                };
                static_cast<void>(::poll(&descriptor, 1, static_cast<int>(remaining.count())));
            }
        }

        [[nodiscard]] std::uint8_t eventType(const xcb_generic_event_t& event) noexcept
        {
            return event.response_type & 0x7FU;
        }

    } // namespace

    struct WaylandTestWindow::State final
    {
        wl_display* display = nullptr;
        wl_registry* registry = nullptr;
        wl_compositor* compositor = nullptr;
        xdg_wm_base* wmBase = nullptr;
        wl_surface* surface = nullptr;
        xdg_surface* xdgSurface = nullptr;
        xdg_toplevel* toplevel = nullptr;
        bool configured = false;

        State() = default;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        State(State&&) = delete;
        State& operator=(State&&) = delete;

        ~State()
        {
            if (toplevel != nullptr)
            {
                xdg_toplevel_destroy(toplevel);
            }
            if (xdgSurface != nullptr)
            {
                xdg_surface_destroy(xdgSurface);
            }
            if (surface != nullptr)
            {
                wl_surface_destroy(surface);
            }
            if (wmBase != nullptr)
            {
                xdg_wm_base_destroy(wmBase);
            }
            if (compositor != nullptr)
            {
                wl_compositor_destroy(compositor);
            }
            if (registry != nullptr)
            {
                wl_registry_destroy(registry);
            }
            if (display != nullptr)
            {
                static_cast<void>(wl_display_flush(display));
                wl_display_disconnect(display);
            }
        }
    };

    WaylandTestWindow::WaylandTestWindow(std::unique_ptr<State> state) noexcept : state_(std::move(state))
    {
    }

    WaylandTestWindow::~WaylandTestWindow() = default;

    bool WaylandTestWindow::compositorReachable()
    {
        wl_display* display = wl_display_connect(nullptr);
        if (display == nullptr)
        {
            return false;
        }
        wl_display_disconnect(display);
        return true;
    }

    std::unique_ptr<WaylandTestWindow> WaylandTestWindow::open(const char* title, std::string& error)
    {
        auto state = std::make_unique<State>();
        state->display = wl_display_connect(nullptr);
        if (state->display == nullptr)
        {
            error = "wl_display_connect failed for WAYLAND_DISPLAY=" + environmentValue("WAYLAND_DISPLAY");
            return nullptr;
        }

        static const wl_registry_listener s_registryListener{
            .global =
                [](void* data,
                   wl_registry* registry,
                   std::uint32_t name,
                   const char* interface,
                   std::uint32_t version) {
                    auto& self = *static_cast<State*>(data);
                    const std::string_view kind(interface);
                    if (kind == wl_compositor_interface.name)
                    {
                        self.compositor = static_cast<wl_compositor*>(
                            wl_registry_bind(registry, name, &wl_compositor_interface, (std::min)(version, 4U)));
                    }
                    else if (kind == xdg_wm_base_interface.name)
                    {
                        self.wmBase =
                            static_cast<xdg_wm_base*>(wl_registry_bind(registry, name, &xdg_wm_base_interface, 1));
                    }
                },
            .global_remove = [](void*, wl_registry*, std::uint32_t) {},
        };
        static const xdg_wm_base_listener s_wmBaseListener{
            .ping = [](void*, xdg_wm_base* base, std::uint32_t serial) { xdg_wm_base_pong(base, serial); },
        };
        static const xdg_surface_listener s_surfaceListener{
            .configure =
                [](void* data, xdg_surface* surface, std::uint32_t serial) {
                    xdg_surface_ack_configure(surface, serial);
                    static_cast<State*>(data)->configured = true;
                },
        };
        static const xdg_toplevel_listener s_toplevelListener{
            .configure = [](void*, xdg_toplevel*, std::int32_t, std::int32_t, wl_array*) {},
            .close = [](void*, xdg_toplevel*) {},
            .configure_bounds = [](void*, xdg_toplevel*, std::int32_t, std::int32_t) {},
            .wm_capabilities = [](void*, xdg_toplevel*, wl_array*) {},
        };

        state->registry = wl_display_get_registry(state->display);
        static_cast<void>(wl_registry_add_listener(state->registry, &s_registryListener, state.get()));
        if (wl_display_roundtrip(state->display) < 0)
        {
            error = "The Wayland registry roundtrip failed.";
            return nullptr;
        }
        if (state->compositor == nullptr || state->wmBase == nullptr)
        {
            error = "The Wayland compositor does not advertise wl_compositor and xdg_wm_base.";
            return nullptr;
        }
        static_cast<void>(xdg_wm_base_add_listener(state->wmBase, &s_wmBaseListener, state.get()));

        state->surface = wl_compositor_create_surface(state->compositor);
        state->xdgSurface = xdg_wm_base_get_xdg_surface(state->wmBase, state->surface);
        static_cast<void>(xdg_surface_add_listener(state->xdgSurface, &s_surfaceListener, state.get()));
        state->toplevel = xdg_surface_get_toplevel(state->xdgSurface);
        static_cast<void>(xdg_toplevel_add_listener(state->toplevel, &s_toplevelListener, state.get()));
        xdg_toplevel_set_title(state->toplevel, title);
        xdg_toplevel_set_app_id(state->toplevel, "dev.paq.rachis.RachisRHITests");

        wl_surface_commit(state->surface);
        for (int attempt = 0; attempt < 8 && !state->configured; ++attempt)
        {
            if (wl_display_roundtrip(state->display) < 0)
            {
                error = "The Wayland connection failed while waiting for the first configure.";
                return nullptr;
            }
        }
        if (!state->configured)
        {
            error = "The Wayland compositor never configured the xdg_toplevel.";
            return nullptr;
        }
        return std::make_unique<WaylandTestWindow>(std::move(state));
    }

    void* WaylandTestWindow::display() const noexcept
    {
        return state_->display;
    }

    void* WaylandTestWindow::surface() const noexcept
    {
        return state_->surface;
    }

    bool WaylandTestWindow::roundtrip()
    {
        return wl_display_roundtrip(state_->display) >= 0;
    }

    int WaylandTestWindow::connectionError() const noexcept
    {
        return wl_display_get_error(state_->display);
    }

    struct XcbTestWindow::State final
    {
        xcb_connection_t* connection = nullptr;
        xcb_window_t window = 0;

        State() = default;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        State(State&&) = delete;
        State& operator=(State&&) = delete;

        ~State()
        {
            if (connection == nullptr)
            {
                return;
            }
            if (window != 0)
            {
                static_cast<void>(xcb_destroy_window(connection, window));
                static_cast<void>(xcb_flush(connection));
            }
            xcb_disconnect(connection);
        }
    };

    XcbTestWindow::XcbTestWindow(std::unique_ptr<State> state) noexcept : state_(std::move(state))
    {
    }

    XcbTestWindow::~XcbTestWindow() = default;

    std::unique_ptr<XcbTestWindow> XcbTestWindow::open(const char* title, WindowExtent extent, std::string& error)
    {
        auto state = std::make_unique<State>();
        int screenNumber = 0;
        state->connection = xcb_connect(nullptr, &screenNumber);
        if (xcb_connection_has_error(state->connection) != 0)
        {
            error = "xcb_connect failed for DISPLAY=" + environmentValue("DISPLAY");
            return nullptr;
        }
        auto screens = xcb_setup_roots_iterator(xcb_get_setup(state->connection));
        for (int index = 0; index < screenNumber && screens.rem > 0; ++index)
        {
            xcb_screen_next(&screens);
        }
        const xcb_screen_t* screen = screens.data;
        if (screen == nullptr)
        {
            error = "The X server reported no screen for DISPLAY=" + environmentValue("DISPLAY");
            return nullptr;
        }

        state->window = xcb_generate_id(state->connection);
        const std::array<std::uint32_t, 2> values{
            screen->black_pixel,
            XCB_EVENT_MASK_STRUCTURE_NOTIFY | XCB_EVENT_MASK_EXPOSURE,
        };
        static_cast<void>(xcb_create_window(state->connection,
                                            XCB_COPY_FROM_PARENT,
                                            state->window,
                                            screen->root,
                                            0,
                                            0,
                                            static_cast<std::uint16_t>(extent.width),
                                            static_cast<std::uint16_t>(extent.height),
                                            0,
                                            XCB_WINDOW_CLASS_INPUT_OUTPUT,
                                            screen->root_visual,
                                            XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK,
                                            values.data()));
        static_cast<void>(xcb_change_property(state->connection,
                                              XCB_PROP_MODE_REPLACE,
                                              state->window,
                                              XCB_ATOM_WM_NAME,
                                              XCB_ATOM_STRING,
                                              8,
                                              static_cast<std::uint32_t>(std::strlen(title)),
                                              title));
        static_cast<void>(xcb_map_window(state->connection, state->window));
        static_cast<void>(xcb_flush(state->connection));
        const auto window = state->window;
        const bool mapped =
            waitForXcbEvent(state->connection, std::chrono::seconds(5), [window](const xcb_generic_event_t& event) {
                return eventType(event) == XCB_MAP_NOTIFY &&
                       reinterpret_cast<const xcb_map_notify_event_t&>(event).window == window;
            });
        if (!mapped)
        {
            error = "The X server did not map the test window within 5 seconds.";
            return nullptr;
        }
        return std::make_unique<XcbTestWindow>(std::move(state));
    }

    void* XcbTestWindow::connection() const noexcept
    {
        return state_->connection;
    }

    void* XcbTestWindow::window() const noexcept
    {
        return reinterpret_cast<void*>(static_cast<std::uintptr_t>(state_->window));
    }

    WindowExtent XcbTestWindow::resize(WindowExtent extent)
    {
        const std::array<std::uint32_t, 2> values{extent.width, extent.height};
        static_cast<void>(xcb_configure_window(
            state_->connection, state_->window, XCB_CONFIG_WINDOW_WIDTH | XCB_CONFIG_WINDOW_HEIGHT, values.data()));
        static_cast<void>(xcb_flush(state_->connection));
        const auto window = state_->window;

        static_cast<void>(waitForXcbEvent(
            state_->connection, std::chrono::seconds(2), [window, extent](const xcb_generic_event_t& event) {
                if (eventType(event) != XCB_CONFIGURE_NOTIFY)
                {
                    return false;
                }
                const auto& configure = reinterpret_cast<const xcb_configure_notify_event_t&>(event);
                return configure.window == window && configure.width == extent.width &&
                       configure.height == extent.height;
            }));
        return geometry();
    }

    WindowExtent XcbTestWindow::geometry() const
    {
        xcb_generic_error_t* rawError = nullptr;
        const XcbReply<xcb_get_geometry_reply_t> reply{xcb_get_geometry_reply(
            state_->connection, xcb_get_geometry(state_->connection, state_->window), &rawError)};
        const XcbReply<xcb_generic_error_t> failure{rawError};
        if (reply == nullptr)
        {
            return {};
        }
        return {.width = reply->width, .height = reply->height};
    }

    int XcbTestWindow::connectionError() const noexcept
    {
        return xcb_connection_has_error(state_->connection);
    }

} // namespace rachis::test
