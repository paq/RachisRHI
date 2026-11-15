#include "Rachis/rachis.h"

#include <catch2/catch_test_macros.hpp>

#include <Windows.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace
{
    using Device = std::unique_ptr<rachis_device, decltype(&rachis_device_destroy)>;

    constexpr std::uint32_t windowWidth = 320;
    constexpr std::uint32_t windowHeight = 240;

    class TestWindow final
    {
    public:
        TestWindow()
            : window_(CreateWindowExW(0,
                                      L"STATIC",
                                      L"Rachis C ABI presentation",
                                      WS_OVERLAPPEDWINDOW,
                                      CW_USEDEFAULT,
                                      CW_USEDEFAULT,
                                      static_cast<int>(windowWidth),
                                      static_cast<int>(windowHeight),
                                      nullptr,
                                      nullptr,
                                      GetModuleHandleW(nullptr),
                                      nullptr))
        {
            if (window_ != nullptr)
            {
                static_cast<void>(ShowWindow(window_, SW_SHOWNOACTIVATE));
            }
        }

        ~TestWindow()
        {
            if (window_ != nullptr)
            {
                static_cast<void>(DestroyWindow(window_));
            }
        }

        TestWindow(const TestWindow&) = delete;
        TestWindow& operator=(const TestWindow&) = delete;
        TestWindow(TestWindow&&) = delete;
        TestWindow& operator=(TestWindow&&) = delete;

        [[nodiscard]] HWND get() const noexcept
        {
            return window_;
        }

    private:
        HWND window_ = nullptr;
    };

    [[nodiscard]] rachis_error makeError() noexcept
    {
        rachis_error error{};
        error.struct_size = sizeof(error);
        return error;
    }

    void requireOk(rachis_status status, const rachis_error& error)
    {
        INFO(error.message);
        REQUIRE(status == RACHIS_STATUS_OK);
    }

    [[nodiscard]] rachis_texture_barrier imageBarrier(rachis_handle texture,
                                                      std::uint32_t stagesBefore,
                                                      std::uint32_t accessBefore,
                                                      std::uint32_t layoutBefore,
                                                      std::uint32_t stagesAfter,
                                                      std::uint32_t accessAfter,
                                                      std::uint32_t layoutAfter) noexcept
    {
        rachis_texture_barrier barrier{};
        barrier.texture = texture;
        barrier.stages_before = stagesBefore;
        barrier.stages_after = stagesAfter;
        barrier.access_before = accessBefore;
        barrier.access_after = accessAfter;
        barrier.layout_before = layoutBefore;
        barrier.layout_after = layoutAfter;
        barrier.aspect = RACHIS_ASPECT_COLOR;
        barrier.mip_level_count = RACHIS_REMAINING_SUBRESOURCES;
        barrier.array_layer_count = RACHIS_REMAINING_SUBRESOURCES;
        return barrier;
    }

    [[nodiscard]] rachis_status recordBarrier(rachis_device* device,
                                              rachis_handle command,
                                              const rachis_texture_barrier& barrier,
                                              rachis_error& error)
    {
        rachis_barrier_batch batch{};
        batch.struct_size = sizeof(batch);
        batch.abi_version = RACHIS_ABI_VERSION;
        batch.texture_barriers = &barrier;
        batch.texture_barrier_count = 1;
        return rachis_command_barriers(device, command, &batch, &error);
    }

    [[nodiscard]] rachis_swapchain_image acquireImage(rachis_device* device,
                                                      rachis_handle swapchain,
                                                      rachis_error& error)
    {
        rachis_swapchain_image image{};
        image.struct_size = sizeof(image);
        requireOk(rachis_swapchain_acquire(device, swapchain, &image, &error), error);
        REQUIRE(image.texture != 0);
        REQUIRE(image.render_target_view != 0);
        CHECK(image.format == RACHIS_FORMAT_BGRA8_UNORM);
        CHECK(image.color_space == RACHIS_COLOR_SPACE_SRGB_NONLINEAR);
        CHECK(image.initial_layout <= RACHIS_LAYOUT_RESOLVE_DESTINATION);
        CHECK(image.required_final_layout == RACHIS_LAYOUT_PRESENT);
        return image;
    }

    void submitAndPresent(rachis_device* device, rachis_handle command, rachis_handle swapchain, rachis_error& error)
    {
        rachis_handle timeline = 0;
        requireOk(rachis_command_submit(device, command, &timeline, &error), error);
        std::uint32_t status = RACHIS_PRESENT_STATUS_SURFACE_OUT_OF_DATE;
        requireOk(rachis_swapchain_present(device, swapchain, timeline, &status, &error), error);
        CHECK(status == RACHIS_PRESENT_STATUS_PRESENTED);
        requireOk(rachis_timeline_wait(device, timeline, &error), error);
        requireOk(rachis_timeline_release(device, timeline, &error), error);
    }
} // namespace

TEST_CASE("C ABI presents cleared and copied swapchain images",
          "[size:medium][requires:InteractiveDesktop]" RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][presentation]")
{
    const TestWindow window;
    REQUIRE(window.get() != nullptr);
    auto error = makeError();
    rachis_device_desc deviceDesc{};
    deviceDesc.struct_size = sizeof(deviceDesc);
    deviceDesc.abi_version = RACHIS_ABI_VERSION;
    deviceDesc.backend = RACHIS_SMOKE_BACKEND;
    deviceDesc.adapter_preference = RACHIS_SMOKE_ADAPTER;
    deviceDesc.validation_level = RACHIS_VALIDATION_STANDARD;
    deviceDesc.application_name = "Rachis C ABI presentation";
    rachis_device* native = nullptr;
    requireOk(rachis_device_create(&deviceDesc, &native, &error), error);
    const Device device{native, &rachis_device_destroy};

    rachis_surface_desc surfaceDesc{};
    surfaceDesc.struct_size = sizeof(surfaceDesc);
    surfaceDesc.abi_version = RACHIS_ABI_VERSION;
    surfaceDesc.kind = RACHIS_SURFACE_WIN32;
    surfaceDesc.window_handle = window.get();
    rachis_handle surface = 0;
    requireOk(rachis_surface_create(device.get(), &surfaceDesc, &surface, &error), error);

    rachis_surface_capabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities) - 1;
    CHECK(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error) ==
          RACHIS_STATUS_ABI_MISMATCH);
    capabilities.struct_size = sizeof(capabilities);
    requireOk(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error), error);
    REQUIRE(capabilities.format_count > 0);
    REQUIRE(capabilities.format_count <= RACHIS_MAX_SURFACE_FORMATS);
    constexpr auto knownPresentModes =
        static_cast<std::uint32_t>(RACHIS_PRESENT_MODE_FIFO_BIT | RACHIS_PRESENT_MODE_IMMEDIATE_BIT);
    CHECK((capabilities.present_modes & RACHIS_PRESENT_MODE_FIFO_BIT) != 0);
    CHECK((capabilities.present_modes & ~knownPresentModes) == 0);
    CHECK((capabilities.usages & RACHIS_TEXTURE_RENDER_TARGET) != 0);
    auto* const bgra = std::find_if(capabilities.formats,
                                    capabilities.formats + capabilities.format_count,
                                    [](const rachis_surface_format& format) {
                                        return format.format == RACHIS_FORMAT_BGRA8_UNORM &&
                                               format.color_space == RACHIS_COLOR_SPACE_SRGB_NONLINEAR;
                                    });
    REQUIRE(bgra != capabilities.formats + capabilities.format_count);
    const bool copyDestination = (capabilities.usages & RACHIS_TEXTURE_COPY_DESTINATION) != 0;

    rachis_swapchain_desc swapchainDesc{};
    swapchainDesc.struct_size = sizeof(swapchainDesc);
    swapchainDesc.abi_version = RACHIS_ABI_VERSION;
    swapchainDesc.surface = surface;
    swapchainDesc.width = capabilities.preferred_width == 0 ? windowWidth : capabilities.preferred_width;
    swapchainDesc.height = capabilities.preferred_height == 0 ? windowHeight : capabilities.preferred_height;
    swapchainDesc.image_count =
        std::max(capabilities.min_image_count,
                 std::min(capabilities.max_image_count == 0 ? 3U : capabilities.max_image_count, 3U));
    swapchainDesc.format = RACHIS_FORMAT_BGRA8_UNORM;
    swapchainDesc.color_space = RACHIS_COLOR_SPACE_SRGB_NONLINEAR;
    swapchainDesc.usage = RACHIS_TEXTURE_RENDER_TARGET | (copyDestination ? RACHIS_TEXTURE_COPY_DESTINATION : 0U);
    swapchainDesc.present_mode = RACHIS_PRESENT_FIFO;
    rachis_handle swapchain = 0;
    requireOk(rachis_swapchain_create(device.get(), &swapchainDesc, &swapchain, &error), error);

    rachis_command_pool_desc poolDesc{};
    poolDesc.struct_size = sizeof(poolDesc);
    poolDesc.abi_version = RACHIS_ABI_VERSION;
    poolDesc.queue_type = RACHIS_QUEUE_GRAPHICS;
    poolDesc.max_command_buffers = 4;
    rachis_handle pool = 0;
    requireOk(rachis_command_pool_create(device.get(), &poolDesc, &pool, &error), error);

    auto image = acquireImage(device.get(), swapchain, error);
    CHECK(image.width == swapchainDesc.width);
    CHECK(image.height == swapchainDesc.height);
    rachis_handle lease = 0;
    CHECK(rachis_texture_lease_acquire(device.get(), image.texture, &lease, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    rachis_texture_view_desc viewDesc{};
    viewDesc.struct_size = sizeof(viewDesc);
    viewDesc.abi_version = RACHIS_ABI_VERSION;
    viewDesc.texture = image.texture;
    viewDesc.usage = RACHIS_VIEW_USAGE_RENDER_TARGET;
    viewDesc.dimension = RACHIS_VIEW_2D;
    viewDesc.mip_level_count = 1;
    viewDesc.array_layer_count = 1;
    rachis_handle view = 0;
    CHECK(rachis_texture_view_create(device.get(), &viewDesc, &view, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_destroy(device.get(), image.texture, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_view_destroy(device.get(), image.render_target_view, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);

    rachis_handle command = 0;
    requireOk(rachis_command_acquire(device.get(), pool, &command, &error), error);
    requireOk(recordBarrier(device.get(),
                            command,
                            imageBarrier(image.texture,
                                         image.initial_stages,
                                         image.initial_access,
                                         image.initial_layout,
                                         RACHIS_STAGE_RENDER_TARGET,
                                         RACHIS_ACCESS_RENDER_TARGET,
                                         RACHIS_LAYOUT_RENDER_TARGET),
                            error),
              error);
    rachis_color_attachment attachment{};
    attachment.view = image.render_target_view;
    attachment.load_op = RACHIS_LOAD_OP_CLEAR;
    attachment.store_op = RACHIS_STORE_OP_STORE;
    attachment.clear_color[0] = 0.1F;
    attachment.clear_color[1] = 0.4F;
    attachment.clear_color[2] = 0.7F;
    attachment.clear_color[3] = 1.0F;
    rachis_rendering_desc rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.abi_version = RACHIS_ABI_VERSION;
    rendering.color_attachments = &attachment;
    rendering.color_attachment_count = 1;
    requireOk(rachis_command_begin_rendering(device.get(), command, &rendering, &error), error);
    requireOk(rachis_command_end_rendering(device.get(), command, &error), error);
    requireOk(recordBarrier(device.get(),
                            command,
                            imageBarrier(image.texture,
                                         RACHIS_STAGE_RENDER_TARGET,
                                         RACHIS_ACCESS_RENDER_TARGET,
                                         RACHIS_LAYOUT_RENDER_TARGET,
                                         image.required_final_stages,
                                         image.required_final_access,
                                         image.required_final_layout),
                            error),
              error);
    submitAndPresent(device.get(), command, swapchain, error);

    requireOk(rachis_command_acquire(device.get(), pool, &command, &error), error);
    CHECK(recordBarrier(device.get(),
                        command,
                        imageBarrier(image.texture,
                                     RACHIS_STAGE_NONE,
                                     RACHIS_ACCESS_NONE,
                                     RACHIS_LAYOUT_PRESENT,
                                     RACHIS_STAGE_COPY,
                                     RACHIS_ACCESS_COPY_DESTINATION,
                                     RACHIS_LAYOUT_COPY_DESTINATION),
                        error) == RACHIS_STATUS_INVALID_ARGUMENT);

    if (copyDestination)
    {
        image = acquireImage(device.get(), swapchain, error);
        rachis_buffer_texture_region region{};
        region.aspect = RACHIS_ASPECT_COLOR;
        region.width = image.width;
        region.height = image.height;
        region.depth = 1;
        rachis_texture_transfer_layout layout{};
        layout.struct_size = sizeof(layout);
        requireOk(rachis_texture_query_transfer_layout(device.get(), image.texture, &region, &layout, &error), error);
        rachis_buffer_desc bufferDesc{};
        bufferDesc.struct_size = sizeof(bufferDesc);
        bufferDesc.abi_version = RACHIS_ABI_VERSION;
        bufferDesc.size = layout.required_buffer_size;
        bufferDesc.usage = RACHIS_BUFFER_COPY_SOURCE;
        bufferDesc.memory_type = RACHIS_MEMORY_UPLOAD;
        rachis_handle upload = 0;
        requireOk(rachis_buffer_create(device.get(), &bufferDesc, &upload, &error), error);
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(layout.required_buffer_size), 0x80);
        requireOk(rachis_buffer_write(device.get(), upload, 0, pixels.data(), pixels.size(), &error), error);

        requireOk(recordBarrier(device.get(),
                                command,
                                imageBarrier(image.texture,
                                             image.initial_stages,
                                             image.initial_access,
                                             image.initial_layout,
                                             RACHIS_STAGE_COPY,
                                             RACHIS_ACCESS_COPY_DESTINATION,
                                             RACHIS_LAYOUT_COPY_DESTINATION),
                                error),
                  error);
        region.bytes_per_row = layout.bytes_per_row;
        rachis_buffer_texture_copy_desc copy{};
        copy.struct_size = sizeof(copy);
        copy.abi_version = RACHIS_ABI_VERSION;
        copy.buffer = upload;
        copy.texture = image.texture;
        copy.regions = &region;
        copy.region_count = 1;
        requireOk(rachis_command_copy_buffer_to_texture(device.get(), command, &copy, &error), error);
        requireOk(recordBarrier(device.get(),
                                command,
                                imageBarrier(image.texture,
                                             RACHIS_STAGE_COPY,
                                             RACHIS_ACCESS_COPY_DESTINATION,
                                             RACHIS_LAYOUT_COPY_DESTINATION,
                                             image.required_final_stages,
                                             image.required_final_access,
                                             image.required_final_layout),
                                error),
                  error);
        submitAndPresent(device.get(), command, swapchain, error);
        requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
    }
    else
    {
        WARN("The surface does not report copy-destination usage; the copy frame was skipped.");
        requireOk(rachis_command_discard(device.get(), command, &error), error);
    }

    image = acquireImage(device.get(), swapchain, error);
    rachis_buffer_texture_region region{};
    region.aspect = RACHIS_ASPECT_COLOR;
    region.width = image.width;
    region.height = image.height;
    region.depth = 1;
    rachis_texture_transfer_layout layout{};
    layout.struct_size = sizeof(layout);
    requireOk(rachis_texture_query_transfer_layout(device.get(), image.texture, &region, &layout, &error), error);
    requireOk(rachis_swapchain_cancel(device.get(), swapchain, &error), error);
    CHECK(rachis_texture_query_transfer_layout(device.get(), image.texture, &region, &layout, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);

    REQUIRE(SetWindowPos(window.get(),
                         nullptr,
                         0,
                         0,
                         static_cast<int>(windowWidth + 80),
                         static_cast<int>(windowHeight + 60),
                         SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE) != FALSE);
    requireOk(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error), error);
    REQUIRE(capabilities.preferred_width != 0);
    REQUIRE(capabilities.preferred_height != 0);
    CHECK(capabilities.preferred_width != swapchainDesc.width);
    requireOk(rachis_swapchain_resize(
                  device.get(), swapchain, capabilities.preferred_width, capabilities.preferred_height, &error),
              error);
    image = acquireImage(device.get(), swapchain, error);
    CHECK(image.width == capabilities.preferred_width);
    CHECK(image.height == capabilities.preferred_height);
    requireOk(rachis_swapchain_cancel(device.get(), swapchain, &error), error);

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_swapchain_destroy(device.get(), swapchain, &error), error);
    requireOk(rachis_surface_destroy(device.get(), surface, &error), error);
    rachis_diagnostic_counts counts{};
    counts.struct_size = sizeof(counts);
    requireOk(rachis_device_diagnostic_counts(device.get(), &counts, &error), error);
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

TEST_CASE("C ABI keeps a surface while its swapchain is alive and zeroes unused capability formats",
          "[size:medium][requires:InteractiveDesktop]" RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][presentation]")
{
    const TestWindow window;
    REQUIRE(window.get() != nullptr);
    auto error = makeError();
    rachis_device_desc deviceDesc{};
    deviceDesc.struct_size = sizeof(deviceDesc);
    deviceDesc.abi_version = RACHIS_ABI_VERSION;
    deviceDesc.backend = RACHIS_SMOKE_BACKEND;
    deviceDesc.adapter_preference = RACHIS_SMOKE_ADAPTER;
    deviceDesc.validation_level = RACHIS_VALIDATION_STANDARD;
    deviceDesc.application_name = "Rachis C ABI surface lifetime";
    rachis_device* native = nullptr;
    requireOk(rachis_device_create(&deviceDesc, &native, &error), error);
    const Device device{native, &rachis_device_destroy};

    rachis_surface_desc surfaceDesc{};
    surfaceDesc.struct_size = sizeof(surfaceDesc);
    surfaceDesc.abi_version = RACHIS_ABI_VERSION;
    surfaceDesc.kind = RACHIS_SURFACE_WIN32;
    surfaceDesc.window_handle = window.get();
    rachis_handle surface = 0;
    requireOk(rachis_surface_create(device.get(), &surfaceDesc, &surface, &error), error);

    rachis_surface_capabilities capabilities;
    std::memset(&capabilities, 0xA5, sizeof(capabilities));
    capabilities.struct_size = sizeof(capabilities);
    requireOk(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error), error);
    REQUIRE(capabilities.format_count > 0);
    REQUIRE(capabilities.format_count <= RACHIS_MAX_SURFACE_FORMATS);
    for (auto index = capabilities.format_count; index < RACHIS_MAX_SURFACE_FORMATS; ++index)
    {
        CAPTURE(index);
        CHECK(capabilities.formats[index].format == RACHIS_FORMAT_UNKNOWN);
        CHECK(capabilities.formats[index].color_space == 0U);
    }

    rachis_swapchain_desc swapchainDesc{};
    swapchainDesc.struct_size = sizeof(swapchainDesc);
    swapchainDesc.abi_version = RACHIS_ABI_VERSION;
    swapchainDesc.surface = surface;
    swapchainDesc.width = capabilities.preferred_width == 0 ? windowWidth : capabilities.preferred_width;
    swapchainDesc.height = capabilities.preferred_height == 0 ? windowHeight : capabilities.preferred_height;
    swapchainDesc.image_count =
        std::max(capabilities.min_image_count,
                 std::min(capabilities.max_image_count == 0 ? 3U : capabilities.max_image_count, 3U));
    swapchainDesc.format = RACHIS_FORMAT_BGRA8_UNORM;
    swapchainDesc.color_space = RACHIS_COLOR_SPACE_SRGB_NONLINEAR;
    swapchainDesc.usage = RACHIS_TEXTURE_RENDER_TARGET;
    swapchainDesc.present_mode = RACHIS_PRESENT_FIFO;
    rachis_handle swapchain = 0;
    requireOk(rachis_swapchain_create(device.get(), &swapchainDesc, &swapchain, &error), error);

    error = makeError();
    CHECK(rachis_surface_destroy(device.get(), surface, &error) == RACHIS_STATUS_INVALID_STATE);
    CHECK(error.code == RACHIS_STATUS_INVALID_STATE);
    requireOk(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error), error);

    requireOk(rachis_swapchain_destroy(device.get(), swapchain, &error), error);
    requireOk(rachis_surface_destroy(device.get(), surface, &error), error);
    CHECK(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    rachis_diagnostic_counts counts{};
    counts.struct_size = sizeof(counts);
    requireOk(rachis_device_diagnostic_counts(device.get(), &counts, &error), error);
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

namespace
{
    constexpr bool presentationProgressSupported = RACHIS_SMOKE_BACKEND == RACHIS_BACKEND_D3D12;

    [[nodiscard]] rachis_swapchain_presentation_progress queryProgress(rachis_device* device,
                                                                       rachis_handle swapchain,
                                                                       rachis_error& error)
    {
        rachis_swapchain_presentation_progress progress{};
        progress.struct_size = sizeof(progress);
        requireOk(rachis_swapchain_query_presentation_progress(device, swapchain, &progress, &error), error);
        CHECK(progress.struct_size == sizeof(progress));
        CHECK(progress.supported <= 1U);
        CHECK(progress.displayed_sequence <= progress.submitted_sequence);
        if (progress.supported == 0)
        {
            CHECK(progress.generation == 0);
            CHECK(progress.submitted_sequence == 0);
            CHECK(progress.displayed_sequence == 0);
        }
        return progress;
    }

    void recordClear(rachis_device* device,
                     rachis_handle command,
                     const rachis_swapchain_image& image,
                     rachis_error& error)
    {
        requireOk(recordBarrier(device,
                                command,
                                imageBarrier(image.texture,
                                             image.initial_stages,
                                             image.initial_access,
                                             image.initial_layout,
                                             RACHIS_STAGE_RENDER_TARGET,
                                             RACHIS_ACCESS_RENDER_TARGET,
                                             RACHIS_LAYOUT_RENDER_TARGET),
                                error),
                  error);
        rachis_color_attachment attachment{};
        attachment.view = image.render_target_view;
        attachment.load_op = RACHIS_LOAD_OP_CLEAR;
        attachment.store_op = RACHIS_STORE_OP_STORE;
        attachment.clear_color[0] = 0.7F;
        attachment.clear_color[1] = 0.4F;
        attachment.clear_color[2] = 0.1F;
        attachment.clear_color[3] = 1.0F;
        rachis_rendering_desc rendering{};
        rendering.struct_size = sizeof(rendering);
        rendering.abi_version = RACHIS_ABI_VERSION;
        rendering.color_attachments = &attachment;
        rendering.color_attachment_count = 1;
        requireOk(rachis_command_begin_rendering(device, command, &rendering, &error), error);
        requireOk(rachis_command_end_rendering(device, command, &error), error);
        requireOk(recordBarrier(device,
                                command,
                                imageBarrier(image.texture,
                                             RACHIS_STAGE_RENDER_TARGET,
                                             RACHIS_ACCESS_RENDER_TARGET,
                                             RACHIS_LAYOUT_RENDER_TARGET,
                                             image.required_final_stages,
                                             image.required_final_access,
                                             image.required_final_layout),
                                error),
                  error);
    }

    void clearAndPresent(rachis_device* device, rachis_handle pool, rachis_handle swapchain, rachis_error& error)
    {
        const auto image = acquireImage(device, swapchain, error);
        rachis_handle command = 0;
        requireOk(rachis_command_acquire(device, pool, &command, &error), error);
        recordClear(device, command, image, error);
        submitAndPresent(device, command, swapchain, error);
    }

    [[nodiscard]] rachis_status queryImageLayout(rachis_device* device,
                                                 const rachis_swapchain_image& image,
                                                 rachis_error& error)
    {
        rachis_buffer_texture_region region{};
        region.aspect = RACHIS_ASPECT_COLOR;
        region.width = image.width;
        region.height = image.height;
        region.depth = 1;
        rachis_texture_transfer_layout layout{};
        layout.struct_size = sizeof(layout);
        return rachis_texture_query_transfer_layout(device, image.texture, &region, &layout, &error);
    }
} // namespace

TEST_CASE("C ABI reports display progress apart from GPU completion",
          "[size:medium][requires:InteractiveDesktop]" RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][presentation]")
{
    const TestWindow window;
    REQUIRE(window.get() != nullptr);
    auto error = makeError();
    rachis_device_desc deviceDesc{};
    deviceDesc.struct_size = sizeof(deviceDesc);
    deviceDesc.abi_version = RACHIS_ABI_VERSION;
    deviceDesc.backend = RACHIS_SMOKE_BACKEND;
    deviceDesc.adapter_preference = RACHIS_SMOKE_ADAPTER;
    deviceDesc.validation_level = RACHIS_VALIDATION_STANDARD;
    deviceDesc.application_name = "Rachis C ABI presentation progress";
    rachis_device* native = nullptr;
    requireOk(rachis_device_create(&deviceDesc, &native, &error), error);
    const Device device{native, &rachis_device_destroy};

    rachis_surface_desc surfaceDesc{};
    surfaceDesc.struct_size = sizeof(surfaceDesc);
    surfaceDesc.abi_version = RACHIS_ABI_VERSION;
    surfaceDesc.kind = RACHIS_SURFACE_WIN32;
    surfaceDesc.window_handle = window.get();
    rachis_handle surface = 0;
    requireOk(rachis_surface_create(device.get(), &surfaceDesc, &surface, &error), error);
    rachis_surface_capabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    requireOk(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error), error);

    rachis_swapchain_desc swapchainDesc{};
    swapchainDesc.struct_size = sizeof(swapchainDesc);
    swapchainDesc.abi_version = RACHIS_ABI_VERSION;
    swapchainDesc.surface = surface;
    swapchainDesc.width = capabilities.preferred_width == 0 ? windowWidth : capabilities.preferred_width;
    swapchainDesc.height = capabilities.preferred_height == 0 ? windowHeight : capabilities.preferred_height;
    swapchainDesc.image_count =
        std::max(capabilities.min_image_count,
                 std::min(capabilities.max_image_count == 0 ? 3U : capabilities.max_image_count, 3U));
    swapchainDesc.format = RACHIS_FORMAT_BGRA8_UNORM;
    swapchainDesc.color_space = RACHIS_COLOR_SPACE_SRGB_NONLINEAR;
    swapchainDesc.usage = RACHIS_TEXTURE_RENDER_TARGET;
    swapchainDesc.present_mode = RACHIS_PRESENT_FIFO;
    rachis_handle swapchain = 0;
    requireOk(rachis_swapchain_create(device.get(), &swapchainDesc, &swapchain, &error), error);

    rachis_swapchain_presentation_progress progress{};
    progress.struct_size = sizeof(progress) - 1;
    CHECK(rachis_swapchain_query_presentation_progress(device.get(), swapchain, &progress, &error) ==
          RACHIS_STATUS_ABI_MISMATCH);
    CHECK(rachis_swapchain_query_presentation_progress(device.get(), swapchain, nullptr, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    progress.struct_size = sizeof(progress);
    CHECK(rachis_swapchain_query_presentation_progress(device.get(), surface, &progress, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);

    progress = queryProgress(device.get(), swapchain, error);
    CHECK(progress.supported == (presentationProgressSupported ? 1U : 0U));
    const auto generation = progress.generation;
    if (presentationProgressSupported)
    {
        CHECK(generation != 0);
        CHECK(progress.submitted_sequence == 0);
        CHECK(progress.displayed_sequence == 0);
    }

    rachis_command_pool_desc poolDesc{};
    poolDesc.struct_size = sizeof(poolDesc);
    poolDesc.abi_version = RACHIS_ABI_VERSION;
    poolDesc.queue_type = RACHIS_QUEUE_GRAPHICS;
    poolDesc.max_command_buffers = 2;
    rachis_handle pool = 0;
    requireOk(rachis_command_pool_create(device.get(), &poolDesc, &pool, &error), error);

    constexpr std::uint64_t frames = 2;
    for (std::uint64_t frame = 1; frame <= frames; ++frame)
    {
        clearAndPresent(device.get(), pool, swapchain, error);
        progress = queryProgress(device.get(), swapchain, error);
        if (presentationProgressSupported)
        {
            CHECK(progress.supported == 1U);
            CHECK(progress.generation == generation);
            CHECK(progress.submitted_sequence == frame);
        }
        else
        {
            CHECK(progress.supported == 0U);
        }
    }

    const auto deadline = GetTickCount64() + 2000;
    while (progress.displayed_sequence < progress.submitted_sequence && GetTickCount64() < deadline)
    {
        MSG message{};
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE) != FALSE)
        {
            static_cast<void>(TranslateMessage(&message));
            static_cast<void>(DispatchMessageW(&message));
        }
        Sleep(2);
        const auto previous = progress.displayed_sequence;
        progress = queryProgress(device.get(), swapchain, error);
        CHECK(progress.displayed_sequence >= previous);
    }
    if (presentationProgressSupported)
    {
        CHECK(progress.generation == generation);
        CHECK(progress.submitted_sequence == frames);
        CHECK(progress.displayed_sequence == frames);
    }
    else
    {
        CHECK(progress.supported == 0U);
    }

    requireOk(
        rachis_swapchain_resize(device.get(), swapchain, swapchainDesc.width + 16, swapchainDesc.height + 8, &error),
        error);
    progress = queryProgress(device.get(), swapchain, error);
    if (presentationProgressSupported)
    {
        CHECK(progress.supported == 1U);
        CHECK(progress.generation > generation);
        CHECK(progress.submitted_sequence == 0);
        CHECK(progress.displayed_sequence == 0);
    }

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_swapchain_destroy(device.get(), swapchain, &error), error);
    progress.struct_size = sizeof(progress);
    CHECK(rachis_swapchain_query_presentation_progress(device.get(), swapchain, &progress, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_surface_destroy(device.get(), surface, &error), error);
    rachis_diagnostic_counts counts{};
    counts.struct_size = sizeof(counts);
    requireOk(rachis_device_diagnostic_counts(device.get(), &counts, &error), error);
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

TEST_CASE("C ABI swapchain cancel keeps borrowed handles while a recorded command uses the image",
          "[size:medium][requires:InteractiveDesktop]" RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][presentation]")
{
    const TestWindow window;
    REQUIRE(window.get() != nullptr);
    auto error = makeError();
    rachis_device_desc deviceDesc{};
    deviceDesc.struct_size = sizeof(deviceDesc);
    deviceDesc.abi_version = RACHIS_ABI_VERSION;
    deviceDesc.backend = RACHIS_SMOKE_BACKEND;
    deviceDesc.adapter_preference = RACHIS_SMOKE_ADAPTER;
    deviceDesc.validation_level = RACHIS_VALIDATION_STANDARD;
    deviceDesc.application_name = "Rachis C ABI recorded cancel";
    rachis_device* native = nullptr;
    requireOk(rachis_device_create(&deviceDesc, &native, &error), error);
    const Device device{native, &rachis_device_destroy};

    rachis_surface_desc surfaceDesc{};
    surfaceDesc.struct_size = sizeof(surfaceDesc);
    surfaceDesc.abi_version = RACHIS_ABI_VERSION;
    surfaceDesc.kind = RACHIS_SURFACE_WIN32;
    surfaceDesc.window_handle = window.get();
    rachis_handle surface = 0;
    requireOk(rachis_surface_create(device.get(), &surfaceDesc, &surface, &error), error);
    rachis_surface_capabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    requireOk(rachis_surface_query_capabilities(device.get(), surface, &capabilities, &error), error);

    rachis_swapchain_desc swapchainDesc{};
    swapchainDesc.struct_size = sizeof(swapchainDesc);
    swapchainDesc.abi_version = RACHIS_ABI_VERSION;
    swapchainDesc.surface = surface;
    swapchainDesc.width = capabilities.preferred_width == 0 ? windowWidth : capabilities.preferred_width;
    swapchainDesc.height = capabilities.preferred_height == 0 ? windowHeight : capabilities.preferred_height;
    swapchainDesc.image_count =
        std::max(capabilities.min_image_count,
                 std::min(capabilities.max_image_count == 0 ? 3U : capabilities.max_image_count, 3U));
    swapchainDesc.format = RACHIS_FORMAT_BGRA8_UNORM;
    swapchainDesc.color_space = RACHIS_COLOR_SPACE_SRGB_NONLINEAR;
    swapchainDesc.usage = RACHIS_TEXTURE_RENDER_TARGET;
    swapchainDesc.present_mode = RACHIS_PRESENT_FIFO;
    rachis_handle swapchain = 0;
    requireOk(rachis_swapchain_create(device.get(), &swapchainDesc, &swapchain, &error), error);

    rachis_command_pool_desc poolDesc{};
    poolDesc.struct_size = sizeof(poolDesc);
    poolDesc.abi_version = RACHIS_ABI_VERSION;
    poolDesc.queue_type = RACHIS_QUEUE_GRAPHICS;
    poolDesc.max_command_buffers = 2;
    rachis_handle pool = 0;
    requireOk(rachis_command_pool_create(device.get(), &poolDesc, &pool, &error), error);

    const auto image = acquireImage(device.get(), swapchain, error);
    rachis_handle command = 0;
    requireOk(rachis_command_acquire(device.get(), pool, &command, &error), error);
    recordClear(device.get(), command, image, error);
    REQUIRE(rachis_swapchain_cancel(device.get(), swapchain, &error) == RACHIS_STATUS_INVALID_STATE);
    requireOk(queryImageLayout(device.get(), image, error), error);
    rachis_handle probe = 0;
    requireOk(rachis_command_acquire(device.get(), pool, &probe, &error), error);
    rachis_color_attachment attachment{};
    attachment.view = image.render_target_view;
    attachment.load_op = RACHIS_LOAD_OP_CLEAR;
    attachment.store_op = RACHIS_STORE_OP_DISCARD;
    rachis_rendering_desc rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.abi_version = RACHIS_ABI_VERSION;
    rendering.color_attachments = &attachment;
    rendering.color_attachment_count = 1;
    requireOk(rachis_command_begin_rendering(device.get(), probe, &rendering, &error), error);
    requireOk(rachis_command_end_rendering(device.get(), probe, &error), error);
    requireOk(rachis_command_discard(device.get(), probe, &error), error);
    CHECK(rachis_swapchain_cancel(device.get(), swapchain, &error) == RACHIS_STATUS_INVALID_STATE);
    submitAndPresent(device.get(), command, swapchain, error);
    CHECK(queryImageLayout(device.get(), image, error) == RACHIS_STATUS_INVALID_ARGUMENT);

    const auto unused = acquireImage(device.get(), swapchain, error);
    requireOk(rachis_swapchain_cancel(device.get(), swapchain, &error), error);
    CHECK(queryImageLayout(device.get(), unused, error) == RACHIS_STATUS_INVALID_ARGUMENT);

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_swapchain_destroy(device.get(), swapchain, &error), error);
    requireOk(rachis_surface_destroy(device.get(), surface, &error), error);
    rachis_diagnostic_counts counts{};
    counts.struct_size = sizeof(counts);
    requireOk(rachis_device_diagnostic_counts(device.get(), &counts, &error), error);
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}
