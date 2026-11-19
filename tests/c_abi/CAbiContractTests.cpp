#include "Rachis/rachis.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef RACHIS_CABI_D3D12_DEVICE_REMOVAL
#include <d3d12.h>
#include <dxgi1_6.h>
#endif

#ifndef RACHIS_CABI_DEVICE_SIZE
#error RACHIS_CABI_DEVICE_SIZE must name the size tag of cases that create a device.
#endif
#ifndef RACHIS_CABI_DEVICE_REQUIRES
#error RACHIS_CABI_DEVICE_REQUIRES must name the capability tags of cases that create a device.
#endif

namespace
{
    using Device = std::unique_ptr<rachis_device, decltype(&rachis_device_destroy)>;

    constexpr std::uint32_t textureSize = 4;
    constexpr std::uint32_t valueCount = 64;

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

    struct DiagnosticCapture final
    {
        std::mutex mutex;
        std::vector<std::uint32_t> severities;
        std::string messages;
    };

    void captureDiagnostic(std::uint32_t severity, std::uint32_t, const char* message, void* userData)
    {
        auto& capture = *static_cast<DiagnosticCapture*>(userData);
        const std::scoped_lock lock(capture.mutex);
        capture.severities.push_back(severity);
        capture.messages.append(message == nullptr ? "" : message).append("\n");
    }

    [[nodiscard]] Device createDevice(rachis_error& error,
                                      std::uint32_t validation = RACHIS_VALIDATION_DISABLED,
                                      DiagnosticCapture* capture = nullptr)
    {
        rachis_device_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.backend = RACHIS_SMOKE_BACKEND;
        desc.adapter_preference = RACHIS_SMOKE_ADAPTER;
        desc.validation_level = validation;
        desc.application_name = "Rachis C ABI contract";
        if (capture != nullptr)
        {
            desc.diagnostic_callback = &captureDiagnostic;
            desc.diagnostic_user_data = capture;
        }
        rachis_device* native = nullptr;
        const auto status = rachis_device_create(&desc, &native, &error);
        if (status == RACHIS_STATUS_BACKEND_UNAVAILABLE)
        {
            SKIP(error.message);
        }
        requireOk(status, error);
        return Device{native, &rachis_device_destroy};
    }

    [[nodiscard]] rachis_handle createBuffer(
        rachis_device* device, std::uint64_t size, std::uint32_t usage, std::uint32_t memoryType, rachis_error& error)
    {
        rachis_buffer_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.size = size;
        desc.usage = usage;
        desc.memory_type = memoryType;
        rachis_handle buffer = 0;
        requireOk(rachis_buffer_create(device, &desc, &buffer, &error), error);
        return buffer;
    }

    [[nodiscard]] rachis_handle createTexture(rachis_device* device,
                                              std::uint32_t format,
                                              std::uint32_t width,
                                              std::uint32_t height,
                                              std::uint32_t usage,
                                              rachis_error& error)
    {
        rachis_texture_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.dimension = RACHIS_TEXTURE_2D;
        desc.format = format;
        desc.usage = usage;
        desc.width = width;
        desc.height = height;
        desc.depth_or_array_layers = 1;
        desc.mip_levels = 1;
        desc.sample_count = 1;
        desc.label = "C ABI texture";
        rachis_handle texture = 0;
        requireOk(rachis_texture_create(device, &desc, &texture, &error), error);
        return texture;
    }

    [[nodiscard]] rachis_texture_view_desc renderTargetViewDesc(rachis_handle texture) noexcept
    {
        rachis_texture_view_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.texture = texture;
        desc.usage = RACHIS_VIEW_USAGE_RENDER_TARGET;
        desc.dimension = RACHIS_VIEW_2D;
        desc.format = RACHIS_FORMAT_UNKNOWN;
        desc.aspect = RACHIS_ASPECT_COLOR;
        desc.mip_level_count = 1;
        desc.array_layer_count = 1;
        return desc;
    }

    [[nodiscard]] rachis_handle acquireCommand(rachis_device* device,
                                               std::uint32_t queueType,
                                               rachis_handle& pool,
                                               rachis_error& error)
    {
        rachis_command_pool_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.queue_type = queueType;
        desc.max_command_buffers = 2;
        requireOk(rachis_command_pool_create(device, &desc, &pool, &error), error);
        rachis_handle command = 0;
        requireOk(rachis_command_acquire(device, pool, &command, &error), error);
        return command;
    }

    [[nodiscard]] rachis_texture_barrier textureBarrier(rachis_handle texture,
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

    [[nodiscard]] rachis_buffer_barrier bufferBarrier(rachis_handle buffer,
                                                      std::uint32_t stagesBefore,
                                                      std::uint32_t accessBefore,
                                                      std::uint32_t stagesAfter,
                                                      std::uint32_t accessAfter) noexcept
    {
        rachis_buffer_barrier barrier{};
        barrier.buffer = buffer;
        barrier.stages_before = stagesBefore;
        barrier.stages_after = stagesAfter;
        barrier.access_before = accessBefore;
        barrier.access_after = accessAfter;
        barrier.size = RACHIS_WHOLE_SIZE;
        return barrier;
    }

    [[nodiscard]] rachis_status recordBarriers(rachis_device* device,
                                               rachis_handle command,
                                               const rachis_buffer_barrier* buffers,
                                               std::uint32_t bufferCount,
                                               const rachis_texture_barrier* textures,
                                               std::uint32_t textureCount,
                                               rachis_error& error)
    {
        rachis_barrier_batch batch{};
        batch.struct_size = sizeof(batch);
        batch.abi_version = RACHIS_ABI_VERSION;
        batch.buffer_barriers = buffers;
        batch.buffer_barrier_count = bufferCount;
        batch.texture_barriers = textures;
        batch.texture_barrier_count = textureCount;
        return rachis_command_barriers(device, command, &batch, &error);
    }

    [[nodiscard]] rachis_buffer_texture_region wholeTextureRegion(std::uint32_t width, std::uint32_t height) noexcept
    {
        rachis_buffer_texture_region region{};
        region.aspect = RACHIS_ASPECT_COLOR;
        region.width = width;
        region.height = height;
        region.depth = 1;
        return region;
    }

    [[nodiscard]] rachis_status copyBufferTexture(rachis_device* device,
                                                  rachis_handle command,
                                                  rachis_handle buffer,
                                                  rachis_handle texture,
                                                  const rachis_buffer_texture_region& region,
                                                  bool toTexture,
                                                  rachis_error& error)
    {
        rachis_buffer_texture_copy_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.buffer = buffer;
        desc.texture = texture;
        desc.regions = &region;
        desc.region_count = 1;
        return toTexture ? rachis_command_copy_buffer_to_texture(device, command, &desc, &error)
                         : rachis_command_copy_texture_to_buffer(device, command, &desc, &error);
    }

    void submitAndWait(rachis_device* device, rachis_handle command, rachis_error& error)
    {
        rachis_handle timeline = 0;
        requireOk(rachis_command_submit(device, command, &timeline, &error), error);
        requireOk(rachis_timeline_wait(device, timeline, &error), error);
        requireOk(rachis_timeline_release(device, timeline, &error), error);
    }

    void requireNoValidationErrors(rachis_device* device, rachis_error& error)
    {
        rachis_diagnostic_counts counts{};
        counts.struct_size = sizeof(counts);
        requireOk(rachis_device_diagnostic_counts(device, &counts, &error), error);
        CHECK(counts.errors == 0);
        CHECK(counts.corruptions == 0);
    }

    [[nodiscard]] bool nearByte(std::uint8_t actual, std::uint8_t expected) noexcept
    {
        return actual + 1 >= expected && actual <= expected + 1;
    }

#ifdef RACHIS_CABI_COMPUTE_SHADER_PATH
    [[nodiscard]] std::vector<char> loadShader()
    {
        std::ifstream input{RACHIS_CABI_COMPUTE_SHADER_PATH, std::ios::binary};
        return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    }
#endif
} // namespace

#ifdef RACHIS_CABI_GRAPHICS_VERTEX_PATH
TEST_CASE("C ABI graphics pipeline draws through the shared Rachis module",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][graphics]")
{
    const auto load = [](const char* path) {
        std::ifstream input{path, std::ios::binary};
        return std::vector<char>{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    };
    const auto vertexBytes = load(RACHIS_CABI_GRAPHICS_VERTEX_PATH);
    const auto pixelBytes = load(RACHIS_CABI_GRAPHICS_PIXEL_PATH);
    REQUIRE_FALSE(vertexBytes.empty());
    REQUIRE_FALSE(pixelBytes.empty());

    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    rachis_shader_module_desc shaderDesc{};
    shaderDesc.struct_size = sizeof(shaderDesc);
    shaderDesc.abi_version = RACHIS_ABI_VERSION;
    shaderDesc.stage = RACHIS_SHADER_VERTEX;
    shaderDesc.bytecode = vertexBytes.data();
    shaderDesc.bytecode_size = vertexBytes.size();
    shaderDesc.entry_point = "vertexMain";
    rachis_handle vertex = 0;
    requireOk(rachis_shader_module_create(device.get(), &shaderDesc, &vertex, &error), error);
    shaderDesc.stage = RACHIS_SHADER_PIXEL;
    shaderDesc.bytecode = pixelBytes.data();
    shaderDesc.bytecode_size = pixelBytes.size();
    shaderDesc.entry_point = "pixelMain";
    const rachis_shader_push_constant_requirement pushRequirement{0, 16};
    shaderDesc.push_constants = &pushRequirement;
    shaderDesc.push_constant_count = 1;
    rachis_handle pixel = 0;
    requireOk(rachis_shader_module_create(device.get(), &shaderDesc, &pixel, &error), error);

    rachis_pipeline_layout_desc layoutDesc{};
    layoutDesc.struct_size = sizeof(layoutDesc);
    layoutDesc.abi_version = RACHIS_ABI_VERSION;
    const rachis_push_constant_range pushRange{RACHIS_SHADER_PIXEL, 0, 16};
    layoutDesc.push_constant_ranges = &pushRange;
    layoutDesc.push_constant_range_count = 1;
    rachis_handle pipelineLayout = 0;
    requireOk(rachis_pipeline_layout_create(device.get(), &layoutDesc, &pipelineLayout, &error), error);
    rachis_color_target target{};
    target.format = RACHIS_FORMAT_RGBA8_UNORM;
    target.write_mask = RACHIS_COLOR_WRITE_ALL;
    target.color_blend.source_factor = RACHIS_BLEND_ONE;
    target.alpha_blend.source_factor = RACHIS_BLEND_ONE;
    rachis_graphics_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.vertex_shader = vertex;
    pipelineDesc.pixel_shader = pixel;
    pipelineDesc.topology = RACHIS_TOPOLOGY_TRIANGLE_LIST;
    pipelineDesc.cull_mode = RACHIS_CULL_NONE;
    pipelineDesc.color_targets = &target;
    pipelineDesc.color_target_count = 1;
    rachis_depth_stencil_state depthState{RACHIS_FORMAT_D32_FLOAT, 1, 1, RACHIS_COMPARE_LESS};
    pipelineDesc.depth_stencil = &depthState;
    rachis_handle pipeline = 0;
    requireOk(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error), error);
    depthState.format = RACHIS_FORMAT_RGBA8_UNORM;
    rachis_handle invalidPipeline = 0;
    CHECK(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &invalidPipeline, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(invalidPipeline == 0);
    depthState.format = RACHIS_FORMAT_D32_FLOAT;
    depthState.depth_test_enabled = 0;
    CHECK(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &invalidPipeline, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(invalidPipeline == 0);
    depthState.depth_test_enabled = 1;

    const auto texture = createTexture(device.get(),
                                       RACHIS_FORMAT_RGBA8_UNORM,
                                       textureSize,
                                       textureSize,
                                       RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_COPY_SOURCE,
                                       error);
    const auto viewDesc = renderTargetViewDesc(texture);
    rachis_handle view = 0;
    requireOk(rachis_texture_view_create(device.get(), &viewDesc, &view, &error), error);
    const auto depthTexture = createTexture(
        device.get(), RACHIS_FORMAT_D32_FLOAT, textureSize, textureSize, RACHIS_TEXTURE_DEPTH_STENCIL, error);
    rachis_texture_view_desc depthViewDesc{};
    depthViewDesc.struct_size = sizeof(depthViewDesc);
    depthViewDesc.abi_version = RACHIS_ABI_VERSION;
    depthViewDesc.texture = depthTexture;
    depthViewDesc.usage = RACHIS_VIEW_USAGE_DEPTH_STENCIL;
    depthViewDesc.dimension = RACHIS_VIEW_2D;
    depthViewDesc.aspect = RACHIS_ASPECT_DEPTH;
    depthViewDesc.mip_level_count = 1;
    depthViewDesc.array_layer_count = 1;
    rachis_handle depthView = 0;
    requireOk(rachis_texture_view_create(device.get(), &depthViewDesc, &depthView, &error), error);
    const auto region = wholeTextureRegion(textureSize, textureSize);
    rachis_texture_transfer_layout transfer{};
    transfer.struct_size = sizeof(transfer);
    requireOk(rachis_texture_query_transfer_layout(device.get(), texture, &region, &transfer, &error), error);
    const auto readback = createBuffer(
        device.get(), transfer.required_buffer_size, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
    const auto toRender = textureBarrier(texture,
                                         RACHIS_STAGE_ALL,
                                         RACHIS_ACCESS_NONE,
                                         RACHIS_LAYOUT_UNDEFINED,
                                         RACHIS_STAGE_RENDER_TARGET,
                                         RACHIS_ACCESS_RENDER_TARGET,
                                         RACHIS_LAYOUT_RENDER_TARGET);
    requireOk(recordBarriers(device.get(), command, nullptr, 0, &toRender, 1, error), error);
    auto depthToRender = textureBarrier(depthTexture,
                                        RACHIS_STAGE_ALL,
                                        RACHIS_ACCESS_NONE,
                                        RACHIS_LAYOUT_UNDEFINED,
                                        RACHIS_STAGE_DEPTH_STENCIL,
                                        RACHIS_ACCESS_DEPTH_STENCIL_WRITE,
                                        RACHIS_LAYOUT_DEPTH_STENCIL_WRITE);
    depthToRender.aspect = RACHIS_ASPECT_DEPTH;
    requireOk(recordBarriers(device.get(), command, nullptr, 0, &depthToRender, 1, error), error);
    rachis_color_attachment attachment{};
    attachment.view = view;
    attachment.load_op = RACHIS_LOAD_OP_CLEAR;
    attachment.store_op = RACHIS_STORE_OP_STORE;
    attachment.clear_color[2] = 1.0F;
    attachment.clear_color[3] = 1.0F;
    rachis_rendering_desc rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.abi_version = RACHIS_ABI_VERSION;
    rendering.color_attachments = &attachment;
    rendering.color_attachment_count = 1;
    rachis_depth_stencil_attachment depthAttachment{};
    depthAttachment.view = depthView;
    depthAttachment.depth_load_op = RACHIS_LOAD_OP_CLEAR;
    depthAttachment.depth_store_op = RACHIS_STORE_OP_STORE;
    depthAttachment.depth_clear_value = 1.0F;
    rendering.depth_stencil_attachment = &depthAttachment;
    requireOk(rachis_command_begin_rendering(device.get(), command, &rendering, &error), error);
    const rachis_viewport viewport{0, 0, textureSize, textureSize, 0, 1};
    const rachis_scissor scissor{0, 0, textureSize, textureSize};
    requireOk(rachis_command_set_viewport(device.get(), command, &viewport, &error), error);
    requireOk(rachis_command_set_scissor(device.get(), command, &scissor, &error), error);
    requireOk(rachis_command_bind_graphics_pipeline(device.get(), command, pipeline, &error), error);
    const std::array<float, 4> tint{1.0F, 0.0F, 0.0F, 1.0F};
    requireOk(rachis_command_set_graphics_push_constants(
                  device.get(), command, RACHIS_SHADER_PIXEL, 0, tint.data(), sizeof(tint), &error),
              error);
    requireOk(rachis_command_draw(device.get(), command, 3, 1, 0, 0, &error), error);
    const std::array<float, 4> occludedTint{0.0F, 1.0F, 0.0F, 1.0F};
    requireOk(rachis_command_set_graphics_push_constants(
                  device.get(), command, RACHIS_SHADER_PIXEL, 0, occludedTint.data(), sizeof(occludedTint), &error),
              error);
    requireOk(rachis_command_draw(device.get(), command, 3, 1, 0, 0, &error), error);
    requireOk(rachis_command_end_rendering(device.get(), command, &error), error);
    const auto toCopy = textureBarrier(texture,
                                       RACHIS_STAGE_RENDER_TARGET,
                                       RACHIS_ACCESS_RENDER_TARGET,
                                       RACHIS_LAYOUT_RENDER_TARGET,
                                       RACHIS_STAGE_COPY,
                                       RACHIS_ACCESS_COPY_SOURCE,
                                       RACHIS_LAYOUT_COPY_SOURCE);
    requireOk(recordBarriers(device.get(), command, nullptr, 0, &toCopy, 1, error), error);
    auto copyRegion = region;
    copyRegion.bytes_per_row = transfer.bytes_per_row;
    requireOk(copyBufferTexture(device.get(), command, readback, texture, copyRegion, false, error), error);
    submitAndWait(device.get(), command, error);
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(transfer.required_buffer_size));
    requireOk(rachis_buffer_read(device.get(), readback, 0, pixels.data(), pixels.size(), &error), error);
    const auto* center = pixels.data() + transfer.bytes_per_row + 4;
    CHECK(center[0] == 255);
    CHECK(center[1] == 0);
    CHECK(center[2] == 0);
    CHECK(center[3] == 255);

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_texture_view_destroy(device.get(), view, &error), error);
    requireOk(rachis_texture_view_destroy(device.get(), depthView, &error), error);
    requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
    requireOk(rachis_texture_destroy(device.get(), depthTexture, &error), error);
    requireOk(rachis_pipeline_destroy(device.get(), pipeline, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), pixel, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), vertex, &error), error);
    requireNoValidationErrors(device.get(), error);
}

TEST_CASE("C ABI scissor rectangles are clipped to the render target on every backend",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][graphics][regression]")
{
    const auto load = [](const char* path) {
        std::ifstream input{path, std::ios::binary};
        return std::vector<char>{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    };
    const auto vertexBytes = load(RACHIS_CABI_GRAPHICS_VERTEX_PATH);
    const auto pixelBytes = load(RACHIS_CABI_GRAPHICS_PIXEL_PATH);
    REQUIRE_FALSE(vertexBytes.empty());
    REQUIRE_FALSE(pixelBytes.empty());

    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    rachis_shader_module_desc shaderDesc{};
    shaderDesc.struct_size = sizeof(shaderDesc);
    shaderDesc.abi_version = RACHIS_ABI_VERSION;
    shaderDesc.stage = RACHIS_SHADER_VERTEX;
    shaderDesc.bytecode = vertexBytes.data();
    shaderDesc.bytecode_size = vertexBytes.size();
    shaderDesc.entry_point = "vertexMain";
    rachis_handle vertex = 0;
    requireOk(rachis_shader_module_create(device.get(), &shaderDesc, &vertex, &error), error);
    shaderDesc.stage = RACHIS_SHADER_PIXEL;
    shaderDesc.bytecode = pixelBytes.data();
    shaderDesc.bytecode_size = pixelBytes.size();
    shaderDesc.entry_point = "pixelMain";
    const rachis_shader_push_constant_requirement pushRequirement{0, 16};
    shaderDesc.push_constants = &pushRequirement;
    shaderDesc.push_constant_count = 1;
    rachis_handle pixel = 0;
    requireOk(rachis_shader_module_create(device.get(), &shaderDesc, &pixel, &error), error);

    rachis_pipeline_layout_desc layoutDesc{};
    layoutDesc.struct_size = sizeof(layoutDesc);
    layoutDesc.abi_version = RACHIS_ABI_VERSION;
    const rachis_push_constant_range pushRange{RACHIS_SHADER_PIXEL, 0, 16};
    layoutDesc.push_constant_ranges = &pushRange;
    layoutDesc.push_constant_range_count = 1;
    rachis_handle pipelineLayout = 0;
    requireOk(rachis_pipeline_layout_create(device.get(), &layoutDesc, &pipelineLayout, &error), error);
    rachis_color_target target{};
    target.format = RACHIS_FORMAT_RGBA8_UNORM;
    target.write_mask = RACHIS_COLOR_WRITE_ALL;
    target.color_blend.source_factor = RACHIS_BLEND_ONE;
    target.alpha_blend.source_factor = RACHIS_BLEND_ONE;
    rachis_graphics_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.vertex_shader = vertex;
    pipelineDesc.pixel_shader = pixel;
    pipelineDesc.topology = RACHIS_TOPOLOGY_TRIANGLE_LIST;
    pipelineDesc.cull_mode = RACHIS_CULL_NONE;
    pipelineDesc.color_targets = &target;
    pipelineDesc.color_target_count = 1;
    rachis_handle pipeline = 0;
    requireOk(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error), error);

    constexpr std::uint32_t targetSize = 64;
    constexpr std::uint32_t largest = std::numeric_limits<std::int32_t>::max();
    const auto texture = createTexture(device.get(),
                                       RACHIS_FORMAT_RGBA8_UNORM,
                                       targetSize,
                                       targetSize,
                                       RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_COPY_SOURCE,
                                       error);
    const auto viewDesc = renderTargetViewDesc(texture);
    rachis_handle view = 0;
    requireOk(rachis_texture_view_create(device.get(), &viewDesc, &view, &error), error);
    const auto region = wholeTextureRegion(targetSize, targetSize);
    rachis_texture_transfer_layout transfer{};
    transfer.struct_size = sizeof(transfer);
    requireOk(rachis_texture_query_transfer_layout(device.get(), texture, &region, &transfer, &error), error);
    const auto readback = createBuffer(
        device.get(), transfer.required_buffer_size, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);

    struct ScissorCase final
    {
        const char* name = "";
        std::vector<rachis_scissor> scissors;
        std::uint32_t litPixels = 0;
    };
    const std::array cases{
        ScissorCase{"full", {{0, 0, targetSize, targetSize}}, 4096},
        ScissorCase{"inside", {{32, 32, 32, 32}}, 1024},
        ScissorCase{"oversized", {{0, 0, 4096, 4096}}, 4096},
        ScissorCase{"largest", {{0, 0, largest, largest}}, 4096},
        ScissorCase{"partly outside", {{32, 32, 64, 64}}, 1024},
        ScissorCase{"one past the edge", {{0, 0, 65, 64}}, 4096},
        ScissorCase{"wholly outside", {{64, 64, 16, 16}}, 0},
        ScissorCase{"small, then oversized", {{0, 0, 16, 16}, {0, 0, 4096, 4096}}, 4096},
        ScissorCase{"small, then wholly outside", {{0, 0, 16, 16}, {64, 64, 16, 16}}, 0},
        ScissorCase{"wholly outside, then small", {{64, 64, 16, 16}, {8, 8, 16, 16}}, 256},
    };
    bool firstCase = true;
    for (const auto& scissorCase : cases)
    {
        INFO(scissorCase.name);
        rachis_handle pool = 0;
        const auto command = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
        const auto toRender = firstCase ? textureBarrier(texture,
                                                         RACHIS_STAGE_ALL,
                                                         RACHIS_ACCESS_NONE,
                                                         RACHIS_LAYOUT_UNDEFINED,
                                                         RACHIS_STAGE_RENDER_TARGET,
                                                         RACHIS_ACCESS_RENDER_TARGET,
                                                         RACHIS_LAYOUT_RENDER_TARGET)
                                        : textureBarrier(texture,
                                                         RACHIS_STAGE_COPY,
                                                         RACHIS_ACCESS_COPY_SOURCE,
                                                         RACHIS_LAYOUT_COPY_SOURCE,
                                                         RACHIS_STAGE_RENDER_TARGET,
                                                         RACHIS_ACCESS_RENDER_TARGET,
                                                         RACHIS_LAYOUT_RENDER_TARGET);
        firstCase = false;
        requireOk(recordBarriers(device.get(), command, nullptr, 0, &toRender, 1, error), error);
        rachis_color_attachment attachment{};
        attachment.view = view;
        attachment.load_op = RACHIS_LOAD_OP_CLEAR;
        attachment.store_op = RACHIS_STORE_OP_STORE;
        attachment.clear_color[3] = 1.0F;
        rachis_rendering_desc rendering{};
        rendering.struct_size = sizeof(rendering);
        rendering.abi_version = RACHIS_ABI_VERSION;
        rendering.color_attachments = &attachment;
        rendering.color_attachment_count = 1;
        requireOk(rachis_command_begin_rendering(device.get(), command, &rendering, &error), error);
        const rachis_viewport viewport{0, 0, targetSize, targetSize, 0, 1};
        requireOk(rachis_command_set_viewport(device.get(), command, &viewport, &error), error);
        for (const auto& scissor : scissorCase.scissors)
        {
            requireOk(rachis_command_set_scissor(device.get(), command, &scissor, &error), error);
        }
        requireOk(rachis_command_bind_graphics_pipeline(device.get(), command, pipeline, &error), error);
        const std::array<float, 4> white{1.0F, 1.0F, 1.0F, 1.0F};
        requireOk(rachis_command_set_graphics_push_constants(
                      device.get(), command, RACHIS_SHADER_PIXEL, 0, white.data(), sizeof(white), &error),
                  error);
        requireOk(rachis_command_draw(device.get(), command, 3, 1, 0, 0, &error), error);
        requireOk(rachis_command_end_rendering(device.get(), command, &error), error);
        const auto toCopy = textureBarrier(texture,
                                           RACHIS_STAGE_RENDER_TARGET,
                                           RACHIS_ACCESS_RENDER_TARGET,
                                           RACHIS_LAYOUT_RENDER_TARGET,
                                           RACHIS_STAGE_COPY,
                                           RACHIS_ACCESS_COPY_SOURCE,
                                           RACHIS_LAYOUT_COPY_SOURCE);
        requireOk(recordBarriers(device.get(), command, nullptr, 0, &toCopy, 1, error), error);
        auto copyRegion = region;
        copyRegion.bytes_per_row = transfer.bytes_per_row;
        requireOk(copyBufferTexture(device.get(), command, readback, texture, copyRegion, false, error), error);
        submitAndWait(device.get(), command, error);
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(transfer.required_buffer_size));
        requireOk(rachis_buffer_read(device.get(), readback, 0, pixels.data(), pixels.size(), &error), error);
        std::uint32_t lit = 0;
        for (std::uint32_t y = 0; y < targetSize; ++y)
        {
            for (std::uint32_t x = 0; x < targetSize; ++x)
            {
                if (pixels[static_cast<std::size_t>(y) * transfer.bytes_per_row + static_cast<std::size_t>(x) * 4] >
                    200)
                {
                    ++lit;
                }
            }
        }
        CHECK(lit == scissorCase.litPixels);
        requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    }

    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_texture_view_destroy(device.get(), view, &error), error);
    requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
    requireOk(rachis_pipeline_destroy(device.get(), pipeline, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), pixel, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), vertex, &error), error);
    requireNoValidationErrors(device.get(), error);
}
#endif

TEST_CASE("C ABI texture and sampler bindings validate handles and publish a set",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][graphics]")
{
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    auto other = createDevice(error);
    const auto texture = createTexture(device.get(), RACHIS_FORMAT_RGBA8_UNORM, 4, 4, RACHIS_TEXTURE_SAMPLED, error);
    rachis_texture_view_desc viewDesc{};
    viewDesc.struct_size = sizeof(viewDesc);
    viewDesc.abi_version = RACHIS_ABI_VERSION;
    viewDesc.texture = texture;
    viewDesc.usage = RACHIS_VIEW_USAGE_SAMPLED;
    viewDesc.dimension = RACHIS_VIEW_2D;
    viewDesc.aspect = RACHIS_ASPECT_COLOR;
    viewDesc.mip_level_count = 1;
    viewDesc.array_layer_count = 1;
    rachis_handle view = 0;
    requireOk(rachis_texture_view_create(device.get(), &viewDesc, &view, &error), error);

    rachis_sampler_desc samplerDesc{};
    samplerDesc.struct_size = sizeof(samplerDesc);
    samplerDesc.abi_version = RACHIS_ABI_VERSION;
    samplerDesc.min_filter = RACHIS_FILTER_LINEAR;
    samplerDesc.mag_filter = RACHIS_FILTER_LINEAR;
    samplerDesc.mip_filter = RACHIS_MIP_FILTER_NONE;
    samplerDesc.address_u = RACHIS_ADDRESS_CLAMP_TO_EDGE;
    samplerDesc.address_v = RACHIS_ADDRESS_CLAMP_TO_EDGE;
    samplerDesc.address_w = RACHIS_ADDRESS_CLAMP_TO_EDGE;
    rachis_handle sampler = 0;
    requireOk(rachis_sampler_create(device.get(), &samplerDesc, &sampler, &error), error);
    rachis_handle foreign = 0;
    CHECK(rachis_sampler_create(other.get(), &samplerDesc, &foreign, &error) == RACHIS_STATUS_OK);

    const std::array entries{
        rachis_binding_layout_entry{0, RACHIS_BINDING_SAMPLED_TEXTURE, RACHIS_SHADER_PIXEL, 1, 0},
        rachis_binding_layout_entry{1, RACHIS_BINDING_SAMPLER, RACHIS_SHADER_PIXEL, 1, 0},
    };
    rachis_binding_layout_desc layoutDesc{};
    layoutDesc.struct_size = sizeof(layoutDesc);
    layoutDesc.abi_version = RACHIS_ABI_VERSION;
    layoutDesc.entries = entries.data();
    layoutDesc.entry_count = static_cast<std::uint32_t>(entries.size());
    rachis_handle layout = 0;
    requireOk(rachis_binding_layout_create(device.get(), &layoutDesc, &layout, &error), error);
    const std::array sizes{
        rachis_binding_pool_size{RACHIS_BINDING_SAMPLED_TEXTURE, 2},
        rachis_binding_pool_size{RACHIS_BINDING_SAMPLER, 2},
    };
    rachis_binding_pool_desc poolDesc{};
    poolDesc.struct_size = sizeof(poolDesc);
    poolDesc.abi_version = RACHIS_ABI_VERSION;
    poolDesc.max_binding_sets = 1;
    poolDesc.sizes = sizes.data();
    poolDesc.size_count = static_cast<std::uint32_t>(sizes.size());
    rachis_handle pool = 0;
    requireOk(rachis_binding_pool_create(device.get(), &poolDesc, &pool, &error), error);
    const rachis_texture_binding textureWrite{0, 0, view};
    rachis_sampler_binding samplerWrite{1, 0, foreign};
    rachis_binding_set_desc setDesc{};
    setDesc.struct_size = sizeof(setDesc);
    setDesc.abi_version = RACHIS_ABI_VERSION;
    setDesc.layout = layout;
    setDesc.pool = pool;
    setDesc.texture_writes = &textureWrite;
    setDesc.texture_write_count = 1;
    setDesc.sampler_writes = &samplerWrite;
    setDesc.sampler_write_count = 1;
    rachis_handle set = 0;
    CHECK(rachis_binding_set_create(device.get(), &setDesc, &set, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(set == 0);
    samplerWrite.sampler = sampler;
    requireOk(rachis_binding_set_create(device.get(), &setDesc, &set, &error), error);
    requireOk(rachis_binding_set_destroy(device.get(), set, &error), error);
    requireOk(rachis_binding_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_binding_layout_destroy(device.get(), layout, &error), error);
    requireOk(rachis_sampler_destroy(device.get(), sampler, &error), error);
    requireOk(rachis_sampler_destroy(other.get(), foreign, &error), error);
    requireOk(rachis_texture_view_destroy(device.get(), view, &error), error);
    requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
    requireNoValidationErrors(device.get(), error);
}

TEST_CASE("C ABI stage masks accept RACHIS_SHADER_ALL and reject stages outside the C ABI",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    constexpr std::uint32_t cAbiStages =
        RACHIS_SHADER_VERTEX | RACHIS_SHADER_PIXEL | RACHIS_SHADER_COMPUTE | RACHIS_SHADER_TASK | RACHIS_SHADER_MESH;
    constexpr std::uint32_t graphicsStages = RACHIS_SHADER_VERTEX | RACHIS_SHADER_PIXEL;
    constexpr std::uint32_t formerShaderAll = 0x3fffU;
    CHECK(cAbiStages == RACHIS_SHADER_ALL);

    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);

    rachis_binding_layout_entry entry{
        0, RACHIS_BINDING_UNIFORM_BUFFER, RACHIS_SHADER_VERTEX | RACHIS_SHADER_MESH, 1, 0};
    rachis_binding_layout_desc bindingLayoutDesc{};
    bindingLayoutDesc.struct_size = sizeof(bindingLayoutDesc);
    bindingLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    bindingLayoutDesc.entries = &entry;
    bindingLayoutDesc.entry_count = 1;
    rachis_handle bindingLayout = 0;
    requireOk(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &bindingLayout, &error), error);
    requireOk(rachis_binding_layout_destroy(device.get(), bindingLayout, &error), error);
    entry.visibility = RACHIS_SHADER_ALL;
    requireOk(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &bindingLayout, &error), error);

    rachis_push_constant_range pushRange{RACHIS_SHADER_ALL, 0, 16};
    rachis_pipeline_layout_desc pipelineLayoutDesc{};
    pipelineLayoutDesc.struct_size = sizeof(pipelineLayoutDesc);
    pipelineLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineLayoutDesc.binding_layouts = &bindingLayout;
    pipelineLayoutDesc.binding_layout_count = 1;
    pipelineLayoutDesc.push_constant_ranges = &pushRange;
    pipelineLayoutDesc.push_constant_range_count = 1;
    rachis_handle pipelineLayout = 0;
    requireOk(rachis_pipeline_layout_create(device.get(), &pipelineLayoutDesc, &pipelineLayout, &error), error);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
    const std::array<std::uint32_t, 4> constants{};
    CHECK(rachis_command_set_graphics_push_constants(
              device.get(), command, graphicsStages, 0, constants.data(), sizeof(constants), &error) ==
          RACHIS_STATUS_INVALID_STATE);
    CHECK(rachis_command_set_graphics_push_constants(device.get(),
                                                     command,
                                                     graphicsStages | RACHIS_SHADER_TASK | RACHIS_SHADER_MESH,
                                                     0,
                                                     constants.data(),
                                                     sizeof(constants),
                                                     &error) == RACHIS_STATUS_INVALID_STATE);
    CHECK(rachis_command_set_graphics_push_constants(
              device.get(), command, RACHIS_SHADER_ALL, 0, constants.data(), sizeof(constants), &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);

    std::vector<std::uint32_t> rejected{formerShaderAll, formerShaderAll & ~std::uint32_t{RACHIS_SHADER_COMPUTE}};
    for (std::uint32_t bit = 3; bit < 32; ++bit)
    {
        if (((1U << bit) & RACHIS_SHADER_ALL) != 0)
        {
            continue;
        }
        rejected.push_back(1U << bit);
        rejected.push_back(graphicsStages | (1U << bit));
    }
    for (const auto visibility : rejected)
    {
        INFO("visibility " << visibility);
        entry.visibility = visibility;
        rachis_handle rejectedBindingLayout = 0;
        CHECK(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &rejectedBindingLayout, &error) ==
              RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rejectedBindingLayout == 0);
        pushRange.visibility = visibility;
        rachis_handle rejectedPipelineLayout = 0;
        CHECK(rachis_pipeline_layout_create(device.get(), &pipelineLayoutDesc, &rejectedPipelineLayout, &error) ==
              RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rejectedPipelineLayout == 0);
        CHECK(rachis_command_set_graphics_push_constants(
                  device.get(), command, visibility, 0, constants.data(), sizeof(constants), &error) ==
              RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(error.code == RACHIS_STATUS_INVALID_ARGUMENT);
    }

    requireOk(rachis_command_discard(device.get(), command, &error), error);
    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_binding_layout_destroy(device.get(), bindingLayout, &error), error);
    requireNoValidationErrors(device.get(), error);
}

#ifdef RACHIS_CABI_INDEXED_VERTEX_PATH
namespace
{
    struct IndexedVertex final
    {
        float x;
        float y;
        std::array<std::uint8_t, 4> color;
    };
    static_assert(sizeof(IndexedVertex) == 12);
} // namespace

TEST_CASE("C ABI draws an indexed quad from mapped vertex and index buffers",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][graphics]")
{
    const auto load = [](const char* path) {
        std::ifstream input{path, std::ios::binary};
        return std::vector<char>{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    };
    const auto vertexBytes = load(RACHIS_CABI_INDEXED_VERTEX_PATH);
    const auto pixelBytes = load(RACHIS_CABI_INDEXED_PIXEL_PATH);
    REQUIRE_FALSE(vertexBytes.empty());
    REQUIRE_FALSE(pixelBytes.empty());

    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    rachis_device_capabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    requireOk(rachis_device_query_capabilities(device.get(), &capabilities, &error), error);
    REQUIRE((capabilities.vertex_formats & (std::uint64_t{1} << RACHIS_VERTEX_FORMAT_FLOAT32X2)) != 0);
    REQUIRE((capabilities.vertex_formats & (std::uint64_t{1} << RACHIS_VERTEX_FORMAT_UNORM8X4)) != 0);
    REQUIRE(capabilities.max_vertex_buffers >= 1);
    REQUIRE(capabilities.max_vertex_attributes >= 2);
    REQUIRE(capabilities.max_vertex_buffer_stride >= sizeof(IndexedVertex));

    const std::array<std::uint32_t, 2> inputLocations{0, 1};
    rachis_shader_module_desc shaderDesc{};
    shaderDesc.struct_size = sizeof(shaderDesc);
    shaderDesc.abi_version = RACHIS_ABI_VERSION;
    shaderDesc.stage = RACHIS_SHADER_VERTEX;
    shaderDesc.bytecode = vertexBytes.data();
    shaderDesc.bytecode_size = vertexBytes.size();
    shaderDesc.entry_point = "vertexMain";
    shaderDesc.vertex_input_locations = inputLocations.data();
    shaderDesc.vertex_input_location_count = static_cast<std::uint32_t>(inputLocations.size());
    rachis_handle vertex = 0;
    requireOk(rachis_shader_module_create(device.get(), &shaderDesc, &vertex, &error), error);
    rachis_handle rejectedShader = 0;
    shaderDesc.vertex_input_locations = nullptr;
    CHECK(rachis_shader_module_create(device.get(), &shaderDesc, &rejectedShader, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rejectedShader == 0);
    shaderDesc.vertex_input_locations = inputLocations.data();
    shaderDesc.vertex_input_location_count = RACHIS_MAX_VERTEX_ATTRIBUTES + 1;
    CHECK(rachis_shader_module_create(device.get(), &shaderDesc, &rejectedShader, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rejectedShader == 0);
    shaderDesc.vertex_input_location_count = static_cast<std::uint32_t>(inputLocations.size());
    shaderDesc.stage = RACHIS_SHADER_PIXEL;
    shaderDesc.bytecode = pixelBytes.data();
    shaderDesc.bytecode_size = pixelBytes.size();
    shaderDesc.entry_point = "pixelMain";
    shaderDesc.vertex_input_locations = inputLocations.data();
    CHECK(rachis_shader_module_create(device.get(), &shaderDesc, &rejectedShader, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rejectedShader == 0);
    shaderDesc.vertex_input_locations = nullptr;
    shaderDesc.vertex_input_location_count = 0;
    rachis_handle pixel = 0;
    requireOk(rachis_shader_module_create(device.get(), &shaderDesc, &pixel, &error), error);

    rachis_pipeline_layout_desc layoutDesc{};
    layoutDesc.struct_size = sizeof(layoutDesc);
    layoutDesc.abi_version = RACHIS_ABI_VERSION;
    rachis_handle pipelineLayout = 0;
    requireOk(rachis_pipeline_layout_create(device.get(), &layoutDesc, &pipelineLayout, &error), error);
    rachis_color_target target{};
    target.format = RACHIS_FORMAT_RGBA8_UNORM;
    target.write_mask = RACHIS_COLOR_WRITE_ALL;
    target.color_blend.source_factor = RACHIS_BLEND_ONE;
    target.alpha_blend.source_factor = RACHIS_BLEND_ONE;
    std::array<rachis_vertex_attribute, 2> attributes{{
        {RACHIS_VERTEX_FORMAT_FLOAT32X2, 0, 0},
        {RACHIS_VERTEX_FORMAT_UNORM8X4, 8, 1},
    }};
    rachis_vertex_buffer_layout vertexLayout{};
    vertexLayout.stride = sizeof(IndexedVertex);
    vertexLayout.step_mode = RACHIS_VERTEX_STEP_VERTEX;
    vertexLayout.attributes = attributes.data();
    vertexLayout.attribute_count = static_cast<std::uint32_t>(attributes.size());
    rachis_graphics_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.vertex_shader = vertex;
    pipelineDesc.pixel_shader = pixel;
    pipelineDesc.topology = RACHIS_TOPOLOGY_TRIANGLE_LIST;
    pipelineDesc.cull_mode = RACHIS_CULL_NONE;
    pipelineDesc.color_targets = &target;
    pipelineDesc.color_target_count = 1;
    pipelineDesc.vertex_buffers = &vertexLayout;
    pipelineDesc.vertex_buffer_count = 1;
    rachis_handle pipeline = 0;
    requireOk(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error), error);

    rachis_handle rejectedPipeline = 0;
    const auto rejectsPipeline = [&](rachis_status expected) {
        CHECK(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &rejectedPipeline, &error) == expected);
        CHECK(rejectedPipeline == 0);
    };
    pipelineDesc.vertex_buffer_count = RACHIS_MAX_VERTEX_BUFFERS + 1;
    rejectsPipeline(RACHIS_STATUS_INVALID_ARGUMENT);
    pipelineDesc.vertex_buffer_count = 1;
    pipelineDesc.vertex_buffers = nullptr;
    rejectsPipeline(RACHIS_STATUS_INVALID_ARGUMENT);
    pipelineDesc.vertex_buffers = &vertexLayout;
    attributes[1].format = RACHIS_VERTEX_FORMAT_FLOAT16X4 + 1;
    rejectsPipeline(RACHIS_STATUS_INVALID_ARGUMENT);
    attributes[1].format = RACHIS_VERTEX_FORMAT_UNORM8X4;
    attributes[1].offset = 9;
    rejectsPipeline(RACHIS_STATUS_INVALID_ARGUMENT);
    attributes[1].offset = 8;
    vertexLayout.step_mode = RACHIS_VERTEX_STEP_INSTANCE + 1;
    rejectsPipeline(RACHIS_STATUS_INVALID_ARGUMENT);
    vertexLayout.step_mode = RACHIS_VERTEX_STEP_VERTEX;
    vertexLayout.attribute_count = RACHIS_MAX_VERTEX_ATTRIBUTES + 1;
    rejectsPipeline(RACHIS_STATUS_INVALID_ARGUMENT);
    vertexLayout.attribute_count = static_cast<std::uint32_t>(attributes.size());
    vertexLayout.attributes = nullptr;
    rejectsPipeline(RACHIS_STATUS_INVALID_ARGUMENT);
    vertexLayout.attributes = attributes.data();
    pipelineDesc.strip_index_format = RACHIS_STRIP_INDEX_UINT32 + 1;
    rejectsPipeline(RACHIS_STATUS_INVALID_ARGUMENT);
    pipelineDesc.strip_index_format = RACHIS_STRIP_INDEX_UINT16;
    rejectsPipeline(RACHIS_STATUS_INVALID_ARGUMENT);
    pipelineDesc.topology = RACHIS_TOPOLOGY_TRIANGLE_STRIP;
    rachis_handle stripPipeline = 0;
    requireOk(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &stripPipeline, &error), error);
    requireOk(rachis_pipeline_destroy(device.get(), stripPipeline, &error), error);
    pipelineDesc.topology = RACHIS_TOPOLOGY_TRIANGLE_LIST;
    pipelineDesc.strip_index_format = RACHIS_STRIP_INDEX_NONE;
    pipelineDesc.vertex_buffer_count = 0;
    CHECK(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &rejectedPipeline, &error) != RACHIS_STATUS_OK);
    CHECK(rejectedPipeline == 0);
    pipelineDesc.vertex_buffer_count = 1;

    const std::array<std::uint8_t, 4> green{0, 255, 0, 255};
    const std::array<IndexedVertex, 5> vertices{{
        {3.0F, 3.0F, {255, 255, 255, 255}},
        {-1.0F, -1.0F, green},
        {0.0F, -1.0F, green},
        {-1.0F, 1.0F, green},
        {0.0F, 1.0F, green},
    }};
    const std::array<std::uint16_t, 2> unusedIndices{3, 3};
    const std::array<std::uint16_t, 6> quadIndices{0, 1, 2, 2, 1, 3};
    const auto vertexBuffer =
        createBuffer(device.get(), sizeof(vertices), RACHIS_BUFFER_VERTEX, RACHIS_MEMORY_UPLOAD, error);
    const auto indexBuffer = createBuffer(
        device.get(), sizeof(unusedIndices) + sizeof(quadIndices), RACHIS_BUFFER_INDEX, RACHIS_MEMORY_UPLOAD, error);

    rachis_mapped_memory mappedVertices{};
    mappedVertices.struct_size = sizeof(mappedVertices);
    requireOk(rachis_buffer_map(device.get(), vertexBuffer, 0, RACHIS_WHOLE_SIZE, &mappedVertices, &error), error);
    REQUIRE(mappedVertices.data != nullptr);
    REQUIRE(mappedVertices.size == sizeof(vertices));
    std::memcpy(mappedVertices.data, vertices.data(), sizeof(vertices));
    const rachis_mapped_range wholeRange{0, RACHIS_WHOLE_SIZE};
    requireOk(rachis_buffer_flush_mapped_ranges(device.get(), vertexBuffer, &wholeRange, 1, &error), error);
    const bool keepVerticesMapped = capabilities.persistent_mapping != 0;
    if (!keepVerticesMapped)
    {
        requireOk(rachis_buffer_unmap(device.get(), vertexBuffer, &error), error);
    }

    requireOk(rachis_buffer_write(device.get(), indexBuffer, 0, unusedIndices.data(), sizeof(unusedIndices), &error),
              error);
    rachis_mapped_memory mappedIndices{};
    mappedIndices.struct_size = sizeof(mappedIndices);
    requireOk(rachis_buffer_map(
                  device.get(), indexBuffer, sizeof(unusedIndices), sizeof(quadIndices), &mappedIndices, &error),
              error);
    REQUIRE(mappedIndices.data != nullptr);
    REQUIRE(mappedIndices.size == sizeof(quadIndices));
    std::memcpy(mappedIndices.data, quadIndices.data(), sizeof(quadIndices));
    requireOk(rachis_buffer_unmap(device.get(), indexBuffer, &error), error);

    const auto texture = createTexture(device.get(),
                                       RACHIS_FORMAT_RGBA8_UNORM,
                                       textureSize,
                                       textureSize,
                                       RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_COPY_SOURCE,
                                       error);
    const auto viewDesc = renderTargetViewDesc(texture);
    rachis_handle view = 0;
    requireOk(rachis_texture_view_create(device.get(), &viewDesc, &view, &error), error);
    const auto region = wholeTextureRegion(textureSize, textureSize);
    rachis_texture_transfer_layout transfer{};
    transfer.struct_size = sizeof(transfer);
    requireOk(rachis_texture_query_transfer_layout(device.get(), texture, &region, &transfer, &error), error);
    const auto readback = createBuffer(
        device.get(), transfer.required_buffer_size, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
    const auto toRender = textureBarrier(texture,
                                         RACHIS_STAGE_ALL,
                                         RACHIS_ACCESS_NONE,
                                         RACHIS_LAYOUT_UNDEFINED,
                                         RACHIS_STAGE_RENDER_TARGET,
                                         RACHIS_ACCESS_RENDER_TARGET,
                                         RACHIS_LAYOUT_RENDER_TARGET);
    requireOk(recordBarriers(device.get(), command, nullptr, 0, &toRender, 1, error), error);
    rachis_color_attachment attachment{};
    attachment.view = view;
    attachment.load_op = RACHIS_LOAD_OP_CLEAR;
    attachment.store_op = RACHIS_STORE_OP_STORE;
    attachment.clear_color[2] = 1.0F;
    attachment.clear_color[3] = 1.0F;
    rachis_rendering_desc rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.abi_version = RACHIS_ABI_VERSION;
    rendering.color_attachments = &attachment;
    rendering.color_attachment_count = 1;
    requireOk(rachis_command_begin_rendering(device.get(), command, &rendering, &error), error);
    const rachis_viewport viewport{0, 0, textureSize, textureSize, 0, 1};
    const rachis_scissor scissor{0, 0, textureSize, textureSize};
    requireOk(rachis_command_set_viewport(device.get(), command, &viewport, &error), error);
    requireOk(rachis_command_set_scissor(device.get(), command, &scissor, &error), error);
    requireOk(rachis_command_bind_graphics_pipeline(device.get(), command, pipeline, &error), error);

    rachis_vertex_buffer_binding vertexBinding{vertexBuffer, 0};
    CHECK(rachis_command_bind_vertex_buffers(device.get(), command, 0, nullptr, 1, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_bind_vertex_buffers(device.get(), command, 0, &vertexBinding, 0, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_bind_vertex_buffers(
              device.get(), command, RACHIS_MAX_VERTEX_BUFFERS, &vertexBinding, 1, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_bind_vertex_buffers(
              device.get(), command, 0, &vertexBinding, RACHIS_MAX_VERTEX_BUFFERS + 1, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    vertexBinding.buffer = pipeline;
    CHECK(rachis_command_bind_vertex_buffers(device.get(), command, 0, &vertexBinding, 1, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    vertexBinding.buffer = vertexBuffer;
    vertexBinding.offset = 2;
    CHECK(rachis_command_bind_vertex_buffers(device.get(), command, 0, &vertexBinding, 1, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    vertexBinding.offset = 0;
    requireOk(rachis_command_bind_vertex_buffers(device.get(), command, 0, &vertexBinding, 1, &error), error);
    CHECK(rachis_command_draw_indexed(device.get(), command, 6, 1, 2, 1, 0, &error) != RACHIS_STATUS_OK);
    CHECK(rachis_command_bind_index_buffer(
              device.get(), command, indexBuffer, RACHIS_INDEX_FORMAT_UINT32 + 1, 0, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_bind_index_buffer(device.get(), command, pipeline, RACHIS_INDEX_FORMAT_UINT16, 0, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(
        rachis_command_bind_index_buffer(device.get(), command, indexBuffer, RACHIS_INDEX_FORMAT_UINT16, 0, &error),
        error);
    requireOk(rachis_command_draw_indexed(device.get(), command, 6, 1, 2, 1, 0, &error), error);
    requireOk(rachis_command_end_rendering(device.get(), command, &error), error);
    const auto toCopy = textureBarrier(texture,
                                       RACHIS_STAGE_RENDER_TARGET,
                                       RACHIS_ACCESS_RENDER_TARGET,
                                       RACHIS_LAYOUT_RENDER_TARGET,
                                       RACHIS_STAGE_COPY,
                                       RACHIS_ACCESS_COPY_SOURCE,
                                       RACHIS_LAYOUT_COPY_SOURCE);
    requireOk(recordBarriers(device.get(), command, nullptr, 0, &toCopy, 1, error), error);
    auto copyRegion = region;
    copyRegion.bytes_per_row = transfer.bytes_per_row;
    requireOk(copyBufferTexture(device.get(), command, readback, texture, copyRegion, false, error), error);
    submitAndWait(device.get(), command, error);
    if (keepVerticesMapped)
    {
        requireOk(rachis_buffer_unmap(device.get(), vertexBuffer, &error), error);
    }

    rachis_mapped_memory mappedPixels{};
    mappedPixels.struct_size = sizeof(mappedPixels);
    requireOk(rachis_buffer_map(device.get(), readback, 0, RACHIS_WHOLE_SIZE, &mappedPixels, &error), error);
    REQUIRE(mappedPixels.data != nullptr);
    REQUIRE(mappedPixels.size == transfer.required_buffer_size);
    requireOk(rachis_buffer_invalidate_mapped_ranges(device.get(), readback, &wholeRange, 1, &error), error);
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(transfer.required_buffer_size));
    std::memcpy(pixels.data(), mappedPixels.data, pixels.size());
    requireOk(rachis_buffer_unmap(device.get(), readback, &error), error);
    for (std::uint32_t y = 0; y < textureSize; ++y)
    {
        for (std::uint32_t x = 0; x < textureSize; ++x)
        {
            INFO("pixel " << x << ", " << y);
            const auto* texel = pixels.data() + static_cast<std::size_t>(y) * transfer.bytes_per_row + x * 4U;
            const bool insideQuad = x < textureSize / 2;
            CHECK(texel[0] == 0);
            CHECK(texel[1] == (insideQuad ? 255 : 0));
            CHECK(texel[2] == (insideQuad ? 0 : 255));
            CHECK(texel[3] == 255);
        }
    }

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), indexBuffer, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), vertexBuffer, &error), error);
    requireOk(rachis_texture_view_destroy(device.get(), view, &error), error);
    requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
    requireOk(rachis_pipeline_destroy(device.get(), pipeline, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), pixel, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), vertex, &error), error);
    requireNoValidationErrors(device.get(), error);
}
#endif

#ifdef RACHIS_CABI_DRAW_INDEX_VERTEX_PATH
namespace
{
    using Texel = std::array<std::uint8_t, 4>;

    constexpr Texel clearTexel{0, 0, 0, 0};
    constexpr Texel redTexel{255, 0, 0, 255};
    constexpr Texel greenTexel{0, 255, 0, 255};
    constexpr Texel blueTexel{0, 0, 255, 255};
    constexpr Texel magentaTexel{255, 0, 255, 255};
    constexpr Texel whiteTexel{255, 255, 255, 255};

    [[nodiscard]] constexpr std::uint32_t packTexel(const Texel& texel) noexcept
    {
        return std::uint32_t{texel[0]} | (std::uint32_t{texel[1]} << 8U) | (std::uint32_t{texel[2]} << 16U) |
               (std::uint32_t{texel[3]} << 24U);
    }

    [[nodiscard]] std::vector<char> loadFile(const char* path)
    {
        std::ifstream input{path, std::ios::binary};
        return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    }

    [[nodiscard]] rachis_device_capabilities queryCapabilities(rachis_device* device, rachis_error& error)
    {
        rachis_device_capabilities capabilities{};
        capabilities.struct_size = sizeof(capabilities);
        requireOk(rachis_device_query_capabilities(device, &capabilities, &error), error);
        return capabilities;
    }

    [[nodiscard]] rachis_shader_module_desc shaderModuleDesc(std::uint32_t stage,
                                                             const std::vector<char>& bytecode,
                                                             const char* entryPoint) noexcept
    {
        rachis_shader_module_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.stage = stage;
        desc.bytecode = bytecode.data();
        desc.bytecode_size = bytecode.size();
        desc.entry_point = entryPoint;
        if (stage == RACHIS_SHADER_COMPUTE || stage == RACHIS_SHADER_TASK || stage == RACHIS_SHADER_MESH)
        {
            desc.workgroup_x = 1;
            desc.workgroup_y = 1;
            desc.workgroup_z = 1;
        }
        return desc;
    }

    [[nodiscard]] rachis_handle createShaderModule(rachis_device* device,
                                                   const rachis_shader_module_desc& desc,
                                                   rachis_error& error)
    {
        rachis_handle shader = 0;
        requireOk(rachis_shader_module_create(device, &desc, &shader, &error), error);
        return shader;
    }

    [[nodiscard]] rachis_status createDrawIndexLayout(rachis_device* device,
                                                      const rachis_handle* bindingLayouts,
                                                      std::uint32_t bindingLayoutCount,
                                                      std::uint32_t drawIndex,
                                                      rachis_handle& layout,
                                                      rachis_error& error)
    {
        rachis_pipeline_layout_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.binding_layouts = bindingLayouts;
        desc.binding_layout_count = bindingLayoutCount;
        desc.draw_index = drawIndex;
        return rachis_pipeline_layout_create(device, &desc, &layout, &error);
    }

    [[nodiscard]] rachis_color_target opaqueTarget() noexcept
    {
        rachis_color_target target{};
        target.format = RACHIS_FORMAT_RGBA8_UNORM;
        target.write_mask = RACHIS_COLOR_WRITE_ALL;
        target.color_blend.source_factor = RACHIS_BLEND_ONE;
        target.alpha_blend.source_factor = RACHIS_BLEND_ONE;
        return target;
    }

    struct CellTarget final
    {
        rachis_handle texture = 0;
        rachis_handle view = 0;
        rachis_handle readback = 0;
        rachis_buffer_texture_region region{};
    };

    [[nodiscard]] CellTarget createCellTarget(rachis_device* device, rachis_error& error)
    {
        CellTarget target;
        target.texture = createTexture(device,
                                       RACHIS_FORMAT_RGBA8_UNORM,
                                       textureSize,
                                       textureSize,
                                       RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_COPY_SOURCE,
                                       error);
        const auto viewDesc = renderTargetViewDesc(target.texture);
        requireOk(rachis_texture_view_create(device, &viewDesc, &target.view, &error), error);
        target.region = wholeTextureRegion(textureSize, textureSize);
        rachis_texture_transfer_layout transfer{};
        transfer.struct_size = sizeof(transfer);
        requireOk(rachis_texture_query_transfer_layout(device, target.texture, &target.region, &transfer, &error),
                  error);
        target.region.bytes_per_row = transfer.bytes_per_row;
        target.readback = createBuffer(
            device, transfer.required_buffer_size, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
        return target;
    }

    void beginCellRendering(rachis_device* device, rachis_handle command, const CellTarget& target, rachis_error& error)
    {
        const auto toRender = textureBarrier(target.texture,
                                             RACHIS_STAGE_ALL,
                                             RACHIS_ACCESS_NONE,
                                             RACHIS_LAYOUT_UNDEFINED,
                                             RACHIS_STAGE_RENDER_TARGET,
                                             RACHIS_ACCESS_RENDER_TARGET,
                                             RACHIS_LAYOUT_RENDER_TARGET);
        requireOk(recordBarriers(device, command, nullptr, 0, &toRender, 1, error), error);
        rachis_color_attachment attachment{};
        attachment.view = target.view;
        attachment.load_op = RACHIS_LOAD_OP_CLEAR;
        attachment.store_op = RACHIS_STORE_OP_STORE;
        rachis_rendering_desc rendering{};
        rendering.struct_size = sizeof(rendering);
        rendering.abi_version = RACHIS_ABI_VERSION;
        rendering.color_attachments = &attachment;
        rendering.color_attachment_count = 1;
        requireOk(rachis_command_begin_rendering(device, command, &rendering, &error), error);
        const rachis_viewport viewport{0, 0, textureSize, textureSize, 0, 1};
        const rachis_scissor scissor{0, 0, textureSize, textureSize};
        requireOk(rachis_command_set_viewport(device, command, &viewport, &error), error);
        requireOk(rachis_command_set_scissor(device, command, &scissor, &error), error);
    }

    void endCellRendering(rachis_device* device, rachis_handle command, const CellTarget& target, rachis_error& error)
    {
        requireOk(rachis_command_end_rendering(device, command, &error), error);
        const auto toCopy = textureBarrier(target.texture,
                                           RACHIS_STAGE_RENDER_TARGET,
                                           RACHIS_ACCESS_RENDER_TARGET,
                                           RACHIS_LAYOUT_RENDER_TARGET,
                                           RACHIS_STAGE_COPY,
                                           RACHIS_ACCESS_COPY_SOURCE,
                                           RACHIS_LAYOUT_COPY_SOURCE);
        requireOk(recordBarriers(device, command, nullptr, 0, &toCopy, 1, error), error);
        requireOk(copyBufferTexture(device, command, target.readback, target.texture, target.region, false, error),
                  error);
    }

    [[nodiscard]] std::vector<Texel> readCells(rachis_device* device, const CellTarget& target, rachis_error& error)
    {
        rachis_mapped_memory mapped{};
        mapped.struct_size = sizeof(mapped);
        requireOk(rachis_buffer_map(device, target.readback, 0, RACHIS_WHOLE_SIZE, &mapped, &error), error);
        REQUIRE(mapped.data != nullptr);
        const rachis_mapped_range wholeRange{0, RACHIS_WHOLE_SIZE};
        requireOk(rachis_buffer_invalidate_mapped_ranges(device, target.readback, &wholeRange, 1, &error), error);
        std::vector<Texel> cells(static_cast<std::size_t>(textureSize) * textureSize);
        for (std::uint32_t y = 0; y < textureSize; ++y)
        {
            for (std::uint32_t x = 0; x < textureSize; ++x)
            {
                const auto* texel = static_cast<const std::uint8_t*>(mapped.data) +
                                    static_cast<std::size_t>(y) * target.region.bytes_per_row + x * 4U;
                std::memcpy(cells[static_cast<std::size_t>(y) * textureSize + x].data(), texel, sizeof(Texel));
            }
        }
        requireOk(rachis_buffer_unmap(device, target.readback, &error), error);
        return cells;
    }

    using CellRow = std::array<Texel, textureSize>;

    void checkCells(const std::vector<Texel>& cells, const std::array<CellRow, textureSize>& expected)
    {
        for (std::uint32_t y = 0; y < textureSize; ++y)
        {
            for (std::uint32_t x = 0; x < textureSize; ++x)
            {
                INFO("cell column " << x << ", row " << y);
                const auto& actual = cells[static_cast<std::size_t>(y) * textureSize + x];
                const auto& wanted = expected[y][x];
                for (std::size_t channel = 0; channel < actual.size(); ++channel)
                {
                    INFO("channel " << channel);
                    CHECK(int{actual[channel]} == int{wanted[channel]});
                }
            }
        }
    }

    void destroyCellTarget(rachis_device* device, const CellTarget& target, rachis_error& error)
    {
        requireOk(rachis_buffer_destroy(device, target.readback, &error), error);
        requireOk(rachis_texture_view_destroy(device, target.view, &error), error);
        requireOk(rachis_texture_destroy(device, target.texture, &error), error);
    }

    [[nodiscard]] rachis_indirect_draw_desc indirectDrawDesc(rachis_handle arguments) noexcept
    {
        rachis_indirect_draw_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.argument_buffer = arguments;
        desc.max_draw_count = 1;
        return desc;
    }

    template<typename Word, std::size_t Count>
    void writeWords(rachis_device* device,
                    rachis_handle buffer,
                    const std::array<Word, Count>& words,
                    rachis_error& error)
    {
        requireOk(rachis_buffer_write(device, buffer, 0, words.data(), sizeof(words), &error), error);
    }
} // namespace

TEST_CASE("C ABI counted indirect draws read the draw index and a variable-count binding array",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][graphics][indirect][bindless]")
{
    const auto vertexBytes = loadFile(RACHIS_CABI_DRAW_INDEX_VERTEX_PATH);
    const auto pixelBytes = loadFile(RACHIS_CABI_BINDING_ARRAY_PIXEL_PATH);
    REQUIRE_FALSE(vertexBytes.empty());
    REQUIRE_FALSE(pixelBytes.empty());

    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    const auto capabilities = queryCapabilities(device.get(), error);
    if (RACHIS_SMOKE_BACKEND == RACHIS_BACKEND_D3D12)
    {
        REQUIRE(capabilities.shader_draw_index == 1);
        REQUIRE(capabilities.multi_draw_indirect == 1);
        REQUIRE(capabilities.draw_indirect_count == 1);
        REQUIRE(capabilities.partially_bound_binding_arrays == 1);
        REQUIRE(capabilities.variable_descriptor_count_binding_arrays == 1);
        REQUIRE(capabilities.update_after_bind_binding_arrays == 1);
    }
    if (capabilities.shader_draw_index == 0 || capabilities.multi_draw_indirect == 0 ||
        capabilities.partially_bound_binding_arrays == 0 || capabilities.variable_descriptor_count_binding_arrays == 0)
    {
        SKIP("The device lacks the draw index, multi-draw indirect, or variable-count binding arrays.");
    }
    const bool counted = capabilities.draw_indirect_count != 0;
    const bool updateAfterBind = capabilities.update_after_bind_binding_arrays != 0;
    REQUIRE(capabilities.max_variable_buffer_descriptors >= 8);

    const std::array<std::uint32_t, 1> inputLocations{0};
    auto vertexDesc = shaderModuleDesc(RACHIS_SHADER_VERTEX, vertexBytes, "indirectVertex");
    vertexDesc.vertex_input_locations = inputLocations.data();
    vertexDesc.vertex_input_location_count = static_cast<std::uint32_t>(inputLocations.size());
    rachis_handle rejectedShader = 0;
    vertexDesc.draw_index = 2;
    CHECK(rachis_shader_module_create(device.get(), &vertexDesc, &rejectedShader, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rejectedShader == 0);
    if (RACHIS_SMOKE_BACKEND != RACHIS_BACKEND_METAL)
    {
        vertexDesc.draw_index = 0;
        CHECK(rachis_shader_module_create(device.get(), &vertexDesc, &rejectedShader, &error) ==
              RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rejectedShader == 0);
    }
    vertexDesc.draw_index = 1;
    const auto vertex = createShaderModule(device.get(), vertexDesc, error);

    const rachis_shader_binding_requirement colorArray{
        0, 0, RACHIS_BINDING_READ_ONLY_STORAGE_BUFFER, RACHIS_UNBOUNDED_BINDING_COUNT};
    auto pixelDesc = shaderModuleDesc(RACHIS_SHADER_PIXEL, pixelBytes, "arrayPixel");
    pixelDesc.bindings = &colorArray;
    pixelDesc.binding_count = 1;
    pixelDesc.draw_index = 1;
    CHECK(rachis_shader_module_create(device.get(), &pixelDesc, &rejectedShader, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rejectedShader == 0);
    pixelDesc.draw_index = 0;
    const auto pixel = createShaderModule(device.get(), pixelDesc, error);

    constexpr std::uint32_t arrayLength = 8;
    std::uint32_t arrayFlags = RACHIS_BINDING_FLAG_PARTIALLY_BOUND | RACHIS_BINDING_FLAG_VARIABLE_DESCRIPTOR_COUNT;
    if (updateAfterBind)
    {
        arrayFlags |= RACHIS_BINDING_FLAG_UPDATE_AFTER_BIND;
    }
    rachis_binding_layout_entry arrayEntry{
        0, RACHIS_BINDING_READ_ONLY_STORAGE_BUFFER, RACHIS_SHADER_PIXEL, RACHIS_UNBOUNDED_BINDING_COUNT, 1U << 3U};
    rachis_binding_layout_desc bindingLayoutDesc{};
    bindingLayoutDesc.struct_size = sizeof(bindingLayoutDesc);
    bindingLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    bindingLayoutDesc.entries = &arrayEntry;
    bindingLayoutDesc.entry_count = 1;
    rachis_handle rejectedLayout = 0;
    CHECK(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &rejectedLayout, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rejectedLayout == 0);
    arrayEntry.flags = RACHIS_BINDING_FLAG_PARTIALLY_BOUND;
    CHECK(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &rejectedLayout, &error) != RACHIS_STATUS_OK);
    CHECK(rejectedLayout == 0);
    arrayEntry.flags = arrayFlags;
    rachis_handle bindingLayout = 0;
    requireOk(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &bindingLayout, &error), error);
    const rachis_binding_pool_size poolSize{RACHIS_BINDING_READ_ONLY_STORAGE_BUFFER, 2 * arrayLength};
    rachis_binding_pool_desc bindingPoolDesc{};
    bindingPoolDesc.struct_size = sizeof(bindingPoolDesc);
    bindingPoolDesc.abi_version = RACHIS_ABI_VERSION;
    bindingPoolDesc.max_binding_sets = 2;
    bindingPoolDesc.sizes = &poolSize;
    bindingPoolDesc.size_count = 1;
    rachis_handle bindingPool = 0;
    requireOk(rachis_binding_pool_create(device.get(), &bindingPoolDesc, &bindingPool, &error), error);

    constexpr std::uint64_t colorBytes = 256;
    const std::array<Texel, 5> colors{redTexel, greenTexel, blueTexel, magentaTexel, whiteTexel};
    std::array<rachis_handle, 5> colorBuffers{};
    for (auto& buffer : colorBuffers)
    {
        buffer = createBuffer(device.get(),
                              colorBytes,
                              RACHIS_BUFFER_STORAGE | RACHIS_BUFFER_COPY_DESTINATION,
                              RACHIS_MEMORY_DEVICE_LOCAL,
                              error);
    }
    std::array<rachis_buffer_binding, 4> colorWrites{};
    for (std::uint32_t index = 0; index < colorWrites.size(); ++index)
    {
        colorWrites[index] = {0, index * 2 + 1, colorBuffers[index], 0, 16, sizeof(std::uint32_t)};
    }
    rachis_binding_set_desc setDesc{};
    setDesc.struct_size = sizeof(setDesc);
    setDesc.abi_version = RACHIS_ABI_VERSION;
    setDesc.layout = bindingLayout;
    setDesc.pool = bindingPool;
    setDesc.writes = colorWrites.data();
    setDesc.write_count = static_cast<std::uint32_t>(colorWrites.size());
    setDesc.variable_descriptor_count = capabilities.max_variable_buffer_descriptors + 1;
    rachis_handle set = 0;
    CHECK(rachis_binding_set_create(device.get(), &setDesc, &set, &error) != RACHIS_STATUS_OK);
    CHECK(set == 0);
    setDesc.variable_descriptor_count = arrayLength;
    requireOk(rachis_binding_set_create(device.get(), &setDesc, &set, &error), error);

    const std::array<rachis_buffer_binding, 2> laterWrites{{
        {0, 5, colorBuffers[4], 0, 16, sizeof(std::uint32_t)},
        {0, 7, 0, 0, 0, 0},
    }};
    rachis_binding_set_update_desc updateDesc{};
    updateDesc.struct_size = sizeof(updateDesc);
    updateDesc.abi_version = RACHIS_ABI_VERSION;
    updateDesc.writes = laterWrites.data();
    updateDesc.write_count = static_cast<std::uint32_t>(laterWrites.size());
    CHECK(rachis_binding_set_update(device.get(), set, nullptr, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_binding_set_update(nullptr, set, &updateDesc, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_binding_set_update(device.get(), bindingLayout, &updateDesc, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    updateDesc.struct_size = sizeof(updateDesc) - 1;
    CHECK(rachis_binding_set_update(device.get(), set, &updateDesc, &error) == RACHIS_STATUS_ABI_MISMATCH);
    updateDesc.struct_size = sizeof(updateDesc);
    updateDesc.abi_version = RACHIS_ABI_VERSION - 1;
    CHECK(rachis_binding_set_update(device.get(), set, &updateDesc, &error) == RACHIS_STATUS_ABI_MISMATCH);
    updateDesc.abi_version = RACHIS_ABI_VERSION;
    updateDesc.writes = nullptr;
    CHECK(rachis_binding_set_update(device.get(), set, &updateDesc, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    const rachis_buffer_binding foreignWrite{0, 5, set, 0, 16, sizeof(std::uint32_t)};
    updateDesc.writes = &foreignWrite;
    updateDesc.write_count = 1;
    CHECK(rachis_binding_set_update(device.get(), set, &updateDesc, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    const rachis_buffer_binding pastEndWrite{0, arrayLength, colorBuffers[4], 0, 16, sizeof(std::uint32_t)};
    updateDesc.writes = &pastEndWrite;
    CHECK(rachis_binding_set_update(device.get(), set, &updateDesc, &error) != RACHIS_STATUS_OK);
    updateDesc.writes = laterWrites.data();
    updateDesc.write_count = static_cast<std::uint32_t>(laterWrites.size());
    if (!updateAfterBind)
    {
        requireOk(rachis_binding_set_update(device.get(), set, &updateDesc, &error), error);
    }

    rachis_handle pipelineLayout = 0;
    CHECK(createDrawIndexLayout(device.get(), &bindingLayout, 1, 2, pipelineLayout, error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(pipelineLayout == 0);
    rachis_handle plainLayout = 0;
    requireOk(createDrawIndexLayout(device.get(), &bindingLayout, 1, 0, plainLayout, error), error);
    requireOk(createDrawIndexLayout(device.get(), &bindingLayout, 1, 1, pipelineLayout, error), error);

    const auto target = opaqueTarget();
    const rachis_vertex_attribute cellAttribute{RACHIS_VERTEX_FORMAT_FLOAT32X2, 0, 0};
    rachis_vertex_buffer_layout vertexLayout{};
    vertexLayout.stride = 2 * sizeof(float);
    vertexLayout.step_mode = RACHIS_VERTEX_STEP_VERTEX;
    vertexLayout.attributes = &cellAttribute;
    vertexLayout.attribute_count = 1;
    rachis_graphics_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = plainLayout;
    pipelineDesc.vertex_shader = vertex;
    pipelineDesc.pixel_shader = pixel;
    pipelineDesc.topology = RACHIS_TOPOLOGY_TRIANGLE_LIST;
    pipelineDesc.cull_mode = RACHIS_CULL_NONE;
    pipelineDesc.color_targets = &target;
    pipelineDesc.color_target_count = 1;
    pipelineDesc.vertex_buffers = &vertexLayout;
    pipelineDesc.vertex_buffer_count = 1;
    rachis_handle pipeline = 0;
    CHECK(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error) ==
          RACHIS_STATUS_VALIDATION_ERROR);
    CHECK(pipeline == 0);
    pipelineDesc.layout = pipelineLayout;
    requireOk(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error), error);

    const std::array<float, 20> cellVertices{
        0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,  1.0F, 1.0F,
        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, -1.0F, 1.0F, -1.0F,
    };
    const std::array<std::uint16_t, 6> quadIndices{0, 1, 2, 2, 1, 3};
    constexpr std::uint32_t drawRecordWords = 5;
    constexpr std::uint32_t indexedRecordWords = 6;
    constexpr std::uint32_t drawRecords = 4;
    constexpr std::uint32_t indexedRecords = 2;
    constexpr std::uint64_t indexedOffset = drawRecords * drawRecordWords * sizeof(std::uint32_t);
    std::array<std::uint32_t, drawRecords * drawRecordWords + indexedRecords * indexedRecordWords> records{};
    for (std::uint32_t index = 0; index < drawRecords; ++index)
    {
        const rachis_draw_indirect_arguments arguments{6, 1, 0, 0};
        records[index * drawRecordWords] = index;
        std::memcpy(&records[index * drawRecordWords + 1], &arguments, sizeof(arguments));
    }
    for (std::uint32_t index = 0; index < indexedRecords; ++index)
    {
        const rachis_draw_indexed_indirect_arguments arguments{6, 1, 0, 6, 0};
        const auto first = drawRecords * drawRecordWords + index * indexedRecordWords;
        records[first] = index;
        std::memcpy(&records[first + 1], &arguments, sizeof(arguments));
    }
    const auto vertexBuffer =
        createBuffer(device.get(), sizeof(cellVertices), RACHIS_BUFFER_VERTEX, RACHIS_MEMORY_UPLOAD, error);
    const auto indexBuffer =
        createBuffer(device.get(), sizeof(quadIndices), RACHIS_BUFFER_INDEX, RACHIS_MEMORY_UPLOAD, error);
    const auto argumentBuffer =
        createBuffer(device.get(), sizeof(records), RACHIS_BUFFER_INDIRECT, RACHIS_MEMORY_UPLOAD, error);
    const auto countBuffer =
        createBuffer(device.get(),
                     16,
                     RACHIS_BUFFER_STORAGE | RACHIS_BUFFER_COPY_DESTINATION | RACHIS_BUFFER_INDIRECT,
                     RACHIS_MEMORY_DEVICE_LOCAL,
                     error);
    writeWords(device.get(), vertexBuffer, cellVertices, error);
    writeWords(device.get(), indexBuffer, quadIndices, error);
    writeWords(device.get(), argumentBuffer, records, error);
    const auto cells = createCellTarget(device.get(), error);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
    std::vector<rachis_buffer_barrier> toFill{bufferBarrier(countBuffer,
                                                            RACHIS_STAGE_ALL,
                                                            RACHIS_ACCESS_COMMON,
                                                            RACHIS_STAGE_COMPUTE_SHADER,
                                                            RACHIS_ACCESS_UNORDERED_ACCESS)};
    for (const auto buffer : colorBuffers)
    {
        toFill.push_back(bufferBarrier(buffer,
                                       RACHIS_STAGE_ALL,
                                       RACHIS_ACCESS_COMMON,
                                       RACHIS_STAGE_COMPUTE_SHADER,
                                       RACHIS_ACCESS_UNORDERED_ACCESS));
    }
    requireOk(recordBarriers(
                  device.get(), command, toFill.data(), static_cast<std::uint32_t>(toFill.size()), nullptr, 0, error),
              error);
    CHECK(rachis_command_fill_buffer(nullptr, command, countBuffer, 0, 4, 3, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_fill_buffer(device.get(), 0, countBuffer, 0, 4, 3, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_fill_buffer(device.get(), command, pipeline, 0, 4, 3, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_fill_buffer(device.get(), command, countBuffer, 2, 4, 3, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_fill_buffer(device.get(), command, countBuffer, 0, 0, 3, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_fill_buffer(device.get(), command, countBuffer, 0, 20, 3, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_fill_buffer(device.get(), command, vertexBuffer, 0, 4, 3, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_command_fill_buffer(device.get(), command, countBuffer, 0, RACHIS_WHOLE_SIZE, 0xFFFFFFFFU, &error),
              error);
    const auto refill = bufferBarrier(countBuffer,
                                      RACHIS_STAGE_COMPUTE_SHADER,
                                      RACHIS_ACCESS_UNORDERED_ACCESS,
                                      RACHIS_STAGE_COMPUTE_SHADER,
                                      RACHIS_ACCESS_UNORDERED_ACCESS);
    requireOk(recordBarriers(device.get(), command, &refill, 1, nullptr, 0, error), error);
    requireOk(rachis_command_fill_buffer(device.get(), command, countBuffer, 0, sizeof(std::uint32_t), 3, &error),
              error);
    for (std::size_t index = 0; index < colorBuffers.size(); ++index)
    {
        requireOk(
            rachis_command_fill_buffer(
                device.get(), command, colorBuffers[index], 0, RACHIS_WHOLE_SIZE, packTexel(colors[index]), &error),
            error);
    }
    std::vector<rachis_buffer_barrier> toRead{bufferBarrier(countBuffer,
                                                            RACHIS_STAGE_COMPUTE_SHADER,
                                                            RACHIS_ACCESS_UNORDERED_ACCESS,
                                                            RACHIS_STAGE_INDIRECT,
                                                            RACHIS_ACCESS_INDIRECT_ARGUMENT)};
    for (const auto buffer : colorBuffers)
    {
        toRead.push_back(bufferBarrier(buffer,
                                       RACHIS_STAGE_COMPUTE_SHADER,
                                       RACHIS_ACCESS_UNORDERED_ACCESS,
                                       RACHIS_STAGE_PIXEL_SHADER,
                                       RACHIS_ACCESS_SHADER_RESOURCE));
    }
    requireOk(recordBarriers(
                  device.get(), command, toRead.data(), static_cast<std::uint32_t>(toRead.size()), nullptr, 0, error),
              error);

    beginCellRendering(device.get(), command, cells, error);
    CHECK(rachis_command_fill_buffer(device.get(), command, countBuffer, 0, 4, 3, &error) ==
          RACHIS_STATUS_INVALID_STATE);
    requireOk(rachis_command_bind_graphics_pipeline(device.get(), command, pipeline, &error), error);
    requireOk(rachis_command_bind_graphics_set(device.get(), command, 0, set, &error), error);
    const rachis_vertex_buffer_binding vertexBinding{vertexBuffer, 0};
    requireOk(rachis_command_bind_vertex_buffers(device.get(), command, 0, &vertexBinding, 1, &error), error);
    requireOk(
        rachis_command_bind_index_buffer(device.get(), command, indexBuffer, RACHIS_INDEX_FORMAT_UINT16, 0, &error),
        error);

    auto draw = indirectDrawDesc(argumentBuffer);
    draw.max_draw_count = counted ? drawRecords : drawRecords - 1;
    draw.count_buffer = counted ? countBuffer : 0;
    draw.draw_index_prefix = 1;
    const auto rejectsDraw = [&](const rachis_indirect_draw_desc& desc, rachis_status expected) {
        CHECK(rachis_command_draw_indirect(device.get(), command, &desc, &error) == expected);
        CHECK(error.code == expected);
    };
    CHECK(rachis_command_draw_indirect(device.get(), command, nullptr, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_draw_indirect(nullptr, command, &draw, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_draw_indirect(device.get(), 0, &draw, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    auto rejected = draw;
    rejected.struct_size = sizeof(rejected) - 1;
    rejectsDraw(rejected, RACHIS_STATUS_ABI_MISMATCH);
    rejected = draw;
    rejected.abi_version = RACHIS_ABI_VERSION - 1;
    rejectsDraw(rejected, RACHIS_STATUS_ABI_MISMATCH);
    rejected = draw;
    rejected.draw_index_prefix = 2;
    rejectsDraw(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = draw;
    rejected.argument_buffer = 0;
    rejectsDraw(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected.argument_buffer = set;
    rejectsDraw(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = draw;
    rejected.count_buffer = pipeline;
    rejectsDraw(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = draw;
    rejected.stride = (drawRecordWords - 1) * sizeof(std::uint32_t);
    rejectsDraw(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = draw;
    rejected.argument_offset = 2;
    rejectsDraw(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = draw;
    rejected.max_draw_count = drawRecords + 3;
    rejectsDraw(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = draw;
    rejected.draw_index_prefix = 0;
    rejectsDraw(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    if (counted)
    {
        rejected = draw;
        rejected.count_offset = 2;
        rejectsDraw(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
        rejected.count_offset = 16;
        rejectsDraw(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    }
    requireOk(rachis_command_draw_indirect(device.get(), command, &draw, &error), error);

    auto indexedDraw = indirectDrawDesc(argumentBuffer);
    indexedDraw.argument_offset = indexedOffset;
    indexedDraw.max_draw_count = indexedRecords;
    indexedDraw.stride = indexedRecordWords * sizeof(std::uint32_t);
    indexedDraw.draw_index_prefix = 1;
    CHECK(rachis_command_draw_indexed_indirect(device.get(), command, nullptr, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = indexedDraw;
    rejected.draw_index_prefix = 3;
    CHECK(rachis_command_draw_indexed_indirect(device.get(), command, &rejected, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = indexedDraw;
    rejected.max_draw_count = indexedRecords + 1;
    CHECK(rachis_command_draw_indexed_indirect(device.get(), command, &rejected, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_command_draw_indexed_indirect(device.get(), command, &indexedDraw, &error), error);
    endCellRendering(device.get(), command, cells, error);
    if (updateAfterBind)
    {
        requireOk(rachis_binding_set_update(device.get(), set, &updateDesc, &error), error);
    }
    submitAndWait(device.get(), command, error);

    const CellRow drawnRow{redTexel, greenTexel, whiteTexel, clearTexel};
    const CellRow indexedRow{redTexel, greenTexel, clearTexel, clearTexel};
    checkCells(readCells(device.get(), cells, error), {drawnRow, drawnRow, indexedRow, indexedRow});

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    destroyCellTarget(device.get(), cells, error);
    for (const auto buffer : {vertexBuffer, indexBuffer, argumentBuffer, countBuffer})
    {
        requireOk(rachis_buffer_destroy(device.get(), buffer, &error), error);
    }
    requireOk(rachis_pipeline_destroy(device.get(), pipeline, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), plainLayout, &error), error);
    requireOk(rachis_binding_set_destroy(device.get(), set, &error), error);
    for (const auto buffer : colorBuffers)
    {
        requireOk(rachis_buffer_destroy(device.get(), buffer, &error), error);
    }
    requireOk(rachis_binding_pool_destroy(device.get(), bindingPool, &error), error);
    requireOk(rachis_binding_layout_destroy(device.get(), bindingLayout, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), pixel, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), vertex, &error), error);
    requireNoValidationErrors(device.get(), error);
}

TEST_CASE("C ABI mesh pipelines dispatch directly, through counted indirect records, and from a task shader",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][graphics][mesh][indirect]")
{
    const auto meshBytes = loadFile(RACHIS_CABI_DRAW_INDEX_MESH_PATH);
    const auto taskBytes = loadFile(RACHIS_CABI_DRAW_INDEX_TASK_PATH);
    const auto taskMeshBytes = loadFile(RACHIS_CABI_DRAW_INDEX_TASK_MESH_PATH);
    const auto pixelBytes = loadFile(RACHIS_CABI_DRAW_INDEX_PIXEL_PATH);
    REQUIRE_FALSE(meshBytes.empty());
    REQUIRE_FALSE(taskBytes.empty());
    REQUIRE_FALSE(taskMeshBytes.empty());
    REQUIRE_FALSE(pixelBytes.empty());

    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    const auto capabilities = queryCapabilities(device.get(), error);
    if (RACHIS_SMOKE_BACKEND == RACHIS_BACKEND_D3D12)
    {
        REQUIRE(capabilities.mesh_shaders == 1);
        REQUIRE(capabilities.task_shaders == 1);
    }
    if (capabilities.mesh_shaders == 0 || capabilities.shader_draw_index == 0 || capabilities.multi_draw_indirect == 0)
    {
        SKIP("The device lacks mesh shaders, the draw index, or multi-draw indirect.");
    }
    const bool counted = capabilities.draw_indirect_count != 0;
    const bool taskShaders = capabilities.task_shaders != 0;
    CHECK(capabilities.max_mesh_workgroup_count[0] >= textureSize);
    CHECK(capabilities.max_mesh_output_vertices >= 4);
    CHECK(capabilities.max_mesh_output_primitives >= 2);

    auto meshDesc = shaderModuleDesc(RACHIS_SHADER_MESH, meshBytes, "meshMain");
    meshDesc.draw_index = 1;
    rachis_handle rejectedShader = 0;
    meshDesc.workgroup_y = 0;
    CHECK(rachis_shader_module_create(device.get(), &meshDesc, &rejectedShader, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rejectedShader == 0);
    meshDesc.workgroup_y = 1;
    const auto mesh = createShaderModule(device.get(), meshDesc, error);
    auto pixelDesc = shaderModuleDesc(RACHIS_SHADER_PIXEL, pixelBytes, "recordPixel");
    pixelDesc.workgroup_x = 1;
    CHECK(rachis_shader_module_create(device.get(), &pixelDesc, &rejectedShader, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rejectedShader == 0);
    pixelDesc.workgroup_x = 0;
    const auto pixel = createShaderModule(device.get(), pixelDesc, error);

    rachis_handle plainLayout = 0;
    rachis_handle pipelineLayout = 0;
    requireOk(createDrawIndexLayout(device.get(), nullptr, 0, 0, plainLayout, error), error);
    requireOk(createDrawIndexLayout(device.get(), nullptr, 0, 1, pipelineLayout, error), error);
    const auto target = opaqueTarget();
    rachis_graphics_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.pixel_shader = pixel;
    pipelineDesc.topology = RACHIS_TOPOLOGY_TRIANGLE_LIST;
    pipelineDesc.cull_mode = RACHIS_CULL_NONE;
    pipelineDesc.color_targets = &target;
    pipelineDesc.color_target_count = 1;
    rachis_handle rejectedPipeline = 0;
    CHECK(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &rejectedPipeline, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rejectedPipeline == 0);
    pipelineDesc.mesh_shader = plainLayout;
    CHECK(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &rejectedPipeline, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    pipelineDesc.mesh_shader = mesh;
    pipelineDesc.vertex_shader = pixel;
    CHECK(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &rejectedPipeline, &error) != RACHIS_STATUS_OK);
    CHECK(rejectedPipeline == 0);
    pipelineDesc.vertex_shader = 0;
    pipelineDesc.layout = plainLayout;
    CHECK(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &rejectedPipeline, &error) ==
          RACHIS_STATUS_VALIDATION_ERROR);
    CHECK(rejectedPipeline == 0);
    pipelineDesc.layout = pipelineLayout;
    rachis_handle meshPipeline = 0;
    requireOk(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &meshPipeline, &error), error);

    rachis_handle task = 0;
    rachis_handle taskMesh = 0;
    rachis_handle taskPipeline = 0;
    if (taskShaders)
    {
        task = createShaderModule(device.get(), shaderModuleDesc(RACHIS_SHADER_TASK, taskBytes, "taskMain"), error);
        taskMesh = createShaderModule(
            device.get(), shaderModuleDesc(RACHIS_SHADER_MESH, taskMeshBytes, "meshFromTask"), error);
        pipelineDesc.task_shader = task;
        pipelineDesc.mesh_shader = 0;
        CHECK(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &rejectedPipeline, &error) ==
              RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rejectedPipeline == 0);
        pipelineDesc.mesh_shader = taskMesh;
        requireOk(rachis_graphics_pipeline_create(device.get(), &pipelineDesc, &taskPipeline, &error), error);
    }

    constexpr std::uint32_t recordWords = 4;
    constexpr std::uint32_t recordCount = 4;
    std::array<std::uint32_t, recordWords * recordCount> records{};
    for (std::uint32_t index = 0; index < recordCount; ++index)
    {
        const rachis_dispatch_indirect_arguments arguments{index + 1, 1, 1};
        records[index * recordWords] = index;
        std::memcpy(&records[index * recordWords + 1], &arguments, sizeof(arguments));
    }
    const auto argumentBuffer =
        createBuffer(device.get(), sizeof(records), RACHIS_BUFFER_INDIRECT, RACHIS_MEMORY_UPLOAD, error);
    writeWords(device.get(), argumentBuffer, records, error);
    const auto countBuffer =
        createBuffer(device.get(),
                     sizeof(std::uint32_t),
                     RACHIS_BUFFER_STORAGE | RACHIS_BUFFER_COPY_DESTINATION | RACHIS_BUFFER_INDIRECT,
                     RACHIS_MEMORY_DEVICE_LOCAL,
                     error);
    const auto cells = createCellTarget(device.get(), error);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
    const auto toFill = bufferBarrier(countBuffer,
                                      RACHIS_STAGE_ALL,
                                      RACHIS_ACCESS_COMMON,
                                      RACHIS_STAGE_COMPUTE_SHADER,
                                      RACHIS_ACCESS_UNORDERED_ACCESS);
    requireOk(recordBarriers(device.get(), command, &toFill, 1, nullptr, 0, error), error);
    requireOk(rachis_command_fill_buffer(device.get(), command, countBuffer, 0, RACHIS_WHOLE_SIZE, 3, &error), error);
    const auto toCount = bufferBarrier(countBuffer,
                                       RACHIS_STAGE_COMPUTE_SHADER,
                                       RACHIS_ACCESS_UNORDERED_ACCESS,
                                       RACHIS_STAGE_INDIRECT,
                                       RACHIS_ACCESS_INDIRECT_ARGUMENT);
    requireOk(recordBarriers(device.get(), command, &toCount, 1, nullptr, 0, error), error);

    beginCellRendering(device.get(), command, cells, error);
    requireOk(rachis_command_bind_graphics_pipeline(device.get(), command, meshPipeline, &error), error);
    CHECK(rachis_command_dispatch_mesh(nullptr, command, 1, 1, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_dispatch_mesh(device.get(), 0, 1, 1, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_command_dispatch_mesh(device.get(), command, textureSize, 1, 1, &error), error);

    rachis_indirect_mesh_dispatch_desc dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = RACHIS_ABI_VERSION;
    dispatch.argument_buffer = argumentBuffer;
    dispatch.max_dispatch_count = counted ? recordCount : recordCount - 1;
    dispatch.count_buffer = counted ? countBuffer : 0;
    dispatch.draw_index_prefix = 1;
    CHECK(rachis_command_dispatch_mesh_indirect(device.get(), command, nullptr, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    auto rejected = dispatch;
    rejected.struct_size = sizeof(rejected) + 8;
    CHECK(rachis_command_dispatch_mesh_indirect(device.get(), command, &rejected, &error) ==
          RACHIS_STATUS_ABI_MISMATCH);
    rejected = dispatch;
    rejected.draw_index_prefix = 2;
    CHECK(rachis_command_dispatch_mesh_indirect(device.get(), command, &rejected, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = dispatch;
    rejected.argument_buffer = meshPipeline;
    CHECK(rachis_command_dispatch_mesh_indirect(device.get(), command, &rejected, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = dispatch;
    rejected.draw_index_prefix = 0;
    CHECK(rachis_command_dispatch_mesh_indirect(device.get(), command, &rejected, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = dispatch;
    rejected.max_dispatch_count = recordCount + 1;
    CHECK(rachis_command_dispatch_mesh_indirect(device.get(), command, &rejected, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_command_dispatch_mesh_indirect(device.get(), command, &dispatch, &error), error);
    if (taskShaders)
    {
        requireOk(rachis_command_bind_graphics_pipeline(device.get(), command, taskPipeline, &error), error);
        requireOk(rachis_command_dispatch_mesh(device.get(), command, 1, 1, 1, &error), error);
    }
    endCellRendering(device.get(), command, cells, error);
    submitAndWait(device.get(), command, error);

    const auto lastCell = taskShaders ? magentaTexel : clearTexel;
    checkCells(readCells(device.get(), cells, error),
               {CellRow{redTexel, greenTexel, blueTexel, clearTexel},
                CellRow{redTexel, greenTexel, blueTexel, clearTexel},
                CellRow{redTexel, clearTexel, blueTexel, clearTexel},
                CellRow{redTexel, clearTexel, clearTexel, lastCell}});

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    destroyCellTarget(device.get(), cells, error);
    requireOk(rachis_buffer_destroy(device.get(), argumentBuffer, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), countBuffer, &error), error);
    if (taskShaders)
    {
        requireOk(rachis_pipeline_destroy(device.get(), taskPipeline, &error), error);
        requireOk(rachis_shader_module_destroy(device.get(), taskMesh, &error), error);
        requireOk(rachis_shader_module_destroy(device.get(), task, &error), error);
    }
    requireOk(rachis_pipeline_destroy(device.get(), meshPipeline, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), plainLayout, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), pixel, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), mesh, &error), error);
    requireNoValidationErrors(device.get(), error);
}

TEST_CASE("C ABI runs indirect dispatch batches with a GPU count and clears storage textures",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][compute][indirect]")
{
    const auto bytecode = loadFile(RACHIS_CABI_INDIRECT_DISPATCH_PATH);
    REQUIRE_FALSE(bytecode.empty());
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    const auto capabilities = queryCapabilities(device.get(), error);
    REQUIRE(capabilities.indirect_dispatch == 1);
    if (RACHIS_SMOKE_BACKEND == RACHIS_BACKEND_D3D12)
    {
        REQUIRE(capabilities.multi_dispatch_indirect == 1);
        REQUIRE(capabilities.dispatch_indirect_count == 1);
    }
    const bool multi = capabilities.multi_dispatch_indirect != 0;
    const bool counted = multi && capabilities.dispatch_indirect_count != 0;

    const rachis_shader_binding_requirement requirement{0, 0, RACHIS_BINDING_STORAGE_BUFFER, 1};
    auto shaderDesc = shaderModuleDesc(RACHIS_SHADER_COMPUTE, bytecode, "countGroups");
    shaderDesc.bindings = &requirement;
    shaderDesc.binding_count = 1;
    const auto shader = createShaderModule(device.get(), shaderDesc, error);
    const rachis_binding_layout_entry entry{0, RACHIS_BINDING_STORAGE_BUFFER, RACHIS_SHADER_COMPUTE, 1, 0};
    rachis_binding_layout_desc bindingLayoutDesc{};
    bindingLayoutDesc.struct_size = sizeof(bindingLayoutDesc);
    bindingLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    bindingLayoutDesc.entries = &entry;
    bindingLayoutDesc.entry_count = 1;
    rachis_handle bindingLayout = 0;
    requireOk(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &bindingLayout, &error), error);
    const rachis_binding_pool_size poolSize{RACHIS_BINDING_STORAGE_BUFFER, 1};
    rachis_binding_pool_desc bindingPoolDesc{};
    bindingPoolDesc.struct_size = sizeof(bindingPoolDesc);
    bindingPoolDesc.abi_version = RACHIS_ABI_VERSION;
    bindingPoolDesc.max_binding_sets = 1;
    bindingPoolDesc.sizes = &poolSize;
    bindingPoolDesc.size_count = 1;
    rachis_handle bindingPool = 0;
    requireOk(rachis_binding_pool_create(device.get(), &bindingPoolDesc, &bindingPool, &error), error);
    constexpr std::uint64_t counterBytes = 16;
    const auto counter =
        createBuffer(device.get(),
                     counterBytes,
                     RACHIS_BUFFER_STORAGE | RACHIS_BUFFER_COPY_SOURCE | RACHIS_BUFFER_COPY_DESTINATION,
                     RACHIS_MEMORY_DEVICE_LOCAL,
                     error);
    const rachis_buffer_binding counterWrite{0, 0, counter, 0, counterBytes, sizeof(std::uint32_t)};
    rachis_binding_set_desc setDesc{};
    setDesc.struct_size = sizeof(setDesc);
    setDesc.abi_version = RACHIS_ABI_VERSION;
    setDesc.layout = bindingLayout;
    setDesc.pool = bindingPool;
    setDesc.writes = &counterWrite;
    setDesc.write_count = 1;
    rachis_handle set = 0;
    requireOk(rachis_binding_set_create(device.get(), &setDesc, &set, &error), error);
    rachis_handle pipelineLayout = 0;
    requireOk(createDrawIndexLayout(device.get(), &bindingLayout, 1, 0, pipelineLayout, error), error);
    rachis_compute_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.shader = shader;
    rachis_handle pipeline = 0;
    requireOk(rachis_compute_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error), error);

    constexpr std::uint32_t recordWords = 3;
    constexpr std::uint32_t batchRecords = 4;
    std::array<std::uint32_t, recordWords*(batchRecords + 1)> records{};
    for (std::uint32_t index = 0; index <= batchRecords; ++index)
    {
        const rachis_dispatch_indirect_arguments arguments{1U << index, 1, 1};
        std::memcpy(&records[index * recordWords], &arguments, sizeof(arguments));
    }
    constexpr std::uint64_t recordBytes = recordWords * sizeof(std::uint32_t);
    const auto argumentBuffer =
        createBuffer(device.get(), sizeof(records), RACHIS_BUFFER_INDIRECT, RACHIS_MEMORY_UPLOAD, error);
    writeWords(device.get(), argumentBuffer, records, error);
    const auto countBuffer =
        createBuffer(device.get(),
                     sizeof(std::uint32_t),
                     RACHIS_BUFFER_STORAGE | RACHIS_BUFFER_COPY_DESTINATION | RACHIS_BUFFER_INDIRECT,
                     RACHIS_MEMORY_DEVICE_LOCAL,
                     error);
    const auto counterReadback =
        createBuffer(device.get(), counterBytes, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);

    const auto storageTexture =
        createTexture(device.get(),
                      RACHIS_FORMAT_R32_UINT,
                      textureSize,
                      textureSize,
                      RACHIS_TEXTURE_STORAGE | RACHIS_TEXTURE_COPY_DESTINATION | RACHIS_TEXTURE_COPY_SOURCE,
                      error);
    rachis_texture_view_desc storageViewDesc{};
    storageViewDesc.struct_size = sizeof(storageViewDesc);
    storageViewDesc.abi_version = RACHIS_ABI_VERSION;
    storageViewDesc.texture = storageTexture;
    storageViewDesc.usage = RACHIS_VIEW_USAGE_STORAGE;
    storageViewDesc.dimension = RACHIS_VIEW_2D;
    storageViewDesc.aspect = RACHIS_ASPECT_COLOR;
    storageViewDesc.mip_level_count = 1;
    storageViewDesc.array_layer_count = 1;
    rachis_handle storageView = 0;
    requireOk(rachis_texture_view_create(device.get(), &storageViewDesc, &storageView, &error), error);
    auto textureRegion = wholeTextureRegion(textureSize, textureSize);
    rachis_texture_transfer_layout transfer{};
    transfer.struct_size = sizeof(transfer);
    requireOk(rachis_texture_query_transfer_layout(device.get(), storageTexture, &textureRegion, &transfer, &error),
              error);
    textureRegion.bytes_per_row = transfer.bytes_per_row;
    const auto textureReadback = createBuffer(
        device.get(), transfer.required_buffer_size, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_COMPUTE, pool, error);
    const std::array<rachis_buffer_barrier, 2> toFill{
        bufferBarrier(counter,
                      RACHIS_STAGE_ALL,
                      RACHIS_ACCESS_COMMON,
                      RACHIS_STAGE_COMPUTE_SHADER,
                      RACHIS_ACCESS_UNORDERED_ACCESS),
        bufferBarrier(countBuffer,
                      RACHIS_STAGE_ALL,
                      RACHIS_ACCESS_COMMON,
                      RACHIS_STAGE_COMPUTE_SHADER,
                      RACHIS_ACCESS_UNORDERED_ACCESS),
    };
    const auto toClear = textureBarrier(storageTexture,
                                        RACHIS_STAGE_ALL,
                                        RACHIS_ACCESS_NONE,
                                        RACHIS_LAYOUT_UNDEFINED,
                                        RACHIS_STAGE_COMPUTE_SHADER,
                                        RACHIS_ACCESS_UNORDERED_ACCESS,
                                        RACHIS_LAYOUT_UNORDERED_ACCESS);
    requireOk(recordBarriers(device.get(), command, toFill.data(), 2, &toClear, 1, error), error);
    requireOk(rachis_command_fill_buffer(device.get(), command, counter, 0, RACHIS_WHOLE_SIZE, 0, &error), error);
    requireOk(rachis_command_fill_buffer(device.get(), command, countBuffer, 0, RACHIS_WHOLE_SIZE, 3, &error), error);

    rachis_storage_texture_clear_value clear{};
    clear.type = RACHIS_STORAGE_TEXTURE_CLEAR_UINT32;
    clear.uint32[0] = 0xC0FFEEU;
    CHECK(rachis_command_clear_storage_texture(device.get(), command, storageView, nullptr, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_clear_storage_texture(nullptr, command, storageView, &clear, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_clear_storage_texture(device.get(), command, storageTexture, &clear, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    auto rejectedClear = clear;
    rejectedClear.type = RACHIS_STORAGE_TEXTURE_CLEAR_SINT32 + 1;
    CHECK(rachis_command_clear_storage_texture(device.get(), command, storageView, &rejectedClear, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    rejectedClear.type = RACHIS_STORAGE_TEXTURE_CLEAR_FLOAT32;
    CHECK(rachis_command_clear_storage_texture(device.get(), command, storageView, &rejectedClear, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_command_clear_storage_texture(device.get(), command, storageView, &clear, &error), error);

    const std::array<rachis_buffer_barrier, 2> afterFill{
        bufferBarrier(counter,
                      RACHIS_STAGE_COMPUTE_SHADER,
                      RACHIS_ACCESS_UNORDERED_ACCESS,
                      RACHIS_STAGE_COMPUTE_SHADER,
                      RACHIS_ACCESS_UNORDERED_ACCESS),
        bufferBarrier(countBuffer,
                      RACHIS_STAGE_COMPUTE_SHADER,
                      RACHIS_ACCESS_UNORDERED_ACCESS,
                      RACHIS_STAGE_INDIRECT,
                      RACHIS_ACCESS_INDIRECT_ARGUMENT),
    };
    requireOk(recordBarriers(device.get(), command, afterFill.data(), 2, nullptr, 0, error), error);
    requireOk(rachis_command_bind_compute_pipeline(device.get(), command, pipeline, &error), error);
    requireOk(rachis_command_bind_compute_set(device.get(), command, 0, set, &error), error);

    rachis_indirect_dispatch_desc batch{};
    batch.struct_size = sizeof(batch);
    batch.abi_version = RACHIS_ABI_VERSION;
    batch.argument_buffer = argumentBuffer;
    batch.max_dispatch_count = counted ? batchRecords : batchRecords - 1;
    batch.count_buffer = counted ? countBuffer : 0;
    const auto rejectsDispatch = [&](const rachis_indirect_dispatch_desc& desc, rachis_status expected) {
        CHECK(rachis_command_dispatch_indirect(device.get(), command, &desc, &error) == expected);
        CHECK(error.code == expected);
    };
    CHECK(rachis_command_dispatch_indirect(device.get(), command, nullptr, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_dispatch_indirect(nullptr, command, &batch, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    auto rejected = batch;
    rejected.struct_size = 0;
    rejectsDispatch(rejected, RACHIS_STATUS_ABI_MISMATCH);
    rejected = batch;
    rejected.argument_buffer = 0;
    rejectsDispatch(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = batch;
    rejected.count_buffer = set;
    rejectsDispatch(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected = batch;
    rejected.max_dispatch_count = 1;
    rejected.stride = recordBytes - sizeof(std::uint32_t);
    rejectsDispatch(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected.stride = recordBytes + 2;
    rejectsDispatch(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected.stride = 0;
    rejected.argument_offset = 2;
    rejectsDispatch(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    rejected.argument_offset = sizeof(records);
    rejectsDispatch(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    if (multi)
    {
        rejected = batch;
        rejected.count_buffer = 0;
        rejected.max_dispatch_count = batchRecords + 2;
        rejectsDispatch(rejected, RACHIS_STATUS_INVALID_ARGUMENT);
    }
    else
    {
        rejected = batch;
        rejected.max_dispatch_count = 2;
        rejectsDispatch(rejected, RACHIS_STATUS_UNSUPPORTED);
    }
    if (multi)
    {
        requireOk(rachis_command_dispatch_indirect(device.get(), command, &batch, &error), error);
    }
    else
    {
        for (std::uint32_t index = 0; index + 1 < batchRecords; ++index)
        {
            auto single = batch;
            single.argument_offset = index * recordBytes;
            single.max_dispatch_count = 1;
            requireOk(rachis_command_dispatch_indirect(device.get(), command, &single, &error), error);
        }
    }
    auto single = batch;
    single.argument_offset = batchRecords * recordBytes;
    single.max_dispatch_count = 1;
    single.count_buffer = 0;
    requireOk(rachis_command_dispatch_indirect(device.get(), command, &single, &error), error);

    const auto counterToCopy = bufferBarrier(counter,
                                             RACHIS_STAGE_COMPUTE_SHADER,
                                             RACHIS_ACCESS_UNORDERED_ACCESS,
                                             RACHIS_STAGE_COPY,
                                             RACHIS_ACCESS_COPY_SOURCE);
    const auto textureToCopy = textureBarrier(storageTexture,
                                              RACHIS_STAGE_COMPUTE_SHADER,
                                              RACHIS_ACCESS_UNORDERED_ACCESS,
                                              RACHIS_LAYOUT_UNORDERED_ACCESS,
                                              RACHIS_STAGE_COPY,
                                              RACHIS_ACCESS_COPY_SOURCE,
                                              RACHIS_LAYOUT_COPY_SOURCE);
    requireOk(recordBarriers(device.get(), command, &counterToCopy, 1, &textureToCopy, 1, error), error);
    requireOk(rachis_command_copy_buffer(device.get(), command, counter, counterReadback, 0, 0, counterBytes, &error),
              error);
    requireOk(copyBufferTexture(device.get(), command, textureReadback, storageTexture, textureRegion, false, error),
              error);
    submitAndWait(device.get(), command, error);

    std::array<std::uint32_t, counterBytes / sizeof(std::uint32_t)> groupCounts{};
    requireOk(rachis_buffer_read(device.get(), counterReadback, 0, groupCounts.data(), counterBytes, &error), error);
    CHECK(groupCounts[0] == 1U + 2U + 4U + 16U);
    CHECK(groupCounts[1] == 0U);
    std::vector<std::uint8_t> texels(static_cast<std::size_t>(transfer.required_buffer_size));
    requireOk(rachis_buffer_read(device.get(), textureReadback, 0, texels.data(), texels.size(), &error), error);
    for (std::uint32_t y = 0; y < textureSize; ++y)
    {
        for (std::uint32_t x = 0; x < textureSize; ++x)
        {
            INFO("texel " << x << ", " << y);
            std::uint32_t value = 0;
            std::memcpy(&value,
                        texels.data() + static_cast<std::size_t>(y) * transfer.bytes_per_row + x * sizeof(value),
                        sizeof(value));
            CHECK(value == 0xC0FFEEU);
        }
    }

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_texture_view_destroy(device.get(), storageView, &error), error);
    requireOk(rachis_texture_destroy(device.get(), storageTexture, &error), error);
    requireOk(rachis_pipeline_destroy(device.get(), pipeline, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_binding_set_destroy(device.get(), set, &error), error);
    requireOk(rachis_binding_pool_destroy(device.get(), bindingPool, &error), error);
    requireOk(rachis_binding_layout_destroy(device.get(), bindingLayout, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), shader, &error), error);
    for (const auto buffer : {counter, argumentBuffer, countBuffer, counterReadback, textureReadback})
    {
        requireOk(rachis_buffer_destroy(device.get(), buffer, &error), error);
    }
    requireNoValidationErrors(device.get(), error);
}
#endif

#if defined(RACHIS_CABI_COMPUTE_SHADER_PATH)
TEST_CASE("C ABI compute pipelines run full subgroups of a required size",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][compute][subgroup]")
{
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    rachis_device_capabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    requireOk(rachis_device_query_capabilities(device.get(), &capabilities, &error), error);
    const bool supported = (capabilities.subgroup_required_size_stages & RACHIS_SHADER_COMPUTE) != 0;

    const auto writeBytecode = loadShader();
    REQUIRE_FALSE(writeBytecode.empty());
    const rachis_shader_binding_requirement writeRequirement{0, 0, RACHIS_BINDING_STORAGE_BUFFER, 1};
    rachis_shader_module_desc writeDesc{};
    writeDesc.struct_size = sizeof(writeDesc);
    writeDesc.abi_version = RACHIS_ABI_VERSION;
    writeDesc.stage = RACHIS_SHADER_COMPUTE;
    writeDesc.bytecode = writeBytecode.data();
    writeDesc.bytecode_size = writeBytecode.size();
    writeDesc.entry_point = "main";
    writeDesc.bindings = &writeRequirement;
    writeDesc.binding_count = 1;
    writeDesc.workgroup_x = valueCount;
    writeDesc.workgroup_y = 1;
    writeDesc.workgroup_z = 1;
    rachis_handle writeShader = 0;
    requireOk(rachis_shader_module_create(device.get(), &writeDesc, &writeShader, &error), error);
    const rachis_binding_layout_entry entry{0, RACHIS_BINDING_STORAGE_BUFFER, RACHIS_SHADER_COMPUTE, 1, 0};
    rachis_binding_layout_desc bindingLayoutDesc{};
    bindingLayoutDesc.struct_size = sizeof(bindingLayoutDesc);
    bindingLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    bindingLayoutDesc.entries = &entry;
    bindingLayoutDesc.entry_count = 1;
    rachis_handle bindingLayout = 0;
    requireOk(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &bindingLayout, &error), error);
    rachis_pipeline_layout_desc pipelineLayoutDesc{};
    pipelineLayoutDesc.struct_size = sizeof(pipelineLayoutDesc);
    pipelineLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineLayoutDesc.binding_layouts = &bindingLayout;
    pipelineLayoutDesc.binding_layout_count = 1;
    rachis_handle pipelineLayout = 0;
    requireOk(rachis_pipeline_layout_create(device.get(), &pipelineLayoutDesc, &pipelineLayout, &error), error);
    rachis_compute_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.shader = writeShader;
    pipelineDesc.required_subgroup_size = 3;
    rachis_handle pipeline = 0;
    CHECK(rachis_compute_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(pipeline == 0);
    if (!supported)
    {
        CHECK(capabilities.subgroup_required_size_stages == 0);
        pipelineDesc.required_subgroup_size = 4;
        CHECK(rachis_compute_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error) ==
              RACHIS_STATUS_UNSUPPORTED);
        CHECK(pipeline == 0);
    }
#ifdef RACHIS_CABI_SUBGROUP_SHADER_PREFIX
    if (supported)
    {
        REQUIRE(capabilities.subgroup_min_size > 0);
        REQUIRE(capabilities.subgroup_min_size <= capabilities.subgroup_max_size);
        const auto size = capabilities.subgroup_min_size;
        constexpr std::uint32_t laneCountThreads = 128;
        REQUIRE(laneCountThreads % size == 0);
        std::string path = RACHIS_CABI_SUBGROUP_SHADER_PREFIX;
        if (RACHIS_CABI_SUBGROUP_SIZE_IN_SHADER != 0)
        {
            path += std::to_string(size);
        }
        path += RACHIS_CABI_SUBGROUP_SHADER_EXTENSION;
        std::ifstream input{path, std::ios::binary};
        const std::vector<char> laneBytecode{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
        REQUIRE_FALSE(laneBytecode.empty());
        auto laneDesc = writeDesc;
        laneDesc.bytecode = laneBytecode.data();
        laneDesc.bytecode_size = laneBytecode.size();
        laneDesc.workgroup_x = laneCountThreads;
        rachis_handle laneShader = 0;
        requireOk(rachis_shader_module_create(device.get(), &laneDesc, &laneShader, &error), error);
        pipelineDesc.shader = laneShader;
        pipelineDesc.required_subgroup_size = size;
        requireOk(rachis_compute_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error), error);

        const rachis_binding_pool_size poolSize{RACHIS_BINDING_STORAGE_BUFFER, 1};
        rachis_binding_pool_desc bindingPoolDesc{};
        bindingPoolDesc.struct_size = sizeof(bindingPoolDesc);
        bindingPoolDesc.abi_version = RACHIS_ABI_VERSION;
        bindingPoolDesc.max_binding_sets = 1;
        bindingPoolDesc.sizes = &poolSize;
        bindingPoolDesc.size_count = 1;
        rachis_handle bindingPool = 0;
        requireOk(rachis_binding_pool_create(device.get(), &bindingPoolDesc, &bindingPool, &error), error);
        constexpr std::uint64_t laneBytes = laneCountThreads * 2 * sizeof(std::uint32_t);
        const auto lanes = createBuffer(device.get(),
                                        laneBytes,
                                        RACHIS_BUFFER_STORAGE | RACHIS_BUFFER_COPY_SOURCE,
                                        RACHIS_MEMORY_DEVICE_LOCAL,
                                        error);
        const auto readback =
            createBuffer(device.get(), laneBytes, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
        const rachis_buffer_binding laneWrite{0, 0, lanes, 0, laneBytes, 2 * sizeof(std::uint32_t)};
        rachis_binding_set_desc setDesc{};
        setDesc.struct_size = sizeof(setDesc);
        setDesc.abi_version = RACHIS_ABI_VERSION;
        setDesc.layout = bindingLayout;
        setDesc.pool = bindingPool;
        setDesc.writes = &laneWrite;
        setDesc.write_count = 1;
        rachis_handle set = 0;
        requireOk(rachis_binding_set_create(device.get(), &setDesc, &set, &error), error);

        rachis_handle pool = 0;
        const auto command = acquireCommand(device.get(), RACHIS_QUEUE_COMPUTE, pool, error);
        const auto toWrite = bufferBarrier(
            lanes, RACHIS_STAGE_ALL, RACHIS_ACCESS_COMMON, RACHIS_STAGE_COMPUTE_SHADER, RACHIS_ACCESS_UNORDERED_ACCESS);
        requireOk(recordBarriers(device.get(), command, &toWrite, 1, nullptr, 0, error), error);
        requireOk(rachis_command_bind_compute_pipeline(device.get(), command, pipeline, &error), error);
        requireOk(rachis_command_bind_compute_set(device.get(), command, 0, set, &error), error);
        requireOk(rachis_command_dispatch(device.get(), command, 1, 1, 1, &error), error);
        const auto toCopy = bufferBarrier(lanes,
                                          RACHIS_STAGE_COMPUTE_SHADER,
                                          RACHIS_ACCESS_UNORDERED_ACCESS,
                                          RACHIS_STAGE_COPY,
                                          RACHIS_ACCESS_COPY_SOURCE);
        requireOk(recordBarriers(device.get(), command, &toCopy, 1, nullptr, 0, error), error);
        requireOk(rachis_command_copy_buffer(device.get(), command, lanes, readback, 0, 0, laneBytes, &error), error);
        submitAndWait(device.get(), command, error);

        std::array<std::uint32_t, laneCountThreads * 2> laneCounts{};
        requireOk(rachis_buffer_read(device.get(), readback, 0, laneCounts.data(), laneBytes, &error), error);
        for (std::uint32_t thread = 0; thread < laneCountThreads; ++thread)
        {
            INFO("thread " << thread << " of subgroup size " << size);
            CHECK(laneCounts[thread * 2] == size);
            CHECK(laneCounts[thread * 2 + 1] == size);
        }

        requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
        requireOk(rachis_binding_set_destroy(device.get(), set, &error), error);
        requireOk(rachis_binding_pool_destroy(device.get(), bindingPool, &error), error);
        requireOk(rachis_buffer_destroy(device.get(), lanes, &error), error);
        requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
        requireOk(rachis_pipeline_destroy(device.get(), pipeline, &error), error);
        requireOk(rachis_shader_module_destroy(device.get(), laneShader, &error), error);
    }
#endif
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_binding_layout_destroy(device.get(), bindingLayout, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), writeShader, &error), error);
    requireNoValidationErrors(device.get(), error);
}
#endif

TEST_CASE("C ABI maps one range per buffer and validates mapped ranges",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    constexpr std::uint64_t bufferSize = 64;
    constexpr std::uint64_t mapOffset = 16;
    constexpr std::uint64_t mapSize = 32;
    auto error = makeError();
    auto device = createDevice(error);
    const auto upload = createBuffer(device.get(), bufferSize, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback =
        createBuffer(device.get(), bufferSize, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    const auto deviceLocal =
        createBuffer(device.get(), bufferSize, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_DEVICE_LOCAL, error);
    std::array<std::uint8_t, bufferSize> zeros{};
    requireOk(rachis_buffer_write(device.get(), upload, 0, zeros.data(), zeros.size(), &error), error);

    rachis_mapped_memory mapped{};
    mapped.struct_size = sizeof(mapped);
    CHECK(rachis_buffer_map(device.get(), upload, 0, RACHIS_WHOLE_SIZE, nullptr, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    mapped.struct_size = sizeof(mapped) - 1;
    CHECK(rachis_buffer_map(device.get(), upload, 0, RACHIS_WHOLE_SIZE, &mapped, &error) == RACHIS_STATUS_ABI_MISMATCH);
    mapped.struct_size = sizeof(mapped);

    struct Range
    {
        std::uint64_t offset;
        std::uint64_t size;
    };
    const std::array<Range, 7> rejected{{
        {0, 0},
        {bufferSize, RACHIS_WHOLE_SIZE},
        {bufferSize, 1},
        {0, bufferSize + 1},
        {8, bufferSize - 7},
        {RACHIS_WHOLE_SIZE, 1},
        {8, RACHIS_WHOLE_SIZE - 3},
    }};
    std::uint8_t sentinel = 0;
    for (const auto& range : rejected)
    {
        INFO("offset " << range.offset << ", size " << range.size);
        for (const auto target : {upload, readback, deviceLocal})
        {
            mapped.data = &sentinel;
            mapped.size = 1;
            CHECK(rachis_buffer_map(device.get(), target, range.offset, range.size, &mapped, &error) ==
                  RACHIS_STATUS_INVALID_ARGUMENT);
            CHECK(mapped.data == nullptr);
            CHECK(mapped.size == 0);
        }
    }
    mapped.data = &sentinel;
    mapped.size = 1;
    CHECK(rachis_buffer_map(device.get(), deviceLocal, 0, RACHIS_WHOLE_SIZE, &mapped, &error) ==
          RACHIS_STATUS_UNSUPPORTED);
    CHECK(mapped.data == nullptr);
    CHECK(mapped.size == 0);
    CHECK(rachis_buffer_map(device.get(), 0, 0, RACHIS_WHOLE_SIZE, &mapped, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_unmap(device.get(), 0, &error) == RACHIS_STATUS_INVALID_ARGUMENT);

    const rachis_mapped_range mappedRange{mapOffset, mapSize};
    CHECK(rachis_buffer_unmap(device.get(), upload, &error) == RACHIS_STATUS_INVALID_STATE);
    CHECK(rachis_buffer_flush_mapped_ranges(device.get(), upload, &mappedRange, 1, &error) ==
          RACHIS_STATUS_INVALID_STATE);
    CHECK(rachis_buffer_invalidate_mapped_ranges(device.get(), readback, &mappedRange, 1, &error) ==
          RACHIS_STATUS_INVALID_STATE);

    requireOk(rachis_buffer_map(device.get(), upload, mapOffset, mapSize, &mapped, &error), error);
    REQUIRE(mapped.data != nullptr);
    REQUIRE(mapped.size == mapSize);
    rachis_mapped_memory second{};
    second.struct_size = sizeof(second);
    CHECK(rachis_buffer_map(device.get(), upload, 0, 8, &second, &error) == RACHIS_STATUS_INVALID_STATE);
    CHECK(second.data == nullptr);
    CHECK(rachis_buffer_write(device.get(), upload, 0, zeros.data(), 8, &error) == RACHIS_STATUS_INVALID_STATE);
    CHECK(rachis_buffer_destroy(device.get(), upload, &error) == RACHIS_STATUS_INVALID_STATE);

    std::array<std::uint8_t, mapSize> expected{};
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        expected[index] = static_cast<std::uint8_t>(index + 1);
    }
    std::memcpy(mapped.data, expected.data(), expected.size());

    CHECK(rachis_buffer_flush_mapped_ranges(device.get(), upload, nullptr, 1, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_flush_mapped_ranges(device.get(), upload, &mappedRange, 0, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    const std::array<Range, 5> outsideMapping{{
        {0, 8},
        {mapOffset, RACHIS_WHOLE_SIZE},
        {mapOffset, mapSize + 1},
        {mapOffset, 0},
        {bufferSize, 1},
    }};
    for (const auto& range : outsideMapping)
    {
        INFO("offset " << range.offset << ", size " << range.size);
        const rachis_mapped_range outside{range.offset, range.size};
        CHECK(rachis_buffer_flush_mapped_ranges(device.get(), upload, &outside, 1, &error) ==
              RACHIS_STATUS_INVALID_ARGUMENT);
    }
    CHECK(rachis_buffer_invalidate_mapped_ranges(device.get(), upload, &mappedRange, 1, &error) ==
          RACHIS_STATUS_UNSUPPORTED);
    const std::array<rachis_mapped_range, 2> splitRanges{{{mapOffset, 8}, {mapOffset + 8, mapSize - 8}}};
    requireOk(rachis_buffer_flush_mapped_ranges(device.get(), upload, splitRanges.data(), 2, &error), error);
    requireOk(rachis_buffer_unmap(device.get(), upload, &error), error);
    CHECK(rachis_buffer_unmap(device.get(), upload, &error) == RACHIS_STATUS_INVALID_STATE);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_COPY, pool, error);
    requireOk(rachis_command_copy_buffer(device.get(), command, upload, readback, 0, 0, bufferSize, &error), error);
    submitAndWait(device.get(), command, error);

    requireOk(rachis_buffer_map(device.get(), readback, 0, RACHIS_WHOLE_SIZE, &mapped, &error), error);
    REQUIRE(mapped.data != nullptr);
    REQUIRE(mapped.size == bufferSize);
    std::array<std::uint8_t, bufferSize> copied{};
    CHECK(rachis_buffer_read(device.get(), readback, 0, copied.data(), copied.size(), &error) ==
          RACHIS_STATUS_INVALID_STATE);
    const rachis_mapped_range wholeRange{0, RACHIS_WHOLE_SIZE};
    CHECK(rachis_buffer_flush_mapped_ranges(device.get(), readback, &wholeRange, 1, &error) ==
          RACHIS_STATUS_UNSUPPORTED);
    requireOk(rachis_buffer_invalidate_mapped_ranges(device.get(), readback, &wholeRange, 1, &error), error);
    std::memcpy(copied.data(), mapped.data, copied.size());
    requireOk(rachis_buffer_unmap(device.get(), readback, &error), error);
    for (std::size_t index = 0; index < copied.size(); ++index)
    {
        INFO("byte " << index);
        const bool written = index >= mapOffset && index < mapOffset + mapSize;
        CHECK(copied[index] == (written ? expected[index - mapOffset] : 0));
    }

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), deviceLocal, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
    CHECK(rachis_buffer_map(device.get(), upload, 0, RACHIS_WHOLE_SIZE, &mapped, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_unmap(device.get(), upload, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_flush_mapped_ranges(device.get(), upload, &wholeRange, 1, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
}

TEST_CASE("C ABI reports mapping and vertex input capabilities",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    auto error = makeError();
    auto device = createDevice(error);
    rachis_device_capabilities capabilities{};
    CHECK(rachis_device_query_capabilities(device.get(), nullptr, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_device_query_capabilities(device.get(), &capabilities, &error) == RACHIS_STATUS_ABI_MISMATCH);
    capabilities.struct_size = sizeof(capabilities);
    capabilities.reserved0 = 7;
    requireOk(rachis_device_query_capabilities(device.get(), &capabilities, &error), error);

    CHECK(capabilities.unified_memory <= 1);
    CHECK(capabilities.persistent_mapping == 1);
    CHECK(capabilities.mapped_memory_host_coherent <= 1);
    CHECK(capabilities.mapped_memory_non_coherent_atom_size >= 1);
    if (RACHIS_SMOKE_BACKEND != RACHIS_BACKEND_VULKAN)
    {
        CHECK(capabilities.mapped_memory_host_coherent == 1);
        CHECK(capabilities.mapped_memory_non_coherent_atom_size == 1);
    }
    constexpr std::uint64_t knownFormats = (std::uint64_t{1} << (RACHIS_VERTEX_FORMAT_FLOAT16X4 + 1)) - 1;
    CHECK((capabilities.vertex_formats & ~knownFormats) == 0);
    const std::array<std::uint32_t, 6> commonFormats{RACHIS_VERTEX_FORMAT_FLOAT32,
                                                     RACHIS_VERTEX_FORMAT_FLOAT32X2,
                                                     RACHIS_VERTEX_FORMAT_FLOAT32X3,
                                                     RACHIS_VERTEX_FORMAT_FLOAT32X4,
                                                     RACHIS_VERTEX_FORMAT_UINT32,
                                                     RACHIS_VERTEX_FORMAT_UNORM8X4};
    for (const auto format : commonFormats)
    {
        INFO("vertex format " << format);
        CHECK((capabilities.vertex_formats & (std::uint64_t{1} << format)) != 0);
    }
    CHECK(capabilities.max_vertex_buffers >= 1);
    CHECK(capabilities.max_vertex_buffers <= RACHIS_MAX_VERTEX_BUFFERS);
    CHECK(capabilities.max_vertex_attributes >= 1);
    CHECK(capabilities.max_vertex_attributes <= RACHIS_MAX_VERTEX_ATTRIBUTES);
    CHECK(capabilities.max_vertex_buffer_stride >= 1);
    CHECK(capabilities.max_vertex_instance_step_rate >= 1);
    CHECK(capabilities.reserved0 == 0);
}

TEST_CASE("C ABI reports indirect, mesh, binding-array, 64-bit atomic and subgroup capabilities",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    auto error = makeError();
    auto device = createDevice(error);
    rachis_device_capabilities capabilities{};
    capabilities.struct_size = offsetof(rachis_device_capabilities, indirect_dispatch);
    CHECK(rachis_device_query_capabilities(device.get(), &capabilities, &error) == RACHIS_STATUS_ABI_MISMATCH);
    capabilities.struct_size = sizeof(capabilities);
    capabilities.reserved1 = 7;
    requireOk(rachis_device_query_capabilities(device.get(), &capabilities, &error), error);
    CHECK(capabilities.reserved1 == 0);

    const std::array<std::uint32_t, 19> flags{
        capabilities.indirect_dispatch,
        capabilities.multi_dispatch_indirect,
        capabilities.dispatch_indirect_count,
        capabilities.indirect_draws,
        capabilities.indirect_draw_base_instance,
        capabilities.multi_draw_indirect,
        capabilities.draw_indirect_count,
        capabilities.shader_draw_index,
        capabilities.conditional_execution,
        capabilities.mesh_shaders,
        capabilities.task_shaders,
        capabilities.partially_bound_binding_arrays,
        capabilities.variable_descriptor_count_binding_arrays,
        capabilities.update_after_bind_binding_arrays,
        capabilities.shader_int64,
        capabilities.shader_int64_atomic_group_shared,
        capabilities.shader_int64_atomic_typed_resource,
        capabilities.shader_int64_atomic_storage_buffer,
        capabilities.subgroup_operations,
    };
    for (std::size_t index = 0; index < flags.size(); ++index)
    {
        INFO("flag " << index);
        CHECK(flags[index] <= 1);
    }
    CHECK(capabilities.indirect_dispatch == 1);
    CHECK(capabilities.indirect_draws == 1);
    if (capabilities.dispatch_indirect_count != 0)
    {
        CHECK(capabilities.multi_dispatch_indirect == 1);
    }
    if (capabilities.task_shaders != 0)
    {
        CHECK(capabilities.mesh_shaders == 1);
    }
    if (capabilities.mesh_shaders != 0)
    {
        CHECK(capabilities.max_mesh_workgroup_invocations >= 1);
        CHECK(capabilities.max_mesh_output_vertices >= 3);
        CHECK(capabilities.max_mesh_output_primitives >= 1);
        CHECK(capabilities.max_mesh_output_memory_bytes >= 1);
        CHECK(capabilities.max_mesh_workgroup_count[0] >= 1);
        CHECK(capabilities.max_mesh_workgroup_count[1] >= 1);
        CHECK(capabilities.max_mesh_workgroup_count[2] >= 1);
        CHECK(capabilities.max_mesh_total_workgroups >= capabilities.max_mesh_workgroup_count[0]);
    }
    if (capabilities.task_shaders != 0)
    {
        CHECK(capabilities.max_task_workgroup_invocations >= 1);
        CHECK(capabilities.max_task_payload_bytes >= 1);
    }
    if (capabilities.variable_descriptor_count_binding_arrays != 0)
    {
        CHECK(capabilities.max_variable_resource_descriptors >= 1);
        CHECK(capabilities.max_variable_buffer_descriptors >= 1);
    }
    if (capabilities.shader_int64_atomic_group_shared != 0 || capabilities.shader_int64_atomic_typed_resource != 0 ||
        capabilities.shader_int64_atomic_storage_buffer != 0)
    {
        CHECK(capabilities.shader_int64 == 1);
    }
    if (capabilities.subgroup_operations != 0)
    {
        CHECK(capabilities.subgroup_min_size >= 1);
        CHECK(capabilities.subgroup_min_size <= capabilities.subgroup_max_size);
    }
    CHECK((capabilities.subgroup_required_size_stages & ~std::uint32_t{RACHIS_SHADER_COMPUTE}) == 0);
    if (capabilities.subgroup_required_size_stages != 0)
    {
        CHECK(capabilities.subgroup_operations == 1);
    }
    if (RACHIS_SMOKE_BACKEND == RACHIS_BACKEND_D3D12)
    {
        CHECK(capabilities.multi_dispatch_indirect == 1);
        CHECK(capabilities.dispatch_indirect_count == 1);
        CHECK(capabilities.indirect_draw_base_instance == 1);
        CHECK(capabilities.multi_draw_indirect == 1);
        CHECK(capabilities.draw_indirect_count == 1);
        CHECK(capabilities.shader_draw_index == 1);
        CHECK(capabilities.partially_bound_binding_arrays == 1);
        CHECK(capabilities.variable_descriptor_count_binding_arrays == 1);
        CHECK(capabilities.update_after_bind_binding_arrays == 1);
        CHECK(capabilities.conditional_execution == 1);
    }
    if (RACHIS_SMOKE_BACKEND == RACHIS_BACKEND_METAL)
    {
        CHECK(capabilities.multi_dispatch_indirect == 1);
        CHECK(capabilities.dispatch_indirect_count == 1);
        CHECK(capabilities.multi_draw_indirect == 1);
        CHECK(capabilities.draw_indirect_count == 1);
        CHECK(capabilities.shader_draw_index == 1);
        CHECK(capabilities.update_after_bind_binding_arrays == 1);
        CHECK(capabilities.conditional_execution == 0);
        CHECK(capabilities.shader_int64_atomic_storage_buffer == 0);
        CHECK(capabilities.subgroup_required_size_stages == RACHIS_SHADER_COMPUTE);
        CHECK(capabilities.subgroup_min_size == 32);
        CHECK(capabilities.subgroup_max_size == 32);
    }
}

TEST_CASE("C ABI rejects descriptor version and layout mismatch", "[size:small]")
{
    auto error = makeError();
    rachis_device* device = nullptr;
    rachis_device_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RACHIS_ABI_VERSION + 1;
    REQUIRE(rachis_device_create(&desc, &device, &error) == RACHIS_STATUS_ABI_MISMATCH);
    REQUIRE(device == nullptr);
    REQUIRE(error.code == RACHIS_STATUS_ABI_MISMATCH);
    REQUIRE(error.message[0] != '\0');
    desc.abi_version = RACHIS_ABI_VERSION;
    desc.struct_size -= 1;
    REQUIRE(rachis_device_create(&desc, &device, &error) == RACHIS_STATUS_ABI_MISMATCH);
    REQUIRE(device == nullptr);
    REQUIRE(rachis_device_create(nullptr, &device, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    REQUIRE(device == nullptr);
    REQUIRE(rachis_abi_version() == RACHIS_ABI_VERSION);
}

#if defined(RACHIS_CABI_SHARED_LIBRARY_PATH) && defined(RACHIS_CABI_MANIFEST_PATH) && defined(RACHIS_CABI_HEADER_PATH)
namespace
{
    [[nodiscard]] std::string readTextFile(const char* path)
    {
        std::ifstream input{path, std::ios::binary};
        return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    }

    [[nodiscard]] bool isIdentChar(char ch) noexcept
    {
        const auto u = static_cast<unsigned char>(ch);
        return std::isalnum(u) != 0 || ch == '_';
    }

    [[nodiscard]] std::string stripComments(std::string_view source)
    {
        std::string out;
        out.reserve(source.size());
        for (std::size_t i = 0; i < source.size();)
        {
            if (i + 1 < source.size() && source[i] == '/' && source[i + 1] == '*')
            {
                i += 2;
                while (i + 1 < source.size() && !(source[i] == '*' && source[i + 1] == '/'))
                {
                    ++i;
                }
                i = std::min(i + 2, source.size());
                out.push_back(' ');
                continue;
            }
            if (i + 1 < source.size() && source[i] == '/' && source[i + 1] == '/')
            {
                while (i < source.size() && source[i] != '\n')
                {
                    ++i;
                }
                continue;
            }
            out.push_back(source[i]);
            ++i;
        }
        return out;
    }

    [[nodiscard]] std::uint32_t parseUnsignedAfter(std::string_view text, std::string_view needle)
    {
        const auto pos = text.find(needle);
        if (pos == std::string_view::npos)
        {
            return 0;
        }
        std::size_t idx = pos + needle.size();
        while (idx < text.size() && (text[idx] == ' ' || text[idx] == '\t' || text[idx] == '='))
        {
            ++idx;
        }
        std::uint32_t value = 0;
        bool any = false;
        while (idx < text.size() && std::isdigit(static_cast<unsigned char>(text[idx])) != 0)
        {
            any = true;
            value = value * 10U + static_cast<std::uint32_t>(text[idx] - '0');
            ++idx;
        }
        return any ? value : 0;
    }

    [[nodiscard]] std::vector<std::string> parseHeaderApiSymbols(std::string_view cleanHeader)
    {
        std::string body;
        body.reserve(cleanHeader.size());
        bool atLineStart = true;
        for (std::size_t i = 0; i < cleanHeader.size(); ++i)
        {
            const char ch = cleanHeader[i];
            if (atLineStart && (ch == ' ' || ch == '\t'))
            {
                continue;
            }
            if (atLineStart && ch == '#')
            {
                while (i < cleanHeader.size() && cleanHeader[i] != '\n')
                {
                    if (cleanHeader[i] == '\\' && i + 1 < cleanHeader.size() && cleanHeader[i + 1] == '\n')
                    {
                        ++i;
                    }
                    ++i;
                }
                atLineStart = true;
                continue;
            }
            atLineStart = (ch == '\n');
            body.push_back(ch);
        }

        std::vector<std::string> symbols;
        constexpr std::string_view marker = "RACHIS_API";
        std::size_t pos = 0;
        while ((pos = body.find(marker, pos)) != std::string::npos)
        {
            const bool startBoundary = pos == 0 || !isIdentChar(body[pos - 1]);
            const bool endBoundary = pos + marker.size() == body.size() || !isIdentChar(body[pos + marker.size()]);
            pos += marker.size();
            if (!startBoundary || !endBoundary)
            {
                continue;
            }
            const auto openParen = body.find('(', pos);
            REQUIRE(openParen != std::string::npos);
            std::size_t nameEnd = openParen;
            while (nameEnd > pos && std::isspace(static_cast<unsigned char>(body[nameEnd - 1])) != 0)
            {
                --nameEnd;
            }
            std::size_t nameStart = nameEnd;
            while (nameStart > pos && isIdentChar(body[nameStart - 1]))
            {
                --nameStart;
            }
            REQUIRE(nameEnd > nameStart);
            symbols.emplace_back(body.substr(nameStart, nameEnd - nameStart));
            pos = openParen + 1;
        }
        std::sort(symbols.begin(), symbols.end());
        return symbols;
    }
} // namespace

TEST_CASE("C ABI shared module exports the pinned manifest for RACHIS_ABI_VERSION", "[size:small][rachis][cabi]")
{
    CHECK(rachis_abi_version() == RACHIS_ABI_VERSION);

    const auto headerClean = stripComments(readTextFile(RACHIS_CABI_HEADER_PATH));
    REQUIRE_FALSE(headerClean.empty());
    const auto headerVersion = parseUnsignedAfter(headerClean, "define RACHIS_ABI_VERSION");
    CHECK(headerVersion == RACHIS_ABI_VERSION);

    const auto manifestRaw = readTextFile(RACHIS_CABI_MANIFEST_PATH);
    REQUIRE_FALSE(manifestRaw.empty());
    std::uint32_t manifestVersion = 0;
    std::vector<std::string> manifestSymbols;
    std::size_t cursor = 0;
    while (cursor < manifestRaw.size())
    {
        const auto nextLine = manifestRaw.find('\n', cursor);
        auto line = std::string_view{manifestRaw}.substr(
            cursor, nextLine == std::string::npos ? std::string::npos : nextLine - cursor);
        cursor = nextLine == std::string::npos ? manifestRaw.size() : nextLine + 1;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
        {
            line.remove_suffix(1);
        }
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
        {
            line.remove_prefix(1);
        }
        if (line.empty() || line.front() == '#')
        {
            continue;
        }
        if (manifestVersion == 0)
        {
            manifestVersion = parseUnsignedAfter(line, "RACHIS_ABI_VERSION");
            REQUIRE(manifestVersion != 0);
            continue;
        }
        manifestSymbols.emplace_back(line);
    }

    CHECK(manifestVersion == RACHIS_ABI_VERSION);
    REQUIRE_FALSE(manifestSymbols.empty());
    CHECK(std::is_sorted(manifestSymbols.begin(), manifestSymbols.end()));

    const auto headerSymbols = parseHeaderApiSymbols(headerClean);
    CHECK(headerSymbols == manifestSymbols);

#ifdef _WIN32
    HMODULE const module = LoadLibraryA(RACHIS_CABI_SHARED_LIBRARY_PATH);
    REQUIRE(module != nullptr);
    struct ModuleGuard final
    {
        HMODULE handle;
        ~ModuleGuard()
        {
            static_cast<void>(FreeLibrary(handle));
        }
    } const guard{module};
    const auto resolveSymbol = [&](const char* name) noexcept -> void* {
        return reinterpret_cast<void*>(GetProcAddress(module, name));
    };
#else
    void* const module = dlopen(RACHIS_CABI_SHARED_LIBRARY_PATH, RTLD_NOW | RTLD_LOCAL);
    INFO((module == nullptr ? dlerror() : ""));
    REQUIRE(module != nullptr);
    struct ModuleGuard final
    {
        void* handle;
        ~ModuleGuard()
        {
            static_cast<void>(dlclose(handle));
        }
    } const guard{module};
    const auto resolveSymbol = [&](const char* name) noexcept -> void* { return dlsym(module, name); };
#endif

    for (const auto& symbol : manifestSymbols)
    {
        INFO("symbol: " << symbol);
        CHECK(resolveSymbol(symbol.c_str()) != nullptr);
    }
}
#endif

TEST_CASE("C ABI leaves an error record with the wrong size untouched", "[size:small]")
{
    for (const auto size : {static_cast<std::uint32_t>(sizeof(rachis_error) - 1),
                            static_cast<std::uint32_t>(sizeof(rachis_error) + 1),
                            std::uint32_t{0}})
    {
        rachis_error error{};
        error.struct_size = size;
        error.code = 77;
        error.native_code = -5;
        std::memcpy(error.message, "sentinel", sizeof("sentinel"));
        rachis_device* device = nullptr;
        rachis_device_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        CHECK(rachis_device_create(&desc, &device, &error) == RACHIS_STATUS_ABI_MISMATCH);
        CHECK(device == nullptr);
        CHECK(rachis_buffer_destroy(nullptr, 1, &error) == RACHIS_STATUS_ABI_MISMATCH);
        CHECK(error.struct_size == size);
        CHECK(error.code == 77);
        CHECK(error.native_code == -5);
        CHECK(std::string_view{error.message} == "sentinel");
    }
}

TEST_CASE("C ABI rejects a null device or output with invalid argument", "[size:small]")
{
    auto error = makeError();
    rachis_swapchain_image image{};
    image.struct_size = sizeof(image);
    CHECK(rachis_swapchain_acquire(nullptr, 1, &image, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(error.code == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_swapchain_acquire(nullptr, 1, nullptr, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    image.struct_size = 0;
    CHECK(rachis_swapchain_acquire(nullptr, 1, &image, &error) == RACHIS_STATUS_INVALID_ARGUMENT);

    rachis_handle handle = 0;
    std::uint32_t state = 0;
    std::uint32_t status = 0;
    std::uint64_t frequency = 0;
    rachis_diagnostic_counts counts{};
    counts.struct_size = sizeof(counts);
    rachis_surface_capabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    rachis_texture_transfer_layout layout{};
    layout.struct_size = sizeof(layout);
    const auto region = wholeTextureRegion(1, 1);
    rachis_texture_desc textureDesc{};
    textureDesc.struct_size = sizeof(textureDesc);
    textureDesc.abi_version = RACHIS_ABI_VERSION;
    rachis_barrier_batch batch{};
    batch.struct_size = sizeof(batch);
    batch.abi_version = RACHIS_ABI_VERSION;
    rachis_rendering_desc rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.abi_version = RACHIS_ABI_VERSION;

    CHECK(rachis_device_diagnostic_counts(nullptr, &counts, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_device_operational_state(nullptr, &state, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_device_backend(nullptr, &state, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_queue_timestamp_frequency(nullptr, RACHIS_QUEUE_GRAPHICS, &frequency, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_create(nullptr, &textureDesc, &handle, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_destroy(nullptr, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_lease_acquire(nullptr, 1, &handle, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_lease_release(nullptr, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_view_destroy(nullptr, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_query_transfer_layout(nullptr, 1, &region, &layout, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_barriers(nullptr, 1, &batch, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_begin_rendering(nullptr, 1, &rendering, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_end_rendering(nullptr, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_reset_queries(nullptr, 1, 1, 0, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_surface_query_capabilities(nullptr, 1, &capabilities, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_swapchain_present(nullptr, 1, 1, &status, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_swapchain_cancel(nullptr, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    std::uint8_t byte = 0;
    CHECK(rachis_buffer_write(nullptr, 1, 0, &byte, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_read(nullptr, 1, 0, &byte, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_write(nullptr, 1, 0, &byte, RACHIS_WHOLE_SIZE, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_read(nullptr, 1, 0, &byte, RACHIS_WHOLE_SIZE, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    rachis_device_capabilities deviceCapabilities{};
    deviceCapabilities.struct_size = sizeof(deviceCapabilities);
    rachis_mapped_memory mapped{};
    mapped.struct_size = sizeof(mapped);
    const rachis_mapped_range range{0, 1};
    const rachis_vertex_buffer_binding vertexBinding{1, 0};
    CHECK(rachis_device_query_capabilities(nullptr, &deviceCapabilities, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_map(nullptr, 1, 0, 1, &mapped, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_unmap(nullptr, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_flush_mapped_ranges(nullptr, 1, &range, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_invalidate_mapped_ranges(nullptr, 1, &range, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_bind_vertex_buffers(nullptr, 1, 0, &vertexBinding, 1, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_bind_index_buffer(nullptr, 1, 1, RACHIS_INDEX_FORMAT_UINT16, 0, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_draw_indexed(nullptr, 1, 3, 1, 0, 0, 0, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    rachis_indirect_draw_desc drawDesc{};
    drawDesc.struct_size = sizeof(drawDesc);
    drawDesc.abi_version = RACHIS_ABI_VERSION;
    drawDesc.argument_buffer = 1;
    drawDesc.max_draw_count = 1;
    rachis_indirect_dispatch_desc dispatchDesc{};
    dispatchDesc.struct_size = sizeof(dispatchDesc);
    dispatchDesc.abi_version = RACHIS_ABI_VERSION;
    dispatchDesc.argument_buffer = 1;
    dispatchDesc.max_dispatch_count = 1;
    rachis_indirect_mesh_dispatch_desc meshDesc{};
    meshDesc.struct_size = sizeof(meshDesc);
    meshDesc.abi_version = RACHIS_ABI_VERSION;
    meshDesc.argument_buffer = 1;
    meshDesc.max_dispatch_count = 1;
    rachis_binding_set_update_desc updateDesc{};
    updateDesc.struct_size = sizeof(updateDesc);
    updateDesc.abi_version = RACHIS_ABI_VERSION;
    const rachis_storage_texture_clear_value clearValue{};
    CHECK(rachis_command_draw_indirect(nullptr, 1, &drawDesc, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_draw_indexed_indirect(nullptr, 1, &drawDesc, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_dispatch_indirect(nullptr, 1, &dispatchDesc, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_dispatch_mesh(nullptr, 1, 1, 1, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_dispatch_mesh_indirect(nullptr, 1, &meshDesc, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_fill_buffer(nullptr, 1, 1, 0, 4, 0, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_clear_storage_texture(nullptr, 1, 1, &clearValue, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_binding_set_update(nullptr, 1, &updateDesc, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    drawDesc.struct_size = sizeof(drawDesc) - 4;
    CHECK(rachis_command_draw_indirect(nullptr, 1, &drawDesc, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    rachis_device_destroy(nullptr);
}

TEST_CASE("C ABI checks handle owner, type and stale lifetime",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    auto error = makeError();
    auto first = createDevice(error);
    auto second = createDevice(error);
    const auto buffer = createBuffer(first.get(), 16, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    REQUIRE(buffer != 0);
    std::array<std::uint8_t, 4> bytes{1, 2, 3, 4};
    CHECK(rachis_buffer_write(first.get(), 0, 0, bytes.data(), bytes.size(), &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_read(first.get(), 0, 0, bytes.data(), bytes.size(), &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_write(second.get(), buffer, 0, bytes.data(), bytes.size(), &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_read(second.get(), buffer, 0, bytes.data(), bytes.size(), &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_destroy(second.get(), buffer, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_pool_destroy(first.get(), buffer, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_destroy(first.get(), buffer, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_lease_acquire(first.get(), buffer, nullptr, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    rachis_handle lease = 0;
    CHECK(rachis_texture_lease_acquire(first.get(), buffer, &lease, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(lease == 0);
    requireOk(rachis_buffer_destroy(first.get(), buffer, &error), error);
    CHECK(rachis_buffer_write(first.get(), buffer, 0, bytes.data(), bytes.size(), &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_read(first.get(), buffer, 0, bytes.data(), bytes.size(), &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_destroy(first.get(), buffer, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
}

TEST_CASE("C ABI host buffer reads and writes reject whole-size and out-of-range byte counts",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    constexpr std::uint64_t bufferSize = 16;
    auto error = makeError();
    auto device = createDevice(error);
    const auto upload = createBuffer(device.get(), bufferSize, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback =
        createBuffer(device.get(), bufferSize, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    const auto deviceLocal = createBuffer(device.get(),
                                          bufferSize,
                                          RACHIS_BUFFER_COPY_SOURCE | RACHIS_BUFFER_COPY_DESTINATION,
                                          RACHIS_MEMORY_DEVICE_LOCAL,
                                          error);
    std::array<std::uint8_t, bufferSize> bytes{};
    const std::array<std::uint8_t, 4> smallExpected{0xAA, 0xBB, 0xCC, 0xDD};
    auto smallBytes = smallExpected;

    for (const auto target : {upload, readback, deviceLocal})
    {
        CHECK(rachis_buffer_write(device.get(), target, 0, bytes.data(), 0, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rachis_buffer_read(device.get(), target, 0, bytes.data(), 0, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rachis_buffer_write(device.get(), target, 0, nullptr, 0, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rachis_buffer_read(device.get(), target, 0, nullptr, 0, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rachis_buffer_write(device.get(), target, 0, nullptr, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rachis_buffer_read(device.get(), target, 0, nullptr, 1, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rachis_buffer_write(device.get(), target, 0, nullptr, RACHIS_WHOLE_SIZE, &error) ==
              RACHIS_STATUS_INVALID_ARGUMENT);
        CHECK(rachis_buffer_read(device.get(), target, 0, nullptr, RACHIS_WHOLE_SIZE, &error) ==
              RACHIS_STATUS_INVALID_ARGUMENT);
    }

    struct Range
    {
        std::uint64_t offset;
        std::uint64_t size;
    };
    const std::array<Range, 10> rejected{{
        {0, RACHIS_WHOLE_SIZE},
        {8, RACHIS_WHOLE_SIZE},
        {bufferSize, RACHIS_WHOLE_SIZE},
        {0, bufferSize + 1},
        {0, bufferSize * 4},
        {8, bufferSize - 7},
        {bufferSize, 1},
        {RACHIS_WHOLE_SIZE, 1},
        {RACHIS_WHOLE_SIZE, bufferSize},
        {8, RACHIS_WHOLE_SIZE - 3},
    }};
    for (const auto& range : rejected)
    {
        INFO("offset " << range.offset << ", size " << range.size);

        for (const auto target : {upload, readback, deviceLocal})
        {
            CHECK(rachis_buffer_write(device.get(), target, range.offset, smallBytes.data(), range.size, &error) ==
                  RACHIS_STATUS_INVALID_ARGUMENT);
            CHECK(rachis_buffer_read(device.get(), target, range.offset, smallBytes.data(), range.size, &error) ==
                  RACHIS_STATUS_INVALID_ARGUMENT);
            CHECK(smallBytes == smallExpected);
        }
    }

    CHECK(rachis_buffer_write(device.get(), deviceLocal, 0, smallBytes.data(), smallBytes.size(), &error) ==
          RACHIS_STATUS_UNSUPPORTED);
    CHECK(rachis_buffer_read(device.get(), deviceLocal, 0, smallBytes.data(), smallBytes.size(), &error) ==
          RACHIS_STATUS_UNSUPPORTED);
    CHECK(rachis_buffer_write(device.get(), readback, 0, smallBytes.data(), smallBytes.size(), &error) ==
          RACHIS_STATUS_UNSUPPORTED);
    CHECK(rachis_buffer_read(device.get(), upload, 0, smallBytes.data(), smallBytes.size(), &error) ==
          RACHIS_STATUS_UNSUPPORTED);
    CHECK(smallBytes == smallExpected);

    rachis_shader_module_desc shaderDesc{};
    shaderDesc.struct_size = sizeof(shaderDesc);
    shaderDesc.abi_version = RACHIS_ABI_VERSION;
    shaderDesc.stage = RACHIS_SHADER_COMPUTE;
    shaderDesc.entry_point = "main";
    shaderDesc.workgroup_x = 1;
    shaderDesc.workgroup_y = 1;
    shaderDesc.workgroup_z = 1;
    rachis_handle shader = 0;
    shaderDesc.bytecode = smallBytes.data();
    shaderDesc.bytecode_size = RACHIS_WHOLE_SIZE;
    CHECK(rachis_shader_module_create(device.get(), &shaderDesc, &shader, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(shader == 0);
    shaderDesc.bytecode_size = 0;
    CHECK(rachis_shader_module_create(device.get(), &shaderDesc, &shader, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(shader == 0);
    shaderDesc.bytecode = nullptr;
    shaderDesc.bytecode_size = smallBytes.size();
    CHECK(rachis_shader_module_create(device.get(), &shaderDesc, &shader, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(shader == 0);
    auto* wrappingAddress = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 1);
    shaderDesc.bytecode = wrappingAddress;
    shaderDesc.bytecode_size = smallBytes.size();
    CHECK(rachis_shader_module_create(device.get(), &shaderDesc, &shader, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(shader == 0);
    CHECK(rachis_buffer_write(device.get(), upload, 0, wrappingAddress, smallBytes.size(), &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_buffer_read(device.get(), readback, 0, wrappingAddress, smallBytes.size(), &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_graphics_push_constants(device.get(), 1, RACHIS_SHADER_PIXEL, 0, nullptr, 4, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_graphics_push_constants(
              device.get(), 1, RACHIS_SHADER_PIXEL, 0, wrappingAddress, 4, &error) == RACHIS_STATUS_INVALID_ARGUMENT);

    requireOk(rachis_buffer_write(device.get(), upload, 0, bytes.data(), bufferSize, &error), error);
    requireOk(rachis_buffer_write(device.get(), upload, 8, bytes.data(), bufferSize - 8, &error), error);
    requireOk(rachis_buffer_read(device.get(), readback, 0, bytes.data(), bufferSize, &error), error);
    requireOk(rachis_buffer_read(device.get(), readback, bufferSize - 1, bytes.data(), 1, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), deviceLocal, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
}

TEST_CASE("C ABI submission consumes command leases and timelines release once",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    auto error = makeError();
    auto device = createDevice(error);
    const std::array<std::uint8_t, 8> expected{1, 2, 3, 5, 8, 13, 21, 34};
    const auto upload =
        createBuffer(device.get(), expected.size(), RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback =
        createBuffer(device.get(), expected.size(), RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    requireOk(rachis_buffer_write(device.get(), upload, 0, expected.data(), expected.size(), &error), error);

    rachis_handle pool = 0;
    const auto discarded = acquireCommand(device.get(), RACHIS_QUEUE_COPY, pool, error);
    requireOk(rachis_command_copy_buffer(device.get(), discarded, upload, readback, 0, 0, expected.size(), &error),
              error);
    requireOk(rachis_command_discard(device.get(), discarded, &error), error);
    rachis_handle timeline = 0;
    CHECK(rachis_command_submit(device.get(), discarded, &timeline, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(timeline == 0);

    rachis_handle command = 0;
    requireOk(rachis_command_acquire(device.get(), pool, &command, &error), error);
    requireOk(rachis_command_copy_buffer(device.get(), command, upload, readback, 0, 0, expected.size(), &error),
              error);
    requireOk(rachis_command_submit(device.get(), command, &timeline, &error), error);
    REQUIRE(timeline != 0);
    rachis_handle again = 0;
    CHECK(rachis_command_submit(device.get(), command, &again, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(again == 0);
    CHECK(rachis_command_discard(device.get(), command, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_copy_buffer(device.get(), command, upload, readback, 0, 0, 1, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_timeline_wait(device.get(), timeline, &error), error);
    requireOk(rachis_timeline_wait(device.get(), timeline, &error), error);

    std::array<std::uint8_t, 8> actual{};
    requireOk(rachis_buffer_read(device.get(), readback, 0, actual.data(), actual.size(), &error), error);
    CHECK(actual == expected);

    requireOk(rachis_timeline_release(device.get(), timeline, &error), error);
    CHECK(rachis_timeline_wait(device.get(), timeline, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_timeline_release(device.get(), timeline, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
}

TEST_CASE("C ABI pre-consume submit failure preserves command recording for recovery or discard",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    const auto texture = createTexture(device.get(),
                                       RACHIS_FORMAT_RGBA8_UNORM,
                                       textureSize,
                                       textureSize,
                                       RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_COPY_SOURCE,
                                       error);
    const auto viewDesc = renderTargetViewDesc(texture);
    rachis_handle view = 0;
    requireOk(rachis_texture_view_create(device.get(), &viewDesc, &view, &error), error);

    rachis_handle pool = 0;
    const auto recoverable = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
    const auto toRender = textureBarrier(texture,
                                         RACHIS_STAGE_ALL,
                                         RACHIS_ACCESS_NONE,
                                         RACHIS_LAYOUT_UNDEFINED,
                                         RACHIS_STAGE_RENDER_TARGET,
                                         RACHIS_ACCESS_RENDER_TARGET,
                                         RACHIS_LAYOUT_RENDER_TARGET);
    requireOk(recordBarriers(device.get(), recoverable, nullptr, 0, &toRender, 1, error), error);

    rachis_color_attachment attachment{};
    attachment.view = view;
    attachment.load_op = RACHIS_LOAD_OP_CLEAR;
    attachment.store_op = RACHIS_STORE_OP_STORE;
    attachment.clear_color[0] = 1.0F;
    attachment.clear_color[3] = 1.0F;
    rachis_rendering_desc rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.abi_version = RACHIS_ABI_VERSION;
    rendering.color_attachments = &attachment;
    rendering.color_attachment_count = 1;
    requireOk(rachis_command_begin_rendering(device.get(), recoverable, &rendering, &error), error);

    rachis_handle timeline = 0;
    CHECK(rachis_command_submit(device.get(), recoverable, &timeline, &error) == RACHIS_STATUS_INVALID_STATE);
    CHECK(timeline == 0);

    requireOk(rachis_command_end_rendering(device.get(), recoverable, &error), error);
    submitAndWait(device.get(), recoverable, error);
    CHECK(rachis_command_discard(device.get(), recoverable, &error) == RACHIS_STATUS_INVALID_ARGUMENT);

    rachis_handle discardable = 0;
    requireOk(rachis_command_acquire(device.get(), pool, &discardable, &error), error);
    requireOk(rachis_command_begin_rendering(device.get(), discardable, &rendering, &error), error);
    CHECK(rachis_command_submit(device.get(), discardable, &timeline, &error) == RACHIS_STATUS_INVALID_STATE);
    CHECK(timeline == 0);
    requireOk(rachis_command_discard(device.get(), discardable, &error), error);
    CHECK(rachis_command_discard(device.get(), discardable, &error) == RACHIS_STATUS_INVALID_ARGUMENT);

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_texture_view_destroy(device.get(), view, &error), error);
    requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
    requireNoValidationErrors(device.get(), error);
}

namespace
{
    struct WaitOverlap final
    {
        rachis_status waitStatus = RACHIS_STATUS_OK;
        std::vector<rachis_status> statuses;
        std::string message;

        int callsDuringWait = 0;
    };

    [[nodiscard]] WaitOverlap overlapTimelineWait(rachis_device* device,
                                                  rachis_handle source,
                                                  rachis_handle destination,
                                                  std::uint64_t copySize,
                                                  rachis_error& error)
    {
        constexpr int copyCount = 96;
        rachis_handle pool = 0;
        const auto command = acquireCommand(device, RACHIS_QUEUE_COPY, pool, error);

        const auto copyAfterCopy = bufferBarrier(destination,
                                                 RACHIS_STAGE_COPY,
                                                 RACHIS_ACCESS_COPY_DESTINATION,
                                                 RACHIS_STAGE_COPY,
                                                 RACHIS_ACCESS_COPY_DESTINATION);
        for (int copy = 0; copy < copyCount; ++copy)
        {
            if (copy != 0)
            {
                requireOk(recordBarriers(device, command, &copyAfterCopy, 1, nullptr, 0, error), error);
            }
            requireOk(rachis_command_copy_buffer(device, command, source, destination, 0, 0, copySize, &error), error);
        }
        rachis_handle timeline = 0;
        requireOk(rachis_command_submit(device, command, &timeline, &error), error);

        WaitOverlap overlap;
        std::atomic_bool waiting = false;
        std::atomic_bool waitReturned = false;
        std::thread waiter([&] {
            auto waitError = makeError();
            waiting.store(true, std::memory_order_release);
            overlap.waitStatus = rachis_timeline_wait(device, timeline, &waitError);
            waitReturned.store(true, std::memory_order_release);
        });
        while (!waiting.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(2));

        auto otherError = makeError();
        const auto call = [&](rachis_status status) {
            overlap.statuses.push_back(status);
            if (status != RACHIS_STATUS_OK && overlap.message.empty())
            {
                overlap.message = otherError.message;
            }
            if (!waitReturned.load(std::memory_order_acquire))
            {
                ++overlap.callsDuringWait;
            }
        };
        call(rachis_timeline_release(device, timeline, &otherError));
        for (int round = 0; round < 100000 && !waitReturned.load(std::memory_order_acquire); ++round)
        {
            rachis_buffer_desc bufferDesc{};
            bufferDesc.struct_size = sizeof(bufferDesc);
            bufferDesc.abi_version = RACHIS_ABI_VERSION;
            bufferDesc.size = 256;
            bufferDesc.usage = RACHIS_BUFFER_COPY_SOURCE;
            bufferDesc.memory_type = RACHIS_MEMORY_UPLOAD;
            rachis_handle buffer = 0;
            call(rachis_buffer_create(device, &bufferDesc, &buffer, &otherError));
            call(rachis_buffer_destroy(device, buffer, &otherError));

            rachis_texture_desc textureDesc{};
            textureDesc.struct_size = sizeof(textureDesc);
            textureDesc.abi_version = RACHIS_ABI_VERSION;
            textureDesc.dimension = RACHIS_TEXTURE_2D;
            textureDesc.format = RACHIS_FORMAT_RGBA8_UNORM;
            textureDesc.usage = RACHIS_TEXTURE_SAMPLED;
            textureDesc.width = textureSize;
            textureDesc.height = textureSize;
            textureDesc.depth_or_array_layers = 1;
            textureDesc.mip_levels = 1;
            textureDesc.sample_count = 1;
            rachis_handle texture = 0;
            call(rachis_texture_create(device, &textureDesc, &texture, &otherError));
            call(rachis_texture_destroy(device, texture, &otherError));
        }
        waiter.join();
        CHECK(rachis_timeline_wait(device, timeline, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
        requireOk(rachis_command_pool_destroy(device, pool, &error), error);
        return overlap;
    }
} // namespace

TEST_CASE("C ABI timeline waits leave the device to other threads",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    constexpr std::uint64_t copySize = 64ULL << 20U;
    constexpr int attempts = 5;
    auto error = makeError();
    auto device = createDevice(error);
    const auto source =
        createBuffer(device.get(), copySize, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_DEVICE_LOCAL, error);
    const auto destination =
        createBuffer(device.get(), copySize, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_DEVICE_LOCAL, error);

    bool overlapped = false;
    for (int attempt = 0; attempt < attempts && !overlapped; ++attempt)
    {
        const auto overlap = overlapTimelineWait(device.get(), source, destination, copySize, error);
        INFO("attempt " << attempt << ": " << overlap.message);
        for (const auto status : overlap.statuses)
        {
            CHECK(status == RACHIS_STATUS_OK);
        }

        CHECK((overlap.waitStatus == RACHIS_STATUS_OK || overlap.waitStatus == RACHIS_STATUS_INVALID_ARGUMENT));
        overlapped = overlap.waitStatus == RACHIS_STATUS_OK && overlap.callsDuringWait > 0;
    }
    CHECK(overlapped);
    requireOk(rachis_buffer_destroy(device.get(), destination, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), source, &error), error);
}

TEST_CASE("C ABI clears a render target and reads it back through explicit barriers",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    const auto texture = createTexture(device.get(),
                                       RACHIS_FORMAT_RGBA8_UNORM,
                                       textureSize,
                                       textureSize,
                                       RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_COPY_SOURCE,
                                       error);
    const auto viewDesc = renderTargetViewDesc(texture);
    rachis_handle view = 0;
    requireOk(rachis_texture_view_create(device.get(), &viewDesc, &view, &error), error);

    const auto region = wholeTextureRegion(textureSize, textureSize);
    rachis_texture_transfer_layout layout{};
    layout.struct_size = sizeof(layout) - 1;
    CHECK(rachis_texture_query_transfer_layout(device.get(), texture, &region, &layout, &error) ==
          RACHIS_STATUS_ABI_MISMATCH);
    layout.struct_size = sizeof(layout);
    requireOk(rachis_texture_query_transfer_layout(device.get(), texture, &region, &layout, &error), error);
    REQUIRE(layout.bytes_per_row >= textureSize * 4);
    REQUIRE(layout.row_count == textureSize);
    const auto readback = createBuffer(
        device.get(), layout.required_buffer_size, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
    auto invalid = textureBarrier(texture,
                                  RACHIS_STAGE_ALL,
                                  RACHIS_ACCESS_NONE,
                                  RACHIS_LAYOUT_RESOLVE_DESTINATION + 1,
                                  RACHIS_STAGE_RENDER_TARGET,
                                  RACHIS_ACCESS_RENDER_TARGET,
                                  RACHIS_LAYOUT_RENDER_TARGET);
    CHECK(recordBarriers(device.get(), command, nullptr, 0, &invalid, 1, error) == RACHIS_STATUS_INVALID_ARGUMENT);
    invalid = textureBarrier(readback,
                             RACHIS_STAGE_ALL,
                             RACHIS_ACCESS_NONE,
                             RACHIS_LAYOUT_UNDEFINED,
                             RACHIS_STAGE_RENDER_TARGET,
                             RACHIS_ACCESS_RENDER_TARGET,
                             RACHIS_LAYOUT_RENDER_TARGET);
    CHECK(recordBarriers(device.get(), command, nullptr, 0, &invalid, 1, error) == RACHIS_STATUS_INVALID_ARGUMENT);
    rachis_barrier_batch mismatched{};
    mismatched.struct_size = sizeof(mismatched);
    CHECK(rachis_command_barriers(device.get(), command, &mismatched, &error) == RACHIS_STATUS_ABI_MISMATCH);

    const auto toRenderTarget = textureBarrier(texture,
                                               RACHIS_STAGE_ALL,
                                               RACHIS_ACCESS_NONE,
                                               RACHIS_LAYOUT_UNDEFINED,
                                               RACHIS_STAGE_RENDER_TARGET,
                                               RACHIS_ACCESS_RENDER_TARGET,
                                               RACHIS_LAYOUT_RENDER_TARGET);
    requireOk(recordBarriers(device.get(), command, nullptr, 0, &toRenderTarget, 1, error), error);

    rachis_color_attachment attachment{};
    attachment.view = view;
    attachment.load_op = RACHIS_LOAD_OP_DISCARD + 1;
    attachment.store_op = RACHIS_STORE_OP_STORE;
    attachment.clear_color[0] = 0.25F;
    attachment.clear_color[1] = 0.5F;
    attachment.clear_color[2] = 0.75F;
    attachment.clear_color[3] = 1.0F;
    rachis_rendering_desc rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.abi_version = RACHIS_ABI_VERSION;
    rendering.color_attachments = &attachment;
    rendering.color_attachment_count = 1;
    CHECK(rachis_command_begin_rendering(device.get(), command, &rendering, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    attachment.load_op = RACHIS_LOAD_OP_CLEAR;
    attachment.view = texture;
    CHECK(rachis_command_begin_rendering(device.get(), command, &rendering, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    attachment.view = view;
    requireOk(rachis_command_begin_rendering(device.get(), command, &rendering, &error), error);
    requireOk(rachis_command_end_rendering(device.get(), command, &error), error);

    const auto toCopy = textureBarrier(texture,
                                       RACHIS_STAGE_RENDER_TARGET,
                                       RACHIS_ACCESS_RENDER_TARGET,
                                       RACHIS_LAYOUT_RENDER_TARGET,
                                       RACHIS_STAGE_COPY,
                                       RACHIS_ACCESS_COPY_SOURCE,
                                       RACHIS_LAYOUT_COPY_SOURCE);
    requireOk(recordBarriers(device.get(), command, nullptr, 0, &toCopy, 1, error), error);
    auto copyRegion = region;
    copyRegion.bytes_per_row = layout.bytes_per_row;
    requireOk(copyBufferTexture(device.get(), command, readback, texture, copyRegion, false, error), error);
    submitAndWait(device.get(), command, error);

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(layout.required_buffer_size));
    requireOk(rachis_buffer_read(device.get(), readback, 0, pixels.data(), pixels.size(), &error), error);
    for (std::uint32_t y = 0; y < textureSize; ++y)
    {
        for (std::uint32_t x = 0; x < textureSize; ++x)
        {
            const auto* pixel =
                pixels.data() + static_cast<std::size_t>(y) * layout.bytes_per_row + static_cast<std::size_t>(x) * 4;
            INFO("pixel " << x << "," << y);
            CHECK(nearByte(pixel[0], 64));
            CHECK(nearByte(pixel[1], 128));
            CHECK(nearByte(pixel[2], 191));
            CHECK(pixel[3] == 255);
        }
    }

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_texture_view_destroy(device.get(), view, &error), error);
    CHECK(rachis_texture_view_destroy(device.get(), view, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireNoValidationErrors(device.get(), error);
}

TEST_CASE("C ABI texture leases and views outlive the owner handle",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    auto other = createDevice(error);
    const auto texture =
        createTexture(device.get(),
                      RACHIS_FORMAT_RGBA8_UNORM,
                      textureSize,
                      textureSize,
                      RACHIS_TEXTURE_COPY_SOURCE | RACHIS_TEXTURE_COPY_DESTINATION | RACHIS_TEXTURE_RENDER_TARGET,
                      error);

    rachis_handle lease = 0;
    CHECK(rachis_texture_lease_acquire(other.get(), texture, &lease, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_texture_lease_acquire(device.get(), texture, &lease, &error), error);
    REQUIRE(lease != 0);
    REQUIRE(lease != texture);
    rachis_handle nested = 0;
    requireOk(rachis_texture_lease_acquire(device.get(), lease, &nested, &error), error);
    requireOk(rachis_texture_lease_release(device.get(), nested, &error), error);
    CHECK(rachis_texture_lease_release(device.get(), nested, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_destroy(device.get(), lease, &error) == RACHIS_STATUS_INVALID_ARGUMENT);

    const auto viewDesc = renderTargetViewDesc(lease);
    rachis_handle view = 0;
    requireOk(rachis_texture_view_create(device.get(), &viewDesc, &view, &error), error);

    requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
    CHECK(rachis_texture_destroy(device.get(), texture, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    rachis_handle stale = 0;
    CHECK(rachis_texture_lease_acquire(device.get(), texture, &stale, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    const auto staleViewDesc = renderTargetViewDesc(texture);
    CHECK(rachis_texture_view_create(device.get(), &staleViewDesc, &stale, &error) == RACHIS_STATUS_INVALID_ARGUMENT);

    const auto region = wholeTextureRegion(textureSize, textureSize);
    rachis_texture_transfer_layout layout{};
    layout.struct_size = sizeof(layout);
    requireOk(rachis_texture_query_transfer_layout(device.get(), lease, &region, &layout, &error), error);
    CHECK(rachis_texture_query_transfer_layout(device.get(), texture, &region, &layout, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    const auto upload =
        createBuffer(device.get(), layout.required_buffer_size, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback = createBuffer(
        device.get(), layout.required_buffer_size, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    std::vector<std::uint8_t> source(static_cast<std::size_t>(layout.required_buffer_size));
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        source[index] = static_cast<std::uint8_t>(index * 7 + 3);
    }
    requireOk(rachis_buffer_write(device.get(), upload, 0, source.data(), source.size(), &error), error);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
    const auto toCopyDestination = textureBarrier(lease,
                                                  RACHIS_STAGE_ALL,
                                                  RACHIS_ACCESS_NONE,
                                                  RACHIS_LAYOUT_UNDEFINED,
                                                  RACHIS_STAGE_COPY,
                                                  RACHIS_ACCESS_COPY_DESTINATION,
                                                  RACHIS_LAYOUT_COPY_DESTINATION);
    requireOk(recordBarriers(device.get(), command, nullptr, 0, &toCopyDestination, 1, error), error);
    auto copyRegion = region;
    copyRegion.bytes_per_row = layout.bytes_per_row;
    requireOk(copyBufferTexture(device.get(), command, upload, lease, copyRegion, true, error), error);
    const auto toCopySource = textureBarrier(lease,
                                             RACHIS_STAGE_COPY,
                                             RACHIS_ACCESS_COPY_DESTINATION,
                                             RACHIS_LAYOUT_COPY_DESTINATION,
                                             RACHIS_STAGE_COPY,
                                             RACHIS_ACCESS_COPY_SOURCE,
                                             RACHIS_LAYOUT_COPY_SOURCE);
    requireOk(recordBarriers(device.get(), command, nullptr, 0, &toCopySource, 1, error), error);
    requireOk(copyBufferTexture(device.get(), command, readback, lease, copyRegion, false, error), error);
    rachis_handle timeline = 0;
    requireOk(rachis_command_submit(device.get(), command, &timeline, &error), error);

    requireOk(rachis_texture_view_destroy(device.get(), view, &error), error);
    requireOk(rachis_texture_lease_release(device.get(), lease, &error), error);
    CHECK(rachis_texture_lease_release(device.get(), lease, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_texture_view_destroy(device.get(), view, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    requireOk(rachis_timeline_wait(device.get(), timeline, &error), error);
    requireOk(rachis_timeline_release(device.get(), timeline, &error), error);

    std::vector<std::uint8_t> actual(source.size());
    requireOk(rachis_buffer_read(device.get(), readback, 0, actual.data(), actual.size(), &error), error);
    for (std::uint32_t y = 0; y < textureSize; ++y)
    {
        const auto offset = static_cast<std::size_t>(y) * layout.bytes_per_row;
        INFO("row " << y);
        CHECK(std::memcmp(actual.data() + offset, source.data() + offset, std::size_t{textureSize} * 4) == 0);
    }

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireNoValidationErrors(device.get(), error);
}

TEST_CASE("C ABI 1D textures accept one mip everywhere and report a Metal mip chain as unsupported",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][regression]")
{
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    for (const std::uint32_t layers : {1U, 2U})
    {
        for (const std::uint32_t mipLevels : {1U, 4U})
        {
            CAPTURE(layers, mipLevels);
            rachis_texture_desc desc{};
            desc.struct_size = sizeof(desc);
            desc.abi_version = RACHIS_ABI_VERSION;
            desc.dimension = RACHIS_TEXTURE_1D;
            desc.format = RACHIS_FORMAT_RGBA8_UNORM;
            desc.usage = RACHIS_TEXTURE_SAMPLED | RACHIS_TEXTURE_COPY_SOURCE | RACHIS_TEXTURE_COPY_DESTINATION;
            desc.width = 13;
            desc.height = 1;
            desc.depth_or_array_layers = layers;
            desc.mip_levels = mipLevels;
            desc.sample_count = 1;
            desc.label = "C ABI 1D texture";
            rachis_handle texture = 0;
            const auto status = rachis_texture_create(device.get(), &desc, &texture, &error);
            if (RACHIS_SMOKE_BACKEND == RACHIS_BACKEND_METAL && mipLevels != 1)
            {
                CHECK(status == RACHIS_STATUS_UNSUPPORTED);
                CHECK(texture == 0);
            }
            else
            {
                requireOk(status, error);
                requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
            }
        }
    }
    requireNoValidationErrors(device.get(), error);
}

TEST_CASE("C ABI 1D textures accept every color usage except render targets on Metal",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][regression]")
{
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);
    for (const std::uint32_t usage :
         {static_cast<std::uint32_t>(RACHIS_TEXTURE_SAMPLED | RACHIS_TEXTURE_STORAGE | RACHIS_TEXTURE_COPY_SOURCE |
                                     RACHIS_TEXTURE_COPY_DESTINATION),
          static_cast<std::uint32_t>(RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_COPY_SOURCE)})
    {
        CAPTURE(usage);
        rachis_texture_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.dimension = RACHIS_TEXTURE_1D;
        desc.format = RACHIS_FORMAT_RGBA8_UNORM;
        desc.usage = usage;
        desc.width = 64;
        desc.height = 1;
        desc.depth_or_array_layers = 1;
        desc.mip_levels = 1;
        desc.sample_count = 1;
        desc.label = "C ABI 1D render target";
        rachis_handle texture = 0;
        const auto status = rachis_texture_create(device.get(), &desc, &texture, &error);
        if (RACHIS_SMOKE_BACKEND == RACHIS_BACKEND_METAL && (usage & RACHIS_TEXTURE_RENDER_TARGET) != 0)
        {
            CHECK(status == RACHIS_STATUS_UNSUPPORTED);
            CHECK(texture == 0);
        }
        else
        {
            requireOk(status, error);
            requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
        }
    }
    requireNoValidationErrors(device.get(), error);
}

#ifdef RACHIS_CABI_COMPUTE_SHADER_PATH
TEST_CASE("C ABI compute output reaches a texture through barriers and timestamps",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);

    const rachis_shader_binding_requirement requirement{0, 0, RACHIS_BINDING_STORAGE_BUFFER, 1};
    rachis_shader_module_desc shaderDesc{};
    shaderDesc.struct_size = sizeof(shaderDesc);
    shaderDesc.abi_version = RACHIS_ABI_VERSION;
    shaderDesc.stage = RACHIS_SHADER_COMPUTE;
    shaderDesc.bytecode = bytecode.data();
    shaderDesc.bytecode_size = bytecode.size();
    shaderDesc.entry_point = "main";
    shaderDesc.bindings = &requirement;
    shaderDesc.binding_count = 1;
    shaderDesc.workgroup_x = valueCount;
    shaderDesc.workgroup_y = 1;
    shaderDesc.workgroup_z = 1;
    rachis_handle shader = 0;
    requireOk(rachis_shader_module_create(device.get(), &shaderDesc, &shader, &error), error);

    const rachis_binding_layout_entry entry{0, RACHIS_BINDING_STORAGE_BUFFER, RACHIS_SHADER_COMPUTE, 1, 0};
    rachis_binding_layout_desc bindingLayoutDesc{};
    bindingLayoutDesc.struct_size = sizeof(bindingLayoutDesc);
    bindingLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    bindingLayoutDesc.entries = &entry;
    bindingLayoutDesc.entry_count = 1;
    rachis_handle bindingLayout = 0;
    requireOk(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &bindingLayout, &error), error);
    const rachis_binding_pool_size poolSize{RACHIS_BINDING_STORAGE_BUFFER, 1};
    rachis_binding_pool_desc bindingPoolDesc{};
    bindingPoolDesc.struct_size = sizeof(bindingPoolDesc);
    bindingPoolDesc.abi_version = RACHIS_ABI_VERSION;
    bindingPoolDesc.max_binding_sets = 1;
    bindingPoolDesc.sizes = &poolSize;
    bindingPoolDesc.size_count = 1;
    rachis_handle bindingPool = 0;
    requireOk(rachis_binding_pool_create(device.get(), &bindingPoolDesc, &bindingPool, &error), error);

    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);
    const auto storage = createBuffer(
        device.get(), outputSize, RACHIS_BUFFER_STORAGE | RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_DEVICE_LOCAL, error);
    rachis_buffer_binding write{};
    write.buffer = storage;
    write.size = outputSize;
    write.structure_byte_stride = sizeof(std::uint32_t);
    rachis_binding_set_desc setDesc{};
    setDesc.struct_size = sizeof(setDesc);
    setDesc.abi_version = RACHIS_ABI_VERSION;
    setDesc.layout = bindingLayout;
    setDesc.pool = bindingPool;
    setDesc.writes = &write;
    setDesc.write_count = 1;
    rachis_handle set = 0;
    requireOk(rachis_binding_set_create(device.get(), &setDesc, &set, &error), error);

    rachis_pipeline_layout_desc pipelineLayoutDesc{};
    pipelineLayoutDesc.struct_size = sizeof(pipelineLayoutDesc);
    pipelineLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineLayoutDesc.binding_layouts = &bindingLayout;
    pipelineLayoutDesc.binding_layout_count = 1;
    rachis_handle pipelineLayout = 0;
    requireOk(rachis_pipeline_layout_create(device.get(), &pipelineLayoutDesc, &pipelineLayout, &error), error);
    rachis_compute_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.shader = shader;
    rachis_handle pipeline = 0;
    requireOk(rachis_compute_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error), error);

    const auto texture = createTexture(device.get(),
                                       RACHIS_FORMAT_R32_UINT,
                                       valueCount,
                                       1,
                                       RACHIS_TEXTURE_COPY_SOURCE | RACHIS_TEXTURE_COPY_DESTINATION,
                                       error);
    const auto region = wholeTextureRegion(valueCount, 1);
    rachis_texture_transfer_layout layout{};
    layout.struct_size = sizeof(layout);
    requireOk(rachis_texture_query_transfer_layout(device.get(), texture, &region, &layout, &error), error);
    REQUIRE(layout.row_data_size == outputSize);
    const auto textureReadback = createBuffer(
        device.get(), layout.required_buffer_size, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    const auto bufferReadback =
        createBuffer(device.get(), outputSize, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);

    rachis_timestamp_pool_desc queryDesc{};
    queryDesc.struct_size = sizeof(queryDesc);
    queryDesc.abi_version = RACHIS_ABI_VERSION;
    queryDesc.queue_type = RACHIS_QUEUE_GRAPHICS;
    queryDesc.query_count = 2;
    rachis_handle queries = 0;
    requireOk(rachis_timestamp_pool_create(device.get(), &queryDesc, &queries, &error), error);
    const auto timestamps = createBuffer(
        device.get(), 2 * sizeof(std::uint64_t), RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    std::uint64_t frequency = 0;
    requireOk(rachis_queue_timestamp_frequency(device.get(), RACHIS_QUEUE_GRAPHICS, &frequency, &error), error);
    CHECK(frequency > 0);
    CHECK(rachis_queue_timestamp_frequency(device.get(), RACHIS_QUEUE_COPY + 1, &frequency, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
    requireOk(rachis_command_reset_queries(device.get(), command, queries, 0, 2, &error), error);
    requireOk(rachis_command_write_timestamp(device.get(), command, queries, 0, &error), error);
    const auto toWrite = bufferBarrier(
        storage, RACHIS_STAGE_ALL, RACHIS_ACCESS_COMMON, RACHIS_STAGE_COMPUTE_SHADER, RACHIS_ACCESS_UNORDERED_ACCESS);
    requireOk(recordBarriers(device.get(), command, &toWrite, 1, nullptr, 0, error), error);
    requireOk(rachis_command_bind_compute_pipeline(device.get(), command, pipeline, &error), error);
    requireOk(rachis_command_bind_compute_set(device.get(), command, 0, set, &error), error);
    requireOk(rachis_command_dispatch(device.get(), command, 1, 1, 1, &error), error);
    requireOk(rachis_command_write_timestamp(device.get(), command, queries, 1, &error), error);

    const auto toCopy = bufferBarrier(storage,
                                      RACHIS_STAGE_COMPUTE_SHADER,
                                      RACHIS_ACCESS_UNORDERED_ACCESS,
                                      RACHIS_STAGE_COPY,
                                      RACHIS_ACCESS_COPY_SOURCE);
    const auto textureToCopy = textureBarrier(texture,
                                              RACHIS_STAGE_ALL,
                                              RACHIS_ACCESS_NONE,
                                              RACHIS_LAYOUT_UNDEFINED,
                                              RACHIS_STAGE_COPY,
                                              RACHIS_ACCESS_COPY_DESTINATION,
                                              RACHIS_LAYOUT_COPY_DESTINATION);
    requireOk(recordBarriers(device.get(), command, &toCopy, 1, &textureToCopy, 1, error), error);
    auto copyRegion = region;
    copyRegion.bytes_per_row = layout.bytes_per_row;
    requireOk(copyBufferTexture(device.get(), command, storage, texture, copyRegion, true, error), error);
    requireOk(rachis_command_copy_buffer(device.get(), command, storage, bufferReadback, 0, 0, outputSize, &error),
              error);
    const auto textureToRead = textureBarrier(texture,
                                              RACHIS_STAGE_COPY,
                                              RACHIS_ACCESS_COPY_DESTINATION,
                                              RACHIS_LAYOUT_COPY_DESTINATION,
                                              RACHIS_STAGE_COPY,
                                              RACHIS_ACCESS_COPY_SOURCE,
                                              RACHIS_LAYOUT_COPY_SOURCE);
    requireOk(recordBarriers(device.get(), command, nullptr, 0, &textureToRead, 1, error), error);
    requireOk(copyBufferTexture(device.get(), command, textureReadback, texture, copyRegion, false, error), error);
    requireOk(rachis_command_resolve_timestamps(device.get(), command, queries, 0, 2, timestamps, 0, &error), error);
    submitAndWait(device.get(), command, error);

    std::array<std::uint32_t, valueCount> fromBuffer{};
    std::array<std::uint32_t, valueCount> fromTexture{};
    requireOk(rachis_buffer_read(device.get(), bufferReadback, 0, fromBuffer.data(), outputSize, &error), error);
    requireOk(rachis_buffer_read(device.get(), textureReadback, 0, fromTexture.data(), outputSize, &error), error);
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        INFO("value " << index);
        CHECK(fromBuffer[index] == index * 3 + 7);
        CHECK(fromTexture[index] == index * 3 + 7);
    }
    std::array<std::uint64_t, 2> ticks{};
    requireOk(rachis_buffer_read(device.get(), timestamps, 0, ticks.data(), sizeof(ticks), &error), error);
#ifdef __APPLE__
    if (RACHIS_SMOKE_BACKEND != RACHIS_BACKEND_VULKAN || ticks[1] != 0)
    {
        CHECK(ticks[1] >= ticks[0]);
    }
#else
    CHECK(ticks[1] >= ticks[0]);
#endif

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_query_pool_destroy(device.get(), queries, &error), error);
    requireOk(rachis_pipeline_destroy(device.get(), pipeline, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_binding_set_destroy(device.get(), set, &error), error);
    requireOk(rachis_binding_pool_destroy(device.get(), bindingPool, &error), error);
    requireOk(rachis_binding_layout_destroy(device.get(), bindingLayout, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), shader, &error), error);
    requireOk(rachis_texture_destroy(device.get(), texture, &error), error);
    for (const auto buffer : {storage, textureReadback, bufferReadback, timestamps})
    {
        requireOk(rachis_buffer_destroy(device.get(), buffer, &error), error);
    }
    requireNoValidationErrors(device.get(), error);
}
#endif

#ifdef RACHIS_CABI_COMPUTE_PUSH_SHADER_PATH
TEST_CASE("C ABI compute push constants reach a dispatch and reject invalid writes",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][compute]")
{
    const auto bytecode = [] {
        std::ifstream input{RACHIS_CABI_COMPUTE_PUSH_SHADER_PATH, std::ios::binary};
        return std::vector<char>{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    }();
    REQUIRE_FALSE(bytecode.empty());
    auto error = makeError();
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD);

    constexpr std::uint32_t outputBinding = RACHIS_SMOKE_BACKEND == RACHIS_BACKEND_METAL ? 1U : 0U;
    constexpr std::uint32_t pushBytes = 16;
    const rachis_shader_binding_requirement requirement{0, outputBinding, RACHIS_BINDING_STORAGE_BUFFER, 1};
    const rachis_shader_push_constant_requirement pushRequirement{0, pushBytes};
    rachis_shader_module_desc shaderDesc{};
    shaderDesc.struct_size = sizeof(shaderDesc);
    shaderDesc.abi_version = RACHIS_ABI_VERSION;
    shaderDesc.stage = RACHIS_SHADER_COMPUTE;
    shaderDesc.bytecode = bytecode.data();
    shaderDesc.bytecode_size = bytecode.size();
    shaderDesc.entry_point = "main";
    shaderDesc.bindings = &requirement;
    shaderDesc.binding_count = 1;
    shaderDesc.workgroup_x = valueCount;
    shaderDesc.workgroup_y = 1;
    shaderDesc.workgroup_z = 1;
    shaderDesc.push_constants = &pushRequirement;
    shaderDesc.push_constant_count = 1;
    rachis_handle shader = 0;
    requireOk(rachis_shader_module_create(device.get(), &shaderDesc, &shader, &error), error);

    const rachis_binding_layout_entry entry{outputBinding, RACHIS_BINDING_STORAGE_BUFFER, RACHIS_SHADER_COMPUTE, 1, 0};
    rachis_binding_layout_desc bindingLayoutDesc{};
    bindingLayoutDesc.struct_size = sizeof(bindingLayoutDesc);
    bindingLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    bindingLayoutDesc.entries = &entry;
    bindingLayoutDesc.entry_count = 1;
    rachis_handle bindingLayout = 0;
    requireOk(rachis_binding_layout_create(device.get(), &bindingLayoutDesc, &bindingLayout, &error), error);
    const rachis_binding_pool_size poolSize{RACHIS_BINDING_STORAGE_BUFFER, 1};
    rachis_binding_pool_desc bindingPoolDesc{};
    bindingPoolDesc.struct_size = sizeof(bindingPoolDesc);
    bindingPoolDesc.abi_version = RACHIS_ABI_VERSION;
    bindingPoolDesc.max_binding_sets = 1;
    bindingPoolDesc.sizes = &poolSize;
    bindingPoolDesc.size_count = 1;
    rachis_handle bindingPool = 0;
    requireOk(rachis_binding_pool_create(device.get(), &bindingPoolDesc, &bindingPool, &error), error);

    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);
    const auto storage = createBuffer(
        device.get(), outputSize, RACHIS_BUFFER_STORAGE | RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_DEVICE_LOCAL, error);
    const auto readback =
        createBuffer(device.get(), outputSize, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    rachis_buffer_binding write{};
    write.binding = outputBinding;
    write.buffer = storage;
    write.size = outputSize;
    write.structure_byte_stride = sizeof(std::uint32_t);
    rachis_binding_set_desc setDesc{};
    setDesc.struct_size = sizeof(setDesc);
    setDesc.abi_version = RACHIS_ABI_VERSION;
    setDesc.layout = bindingLayout;
    setDesc.pool = bindingPool;
    setDesc.writes = &write;
    setDesc.write_count = 1;
    rachis_handle set = 0;
    requireOk(rachis_binding_set_create(device.get(), &setDesc, &set, &error), error);

    const rachis_push_constant_range pushRange{RACHIS_SHADER_COMPUTE, 0, pushBytes};
    rachis_pipeline_layout_desc pipelineLayoutDesc{};
    pipelineLayoutDesc.struct_size = sizeof(pipelineLayoutDesc);
    pipelineLayoutDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineLayoutDesc.binding_layouts = &bindingLayout;
    pipelineLayoutDesc.binding_layout_count = 1;
    pipelineLayoutDesc.push_constant_ranges = &pushRange;
    pipelineLayoutDesc.push_constant_range_count = 1;
    rachis_handle pipelineLayout = 0;
    requireOk(rachis_pipeline_layout_create(device.get(), &pipelineLayoutDesc, &pipelineLayout, &error), error);
    rachis_compute_pipeline_desc pipelineDesc{};
    pipelineDesc.struct_size = sizeof(pipelineDesc);
    pipelineDesc.abi_version = RACHIS_ABI_VERSION;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.shader = shader;
    rachis_handle pipeline = 0;
    requireOk(rachis_compute_pipeline_create(device.get(), &pipelineDesc, &pipeline, &error), error);

    constexpr std::uint32_t wordBytes = 4;
    constexpr std::uint32_t maxPushConstantWords = 32;
    constexpr std::uint32_t multiplier = 5;
    constexpr std::uint32_t addend = 11;
    const std::array<std::uint32_t, pushBytes / wordBytes> constants{multiplier, 0, 0, 0};
    const std::array<std::uint32_t, maxPushConstantWords + 1> oversized{};
    auto* const wrappingAddress = reinterpret_cast<const void*>(std::numeric_limits<std::uintptr_t>::max() - 1);

    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_GRAPHICS, pool, error);
    CHECK(rachis_command_set_compute_push_constants(nullptr, command, 0, constants.data(), pushBytes, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_compute_push_constants(device.get(), command, 0, nullptr, pushBytes, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_compute_push_constants(device.get(), command, 0, wrappingAddress, pushBytes, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_compute_push_constants(device.get(), 0, 0, constants.data(), pushBytes, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_compute_push_constants(device.get(), command, 0, constants.data(), pushBytes, &error) ==
          RACHIS_STATUS_INVALID_STATE);
    CHECK(error.code == RACHIS_STATUS_INVALID_STATE);
    CHECK(rachis_command_set_graphics_push_constants(
              device.get(), command, RACHIS_SHADER_COMPUTE, 0, constants.data(), pushBytes, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_graphics_push_constants(device.get(),
                                                     command,
                                                     RACHIS_SHADER_PIXEL | RACHIS_SHADER_COMPUTE,
                                                     0,
                                                     constants.data(),
                                                     pushBytes,
                                                     &error) == RACHIS_STATUS_INVALID_ARGUMENT);

    requireOk(rachis_command_bind_compute_pipeline(device.get(), command, pipeline, &error), error);
    CHECK(rachis_command_set_compute_push_constants(device.get(), command, 0, nullptr, 0, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_compute_push_constants(device.get(), command, 0, constants.data(), 0, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_compute_push_constants(device.get(), command, 2, constants.data(), wordBytes, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_compute_push_constants(
              device.get(), command, 0, constants.data(), wordBytes + 2, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_compute_push_constants(
              device.get(), command, maxPushConstantWords * wordBytes, constants.data(), wordBytes, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    CHECK(rachis_command_set_compute_push_constants(
              device.get(), command, 0, oversized.data(), (maxPushConstantWords + 1) * wordBytes, &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);

    requireOk(rachis_command_set_compute_push_constants(device.get(), command, 0, constants.data(), pushBytes, &error),
              error);
    requireOk(rachis_command_set_compute_push_constants(device.get(), command, wordBytes, &addend, wordBytes, &error),
              error);
    requireOk(rachis_command_bind_compute_set(device.get(), command, 0, set, &error), error);
    const auto toWrite = bufferBarrier(
        storage, RACHIS_STAGE_ALL, RACHIS_ACCESS_COMMON, RACHIS_STAGE_COMPUTE_SHADER, RACHIS_ACCESS_UNORDERED_ACCESS);
    requireOk(recordBarriers(device.get(), command, &toWrite, 1, nullptr, 0, error), error);
    requireOk(rachis_command_dispatch(device.get(), command, 1, 1, 1, &error), error);
    const auto toCopy = bufferBarrier(storage,
                                      RACHIS_STAGE_COMPUTE_SHADER,
                                      RACHIS_ACCESS_UNORDERED_ACCESS,
                                      RACHIS_STAGE_COPY,
                                      RACHIS_ACCESS_COPY_SOURCE);
    requireOk(recordBarriers(device.get(), command, &toCopy, 1, nullptr, 0, error), error);
    requireOk(rachis_command_copy_buffer(device.get(), command, storage, readback, 0, 0, outputSize, &error), error);
    submitAndWait(device.get(), command, error);

    std::array<std::uint32_t, valueCount> values{};
    requireOk(rachis_buffer_read(device.get(), readback, 0, values.data(), outputSize, &error), error);
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        INFO("value " << index);
        CHECK(values[index] == index * multiplier + addend);
    }

    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_pipeline_destroy(device.get(), pipeline, &error), error);
    requireOk(rachis_pipeline_layout_destroy(device.get(), pipelineLayout, &error), error);
    requireOk(rachis_binding_set_destroy(device.get(), set, &error), error);
    requireOk(rachis_binding_pool_destroy(device.get(), bindingPool, &error), error);
    requireOk(rachis_binding_layout_destroy(device.get(), bindingLayout, &error), error);
    requireOk(rachis_shader_module_destroy(device.get(), shader, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), storage, &error), error);
    requireNoValidationErrors(device.get(), error);
}
#endif

TEST_CASE("C ABI reports diagnostics and rejects wrong output layouts",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi]")
{
    auto error = makeError();
    DiagnosticCapture capture;
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD, &capture);

    rachis_diagnostic_counts counts{};
    counts.struct_size = sizeof(counts) + 1;
    CHECK(rachis_device_diagnostic_counts(device.get(), &counts, &error) == RACHIS_STATUS_ABI_MISMATCH);
    CHECK(error.code == RACHIS_STATUS_ABI_MISMATCH);
    std::uint32_t state = RACHIS_DEVICE_INVALID;
    requireOk(rachis_device_operational_state(device.get(), &state, &error), error);
    CHECK(state == RACHIS_DEVICE_ACTIVE);
    std::uint32_t backend = RACHIS_BACKEND_METAL;
    requireOk(rachis_device_backend(device.get(), &backend, &error), error);
    CHECK(backend == RACHIS_SMOKE_BACKEND);
    rachis_adapter_info adapter{};
    adapter.struct_size = sizeof(adapter) - 1;
    CHECK(rachis_device_adapter_info(device.get(), &adapter, &error) == RACHIS_STATUS_ABI_MISMATCH);
    adapter.struct_size = sizeof(adapter);
    requireOk(rachis_device_adapter_info(device.get(), &adapter, &error), error);
    CHECK(adapter.backend == backend);
    CHECK(adapter.type <= RACHIS_ADAPTER_TYPE_SOFTWARE);
    if constexpr (RACHIS_SMOKE_ADAPTER == RACHIS_ADAPTER_WARP)
    {
        CHECK(adapter.type == RACHIS_ADAPTER_TYPE_SOFTWARE);
    }
    CHECK(adapter.name[0] != '\0');
    CHECK(adapter.name[sizeof(adapter.name) - 1] == '\0');
    auto loss = makeError();
    loss.struct_size = 0;
    CHECK(rachis_device_loss_error(device.get(), &loss, &error) == RACHIS_STATUS_ABI_MISMATCH);
    loss = makeError();
    requireOk(rachis_device_loss_error(device.get(), &loss, &error), error);
    CHECK(loss.code == RACHIS_STATUS_OK);

    const std::array<std::uint8_t, 4> bytes{9, 8, 7, 6};
    const auto upload =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    requireOk(rachis_buffer_write(device.get(), upload, 0, bytes.data(), bytes.size(), &error), error);
    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_COPY, pool, error);
    requireOk(rachis_command_copy_buffer(device.get(), command, upload, readback, 0, 0, bytes.size(), &error), error);
    submitAndWait(device.get(), command, error);

    counts.struct_size = sizeof(counts);
    requireOk(rachis_device_diagnostic_counts(device.get(), &counts, &error), error);
    {
        const std::scoped_lock lock(capture.mutex);
        INFO(capture.messages);
        CHECK(counts.errors == 0);
        CHECK(counts.corruptions == 0);
        for (const auto severity : capture.severities)
        {
            CHECK(severity <= RACHIS_DIAGNOSTIC_CORRUPTION);
            CHECK(severity < RACHIS_DIAGNOSTIC_ERROR);
        }
    }
    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
}

TEST_CASE("C ABI supports concurrent command recording across pools and concurrent read queries",
          RACHIS_CABI_DEVICE_SIZE RACHIS_CABI_DEVICE_REQUIRES "[rachis][cabi][concurrency]")
{
    constexpr std::size_t workerCount = 4;
    constexpr std::size_t readerCount = 2;
    constexpr std::size_t iterationCount = 3;
    constexpr std::size_t wordCount = 16;
    constexpr std::uint64_t bufferByteSize = wordCount * sizeof(std::uint32_t);

    auto error = makeError();
    DiagnosticCapture capture;
    auto device = createDevice(error, RACHIS_VALIDATION_STANDARD, &capture);

    std::array<std::uint32_t, wordCount> probeWords{};
    for (std::size_t i = 0; i < wordCount; ++i)
    {
        probeWords[i] = static_cast<std::uint32_t>(0xA5A50000U + i * 17U);
    }
    const auto probeUpload =
        createBuffer(device.get(), bufferByteSize, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto probeReadback =
        createBuffer(device.get(), bufferByteSize, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    requireOk(rachis_buffer_write(device.get(), probeUpload, 0, probeWords.data(), bufferByteSize, &error), error);
    rachis_handle setupPool = 0;
    const auto setupCommand = acquireCommand(device.get(), RACHIS_QUEUE_COPY, setupPool, error);
    requireOk(rachis_command_copy_buffer(
                  device.get(), setupCommand, probeUpload, probeReadback, 0, 0, bufferByteSize, &error),
              error);
    submitAndWait(device.get(), setupCommand, error);
    requireOk(rachis_command_pool_destroy(device.get(), setupPool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), probeUpload, &error), error);

    const auto probeTexture = createTexture(
        device.get(), RACHIS_FORMAT_RGBA8_UNORM, textureSize, textureSize, RACHIS_TEXTURE_COPY_SOURCE, error);
    const auto fullRegion = wholeTextureRegion(textureSize, textureSize);
    rachis_texture_transfer_layout expectedTransfer{};
    expectedTransfer.struct_size = sizeof(expectedTransfer);
    requireOk(rachis_texture_query_transfer_layout(device.get(), probeTexture, &fullRegion, &expectedTransfer, &error),
              error);

    struct IterationSlot final
    {
        rachis_handle command = 0;
        rachis_handle timeline = 0;
        rachis_handle upload = 0;
        rachis_handle deviceLocal = 0;
        rachis_handle bufferReadback = 0;
        rachis_handle colorTexture = 0;
        rachis_handle colorView = 0;
        rachis_handle textureReadback = 0;
        rachis_texture_transfer_layout transfer{};
        std::array<std::uint32_t, wordCount> expectedWords{};
        std::array<std::uint8_t, 4> expectedPixel{};
        std::array<std::uint32_t, wordCount> actualWords{};
        std::vector<std::uint8_t> actualPixels;
    };

    struct WorkerContext final
    {
        rachis_handle pool = 0;
        std::array<IterationSlot, iterationCount> slots{};
        rachis_status status = RACHIS_STATUS_OK;
        std::string failureMessage;
    };

    struct ReaderOutcome final
    {
        rachis_status status = RACHIS_STATUS_OK;
        std::string failureMessage;
        bool dataMatched = true;
    };

    std::array<WorkerContext, workerCount> workers{};
    for (std::size_t w = 0; w < workerCount; ++w)
    {
        rachis_command_pool_desc poolDesc{};
        poolDesc.struct_size = sizeof(poolDesc);
        poolDesc.abi_version = RACHIS_ABI_VERSION;
        poolDesc.queue_type = RACHIS_QUEUE_GRAPHICS;
        poolDesc.max_command_buffers = static_cast<std::uint32_t>(iterationCount + 1);
        requireOk(rachis_command_pool_create(device.get(), &poolDesc, &workers[w].pool, &error), error);

        for (std::size_t iter = 0; iter < iterationCount; ++iter)
        {
            auto& slot = workers[w].slots[iter];
            requireOk(rachis_command_acquire(device.get(), workers[w].pool, &slot.command, &error), error);
            slot.upload =
                createBuffer(device.get(), bufferByteSize, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
            slot.deviceLocal = createBuffer(device.get(),
                                            bufferByteSize,
                                            RACHIS_BUFFER_COPY_SOURCE | RACHIS_BUFFER_COPY_DESTINATION,
                                            RACHIS_MEMORY_DEVICE_LOCAL,
                                            error);
            slot.bufferReadback = createBuffer(
                device.get(), bufferByteSize, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
            slot.colorTexture = createTexture(device.get(),
                                              RACHIS_FORMAT_RGBA8_UNORM,
                                              textureSize,
                                              textureSize,
                                              RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_COPY_SOURCE,
                                              error);
            const auto viewDesc = renderTargetViewDesc(slot.colorTexture);
            requireOk(rachis_texture_view_create(device.get(), &viewDesc, &slot.colorView, &error), error);
            slot.transfer.struct_size = sizeof(slot.transfer);
            requireOk(rachis_texture_query_transfer_layout(
                          device.get(), slot.colorTexture, &fullRegion, &slot.transfer, &error),
                      error);
            slot.textureReadback = createBuffer(device.get(),
                                                slot.transfer.required_buffer_size,
                                                RACHIS_BUFFER_COPY_DESTINATION,
                                                RACHIS_MEMORY_READBACK,
                                                error);
            slot.actualPixels.resize(static_cast<std::size_t>(slot.transfer.required_buffer_size));

            for (std::size_t word = 0; word < wordCount; ++word)
            {
                slot.expectedWords[word] =
                    static_cast<std::uint32_t>((w + 1U) * 10000U + (iter + 1U) * 100U + word * 3U + 5U);
            }
            slot.expectedPixel = {
                static_cast<std::uint8_t>(24U + w * 52U),
                static_cast<std::uint8_t>(40U + iter * 64U),
                static_cast<std::uint8_t>(210U - w * 40U),
                255U,
            };
        }
    }

    std::mutex startMutex;
    std::condition_variable startCv;
    std::size_t readyThreads = 0;
    bool go = false;
    const auto waitForStart = [&] {
        std::unique_lock lock(startMutex);
        ++readyThreads;
        if (readyThreads == workerCount + readerCount)
        {
            go = true;
            startCv.notify_all();
        }
        else
        {
            startCv.wait(lock, [&] { return go; });
        }
    };

    std::array<std::thread, workerCount> workerThreads{};
    std::array<std::thread, readerCount> readerThreads{};
    std::array<ReaderOutcome, readerCount> readerOutcomes{};

    for (std::size_t w = 0; w < workerCount; ++w)
    {
        workerThreads[w] = std::thread([&, w] {
            waitForStart();
            auto& worker = workers[w];
            auto localError = makeError();
            const auto recordStep = [&](rachis_status stepStatus) {
                if (stepStatus != RACHIS_STATUS_OK && worker.status == RACHIS_STATUS_OK)
                {
                    worker.status = stepStatus;
                    worker.failureMessage = localError.message;
                }
                return stepStatus == RACHIS_STATUS_OK;
            };

            for (auto& slot : worker.slots)
            {
                if (!recordStep(rachis_buffer_write(
                        device.get(), slot.upload, 0, slot.expectedWords.data(), bufferByteSize, &localError)))
                {
                    return;
                }
                if (!recordStep(rachis_command_copy_buffer(
                        device.get(), slot.command, slot.upload, slot.deviceLocal, 0, 0, bufferByteSize, &localError)))
                {
                    return;
                }

                const auto bufToCopySrc = bufferBarrier(slot.deviceLocal,
                                                        RACHIS_STAGE_COPY,
                                                        RACHIS_ACCESS_COPY_DESTINATION,
                                                        RACHIS_STAGE_COPY,
                                                        RACHIS_ACCESS_COPY_SOURCE);
                const auto texToRender = textureBarrier(slot.colorTexture,
                                                        RACHIS_STAGE_ALL,
                                                        RACHIS_ACCESS_NONE,
                                                        RACHIS_LAYOUT_UNDEFINED,
                                                        RACHIS_STAGE_RENDER_TARGET,
                                                        RACHIS_ACCESS_RENDER_TARGET,
                                                        RACHIS_LAYOUT_RENDER_TARGET);
                if (!recordStep(
                        recordBarriers(device.get(), slot.command, &bufToCopySrc, 1, &texToRender, 1, localError)))
                {
                    return;
                }
                if (!recordStep(rachis_command_copy_buffer(device.get(),
                                                           slot.command,
                                                           slot.deviceLocal,
                                                           slot.bufferReadback,
                                                           0,
                                                           0,
                                                           bufferByteSize,
                                                           &localError)))
                {
                    return;
                }

                rachis_color_attachment colorAttachment{};
                colorAttachment.view = slot.colorView;
                colorAttachment.load_op = RACHIS_LOAD_OP_CLEAR;
                colorAttachment.store_op = RACHIS_STORE_OP_STORE;
                colorAttachment.clear_color[0] = static_cast<float>(slot.expectedPixel[0]) / 255.0F;
                colorAttachment.clear_color[1] = static_cast<float>(slot.expectedPixel[1]) / 255.0F;
                colorAttachment.clear_color[2] = static_cast<float>(slot.expectedPixel[2]) / 255.0F;
                colorAttachment.clear_color[3] = 1.0F;
                rachis_rendering_desc rendering{};
                rendering.struct_size = sizeof(rendering);
                rendering.abi_version = RACHIS_ABI_VERSION;
                rendering.color_attachments = &colorAttachment;
                rendering.color_attachment_count = 1;
                if (!recordStep(rachis_command_begin_rendering(device.get(), slot.command, &rendering, &localError)))
                {
                    return;
                }
                if (!recordStep(rachis_command_end_rendering(device.get(), slot.command, &localError)))
                {
                    return;
                }

                const auto texToCopySrc = textureBarrier(slot.colorTexture,
                                                         RACHIS_STAGE_RENDER_TARGET,
                                                         RACHIS_ACCESS_RENDER_TARGET,
                                                         RACHIS_LAYOUT_RENDER_TARGET,
                                                         RACHIS_STAGE_COPY,
                                                         RACHIS_ACCESS_COPY_SOURCE,
                                                         RACHIS_LAYOUT_COPY_SOURCE);
                if (!recordStep(recordBarriers(device.get(), slot.command, nullptr, 0, &texToCopySrc, 1, localError)))
                {
                    return;
                }

                auto copyRegion = fullRegion;
                copyRegion.bytes_per_row = slot.transfer.bytes_per_row;
                if (!recordStep(copyBufferTexture(device.get(),
                                                  slot.command,
                                                  slot.textureReadback,
                                                  slot.colorTexture,
                                                  copyRegion,
                                                  false,
                                                  localError)))
                {
                    return;
                }
            }
        });
    }

    for (std::size_t r = 0; r < readerCount; ++r)
    {
        readerThreads[r] = std::thread([&, r] {
            waitForStart();
            auto& outcome = readerOutcomes[r];
            auto localError = makeError();
            const auto checkStep = [&](rachis_status stepStatus) {
                if (stepStatus != RACHIS_STATUS_OK && outcome.status == RACHIS_STATUS_OK)
                {
                    outcome.status = stepStatus;
                    outcome.failureMessage = localError.message;
                }
                return stepStatus == RACHIS_STATUS_OK;
            };

            for (int poll = 0; poll < 32; ++poll)
            {
                std::array<std::uint32_t, wordCount> readWords{};
                if (!checkStep(rachis_buffer_read(
                        device.get(), probeReadback, 0, readWords.data(), bufferByteSize, &localError)))
                {
                    return;
                }
                if (readWords != probeWords)
                {
                    outcome.dataMatched = false;
                }

                rachis_texture_transfer_layout queriedLayout{};
                queriedLayout.struct_size = sizeof(queriedLayout);
                if (!checkStep(rachis_texture_query_transfer_layout(
                        device.get(), probeTexture, &fullRegion, &queriedLayout, &localError)))
                {
                    return;
                }
                if (queriedLayout.bytes_per_row != expectedTransfer.bytes_per_row ||
                    queriedLayout.required_buffer_size != expectedTransfer.required_buffer_size)
                {
                    outcome.dataMatched = false;
                }

                rachis_diagnostic_counts counts{};
                counts.struct_size = sizeof(counts);
                if (!checkStep(rachis_device_diagnostic_counts(device.get(), &counts, &localError)))
                {
                    return;
                }
                std::uint32_t operationalState = RACHIS_DEVICE_INVALID;
                if (!checkStep(rachis_device_operational_state(device.get(), &operationalState, &localError)))
                {
                    return;
                }
                if (operationalState != RACHIS_DEVICE_ACTIVE)
                {
                    outcome.dataMatched = false;
                }
                std::uint64_t frequency = 0;
                if (!checkStep(
                        rachis_queue_timestamp_frequency(device.get(), RACHIS_QUEUE_GRAPHICS, &frequency, &localError)))
                {
                    return;
                }
                if (frequency == 0)
                {
                    outcome.dataMatched = false;
                }
            }
        });
    }

    for (auto& thread : workerThreads)
    {
        thread.join();
    }
    for (auto& thread : readerThreads)
    {
        thread.join();
    }

    for (std::size_t w = 0; w < workerCount; ++w)
    {
        INFO("worker " << w << ": " << workers[w].failureMessage);
        REQUIRE(workers[w].status == RACHIS_STATUS_OK);
    }
    for (std::size_t r = 0; r < readerCount; ++r)
    {
        INFO("reader " << r << ": " << readerOutcomes[r].failureMessage);
        REQUIRE(readerOutcomes[r].status == RACHIS_STATUS_OK);
        CHECK(readerOutcomes[r].dataMatched);
    }

    for (std::size_t w = 0; w < workerCount; ++w)
    {
        workerThreads[w] = std::thread([&, w] {
            auto& worker = workers[w];
            auto localError = makeError();
            const auto checkStep = [&](rachis_status stepStatus) {
                if (stepStatus != RACHIS_STATUS_OK && worker.status == RACHIS_STATUS_OK)
                {
                    worker.status = stepStatus;
                    worker.failureMessage = localError.message;
                }
                return stepStatus == RACHIS_STATUS_OK;
            };

            for (auto& slot : worker.slots)
            {
                if (!checkStep(rachis_command_submit(device.get(), slot.command, &slot.timeline, &localError)))
                {
                    return;
                }
            }
            for (auto& slot : worker.slots)
            {
                if (!checkStep(rachis_timeline_wait(device.get(), slot.timeline, &localError)))
                {
                    return;
                }
                if (!checkStep(rachis_buffer_read(
                        device.get(), slot.bufferReadback, 0, slot.actualWords.data(), bufferByteSize, &localError)))
                {
                    return;
                }
                if (!checkStep(rachis_buffer_read(device.get(),
                                                  slot.textureReadback,
                                                  0,
                                                  slot.actualPixels.data(),
                                                  slot.actualPixels.size(),
                                                  &localError)))
                {
                    return;
                }
                if (!checkStep(rachis_timeline_release(device.get(), slot.timeline, &localError)))
                {
                    return;
                }
                slot.timeline = 0;
            }
        });
    }
    for (auto& thread : workerThreads)
    {
        thread.join();
    }

    for (std::size_t w = 0; w < workerCount; ++w)
    {
        INFO("worker submit/wait " << w << ": " << workers[w].failureMessage);
        REQUIRE(workers[w].status == RACHIS_STATUS_OK);
        for (std::size_t iter = 0; iter < iterationCount; ++iter)
        {
            const auto& slot = workers[w].slots[iter];
            INFO("worker " << w << " iter " << iter);
            CHECK(slot.actualWords == slot.expectedWords);
            for (std::uint32_t row = 0; row < textureSize; ++row)
            {
                for (std::uint32_t col = 0; col < textureSize; ++col)
                {
                    const auto* pixel = slot.actualPixels.data() + row * slot.transfer.bytes_per_row + col * 4U;
                    CHECK(nearByte(pixel[0], slot.expectedPixel[0]));
                    CHECK(nearByte(pixel[1], slot.expectedPixel[1]));
                    CHECK(nearByte(pixel[2], slot.expectedPixel[2]));
                    CHECK(nearByte(pixel[3], slot.expectedPixel[3]));
                }
            }
        }
    }

    for (auto& worker : workers)
    {
        for (const auto& slot : worker.slots)
        {
            requireOk(rachis_buffer_destroy(device.get(), slot.textureReadback, &error), error);
            requireOk(rachis_texture_view_destroy(device.get(), slot.colorView, &error), error);
            requireOk(rachis_texture_destroy(device.get(), slot.colorTexture, &error), error);
            requireOk(rachis_buffer_destroy(device.get(), slot.bufferReadback, &error), error);
            requireOk(rachis_buffer_destroy(device.get(), slot.deviceLocal, &error), error);
            requireOk(rachis_buffer_destroy(device.get(), slot.upload, &error), error);
        }
        requireOk(rachis_command_pool_destroy(device.get(), worker.pool, &error), error);
    }
    requireOk(rachis_texture_destroy(device.get(), probeTexture, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), probeReadback, &error), error);
    requireNoValidationErrors(device.get(), error);
    {
        const std::scoped_lock lock(capture.mutex);
        INFO(capture.messages);
        for (const auto severity : capture.severities)
        {
            CHECK(severity < RACHIS_DIAGNOSTIC_ERROR);
        }
    }
}

#ifdef RACHIS_CABI_D3D12_DEVICE_REMOVAL
TEST_CASE("C ABI publishes D3D12 device loss through status, state, record and callback",
          "[size:medium][rachis][cabi][warp]")
{
    auto error = makeError();
    DiagnosticCapture capture;
    auto device = createDevice(error, RACHIS_VALIDATION_DISABLED, &capture);
    const std::array<std::uint8_t, 4> bytes{1, 2, 3, 4};
    const auto upload =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback =
        createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    const auto stale = createBuffer(device.get(), bytes.size(), RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    requireOk(rachis_buffer_destroy(device.get(), stale, &error), error);
    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_COPY, pool, error);
    rachis_handle unsubmitted = 0;
    requireOk(rachis_command_acquire(device.get(), pool, &unsubmitted, &error), error);
    requireOk(rachis_command_copy_buffer(device.get(), command, upload, readback, 0, 0, bytes.size(), &error), error);

    IDXGIFactory4* factory = nullptr;
    REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
    IDXGIAdapter* adapter = nullptr;
    REQUIRE(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))));
    ID3D12Device5* removable = nullptr;
    REQUIRE(SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&removable))));
    removable->RemoveDevice();
    removable->Release();
    adapter->Release();
    factory->Release();

    rachis_handle timeline = 0;
    auto status = rachis_command_submit(device.get(), command, &timeline, &error);
    if (status == RACHIS_STATUS_OK)
    {
        status = rachis_timeline_wait(device.get(), timeline, &error);
    }
    CHECK(status == RACHIS_STATUS_DEVICE_LOST);
    CHECK(error.code == RACHIS_STATUS_DEVICE_LOST);

    std::uint32_t state = RACHIS_DEVICE_INVALID;
    requireOk(rachis_device_operational_state(device.get(), &state, &error), error);
    CHECK(state == RACHIS_DEVICE_LOST);
    auto loss = makeError();
    requireOk(rachis_device_loss_error(device.get(), &loss, &error), error);
    CHECK(loss.code == RACHIS_STATUS_DEVICE_LOST);
    CHECK(loss.native_code != 0);
    CHECK(loss.message[0] != '\0');

    rachis_buffer_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RACHIS_ABI_VERSION;
    desc.size = 16;
    desc.usage = RACHIS_BUFFER_COPY_SOURCE;
    desc.memory_type = RACHIS_MEMORY_UPLOAD;
    rachis_handle rejected = 0;
    CHECK(rachis_buffer_create(device.get(), &desc, &rejected, &error) == RACHIS_STATUS_DEVICE_LOST);
    CHECK(rejected == 0);

    std::array<std::uint8_t, 4> copied{};
    CHECK(rachis_buffer_write(device.get(), upload, 0, bytes.data(), bytes.size(), &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_read(device.get(), readback, 0, copied.data(), copied.size(), &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    error = makeError();
    CHECK(rachis_buffer_write(device.get(), stale, 0, bytes.data(), bytes.size(), &error) == RACHIS_STATUS_DEVICE_LOST);
    CHECK(error.code == RACHIS_STATUS_DEVICE_LOST);
    error = makeError();
    CHECK(rachis_buffer_read(device.get(), stale, 0, copied.data(), copied.size(), &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(error.code == RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_write(device.get(), upload, bytes.size(), bytes.data(), bytes.size(), &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_read(device.get(), readback, bytes.size(), copied.data(), copied.size(), &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(rachis_buffer_write(device.get(), upload, 0, nullptr, bytes.size(), &error) ==
          RACHIS_STATUS_INVALID_ARGUMENT);
    {
        const std::scoped_lock lock(capture.mutex);
        INFO(capture.messages);
        bool reported = false;
        for (const auto severity : capture.severities)
        {
            reported = reported || severity == RACHIS_DIAGNOSTIC_ERROR;
        }
        CHECK(reported);
    }

    CHECK(rachis_command_discard(device.get(), command, &error) == RACHIS_STATUS_INVALID_ARGUMENT);
    if (timeline != 0)
    {
        requireOk(rachis_timeline_release(device.get(), timeline, &error), error);
    }
    rachis_handle rejectedTimeline = 0;
    CHECK(rachis_command_submit(device.get(), unsubmitted, &rejectedTimeline, &error) == RACHIS_STATUS_DEVICE_LOST);
    CHECK(rejectedTimeline == 0);
    requireOk(rachis_command_discard(device.get(), unsubmitted, &error), error);
    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), upload, &error), error);
}

TEST_CASE("C ABI keeps a mapped pointer writable after D3D12 device loss and still unmaps it",
          "[size:medium][rachis][cabi][warp]")
{
    constexpr std::uint64_t bufferSize = 256;
    auto error = makeError();
    auto device = createDevice(error);
    const auto mappedUpload =
        createBuffer(device.get(), bufferSize, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto unmappedUpload =
        createBuffer(device.get(), bufferSize, RACHIS_BUFFER_COPY_SOURCE, RACHIS_MEMORY_UPLOAD, error);
    const auto readback =
        createBuffer(device.get(), bufferSize, RACHIS_BUFFER_COPY_DESTINATION, RACHIS_MEMORY_READBACK, error);
    rachis_mapped_memory mapped{};
    mapped.struct_size = sizeof(mapped);
    requireOk(rachis_buffer_map(device.get(), mappedUpload, 0, RACHIS_WHOLE_SIZE, &mapped, &error), error);
    REQUIRE(mapped.data != nullptr);
    std::memset(mapped.data, 0x5A, static_cast<std::size_t>(mapped.size));
    rachis_handle pool = 0;
    const auto command = acquireCommand(device.get(), RACHIS_QUEUE_COPY, pool, error);
    requireOk(rachis_command_copy_buffer(device.get(), command, mappedUpload, readback, 0, 0, bufferSize, &error),
              error);

    IDXGIFactory4* factory = nullptr;
    REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
    IDXGIAdapter* adapter = nullptr;
    REQUIRE(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))));
    ID3D12Device5* removable = nullptr;
    REQUIRE(SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&removable))));
    removable->RemoveDevice();
    removable->Release();
    adapter->Release();
    factory->Release();

    rachis_handle timeline = 0;
    auto status = rachis_command_submit(device.get(), command, &timeline, &error);
    if (status == RACHIS_STATUS_OK)
    {
        status = rachis_timeline_wait(device.get(), timeline, &error);
    }
    CHECK(status == RACHIS_STATUS_DEVICE_LOST);
    std::uint32_t state = RACHIS_DEVICE_INVALID;
    requireOk(rachis_device_operational_state(device.get(), &state, &error), error);
    REQUIRE(state == RACHIS_DEVICE_LOST);

    std::memset(mapped.data, 0xA5, static_cast<std::size_t>(mapped.size));
    CHECK(static_cast<const std::uint8_t*>(mapped.data)[bufferSize - 1] == 0xA5);

    const rachis_mapped_range wholeRange{0, RACHIS_WHOLE_SIZE};
    CHECK(rachis_buffer_flush_mapped_ranges(device.get(), mappedUpload, &wholeRange, 1, &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    rachis_mapped_memory rejected{};
    rejected.struct_size = sizeof(rejected);
    CHECK(rachis_buffer_map(device.get(), unmappedUpload, 0, RACHIS_WHOLE_SIZE, &rejected, &error) ==
          RACHIS_STATUS_DEVICE_LOST);
    CHECK(rejected.data == nullptr);
    CHECK(rejected.size == 0);
    CHECK(rachis_buffer_destroy(device.get(), mappedUpload, &error) == RACHIS_STATUS_INVALID_STATE);
    rachis_device_capabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    requireOk(rachis_device_query_capabilities(device.get(), &capabilities, &error), error);
    CHECK(capabilities.persistent_mapping == 1);

    requireOk(rachis_buffer_unmap(device.get(), mappedUpload, &error), error);
    if (timeline != 0)
    {
        requireOk(rachis_timeline_release(device.get(), timeline, &error), error);
    }
    requireOk(rachis_command_pool_destroy(device.get(), pool, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), readback, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), unmappedUpload, &error), error);
    requireOk(rachis_buffer_destroy(device.get(), mappedUpload, &error), error);
}
#endif

#ifdef RACHIS_CABI_D3D12_DEVICE_REMOVAL
TEST_CASE("C ABI reports a D3D12 hardware adapter as integrated exactly when it has unified memory",
          "[size:medium][requires:HardwareD3D12][rachis][cabi]")
{
    auto error = makeError();
    rachis_device_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RACHIS_ABI_VERSION;
    desc.backend = RACHIS_BACKEND_D3D12;
    desc.adapter_preference = RACHIS_ADAPTER_HIGH_PERFORMANCE;
    desc.validation_level = RACHIS_VALIDATION_DISABLED;
    desc.application_name = "Rachis C ABI adapter type";
    rachis_device* native = nullptr;
    const auto status = rachis_device_create(&desc, &native, &error);
    if (status == RACHIS_STATUS_ADAPTER_NOT_FOUND)
    {
        SKIP(error.message);
    }
    requireOk(status, error);
    const Device device{native, &rachis_device_destroy};

    rachis_adapter_info adapter{};
    adapter.struct_size = sizeof(adapter);
    requireOk(rachis_device_adapter_info(device.get(), &adapter, &error), error);
    INFO(adapter.name);
    REQUIRE(adapter.type != RACHIS_ADAPTER_TYPE_SOFTWARE);

    IDXGIFactory4* factory = nullptr;
    REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
    const LUID luid{
        .LowPart = static_cast<DWORD>(adapter.id & 0xFFFFFFFFU),
        .HighPart = static_cast<LONG>(adapter.id >> 32U),
    };
    IDXGIAdapter1* nativeAdapter = nullptr;
    const auto found = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&nativeAdapter));
    factory->Release();
    REQUIRE(SUCCEEDED(found));
    ID3D12Device* nativeDevice = nullptr;
    const auto opened = D3D12CreateDevice(nativeAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&nativeDevice));
    nativeAdapter->Release();
    REQUIRE(SUCCEEDED(opened));
    D3D12_FEATURE_DATA_ARCHITECTURE1 architecture{};
    const auto queried =
        nativeDevice->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &architecture, sizeof(architecture));
    nativeDevice->Release();
    REQUIRE(SUCCEEDED(queried));
    const std::uint32_t expected =
        architecture.UMA != FALSE ? RACHIS_ADAPTER_TYPE_INTEGRATED : RACHIS_ADAPTER_TYPE_DISCRETE;
    CHECK(adapter.type == expected);
}
#endif

#ifdef RACHIS_CABI_D3D12_DEVICE_REMOVAL
TEST_CASE("C ABI opens the D3D12 WARP adapter by ID under the hardware preferences",
          "[size:medium][rachis][cabi][warp]")
{
    auto error = makeError();
    const auto open = [&](std::uint32_t preference, std::uint64_t adapterId) {
        rachis_device_desc desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = RACHIS_ABI_VERSION;
        desc.backend = RACHIS_BACKEND_D3D12;
        desc.adapter_preference = preference;
        desc.validation_level = RACHIS_VALIDATION_DISABLED;
        desc.adapter_id = adapterId;
        desc.application_name = "Rachis C ABI adapter ID";
        rachis_device* native = nullptr;
        const auto status = rachis_device_create(&desc, &native, &error);
        requireOk(status, error);
        REQUIRE(native != nullptr);
        Device device{native, &rachis_device_destroy};
        rachis_adapter_info adapter{};
        adapter.struct_size = sizeof(adapter);
        requireOk(rachis_device_adapter_info(device.get(), &adapter, &error), error);
        return adapter;
    };

    const auto warp = open(RACHIS_ADAPTER_WARP, 0);
    REQUIRE(warp.type == RACHIS_ADAPTER_TYPE_SOFTWARE);
    REQUIRE(warp.id != 0);

    for (const std::uint32_t preference : {RACHIS_ADAPTER_HIGH_PERFORMANCE, RACHIS_ADAPTER_MINIMUM_POWER})
    {
        INFO("adapter_preference " << preference);
        const auto adapter = open(preference, warp.id);
        CHECK(adapter.id == warp.id);
        CHECK(adapter.type == RACHIS_ADAPTER_TYPE_SOFTWARE);
    }
}
#endif
