# C ABI

`Rachis/rachis.h` exposes a **C11 interface** to RachisRHI so independently compiled native modules can share a single GPU device across binary boundaries. It provides a focused subset of the C++ API (see [Coverage](#coverage)).

The C ABI lives in the `rachis` shared library (`Rachis::C` in CMake) and exports only `rachis_*` symbols with C linkage.

## Contents

- [Version](#version)
- [Creating a device](#creating-a-device)
- [Descriptors and errors](#descriptors-and-errors)
- [Coverage](#coverage)
- [Device capabilities](#device-capabilities)
- [Mapped buffers](#mapped-buffers)
- [Vertex buffers and indexed draws](#vertex-buffers-and-indexed-draws)
- [Adapters](#adapters)
- [Surfaces](#surfaces)
- [Handles and lifetime](#handles-and-lifetime)
  - [Timelines](#timelines)
  - [Texture leases](#texture-leases)
  - [Swapchain images](#swapchain-images)
- [Threads](#threads)
- [Device loss](#device-loss)
- [Timestamps and memory budget](#timestamps-and-memory-budget)

## Version

Call `rachis_abi_version()` before creating a device and abort unless it equals the `RACHIS_ABI_VERSION` your module compiled against (currently `3`). ABI versions are not backwards-compatible.

## Creating a device

```c
#include <Rachis/rachis.h>

int main(void)
{
    if (rachis_abi_version() != RACHIS_ABI_VERSION)
    {
        return 1;
    }

    rachis_device_desc desc = {0};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RACHIS_ABI_VERSION;
    desc.backend = RACHIS_BACKEND_VULKAN;
    desc.adapter_preference = RACHIS_ADAPTER_HIGH_PERFORMANCE;

    rachis_error error = {0};
    error.struct_size = sizeof(error);

    rachis_device* device = NULL;
    if (rachis_device_create(&desc, &device, &error) != RACHIS_STATUS_OK)
    {
        return 1;
    }

    rachis_device_destroy(device);
    return 0;
}
```

A zero-initialized `rachis_device_desc` leaves validation off (matching C++ `DeviceDesc`). Set `validation_level` to `RACHIS_VALIDATION_STANDARD` or `RACHIS_VALIDATION_GPU` to enable it; see [Fallback and validation](platforms.md#fallback-and-validation).

For a complete end-to-end C sample (buffer upload, copy, submit, wait, readback), see [`tests/c_abi/CAbiSmoke.c`](../tests/c_abi/CAbiSmoke.c).

## Descriptors and errors

- **Struct versioning:** Every input descriptor carries `struct_size` and `abi_version`. If either mismatches, the call returns `RACHIS_STATUS_ABI_MISMATCH`. Null device or descriptor pointers return `RACHIS_STATUS_INVALID_ARGUMENT`. Output structs carry `struct_size`. Nested element structs (barriers, attachments, copy regions) are covered by their parent descriptor's version check.
- **Error output:** Fallible calls return `rachis_status` and accept an optional `rachis_error*` (set `error.struct_size = sizeof(rachis_error)`). A wrong `struct_size` fails with `RACHIS_STATUS_ABI_MISMATCH` before the operation starts and leaves `error` untouched. Messages are UTF-8 up to `RACHIS_ERROR_MESSAGE_CAPACITY - 1` bytes, truncated on a UTF-8 character boundary if longer.
- **Diagnostics:** `diagnostic_callback` receives each complete message as a NUL-terminated UTF-8 string valid only for the duration of the callback (truncated like `rachis_error` only on out-of-memory).
- **Caller-owned inputs:** Strings, shader bytecode, and input arrays remain caller-owned and need only stay valid for the duration of the call.
- **Shader bytecode rules:**
  - Bytecode must already be compiled for the selected backend.
  - **Vulkan SPIR-V** must pass `spirv-val`. The backend validates structural elements it reflects (function boundaries, entry points, declared interface, and targets of entry points/execution modes/calls/names/decorations) rather than the entire module. Compile HLSL with `dxc -fvk-support-nonzero-base-vertex -fvk-support-nonzero-base-instance` so `SV_VertexID` and `SV_InstanceID` start at `0` as on D3D12 (see [Shaders](using.md#shaders)).
  - **Metal** requires a packed Metal library (`.metallib` plus reflection) produced by `RachisShaderBinaryTool`; passing a plain `.metallib` fails with `RACHIS_STATUS_INVALID_ARGUMENT`. Since `rachis_add_shader` generates C++ headers, pure C modules invoke `RachisShaderBinaryTool` directly (see [Shaders](using.md#shaders)).
- **Buffer I/O size:** `rachis_buffer_write` and `rachis_buffer_read` copy `size` bytes literally and reject `RACHIS_WHOLE_SIZE` (which is valid only where this page says so: buffer bindings, barriers, `rachis_buffer_map`, and mapped ranges).

## Coverage

Supported C ABI surface:
- Buffers, textures, texture leases, and texture views
- Buffer mapping, with flush and invalidate for mapped ranges
- Buffer copies and bidirectional buffer-texture copies
- Global, buffer, and texture barriers
- Render passes with `load`, `clear`, or `discard` load ops and `store`
- Buffer, texture-view, and sampler bindings
- Vertex, pixel, and compute shader modules; graphics and compute pipelines; graphics and compute push constants; viewport, scissor, draw, and dispatch
- Vertex buffer layouts, vertex and index buffer binding, and indexed draws
- Device capabilities for mapping and vertex input
- Timestamp queries, valid bits, and CPU-clock calibration
- Adapter identification and runtime memory budget
- Surfaces, swapchains, presentation status, and bounded timeline waits/queries

### C++-only features and stage rules

- **Not exposed in C ABI:** Shared resources and cross-device ownership transfers (C ABI barriers always use `OwnershipTransfer::none`), variable-count binding arrays, device limits other than the vertex-input limits, indirect draws, stencil state, and optional pipeline stages (geometry, tessellation, mesh, task, ray tracing).
- **Stage masks (`visibility` and push constants):** Combine `RACHIS_SHADER_VERTEX`, `RACHIS_SHADER_PIXEL`, and `RACHIS_SHADER_COMPUTE` (or `RACHIS_SHADER_ALL` for all three). `0` or unknown bits return `RACHIS_STATUS_INVALID_ARGUMENT`.
- **Push constants:**
  - `rachis_command_set_graphics_push_constants` requires graphics stage bits only (including `RACHIS_SHADER_COMPUTE` fails with `RACHIS_STATUS_INVALID_ARGUMENT`) and a bound graphics pipeline inside a render pass.
  - `rachis_command_set_compute_push_constants` requires a bound compute pipeline (otherwise returns `RACHIS_STATUS_INVALID_STATE`).
  - Both require `offset` to be a multiple of `4`, `size` to be a nonzero multiple of `4`, and `offset + size <= 128`.

## Device capabilities

`rachis_device_query_capabilities` fills a `rachis_device_capabilities` (set `struct_size` first). The values do not change during the life of the device and stay readable after device loss.

| Field | Meaning |
| --- | --- |
| `unified_memory` | Nonzero when the CPU and GPU share memory, so a host-visible buffer costs no extra copy |
| `persistent_mapping` | Nonzero when a buffer may stay mapped while the GPU uses it (see [Mapped buffers](#mapped-buffers)) |
| `mapped_memory_host_coherent` | Nonzero when mapped memory needs no flush or invalidate |
| `mapped_memory_non_coherent_atom_size` | Granularity, in bytes, at which the device flushes and invalidates non-coherent memory; `1` when coherent |
| `vertex_formats` | Bit `1 << RACHIS_VERTEX_FORMAT_*` is set for each vertex format the device accepts |
| `max_vertex_buffers`, `max_vertex_attributes` | Vertex buffer slots and vertex attributes one pipeline may use (at most `RACHIS_MAX_VERTEX_BUFFERS` and `RACHIS_MAX_VERTEX_ATTRIBUTES`) |
| `max_vertex_buffer_stride`, `max_vertex_attribute_offset` | Largest `stride` and attribute `offset` in a vertex buffer layout |
| `max_vertex_instance_step_rate` | Largest `instance_step_rate` (`1` on Vulkan) |

## Mapped buffers

`rachis_buffer_map` returns a pointer into the memory of a `RACHIS_MEMORY_UPLOAD` or `RACHIS_MEMORY_READBACK` buffer. A `RACHIS_MEMORY_DEVICE_LOCAL` buffer returns `RACHIS_STATUS_UNSUPPORTED`.

- **Range:** `offset` and `size` select the mapped bytes; `size` may be `RACHIS_WHOLE_SIZE` for everything from `offset` to the end of the buffer. `rachis_mapped_memory.data` points at byte `offset` of the buffer and `size` is the mapped byte count (set `struct_size` first). An empty range or one outside the buffer returns `RACHIS_STATUS_INVALID_ARGUMENT`.
- **One mapping per buffer:** A buffer has at most one mapping at a time. While it is mapped, `rachis_buffer_map`, `rachis_buffer_write`, `rachis_buffer_read`, and `rachis_buffer_destroy` on it return `RACHIS_STATUS_INVALID_STATE`; call `rachis_buffer_unmap` first. Unmapping a buffer that is not mapped also returns `RACHIS_STATUS_INVALID_STATE`.
- **Pointer lifetime:** The pointer stays valid until `rachis_buffer_unmap` on that buffer or `rachis_device_destroy`. It does not depend on any device lock: any thread may read and write through it while other threads call into the device, and it may be kept across frames.
- **Mapped while the GPU uses the buffer:** When `persistent_mapping` is nonzero, a buffer may stay mapped while command buffers that use it are recorded, submitted, and executed. Direct3D 12 and Metal always report it, and so does every Vulkan driver that follows the specification. When it is zero, unmap the buffer before submitting work that uses it.
- **Synchronization is yours:** RachisRHI does not order your accesses against the GPU or against your other threads. Do not write bytes that a submitted command buffer still reads, and read bytes the GPU writes only after that submission's timeline has completed.

```c
rachis_mapped_memory mapped = {0};
mapped.struct_size = sizeof(mapped);
if (rachis_buffer_map(device, upload, 0, RACHIS_WHOLE_SIZE, &mapped, &error) == RACHIS_STATUS_OK)
{
    memcpy(mapped.data, vertices, sizeof(vertices));
    const rachis_mapped_range written = {0, sizeof(vertices)};
    rachis_buffer_flush_mapped_ranges(device, upload, &written, 1, &error);
}
```

### Flush and invalidate

Whether a write through the pointer reaches the GPU by itself depends on `mapped_memory_host_coherent`:

- **Nonzero** (always on Direct3D 12 and Metal, and on most Vulkan devices): writes are visible to the GPU and GPU writes are visible through the pointer with no further call. `rachis_buffer_flush_mapped_ranges` and `rachis_buffer_invalidate_mapped_ranges` check their arguments and do nothing else, so portable code can call them unconditionally.
- **Zero** (some Vulkan devices): after writing to an upload buffer, call `rachis_buffer_flush_mapped_ranges` on the written bytes before submitting work that reads them. Before reading a readback buffer through the pointer, call `rachis_buffer_invalidate_mapped_ranges` on the bytes the GPU wrote. `rachis_buffer_unmap` flushes the whole mapping of an upload buffer.

Each `rachis_mapped_range` is an `offset` from the start of the buffer (not of the mapping) and a `size`, which may be `RACHIS_WHOLE_SIZE`. Every range must lie inside the mapped range; it need not be a multiple of `mapped_memory_non_coherent_atom_size`, because RachisRHI widens it. Failures:

| Status | Cause |
| --- | --- |
| `RACHIS_STATUS_INVALID_ARGUMENT` | No ranges, or a range that is empty, outside the buffer, or outside the mapped range |
| `RACHIS_STATUS_UNSUPPORTED` | Flush on a buffer that is not `RACHIS_MEMORY_UPLOAD`, or invalidate on one that is not `RACHIS_MEMORY_READBACK` |
| `RACHIS_STATUS_INVALID_STATE` | The buffer is not mapped |

## Vertex buffers and indexed draws

- **Shader inputs:** A vertex shader module lists every input location it reads in `vertex_input_locations` (each one unique and below `RACHIS_MAX_VERTEX_ATTRIBUTES`). Other stages leave the list empty. [Shaders](using.md#shaders) explains how shader inputs get their locations on each backend.
- **Layouts:** `vertex_buffers` in `rachis_graphics_pipeline_desc` holds one `rachis_vertex_buffer_layout` per vertex buffer slot, in slot order. Each layout gives the `stride`, the `step_mode`, and its attributes (`format`, byte `offset`, and `shader_location`). The locations of all attributes together must be exactly the locations the vertex shader declares. Offsets and strides follow the alignment rules in [Vertex input](using.md#vertex-input).
- **Step rate:** `instance_step_rate` `0` means `1`. `RACHIS_VERTEX_STEP_VERTEX` accepts only that. `RACHIS_VERTEX_STEP_INSTANCE` advances once every `instance_step_rate` instances, up to `max_vertex_instance_step_rate`.
- **Binding:** `rachis_command_bind_vertex_buffers` binds `binding_count` buffers created with `RACHIS_BUFFER_VERTEX` to consecutive slots from `first_slot`; if any binding is rejected, none of them changes. `rachis_command_bind_index_buffer` binds a `RACHIS_BUFFER_INDEX` buffer as `RACHIS_INDEX_FORMAT_UINT16` or `RACHIS_INDEX_FORMAT_UINT32`.
- **Drawing:** `rachis_command_draw_indexed` reads `index_count` indices starting at index `first_index` of the bound index buffer and adds `base_vertex` to each.
- **Indexed strips:** For a line or triangle strip drawn with an index buffer, set `strip_index_format` to the format of that index buffer. This turns on primitive restart at index `0xFFFF` or `0xFFFFFFFF`, and Metal requires it for indexed strip draws. Leave it `RACHIS_STRIP_INDEX_NONE` for list topologies, where any other value returns `RACHIS_STATUS_INVALID_ARGUMENT`.

## Adapters

- **Selecting an adapter:** A nonzero `adapter_id` in `rachis_device_desc` overrides `adapter_preference`. With `RACHIS_ADAPTER_HIGH_PERFORMANCE` or `RACHIS_ADAPTER_MINIMUM_POWER`, the device opens that adapter if supported by the backend (including D3D12 WARP). With `RACHIS_ADAPTER_WARP` (D3D12 only), `adapter_id` must refer to WARP. An unknown `adapter_id` returns `RACHIS_STATUS_ADAPTER_NOT_FOUND`. `0` defers to `adapter_preference`.
- **Querying adapter info:** `rachis_device_adapter_info` sets `type` to one of `RACHIS_ADAPTER_TYPE_UNKNOWN`, `RACHIS_ADAPTER_TYPE_INTEGRATED`, `RACHIS_ADAPTER_TYPE_DISCRETE`, or `RACHIS_ADAPTER_TYPE_SOFTWARE`. Compare against `RACHIS_ADAPTER_TYPE_SOFTWARE` (not `RACHIS_ADAPTER_WARP`) to detect software rasterization. On D3D12, `INTEGRATED` vs `DISCRETE` comes from `D3D12_FEATURE_ARCHITECTURE1`, so adapter enumeration briefly opens a device on each hardware adapter. `name` is UTF-8, truncated at a character boundary if needed.

## Surfaces

`rachis_surface_desc` wraps a native window using `kind`, `window_handle`, and `display_handle`:

| `kind` | `window_handle` | `display_handle` | Backends |
| --- | --- | --- | --- |
| `RACHIS_SURFACE_WIN32` | Live `HWND` | Unused (`NULL`) | D3D12, Vulkan (Windows) |
| `RACHIS_SURFACE_WAYLAND` | `wl_surface*` | `wl_display*` (required) | Vulkan (Linux) |
| `RACHIS_SURFACE_XCB` | `(void*)(uintptr_t)window` (`xcb_window_t` value, nonzero) | `xcb_connection_t*` (required) | Vulkan (Linux) |
| `RACHIS_SURFACE_CA_METAL_LAYER` | `CAMetalLayer*` | Hosting `NSView*` or `NULL` | Metal |

> [!WARNING]
> For XCB, pass the `xcb_window_t` value itself cast to `void*`, **not** `&window`. D3D12 accepts only `RACHIS_SURFACE_WIN32`; Metal accepts only `RACHIS_SURFACE_CA_METAL_LAYER`; Vulkan (including MoltenVK) rejects `RACHIS_SURFACE_CA_METAL_LAYER`.

### Surface and swapchain rules

- **Ownership:** Native windows, displays, and layers remain caller-owned and must outlive the surface. Destroy all swapchains on a surface before calling `rachis_surface_destroy` (otherwise returns `RACHIS_STATUS_INVALID_STATE`).
- **One swapchain per window:** Only one live swapchain may exist per native window on a device. Creating a second on that surface—or on another surface for the same window—returns `RACHIS_STATUS_INVALID_STATE` (creating multiple `rachis_surface` handles for one window is allowed). See [Surfaces](using.md#surfaces) for cross-device caveats.
- **Metal layers:** Creating a surface leaves `CAMetalLayer` untouched (if `layer.device` is set, it must match the device's GPU). Creating a swapchain sets `device`, `pixelFormat`, `drawableSize`, `colorspace`, `opaque`, `allowsNextDrawableTimeout`, `displaySyncEnabled` (`NO` for `RACHIS_PRESENT_IMMEDIATE`), `framebufferOnly` (unless copy-source usage is requested), and `name` (if the surface has a label). With an `NSView`, hidden views or non-visible windows cause `rachis_swapchain_acquire` to return `RACHIS_STATUS_NOT_READY` and `rachis_swapchain_present` to report `RACHIS_PRESENT_STATUS_OCCLUDED` (without an `NSView`, only zero drawable size counts as hidden). View visibility is read on the main thread, so off-main-thread callers observe updates after the main run loop runs.
- **Capabilities bitmasks vs ordinals:** `rachis_surface_query_capabilities` populates the first `format_count` slots of `formats` (zeroing the rest) and returns two bitmasks: `present_modes` (`RACHIS_PRESENT_MODE_FIFO_BIT` / `RACHIS_PRESENT_MODE_IMMEDIATE_BIT`) and `usages` (`RACHIS_TEXTURE_*`). In `rachis_swapchain_desc`, `present_mode` takes an **ordinal** (`RACHIS_PRESENT_FIFO` or `RACHIS_PRESENT_IMMEDIATE`), not a bit flag—test support with `present_modes & RACHIS_PRESENT_MODE_IMMEDIATE_BIT`, never `& RACHIS_PRESENT_IMMEDIATE`.
- **D3D12 sRGB sampling:** On D3D12 devices lacking fully typed format casting, `rachis_swapchain_create` returns `RACHIS_STATUS_UNSUPPORTED` for `RACHIS_FORMAT_RGBA8_UNORM_SRGB` or `RACHIS_FORMAT_BGRA8_UNORM_SRGB` combined with `RACHIS_TEXTURE_SAMPLED` (see [Surfaces](using.md#surfaces)).

## Handles and lifetime

`rachis_device*` owns all child objects, represented as opaque `rachis_handle` values:
- Every call checks handle type, owning device, and liveness. `0` is never valid, and handle values are never reused.
- Destroy all child objects before `rachis_device_destroy` (which invalidates any remaining handles).
- Command buffers created via the C ABI **always retain** referenced objects until GPU completion (`rachis_command_pool_desc` has no unretained flag).
- `rachis_command_submit` consumes the command handle on success and on failures occurring after it takes ownership of the recording; on earlier failures, the handle stays valid to resubmit or discard.

### Timelines

`rachis_command_submit` outputs a caller-owned timeline handle:
- Pass it to `rachis_timeline_wait`, `rachis_timeline_wait_for`, `rachis_timeline_query`, and `rachis_swapchain_present` as many times as needed (none consumes it).
- `rachis_timeline_query` and `rachis_timeline_wait_for` write `RACHIS_TIMELINE_STATUS_PENDING` or `RACHIS_TIMELINE_STATUS_COMPLETE` and return `RACHIS_STATUS_OK` in both cases (`timeout_ns == 0` polls without waiting; values above `INT64_MAX` wait unboundedly).
- Release it with `rachis_timeline_release` once no further wait or present needs it. Releasing only frees the handle—it neither waits nor cancels GPU work, so it is safe immediately after `submit` (if you don't wait) or right after the last wait/present returns.
- Failing to release timelines leaks slots until `rachis_command_submit` returns `RACHIS_STATUS_RESOURCE_EXHAUSTED`.

```c
rachis_handle timeline = 0;
uint32_t present_status = RACHIS_PRESENT_STATUS_PRESENTED;
if (rachis_command_submit(device, command, &timeline, &error) == RACHIS_STATUS_OK)
{
    rachis_swapchain_present(device, swapchain, timeline, &present_status, &error);
    rachis_timeline_release(device, timeline, &error);
}
```

### Texture leases

- `rachis_texture_create` returns the owner handle. Call `rachis_texture_lease_acquire` to obtain another handle to the same texture for another module on the same device.
- A texture stays alive as long as its owner handle, any lease, or any view remains live. `rachis_texture_destroy` releases only the owner handle; release leases after GPU work using them has completed.
- Once the owner handle is destroyed, the call releasing the final lease (`rachis_texture_lease_release`) or view (`rachis_texture_view_destroy`) destroys the underlying texture. If destruction fails, the call returns an error and preserves the handle for retry (a retained failed view handle holds no view and accepts only `rachis_texture_view_destroy`).

### Swapchain images

- Image and view handles from `rachis_swapchain_acquire` are **borrowed**: they cannot be leased or used to create views, and `present`, `cancel`, `resize`, or `destroy` invalidates them.
- On Metal, submitting a command buffer referencing an invalidated swapchain image returns `RACHIS_STATUS_INVALID_STATE`.
- On Direct3D 12 and Vulkan, `rachis_swapchain_cancel` returns `RACHIS_STATUS_INVALID_STATE` while a command buffer referencing the acquired image is recording or submitted-but-unpresented (see [Surfaces](using.md#surfaces)).
- Frame flow: transition from `rachis_swapchain_acquire`'s reported `initial` state -> write -> transition to required presentation state -> submit -> pass the resulting timeline handle to `rachis_swapchain_present` -> release the timeline handle. On Vulkan, the timeline must come from a graphics-queue submission made after `rachis_swapchain_acquire` (otherwise `rachis_swapchain_present` returns `RACHIS_STATUS_INVALID_ARGUMENT` and leaves the image acquired).

## Threads

Calls on a single `rachis_device*` are thread-safe, provided nothing else calls into the device during `rachis_device_destroy`:

- **Synchronization groups:**
  - *Exclusive operations* serialize with each other and wait for active concurrent operations: object create/destroy, lease acquire/release, timeline release, command acquire/discard/submit, swapchain acquire/cancel/resize, and pre/post-present bookkeeping.
  - *Concurrent operations* run concurrently with each other (subject to backend locks below): command recording on distinct command handles, `rachis_buffer_write`, `rachis_buffer_read`, `rachis_buffer_map`, `rachis_buffer_unmap`, flushing and invalidating mapped ranges, and queries.
  - *No call at all:* reading and writing through a pointer from `rachis_buffer_map` takes no lock and never waits for another thread's call (see [Mapped buffers](#mapped-buffers)).
  - *Fairness:* Waiters proceed in arrival order, and adjacent concurrent calls execute together—continuous recording/copying cannot starve exclusive calls, nor vice versa.
- **Per-command-handle restriction:** Only one thread may record into a given command handle at a time; concurrent recording on the same handle returns `RACHIS_STATUS_INVALID_STATE`.
- **Non-blocking waits and swapchain serialization:** `rachis_timeline_wait` and `rachis_timeline_wait_for` do not block other calls. Presents serialize per swapchain (and on Vulkan block work targeting the presenting queue during the driver present call). On D3D12 and Vulkan, present may wait for the display; on Metal it does not. `rachis_swapchain_acquire`, `resize`, and `destroy` can block on the GPU or window system while holding device locks.
- **Vulkan & Metal copy/recording locks:** On Vulkan and Metal, `rachis_buffer_write` and `rachis_buffer_read` hold an internal device-wide lock for the entire copy, blocking other copies (even on different buffers), `rachis_buffer_map`, `rachis_buffer_unmap`, and flushing or invalidating mapped ranges, plus:
  - *Vulkan:* `rachis_command_copy_buffer`, buffer-texture copies, `rachis_command_barriers`, `rachis_command_begin_rendering`, and `rachis_texture_query_transfer_layout` (copies and barriers across distinct command handles also serialize).
  - *Metal:* all command recording calls.
  - *Direct3D 12* locks per buffer and has none of these stalls. To keep other threads responsive on Vulkan/Metal, split large uploads or readbacks into chunks of a few megabytes, or write and read through a mapped pointer, which takes no lock.
- **Diagnostic callbacks:** May execute on backend threads and must never re-enter the `rachis_device*`.

## Device loss

- Query `rachis_device_operational_state` to check whether a device is active or lost, and `rachis_device_loss_error` to read the stored cause (check `error.code`, since the function's return status only indicates whether querying succeeded). The first loss is also sent to `diagnostic_callback`.
- Once lost, all create, record, submit, acquire, and present calls—plus `rachis_buffer_write`, `rachis_buffer_read`, `rachis_buffer_map`, `rachis_buffer_flush_mapped_ranges`, and `rachis_buffer_invalidate_mapped_ranges` (even with stale handles or out-of-bounds ranges)—return `RACHIS_STATUS_DEVICE_LOST`.
- A pointer mapped before the loss stays valid memory until `rachis_buffer_unmap`: reading and writing through it does not fault, but the bytes read are undefined and writes reach no GPU. `rachis_buffer_unmap` still succeeds, and a mapped buffer still needs it before `rachis_buffer_destroy`.
- Diagnostics, `destroy`, and `release` calls continue working (including `rachis_swapchain_destroy` while an image is still acquired). Create a new device to recover.

## Timestamps and memory budget

- **Valid bits:** `rachis_queue_timestamp_valid_bits` returns the number of low valid bits per queue (`64` on Direct3D 12 and Metal, `timestampValidBits` on Vulkan; take timestamp deltas modulo $2^{\text{bits}}$), or `RACHIS_STATUS_UNSUPPORTED` when the queue has no timestamps.
- **Clock calibration:** `rachis_queue_calibrate_timestamps` fills `rachis_timestamp_calibration` (set `struct_size` first) with paired GPU and host monotonic counters (`RACHIS_CPU_CLOCK_QUERY_PERFORMANCE_COUNTER` on Windows, `RACHIS_CPU_CLOCK_MONOTONIC` in ns on Linux, `RACHIS_CPU_CLOCK_MACH_ABSOLUTE_TIME` in ticks on macOS Metal), frequencies, and `max_deviation_ns` (see [Queries](using.md#queries) for conversion). Returns `RACHIS_STATUS_UNSUPPORTED` when unavailable (including Vulkan on macOS).
- **Memory budget:** `rachis_device_memory_budget` fills `rachis_memory_budget` (set `struct_size` first) with process-wide local and non-local adapter budget and usage (`QueryVideoMemoryInfo` on D3D12, `VK_EXT_memory_budget` on Vulkan, `MTLDevice` local-only on Metal; usage may exceed budget). Returns `RACHIS_STATUS_UNSUPPORTED` on Vulkan without `VK_EXT_memory_budget`.
- **Errors:** On failure (after checking `struct_size`), output structs are zeroed except `struct_size`. Both calls return `RACHIS_STATUS_DEVICE_LOST` on a lost device.
