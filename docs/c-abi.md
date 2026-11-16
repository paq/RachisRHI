# C ABI

`Rachis/rachis.h` exposes a **C11 interface** to RachisRHI so independently compiled native modules can share a single GPU device across binary boundaries. It provides a focused subset of the C++ API (see [Coverage](#coverage)).

The C ABI lives in the `rachis` shared library (`Rachis::C` in CMake) and exports only `rachis_*` symbols with C linkage.

## Contents

- [Version](#version)
- [Creating a device](#creating-a-device)
- [Descriptors and errors](#descriptors-and-errors)
- [Coverage](#coverage)
- [Adapters](#adapters)
- [Surfaces](#surfaces)
- [Handles and lifetime](#handles-and-lifetime)
  - [Timelines](#timelines)
  - [Texture leases](#texture-leases)
  - [Swapchain images](#swapchain-images)
- [Threads](#threads)
- [Device loss](#device-loss)

## Version

Call `rachis_abi_version()` before creating a device and abort unless it equals the `RACHIS_ABI_VERSION` your module compiled against (currently `1`). ABI versions are not backwards-compatible.

Releases that share a `RACHIS_ABI_VERSION` ship the same `rachis.h` apart from whitespace, so a module built against one of them can load the `rachis` library from any other. Any other change to the header raises the version. CI compares `rachis.h` with the newest release tag using [`cmake/CheckCAbiFreeze.cmake`](../cmake/CheckCAbiFreeze.cmake); run `cmake -P cmake/CheckCAbiFreeze.cmake` from the source root to check locally.

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
- **Buffer I/O size:** `rachis_buffer_write` and `rachis_buffer_read` copy `size` bytes literally and reject `RACHIS_WHOLE_SIZE` (which is valid only in descriptors that explicitly document it, such as buffer bindings and barriers).

## Coverage

Supported C ABI surface:
- Buffers, textures, texture leases, and texture views
- Buffer copies and bidirectional buffer-texture copies
- Global, buffer, and texture barriers
- Render passes with `load`, `clear`, or `discard` load ops and `store`
- Buffer, texture-view, and sampler bindings
- Vertex, pixel, and compute shader modules; graphics and compute pipelines; graphics and compute push constants; viewport, scissor, draw, and dispatch
- Timestamp queries
- Adapter identification
- Surfaces, swapchains, and presentation status

### C++-only features and stage rules

- **Not exposed in C ABI:** Shared resources and cross-device ownership transfers (C ABI barriers always use `OwnershipTransfer::none`), variable-count binding arrays, device limits, vertex buffer layouts, indexed draws, stencil state, and optional pipeline stages (geometry, tessellation, mesh, task, ray tracing).
- **Stage masks (`visibility` and push constants):** Combine `RACHIS_SHADER_VERTEX`, `RACHIS_SHADER_PIXEL`, and `RACHIS_SHADER_COMPUTE` (or `RACHIS_SHADER_ALL` for all three). `0` or unknown bits return `RACHIS_STATUS_INVALID_ARGUMENT`.
- **Push constants:**
  - `rachis_command_set_graphics_push_constants` requires graphics stage bits only (including `RACHIS_SHADER_COMPUTE` fails with `RACHIS_STATUS_INVALID_ARGUMENT`) and a bound graphics pipeline inside a render pass.
  - `rachis_command_set_compute_push_constants` requires a bound compute pipeline (otherwise returns `RACHIS_STATUS_INVALID_STATE`).
  - Both require `offset` to be a multiple of `4`, `size` to be a nonzero multiple of `4`, and `offset + size <= 128`.

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
- **Metal layers:** Creating a surface leaves `CAMetalLayer` untouched (if `layer.device` is set, it must match the device's GPU). Creating a swapchain sets `device`, `pixelFormat`, `drawableSize`, `colorspace`, `opaque`, `allowsNextDrawableTimeout`, `framebufferOnly` (unless copy-source usage is requested), and `name` (if the surface has a label). With an `NSView`, hidden views or non-visible windows cause `rachis_swapchain_acquire` to return `RACHIS_STATUS_NOT_READY` and `rachis_swapchain_present` to report `RACHIS_PRESENT_STATUS_OCCLUDED` (without an `NSView`, only zero drawable size counts as hidden). View visibility is read on the main thread, so off-main-thread callers observe updates after the main run loop runs.
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
- Pass it to `rachis_timeline_wait` and `rachis_swapchain_present` as many times as needed (neither consumes it).
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
  - *Concurrent operations* run concurrently with each other (subject to backend locks below): command recording on distinct command handles, `rachis_buffer_write`, `rachis_buffer_read`, and queries.
  - *Fairness:* Waiters proceed in arrival order, and adjacent concurrent calls execute together—continuous recording/copying cannot starve exclusive calls, nor vice versa.
- **Per-command-handle restriction:** Only one thread may record into a given command handle at a time; concurrent recording on the same handle returns `RACHIS_STATUS_INVALID_STATE`.
- **Non-blocking waits and swapchain serialization:** `rachis_timeline_wait` does not block other calls. Presents serialize per swapchain (and on Vulkan block work targeting the presenting queue during the driver present call). On D3D12 and Vulkan, present may wait for the display; on Metal it does not. `rachis_swapchain_acquire`, `resize`, and `destroy` can block on the GPU or window system while holding device locks.
- **Vulkan & Metal copy/recording locks:** On Vulkan and Metal, `rachis_buffer_write` and `rachis_buffer_read` hold an internal device-wide lock for the entire copy, blocking other copies (even on different buffers) plus:
  - *Vulkan:* `rachis_command_copy_buffer`, buffer-texture copies, `rachis_command_barriers`, `rachis_command_begin_rendering`, and `rachis_texture_query_transfer_layout` (copies and barriers across distinct command handles also serialize).
  - *Metal:* all command recording calls.
  - *Direct3D 12* locks per buffer and has none of these stalls. To keep other threads responsive on Vulkan/Metal, split large uploads or readbacks into chunks of a few megabytes.
- **Diagnostic callbacks:** May execute on backend threads and must never re-enter the `rachis_device*`.

## Device loss

- Query `rachis_device_operational_state` to check whether a device is active or lost, and `rachis_device_loss_error` to read the stored cause (check `error.code`, since the function's return status only indicates whether querying succeeded). The first loss is also sent to `diagnostic_callback`.
- Once lost, all create, record, submit, acquire, and present calls—plus `rachis_buffer_write` and `rachis_buffer_read` (even with stale handles or out-of-bounds ranges)—return `RACHIS_STATUS_DEVICE_LOST`.
- Diagnostics, `destroy`, and `release` calls continue working (including `rachis_swapchain_destroy` while an image is still acquired). Create a new device to recover.
