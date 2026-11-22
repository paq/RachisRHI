# Using RachisRHI

## Contents

- **Setup & basics:** [CMake targets](#cmake-targets) · [Windows runtime DLLs](#windows-runtime-dlls) · [Headers](#headers) · [Creating a device](#creating-a-device) · [Errors](#errors)
- **Shaders & pipelines:** [Shaders](#shaders) ([Optional shader stages](#optional-shader-stages), [Stage inputs and outputs](#stage-inputs-and-outputs)) · [Vertex input](#vertex-input) · [Graphics pipeline limits on Metal](#graphics-pipeline-limits-on-metal) · [Point lists](#point-lists) · [Multiview rendering](#multiview-rendering)
- **Presentation:** [Surfaces](#surfaces)
- **Resources & lifetimes:** [Handles and lifetime](#handles-and-lifetime) · [Texture layouts](#texture-layouts) · [Readback buffers](#readback-buffers) · [Acceleration structures](#acceleration-structures) · [Sharing resources](#sharing-resources)
- **Bindings & barriers:** [Samplers](#samplers) · [Binding pools](#binding-pools) · [Variable-count binding arrays](#variable-count-binding-arrays) · [Direct3D 12 without enhanced barriers](#direct3d-12-without-enhanced-barriers)
- **Execution & operations:** [Threads](#threads) · [Queries](#queries) · [Device loss](#device-loss) · [Diagnostics](#diagnostics) · [Examples](#examples)

---

## CMake targets

An installed Rachis package (`find_package(Rachis 0.4 CONFIG REQUIRED)`) exports the following targets:

| Target | Kind | Purpose |
| --- | --- | --- |
| `Rachis::RHI` | Static library | C++23 core API (`rachis::`) and headers |
| `Rachis::C` | Shared library | `rachis` C11 ABI library ([C ABI guide](c-abi.md)) |
| `Rachis::RHID3D12` | Backend library | Direct3D 12 backend (present when built) |
| `Rachis::RHIVulkan` | Backend library | Vulkan 1.3 backend (present when built) |
| `Rachis::RHIMetal` | Backend library | Metal backend (present when built) |
| `Rachis::ShaderBinaryTool` | Host executable | Shader cooking tool invoked by `rachis_add_shader` |

Minimal C++ consumer configuration:

```cmake
find_package(Rachis 0.4 CONFIG REQUIRED)

add_executable(app main.cpp)
target_compile_features(app PRIVATE cxx_std_23)
target_link_libraries(app PRIVATE Rachis::RHI)
```

Point CMake at the installed prefix during configuration:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=<prefix>
```

CMake searches `<prefix>/<libdir>/cmake/Rachis` (or an explicit `Rachis_DIR` directory). `<libdir>` is `lib` on Windows, macOS, Debian, and Arch; `lib64` on Fedora, openSUSE, RHEL, and NixOS; or `lib/<multiarch>` under `/usr` on Debian.

### Package discovery, components, and private dependencies

- **Cross-compiled shader tool:** Packages built with `RACHIS_SHADER_BINARY_TOOL` omit `Rachis::ShaderBinaryTool`. Consumers calling `rachis_add_shader` against such a package must set `RACHIS_SHADER_BINARY_TOOL` to a runnable host build of the tool.
- **Bundled internal libraries:** The installed package bundles private copies of `volk` and `D3D12MemoryAllocator` under `<libdir>/Rachis` with no headers or CMake package configs exported, preventing other copies on `CMAKE_PREFIX_PATH` from replacing them (`find_package(Rachis)` only looks up `Threads` for Vulkan). If your own targets also link static `volk` or `D3D12MemoryAllocator` (or define `VMA_IMPLEMENTATION`), embed RachisRHI from source instead of consuming the installed package to avoid duplicate symbols; see [Using the source tree directly](building.md#using-the-source-tree-directly).
- **MSVC multi-configuration linking:** `Rachis::RHI` must link with a matching Debug or Release CRT configuration (mismatching triggers `LNK2038`). A combined prefix built per [Debug and Release in one prefix](building.md#debug-and-release-in-one-prefix) satisfies both (`RelWithDebInfo` and `MinSizeRel` resolve to the Release libraries). `Rachis::C` (`rachis.dll`) is configuration-independent.
- **Backend detection and components:** Enabled backends are exposed as boolean target properties on `Rachis::RHI`: `RACHIS_ENABLE_D3D12`, `RACHIS_ENABLE_VULKAN`, and `RACHIS_ENABLE_METAL`. Package components drop the `Rachis::` prefix and report `Rachis_<component>_FOUND`:
  - `find_package(Rachis 0.4 CONFIG REQUIRED COMPONENTS RHI RHIVulkan)` fails configuration if the Vulkan backend is absent.
  - Passing components under `OPTIONAL_COMPONENTS` leaves `Rachis_<component>_FOUND` for conditional checks without failing configure.
- **C ABI consumers:** Link `Rachis::C` instead of `Rachis::RHI`; no C++ toolchain or standard library linkage is required by the caller. On Windows, deploy `rachis.dll` alongside the executable. See [`tests/package/consumer`](../tests/package/consumer) for a complete package consumer example.

### Windows runtime DLLs

> [!IMPORTANT]
> When RachisRHI is built with Direct3D 12 and `RACHIS_ENABLE_DEBUG_MARKERS` enabled (the default), every executable linking `Rachis::RHI` or `Rachis::C` loads `WinPixEventRuntime.dll` at process startup—even if debug markers are never called or the application selects Vulkan at runtime. Executables linking `Rachis::C` also load `rachis.dll`. Both DLLs live in `<prefix>/bin`; missing either fails launch with `STATUS_DLL_NOT_FOUND` (`0xC0000135`). Building with `-DRACHIS_ENABLE_DEBUG_MARKERS=OFF` removes the `WinPixEventRuntime.dll` dependency.

For a single executable, copy runtime DLLs post-build and install them alongside the binary:

```cmake
if(WIN32)
    add_custom_command(TARGET app POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy -t "$<TARGET_FILE_DIR:app>" $<TARGET_RUNTIME_DLLS:app>
        COMMAND_EXPAND_LISTS)

    install(FILES $<TARGET_RUNTIME_DLLS:app> TYPE BIN)
endif()
```

Use `copy -t <dir>` rather than `copy_if_different $<TARGET_RUNTIME_DLLS:app> <dir>`: `copy -t` succeeds even when `$<TARGET_RUNTIME_DLLS:app>` expands to empty (for instance, when linking `Rachis::RHI` without D3D12 or debug markers), whereas `copy_if_different` errors on an empty file list.

When multiple executables share one output directory in a single build, avoid parallel post-build copy races by dedicating one custom target with `copy_if_different` (which skips rewriting loaded DLLs already on disk) ordered before the executables link. [`tests/package/consumer/CMakeLists.txt`](../tests/package/consumer/CMakeLists.txt) illustrates this pattern: its list is guaranteed non-empty because it links `Rachis::C` (`rachis.dll`), and it declares an explicit dependency on `Rachis::C` so `$<TARGET_RUNTIME_DLLS>` waits for `rachis.dll` when built via `add_subdirectory`.

---

## Headers

All public headers live under `Rachis/`, and C++ symbols live in namespace `rachis::`:

| Area | Headers |
| --- | --- |
| Core device, errors, handles | `Device.h`, `Error.h`, `Handle.h`, `Diagnostics.h` |
| Resources, samplers, bindings | `Resource.h`, `Sampler.h`, `Binding.h` |
| Commands, rendering, pipelines | `Command.h`, `Rendering.h`, `Pipeline.h` |
| Presentation, queries, ray tracing, interop | `Presentation.h`, `Query.h`, `AccelerationStructure.h`, `Interop.h` |
| C11 ABI | `rachis.h` |

---

## Creating a device

```cpp
#include <Rachis/Device.h>

#include <cstdio>
#include <utility>

int main()
{
    rachis::DeviceDesc desc{};
#ifndef NDEBUG
    desc.validation = rachis::ValidationLevel::standard;
#endif

    auto deviceResult = rachis::createDevice(desc);
    if (!deviceResult)
    {
        std::fprintf(stderr, "createDevice failed: %s\n", deviceResult.error().message.c_str());
        return 1;
    }

    rachis::Device device = std::move(*deviceResult);
    (void)device;
    return 0;
}
```

Key `DeviceDesc` selection and validation rules:

- **Default backend:** `defaultHostBackend()` selects `Backend::d3d12` on Windows, `Backend::metal` on macOS, and `Backend::vulkan` elsewhere. Override `desc.backend` to target another built-in backend. If the requested backend was not compiled in or lacks host support, `createDevice` returns `ErrorCode::backendUnavailable`.
- **Adapter selection:** `adapterId` takes precedence over `adapterPreference` (`highPerformance`, `minimumPower`, `warp`). When `adapterPreference` is `highPerformance` or `minimumPower`, specifying `adapterId` opens that exact adapter—including software WARP, which `enumerateAdapters(Backend::d3d12)` reports with `AdapterType::software`. Setting `adapterPreference = AdapterPreference::warp` is Direct3D 12-only and requires any provided `adapterId` to refer to WARP. Unknown IDs fail with `ErrorCode::adapterNotFound`. Without an `adapterId`, `highPerformance` and `minimumPower` pick only hardware adapters: on Direct3D 12 they skip adapters that DXGI flags as software and the Microsoft Basic Render Driver (vendor `0x1414`, device `0x8c`), which Windows exposes on machines without a GPU driver, such as virtual machines, even when DXGI does not flag it as software. Use `AdapterPreference::warp` for software rendering there.
- **Validation levels:** Validation defaults to `ValidationLevel::disabled`:
  - `standard`: enables the Direct3D 12 debug layer, Vulkan validation layer (`VK_LAYER_KHRONOS_validation`), or Metal `MTLLogState` shader logging.
  - `gpu`: adds Direct3D 12 GPU-based validation, Vulkan GPU-assisted validation, and Metal debug log level. Metal API and shader validation (`MTL_DEBUG_LAYER`, `MTL_SHADER_VALIDATION`) remain external runtime environment variables.
  - Missing validation layers/tools are silently skipped unless `requireValidation = true`.

---

## Errors

RachisRHI does not throw exceptions across its public API. Operations return either `rachis::Result<T>` (`std::expected<T, rachis::Error>`) or `rachis::Status` (`Result<void>`).

Each `rachis::Error` carries:
- `code`: portable `rachis::ErrorCode`
- `nativeCode`: backend-specific HRESULT, `VkResult`, or OS/driver status (`std::int64_t`)
- `message`: diagnostic `std::string` describing the failure

---

## Shaders

RachisRHI consumes precompiled shader bytecode—DXIL for Direct3D 12, SPIR-V for Vulkan, and packed Metal libraries (`.pnms`) for Metal—paired with an explicit `ShaderModuleDesc` or `ShaderLibraryDesc` (`Rachis/Pipeline.h`). The RHI does not compile HLSL/Slang at runtime and enforces these interface rules at module/library creation:

- **SPIR-V structural checks:** Submitted SPIR-V must be valid (`spirv-val`). Before accepting a module or library, Vulkan inspects function boundaries, entry points, declared shader interfaces, and defined targets of entry points, execution modes, calls, names, and decorations—rejecting truncated modules or missing referenced definitions without re-implementing a full validator.
- **Vertex attribute locations:** `vertexInputLocations` must list every input location consumed by a vertex shader (`float4x4 world : TEXCOORD1` spans locations `1..4`; `float4 weights[2] : TEXCOORD1` spans `1..2`), and every location must be `< rachis::maxVertexAttributes` (`16`). Vulkan assigns vertex attribute locations strictly according to SPIR-V rules and rejects inputs exceeding the limit or struct-typed vertex inputs.
- **Specialization constants:** Only 32-bit scalar `SpecializationConstant` entries (`boolean`, `signedInt32`, `unsignedInt32`, `float32`) are supported. Vulkan rejects 8-bit, 16-bit, or 64-bit specialization entries with `ErrorCode::invalidArgument`; omit an entry to keep its shader default.
- **Packed Metal libraries:** `.pnms` blobs combine `.metallib` bytecode with Metal reflection metadata and are produced exclusively by `RachisShaderBinaryTool`.

### Cooking and reflecting shaders with CMake

`rachis_add_shader` compiles one HLSL, Slang, or GLSL source file at build time for every enabled backend and generates a header under `${CMAKE_CURRENT_BINARY_DIR}` containing embedded bytecode and optional reflection metadata. The target comes first:

```cmake
rachis_add_shader(app
    NAME BlurShader
    SOURCE shaders/Blur.hlsl
    ENTRY blurMain
    STAGE compute
    REFLECT
)
```

Include `"BlurShader.h"` and pass the generated descriptors directly into RachisRHI:

```cpp
#include "BlurShader.h"

namespace blur = rachis_shaders::reflection::BlurShader;

auto shader = device.createShaderModule(blur::moduleDesc(backend, "Blur"));
auto bindingLayout = device.createBindingLayout({
    .entries = blur::bindingLayout(backend, 0),
    .label = "Blur bindings",
});
auto bindingPool = device.createBindingPool({
    .maxBindingSets = 1,
    .sizes = blur::bindingPoolSizes(backend, 0),
});
```

Without `REFLECT`, only raw bytecode arrays are generated (`rachis_shaders::BlurShader(backend)`), and `slangc` is required only for Slang shaders or when building with Metal enabled. Adding `REFLECT` to an HLSL or Slang shader requires `slangc` on **every** backend; a GLSL shader's interface comes from its SPIR-V. `REFLECT` generates `rachis_shaders::reflection::<NAME>` (plus backend sub-namespaces such as `rachis_shaders::reflection::BlurShader::vulkan`) containing:

| Generated member | Description |
| --- | --- |
| `moduleDesc(backend, label)` | Complete `ShaderModuleDesc` pairing bytecode with reflected interface |
| `shaderInterface(backend)` | Reflected `ShaderInterface` (entry point, workgroup size, stage I/O) |
| `bindingLayout(backend, set)` | `BindingLayoutEntry` span for set `set` |
| `bindingPoolSizes(backend, set)` | Required pool counts for set `set` (excludes unbounded binding arrays) |
| `pushConstantRanges(backend)` | Push constant ranges declared by the shader |
| `bindingSetCount(backend)` | Number of descriptor sets referenced |
| `stage`, `entryPoint` | Compile-time stage enum and entry-point name |

`createShaderModule` still cross-checks the supplied `ShaderModuleDesc` against the bytecode and rejects mismatches rather than silently deriving an unverified layout.

Reflected C++ structs are also emitted for every constant buffer, push-constant block, and structured-buffer element type with exact field offsets, padding, and `static_assert` size/offset checks:
- Vectors map to `std::array`, matrices to arrays of columns, and strided arrays to `rachis_shaders::layout::StridedArray` (`sizeof(Sample)` matches element stride).
- D3D12 and Metal buffer layouts are extracted via `slangc`. Vulkan layouts come from the SPIR-V the Vulkan compiler emits: `dxc`'s relaxed `std140`/`std430` for HLSL, Slang's `std140`/`std430` for Slang, and the declared block layouts for GLSL. The same struct can therefore differ between backends and between languages.
- Limitations: buffer arrays require an explicit element count, arrays of matrices are unsupported, reflected buffer matrices must use `column_major` order, and Vulkan does not accept reflected column-major matrices directly inside buffer blocks.
- Multiple targets in one executable can safely define shaders with the same `NAME`: `rachis_add_shader` scopes definitions into project/target-unique inline namespaces inside `rachis_shaders` and `rachis_shaders::reflection::<NAME>`, while duplicate `NAME`s within the same target fail CMake configure.

### Shader languages

`LANGUAGE HLSL`, `LANGUAGE SLANG`, or `LANGUAGE GLSL` picks the compilers. Without it, the extension of `SOURCE` decides: `.hlsl` is HLSL, `.slang` is Slang, and `.glsl`, `.vert`, `.frag`, and `.comp` are GLSL. Any other extension fails CMake configure until `LANGUAGE` names the language.

| Language | Direct3D 12 | Vulkan | Metal | `REFLECT` reads the interface from |
| --- | --- | --- | --- | --- |
| HLSL | `dxc` DXIL | Vulkan SDK `dxc` SPIR-V | `slangc` + `RachisShaderBinaryTool` | `slangc`, once per backend |
| Slang | `slangc` DXIL | `slangc` SPIR-V | `slangc` + `RachisShaderBinaryTool` | `slangc`, once per backend |
| GLSL | not compiled | `glslangValidator` SPIR-V | not compiled | the SPIR-V `glslangValidator` emits |

- **Slang** follows the HLSL rules below: `register(<class>N, spaceM)` is binding `N` in set `M`, and a push-constant block is `[[vk::push_constant]] ConstantBuffer<T> name : register(b0, space4);`, which Slang accepts on every target without an `#if`. `slangc` reads the dialect from the file extension, so `LANGUAGE SLANG` on a `.hlsl` file compiles that HLSL with `slangc` on every backend. Vulkan builds define `__spirv__=1`, as `dxc` does for HLSL.
- **GLSL** is Vulkan-only. On Direct3D 12 and Metal its bytecode and its reflection are empty, and configuring a GLSL shader against a package without Vulkan fails. The entry function is always `main`; `ENTRY`, which defaults to `main`, names the SPIR-V entry point. `layout(set = M, binding = N)` is binding `N` in set `M`, `layout(push_constant)` declares the push constants, and `layout(location = N) in` is vertex input location `N`. A `readonly buffer` is a `readOnlyStorageBuffer`, and a block holding one runtime array of structs gets a C++ struct for its element, like an HLSL `StructuredBuffer`. `REFLECT` rejects combined image samplers (`sampler2D`); declare a `texture2D` and a `sampler` and combine them with `sampler2D(texture, sampler)` in the shader.
- `gl_VertexIndex` and `gl_InstanceIndex` keep their Vulkan meaning and include `firstVertex`, `baseVertex`, and `firstInstance`, unlike `SV_VertexID` and `SV_InstanceID` in HLSL and Slang shaders (see below).

### Compiler targets, register mapping, and matrix packing

For HLSL:

| Backend | Compiler / tool | Output format | Preprocessor define |
| --- | --- | --- | --- |
| Direct3D 12 | `dxc` (Shader Model 6.0) | DXIL | `RACHIS_D3D12=1` |
| Vulkan | Vulkan SDK `dxc` + `spirv-val` (Vulkan 1.3) | SPIR-V | `RACHIS_VULKAN=1` |
| Metal | `slangc` + `RachisShaderBinaryTool` | `.pnms` packed metallib | `RACHIS_METAL=1` |

Slang and GLSL shaders get the same defines, and their Vulkan SPIR-V is checked with `spirv-val` for Vulkan 1.3 too.

Cooking conventions and cross-backend rules:
- `#include` dependencies are tracked automatically; extra macros can be passed via `DEFINES NAME=VALUE...`. GLSL includes need `#extension GL_GOOGLE_include_directive : require`.
- `STAGE` passes the profile directly to `dxc`, `-stage` to `slangc`, and `-S` to `glslangValidator`, so shaders do not need `[shader(...)]` attributes.
- **Registers & binding slots:** `register(<class>N, spaceM)` maps to binding `N` in set `M`. Push constants use `register(b0, space4)` on Direct3D 12 and `[[vk::push_constant]]` on Vulkan and Metal. D3D12 vertex inputs at location `N` use semantic `TEXCOORDn`.
- **Storage buffer writability:** `t` (`StructuredBuffer`, `ByteAddressBuffer`) maps to `BindingType::readOnlyStorageBuffer`, while `u` (`RWStructuredBuffer`) maps to `BindingType::storageBuffer`. On Vulkan, `createShaderModule` checks `NonWritable` on the SPIR-V variable (Slang) or on every block member (`dxc`).
- **Metal single resource namespace per kind:** An array binding at `N` of count `C` occupies slots `N .. N+C-1`. Direct3D 12 and Vulkan separate namespaces for `b`, `t`, `u`, and `s`, whereas Metal allocates **one shared slot namespace across all buffers** (`b`, `t`, and `u` buffer bindings), one for textures, and one for samplers. For example, placing `ConstantBuffer<Source> sources[2] : register(b0, space0)` alongside an `RWStructuredBuffer` at `register(u1, space0)` compiles on Direct3D 12 and Vulkan, but fails `createBindingLayout` on Metal because `sources[1]` occupies buffer slot `1`—move the `RWStructuredBuffer` to `register(u2, space0)`.
- **Matrix packing:** RachisRHI assumes HLSL's default `column_major` order (do not pass `-Zpr` or `-Zpc`). In constant buffers and push constants, `float3x4` (4 columns of 3 floats) pads every column to 16 bytes (`64` bytes total) on all backends. In structured buffers, `float3x4` is tightly packed (`48` bytes) on Direct3D 12 and Metal, but padded to 16-byte columns (`64` bytes) on Vulkan; `float4x4` and `float4x3` use 16-byte columns on all backends. Explicit `row_major` packs rows contiguously (`dxc` writes SPIR-V `RowMajor` for `column_major` and `ColMajor` for `row_major`, and Vulkan accepts both in push-constant blocks).

### Base vertex, base instance, and cross-compiling

- **Zero-based `SV_VertexID` / `SV_InstanceID`:** In HLSL and Slang shaders, both semantics are zero-based relative to the draw call (`firstVertex`/`baseVertex` and `firstInstance` are excluded, including in indirect draws).
  - Vulkan compiles HLSL with `-fvk-support-nonzero-base-vertex` and `-fvk-support-nonzero-base-instance`, and `slangc` subtracts the same bases from Slang's, emitting SPIR-V `DrawParameters` (requires `shaderDrawParameters`, otherwise `createShaderModule` returns `ErrorCode::unsupported`).
  - Metal: `slangc` subtracts `[[base_instance]]` automatically. When a vertex shader reads `SV_VertexID`, `RachisShaderBinaryTool cook-metallib` emits `.metal` source, subtracts `[[base_vertex]]` from `[[vertex_id]]` (reusing the `SV_StartVertexLocation` parameter's `[[base_vertex]]`), and compiles via `xcrun --sdk macosx metal`. Enable `RACHIS_METAL_EMIT_DEBUG_SOURCE` to inspect the generated `.metallib.metal` beside `.metallib`.
- **Cross-compiling:** When `CMAKE_CROSSCOMPILING` is set, provide `RACHIS_SHADER_BINARY_TOOL` pointing to a host build of `RachisShaderBinaryTool`.

### Cooking manually for C or custom build systems

C ABI callers and custom build pipelines can cook shaders outside `rachis_add_shader` (preserving `<NAME>.<backend>.reflection.json` beside the bytecode when reflection is needed):

- **Vulkan (`dxc`):** Pass `-spirv -fspv-target-env=vulkan1.3 -fvk-support-nonzero-base-vertex -fvk-support-nonzero-base-instance` and validate with `spirv-val`.
- **Slang (`slangc`):** Pass `-profile sm_6_0 -matrix-layout-column-major` with `-target dxil`, or with `-target spirv -fvk-use-entrypoint-name -fvk-b-shift 0 all -fvk-t-shift 0 all -fvk-s-shift 0 all -fvk-u-shift 0 all`, and validate SPIR-V with `spirv-val`.
- **GLSL (`glslangValidator`):** Pass `-V --target-env vulkan1.3 -S <vert|frag|comp>` and validate with `spirv-val`. `RachisShaderBinaryTool reflect-interface --spirv-reflection vulkan=<shader.spv>` generates the interface header from the module.
- **Metal (`RachisShaderBinaryTool cook-metallib`):**

```bash
RachisShaderBinaryTool cook-metallib \
    --slangc <path/to/slangc> \
    --source <shader.hlsl> \
    --entry <entryPoint> \
    --stage <vertex|pixel|compute|...> \
    --output <shader.metallib> \
    --pnms <shader.pnms> \
    --json <shader.metal.reflection.json> \
    [--define KEY=VALUE ...] \
    [--no-warnings-as-errors] \
    [--depfile <shader.d>]
```

### Optional shader stages

On Vulkan, binding visibility masks, push-constant stage masks, and pipeline barriers ignore unsupported optional shader stages as long as at least one supported stage remains (`ShaderStage::all` and `ShaderStage::allGraphics` always work). Passing a stage mask containing **only** unsupported stages fails with `ErrorCode::unsupported`.

Similarly, barriers whose `stagesBefore` or `stagesAfter` mask contains no supported stage (or whose `Access` flags belong exclusively to unsupported stages) return `ErrorCode::unsupported`. Specifically on Vulkan:
- `conditionalExecution`, `streamOutput`, and `accelerationStructureCopy` are rejected as unsupported when targeted exclusively.
- `Access::shadingRateSource` requires `DeviceCapabilities::variableRateShadingTier2`.

### Stage inputs and outputs

Graphics pipeline creation validates pixel shader inputs against the last enabled pre-rasterization stage (`geometry` > `domain` > `mesh` or `vertex`) using bytecode signatures across all backends and the C ABI:

- Every pixel input location and component type must be produced by the preceding stage (`SV_*` system values are excluded; arrays and matrices count as one location per element or column; 64-bit scalars consume two locations). Reading fewer vector components or leaving upstream outputs unused is valid; mismatches fail with `ErrorCode::validationError`.
- HLSL assigns locations in declaration order unless `[[vk::location(N)]]` overrides them on Vulkan (`[[vk::location]]` has no effect on DXIL/Metal). Matrices occupy one register per row in DXIL/Metal versus SPIR-V conventions unless decomposed consistently.
- Metal additionally checks exact semantic names and exact scalar/vector types across stage boundaries.
- For Direct3D 12 `SV_Position` and custom stage outputs, keep semantic names, types, and declaration order identical between paired stages.

---

## Vertex input

For every vertex attribute format, alignment is $\min(4, \text{formatBytes})$ (`unorm8x2` requires 2-byte alignment; all other vertex formats require 4-byte alignment):

- `VertexAttribute::offset` must be a multiple of that attribute's alignment.
- `VertexBufferLayout::stride` and `VertexBufferBinding::offset` must be multiples of the maximum alignment required by any attribute in that buffer slot. For example, three `unorm8x2` attributes permit stride `6`, whereas adding a `unorm8x4` attribute increases the required stride and binding offset multiple to `4`.

Violations fail with `ErrorCode::invalidArgument`, and `bindVertexBuffers` updates bindings atomically (a failed call leaves prior bindings unchanged).

---

## Surfaces

Create a surface by populating `SurfaceDesc` (`Rachis/Presentation.h`):

| `SurfaceKind` | `windowHandle` | `displayHandle` | Notes |
| --- | --- | --- | --- |
| `win32` | `HWND` | Ignored | Windows only |
| `wayland` | `wl_surface*` | `wl_display*` | Both non-null |
| `xcb` | `xcb_window_t` cast via `uintptr_t` (`void*`) | `xcb_connection_t*` | Window ID passed as pointer integer value, **not** `&window` |
| `caMetalLayer` | `CAMetalLayer*` or `NSView*` | Ignored | Native Metal only; rejected by MoltenVK |

Surface and swapchain lifecycle rules:
- **Ownership & destruction:** RachisRHI does not take ownership of the window, display, or layer; keep native objects alive until `destroySurface` completes. Destroy all swapchains attached to a surface before calling `destroySurface` (`ErrorCode::invalidState` otherwise).
- **One swapchain per native window:** RachisRHI allows at most one active swapchain per `HWND`, `(wl_display*, wl_surface*)`, `(xcb_connection_t*, xcb_window_t)`, or `CAMetalLayer*` within a `Device`. Cross-device swapchains on the same Wayland surface violate the Wayland protocol inside Mesa.
- **Metal layer & `NSView` handling:** If passed an `NSView*`, RachisRHI expects a layer-backed view carrying a `CAMetalLayer` (reads occur on the main thread when needed). Occluded or unavailable drawables return `ErrorCode::notReady` on acquire or `PresentStatus::occluded` on present.
- **Metal threads & shared-event limit:** Surface and swapchain calls drain an internal autorelease pool and may run on threads without an active `NSAutoreleasePool`. If the system-wide IOSurface shared-event table is exhausted (its live `IOSurfaceSharedEvent` count is at or near 65,535), `createDevice` returns `ErrorCode::resourceExhausted` until the holding process exits; other `nil` command-queue results return `ErrorCode::notReady` with retry guidance.
- **Surface usage flags:** Query `SurfaceCapabilities::usages` first. On Direct3D 12, requesting `TextureUsage::sampled` with `rgba8UnormSrgb` or `bgra8UnormSrgb` fails with `ErrorCode::unsupported` unless the adapter supports `CastingFullyTypedFormatSupported`.
- **Acquired image & queue rules:**
  - On Vulkan, the `TimelinePoint` passed to `present` must come from a graphics queue and satisfy post-acquire ordering (`ErrorCode::invalidArgument` otherwise).
  - `acquireNextImage` returns `ErrorCode::notReady` on transient timeout, hidden surfaces, or a pending present; `acquireNextImage(swapchain, timeout)` returns `ErrorCode::timeout` when `timeout` expires (`nanoseconds::max()` waits unboundedly).
  - On Direct3D 12 and Vulkan, only one acquired image may be active at a time (`acquire` -> record/submit -> `present` or `cancelSwapchainImage`); out-of-order calls return `ErrorCode::invalidState`.
  - If Direct3D 12 `present` or `resizeSwapchain` fails, `querySwapchainPresentation` and `querySwapchainFrameTiming` still succeed and let you inspect state before recovering or recreating.
  - On Vulkan, a failed `resizeSwapchain` retires the old swapchain cleanly—call `resizeSwapchain` again to recover or `destroySwapchain` to tear down.
  - On Vulkan, surfaces with a fixed extent (Win32) create swapchains at the surface's current extent rather than the requested size—read the actual dimensions from `AcquiredSwapchainImage`. A zero surface extent at `createSwapchain` or a minimized `present` (`PresentStatus::occluded`) suspends the swapchain (`acquireNextImage` returns `ErrorCode::invalidState`, or `notReady` for the timed overload while minimized) until `resizeSwapchain` recreates it, even if restored to the same size.
  - On Direct3D 12, `resizeSwapchain` fails with `ErrorCode::invalidState` if any C++ swapchain image views or binding sets referencing them are still alive.

---

## Multiview rendering

Set `RenderingDesc::viewMask` (non-zero bitmask) when `DeviceCapabilities::multiviewRendering` is `true` (Metal rejects non-zero `viewMask` with `ErrorCode::unsupported`):

- **Attachment clears differ by backend:** Vulkan clears only the layers selected by `viewMask`, whereas Direct3D 12 clears all attachment layers. Avoid relying on out-of-mask layers keeping prior contents across a multiview clear.
- **Query restrictions on Vulkan:** See [Queries](#queries) for query calls disabled inside multiview scopes on Vulkan.

---

## Point lists

Rasterized point lists render at a fixed **1 px** point size across all backends. On Vulkan without `VK_KHR_maintenance5`, any vertex shader rendered with a point-list topology must declare `[[vk::builtin("PointSize")]] float pointSize : PSIZE;` and assign `1.0`—otherwise pipeline creation fails with `ErrorCode::unsupported`.

---

## Graphics pipeline limits on Metal

- **Primitive restart for indexed strips:** Metal always enables index-buffer primitive restart at `0xFFFF` (`uint16`) / `0xFFFFFFFF` (`uint32`) for triangle and line strips. Any graphics pipeline using a strip topology with `drawIndexed` or `drawIndexedIndirect` must explicitly set `GraphicsPipelineDesc::stripIndexFormat` or Metal pipeline creation fails with `ErrorCode::unsupported` (non-indexed strip draws are unaffected). Both Direct3D 12 and Metal reject an index buffer whose index format disagrees with `stripIndexFormat` (`ErrorCode::invalidArgument`).
- **Sample masks:** Metal rejects `GraphicsPipelineDesc::sampleMask` values that mask out active pipeline samples with `ErrorCode::unsupported`.

---

## Handles and lifetime

All GPU objects are returned as typed generational handles backed by monotonic process-wide slot identities:

- **Process handle ceiling:** Each handle kind can allocate up to `16,777,215` (`2^24 - 1`) unique identities over the lifetime of a process before returning `ErrorCode::resourceExhausted`.
- **Automatic retention vs. unretained pools:** By default, submitted command buffers retain referenced resources, views, samplers, pipelines, binding sets, heaps, and query pools until the submission retires on the GPU, so destroying handles right after `submit` is safe. Pools created with `CommandPoolFlag::unretainedResources` (C++ only) skip retention tracking; callers using unretained pools **must** observe GPU completion—via `Device::wait`, `TimelinePointStatus::complete` from `queryTimelinePoint`/timed `wait`, or `completedValue >= point`—before destroying referenced handles (backend-owned internal temporaries remain retained automatically).
- **Texture views:** Keeping a `TextureViewHandle` alive extends the parent texture's native lifetime on Direct3D 12, whereas Vulkan and Metal require destroying all views before `destroyTexture` (`ErrorCode::invalidState` otherwise). Destroy swapchain image views before `resizeSwapchain` or `destroySwapchain` (except `destroySwapchain` after device loss; on Vulkan surviving swapchain views become stale and accept only `destroyTextureView`).

---

## Texture layouts

Texture layout tracking is **always active**, even when `ValidationLevel::disabled` is set:

- **Deterministic validation:** An intra-command-buffer layout mismatch fails that recording call atomically (`ErrorCode::invalidState`). At `submit`, RachisRHI validates the first barrier/access of each command buffer across the batch in order and fails the entire batch atomically (`ErrorCode::invalidState`) on mismatch.
- **Literal portable comparison:** RachisRHI compares logical `TextureLayout` enums directly (`common` != `unorderedAccess`, `copySource`/`copyDestination` != `resolveSource`/`resolveDestination`), matching backend state rather than silently coalescing them.
- **Initial and transition states:**
  - `TextureLayout::undefined` is always valid as `layoutBefore` (discarding prior contents). Newly created textures start in `undefined` (`common` on Direct3D 12 under the hood, `present` for swapchain images), and an acquired swapchain image starts in `AcquiredSwapchainImage::initialLayout`.
  - Copies and resolves require the appropriate layouts on entry and leave subresources in those same layouts.
  - Shared textures imported into another device have unknown external entry layout—only intra-command-buffer transitions are checked until the caller re-establishes known state.
  - `generateMipmaps` transitions the base mip from `baseLayout` and leaves all touched mips in `finalLayout`.
  - `activateTextureAlias` sets all activated subresources to `layoutAfter`.

---

## Readback buffers

Buffers created with `MemoryType::readback` are host-visible once GPU work completes (`Device::wait`):

| Backend | Allowed usages / notes |
| --- | --- |
| Direct3D 12 | Supports only `BufferUsage::copyDestination` |
| Vulkan | Rejects `BufferUsage::storage` with `ErrorCode::unsupported` |
| Metal | Permits `BufferUsage::storage` |

---

## Acceleration structures

Acceleration structures follow a single-build / in-place-update contract:

- **Creation & build order:** An acceleration structure accepts one initial build, out-of-place update, compaction copy, or deserialization, after which only in-place updates are valid. Recording a build marks the target as built—submit the builder's command buffer before submitting any dependent command buffer. Resetting or abandoning a command buffer, or a failed `submit`, rolls back its recorded acceleration-structure state transitions.
- **Direct3D 12 in-place update exclusivity:** Direct3D 12 rejects submitting concurrent command buffers where one records an in-place update of an acceleration structure while another reads the same structure (binding it as a shader resource, building/updating a TLAS referencing it, compacting, serializing, or writing compacted-size / serialization-info queries) with `ErrorCode::invalidState`. Sequential commands within the same command buffer remain valid.
- **Update sizes:** An update's destination (the target itself for an in-place update) must hold the `accelerationStructureSize` that `queryBottomLevelAccelerationStructureBuildSizes` or `queryTopLevelAccelerationStructureBuildSizes` reports for the update's input, even when the source was compacted or deserialized into less memory; a smaller destination fails with `ErrorCode::invalidArgument`. The scratch range must hold the reported `updateScratchSize`, which may be `0`.

---

## Sharing resources

When `DeviceCapabilities::sharedBufferHandles`, `sharedTextureHandles`, and `sharedTimelineHandles` report `true` (Windows NT handles via `createShared*`, `exportShared*`, `importShared*`, `exportTimelinePoint`, and `enqueueSharedTimelineWait`), follow a strict two-step cross-device handoff:

1. Transition the resource to `TextureLayout::common` (for textures), record a release barrier with `OwnershipTransfer::releaseExternal`, `submit`, and call `exportTimelinePoint`.
2. On the importing device, call `enqueueSharedTimelineWait` on the target queue, record an acquire barrier (`OwnershipTransfer::acquireExternal`, `common` -> `common`) before any other use, and then transition to the working layout.

Using an invalid transfer target or releasing from a non-`common` layout returns `ErrorCode::invalidArgument`. For Windows NT handle rules and backend compatibility details, see [Platforms: Shared resources](platforms.md#shared-resources).

`importSharedTexture` requires matching `dimension`, `width`, `height`, `depthOrArrayLayers`, `mipLevels`, `sampleCount`, `format`, `usage`, and unordered `viewFormats`. Direct3D 12 validates descriptors against native allocation properties; Vulkan detects descriptor mismatches across same-process exports (cross-process OS imports cannot detect descriptor mismatches automatically).

---

## Samplers

- Setting `maxAnisotropy > 1` forces `minFilter` and `magFilter` to `FilterMode::linear`.
- `mipFilter` supports `MipmapFilterMode::none`, `nearest`, and `linear`. On Direct3D 12, combining `MipmapFilterMode::nearest` with `maxAnisotropy > 1` requires `AnisoFilterWithPointMipSupported` (`D3D12_FEATURE_DATA_D3D12_OPTIONS19`) and fails with `ErrorCode::unsupported` otherwise.
- `lodBias` must be finite and within `[-maxSamplerLodBias, +maxSamplerLodBias]` (`ErrorCode::invalidArgument` outside range—RachisRHI does not clamp). Hardware bounds are at least `2` on Vulkan, `15.99` on Direct3D 12, and `15.999` on Metal. (In the C ABI, `lodBias` is fixed to `0.0f`.)

---

## Binding pools

`createBindingPool` reserves capacity up to `maxBindingSets` and per-type descriptor `sizes`:

- Calling `updateBindingSet` preserves previous descriptor versions if any active or unobserved-completion command buffer may still reference them (even when recorded from `CommandPoolFlag::unretainedResources` pools), and updates in place once idle. Exhausting pool space returns `ErrorCode::resourceExhausted`; targeting a destroyed pool returns `ErrorCode::invalidState`.
- **Direct3D 12 descriptor heap budgets (`DeviceLimits`):**
  - `maxBindingPoolResourceDescriptors` (`868,928`): reserved up front by `createBindingPool` until the pool and all its sets and retained versions finish and are destroyed.
  - `maxBindingSamplerDescriptors` (`2,048`): shared online sampler heap entries deduplicated by each binding set's ordered `SamplerHandle` sequence and checked at `createBindingSet` / `updateBindingSet`.
  - Two separate `65,536`-descriptor ranges remain unreserved for internal utilities (mipmap generation, depth resolve, sampler feedback clears) and non-root dynamic buffer offsets (`count == 1` `dynamicUniformBuffer` bindings promote directly to root descriptors while root-signature space remains; remaining dynamic buffers allocate per-CB offset tables deduplicated in doubling pages up to `256` descriptors and reclaimed on CB retirement).
  - Vulkan and Metal report `UINT32_MAX` for both limits.

---

## Variable-count binding arrays

Declare an unbounded trailing binding with `BindingFlag::variableDescriptorCount` and `unboundedBindingCount`, then supply the concrete allocation size in `BindingSetDesc::variableDescriptorCount`:

- Query `DeviceLimits::maxVariableBufferDescriptors`, `maxVariableSamplerDescriptors`, and `maxVariableResourceDescriptors` before choosing array capacities.
- On Vulkan, pipeline/binding layout limits cap variable array lengths at `3/4` of the backend stage/pipeline limit (or `limit - 1024` when `1/4 > 1024`) minus other descriptors of the same category in that layout (dynamic and read-only storage buffers share the storage-buffer budget; acceleration structure arrays are capped by the stricter AS descriptor limits).
- Exceeding per-type variable limits fails with `ErrorCode::invalidArgument`; combining sets in `createPipelineLayout` beyond cross-set limits fails with `ErrorCode::unsupported`.

---

## Direct3D 12 without enhanced barriers

When `DeviceCapabilities::enhancedBarriers` is `false`, Direct3D 12 falls back to legacy resource transitions with stricter constraints:

| Operation | Legacy fallback behavior (`enhancedBarriers == false`) |
| --- | --- |
| Write-combined buffer barriers | Rejected (`ErrorCode::unsupported`) |
| Combined `common` + `copyDestination` texture barriers | Rejected (`ErrorCode::unsupported`; `depthStencilRead \| depthStencilWrite` in `depthStencilWrite` layout remains valid) |
| Global memory barriers | Only `Access::unorderedAccess` -> `Access::unorderedAccess` allowed; others return `ErrorCode::unsupported` |
| Same-buffer `copyBuffer` | Rejected (`ErrorCode::unsupported`); stage through a second buffer instead (distinct-subresource `copyTexture` within one texture remains valid) |
| `activateBufferAlias` / `activateTextureAlias` | Rejected (`ErrorCode::unsupported`) |
| Cross-queue texture handoff (graphics <-> compute) | Must transition through `TextureLayout::common` + `Access::common` (omitting `common` in favor of `shaderResource` triggers D3D12 debug error `527` / undefined behavior) |

---

## Threads

`Device` methods are thread-safe, subject to explicit caller invariants:

- **Per-command-buffer thread affinity:** At most one thread may record into a given `CommandBuffer` at any time. Debug groups and GPU profile scopes must begin and end on the same thread while open (`ErrorCode::invalidState`).
- **No concurrent destruction or device move:** Never destroy a handle concurrently with another call using it, destroy/move a `Device` while other calls are executing on it, or re-enter `Device` methods from inside `diagnosticCallback`.
- **Mapped memory & synchronization:** Host reads/writes through `mapBuffer` require manual synchronization against GPU timelines. `Device::waitIdle()` drains every queue on the device.
- **Internal serialization characteristics:**
  - Submissions serialize per queue (except shared native Vulkan queues, which serialize together). Swapchain calls serialize per swapchain (and on Vulkan lock the presenting queue during driver present).
  - Vulkan and Metal acquire an internal device lock during `writeBuffer` / `readBuffer` (plus Vulkan resource create/destroy, copies, barriers, `beginRendering`, `queryTextureTransferLayout`, and handle lookups; and all Metal recording calls), and Vulkan copies/barriers serialize across command buffers. Direct3D 12 `writeBuffer` / `readBuffer` lock per buffer instead—on Vulkan/Metal, chunk large CPU transfers or stream via `mapBuffer`.

---

## Queries

- **Scope rules:**
  - Occlusion queries must begin and end inside the same `beginRendering` / `endRendering` scope.
  - Pipeline-statistics and stream-output-statistics queries must begin and end on the same side of any rendering boundary (mismatches return `ErrorCode::invalidState` and leave both the query and rendering scope open).
- **Vulkan multiview restrictions:** Inside a multiview rendering pass (`viewMask` selecting multiple views) on Vulkan, `writeTimestamp`, `beginGpuProfileScope` / `endGpuProfileScope`, `beginOcclusionQuery`, and `beginPipelineStatisticsQuery` fail with `ErrorCode::unsupported`. `RenderingDesc::timestampWrites` works on all backends (Vulkan writes timestamps immediately before and after the render pass).
- **Queue clock calibration:** `calibrateQueueClock` pairs a GPU timestamp with the host monotonic counter named by `cpuClock` (`queryPerformanceCounter` on Windows, `monotonic` on Linux, `machAbsoluteTime` on Metal, `monotonicRaw` on MoltenVK), with `maxDeviationNanoseconds` bounding the sampling gap and `VK_KHR_calibrated_timestamps` / `VK_EXT_calibrated_timestamps` on Vulkan. Convert a GPU timestamp $t$ via $\text{cpuTimestamp} + (t - \text{gpuTimestamp}) \cdot \text{cpuFrequency} / \text{gpuFrequency}$; unsupported queues or Vulkan devices return `ErrorCode::unsupported`.

---

## Device loss

Once a device enters a lost state, recovery is permanent:

- Query `Device::operationalState()` and `Device::deviceLossError()` at any time.
- Resource creation, submissions, `mapBuffer`, `flushMappedMemoryRanges`, `invalidateMappedMemoryRanges`, `writeBuffer`, and `readBuffer` return `ErrorCode::deviceLost` (`completedValue` freezes at its last observed value).
- Cleanup calls (`destroy*`, `unmapBuffer`) continue to succeed (`destroySwapchain` allows tearing down with still-acquired images or live swapchain views; on Vulkan those swapchain views become stale and accept only `destroyTextureView`; normal non-swapchain textures still require destroying their views first).
- Direct3D 12 resource creation calls may be the first place `deviceLost` surfaces; command recording calls that hit driver failures report `ErrorCode::nativeFailure` or `ErrorCode::outOfMemory` until the next `submit` or `wait` latches device loss.
- If an imported Direct3D 12 shared timeline fence returns `UINT64_MAX`, `enqueueSharedTimelineWait` fails with `ErrorCode::invalidState` without marking the importing device lost.

---

## Diagnostics

RachisRHI writes nothing to `stdout`/`stderr` on its own. Attach `DeviceDesc::diagnosticCallback` (and `diagnosticUserData`) to stream backend messages, and poll `Device::diagnosticCounts()` regardless of whether a callback is installed.

> [!WARNING]
> `diagnosticCallback` may fire on backend or validation threads: never re-enter `Device` methods from inside the callback, copy any string data you need before returning, and keep `diagnosticUserData` alive until `~Device()` returns (Vulkan validation reports leaked handles during device teardown).

---

## Examples

Runnable end-to-end programs and smoke tests:

- Compute buffer squaring pipeline: [`examples/compute/main.cpp`](../examples/compute/main.cpp)
- Offscreen triangle rendered to PPM: [`examples/triangle/main.cpp`](../examples/triangle/main.cpp)
- Building and CLI options for all examples: [`examples/README.md`](../examples/README.md)
- Minimal C11 ABI program: [`tests/c_abi/CAbiSmoke.c`](../tests/c_abi/CAbiSmoke.c)
