# Tests

The RachisRHI test suite is organized by test scope (`core`, `portable`, backend-specific, `c_abi`, `package`) and then by feature area.

## Contents

- [Directory layout](#directory-layout)
- [Feature areas](#feature-areas)
- [Sizes](#sizes)
- [Requirements](#requirements)
- [Running a subset](#running-a-subset)
  - [Package and embedding tests](#package-and-embedding-tests)
  - [Build-rule and manifest checks](#build-rule-and-manifest-checks)
  - [C ABI tests](#c-abi-tests)
- [Adding a test](#adding-a-test)

## Directory layout

| Folder | Under test | Hardware / environment needed |
| --- | --- | --- |
| `core/` | Backend-independent code: `Device` front end, handle tables, and `RachisShaderBinaryTool` | None |
| `portable/` | Contracts executed identically across all enabled backends | Device for each enabled backend |
| `d3d12/` | Direct3D 12 backend via public C++ API | WARP (or physical GPU where tagged) |
| `vulkan/` | Vulkan backend via public C++ API | Hardware Vulkan 1.3 driver |
| `metal/` | Metal backend via public C++ API | Apple silicon GPU (`MTLGPUFamilyMetal4`) |
| `c_abi/` | C11 interface in `rachis.h`, including pure C smoke test | Device for enabled backends |
| `package/` | Installed CMake package & `add_subdirectory` embedding (`consumer/`, `subdirectories/`, `shadernames/`, `includes/`, `buildpaths/`, `config/`, [`examples/`](../examples), `CheckEmbeddedProject.cmake`) | Built package and device for examples |
| `shaders/` | HLSL shader sources compiled for tests | — |
| `support/` | Shared test helpers: device descriptors, window hosts, fault injection, a stand-in `slangc`, and contract bodies that several backends run (`DrawIndexContract.h` for the draw index, `FillAndClearContract.h` for `fillBuffer` and `clearStorageTexture`) | — |

## Feature areas

`d3d12/`, `vulkan/`, and `metal/` mirror the same subdirectory structure:

| Area | Scope |
| --- | --- |
| `device/` | Device creation, adapter selection, host profiles, handle provenance, and device loss |
| `diagnostics/` | Diagnostic messages, fault reports, and debug markers |
| `resources/` | Buffers, textures, views, samplers, formats, copies, mipmaps, and resolves |
| `memory/` | Heaps, placed/sparse resources, residency, aliasing, and budgets |
| `commands/` | Queues, command pools, barriers, and timelines |
| `binding/` | Binding layouts, pools, and sets |
| `shaders/` | Shader modules, libraries, interfaces, specialization, and shader arithmetic |
| `pipelines/` | Pipeline creation, asynchronous compilation, and caches |
| `rendering/` | Render passes, raster/depth-stencil/blend states, optional stages, and headless parity |
| `compute/` | Compute dispatch |
| `queries/` | Timestamp, occlusion, and pipeline-statistics queries; conditional execution |
| `raytracing/` | Acceleration structures, ray queries, and ray-tracing pipelines |
| `presentation/` | Surfaces, swapchains, presentation, and resizing |
| `interop/` | Shared buffers, textures, and timelines across APIs/devices |
| `internals/` | GPU-less backend unit tests (fake Vulkan driver; plain D3D12 helpers) |
| `performance/` | *(Metal only)* Command recording and pooling performance |

`d3d12/internals/` and `vulkan/internals/` compile on every host regardless of enabled backends (cases in `vulkan/internals/` requiring Vulkan headers build only when Vulkan is enabled).

## Sizes

Every test carries a `size:` label exposed to CTest:

| Label | Meaning |
| --- | --- |
| `size:small` | No GPU or device required; runs anywhere |
| `size:medium` | Offscreen device per enabled backend: WARP for most D3D12 cases, headless hardware device for Vulkan, Apple silicon GPU for Metal (see [Requirements](#requirements) for exceptions) |
| `size:large` | Physical GPU or active window system (Metal has no software device, so all but one Metal device test are `size:large`) |

## Requirements

Medium tests with extra environment requirements carry `requires:` CTest labels:

| Label | Extra requirement |
| --- | --- |
| `requires:HardwareD3D12` | Physical Direct3D 12 adapter instead of WARP |
| `requires:HardwareVulkan` | Hardware Vulkan adapter *(untagged Vulkan cases need one too; see below)* |
| `requires:InteractiveDesktop` | Visible windows (signed-in Windows desktop session, or Linux display server named by `requires:Wayland` / `requires:X11`) |
| `requires:Wayland` | Running Wayland compositor (skips if unavailable) |
| `requires:X11` | Running X server (skips when `DISPLAY` is unset) |

Exclude all `requires:` tests with `-LE "requires:"`:

```sh
ctest --preset windows-x64-d3d12-small-medium -LE "requires:" --output-on-failure
```

### Important untagged dependencies

- **Hardware Vulkan everywhere:** Every Vulkan device test requires a **hardware Vulkan 1.3 driver**, tagged or not. Software drivers (`lavapipe`, `llvmpipe`, `SwiftShader`) are always rejected by the backend.
- **Vulkan skip vs fail policy:**
  - `rachis::test::skipWithoutCreatedDevice` and `rachis::test::skipWithoutVulkanAdapterList` **fail on Windows** and **skip on Linux/macOS** only when no Vulkan loader/driver exists or all drivers are rejected CPU implementations (other errors like `adapterNotFound` or non-conforming hardware drivers still fail).
  - Pass `--force-vulkan` to `RachisRHITests` to force failure instead of skipping on Linux/macOS too.
  - `RachisExample.<example>.vulkan` tests follow `skipWithoutCreatedDevice` (without `--force-vulkan`).
- **Windows Vulkan reference features:** On Windows, Vulkan tests also expect optional features exposed by the reference desktop GPU (geometry/tessellation shaders, conservative rasterization, fragment barycentrics, rasterizer-ordered storage, 16- and 64-bit shader arithmetic). On other hosts, tests skip or omit checks when unsupported.
- **Windows interactive sessions:** Some untagged medium tests (including D3D12 swapchain contracts and a Vulkan surface contract) create hidden/auxiliary windows—run the medium tier from a **signed-in desktop session**, not a headless Windows service.
- **GPU-less machines:** On a GPU-less Windows runner, only `windows-x64-d3d12` with `-LE "requires:"` can pass the medium tier (`windows-x64-vulkan` and `windows-x64-both` fail without hardware Vulkan). On a GPU-less Linux runner, `-LE "requires:"` skips window-system and hardware tests and remaining Vulkan tests skip. On a macOS virtual machine whose GPU fails the Metal 4 profile (such as a GitHub-hosted runner), `-LE hardware` leaves out the one medium Metal device test.

## Running a subset

C++ unit/contract tests live in `RachisRHITests` and register in CTest as `RachisRHI.<test case name>`:

```sh
# Preset small + medium tests
ctest --preset macos-arm64-metal-small-medium --output-on-failure

# Multiple -L flags intersect labels (metal AND present)
ctest --test-dir out/build/macos-arm64-metal -C Debug -L metal -L present --output-on-failure

# Regex filter on test name
ctest --test-dir out/build/macos-arm64-metal -C Debug -R "RachisRHI.Metal texture" --output-on-failure
```

### Package and embedding tests

| Test | Size | What it verifies |
| --- | --- | --- |
| `RachisPackageConsumer` | `medium` (with D3D12) / `large` | Builds and runs `package/consumer/` against the installed package from both C++ and pure C |
| `RachisPackageExamples` | `medium` (with D3D12) / `large` | Builds and runs [`examples/`](../examples) against the installed package |
| `RachisPackageSiblings` | `small` | Configures `package/subdirectories/` using `find_package(Rachis)` and `rachis_add_shader` across sibling directories |
| `RachisPackageCrossCompiling` | `small` | Configures `package/subdirectories/` with cross-compiling variables set: verifies configure stops without `RACHIS_SHADER_BINARY_TOOL` and succeeds when pointing at the installed tool |
| `RachisPackageConfig` | `small` | Configures `package/config/`, testing `find_package(Rachis COMPONENTS ...)` per target and confirming `RelWithDebInfo`/`MinSizeRel` select Release exports |
| `RachisPackageShaderNames` | `small` | Builds and runs `package/shadernames/`, verifying per-target isolation when multiple targets call `rachis_add_shader` with the same `NAME` (`FillShader`), duplicate `NAME` errors, `REFLECT` failure without `slangc`, `LANGUAGE` errors for an unknown extension or language, and `STAGE` forwarding to `slangc` on Metal |
| `RachisPackageShaderIncludes` | `small` | Builds `package/includes/` (nested shader includes in a space-containing path, and a shader that includes `Rachis/DrawIndex.hlsli` from the installed package), modifies a nested header, and verifies rebuilds plus updated hashes and macOS SDK versions in `.key` files |
| `RachisPackageShaderBuildPaths` | `small` | Builds `package/buildpaths/` in a directory containing a comma (`build,comma`) and hashes the compiled `.dxil`/`.spv`/`.pnms` artifacts |
| `RachisEmbeddedProject` | `small` | Configures `package/subdirectories/` via `add_subdirectory` (with CTest enabled): verifies no RachisRHI tests/test-shaders leak into CTest or CMake File API, checks `RachisShaderBinaryTool` stays in its binary dir, verifies cross-compiling error without `RACHIS_SHADER_BINARY_TOOL`, and hides `slangc` on non-Metal builds |

### Build-rule and manifest checks

| Test | Condition | What it verifies (`size:small`) |
| --- | --- | --- |
| `RachisRuntimeDllCopies` | Ninja + D3D12 + debug markers | Inspects `ninja -t commands` to confirm each executable/DLL directory receives `WinPixEventRuntime.dll` from exactly one copy rule (no parallel race) |
| `RachisTestShaderRules` | Ninja + D3D12 | Inspects `ninja -t query` to verify each test `.dxil` builds via its own rule depending on its `.hlsl` source |
| `RachisShaderArtifactKeys` | Vulkan enabled | Verifies Vulkan mipmap `.key` files record every `dxc` flag including register shifts |
| `RachisTlcTimeout` | Linux/macOS with `bash` + `perl` | Tests [`tools/tlc.sh`](../tools/tlc.sh) `TLC_TIMEOUT=1` Perl process-group termination against a sleeping stand-in `java` |
| `RachisThirdPartyNotices` | Always | Verifies [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) links match all installed third-party license files |
| `RachisDeviceCreationGuards` | Always | Ensures Vulkan test code checks failures via `skipWithoutCreatedDevice` / `skipWithoutVulkanAdapterList` instead of silently passing |

### C ABI tests

- `RachisCAbiManifest` checks agreement between `include/Rachis/rachis.h`, `src/C/rachis_abi_exports.txt`, and exported library symbols (`nm` on Linux/macOS, `dumpbin`/`llvm-readobj` on Windows). `RachisCAbiManifestFindsLeaks` verifies the check catches non-`rachis_` functions, `rachis_` constants, and weak `rachis_` symbols in `c_abi/exportleak/`.
- `RachisCAbiFreeze*` run [`cmake/CheckCAbiFreeze.cmake`](../cmake/CheckCAbiFreeze.cmake) on the fixtures in `c_abi/freeze/`: whitespace-only changes and a raised `RACHIS_ABI_VERSION` pass, while a changed declaration, a joined preprocessor line, or a lowered version fail. Comparing the real `rachis.h` with the last release tag needs the git history, so the *C ABI freeze* CI job runs that instead of a test.
- Size classification across `RachisCAbi*`:
  - `size:small`: `RachisCAbiManifest`, `RachisCAbiFreeze*`, `RachisCAbiLibraryVersion`, and device-free cases of `RachisCAbiTests` and `RachisCAbiFaultInjectionTests`.
  - `size:medium`: `RachisCAbiExports` (Linux/macOS) and Windows presentation cases.
  - `size:large`: presentation fault cases and `RachisCAbiVulkanFaultInjectionTests`.
  - Remaining device cases are `size:medium` when D3D12 is enabled and `size:large` otherwise.

## Adding a test

1. Place the test file under `<backend>/<area>/` (or `portable/`, `core/`, `c_abi/`).
2. Add it to the appropriate `RachisRHITests` source list in the root [`CMakeLists.txt`](../CMakeLists.txt).
3. Tag every `TEST_CASE` with a `size:` tag (`[size:small]`, `[size:medium]`, or `[size:large]`), plus any needed `requires:` tags.
4. For Vulkan device creation or adapter enumeration, always handle failure through `rachis::test::skipWithoutCreatedDevice(created)` or `rachis::test::skipWithoutVulkanAdapterList(adapters)` so `RachisDeviceCreationGuards` passes and unsupported hosts do not report false passes.
