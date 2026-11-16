# Building

This guide covers toolchain requirements, Nix development shells, CMake presets and options, test execution, packaging, and embedding RachisRHI directly into another CMake project.

## Contents

- [Tools](#tools)
  - [Nix development shell](#nix-development-shell)
- [Presets](#presets)
- [Options](#options)
- [Tests](#tests)
- [Installing](#installing)
  - [Debug and Release in one prefix](#debug-and-release-in-one-prefix)
  - [C ABI library version](#c-abi-library-version)
  - [Development install](#development-install)
- [Using the source tree directly](#using-the-source-tree-directly)

## Tools

Building RachisRHI requires:
- **CMake 4.0 or later**
- **Ninja** (repository presets use the `Ninja Multi-Config` generator)
- **A C++23 compiler** (on Windows, run CMake from an **x64 Visual Studio developer shell**)

RachisRHI compiles its internal shaders at build time and never invokes a shader compiler at run time. Consuming an installed package requires shader tools only if your project calls `rachis_add_shader` (see [Shaders](using.md#shaders)).

| Backend | Required build tools |
| --- | --- |
| **Direct3D 12** | DirectX Shader Compiler (`dxc`) on `PATH` |
| **Vulkan** | SPIR-V-capable `dxc` and `spirv-val` (both from the Vulkan SDK; the Windows SDK `dxc` cannot emit SPIR-V). Building tests also requires Slang (`slangc`) and glslang (`glslangValidator`) |
| **Metal** | Xcode 26+ with the macOS 26 SDK, and Slang (`slangc`) `2026.17.1+` built for `arm64` |

- **Shader languages:** The table lists the tools for HLSL shaders. Slang shaders (`LANGUAGE SLANG`) compile with `slangc` on every backend; Direct3D 12 needs the `dxcompiler` library that ships beside `slangc` in the Vulkan SDK. GLSL shaders (`LANGUAGE GLSL`, Vulkan only) compile with `glslangValidator` from the Vulkan SDK, or `RACHIS_GLSLANG_EXECUTABLE`, and are checked with `spirv-val`.
- **`REFLECT` shaders:** Calling `rachis_add_shader(... REFLECT)` on an HLSL or Slang shader generates shader interfaces and C++ buffer structs using `slangc`, so `slangc` is required on **all backends** when `REFLECT` is used (including for building the examples and tests). A GLSL shader's interface comes from its SPIR-V and needs no `slangc`. Without `REFLECT`, only the compilers above are needed.
- **`slangc` lookup order:** CMake checks the `SLANGC` environment variable first, then `PATH`, then the SDK referenced by `VULKAN_SDK`.
- **Linux presentation tests:** Require `wayland-scanner` and development packages for `wayland-client`, `xcb`, and `wayland-protocols`.

During configuration, CMake downloads pinned revisions of Vulkan-Headers, volk, Vulkan Memory Allocator, D3D12 Memory Allocator, WinPixEventRuntime, and (when tests are enabled and Catch2 is not already installed) Catch2. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

### Nix development shell

On Linux x64 and Apple silicon macOS, `nix develop` enters a shell with CMake, Ninja, `clang-format`, `slangc`, a SPIR-V-capable `dxc`, `glslangValidator`, `spirv-val`, the Vulkan loader, `vulkaninfo`, and the Khronos validation layers, with `SLANGC` preconfigured:

| Host | Additional environment details |
| --- | --- |
| **Linux x64** | Adds GCC, `pkg-config`, and the Wayland/XCB packages needed by presentation tests. Does not supply a Vulkan driver—the loader uses your host hardware driver (CPU drivers like lavapipe are rejected). |
| **macOS arm64** | Adds MoltenVK via `VK_DRIVER_FILES` and validation layers via `VK_LAYER_PATH`, while keeping Xcode's `clang` for the macOS 26 SDK. |

```sh
nix develop
cmake --preset linux-x64-vulkan
cmake --build --preset linux-x64-vulkan-debug
ctest --preset linux-x64-vulkan-small-medium --output-on-failure
```

There is no Windows Nix shell; install the native tools from the table above instead.

## Presets

### Development presets

| Configure preset | Host | Backends |
| --- | --- | --- |
| `windows-x64-d3d12` | Windows | Direct3D 12 |
| `windows-x64-vulkan` | Windows | Vulkan |
| `windows-x64-both` | Windows | Direct3D 12 and Vulkan |
| `linux-x64-vulkan` | Linux | Vulkan |
| `macos-arm64-metal` | macOS | Metal |
| `macos-arm64-vulkan` | macOS | Vulkan via MoltenVK *(unsupported investigation preset)* |

Each development preset has matching `<preset>-debug` build and `<preset>-small-medium` test presets (and `macos-arm64-metal-all` on macOS):

```sh
cmake --preset macos-arm64-metal
cmake --build --preset macos-arm64-metal-debug
ctest --preset macos-arm64-metal-small-medium --output-on-failure
```

### Product presets (for packaging)

| Product preset | Host | Backends |
| --- | --- | --- |
| `windows-x64-both-product` | Windows | Direct3D 12 and Vulkan |
| `linux-x64-vulkan-product` | Linux | Vulkan |
| `macos-arm64-metal-product` | macOS | Metal |

Product presets set `BUILD_TESTING=OFF`: they do not download Catch2, build tests, or compile `RACHIS_TESTING` hooks into the libraries. Each supplies `<preset>-debug` and `<preset>-release` build presets (no test presets). Use these presets for packaging as shown in [Installing](#installing).

All build outputs go to `out/build/<preset>`.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `RACHIS_BACKENDS` | `both` (Windows), `metal` (macOS), `vulkan` (Linux) | Backends to build (`d3d12`, `vulkan`, or `both` on Windows) |
| `RACHIS_ENABLE_DEBUG_MARKERS` | `ON` | Enables PIX debug markers on D3D12. Windows executables linking the package then depend on `WinPixEventRuntime.dll` (see [Windows runtime DLLs](using.md#windows-runtime-dlls)) |
| `RACHIS_BUILD_TESTS` | `ON` (top-level), `OFF` (embedded) | Builds tests, test shaders, and `RACHIS_TESTING` hooks when `BUILD_TESTING` is also `ON` |
| `RACHIS_INSTALL` | `ON` (top-level), `OFF` (embedded) | Enables install rules and CMake package generation (required by `RachisPackage*` tests) |
| `BUILD_TESTING` | `ON` (top-level), `OFF` (product presets) | Controls Catch2 tests and `RACHIS_TESTING` hooks alongside `RACHIS_BUILD_TESTS`. Manual packaging builds not using a product preset must pass `-DBUILD_TESTING=OFF` |
| `RACHIS_BUILD_EXAMPLES` | `ON` (top-level with tests) | Builds programs in [`examples/`](../examples) and registers `RachisExample.<example>.<backend>` tests |
| `RACHIS_SHADER_BINARY_TOOL` | Empty | Path to a prebuilt host `RachisShaderBinaryTool` for cross-compiling. When set, RachisRHI uses it instead of installing its own; downstream projects using `rachis_add_shader` against that package must also set it. Required at configure time for cross-compiling builds that run the tool (all Metal builds, or builds with tests/examples; D3D12/Vulkan library-only builds configure without it and install a target tool) |

## Tests

CTest exposes each test's size and extra hardware/session dependencies as labels:

| Size label | Requirements |
| --- | --- |
| `size:small` | No GPU or device required; runs anywhere |
| `size:medium` | Offscreen device per enabled backend: WARP for most D3D12 cases; a hardware Vulkan 1.3 driver (CPU implementations like lavapipe/SwiftShader are rejected; on Windows, reference desktop GPU optional features are also required); or Apple silicon for the single non-large Metal device test |
| `size:large` | Physical GPU or active window system (Metal has no software device, so almost all Metal device tests are `size:large`) |

Some medium tests carry `requires:` labels (such as `requires:HardwareD3D12`, `requires:HardwareVulkan`, `requires:InteractiveDesktop`, `requires:Wayland`, `requires:X11`) and can be excluded with `-LE "requires:"`. However, not every hardware or desktop dependency is tagged—see [`tests/README.md`](../tests/README.md#requirements) for exact rules on skips vs failures.

Common test runs:

```sh
# Windows machine without a GPU (VM/CI runner; run from a signed-in desktop session):
ctest --preset windows-x64-d3d12-small-medium -LE "requires:" --output-on-failure

# Linux machine without a GPU (excludes Wayland/X11/hardware cases; remaining Vulkan cases skip):
ctest --preset linux-x64-vulkan-small-medium -LE "requires:" --output-on-failure

# macOS virtual machine whose GPU fails the Metal 4 profile (such as a GitHub-hosted runner):
ctest --preset macos-arm64-metal-small-medium -LE hardware --output-on-failure

# Large tier tests:
ctest --test-dir out/build/windows-x64-d3d12 -C Debug -L size:large --output-on-failure

# All three tiers on macOS:
ctest --preset macos-arm64-metal-all --output-on-failure
```

See [`tests/README.md`](../tests/README.md) for the full test directory structure, subset filtering, and package tests.

## Installing

Package from a product preset in the **Release** configuration (replace `windows-x64-both-product` with `linux-x64-vulkan-product` or `macos-arm64-metal-product` on those hosts):

```sh
cmake --preset windows-x64-both-product
cmake --build --preset windows-x64-both-product-release
cmake --install out/build/windows-x64-both-product --config Release --prefix out/install/rachis
```

> [!IMPORTANT]
> Redistribute **only Release builds** of `Rachis::C`. On Windows, a Debug `rachis.dll` links the MSVC debug runtime DLLs, which Microsoft does not license for redistribution.

### Installed prefix layout

```text
<prefix>/
├── bin/
│   ├── RachisShaderBinaryTool(.exe)   # omitted when RACHIS_SHADER_BINARY_TOOL was set
│   ├── rachis.dll                     # Windows C ABI shared library
│   └── WinPixEventRuntime.dll         # Windows D3D12 with debug markers enabled
├── include/
│   └── Rachis/                        # public C++ and C ABI headers
├── <libdir>/
│   ├── cmake/Rachis/                  # CMake package config + rachis_add_shader
│   ├── Rachis/                        # private bundled volk, D3D12MA, and PIX import libs (no headers/CMake configs)
│   └── ...                            # RachisRHI static libs and librachis shared lib
└── share/
    ├── doc/Rachis/                    # README, CHANGELOG, docs/, LICENSE, THIRD_PARTY_NOTICES, third_party/
    └── licenses/Rachis/               # LICENSE, THIRD_PARTY_NOTICES, third_party/ (kept even if docs are stripped)
```

- `<libdir>` follows `GNUInstallDirs` (`CMAKE_INSTALL_LIBDIR`): `lib` on Windows, macOS, and Debian/Arch Linux; `lib64` on Fedora, openSUSE, RHEL, and NixOS; `lib/<multiarch>` under Debian `/usr`. Pass `-DCMAKE_INSTALL_LIBDIR=lib` for a uniform layout.
- Bundled dependencies live privately in `<libdir>/Rachis` with no installed headers or CMake configs so they never collide with system copies in a shared prefix.
- `CMAKE_INSTALL_LIBDIR` and `CMAKE_INSTALL_BINDIR` may be absolute paths (as used in Nix packaging); note that `cmake --install --prefix` cannot relocate absolute destinations.

### Debug and Release in one prefix

`Rachis::RHI` and its backend libraries are static C++ libraries. On MSVC, linking mismatched configurations fails with `LNK2038`. To provide both configurations in one prefix for C++ consumers, install **Debug first** and **Release second**:

```sh
cmake --preset windows-x64-both-product
cmake --build --preset windows-x64-both-product-debug
cmake --install out/build/windows-x64-both-product --config Debug --prefix out/install/rachis
cmake --build --preset windows-x64-both-product-release
cmake --install out/build/windows-x64-both-product --config Release --prefix out/install/rachis
```

- Debug static libraries carry a `d` suffix (e.g., `RachisRHId.lib`, `Rachis/volkd.lib`, `libRachisRHId.a`), allowing side-by-side installation. `find_package(Rachis)` selects the matching configuration (`RelWithDebInfo` and `MinSizeRel` use Release).
- The C ABI shared library has no configuration suffix (`rachis.dll`, `librachis.2.2.0.dylib`), so installing Release last ensures the redistributable DLL remains in the prefix. `Rachis::C` does not depend on the C runtime configuration and works with either.

### C ABI library version

The C ABI shared library is versioned from `RACHIS_ABI_VERSION` in `rachis.h` (`<abi>.<package minor>.<package patch>`), not the package major version. For package `0.2.0` and C ABI `2`:

| Host | Library artifacts |
| --- | --- |
| **Linux** | `librachis.so.2.2.0`, `SONAME` `librachis.so.2`, symlinks `librachis.so.2` and `librachis.so` |
| **macOS** | `librachis.2.2.0.dylib`, install name `@rpath/librachis.2.dylib`, compatibility `2.0.0`, current `2.2.0`, symlinks `librachis.2.dylib` and `librachis.dylib` |
| **Windows** | `rachis.dll` with image version `2.2` |

Different C ABI versions produce distinct `SONAME`s and install names so the dynamic loader never binds an incompatible version.

### Development install

Installing from a test-enabled preset (e.g., `windows-x64-both`) is strictly for local development:

```sh
cmake --install out/build/windows-x64-both --config Debug --prefix out/install/rachis-dev
```

Its libraries include `RACHIS_TESTING` hooks, and on Windows `rachis.dll` links the debug CRT—never redistribute it.

See [Using the library](using.md) for consuming an installed package.

## Using the source tree directly

You can embed RachisRHI via `add_subdirectory` or `FetchContent`:

- Exposes the same `Rachis::` targets as `find_package(Rachis)`, and `rachis_add_shader` works in any directory of the parent project.
- Targets require C++23 via compile features. RachisRHI declares its own `project()` and `cmake_minimum_required(VERSION 4.0)`, so CMake 4.0+ is required and RachisRHI retains its own policy settings.
- When embedded, RachisRHI disables tests, test shaders, Catch2 fetching, `RACHIS_TESTING` hooks, and install rules regardless of the parent's `BUILD_TESTING`. To override, set `RACHIS_BUILD_TESTS` or `RACHIS_INSTALL` to `ON` before adding RachisRHI (tests also require `BUILD_TESTING=ON`).
- Dependency options are passed as directory-scoped variables so your cache stays clean.

If your parent project defines any of these targets **before** adding RachisRHI, RachisRHI reuses them instead of downloading pinned copies:

| Target | Requirement |
| --- | --- |
| `Vulkan-Headers` | `VK_HEADER_VERSION` >= `296` (older versions fail configuration) |
| `volk` | Static library with `VOLK_STATIC_DEFINES` including `VK_ENABLE_BETA_EXTENSIONS` (plus `VK_USE_PLATFORM_WIN32_KHR` on Windows) |
| `VulkanMemoryAllocator` | `3.2.1` or compatible `3.x` release |
| `D3D12MemoryAllocator` | `3.2.0` or compatible `3.x` release |

- With `RACHIS_INSTALL=ON`, `volk` and `D3D12MemoryAllocator` must be static libraries.
- Defining any of these target names *after* RachisRHI will collide with RachisRHI's targets.
- The Vulkan backend compiles `VMA_IMPLEMENTATION` itself, so the parent project must not compile it again.
- On Windows, executables still require runtime DLLs as described in [Windows runtime DLLs](using.md#windows-runtime-dlls).
