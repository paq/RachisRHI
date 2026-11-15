# Examples

Two self-verifying C++ examples demonstrate the RHI end to end without creating a window, running across every enabled backend.

| Example | What it demonstrates |
| --- | --- |
| [`compute`](compute/main.cpp) | Compute dispatch squaring 1,024 integers: `rachis_add_shader` + `REFLECT` interfaces, binding layouts/pools/sets, device-local/upload/readback buffers, buffer barriers, submission, and waiting |
| [`triangle`](triangle/main.cpp) | Offscreen triangle rendered to a texture and saved as a PPM image: graphics pipelines, render passes, texture barriers, and texture-to-buffer readback with queried row alignment |

Both examples validate their own output and exit non-zero on error, allowing CTest to execute them directly. CLI parsing and error helpers live in [`common/ExampleSupport.h`](common/ExampleSupport.h).

## Shaders

Each example compiles its HLSL via `rachis_add_shader(... REFLECT)` in [`examples/CMakeLists.txt`](CMakeLists.txt). CMake builds DXIL (D3D12), SPIR-V (Vulkan), and a packed Metal library (Metal), embedding both the bytecode and `slangc`-reflected `constexpr` descriptors into a generated header so no files are loaded at run time:

```cpp
#include "SquareShader.h"

namespace square = rachis_shaders::reflection::SquareShader;

auto shader = device.createShaderModule(square::moduleDesc(backend, "Square"));
auto layout = device.createBindingLayout({.entries = square::bindingLayout(backend, 0)});
```

See [Shaders](../docs/using.md#shaders) for details on `rachis_add_shader`.

## Building in this repository

Examples build automatically when `RACHIS_BUILD_EXAMPLES` is `ON` (the default for top-level builds with tests):

```sh
cmake --preset windows-x64-both
cmake --build --preset windows-x64-both-debug
ctest --test-dir out/build/windows-x64-both -C Debug -L example --output-on-failure
```

- CTest names: `RachisExample.<example>.<backend>`
- Executable output: `out/build/<preset>/examples/<config>`

## Building against an installed package

`examples/` is also a standalone CMake project. Point `CMAKE_PREFIX_PATH` at your install prefix (or set `Rachis_DIR` to `<prefix>/<libdir>/cmake/Rachis`; see [CMake targets](../docs/using.md#cmake-targets)) and build in a configuration present in the package:

```sh
cmake -S examples -B build/examples -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<prefix>
cmake --build build/examples
ctest --test-dir build/examples --output-on-failure
```

- Packages built via [Installing](../docs/building.md#installing) contain `Release` only. On MSVC, a `Debug` example build requires both configurations in one prefix (see [Debug and Release in one prefix](../docs/building.md#debug-and-release-in-one-prefix)).
- Requires shader compilers for the package's backends (see [Tools](../docs/building.md#tools)), and on Windows an x64 Visual Studio developer shell.

## Running

```sh
rachis-example-compute  [--backend d3d12|vulkan|metal] [--warp] [--validation disabled|standard|gpu]
rachis-example-triangle [--backend d3d12|vulkan|metal] [--warp] [--validation disabled|standard|gpu] [--output <path>]
```

| Flag | Behavior |
| --- | --- |
| `--backend` | Defaults to Direct3D 12 on Windows, Metal on macOS, and Vulkan on Linux |
| `--warp` | Selects the WARP software adapter on Direct3D 12 |
| `--validation` | Defaults to `standard` (examples run as tests and fail if validation reports any error, skipping missing layers unless present); see [Fallback and validation](../docs/platforms.md#fallback-and-validation) |
| `--output` | Output path for `triangle` (defaults to `triangle.ppm` in the working directory) |

### Exit code `77` (CTest skip)

An example exits with `77` (reported as skipped by CTest) when its backend cannot run on the host: Direct3D 12 or Metal has no available adapter, or on Linux/macOS no Vulkan driver is present (or only rejected CPU drivers exist). On Windows, Vulkan failures fail the test rather than skipping, matching the test suite rule in [`tests/README.md`](../tests/README.md#requirements).
