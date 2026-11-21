# RachisRHI

RachisRHI is a C++23 GPU abstraction targeting **Direct3D 12**, **Vulkan 1.3**, and **Metal 4**, paired with a versioned **C11 ABI** for cross-module device sharing and **TLA+** safety specifications verified with TLC.

The C++23 API may change between minor releases before `1.0.0`.

> [!NOTE]
> Backends differ in optional hardware/API feature coverage. Check [Optional features by backend](docs/platforms.md#optional-features-by-backend) before relying on optional capabilities across backends.

---

## Backend support

| Backend | Tested hosts | Minimum details & notes |
| --- | --- | --- |
| **Direct3D 12** | Windows 11 (`x64`) | Native DXIL via `dxc`; optional WARP software adapter |
| **Vulkan 1.3** | Windows 11 (`x64`), Linux (`x64`) | SPIR-V via Vulkan SDK `dxc` + `spirv-val`; rejects layered Vulkan-on-D3D12 |
| **Metal 4** | macOS 26+ (Apple silicon) | Packed metallib (`.pnms`) via `slangc` + `RachisShaderBinaryTool` |

Each row has been tested on GPU hardware. GitHub Actions has no GPU, so CI runs the Direct3D 12 tests on WARP and only the GPU-less tests for Vulkan and Metal. The rows list what was tested, not a minimum; see [Platforms](docs/platforms.md) for backend requirements and what older OS versions would need.

---

## API equivalence

RachisRHI models a Vulkan 1.3–style explicit API (dynamic rendering, binding sets, and timeline synchronization) over all three backends. Cells marked *no native object* are state RachisRHI tracks itself.

| RachisRHI | Direct3D 12 | Vulkan 1.3 | Metal 4 |
| --- | --- | --- | --- |
| [`Device`](include/Rachis/Device.h) | `ID3D12Device` | `VkInstance` + `VkDevice` | `id<MTLDevice>` |
| Queue, chosen by [`QueueType`](include/Rachis/Command.h) | `ID3D12CommandQueue` | `VkQueue` | `id<MTL4CommandQueue>` |
| [`CommandPoolHandle`](include/Rachis/Command.h) | `ID3D12CommandAllocator` | `VkCommandPool` | `id<MTL4CommandAllocator>` |
| [`CommandBuffer`](include/Rachis/Command.h) | `ID3D12GraphicsCommandList` | `VkCommandBuffer` | `id<MTL4CommandBuffer>` |
| [`TimelinePoint`](include/Rachis/Command.h) | `ID3D12Fence` + value | Timeline `VkSemaphore` + value | `id<MTLSharedEvent>` + value |
| [`ResourceHeapHandle`](include/Rachis/Resource.h) | `ID3D12Heap` | `VkDeviceMemory` | `id<MTLHeap>` |
| [`BufferHandle`](include/Rachis/Resource.h) / [`TextureHandle`](include/Rachis/Resource.h) | `ID3D12Resource` | `VkBuffer` / `VkImage` | `id<MTLBuffer>` / `id<MTLTexture>` |
| [`TextureViewHandle`](include/Rachis/Resource.h) | Descriptor (SRV/UAV/RTV/DSV) | `VkImageView` | `id<MTLTexture>` (view) |
| [`SamplerHandle`](include/Rachis/Sampler.h) | Sampler descriptor | `VkSampler` | `id<MTLSamplerState>` |
| [`AccelerationStructureHandle`](include/Rachis/AccelerationStructure.h) | Acceleration structure buffer (`ID3D12Resource`) | `VkAccelerationStructureKHR` | Not supported ([optional features](docs/platforms.md#optional-features-by-backend)) |
| [`BindingLayoutHandle`](include/Rachis/Binding.h) | Root parameters / table layout | `VkDescriptorSetLayout` | *No native object* (slot layout) |
| [`BindingPoolHandle`](include/Rachis/Binding.h) | `ID3D12DescriptorHeap` (suballocated) | `VkDescriptorPool` | *No native object* (set and descriptor quotas) |
| [`BindingSetHandle`](include/Rachis/Binding.h) | Descriptor table + root descriptors | `VkDescriptorSet` | `id<MTL4ArgumentTable>` |
| [`PipelineLayoutHandle`](include/Rachis/Pipeline.h) | `ID3D12RootSignature` | `VkPipelineLayout` | *No native object* (slot remap and push-constant buffer) |
| [`ShaderModuleHandle`](include/Rachis/Pipeline.h) | `D3D12_SHADER_BYTECODE` (DXIL) | `VkShaderModule` (SPIR-V) | `id<MTLLibrary>` + `id<MTLFunction>` |
| [`PipelineHandle`](include/Rachis/Pipeline.h) | `ID3D12PipelineState` / `ID3D12StateObject` | `VkPipeline` | `id<MTLRenderPipelineState>` / `id<MTLComputePipelineState>` |
| [`beginRendering` / `endRendering`](include/Rachis/Rendering.h) | `OMSetRenderTargets` | `vkCmdBeginRendering` (dynamic) | `id<MTL4RenderCommandEncoder>` |
| [`textureBarrier` / `bufferBarrier` / `globalBarrier`](include/Rachis/Resource.h) | Enhanced `Barrier`, legacy `ResourceBarrier` fallback | `vkCmdPipelineBarrier2` | `barrierAfterQueueStages` / `barrierAfterStages` |
| [`QueryPoolHandle`](include/Rachis/Query.h) | `ID3D12QueryHeap` | `VkQueryPool` | `id<MTL4CounterHeap>` |
| [`SurfaceHandle`](include/Rachis/Presentation.h) | `HWND` | `VkSurfaceKHR` | `CAMetalLayer` |
| [`SwapchainHandle`](include/Rachis/Presentation.h) | `IDXGISwapChain4` | `VkSwapchainKHR` | `id<CAMetalDrawable>` from the layer |

---

## Key guarantees

| Guarantee | Behavior |
| --- | --- |
| **No backend fallback** | `createDevice` opens the exact requested backend or fails naming the missing requirement (layered Vulkan-on-D3D12 is rejected). Validation layers are the sole exception—skipped if unavailable unless `requireValidation` is set. |
| **No stale handles** | Every handle embeds a generation tag; destroyed handles and handles from another `Device` are rejected. |
| **Destroy after submit** | Submitted commands retain referenced resources until GPU completion unless the pool uses `CommandPoolFlag::unretainedResources`. On Vulkan and Metal, destroy texture views before destroying the parent texture. |
| **One device across modules** | The C11 ABI validates its version and every descriptor size strictly, and texture leases keep shared textures alive across FFI boundaries. |
| **Explicit device loss** | Device loss is permanent on that `Device`; subsequent state-mutating calls fail cleanly without partial side effects. |
| **Modeled & tested** | Lifetime, synchronization, barrier/layout rules, presentation, and device loss are specified in TLA+ (`tools/tlc.sh`) alongside Catch2 backend contracts; most models follow Direct3D 12 (see [the TLA+ specification index](formal/tla/README.md) for correspondence). |

---

## Quick start

Prerequisites: **CMake 4.0+**, **Ninja**, a **C++23** compiler, and the backend shader compilers listed in [Building](docs/building.md).

```sh
# 1. Configure a host/backend preset
cmake --preset macos-arm64-metal     # macOS, Metal
cmake --preset windows-x64-d3d12     # Windows, Direct3D 12
cmake --preset windows-x64-vulkan    # Windows, Vulkan
cmake --preset windows-x64-both      # Windows, Direct3D 12 + Vulkan
cmake --preset linux-x64-vulkan      # Linux, Vulkan

# 2. Build Debug and run small+medium tests (<preset>-debug, <preset>-small-medium)
cmake --build --preset macos-arm64-metal-debug
ctest --preset macos-arm64-metal-small-medium --output-on-failure
```

---

## CMake integration

```cmake
find_package(Rachis 0.3 CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE Rachis::RHI)  # or Rachis::C for the C11 ABI
```

- **Installed package vs. source tree:** Pass `-DCMAKE_PREFIX_PATH=<prefix>` from a Release product preset build ([Installing](docs/building.md#installing)), or embed directly via `add_subdirectory` with no Rachis tests or install rules added by default ([Using the source tree directly](docs/building.md#using-the-source-tree-directly)).
- **Shader cooking:** Use `rachis_add_shader` to compile HLSL or Slang at build time for every backend enabled in the package, or GLSL for Vulkan ([Shaders](docs/using.md#shaders) · [Examples](examples/README.md)).
- **Windows runtime DLLs:** Copy `WinPixEventRuntime.dll` (and `rachis.dll` when linking `Rachis::C`) from `<prefix>/bin` beside the executable; [Windows runtime DLLs](docs/using.md#windows-runtime-dlls) shows the CMake copy step.

---

## Documentation

| Guide | Covers |
| --- | --- |
| [Documentation index](docs/README.md) | Overview map of core guides and repository docs |
| [Platforms](docs/platforms.md) | Host support, validation rules, backend limits, and feature parity |
| [Building](docs/building.md) | Tools, Nix shell, presets, CMake options, test tiers, installing, embedding |
| [Using RachisRHI](docs/using.md) | CMake targets, device creation, errors, shaders, lifetimes, threading, queries, device loss |
| [C ABI](docs/c-abi.md) | Versioning, opaque handles, lifetimes, texture leases, surfaces, and concurrency |
| [Examples](examples/README.md) | Compute dispatch and offscreen PPM triangle across every backend |
| [Tests](tests/README.md) | Suite layout, size/hardware labels, CTest filters, and authoring rules |
| [TLA+ models](formal/tla/README.md) | 97 formal specifications, counterexample checks, and TLC runner usage |
| [Changelog](CHANGELOG.md) | Release history |

---

## License

Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Trademarks

Vulkan is a registered trademark of the Khronos Group Inc. Direct3D is a trademark of the Microsoft group of companies. Metal is a trademark of Apple Inc., registered in the U.S. and other countries and regions. RachisRHI is not affiliated with or endorsed by any of them. All other trademarks are the property of their respective owners.
