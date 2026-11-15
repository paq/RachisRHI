# RachisRHI third-party notices

RachisRHI is licensed under the Apache License 2.0; see [LICENSE](LICENSE). The
components below are compiled into, linked into, or redistributed with the
RachisRHI package. Keep this notice, `LICENSE`, and the files in `third_party/`
with every copy of the package that you redistribute.

"The `rachis` shared library" below is the C ABI library: `rachis.dll`,
`librachis.so`, or `librachis.dylib`. The static C++ libraries are
`Rachis::RHI` and the backend libraries it links.

| Component | Pinned version | How it ships | License and notice files |
| --- | --- | --- | --- |
| [WinPixEventRuntime](https://github.com/microsoft/PixEvents) | NuGet `1.0.240308001` | Windows builds with the D3D12 backend and `RACHIS_ENABLE_DEBUG_MARKERS` (on by default). Its headers are compiled into the D3D12 backend. `WinPixEventRuntime.dll` is copied into the package's `bin` directory and its import library into the package's library directory. The `rachis` shared library, and programs linked with the static C++ libraries, load the DLL at startup, so it has to be redistributed with them. | [MIT](third_party/WinPixEventRuntime-LICENSE.txt), [NuGet third-party notices](third_party/WinPixEventRuntime-ThirdPartyNotices.txt) |
| [D3D12 Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator) | `v3.2.0` | Windows builds with the D3D12 backend. Statically linked into the `rachis` shared library. Its static library is also installed, because programs that link the static `Rachis::RHID3D12` library link it too. | [MIT](third_party/D3D12MemoryAllocator-LICENSE.txt), [upstream notices](third_party/D3D12MemoryAllocator-NOTICES.txt) |
| [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | `v3.2.1` | Builds with the Vulkan backend. Header-only; compiled into the Vulkan backend, both in the `rachis` shared library and in the static `Rachis::RHIVulkan` library. Its header is not installed. | [MIT](third_party/VulkanMemoryAllocator-LICENSE.txt) |
| [volk](https://github.com/zeux/volk) | `vulkan-sdk-1.3.296.0` | Builds with the Vulkan backend. Statically linked into the `rachis` shared library. Its static library is also installed, because programs that link the static `Rachis::RHIVulkan` library link it too. Its header is not installed. | [MIT](third_party/volk-LICENSE.md) |
| [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | `v1.3.296` | Builds with the Vulkan backend. Compiled into the Vulkan backend and into volk. Not installed. | [upstream license summary](third_party/Vulkan-Headers-LICENSE.md), [Apache-2.0 text](third_party/Apache-2.0-LICENSE.txt) |

WinPixEventRuntime's NuGet notices cover the code that it incorporates. D3D12
Memory Allocator's upstream notices cover the MIT code that it incorporates.

Every package carries all of these notices, whichever backends it was built
with. A notice alone does not mean that a package contains the component: a
Linux or macOS package, for example, contains neither D3D12 Memory Allocator
nor WinPixEventRuntime.

GPU drivers, the Vulkan loader, the validation layers, the compilers, Slang,
DXC, SPIRV-Tools, and Catch2 (used only by the tests) are development or system
dependencies. None of them ships in the package.

## Where the installed package keeps these files

`cmake --install` puts this notice, `LICENSE`, and a `third_party/` directory
with the files linked above in two places, so that the license texts remain
when a packager leaves out the documentation:

- `share/licenses/Rachis/`
- `share/doc/Rachis/`, next to `README.md` and the rest of the documentation

Both directories keep the layout of the source tree, so the links in this
notice work from either of them. The paths are relative to the install prefix,
and `share` is the default value of `CMAKE_INSTALL_DATADIR`.
