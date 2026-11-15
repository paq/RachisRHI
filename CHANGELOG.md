# Changelog

## 0.1.0

First release.

- C++23 API over Direct3D 12 (Windows 11 x64), Vulkan 1.3 (Windows 11 x64 and
  Linux x64), and Metal 4 (macOS 26 or later on Apple silicon).
- A C11 ABI in `rachis.h`, starting at `RACHIS_ABI_VERSION` 1, so separately
  compiled modules can share one device.
- CMake package `Rachis` with the `Rachis::RHI` and `Rachis::C` targets.
- `rachis_add_shader` in the package, which compiles an HLSL entry point for
  every backend in the package and embeds the bytecode in a header, and the
  `Rachis::ShaderBinaryTool` it uses to embed the bytecode and to package
  Metal shaders.
- `rachis_add_shader` also compiles Slang, with `slangc` on every backend, and
  GLSL, with `glslangValidator` for Vulkan only. `LANGUAGE HLSL`, `SLANG`, or
  `GLSL` picks the language, which otherwise comes from the source's extension;
  a source with any other extension now needs `LANGUAGE`. A GLSL shader's
  `REFLECT` interface comes from its SPIR-V, through the new
  `RachisShaderBinaryTool reflect-interface --spirv-reflection` option.
  [Using](docs/using.md#shader-languages) lists the rules for each language.
- A `StructuredBuffer` or `ByteAddressBuffer` in an HLSL `t` register is a
  `readOnlyStorageBuffer` binding on every backend, including Vulkan with the
  `dxc` SPIR-V that `rachis_add_shader` cooks. A shader interface that declared
  `storageBuffer` for such a binding to pass the Vulkan check is now rejected
  with `ErrorCode::invalidArgument`; declare `readOnlyStorageBuffer`, as
  Direct3D 12 and Metal already require.
- HLSL matrices in constant buffers, push constants, and structured buffers
  work on Vulkan with the `dxc` SPIR-V that `rachis_add_shader` cooks. The
  Vulkan backend accepts the `RowMajor` decoration that `dxc` writes for HLSL's
  default `column_major` packing instead of rejecting the module, and sizes a
  push-constant block by each matrix's decoration. [Using](docs/using.md#shaders)
  describes the layout of a matrix on each backend.
- `SV_VertexID` and `SV_InstanceID` count from zero in every draw on every
  backend, as on Direct3D 12, in shaders that `rachis_add_shader` and
  `RachisShaderBinaryTool cook-metallib` cook. `firstVertex`, `baseVertex`, and
  `firstInstance` stay out of them, in direct and indirect draws.
  [Using](docs/using.md#shaders) explains.
- Compute and offscreen triangle examples in [`examples/`](examples).
- A `nix develop` shell for Linux and macOS. See [Building](docs/building.md).

Until 1.0.0, the C++ API can change between minor versions. The C ABI changes
only together with its version number.
