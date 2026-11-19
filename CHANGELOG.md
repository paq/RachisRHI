# Changelog

## Unreleased

- The C ABI is at `RACHIS_ABI_VERSION` 3, and the shared library is
  `librachis.so.3` and `librachis.3.dylib`. Modules built against version 2
  must be rebuilt. [C ABI](docs/c-abi.md) describes what version 3 adds:
  - `rachis_command_draw_indirect`, `rachis_command_draw_indexed_indirect`,
    `rachis_command_dispatch_indirect`, and
    `rachis_command_dispatch_mesh_indirect`, with several records, a count
    buffer, and the draw-index prefix, and structs for the records.
  - Task and mesh shaders: `task_shader` and `mesh_shader` in
    `rachis_graphics_pipeline_desc`, `rachis_command_dispatch_mesh`, and the
    `RACHIS_SHADER_TASK` and `RACHIS_SHADER_MESH` stage bits, which
    `RACHIS_SHADER_ALL` now includes.
  - `draw_index` in `rachis_shader_module_desc` and
    `rachis_pipeline_layout_desc`.
  - Binding arrays: `flags` in `rachis_binding_layout_entry` (partially bound,
    variable count, update after bind), `variable_descriptor_count` in
    `rachis_binding_set_desc`, and `rachis_binding_set_update`.
  - `rachis_command_fill_buffer` and `rachis_command_clear_storage_texture`.
  - `required_subgroup_size` in `rachis_compute_pipeline_desc`.
  - `rachis_device_capabilities` reports the indirect-call, mesh-shader,
    binding-array, 64-bit atomic, subgroup, and conditional-execution
    capabilities, the mesh-shader limits, the variable-count binding limits,
    and `max_draw_indirect_count`.
- Multi-record indirect calls can give each record a draw index that shaders
  read. Set `drawIndexPrefix` in `IndirectDrawDesc` or
  `IndirectMeshDispatchDesc` and start each record with a `uint32` equal to its
  position, set `PipelineLayoutDesc::drawIndex`, and call `rachisDrawIndex()`
  from `#include "Rachis/DrawIndex.hlsli"` in HLSL or Slang.
  `DeviceCapabilities::shaderDrawIndex` reports support, on every backend.
  `rachis_add_shader` passes the RachisRHI include directory to every HLSL and
  Slang compiler run, and `RachisShaderBinaryTool reflect-interface` reads from
  the bytecode whether a vertex, task, or mesh entry point reads the draw
  index, now also from a Metal manifest with `--bytecode metal=<shader.pnms>`.
  Register space 5 is reserved for the draw index.
  [Using](docs/using.md#draw-index-in-indirect-calls) explains.
- `fillBuffer` writes a 32-bit value to a buffer range and
  `clearStorageTexture` clears a storage-texture view, on the GPU and in
  command-buffer order, on every backend. Both count as compute-stage
  unordered-access writes for barriers.
  [Using](docs/using.md#buffer-fills-and-storage-texture-clears) lists the
  rules.
- `DeviceCapabilities::shaderInt64AtomicStorageBuffer` reports 64-bit atomics
  on storage buffers: on Direct3D 12 with shader model 6.6 and 64-bit atomics
  on descriptor-heap resources, and on Vulkan with `shaderBufferInt64Atomics`,
  which RachisRHI now enables. Metal reports no
  64-bit atomics. Vulkan's `createShaderModule` accepts such shaders on devices
  that report it instead of rejecting them all, and no longer counts a 32-bit
  atomic as a 64-bit one because the module also declares 64-bit atomics.
- `rachis_add_shader(... SHADER_MODEL 6_6)` builds a shader for a later shader
  model, from `6_0`, the default, to `6_6`, for every backend.
  `RachisShaderBinaryTool cook-metallib` takes the matching `--profile`.
  [Using](docs/using.md#shader-model) explains.
- `ComputePipelineDesc::requiredSubgroupSize` makes a compute pipeline run full
  subgroups of one size. `SubgroupCapabilities::requiredSizeStages` reports
  support. On Direct3D 12 the shader must declare the same `[WaveSize]`;
  Vulkan uses `requiredSubgroupSize` with full subgroups; Metal accepts only
  32. [Using](docs/using.md#required-subgroup-size) lists the checks.
- `BindingFlag::updateAfterBind` lets `updateBindingSet` change elements of a
  large binding array in place while work in flight uses the set, without
  copying the whole set into a new version. The application promises not to
  change elements that such work reads. `DeviceCapabilities::updateAfterBindBindingArrays`
  reports support: always on Direct3D 12, with the update-after-bind
  descriptor-indexing features on Vulkan, and for variable-count entries only
  on Metal. An element written in place stays alive, and on Direct3D 12
  resident, for every command buffer that bound the set before the write,
  until it completes. On Direct3D 12 such a write cannot name a sparse
  resource or a swapchain texture.
  [Using](docs/using.md#update-after-bind-binding-arrays) explains.
- Top-level acceleration structures build and update from instance records
  that the GPU writes into a buffer: `buildTopLevelAccelerationStructureFromBuffer`,
  `updateTopLevelAccelerationStructureFromBuffer`, and
  `updateTopLevelAccelerationStructureInPlaceFromBuffer`, with
  `queryAccelerationStructureInstanceLayout` for the record layout
  (`AccelerationStructureInstanceRecord`) and
  `queryAccelerationStructureReference` for the 64-bit reference a record
  stores for a bottom-level structure. On Direct3D 12 and Vulkan. A command
  buffer may read a top-level structure and then, after a barrier, update it
  in place.
- Vulkan implements top-level acceleration-structure updates, out of place and
  in place.
- Vulkan reports `conditionalExecution` on devices with
  `VK_EXT_conditional_rendering`, and `multiDispatchIndirect` and
  `dispatchIndirectCount`, which it emulates with one dispatch per record and,
  for a count buffer, an internal compute pass.
  [Platforms](docs/platforms.md#indirect-dispatch-batches-on-vulkan) describes
  the cost. Inside a conditional-execution scope, `fillBuffer`,
  `clearStorageTexture`, and the top-level builds and updates from instance
  buffers fail with `ErrorCode::invalidState`, as copies already do.
- `DeviceLimits::maxDrawIndirectCount` reports the largest count a counted
  draw may read from its count buffer. Vulkan requires the GPU-written count
  to stay within the device's limit; Direct3D 12 and Metal report no limit.
- Metal:
  - Mesh and task shaders on Apple9 GPUs (M3, A17 Pro) and later, with
    `dispatchMesh` and `dispatchMeshIndirect`. Mesh shaders written in Slang
    must write each output vertex and primitive as a whole struct. A
    `dispatchMeshIndirect` with more than one record needs a task or mesh
    shader that reads a binding, push constants, or the draw index, and fails
    with `ErrorCode::unsupported` otherwise.
    [Platforms](docs/platforms.md#mesh-shaders-on-metal) lists the limits.
  - `multiDrawIndirect`, `drawIndirectCount`, `multiDispatchIndirect`, and
    `dispatchIndirectCount`, built from one Metal call per record and a small
    internal pass for count buffers. Recording cost grows with `maxDrawCount`
    and `maxDispatchCount`; [Platforms](docs/platforms.md#indirect-calls-on-metal)
    explains.
  - Variable-count binding arrays live in argument buffers, so
    `maxVariableBufferDescriptors` and `maxVariableResourceDescriptors` are
    500,000, and `maxVariableSamplerDescriptors` is 1,024 on Apple9 GPUs and
    996 on Apple7 and Apple8, instead of the argument-table slot counts.
    `RachisShaderBinaryTool cook-metallib` rewrites the unbounded arrays that
    `slangc` emits for Metal.
- [Using](docs/using.md#per-record-state) records that indirect calls do not
  change bindings or pipelines from one record to the next, and how to reach
  per-record data with binding arrays and the draw index instead.
- Fixes:
  - Vulkan keeps a bottom-level acceleration structure's storage range
    reserved while a top-level structure that references it is alive.
    Before, a new structure could be placed over it after the bottom-level
    handle was destroyed.
  - Metal keeps a texture's placed-heap range reserved while a binding set,
    a render attachment, or a recorded command holds one of its views.
  - Vulkan never gives the compute queue role to a queue family without
    compute, and prefers a graphics family that also computes.
  - Vulkan accepts a driver that reports a zero scratch size for building or
    updating an acceleration structure.
  - `rachis_add_shader(... REFLECT)` rebuilds the reflected interface when a
    file the shader includes changes.

## 0.2.0

- The C ABI is at `RACHIS_ABI_VERSION` 2, and the shared library is
  `librachis.so.2` and `librachis.2.dylib`. Modules built against version 1
  must be rebuilt. [C ABI](docs/c-abi.md) describes what version 2 adds:
  - `rachis_buffer_map` and `rachis_buffer_unmap`, with
    `rachis_buffer_flush_mapped_ranges` and
    `rachis_buffer_invalidate_mapped_ranges` for memory that is not host
    coherent. A mapped pointer needs no device lock and stays valid until the
    buffer is unmapped, also after device loss.
  - Vertex buffer layouts and `strip_index_format` in
    `rachis_graphics_pipeline_desc`, and `vertex_input_locations` in
    `rachis_shader_module_desc`.
  - `rachis_command_bind_vertex_buffers`, `rachis_command_bind_index_buffer`,
    and `rachis_command_draw_indexed`.
  - `rachis_device_query_capabilities`, which reports unified memory,
    persistent mapping, mapped-memory coherence, and the vertex-input limits.
- Mapped buffers now follow the same rules on every backend:
  - On Metal, `destroyBuffer` fails with `ErrorCode::invalidState` while the
    buffer is mapped, and `flushMappedMemoryRanges` and
    `invalidateMappedMemoryRanges` check the buffer's memory type and that
    every range lies inside the mapping, as Direct3D 12 and Vulkan already do.
  - On Vulkan and Metal, an `unmapBuffer` range must lie inside the mapping, as
    on Direct3D 12. Vulkan flushes only the mapped range when it unmaps the
    whole buffer.
  - On Direct3D 12 and Metal, `flushMappedMemoryRanges` and
    `invalidateMappedMemoryRanges` return `ErrorCode::deviceLost` on a lost
    device, as on Vulkan.
  - On Vulkan, `unmapBuffer` on a lost device skips the flush, so it cannot
    fail and leave a buffer that can never be destroyed.

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
