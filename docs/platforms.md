# Platforms

This page describes the host platforms RachisRHI targets, optional feature availability per backend, validation behavior, and backend-specific constraints that affect portable code.

## Contents

- [Supported hosts](#supported-hosts)
- [Optional features by backend](#optional-features-by-backend) ([Acceleration-structure operations](#acceleration-structure-operations))
- [Vulkan profile](#vulkan-profile)
- [Fallback and validation](#fallback-and-validation)
- [Direct3D 12 validation is process-wide](#direct3d-12-validation-is-process-wide)
- [Vulkan copy queues](#vulkan-copy-queues)
- [Indirect dispatch batches on Vulkan](#indirect-dispatch-batches-on-vulkan)
- [Conditional execution on Vulkan](#conditional-execution-on-vulkan)
- [Attachment clears on Metal](#attachment-clears-on-metal)
- [Scissors on Metal](#scissors-on-metal)
- [Shared resources](#shared-resources)
- [1D textures on Metal](#1d-textures-on-metal)
- [Indirect calls on Metal](#indirect-calls-on-metal)
- [Mesh shaders on Metal](#mesh-shaders-on-metal)
- [Storage texture formats on Metal](#storage-texture-formats-on-metal)
- [Variable-count binding arrays](#variable-count-binding-arrays)
- [Update-after-bind binding arrays](#update-after-bind-binding-arrays)
- [Vulkan on macOS](#vulkan-on-macos)

## Supported hosts

| Host | Backend | Runtime requirement |
| --- | --- | --- |
| Windows 11 x64 | Direct3D 12 | Direct3D 12 adapter at feature level 12.0+ (or WARP software adapter) and DXGI 1.6 |
| Windows 11 x64 | Vulkan 1.3 | Vulkan 1.3 loader (`vulkan-1.dll`) and a driver meeting the [Vulkan profile](#vulkan-profile) |
| Linux x64 | Vulkan 1.3 | Vulkan 1.3 loader (`libvulkan.so.1`) and a driver meeting the [Vulkan profile](#vulkan-profile); presentation requires Wayland or XCB (offscreen work does not) |
| macOS 26.0+ (Apple silicon) | Metal 4 | GPU in `MTLGPUFamilyMetal4` (Intel Macs are rejected) |

On macOS, RachisRHI records commands with the **Metal 4** command API and presents through an application-owned `CAMetalLayer`.

> [!NOTE]
> This table reflects what the project builds and tests today, not a policy against older OS versions:
> - **Windows 10:** Neither Windows backend checks the Windows version, though neither is tested on Windows 10.
> - **Earlier macOS / Intel Macs:** The Metal backend relies on the Metal 4 command API, so older macOS versions and Intel Macs (whose GPUs lack Metal 4) would require a separate code path.

## Optional features by backend

Backends do not implement the same optional features. Query `Device::capabilities()` for what the active device supports; calling an operation that needs an unsupported feature fails with `ErrorCode::unsupported`.

The table below shows which backends *can* report each feature (subject to adapter and driver support):

| `DeviceCapabilities` field | Direct3D 12 | Vulkan | Metal |
| --- | --- | --- | --- |
| `rayTracing`, `rayQuery` | Yes | Yes | No |
| `meshShaders`, `taskShaders` | Yes | Yes | Apple9 GPUs and later ([Mesh shaders on Metal](#mesh-shaders-on-metal)) |
| `shaderDrawIndex` ([draw index in indirect calls](using.md#draw-index-in-indirect-calls)) | Yes | Yes, with `shaderDrawParameters` | No |
| `geometryShaders`, `tessellationShaders` | Yes | Yes | No |
| `variableRateShading`, `variableRateShadingTier2` | Yes | Yes | No |
| `sparseBuffers`, `sparseTextures` | Yes | No | No |
| `samplerFeedback` | Yes | No | No |
| `conditionalExecution` | Yes | Yes, with `VK_EXT_conditional_rendering` | No |
| `multiDispatchIndirect` | Yes | Yes | No |
| `dispatchIndirectCount` | Yes | Yes, with the `bufferDeviceAddress` and `shaderInt64` features | No |
| `updateAfterBindBindingArrays` | Yes | Yes | Yes, variable-count arrays only |

On Vulkan, `multiDispatchIndirect` and `dispatchIndirectCount` are emulated; see [Indirect dispatch batches on Vulkan](#indirect-dispatch-batches-on-vulkan).

### Acceleration-structure operations

On devices that report `rayTracing` or `rayQuery`, the backends implement these acceleration-structure operations. The others fail with `ErrorCode::unsupported`. Metal has no acceleration structures.

| Operation | Direct3D 12 | Vulkan |
| --- | --- | --- |
| Bottom-level and top-level builds, including top-level builds from instance buffers | Yes | Yes |
| Top-level updates, out-of-place and in place, from spans or instance buffers | Yes | Yes |
| Bottom-level updates | Yes | No |
| Compaction, serialization, and compacted-size and serialization-info queries | Yes | No |

## Vulkan profile

### Loader and adapter filtering

RachisRHI loads the Vulkan loader dynamically at run time rather than linking against it:

- `vkEnumerateInstanceVersion` must report **Vulkan 1.3 or later**; otherwise `createDevice` and `enumerateAdapters(Backend::vulkan)` fail with `ErrorCode::backendUnavailable`.
- Drivers layered on Direct3D 12 (such as Mesa Dozen) are rejected.
- CPU implementations (`VK_PHYSICAL_DEVICE_TYPE_CPU`, such as lavapipe and SwiftShader) are rejected on every host. `enumerateAdapters(Backend::vulkan)` may still list them as `AdapterType::software`, but `createDevice` rejects them—even when requested explicitly via `DeviceDesc::adapterId`.

### Required features, formats, and limits

A Vulkan device must support Vulkan 1.3 with:

- **Core features:** dynamic rendering, `synchronization2`, timeline semaphores, independent blending, sampler anisotropy, BC texture compression, dynamically indexed sampled-image arrays, and timestamps on graphics and compute queues.
- **Formats:** `RGBA8` and `BGRA8` (with their sRGB variants), `D32`, at least one packed depth-stencil format, `RGBA16F`, and `R32` unsigned.
- **Minimum limits:** 2D images up to `8192` texels, `4` binding sets, `128` bytes of push constants, `8` color attachments, and `256` compute invocations per workgroup.

When provided by the driver, RachisRHI also enables swapchain maintenance, present wait, device fault extensions, `VK_KHR_maintenance5`, `VK_EXT_conditional_rendering` (for `conditionalExecution`), and `shaderDrawParameters` (required by shader modules declaring SPIR-V `DrawParameters`, such as vertex shaders compiled via `rachis_add_shader` that read `SV_VertexID` or `SV_InstanceID`; see [Shaders](using.md#shaders)). `DeviceCapabilities::shaderDrawIndex` is true only when the driver provides `shaderDrawParameters`.

### Native queue mapping

Each `QueueType` receives its own native Vulkan queue when the device exposes enough queues:

- **Compute queue selection order:**
  1. A second queue from the graphics family (preferred first because concurrent submissions to the graphics queue and compute-only family on an RTX 4080 can trigger device loss under NVIDIA Windows driver `621.14`).
  2. A compute family without graphics.
  3. Any remaining compute-capable family.
- **Copy queue selection order:**
  1. A transfer-only family (no graphics or compute) with `minImageTransferGranularity == (1, 1, 1)`.
  2. The graphics family.
  3. Any remaining graphics- or compute-capable family.
- **Fallback sharing:** On devices with fewer native queues than queue types (such as single-queue devices), unallocated queue types share the graphics queue. Shared queue types execute in a single submission order: work on one waits behind earlier work on the others, `waitIdle(queueType)` waits on all of them, and a shared timeline wait on one blocks the others until satisfied.

### Windows timeline point export

On Windows, `exportTimelinePoint` gives each exported point its own Vulkan timeline semaphore, allowing consumers to wait on multiple exports from one queue concurrently:

- An already-completed point starts its exported semaphore at the point's value.
- A pending point starts one below its value, and a dedicated RachisRHI background thread (spawned on first such export) signals it from the host once the point completes—so later work or shared timeline waits on that queue do not delay it.

## Fallback and validation

Device creation validates every requirement above and fails with an error naming the missing requirement. **Backends never fall back to another API**—a device either opens on the requested backend or fails.

Validation is a development tool and is **disabled by default** (both a default `DeviceDesc` and a zero-initialized `rachis_device_desc` leave it off):

| `ValidationLevel` | Direct3D 12 | Vulkan | Metal |
| --- | --- | --- | --- |
| `disabled` | Off | Off | Off |
| `standard` | D3D12 debug layer | Khronos validation + synchronization validation | `MTLLogState` forwarding shader log messages to the diagnostic callback |
| `gpu` | Adds GPU-based validation (GBV) | Adds GPU-assisted validation | `MTLLogState` at debug level |

If the requested layers or log state are not installed, device creation succeeds without them unless `DeviceDesc::requireValidation` is set, in which case device creation fails instead.

## Direct3D 12 validation is process-wide

> [!WARNING]
> On Direct3D 12, both the debug layer and GPU-based validation (`ValidationLevel::gpu`) apply **to the entire process** and remain active for every subsequent Direct3D 12 device. Configure validation before creating your first D3D12 device.

- **Enabling the debug layer removes existing devices:** Turning on the D3D12 debug layer removes every live Direct3D 12 device in the process. RachisRHI therefore refuses to enable it while a non-validated RachisRHI device is still alive—instead creating the new device without validation and reporting a warning via its diagnostic callback (or failing with `ErrorCode::validationError` if `requireValidation` is set). Once those devices are destroyed, validation can be enabled again.
- **Enabling GPU-based validation removes non-GBV devices:** Turning on GBV removes every live D3D12 device, including devices using only the standard debug layer. While such a device is alive, requesting `ValidationLevel::gpu` falls back to standard validation with a diagnostic warning (or fails with `ErrorCode::validationError` if `requireValidation` is set).
- **Sticky process-wide state:** Once enabled, the debug layer (and GBV, once enabled) stays active for the rest of the process. Later devices created with `ValidationLevel::disabled` still run under the active layer, though their own diagnostic callbacks receive no validation messages.
- **Shared native devices:** Multiple RachisRHI devices on the same adapter can share an underlying `ID3D12Device`, so debug-layer messages triggered by one device may arrive at another device's diagnostic callback on that adapter.
- **Visibility boundary:** RachisRHI tracks only the devices it created itself. It cannot see devices created directly by the application or by a second copy of RachisRHI in the process (such as a statically linked copy alongside `rachis.dll`), and enabling validation will still remove those devices.

## Vulkan copy queues

When a Vulkan device exposes a transfer-only queue family (supporting neither graphics nor compute) with `minImageTransferGranularity == (1, 1, 1)`, `QueueType::copy` uses it. Because RachisRHI does not enable `VK_KHR_maintenance10`:

- On a transfer-only copy queue (or a compute queue whose family lacks graphics support), `copyBufferToTexture` and `copyTextureToBuffer` fail with `ErrorCode::unsupported` without recording anything if any region specifies `TextureAspect::depth` or `TextureAspect::stencil`.
- Color buffer-texture copies and `copyTexture` between depth-stencil textures still work on the copy queue.
- **Portable rule:** Record depth and stencil buffer uploads/readbacks on a **graphics** command buffer, which works on every backend. (Vulkan devices without a qualifying transfer-only family fall back as noted in [Native queue mapping](#native-queue-mapping) and allow depth/stencil buffer copies when the selected family supports graphics; D3D12 and Metal accept them on `QueueType::copy`.)

## Indirect dispatch batches on Vulkan

Core Vulkan dispatches one indirect record per call and has no GPU-counted dispatch, so RachisRHI emulates both:

- **Multi-record batches** (`maxDispatchCount > 1`, no `countBuffer`): RachisRHI records one `vkCmdDispatchIndirect` per record, at `argumentOffset + index * stride`. Recording time grows with `maxDispatchCount`.
- **GPU-counted batches** (`countBuffer` set): RachisRHI records an internal compute pass before the dispatches. One thread per record (64 threads per workgroup) copies the records into a new device-local scratch buffer of `12 * maxDispatchCount` bytes and sets the group counts of every record at or past `min(count, maxDispatchCount)` to zero. RachisRHI then records `maxDispatchCount` `vkCmdDispatchIndirect` calls on the scratch records; records past the count dispatch zero workgroups.
- **Barriers:** The pass adds two barriers: indirect-argument reads to compute-shader reads before it, and compute-shader writes of the scratch buffer to indirect-argument reads after it. Work before the dispatch that writes the arguments or count still needs your own barrier to `PipelineStage::indirect`.
- **Bound state:** The pass binds its own pipeline and push constants, then rebinds your compute pipeline and pushes your compute push constants again. Your binding sets stay bound. An active conditional execution scope is paused for the pass, so the pass always runs and only your dispatches are conditional.
- **Lifetime:** The scratch buffer is freed when the command buffer completes, also for command pools with `CommandPoolFlag::unretainedResources`.
- **Queries:** The pass counts towards `computeShaderInvocations` of an active pipeline-statistics query and adds its time to timestamps around the dispatch.

## Conditional execution on Vulkan

Vulkan implements conditional execution with `VK_EXT_conditional_rendering`, which reads a 32-bit predicate. RachisRHI passes the low 32 bits of the `uint64` predicate, so a value whose low 32 bits are zero counts as zero on Vulkan, while Direct3D 12 reads all 64 bits. Write predicate values below 2^32 for the same result on every backend.

## Attachment clears on Metal

- On Metal, `clearColorAttachment` and `clearDepthStencilAttachment` must clear the **entire render target** (either pass no rectangles, or rectangles whose union covers the full target). Clearing a sub-region fails with `ErrorCode::unsupported`, whereas Direct3D 12 and Vulkan clear only the specified rectangles.
- Empty rectangles, negative offsets, or rectangles extending past the render target fail with `ErrorCode::invalidArgument` on all backends.
- A clear on Metal ends the current render pass and begins a new one, writing other attachments to memory and reading them back.

## Scissors on Metal

All backends clip scissor rectangles to the render target. However, when the clipped scissor lies **wholly outside** the render target:

- **Metal** (and Vulkan on `VK_KHR_portability_subset` devices such as MoltenVK; see [Vulkan on macOS](#vulkan-on-macos)) skips draws whose pipeline has rasterization enabled once validation passes—so the vertex shader does not run and any buffer writes from it are skipped.
- **Direct3D 12 and standard Vulkan** still submit the draw, running the vertex shader and discarding only fragments.
- Draws with rasterization disabled execute on every backend.

## Shared resources

Shared buffers and textures move between two devices using the release/acquire barriers described in [Sharing resources](using.md#sharing-resources):

| Backend | Ownership transfer behavior |
| --- | --- |
| **Vulkan** | Each RachisRHI device owns a separate `VkInstance`, so external queues map to `VK_QUEUE_FAMILY_EXTERNAL`. Release transfers from the recording queue's family to `VK_QUEUE_FAMILY_EXTERNAL` and acquire transfers back, both with `oldLayout = newLayout = VK_IMAGE_LAYOUT_GENERAL` (`VK_QUEUE_FAMILY_IGNORED` instead of the recording family when `VK_SHARING_MODE_CONCURRENT` is used). Transfers are required in both sharing modes; omitting them risks stale or compressed contents on drivers that track compression across transfers. |
| **Direct3D 12** | No ownership transfer is needed; barriers record as ordinary `common` -> `common` transitions with the specified stages and accesses. |
| **Metal** | Shared resources are not supported; every ownership transfer fails with `ErrorCode::invalidArgument`. |

## 1D textures on Metal

Metal restricts 1D textures and 1D texture arrays in two ways:

- **Single mip level:** `createTexture`, `createPlacedTexture`, and `queryTextureMemoryRequirements` fail with `ErrorCode::unsupported` (`RACHIS_STATUS_UNSUPPORTED` in `rachis_texture_create`) if `TextureDimension::texture1D` (`RACHIS_TEXTURE_1D`) has `mipLevels > 1`. `generateMipmaps` fails with `ErrorCode::unsupported` on every 1D texture. `DeviceLimits::maxTextureMipLevels` applies only to 2D and 3D textures on Metal.
- **No attachment usage:** Metal cannot render to 1D textures. Descriptions with `TextureUsage::renderTarget` (`RACHIS_TEXTURE_RENDER_TARGET`) or `TextureUsage::depthStencil` (`RACHIS_TEXTURE_DEPTH_STENCIL`) fail with `ErrorCode::unsupported`, and `colorAttachment` / `depthStencilAttachment` bits in `queryFormatProperties` do not cover 1D textures. Sampled, storage, and copy usages work.

> [!TIP]
> Direct3D 12 and Vulkan (including MoltenVK) allow mipmapped and renderable 1D textures. For portable mipmapped or renderable 1D textures across all backends, use a **2D texture with `height = 1`**.

## Indirect calls on Metal

Metal has no multi-record indirect calls and no GPU-written counts, so RachisRHI builds them from single-record calls. `multiDrawIndirect`, `drawIndirectCount`, `multiDispatchIndirect`, and `dispatchIndirectCount` are all reported, and the results match Direct3D 12 and Vulkan.

- **Several records:** `drawIndirect`, `drawIndexedIndirect`, `dispatchIndirect`, and `dispatchMeshIndirect` encode one Metal call per record, `maxDrawCount` or `maxDispatchCount` of them. The CPU cost of recording grows with that maximum, not with the count the GPU writes.
- **Count buffer:** before those calls, RachisRHI runs a small internal pass that copies the records into scratch memory and sets the instance or workgroup count of every record at or past the count to zero. Inside a render pass this pass is an object-shader draw with rasterization disabled; in a compute pass it is a compute dispatch. Each such call allocates `maxDrawCount` (or `maxDispatchCount`) times the record size of scratch memory, freed when the command buffer completes.
- Keep `maxDrawCount` and `maxDispatchCount` close to the real upper bound on Metal.

## Mesh shaders on Metal

Metal reports `meshShaders` and `taskShaders` on Apple9 GPUs (M3, A17 Pro) and later, the first family whose mesh draws can read their counts from a buffer. Task shaders run as Metal object shaders.

- **Limits:** `DeviceCapabilities::meshShader` reports a task payload of `16,352` bytes (Metal's 16 KB less what Metal can reserve), 256 output vertices and 512 output primitives per mesh workgroup, and `1,048,575` mesh workgroups per dispatch. A task workgroup can launch at most `1,048,575` mesh workgroups in total, a lower bound than Direct3D 12's.
- **Writing outputs:** `slangc` compiles Metal mesh shaders only when each output vertex and primitive is written as a whole struct. Fill a local struct and assign it (`vertices[i] = vertex;`) instead of writing members (`vertices[i].position = ...`). This form works on every backend.
- **Stage linkage:** Metal's pipeline compiler checks that the pixel shader's inputs match the mesh shader's outputs. A mismatch still fails `createGraphicsPipeline` with `ErrorCode::validationError`, but the message comes from Metal.

## Storage texture formats on Metal

Metal binds storage textures with `read_write` access, which Apple supports only on formats included in the device's read-write texture tier:

- **Tier 1:** `r32Float`, `r32Uint`, `r32Sint`
- **Tier 2 (Apple silicon):** adds `rgba32Float`, `rgba32Uint`, `rgba32Sint`, `rgba16Float`, `rgba16Uint`, `rgba16Sint`, `rgba8Unorm`, `rgba8Uint`, `rgba8Sint`, `r16Float`, `r16Uint`, `r16Sint`, `r8Unorm`, `r8Uint`, `r8Sint`

On Metal:
- `queryFormatProperties` reports `storageRead` and `storageWrite` only for those formats.
- Other formats (including `rg8Unorm`, `rg16Float`, `rg32Float`, `bgra8Unorm`) fail `createTexture`, `createPlacedTexture`, `queryTextureMemoryRequirements`, `createTextureView` (for storage views), and `rachis_texture_create` with `ErrorCode::unsupported` (`RACHIS_STATUS_UNSUPPORTED`) when requesting `TextureUsage::storage`.
- **sRGB and mipmap generation:** sRGB formats have no direct storage support. Like Direct3D 12, `rgba8UnormSrgb` allows `TextureUsage::storage` (and thus `generateMipmaps`) if `TextureDesc::viewFormats` includes `rgba8Unorm`, using that linear format for storage views. Because `bgra8Unorm` lacks Metal storage support, neither `bgra8Unorm` nor `bgra8UnormSrgb` supports `generateMipmaps` on Metal. Use `rgba8Unorm` or `rgba8UnormSrgb` for portable mipmapped color textures.

## Variable-count binding arrays

A binding layout entry with `BindingFlag::variableDescriptorCount` gets its size per set from `BindingSetDesc::variableDescriptorCount`. `DeviceCapabilities::limits` exposes separate maximums by binding kind:

| Limit field | Applies to | Direct3D 12 | Vulkan | Metal |
| --- | --- | --- | --- | --- |
| `maxVariableBufferDescriptors` | `uniformBuffer`, `readOnlyStorageBuffer`, `storageBuffer` | `868,928` | Min of resource limit and stage/set uniform-buffer quota (e.g., `12` when stage limit is `15`) | `500,000` |
| `maxVariableSamplerDescriptors` | `sampler` | — | Device sampler quota | `1,024` |
| `maxVariableResourceDescriptors` | Other types (`sampledTexture`, `storageTexture`, etc.) | `868,928` | Device resource quota (e.g., `65,536`) | `500,000` |

Rules and caveats:
- **Shared capacity:** A set reaches the full limit only when the variable-count entry is alone in its layout; other entries in the layout consume slots from the same argument table or heap.
- **Metal argument tables and argument buffers:** Each Metal shader stage has `31` buffer, `128` texture, and `16` sampler slots, shared by every set visible to that stage, one slot for push constants (if visible to that stage), and vertex buffers in the vertex stage. A fixed-size binding at slot `b` of count `c` uses slots `b` to `b + c - 1`. A variable-count array instead lives in an argument buffer that takes one buffer slot, so its length is limited only by the table above, whatever its binding number. RachisShaderBinaryTool `cook-metallib` (and so `rachis_add_shader`) compiles an unbounded HLSL or Slang array, such as `Texture2D textures[] : register(t0, space1)`, to read that argument buffer; the shader source stays the same on every backend.
- **Size buffer arrays correctly:** On D3D12, buffers and other resources share the `868,928`-descriptor heap range (`maxBindingPoolResourceDescriptors`), whereas on Vulkan and Metal `maxVariableBufferDescriptors` can be much smaller than `maxVariableResourceDescriptors`. Sizing a buffer array from `maxVariableResourceDescriptors` causes `createBindingSet` to fail with `ErrorCode::invalidArgument` on Metal or on Vulkan devices with tighter uniform-buffer limits. **Always size buffer arrays from `maxVariableBufferDescriptors`.**
- The C ABI exposes neither variable-count binding arrays nor device limits.

## Update-after-bind binding arrays

A binding layout entry with `BindingFlag::updateAfterBind` changes its elements in place (see [Update-after-bind binding arrays](using.md#update-after-bind-binding-arrays)). `DeviceCapabilities::updateAfterBindBindingArrays` reports whether the device supports it:

| Backend | Reported when | Limits that apply |
| --- | --- | --- |
| Direct3D 12 | Always | The usual shader-visible heap budget (`maxBindingPoolResourceDescriptors`). Descriptor ranges of these entries are volatile in the root signature, so the GPU reads each descriptor when a shader uses it. |
| Vulkan | The device supports `descriptorBindingPartiallyBound`, `descriptorBindingUpdateUnusedWhilePending`, and `descriptorBindingSampledImageUpdateAfterBind`, `descriptorBindingStorageImageUpdateAfterBind`, and `descriptorBindingStorageBufferUpdateAfterBind` | See below |
| Metal | Always | Only on variable-count entries, which live in an argument buffer (see [Variable-count binding arrays](#variable-count-binding-arrays)). The flag on a fixed-size entry fails `createBindingLayout` with `ErrorCode::unsupported`. |

On Vulkan:
- **Pipeline layouts:** When a pipeline layout holds a set with an update-after-bind entry, the descriptors of all its sets must fit the device's update-after-bind limits (`maxPerStageDescriptorUpdateAfterBind*`, `maxPerStageUpdateAfterBindResources`, and `maxDescriptorSetUpdateAfterBind*`), and the sets without such an entry must still fit the regular limits. `createPipelineLayout` fails with `ErrorCode::unsupported` otherwise.
- **Variable-count arrays:** An update-after-bind variable-count entry gets its capacity from the update-after-bind limits, capped by the `DeviceLimits` maximums in [Variable-count binding arrays](#variable-count-binding-arrays).
- **Binding pools:** The first set with an update-after-bind entry in a binding pool adds a second native descriptor pool, created with `VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT` and sized like the first. Its descriptors count against the device's `maxUpdateAfterBindDescriptorsInAllPools`.

## Vulkan on macOS

> [!NOTE]
> The `macos-arm64-vulkan` preset builds the Vulkan backend against **MoltenVK** for investigation only and is **not** a supported production configuration.

Known MoltenVK / `VK_KHR_portability_subset` caveats and workarounds:

- **Timestamp queries:** Resolved timestamps can intermittently read `0` even after submission completion, so Vulkan timestamp tests verify ordering on macOS only when all stamps were written.
- **Clock calibration:** MoltenVK reports `CLOCK_MONOTONIC_RAW` (rather than `CLOCK_MONOTONIC`) as its calibratable host time domain, so `calibrateQueueClock` uses that clock on macOS; the Vulkan calibration test checks field validity without comparing resolved timestamps against calibration.
- **Empty scissor handling:** MoltenVK clamps out-of-bounds scissor origins into the render area, which would otherwise light a corner pixel. On `VK_KHR_portability_subset` devices, RachisRHI clips scissors to the render area itself, passes empty scissors as a zero-sized rectangle at `(0, 0)`, and skips rasterized draws and mesh dispatches when every scissor is empty (matching [Scissors on Metal](#scissors-on-metal); the C ABI scissor test allows at most one lit pixel on macOS Vulkan).
- **Barrier source stage workaround (MoltenVK 1.4.2):** With Metal argument buffers, MoltenVK 1.4.2 drops execution-dependency chains (a barrier waits only on its own source stages, risking races like a copy waiting on a pixel shader before an earlier compute `generateMipmaps` finishes). On `VK_KHR_portability_subset` devices, RachisRHI records all texture, buffer, global, and alias-activation barriers with `VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT` as the source stage mask regardless of `stagesBefore`.
- **Dynamic buffer pipeline translation workaround (MoltenVK 1.4.2):** With Metal argument buffers, MoltenVK 1.4.2 reuses the Metal translation of a SPIR-V module across pipelines even when layouts differ in dynamic buffers, causing dynamic offsets to read from `0`. On `VK_KHR_portability_subset` devices, pipelines whose layouts contain a dynamic buffer are built from private copies of their shader modules without the native pipeline cache (`PipelineCacheKey` is still tracked, but MoltenVK compiles those shaders on every creation).
- **Vertex stride alignment:** When `VK_KHR_portability_subset` reports `minVertexInputBindingStrideAlignment`, graphics pipelines with non-multiple vertex strides fail with `ErrorCode::unsupported`.
