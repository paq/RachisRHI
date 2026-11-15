-------------------- MODULE SurfaceCapabilityNegotiation --------------------
(***************************************************************************)
(* Formal safety model for surface capability discovery, swapchain         *)
(* descriptor validation, and native surface retention across handle       *)
(* destruction:                                                            *)
(*   - include/Rachis/Presentation.h (SurfaceCapabilities,                 *)
(*     supportsPresentMode)                                                *)
(*   - src/D3D12/D3D12Swapchain.cpp (capabilitiesForSurface,               *)
(*     validateSwapchainDescForSurface, destroySurface)                    *)
(*   - src/Vulkan/VulkanSwapchain.cpp (capabilitiesFor,                    *)
(*     validateSwapchainDesc, destroySurface)                              *)
(*                                                                         *)
(* Models universal FIFO vs tearing-gated immediate presentation, format/  *)
(* color-space/usage/extent/image-count validation, independent native     *)
(* surface reclamation (surfacePhysical), destroySurface rejecting while a *)
(* swapchain still claims the surface, and failure-atomic rejection.       *)
(*                                                                         *)
(* Trusted boundary: host OS window lifetime (caller obligation) and       *)
(* native DXGI/WSI surface capability queries.                             *)
(***************************************************************************)
EXTENDS Naturals

Formats == {"bgra8", "rgba8", "hdr10"}
ColorSpaces == {"srgb", "extended", "hdr10"}
PresentModes == {"fifo", "immediate", "invalid"}
Usages == {"render", "render-copy", "storage", "copy-only"}
Extents == {0, 64, 20000}
ImageCounts == {1, 2, 3, 17}
Outcomes == {"none", "accepted", "rejected"}
MaxExtent == 16384
MinImages == 2
MaxImages == 16

VARIABLES
  tearingSupported,    \* Whether the display/backend supports tearing (immediate mode).
  surfaceLogical,      \* Whether the public surface handle is still published.
  surfacePhysical,     \* Whether the underlying native surface object is alive.
  capabilitiesQueried, \* Whether a valid surface capability snapshot was queried.
  swapchainLive,       \* Whether a swapchain created from the surface is live.
  swapchainFormat,     \* Format of the live swapchain.
  swapchainColorSpace, \* Color space of the live swapchain.
  swapchainPresentMode,\* Present mode of the live swapchain.
  swapchainUsage,      \* Usage flags of the live swapchain.
  swapchainExtent,     \* Width/height extent of the live swapchain.
  swapchainImageCount, \* Back-buffer count of the live swapchain.
  lastOutcome          \* Outcome of the latest fallible query or creation call.

coreVars == <<
  tearingSupported,
  surfaceLogical,
  surfacePhysical,
  capabilitiesQueried,
  swapchainLive,
  swapchainFormat,
  swapchainColorSpace,
  swapchainPresentMode,
  swapchainUsage,
  swapchainExtent,
  swapchainImageCount
>>

vars == <<
  tearingSupported,
  surfaceLogical,
  surfacePhysical,
  capabilitiesQueried,
  swapchainLive,
  swapchainFormat,
  swapchainColorSpace,
  swapchainPresentMode,
  swapchainUsage,
  swapchainExtent,
  swapchainImageCount,
  lastOutcome
>>

FormatSupported(format, colorSpace) ==
  (format = "bgra8" \/ format = "rgba8") /\ colorSpace = "srgb"

PresentModeSupported(mode) ==
  mode = "fifo" \/ (mode = "immediate" /\ tearingSupported)

UsageSupported(usage) ==
  usage = "render" \/ usage = "render-copy"

ExtentSupported(extent) ==
  extent > 0 /\ extent <= MaxExtent

ImageCountSupported(imageCount) ==
  imageCount >= MinImages /\ imageCount <= MaxImages

DescriptorSupported(format, colorSpace, mode, usage, extent, imageCount) ==
  /\ surfaceLogical
  /\ FormatSupported(format, colorSpace)
  /\ PresentModeSupported(mode)
  /\ UsageSupported(usage)
  /\ ExtentSupported(extent)
  /\ ImageCountSupported(imageCount)

Init ==
  \E tearing \in BOOLEAN :
    /\ tearingSupported = tearing
    /\ surfaceLogical = TRUE
    /\ surfacePhysical = TRUE
    /\ capabilitiesQueried = FALSE
    /\ swapchainLive = FALSE
    /\ swapchainFormat = "bgra8"
    /\ swapchainColorSpace = "srgb"
    /\ swapchainPresentMode = "fifo"
    /\ swapchainUsage = "render"
    /\ swapchainExtent = 64
    /\ swapchainImageCount = 2
    /\ lastOutcome = "none"

attemptQueryCapabilities ==
  IF surfaceLogical
  THEN /\ capabilitiesQueried' = TRUE
       /\ lastOutcome' = "accepted"
       /\ UNCHANGED <<
            tearingSupported,
            surfaceLogical,
            surfacePhysical,
            swapchainLive,
            swapchainFormat,
            swapchainColorSpace,
            swapchainPresentMode,
            swapchainUsage,
            swapchainExtent,
            swapchainImageCount
          >>
  ELSE /\ lastOutcome' = "rejected"
       /\ UNCHANGED coreVars

attemptCreateSwapchain(format, colorSpace, mode, usage, extent, imageCount) ==
  IF ~swapchainLive
     /\ DescriptorSupported(format, colorSpace, mode, usage, extent, imageCount)
  THEN /\ swapchainLive' = TRUE
       /\ swapchainFormat' = format
       /\ swapchainColorSpace' = colorSpace
       /\ swapchainPresentMode' = mode
       /\ swapchainUsage' = usage
       /\ swapchainExtent' = extent
       /\ swapchainImageCount' = imageCount
       /\ lastOutcome' = "accepted"
       /\ UNCHANGED <<
            tearingSupported,
            surfaceLogical,
            surfacePhysical,
            capabilitiesQueried
          >>
  ELSE /\ lastOutcome' = "rejected"
       /\ UNCHANGED coreVars

destroySurface ==
  /\ surfaceLogical
  /\ ~swapchainLive
  /\ surfaceLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       tearingSupported,
       surfacePhysical,
       capabilitiesQueried,
       swapchainLive,
       swapchainFormat,
       swapchainColorSpace,
       swapchainPresentMode,
       swapchainUsage,
       swapchainExtent,
       swapchainImageCount
     >>

collectSurface ==
  /\ surfacePhysical
  /\ ~surfaceLogical
  /\ ~swapchainLive
  /\ surfacePhysical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       tearingSupported,
       surfaceLogical,
       capabilitiesQueried,
       swapchainLive,
       swapchainFormat,
       swapchainColorSpace,
       swapchainPresentMode,
       swapchainUsage,
       swapchainExtent,
       swapchainImageCount
     >>

destroySwapchain ==
  /\ swapchainLive
  /\ swapchainLive' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       tearingSupported,
       surfaceLogical,
       surfacePhysical,
       capabilitiesQueried,
       swapchainFormat,
       swapchainColorSpace,
       swapchainPresentMode,
       swapchainUsage,
       swapchainExtent,
       swapchainImageCount
     >>

Next ==
  \/ attemptQueryCapabilities
  \/ \E f \in Formats, cs \in ColorSpaces, m \in PresentModes,
        u \in Usages, ext \in Extents, cnt \in ImageCounts :
       attemptCreateSwapchain(f, cs, m, u, ext, cnt)
  \/ destroySurface
  \/ collectSurface
  \/ destroySwapchain

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ tearingSupported \in BOOLEAN
  /\ surfaceLogical \in BOOLEAN
  /\ surfacePhysical \in BOOLEAN
  /\ capabilitiesQueried \in BOOLEAN
  /\ swapchainLive \in BOOLEAN
  /\ swapchainFormat \in Formats
  /\ swapchainColorSpace \in ColorSpaces
  /\ swapchainPresentMode \in PresentModes
  /\ swapchainUsage \in Usages
  /\ swapchainExtent \in Extents
  /\ swapchainImageCount \in ImageCounts
  /\ lastOutcome \in Outcomes

LiveSwapchainMatchesCapabilities ==
  swapchainLive =>
    /\ FormatSupported(swapchainFormat, swapchainColorSpace)
    /\ PresentModeSupported(swapchainPresentMode)
    /\ UsageSupported(swapchainUsage)
    /\ ExtentSupported(swapchainExtent)
    /\ ImageCountSupported(swapchainImageCount)

NativeSurfaceRetainedByOwners ==
  (surfaceLogical \/ swapchainLive) => surfacePhysical

Safety ==
  /\ LiveSwapchainMatchesCapabilities
  /\ NativeSurfaceRetainedByOwners

StepSafety ==
  [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
     /\ (~surfaceLogical => ~(~swapchainLive /\ swapchainLive'))
     /\ (surfaceLogical /\ ~surfaceLogical' => ~swapchainLive)]_vars

=============================================================================
