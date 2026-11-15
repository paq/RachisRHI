---------------------- MODULE VariableRateShadingTier2 ----------------------
(***************************************************************************)
(* Formal safety model for Tier 2 variable-rate shading capability         *)
(* reporting, render-pass combiner/image state resets, combiner and        *)
(* shading-rate image validation, and recording/submission image           *)
(* retention:                                                              *)
(*   - include/Rachis/Device.h (variableRateShadingTier2,                  *)
(*     shadingRateImageTileSize, setShadingRate, setShadingRateImage) and  *)
(*     include/Rachis/Rendering.h (ShadingRateCombiner)                    *)
(*   - src/D3D12/D3D12Device.cpp (detectVariableRateShading reads the      *)
(*     OPTIONS6 tier and tile size)                                        *)
(*   - src/D3D12/D3D12CommandRecording.cpp (beginRendering calls           *)
(*     resetShadingRate; PipelineManager::setShadingRate and               *)
(*     setShadingRateImage validate before recording)                      *)
(*   - src/D3D12/D3D12Resource.cpp (validateShadingRateImage at texture    *)
(*     creation; textureViewSource rejects stale handles)                  *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::retainResource keeps a      *)
(*     selected image alive until submission completion)                   *)
(*   - src/Vulkan/VulkanRuntime.cpp and src/Vulkan/VulkanGraphics.cpp      *)
(*     implement the same reporting, reset, Tier 2 gating, and retention,  *)
(*     but also reject the sum combiner and rebind the image by splitting  *)
(*     dynamic rendering                                                   *)
(*   - Metal rejects the feature: src/Metal/MetalResources.mm rejects      *)
(*     shading-rate-source textures, and setShadingRate and                *)
(*     setShadingRateImage fall back to the unsupported defaults in        *)
(*     src/Internal/BackendDevice.h                                        *)
(*                                                                         *)
(* Removes write-only state-bloating request history and 1-step before*    *)
(* copies; separates recording retention, abandonment, GPU submission,     *)
(* GPU completion, and independent physical image reclamation.             *)
(*                                                                         *)
(* Trusted boundary: hardware rasterizer per-tile rate evaluation once     *)
(* validated combiners and shading-rate images are bound.                  *)
(***************************************************************************)
EXTENDS Naturals

\* Tiers: 0 = unsupported, 1 = per-draw only, 2 = primitive/image combiners.
Tiers == {0, 1, 2}
\* Combiners: 0 = passthrough, 1..4 = override/min/max/sum, 5 = invalid.
Combiners == 0..5
ValidCombiners == 0..4
\* Image requests: 0 = unbind, 1 = valid live image, 2 = invalid desc, 3 = stale.
ImageRequests == {0, 1, 2, 3}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  nativeTier,             \* Hardware VRS tier (0..2).
  nativeTileSize,         \* Hardware shading-rate image tile size (0 or 16).
  reportedTier,           \* Publicly reported VRS tier.
  reportedTileSize,       \* Publicly reported VRS tile size.
  scopeLive,              \* Whether a render pass is currently open.
  primitiveCombiner,      \* Active primitive VRS combiner in current scope.
  imageCombiner,          \* Active image VRS combiner in current scope.
  imageBound,             \* Whether a shading-rate image is currently bound.
  logicalImageLive,       \* Public shading-rate texture handle liveness.
  physicalImageLive,      \* Native shading-rate texture storage liveness.
  recordingImageRetained, \* Whether the open command buffer retains the image.
  inFlightImageRetained,  \* Whether an in-flight submission retains the image.
  recordedWork,           \* Whether the command buffer holds recorded VRS work.
  inFlightWork,           \* Whether the command buffer is in flight on the GPU.
  lastOp,                 \* Tag of the latest VRS API attempt.
  lastOutcome             \* Outcome of the latest fallible VRS API attempt.

coreVars == <<
  nativeTier,
  nativeTileSize,
  reportedTier,
  reportedTileSize,
  scopeLive,
  primitiveCombiner,
  imageCombiner,
  imageBound,
  logicalImageLive,
  physicalImageLive,
  recordingImageRetained,
  inFlightImageRetained,
  recordedWork,
  inFlightWork
>>

vars == <<
  nativeTier,
  nativeTileSize,
  reportedTier,
  reportedTileSize,
  scopeLive,
  primitiveCombiner,
  imageCombiner,
  imageBound,
  logicalImageLive,
  physicalImageLive,
  recordingImageRetained,
  inFlightImageRetained,
  recordedWork,
  inFlightWork,
  lastOp,
  lastOutcome
>>

CanSetCombiners(primitive, image) ==
  /\ scopeLive
  /\ ~inFlightWork
  /\ reportedTier >= 1
  /\ primitive \in ValidCombiners
  /\ image \in ValidCombiners
  /\ ((primitive = 0 /\ image = 0) \/ reportedTier = 2)

CanSetImage(request) ==
  /\ scopeLive
  /\ ~inFlightWork
  /\ reportedTier = 2
  /\ (request = 0 \/ (request = 1 /\ logicalImageLive /\ physicalImageLive))

Init ==
  \E tier \in Tiers, live \in BOOLEAN :
    LET tile == IF tier = 2 THEN 16 ELSE 0 IN
    /\ nativeTier = tier
    /\ nativeTileSize = tile
    /\ reportedTier = tier
    /\ reportedTileSize = tile
    /\ scopeLive = FALSE
    /\ primitiveCombiner = 0
    /\ imageCombiner = 0
    /\ imageBound = FALSE
    /\ logicalImageLive = live
    /\ physicalImageLive = live
    /\ recordingImageRetained = FALSE
    /\ inFlightImageRetained = FALSE
    /\ recordedWork = FALSE
    /\ inFlightWork = FALSE
    /\ lastOp = "none"
    /\ lastOutcome = "none"

attemptBegin ==
  /\ lastOp' = "begin"
  /\ IF ~scopeLive /\ ~inFlightWork
     THEN /\ scopeLive' = TRUE
          /\ primitiveCombiner' = 0
          /\ imageCombiner' = 0
          /\ imageBound' = FALSE
          /\ recordedWork' = TRUE
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               nativeTier,
               nativeTileSize,
               reportedTier,
               reportedTileSize,
               logicalImageLive,
               physicalImageLive,
               recordingImageRetained,
               inFlightImageRetained,
               inFlightWork
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptSetCombiners(primitive, image) ==
  /\ primitive \in Combiners
  /\ image \in Combiners
  /\ lastOp' = "setCombiners"
  /\ IF CanSetCombiners(primitive, image)
     THEN /\ primitiveCombiner' = primitive
          /\ imageCombiner' = image
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               nativeTier,
               nativeTileSize,
               reportedTier,
               reportedTileSize,
               scopeLive,
               imageBound,
               logicalImageLive,
               physicalImageLive,
               recordingImageRetained,
               inFlightImageRetained,
               recordedWork,
               inFlightWork
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptSetImage(request) ==
  /\ request \in ImageRequests
  /\ lastOp' = "setImage"
  /\ IF CanSetImage(request)
     THEN LET binding == request = 1 IN
          /\ imageBound' = binding
          /\ IF binding
             THEN recordingImageRetained' = TRUE
             ELSE UNCHANGED recordingImageRetained
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               nativeTier,
               nativeTileSize,
               reportedTier,
               reportedTileSize,
               scopeLive,
               primitiveCombiner,
               imageCombiner,
               logicalImageLive,
               physicalImageLive,
               inFlightImageRetained,
               recordedWork,
               inFlightWork
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptEnd ==
  /\ lastOp' = "end"
  /\ IF scopeLive
     THEN /\ scopeLive' = FALSE
          /\ imageBound' = FALSE
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               nativeTier,
               nativeTileSize,
               reportedTier,
               reportedTileSize,
               primitiveCombiner,
               imageCombiner,
               logicalImageLive,
               physicalImageLive,
               recordingImageRetained,
               inFlightImageRetained,
               recordedWork,
               inFlightWork
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

abandonRecording ==
  /\ ~scopeLive
  /\ recordedWork
  /\ ~inFlightWork
  /\ primitiveCombiner' = 0
  /\ imageCombiner' = 0
  /\ recordingImageRetained' = FALSE
  /\ recordedWork' = FALSE
  /\ lastOp' = "none"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeTier,
       nativeTileSize,
       reportedTier,
       reportedTileSize,
       scopeLive,
       imageBound,
       logicalImageLive,
       physicalImageLive,
       inFlightImageRetained,
       inFlightWork
     >>

submit ==
  /\ ~scopeLive
  /\ recordedWork
  /\ ~inFlightWork
  /\ recordingImageRetained' = FALSE
  /\ inFlightImageRetained' = recordingImageRetained
  /\ recordedWork' = FALSE
  /\ inFlightWork' = TRUE
  /\ lastOp' = "none"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeTier,
       nativeTileSize,
       reportedTier,
       reportedTileSize,
       scopeLive,
       primitiveCombiner,
       imageCombiner,
       imageBound,
       logicalImageLive,
       physicalImageLive
     >>

completeSubmission ==
  /\ inFlightWork
  /\ inFlightImageRetained' = FALSE
  /\ inFlightWork' = FALSE
  /\ lastOp' = "none"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeTier,
       nativeTileSize,
       reportedTier,
       reportedTileSize,
       scopeLive,
       primitiveCombiner,
       imageCombiner,
       imageBound,
       logicalImageLive,
       physicalImageLive,
       recordingImageRetained,
       recordedWork
     >>

destroyImage ==
  /\ logicalImageLive
  /\ logicalImageLive' = FALSE
  /\ lastOp' = "none"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeTier,
       nativeTileSize,
       reportedTier,
       reportedTileSize,
       scopeLive,
       primitiveCombiner,
       imageCombiner,
       imageBound,
       physicalImageLive,
       recordingImageRetained,
       inFlightImageRetained,
       recordedWork,
       inFlightWork
     >>

collectImage ==
  /\ physicalImageLive
  /\ ~logicalImageLive
  /\ ~recordingImageRetained
  /\ ~inFlightImageRetained
  /\ physicalImageLive' = FALSE
  /\ lastOp' = "none"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeTier,
       nativeTileSize,
       reportedTier,
       reportedTileSize,
       scopeLive,
       primitiveCombiner,
       imageCombiner,
       imageBound,
       logicalImageLive,
       recordingImageRetained,
       inFlightImageRetained,
       recordedWork,
       inFlightWork
     >>

Next ==
  \/ attemptBegin
  \/ \E p, i \in Combiners : attemptSetCombiners(p, i)
  \/ \E req \in ImageRequests : attemptSetImage(req)
  \/ attemptEnd
  \/ abandonRecording
  \/ submit
  \/ completeSubmission
  \/ destroyImage
  \/ collectImage

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ nativeTier \in Tiers
  /\ nativeTileSize \in {0, 16}
  /\ reportedTier \in Tiers
  /\ reportedTileSize \in {0, 16}
  /\ scopeLive \in BOOLEAN
  /\ primitiveCombiner \in ValidCombiners
  /\ imageCombiner \in ValidCombiners
  /\ imageBound \in BOOLEAN
  /\ logicalImageLive \in BOOLEAN
  /\ physicalImageLive \in BOOLEAN
  /\ recordingImageRetained \in BOOLEAN
  /\ inFlightImageRetained \in BOOLEAN
  /\ recordedWork \in BOOLEAN
  /\ inFlightWork \in BOOLEAN
  /\ lastOp \in {"none", "begin", "setCombiners", "setImage", "end"}
  /\ lastOutcome \in Outcomes

CapabilityMappingIsExact ==
  /\ reportedTier = nativeTier
  /\ reportedTileSize = nativeTileSize
  /\ (reportedTier = 2 <=> reportedTileSize = 16)

Tier2ControlsRequireTier2Support ==
  /\ ((primitiveCombiner # 0 \/ imageCombiner # 0) => reportedTier = 2)
  /\ (imageBound => scopeLive /\ reportedTier = 2 /\ recordingImageRetained)

BeginResetsTier2State ==
  (lastOp = "begin" /\ lastOutcome = "accepted") =>
    /\ scopeLive
    /\ primitiveCombiner = 0
    /\ imageCombiner = 0
    /\ ~imageBound

BoundAndRetainedImagesRemainPhysical ==
  /\ (logicalImageLive => physicalImageLive)
  /\ (imageBound => physicalImageLive)
  /\ ((recordingImageRetained \/ inFlightImageRetained) => physicalImageLive)
  /\ ~(recordedWork /\ inFlightWork)
  /\ ~(recordingImageRetained /\ inFlightImageRetained)

Safety ==
  /\ CapabilityMappingIsExact
  /\ Tier2ControlsRequireTier2Support
  /\ BeginResetsTier2State
  /\ BoundAndRetainedImagesRemainPhysical

StepSafety ==
  [][lastOutcome' = "rejected" => coreVars' = coreVars]_vars

=============================================================================
