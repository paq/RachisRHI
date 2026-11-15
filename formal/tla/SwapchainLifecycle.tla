------------------------- MODULE SwapchainLifecycle -------------------------
(***************************************************************************)
(* Formal safety model for D3D12 swapchain acquisition, borrowed-image     *)
(* cancellation, post-present timeline markers, surface-out-of-date        *)
(* refresh, native presentation poisoning, and terminal device loss:       *)
(*   - src/D3D12/D3D12Swapchain.cpp (waitForBackBuffer, acquireNextImage,  *)
(*     cancelSwapchainImage, present, resizeSwapchain, destroySwapchain)   *)
(*   - src/D3D12/D3D12SwapchainLease.h (SwapchainImageLeaseState cancel    *)
(*     and presentation-point validation)                                  *)
(*   - src/D3D12/D3D12Queue.cpp (reserveSignal post-present markers),      *)
(*     src/D3D12/D3D12Device.cpp (terminal device-loss observation)        *)
(*                                                                         *)
(* Aligns with D3D12Swapchain.cpp:                                         *)
(*   - waitForBackBuffer device removal (during acquire, resize, or        *)
(*     destroySwapchain) records terminal deviceLost without requiring     *)
(*     swapchain poisoning;                                                *)
(*   - ResizeBuffers device removal releases back buffers and sets         *)
(*     poisoned = TRUE /\ deviceLost = TRUE while preserving any           *)
(*     pre-existing requiresRefresh flag;                                  *)
(*   - deviceLost is monotonic across destroySwapchain (never reset);      *)
(*   - accepted presents (ok or surfaceOutOfDate) emit a post-present      *)
(*     marker strictly past renderPoint, while device-loss present         *)
(*     clears lastUse instead of claiming renderPoint retired DXGI.        *)
(*                                                                         *)
(* Trusted boundary: DXGI flip-model presentation queue and OS HWND.       *)
(***************************************************************************)
EXTENDS Integers

Images == {0, 1}
NoImage == -1
ImageOrNone == Images \cup {NoImage}
PresentOutcomes == {"none", "ok", "outOfDate", "failedBeforeDxgi", "deviceLost"}
MaxGeneration == 3
MaxTimeline == 4

VARIABLES
  surfaceLogical,        \* Whether the public surface handle is live.
  surfacePhysical,       \* Whether the native surface object is physically alive.
  swapchainAlive,        \* Whether the public swapchain handle is live.
  generation,            \* Monotonic swapchain resize/creation generation (0..3).
  imageGeneration,       \* Per-image generation of live borrowed handles.
  imageHandleAlive,      \* Per-image borrowed handle validity.
  acquiredImage,         \* Currently acquired back-buffer index or NoImage.
  recordedReference,     \* Whether an unsubmitted recording references acquiredImage.
  renderPoint,           \* Submitted render timeline point awaiting present (0..4).
  submitted,             \* Graphics queue submitted timeline frontier (0..4).
  completed,             \* Graphics queue completed timeline frontier (0..4).
  lastUse,               \* Per-image timeline point required before reuse/retire.
  poisoned,              \* Whether native present/resize failed terminally.
  deviceLost,            \* Monotonic terminal device-loss state.
  requiresRefresh,       \* Whether DXGI_STATUS_MODE_CHANGED requires resize.
  lastPresentOutcome,    \* Outcome of the latest present attempt.
  lastPresentRenderPoint,\* renderPoint supplied to the latest present attempt.
  lastPresentMarker      \* Post-present timeline marker emitted (0 on failure/loss).

vars == <<
  surfaceLogical,
  surfacePhysical,
  swapchainAlive,
  generation,
  imageGeneration,
  imageHandleAlive,
  acquiredImage,
  recordedReference,
  renderPoint,
  submitted,
  completed,
  lastUse,
  poisoned,
  deviceLost,
  requiresRefresh,
  lastPresentOutcome,
  lastPresentRenderPoint,
  lastPresentMarker
>>

AllImagesComplete ==
  \A img \in Images : lastUse[img] <= completed

Init ==
  /\ surfaceLogical = FALSE
  /\ surfacePhysical = FALSE
  /\ swapchainAlive = FALSE
  /\ generation = 0
  /\ imageGeneration = [img \in Images |-> 0]
  /\ imageHandleAlive = [img \in Images |-> FALSE]
  /\ acquiredImage = NoImage
  /\ recordedReference = FALSE
  /\ renderPoint = 0
  /\ submitted = 0
  /\ completed = 0
  /\ lastUse = [img \in Images |-> 0]
  /\ poisoned = FALSE
  /\ deviceLost = FALSE
  /\ requiresRefresh = FALSE
  /\ lastPresentOutcome = "none"
  /\ lastPresentRenderPoint = 0
  /\ lastPresentMarker = 0

createSurface ==
  /\ ~deviceLost
  /\ ~surfaceLogical
  /\ ~surfacePhysical
  /\ ~swapchainAlive
  /\ surfaceLogical' = TRUE
  /\ surfacePhysical' = TRUE
  /\ UNCHANGED <<
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       lastUse,
       poisoned,
       deviceLost,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

destroySurface ==
  /\ surfaceLogical
  /\ ~swapchainAlive
  /\ surfaceLogical' = FALSE
  /\ UNCHANGED <<
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       lastUse,
       poisoned,
       deviceLost,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

collectSurface ==
  /\ surfacePhysical
  /\ ~surfaceLogical
  /\ ~swapchainAlive
  /\ surfacePhysical' = FALSE
  /\ UNCHANGED <<
       surfaceLogical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       lastUse,
       poisoned,
       deviceLost,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

createSwapchain ==
  /\ ~deviceLost
  /\ surfaceLogical
  /\ ~swapchainAlive
  /\ generation = 0
  /\ swapchainAlive' = TRUE
  /\ generation' = 1
  /\ imageGeneration' = [img \in Images |-> 1]
  /\ imageHandleAlive' = [img \in Images |-> TRUE]
  /\ acquiredImage' = NoImage
  /\ recordedReference' = FALSE
  /\ renderPoint' = 0
  /\ lastUse' = [img \in Images |-> 0]
  /\ poisoned' = FALSE
  /\ requiresRefresh' = FALSE
  /\ lastPresentOutcome' = "none"
  /\ lastPresentRenderPoint' = 0
  /\ lastPresentMarker' = 0
  /\ UNCHANGED <<surfaceLogical, surfacePhysical, submitted, completed, deviceLost>>

acquire(image) ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage = NoImage
  /\ ~recordedReference
  /\ renderPoint = 0
  /\ ~poisoned
  /\ ~requiresRefresh
  /\ imageHandleAlive[image]
  /\ lastUse[image] <= completed
  /\ acquiredImage' = image
  /\ lastUse' = [lastUse EXCEPT ![image] = 0]
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       poisoned,
       deviceLost,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

\* waitForBackBuffer in acquireNextImage observes device removal without poisoning.
loseDeviceInAcquireWait(image) ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage = NoImage
  /\ ~recordedReference
  /\ renderPoint = 0
  /\ ~poisoned
  /\ ~requiresRefresh
  /\ imageHandleAlive[image]
  /\ lastUse[image] > completed
  /\ deviceLost' = TRUE
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       lastUse,
       poisoned,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

recordUse ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage \in Images
  /\ ~recordedReference
  /\ renderPoint = 0
  /\ ~poisoned
  /\ recordedReference' = TRUE
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       acquiredImage,
       renderPoint,
       submitted,
       completed,
       lastUse,
       poisoned,
       deviceLost,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

abandonRecording ==
  /\ acquiredImage \in Images
  /\ recordedReference
  /\ renderPoint = 0
  /\ recordedReference' = FALSE
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       acquiredImage,
       renderPoint,
       submitted,
       completed,
       lastUse,
       poisoned,
       deviceLost,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

cancel ==
  /\ ~deviceLost
  /\ acquiredImage \in Images
  /\ ~recordedReference
  /\ renderPoint = 0
  /\ ~poisoned
  /\ acquiredImage' = NoImage
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       lastUse,
       poisoned,
       deviceLost,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

submitRender ==
  /\ ~deviceLost
  /\ acquiredImage \in Images
  /\ recordedReference
  /\ renderPoint = 0
  /\ ~poisoned
  /\ submitted < MaxTimeline - 1
  /\ recordedReference' = FALSE
  /\ renderPoint' = submitted + 1
  /\ submitted' = submitted + 1
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       acquiredImage,
       completed,
       lastUse,
       poisoned,
       deviceLost,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

present ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage \in Images
  /\ ~recordedReference
  /\ renderPoint > 0
  /\ renderPoint <= submitted
  /\ submitted < MaxTimeline
  /\ ~poisoned
  /\ LET postMarker == submitted + 1
     IN /\ acquiredImage' = NoImage
        /\ renderPoint' = 0
        /\ submitted' = postMarker
        /\ lastUse' = [lastUse EXCEPT ![acquiredImage] = postMarker]
        /\ requiresRefresh' = FALSE
        /\ lastPresentOutcome' = "ok"
        /\ lastPresentRenderPoint' = renderPoint
        /\ lastPresentMarker' = postMarker
        /\ UNCHANGED <<
             surfaceLogical,
             surfacePhysical,
             swapchainAlive,
             generation,
             imageGeneration,
             imageHandleAlive,
             recordedReference,
             completed,
             poisoned,
             deviceLost
           >>

presentSurfaceOutOfDate ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage \in Images
  /\ ~recordedReference
  /\ renderPoint > 0
  /\ renderPoint <= submitted
  /\ submitted < MaxTimeline
  /\ ~poisoned
  /\ LET postMarker == submitted + 1
     IN /\ acquiredImage' = NoImage
        /\ renderPoint' = 0
        /\ submitted' = postMarker
        /\ lastUse' = [lastUse EXCEPT ![acquiredImage] = postMarker]
        /\ requiresRefresh' = TRUE
        /\ lastPresentOutcome' = "outOfDate"
        /\ lastPresentRenderPoint' = renderPoint
        /\ lastPresentMarker' = postMarker
        /\ UNCHANGED <<
             surfaceLogical,
             surfacePhysical,
             swapchainAlive,
             generation,
             imageGeneration,
             imageHandleAlive,
             recordedReference,
             completed,
             poisoned,
             deviceLost
           >>

\* Post-submit failure before DXGI ownership begins (e.g. reserveSignal failure).
failPresent ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage \in Images
  /\ ~recordedReference
  /\ renderPoint > 0
  /\ renderPoint <= submitted
  /\ ~poisoned
  /\ acquiredImage' = NoImage
  /\ renderPoint' = 0
  /\ lastUse' = [lastUse EXCEPT ![acquiredImage] = renderPoint]
  /\ poisoned' = TRUE
  /\ lastPresentOutcome' = "failedBeforeDxgi"
  /\ lastPresentRenderPoint' = renderPoint
  /\ lastPresentMarker' = 0
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       recordedReference,
       submitted,
       completed,
       deviceLost,
       requiresRefresh
     >>

\* DXGI Present returns DXGI_ERROR_DEVICE_REMOVED / RESET: clears buffer.lastUse
\* without claiming renderPoint retired presentation, poisons swapchain, marks lost.
loseDeviceAfterPresent ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage \in Images
  /\ ~recordedReference
  /\ renderPoint > 0
  /\ renderPoint <= submitted
  /\ ~poisoned
  /\ acquiredImage' = NoImage
  /\ renderPoint' = 0
  /\ lastUse' = [lastUse EXCEPT ![acquiredImage] = 0]
  /\ poisoned' = TRUE
  /\ deviceLost' = TRUE
  /\ lastPresentOutcome' = "deviceLost"
  /\ lastPresentRenderPoint' = renderPoint
  /\ lastPresentMarker' = 0
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       recordedReference,
       submitted,
       completed,
       requiresRefresh
     >>

completeNext ==
  /\ ~deviceLost
  /\ completed < submitted
  /\ completed' = completed + 1
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       lastUse,
       poisoned,
       deviceLost,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

resize ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage = NoImage
  /\ ~recordedReference
  /\ renderPoint = 0
  /\ ~poisoned
  /\ AllImagesComplete
  /\ generation < MaxGeneration
  /\ generation' = generation + 1
  /\ imageGeneration' = [img \in Images |-> generation + 1]
  /\ imageHandleAlive' = [img \in Images |-> TRUE]
  /\ lastUse' = [img \in Images |-> 0]
  /\ requiresRefresh' = FALSE
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       poisoned,
       deviceLost,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

\* Non-removal ResizeBuffers failure recovers back buffers and keeps requiresRefresh.
failResize ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage = NoImage
  /\ ~recordedReference
  /\ renderPoint = 0
  /\ ~poisoned
  /\ requiresRefresh
  /\ AllImagesComplete
  /\ lastUse' = [img \in Images |-> 0]
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       poisoned,
       deviceLost,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

\* waitForBackBuffers inside resizeSwapchain observes device removal.
loseDeviceInResizeWait ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage = NoImage
  /\ ~recordedReference
  /\ renderPoint = 0
  /\ ~poisoned
  /\ ~AllImagesComplete
  /\ deviceLost' = TRUE
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       imageHandleAlive,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       lastUse,
       poisoned,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

\* ResizeBuffers returns DXGI_ERROR_DEVICE_REMOVED after releasing back buffers;
\* sets poisoned and deviceLost while preserving requiresRefresh.
loseDeviceInResizeBuffers ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage = NoImage
  /\ ~recordedReference
  /\ renderPoint = 0
  /\ ~poisoned
  /\ AllImagesComplete
  /\ imageHandleAlive' = [img \in Images |-> FALSE]
  /\ lastUse' = [img \in Images |-> 0]
  /\ poisoned' = TRUE
  /\ deviceLost' = TRUE
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       swapchainAlive,
       generation,
       imageGeneration,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       requiresRefresh,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

\* Cleanup remains available on a lost device and preserves terminal deviceLost.
destroySwapchain ==
  /\ swapchainAlive
  /\ acquiredImage = NoImage
  /\ ~recordedReference
  /\ renderPoint = 0
  /\ AllImagesComplete \/ deviceLost
  /\ swapchainAlive' = FALSE
  /\ imageHandleAlive' = [img \in Images |-> FALSE]
  /\ poisoned' = FALSE
  /\ requiresRefresh' = FALSE
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       generation,
       imageGeneration,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       lastUse,
       deviceLost,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

\* waitForBackBuffers during destroySwapchain observes device removal and still
\* finishes cleanup while recording terminal deviceLost on the device.
destroySwapchainObservingDeviceLoss ==
  /\ ~deviceLost
  /\ swapchainAlive
  /\ acquiredImage = NoImage
  /\ ~recordedReference
  /\ renderPoint = 0
  /\ ~AllImagesComplete
  /\ swapchainAlive' = FALSE
  /\ imageHandleAlive' = [img \in Images |-> FALSE]
  /\ poisoned' = FALSE
  /\ deviceLost' = TRUE
  /\ requiresRefresh' = FALSE
  /\ UNCHANGED <<
       surfaceLogical,
       surfacePhysical,
       generation,
       imageGeneration,
       acquiredImage,
       recordedReference,
       renderPoint,
       submitted,
       completed,
       lastUse,
       lastPresentOutcome,
       lastPresentRenderPoint,
       lastPresentMarker
     >>

Next ==
  \/ createSurface
  \/ destroySurface
  \/ collectSurface
  \/ createSwapchain
  \/ \E image \in Images : acquire(image)
  \/ \E image \in Images : loseDeviceInAcquireWait(image)
  \/ recordUse
  \/ abandonRecording
  \/ cancel
  \/ submitRender
  \/ present
  \/ presentSurfaceOutOfDate
  \/ failPresent
  \/ loseDeviceAfterPresent
  \/ completeNext
  \/ resize
  \/ failResize
  \/ loseDeviceInResizeWait
  \/ loseDeviceInResizeBuffers
  \/ destroySwapchain
  \/ destroySwapchainObservingDeviceLoss

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ surfaceLogical \in BOOLEAN
  /\ surfacePhysical \in BOOLEAN
  /\ swapchainAlive \in BOOLEAN
  /\ generation \in 0..MaxGeneration
  /\ imageGeneration \in [Images -> 0..MaxGeneration]
  /\ imageHandleAlive \in [Images -> BOOLEAN]
  /\ acquiredImage \in ImageOrNone
  /\ recordedReference \in BOOLEAN
  /\ renderPoint \in 0..MaxTimeline
  /\ submitted \in 0..MaxTimeline
  /\ completed \in 0..MaxTimeline
  /\ lastUse \in [Images -> 0..MaxTimeline]
  /\ poisoned \in BOOLEAN
  /\ deviceLost \in BOOLEAN
  /\ requiresRefresh \in BOOLEAN
  /\ lastPresentOutcome \in PresentOutcomes
  /\ lastPresentRenderPoint \in 0..MaxTimeline
  /\ lastPresentMarker \in 0..MaxTimeline
  /\ renderPoint <= submitted
  /\ completed <= submitted
  /\ \A img \in Images : lastUse[img] <= submitted

SurfaceRetention ==
  (surfaceLogical \/ swapchainAlive) => surfacePhysical

HealthyLiveImagesAreCurrent ==
  (swapchainAlive /\ ~poisoned) =>
    /\ generation > 0
    /\ \A img \in Images :
         imageHandleAlive[img] /\ imageGeneration[img] = generation

AcquiredImageIsReusable ==
  acquiredImage \in Images =>
    /\ swapchainAlive
    /\ ~poisoned
    /\ ~requiresRefresh
    /\ imageHandleAlive[acquiredImage]
    /\ imageGeneration[acquiredImage] = generation
    /\ lastUse[acquiredImage] <= completed

NoBorrowedHandlesAfterDestroy ==
  ~swapchainAlive => \A img \in Images : ~imageHandleAlive[img]

PoisonedSwapchainHasNoActiveFrame ==
  poisoned =>
    /\ swapchainAlive
    /\ acquiredImage = NoImage
    /\ ~recordedReference
    /\ renderPoint = 0

RefreshBlocksAcquire ==
  requiresRefresh =>
    /\ swapchainAlive
    /\ acquiredImage = NoImage
    /\ ~recordedReference
    /\ renderPoint = 0

PresentationMarkerDiscipline ==
  /\ (lastPresentOutcome \in {"ok", "outOfDate"}
      => lastPresentMarker > lastPresentRenderPoint /\ lastPresentRenderPoint > 0)
  /\ (lastPresentOutcome \in {"failedBeforeDxgi", "deviceLost"}
      => lastPresentMarker = 0)

Safety ==
  /\ SurfaceRetention
  /\ HealthyLiveImagesAreCurrent
  /\ AcquiredImageIsReusable
  /\ NoBorrowedHandlesAfterDestroy
  /\ PoisonedSwapchainHasNoActiveFrame
  /\ RefreshBlocksAcquire
  /\ PresentationMarkerDiscipline

StepSafety ==
  [][/\ (deviceLost => deviceLost')
     /\ (swapchainAlive /\ generation' > generation
         => AllImagesComplete /\ ~deviceLost /\ acquiredImage = NoImage)
     /\ (swapchainAlive /\ ~swapchainAlive'
         => acquiredImage = NoImage /\ ~recordedReference
            /\ (AllImagesComplete \/ deviceLost'))]_vars

=============================================================================
