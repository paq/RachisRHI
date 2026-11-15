-------------------------- MODULE PresentEndpoint ---------------------------
(***************************************************************************)
(* Swapchain acquisition endpoint metadata (undefined before first         *)
(* presentation, present thereafter), no-stage/no-access present-layout    *)
(* barrier rules, present acceptance at the exact endpoint, and defensive  *)
(* D3D12 legacy resource-state conversion.                                 *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12ResourceBarriers.cpp (validTextureBarrierScope and   *)
(*     usesPresentOutsideNoAccessScope present-endpoint validation,        *)
(*     textureBarrierAccessesSupported, legacyTextureState)                *)
(*   - src/D3D12/D3D12BarrierTranslation.h (accessMatchesTextureLayout,    *)
(*     translateLegacyTextureState: present maps to PRESENT only with no   *)
(*     access)                                                             *)
(*   - src/Vulkan/VulkanSwapchain.cpp (describeAcquiredImage:              *)
(*     initialLayout = contentsDefined ? present : undefined)              *)
(*   - src/D3D12/D3D12Swapchain.cpp (acquireNextImage,                     *)
(*     cancelSwapchainImage, and present lease gating; D3D12 acquisition   *)
(*     always publishes the default present initial endpoint rather than   *)
(*     undefined)                                                          *)
(*   - include/Rachis/Presentation.h (AcquiredSwapchainImage endpoint      *)
(*     fields)                                                             *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Models one swapchain image across acquire, barrier transitions,       *)
(*   cancel, and present. Native WSI semaphore/fence signaling is covered  *)
(*   by SwapchainLifecycle.tla and VulkanWsiLifecycle.tla.                 *)
(***************************************************************************)

EXTENDS Naturals

Layouts == {"undefined", "renderTarget", "present"}
NoneStage == 0
RenderStage == 1
Stages == {NoneStage, RenderStage}
NoneAccess == 0
RenderAccess == 1
CopyAccess == 2
Accesses == {NoneAccess, RenderAccess, CopyAccess}
LegacyStates == {"present", "renderTarget", "copySource", "invalidScope"}
MaxWork == 2
Operations == {"none", "acquire", "barrier", "present", "cancel"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  acquired,              \* Whether the swapchain image is currently leased
  contentsDefined,       \* True once the image has completed a presentation
  acquiredInitialLayout, \* initialLayout published by the current acquisition
  currentLayout,         \* Current tracked layout of the leased image
  currentStage,          \* Current synchronization stage of the leased image
  currentAccess,         \* Current access mask of the leased image
  recordedWork,          \* Number of barrier transitions recorded this frame
  lastOp,                \* Most recent operation kind
  lastOutcome,           \* Outcome of most recent operation
  lastAfterLayout,       \* Target layout of most recent barrier attempt
  lastAfterStage,        \* Target stage of most recent barrier attempt
  lastAfterAccess,       \* Target access of most recent barrier attempt
  lastLegacyAfter        \* Defensive D3D12 legacy state for target scope

coreVars == <<
  acquired, contentsDefined, acquiredInitialLayout,
  currentLayout, currentStage, currentAccess, recordedWork
>>

vars == <<
  acquired, contentsDefined, acquiredInitialLayout,
  currentLayout, currentStage, currentAccess, recordedWork,
  lastOp, lastOutcome, lastAfterLayout, lastAfterStage,
  lastAfterAccess, lastLegacyAfter
>>

EndpointScopeValid(layout, stage, access) ==
  CASE layout = "undefined" -> stage = NoneStage /\ access = NoneAccess
    [] layout = "present" -> stage = NoneStage /\ access = NoneAccess
    [] layout = "renderTarget" -> stage = RenderStage /\ access = RenderAccess
    [] OTHER -> FALSE

\* Mirrors legacyTextureState(TextureLayout::present, access, stages,
\* queueType) in src/D3D12/D3D12ResourceBarriers.cpp
\* (translateLegacyTextureState in src/D3D12/D3D12BarrierTranslation.h) for
\* non-copy queues: nonzero access never collapses to PRESENT.
DefensiveLegacyState(layout, access) ==
  IF layout = "present" /\ access = NoneAccess
  THEN "present"
  ELSE IF access = RenderAccess
  THEN "renderTarget"
  ELSE IF access = CopyAccess
  THEN "copySource"
  ELSE "invalidScope"

BarrierValid(beforeLayout, beforeStage, beforeAccess,
             afterLayout, afterStage, afterAccess) ==
  /\ acquired
  /\ recordedWork < MaxWork
  /\ beforeLayout = currentLayout
  /\ beforeStage = currentStage
  /\ beforeAccess = currentAccess
  /\ EndpointScopeValid(beforeLayout, beforeStage, beforeAccess)
  /\ afterLayout # "undefined"
  /\ EndpointScopeValid(afterLayout, afterStage, afterAccess)

PresentValid ==
  /\ acquired
  /\ recordedWork > 0
  /\ currentLayout = "present"
  /\ currentStage = NoneStage
  /\ currentAccess = NoneAccess

Init ==
  /\ acquired = FALSE
  /\ contentsDefined = FALSE
  /\ acquiredInitialLayout = "undefined"
  /\ currentLayout = "undefined"
  /\ currentStage = NoneStage
  /\ currentAccess = NoneAccess
  /\ recordedWork = 0
  /\ lastOp = "none"
  /\ lastOutcome = "none"
  /\ lastAfterLayout = "present"
  /\ lastAfterStage = NoneStage
  /\ lastAfterAccess = NoneAccess
  /\ lastLegacyAfter = "present"

AcquireImage ==
  LET initLayout == IF contentsDefined THEN "present" ELSE "undefined" IN
  /\ ~acquired
  /\ acquired' = TRUE
  /\ acquiredInitialLayout' = initLayout
  /\ currentLayout' = initLayout
  /\ currentStage' = NoneStage
  /\ currentAccess' = NoneAccess
  /\ recordedWork' = 0
  /\ UNCHANGED contentsDefined
  /\ lastOp' = "acquire"
  /\ lastOutcome' = "accepted"
  /\ lastAfterLayout' = initLayout
  /\ lastAfterStage' = NoneStage
  /\ lastAfterAccess' = NoneAccess
  /\ lastLegacyAfter' = DefensiveLegacyState(initLayout, NoneAccess)

AttemptBarrier(beforeLayout, beforeStage, beforeAccess,
               afterLayout, afterStage, afterAccess) ==
  /\ IF BarrierValid(beforeLayout, beforeStage, beforeAccess,
                     afterLayout, afterStage, afterAccess)
     THEN /\ currentLayout' = afterLayout
          /\ currentStage' = afterStage
          /\ currentAccess' = afterAccess
          /\ recordedWork' = recordedWork + 1
          /\ UNCHANGED <<acquired, contentsDefined, acquiredInitialLayout>>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "barrier"
  /\ lastAfterLayout' = afterLayout
  /\ lastAfterStage' = afterStage
  /\ lastAfterAccess' = afterAccess
  /\ lastLegacyAfter' = DefensiveLegacyState(afterLayout, afterAccess)

AttemptPresent ==
  /\ IF PresentValid
     THEN /\ acquired' = FALSE
          /\ contentsDefined' = TRUE
          /\ recordedWork' = 0
          /\ UNCHANGED <<
               acquiredInitialLayout, currentLayout,
               currentStage, currentAccess
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "present"
  /\ lastAfterLayout' = currentLayout
  /\ lastAfterStage' = currentStage
  /\ lastAfterAccess' = currentAccess
  /\ lastLegacyAfter' = DefensiveLegacyState(currentLayout, currentAccess)

AttemptCancel ==
  /\ IF acquired /\ recordedWork = 0
     THEN /\ acquired' = FALSE
          /\ UNCHANGED <<
               contentsDefined, acquiredInitialLayout,
               currentLayout, currentStage, currentAccess, recordedWork
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "cancel"
  /\ lastAfterLayout' = currentLayout
  /\ lastAfterStage' = currentStage
  /\ lastAfterAccess' = currentAccess
  /\ lastLegacyAfter' = DefensiveLegacyState(currentLayout, currentAccess)

Next ==
  \/ AcquireImage
  \/ \E bl \in Layouts, bs \in Stages, ba \in Accesses,
        al \in Layouts, as \in Stages, aa \in Accesses :
       AttemptBarrier(bl, bs, ba, al, as, aa)
  \/ AttemptPresent
  \/ AttemptCancel

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ acquired \in BOOLEAN
  /\ contentsDefined \in BOOLEAN
  /\ acquiredInitialLayout \in Layouts
  /\ currentLayout \in Layouts
  /\ currentStage \in Stages
  /\ currentAccess \in Accesses
  /\ recordedWork \in 0..MaxWork
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes
  /\ lastAfterLayout \in Layouts
  /\ lastAfterStage \in Stages
  /\ lastAfterAccess \in Accesses
  /\ lastLegacyAfter \in LegacyStates

AcquiredInitialLayoutMatchesHistory ==
  acquired =>
    acquiredInitialLayout = (IF contentsDefined THEN "present" ELSE "undefined")

CurrentStateSatisfiesEndpointScope ==
  acquired => EndpointScopeValid(currentLayout, currentStage, currentAccess)

UndefinedOnlyBeforeFirstPresentAndWork ==
  acquired /\ currentLayout = "undefined" =>
    /\ ~contentsDefined
    /\ recordedWork = 0
    /\ currentStage = NoneStage
    /\ currentAccess = NoneAccess

InvalidPresentAccessNeverMapsToPresent ==
  lastOp = "barrier" /\ lastAfterLayout = "present" /\ lastAfterAccess # NoneAccess =>
    /\ lastOutcome = "rejected"
    /\ lastLegacyAfter # "present"

Safety ==
  /\ AcquiredInitialLayoutMatchesHistory
  /\ CurrentStateSatisfiesEndpointScope
  /\ UndefinedOnlyBeforeFirstPresentAndWork
  /\ InvalidPresentAccessNeverMapsToPresent

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (contentsDefined => contentsDefined')
    /\ (lastOp' = "present" /\ lastOutcome' = "accepted" =>
          /\ PresentValid
          /\ ~acquired'
          /\ contentsDefined')
    /\ (lastOp' = "barrier" /\ lastOutcome' = "accepted" =>
          /\ recordedWork' = recordedWork + 1
          /\ currentLayout' # "undefined"
          /\ (currentLayout' = "present" =>
                currentStage' = NoneStage /\ currentAccess' = NoneAccess))
  ]_vars

=============================================================================
