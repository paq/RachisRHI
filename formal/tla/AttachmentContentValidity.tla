--------------------- MODULE AttachmentContentValidity ----------------------
(***************************************************************************)
(* Portable temporal attachment-content validity protocol across rendering *)
(* passes for color, depth, and stencil planes.                            *)
(*                                                                         *)
(* Contract & caller-boundary correspondence:                              *)
(*   - Public API contract: include/Rachis/Rendering.h (LoadOperation:     *)
(*     load, clear, discard; StoreOperation: store, discard).              *)
(*   - Backend lowering:                                                   *)
(*     src/D3D12/D3D12CommandRecording.cpp (PipelineManager::              *)
(*     beginRendering/endRendering: load clears, load- and store-discard   *)
(*     DiscardResource calls); src/Vulkan/VulkanGraphics.cpp               *)
(*     (vulkanLoadOp, vulkanRenderingStoreOp; on VRS tier 2 devices        *)
(*     StoreOperation::discard is lowered to store, which only preserves   *)
(*     more than the contract requires); src/Metal/MetalEncoding.mm        *)
(*     (metalLoadAction, metalStoreAction).                                *)
(*   - IMPORTANT: No Rachis backend tracks temporal attachment             *)
(*     initialization on the CPU or rejects LoadOperation::load on         *)
(*     undefined attachments at beginRendering time. Undefined attachment  *)
(*     memory after creation or StoreOperation::discard is an unobservable *)
(*     GPU hazard, so calling LoadOperation::load only when                *)
(*     storedContent # 0 is a caller obligation, while each backend is     *)
(*     responsible for lowering load/clear/discard and store/discard       *)
(*     faithfully.                                                         *)
(*   - Design rule DiscardInvalidatesStored (TRUE in the design) ensures   *)
(*     StoreOperation::discard marks stored attachment contents undefined  *)
(*     (0) at pass end so a subsequent pass cannot treat discarded pixels  *)
(*     as preserved content.                                               *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Sub-rect partial clears/draws (one whole attachment plane is        *)
(*     modeled; color, depth, and stencil planes evolve independently).    *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT DiscardInvalidatesStored

ASSUME DiscardInvalidatesStored \in BOOLEAN

MaxPasses == 3
LoadOps == {"load", "clear", "discard"}
StoreOps == {"store", "discard"}
OptLoadOps == {"none"} \cup LoadOps
OptStoreOps == {"none"} \cup StoreOps
\* Content tags: 0 = undefined, 1 = cleared value, 2 = drawn value.
ContentValues == {0, 1, 2}

VARIABLES
    scopeActive,            \* TRUE while inside a beginRendering..endRendering scope
    passCount,              \* number of completed rendering passes (0..MaxPasses)
    storedContent,          \* attachment content outside/between passes (0 = undefined)
    activeContent,          \* attachment content inside the active pass (0 = undefined)
    storedBeforeScope,      \* snapshot of storedContent at BeginScope
    activeLoadOp,           \* LoadOperation of the active pass ("none" when inactive)
    activeStoreOp,          \* StoreOperation of the active pass ("none" when inactive)
    drewInScope,            \* TRUE if at least one draw occurred in the active pass
    lastLoadOp,             \* LoadOperation of the most recently ended pass
    lastStoreOp,            \* StoreOperation of the most recently ended pass
    lastDrew,               \* whether the most recently ended pass drew
    lastPriorStored         \* storedContent entering the most recently ended pass

vars == <<scopeActive, passCount, storedContent, activeContent,
          storedBeforeScope, activeLoadOp, activeStoreOp, drewInScope,
          lastLoadOp, lastStoreOp, lastDrew, lastPriorStored>>

TypeOK ==
    /\ scopeActive \in BOOLEAN
    /\ passCount \in 0..MaxPasses
    /\ storedContent \in ContentValues
    /\ activeContent \in ContentValues
    /\ storedBeforeScope \in ContentValues
    /\ activeLoadOp \in OptLoadOps
    /\ activeStoreOp \in OptStoreOps
    /\ drewInScope \in BOOLEAN
    /\ lastLoadOp \in OptLoadOps
    /\ lastStoreOp \in OptStoreOps
    /\ lastDrew \in BOOLEAN
    /\ lastPriorStored \in ContentValues

Init ==
    /\ scopeActive = FALSE
    /\ passCount = 0
    /\ storedContent = 0
    /\ activeContent = 0
    /\ storedBeforeScope = 0
    /\ activeLoadOp = "none"
    /\ activeStoreOp = "none"
    /\ drewInScope = FALSE
    /\ lastLoadOp = "none"
    /\ lastStoreOp = "none"
    /\ lastDrew = FALSE
    /\ lastPriorStored = 0

\* Caller begins a rendering scope; LoadOperation::load is legal only when
\* storedContent is defined (# 0).
BeginScope(loadOp, storeOp) ==
    /\ ~scopeActive
    /\ passCount < MaxPasses
    /\ (loadOp = "load" => storedContent # 0)
    /\ scopeActive' = TRUE
    /\ storedBeforeScope' = storedContent
    /\ activeLoadOp' = loadOp
    /\ activeStoreOp' = storeOp
    /\ drewInScope' = FALSE
    /\ activeContent' =
         IF loadOp = "load" THEN storedContent
         ELSE IF loadOp = "clear" THEN 1
         ELSE 0
    /\ UNCHANGED <<passCount, storedContent, lastLoadOp, lastStoreOp,
                   lastDrew, lastPriorStored>>

Draw ==
    /\ scopeActive
    /\ drewInScope' = TRUE
    /\ activeContent' = 2
    /\ UNCHANGED <<scopeActive, passCount, storedContent, storedBeforeScope,
                   activeLoadOp, activeStoreOp, lastLoadOp, lastStoreOp,
                   lastDrew, lastPriorStored>>

EndScope ==
    /\ scopeActive
    /\ scopeActive' = FALSE
    /\ passCount' = passCount + 1
    /\ storedContent' =
         IF activeStoreOp = "store" THEN activeContent
         ELSE IF DiscardInvalidatesStored THEN 0
         ELSE activeContent
    /\ activeContent' = 0
    /\ activeLoadOp' = "none"
    /\ activeStoreOp' = "none"
    /\ drewInScope' = FALSE
    /\ lastLoadOp' = activeLoadOp
    /\ lastStoreOp' = activeStoreOp
    /\ lastDrew' = drewInScope
    /\ lastPriorStored' = storedBeforeScope
    /\ UNCHANGED storedBeforeScope

Next ==
    \/ \E loadOp \in LoadOps, storeOp \in StoreOps : BeginScope(loadOp, storeOp)
    \/ Draw
    \/ EndScope

Spec == Init /\ [][Next]_vars

InactiveScopeHasNoActiveState ==
    ~scopeActive =>
        /\ activeContent = 0
        /\ activeLoadOp = "none"
        /\ activeStoreOp = "none"
        /\ ~drewInScope

LoadPreservesDefinedPriorContent ==
    (scopeActive /\ activeLoadOp = "load") =>
        /\ storedBeforeScope # 0
        /\ (~drewInScope => activeContent = storedBeforeScope)

ClearDefinesInitialContent ==
    (scopeActive /\ activeLoadOp = "clear" /\ ~drewInScope) =>
        activeContent = 1

LoadDiscardRequiresDrawToDefine ==
    (scopeActive /\ activeLoadOp = "discard") =>
        (activeContent # 0 <=> drewInScope)

DrawDefinesActiveContent ==
    (scopeActive /\ drewInScope) => activeContent = 2

StoreDiscardLeavesAttachmentUndefined ==
    (~scopeActive /\ lastStoreOp = "discard") => storedContent = 0

StorePreservesExactPassOutcome ==
    (~scopeActive /\ lastStoreOp = "store") =>
        /\ (lastDrew => storedContent = 2)
        /\ (~lastDrew /\ lastLoadOp = "clear" => storedContent = 1)
        /\ (~lastDrew /\ lastLoadOp = "load" =>
              storedContent # 0 /\ storedContent = lastPriorStored)
        /\ (~lastDrew /\ lastLoadOp = "discard" => storedContent = 0)

Safety ==
    /\ InactiveScopeHasNoActiveState
    /\ LoadPreservesDefinedPriorContent
    /\ ClearDefinesInitialContent
    /\ LoadDiscardRequiresDrawToDefine
    /\ DrawDefinesActiveContent
    /\ StoreDiscardLeavesAttachmentUndefined
    /\ StorePreservesExactPassOutcome

StepSafety ==
    [][/\ (~scopeActive /\ scopeActive' /\ activeLoadOp' = "load") => storedContent # 0
       /\ (scopeActive /\ scopeActive') => storedContent' = storedContent
       /\ passCount' >= passCount]_vars
=============================================================================
