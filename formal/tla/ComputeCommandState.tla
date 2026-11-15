------------------------ MODULE ComputeCommandState -------------------------
(***************************************************************************)
(* Compute command-buffer pipeline retention, pipeline-scoped binding-set  *)
(* invalidation on pipeline switches, dispatch completeness checking, and  *)
(* recording vs in-flight reference separation.                            *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12CommandRecording.cpp                                 *)
(*     (PipelineManager::bindComputePipeline,                              *)
(*     PipelineManager::bindComputeBindingSets, PipelineManager::dispatch) *)
(*   - src/D3D12/D3D12Pipeline.cpp (PipelineManager::createComputePipeline *)
(*     and destroyPipeline)                                                *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::retainResource,             *)
(*     QueueManager::submit, D3D12CommandBuffer::abandon)                  *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Pipelines = {0, 1} have distinct descriptor-set layout contracts    *)
(*     over Slots = {0, 1}:                                                *)
(*       * Pipeline 0 requires slot 0 with layout 0; slot 1 is empty       *)
(*         (!slot.requiresBindingSet, initialized true on bind).           *)
(*       * Pipeline 1 requires slot 0 with incompatible layout 1 and       *)
(*         slot 1 with layout 1.                                           *)
(*   - When InvalidateOnPipelineChange = TRUE (D3D12CommandRecording.cpp), *)
(*     binding a compute pipeline clears prior bound-set flags and marks   *)
(*     only non-required slots satisfied.                                  *)
(*   - AttemptDispatch requires a bound compute pipeline and all slots     *)
(*     satisfied; rejected dispatches leave recording state untouched.     *)
(*   - Both empty and non-empty command buffers may be submitted or        *)
(*     abandoned; physical pipeline storage stays alive while logically    *)
(*     alive or retained by recording/in-flight command state.             *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Root-constant range coverage and per-dimension threadgroup counts.  *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT InvalidateOnPipelineChange

ASSUME InvalidateOnPipelineChange \in BOOLEAN

Pipelines == {0, 1}
NoPipeline == 2
Slots == {0, 1}
Layouts == {0, 1}
NoLayout == 2
MaxDispatches == 2
Phases == {"idle", "recording", "inFlight"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    logicalAlive,           \* [Pipelines -> BOOLEAN] public handle alive
    physicalAlive,          \* [Pipelines -> BOOLEAN] ComputePipelineRecord alive
    cmdPhase,               \* command-buffer phase in Phases
    recordingRefs,          \* subset of Pipelines retained while recording
    inFlightRefs,           \* subset of Pipelines retained while in flight
    activePipeline,         \* Pipelines \cup {NoPipeline}
    slotSatisfied,          \* [Slots -> BOOLEAN] state.boundBindingSets
    slotBoundLayout,        \* [Slots -> Layouts \cup {NoLayout}] root-table layout
    dispatches,             \* 0..MaxDispatches recorded in current recording
    lastDispatchedPipeline, \* Pipelines \cup {NoPipeline}
    lastDispatchedLayouts,  \* [Slots -> Layouts \cup {NoLayout}]
    lastOutcome             \* Outcomes

coreVars == <<logicalAlive, physicalAlive, cmdPhase, recordingRefs,
              inFlightRefs, activePipeline, slotSatisfied, slotBoundLayout,
              dispatches, lastDispatchedPipeline, lastDispatchedLayouts>>
vars == <<logicalAlive, physicalAlive, cmdPhase, recordingRefs,
          inFlightRefs, activePipeline, slotSatisfied, slotBoundLayout,
          dispatches, lastDispatchedPipeline, lastDispatchedLayouts,
          lastOutcome>>

RequiresSet(p, s) ==
    (s = 0) \/ (p = 1)

RequiredLayout(p, s) ==
    IF p = 0 THEN 0 ELSE 1

Retained(p) ==
    (p \in recordingRefs) \/ (p \in inFlightRefs)

AllSlotsSatisfied ==
    \A s \in Slots : slotSatisfied[s]

TypeOK ==
    /\ logicalAlive \in [Pipelines -> BOOLEAN]
    /\ physicalAlive \in [Pipelines -> BOOLEAN]
    /\ cmdPhase \in Phases
    /\ recordingRefs \subseteq Pipelines
    /\ inFlightRefs \subseteq Pipelines
    /\ activePipeline \in (Pipelines \cup {NoPipeline})
    /\ slotSatisfied \in [Slots -> BOOLEAN]
    /\ slotBoundLayout \in [Slots -> (Layouts \cup {NoLayout})]
    /\ dispatches \in 0..MaxDispatches
    /\ lastDispatchedPipeline \in (Pipelines \cup {NoPipeline})
    /\ lastDispatchedLayouts \in [Slots -> (Layouts \cup {NoLayout})]
    /\ lastOutcome \in Outcomes

Init ==
    /\ logicalAlive = [p \in Pipelines |-> FALSE]
    /\ physicalAlive = [p \in Pipelines |-> FALSE]
    /\ cmdPhase = "recording"
    /\ recordingRefs = {}
    /\ inFlightRefs = {}
    /\ activePipeline = NoPipeline
    /\ slotSatisfied = [s \in Slots |-> FALSE]
    /\ slotBoundLayout = [s \in Slots |-> NoLayout]
    /\ dispatches = 0
    /\ lastDispatchedPipeline = NoPipeline
    /\ lastDispatchedLayouts = [s \in Slots |-> NoLayout]
    /\ lastOutcome = "none"

CreatePipeline(p) ==
    /\ ~logicalAlive[p]
    /\ ~physicalAlive[p]
    /\ logicalAlive' = [logicalAlive EXCEPT ![p] = TRUE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![p] = TRUE]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<cmdPhase, recordingRefs, inFlightRefs, activePipeline,
                   slotSatisfied, slotBoundLayout, dispatches,
                   lastDispatchedPipeline, lastDispatchedLayouts>>

DestroyPipeline(p) ==
    /\ logicalAlive[p]
    /\ logicalAlive' = [logicalAlive EXCEPT ![p] = FALSE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![p] = Retained(p)]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<cmdPhase, recordingRefs, inFlightRefs, activePipeline,
                   slotSatisfied, slotBoundLayout, dispatches,
                   lastDispatchedPipeline, lastDispatchedLayouts>>

Acquire ==
    /\ cmdPhase = "idle"
    /\ cmdPhase' = "recording"
    /\ recordingRefs' = {}
    /\ activePipeline' = NoPipeline
    /\ slotSatisfied' = [s \in Slots |-> FALSE]
    /\ slotBoundLayout' = [s \in Slots |-> NoLayout]
    /\ dispatches' = 0
    /\ lastDispatchedPipeline' = NoPipeline
    /\ lastDispatchedLayouts' = [s \in Slots |-> NoLayout]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, physicalAlive, inFlightRefs>>

BindPipeline(p) ==
    /\ cmdPhase = "recording"
    /\ logicalAlive[p]
    /\ physicalAlive[p]
    /\ recordingRefs' = recordingRefs \cup {p}
    /\ activePipeline' = p
    /\ slotSatisfied' =
         IF InvalidateOnPipelineChange
         THEN [s \in Slots |-> ~RequiresSet(p, s)]
         ELSE [s \in Slots |-> slotSatisfied[s] \/ ~RequiresSet(p, s)]
    /\ slotBoundLayout' =
         IF InvalidateOnPipelineChange
         THEN [s \in Slots |-> NoLayout]
         ELSE slotBoundLayout
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<logicalAlive, physicalAlive, cmdPhase, inFlightRefs,
                   dispatches, lastDispatchedPipeline, lastDispatchedLayouts>>

BindSet(s) ==
    /\ cmdPhase = "recording"
    /\ activePipeline \in Pipelines
    /\ RequiresSet(activePipeline, s)
    /\ slotSatisfied' = [slotSatisfied EXCEPT ![s] = TRUE]
    /\ slotBoundLayout' =
         [slotBoundLayout EXCEPT ![s] = RequiredLayout(activePipeline, s)]
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<logicalAlive, physicalAlive, cmdPhase, recordingRefs,
                   inFlightRefs, activePipeline, dispatches,
                   lastDispatchedPipeline, lastDispatchedLayouts>>

AttemptDispatch ==
    /\ cmdPhase = "recording"
    /\ dispatches < MaxDispatches
    /\ LET valid == (activePipeline \in Pipelines) /\ AllSlotsSatisfied IN
       IF valid
       THEN /\ dispatches' = dispatches + 1
            /\ lastDispatchedPipeline' = activePipeline
            /\ lastDispatchedLayouts' = slotBoundLayout
            /\ lastOutcome' = "accepted"
            /\ UNCHANGED <<logicalAlive, physicalAlive, cmdPhase, recordingRefs,
                           inFlightRefs, activePipeline, slotSatisfied,
                           slotBoundLayout>>
       ELSE /\ lastOutcome' = "rejected"
            /\ UNCHANGED coreVars

Submit ==
    /\ cmdPhase = "recording"
    /\ cmdPhase' = "inFlight"
    /\ recordingRefs' = {}
    /\ inFlightRefs' = recordingRefs
    /\ activePipeline' = NoPipeline
    /\ slotSatisfied' = [s \in Slots |-> FALSE]
    /\ slotBoundLayout' = [s \in Slots |-> NoLayout]
    /\ dispatches' = 0
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, physicalAlive,
                   lastDispatchedPipeline, lastDispatchedLayouts>>

Complete ==
    /\ cmdPhase = "inFlight"
    /\ cmdPhase' = "idle"
    /\ inFlightRefs' = {}
    /\ physicalAlive' =
         [p \in Pipelines |-> logicalAlive[p] \/ (p \in recordingRefs)]
    /\ lastDispatchedPipeline' = NoPipeline
    /\ lastDispatchedLayouts' = [s \in Slots |-> NoLayout]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, recordingRefs, activePipeline,
                   slotSatisfied, slotBoundLayout, dispatches>>

Abandon ==
    /\ cmdPhase = "recording"
    /\ cmdPhase' = "idle"
    /\ recordingRefs' = {}
    /\ activePipeline' = NoPipeline
    /\ slotSatisfied' = [s \in Slots |-> FALSE]
    /\ slotBoundLayout' = [s \in Slots |-> NoLayout]
    /\ dispatches' = 0
    /\ physicalAlive' =
         [p \in Pipelines |-> logicalAlive[p] \/ (p \in inFlightRefs)]
    /\ lastDispatchedPipeline' = NoPipeline
    /\ lastDispatchedLayouts' = [s \in Slots |-> NoLayout]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, inFlightRefs>>

Next ==
    \/ \E p \in Pipelines : CreatePipeline(p) \/ DestroyPipeline(p) \/ BindPipeline(p)
    \/ \E s \in Slots : BindSet(s)
    \/ Acquire
    \/ AttemptDispatch
    \/ Submit
    \/ Complete
    \/ Abandon

Spec == Init /\ [][Next]_vars

PhysicalLifetimeIsExact ==
    \A p \in Pipelines :
        physicalAlive[p] <=> (logicalAlive[p] \/ Retained(p))

ActivePipelineIsRecordingRetained ==
    activePipeline \in Pipelines =>
        /\ cmdPhase = "recording"
        /\ activePipeline \in recordingRefs
        /\ physicalAlive[activePipeline]

SubmittedReferencesSeparated ==
    cmdPhase # "recording" =>
        /\ recordingRefs = {}
        /\ activePipeline = NoPipeline
        /\ dispatches = 0

\* Every accepted dispatch bound descriptor tables matching every required
\* slot's exact layout for the active compute pipeline.
DispatchedBindingsMatchPipelineLayout ==
    lastDispatchedPipeline \in Pipelines =>
        \A s \in Slots :
            RequiresSet(lastDispatchedPipeline, s) =>
                lastDispatchedLayouts[s] = RequiredLayout(lastDispatchedPipeline, s)

Safety ==
    /\ PhysicalLifetimeIsExact
    /\ ActivePipelineIsRecordingRetained
    /\ SubmittedReferencesSeparated
    /\ DispatchedBindingsMatchPipelineLayout

StepSafety ==
    [][lastOutcome' = "rejected" => coreVars' = coreVars]_vars
=============================================================================
