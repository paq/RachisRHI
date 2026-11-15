-------------------------- MODULE BindingSetBatch ---------------------------
(***************************************************************************)
(* Atomic multi-set binding validation, ordered 4-family dependency        *)
(* staging, tentative-prefix rollback, and recording/submission retention. *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12CommandRecording.cpp (PipelineManager::              *)
(*     bindComputeBindingSets, bindGraphicsBindingSets, and the free       *)
(*     function retainBindingSetBatchDependencies)                         *)
(*   - src/D3D12/D3D12RayTracingPipeline.cpp (PipelineManager::            *)
(*     bindRayTracingBindingSets)                                          *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::retainCommandDependencies)  *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - A contiguous batch over Sets = {0, 1} validates bound pipeline,     *)
(*     firstSet = 0, setCount = 2, flattened dynamic-offset count = 2,     *)
(*     per-set compatibility/alignment, and live set handles before any    *)
(*     command dependency vector is mutated.                               *)
(*   - When valid, retainBindingSetBatchDependencies stages dependencies   *)
(*     in order across 4 families: "resource" (stage 0, including          *)
(*     prepared and internalOwner), "residency" (stage 1), "swapchain"     *)
(*     (stage 2), and "sparse" (stage 3).                                  *)
(*   - BeginBatch stages any tentative prefix before failureStage directly *)
(*     into recordingDepCount while leaving bound/boundOffset uncommitted; *)
(*     FinishBatch either commits the full batch or rolls                  *)
(*     recordingDepCount back to snapRecordingDepCount.                    *)
(*   - Design rule RollbackBatchPrefix (TRUE in the design) restores exact *)
(*     prior dependency counts when mid-transaction publication fails.     *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - D3D12 SetComputeRootDescriptorTable /                               *)
(*     SetGraphicsRootDescriptorTable calls executed after                 *)
(*     retainBindingSetBatchDependencies commits.                          *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RollbackBatchPrefix

ASSUME RollbackBatchPrefix \in BOOLEAN

Sets == {0, 1}
Offsets == {0, 256}
InvalidKinds == 0..4
\* Stages 0..3 fail at that family ordinal; 4 means all 4 families succeed.
FailureStages == 0..4
Families == {"resource", "residency", "swapchain", "sparse"}
Outcomes == {"none", "pending", "accepted", "rejected"}
MaxBatches == 2

VARIABLES
    pipelineBound,          \* active pipeline is bound on recording command buffer
    logicalAlive,           \* [Sets -> BOOLEAN] public set handle liveness
    physicalAlive,          \* [Sets -> BOOLEAN] physical set version liveness
    slotBound,              \* [Sets -> BOOLEAN] whether slot currently has a bound set
    boundOffset,            \* [Sets -> Offsets] dynamic offset bound on slot
    recordingLease,         \* [Sets -> BOOLEAN] retained by open recording
    inFlightLease,          \* [Sets -> BOOLEAN] retained by in-flight submission
    recordingDepCount,      \* [Families -> 0..(MaxBatches+1)] command dep vector sizes
    inFlightDepCount,       \* [Families -> 0..MaxBatches] submitted dep counts
    txActive,               \* TRUE while tentative dependency prefix is staged
    txFailureStage,         \* 0..3 fail at family ordinal; 4 commits
    txOffsets,              \* [Sets -> Offsets] candidate dynamic offsets
    txReqDeps,              \* [Families -> BOOLEAN] families needed by candidate
    snapSlotBound,          \* pre-attempt snapshot of slotBound
    snapBoundOffset,        \* pre-attempt snapshot of boundOffset
    snapRecordingLease,     \* pre-attempt snapshot of recordingLease
    snapRecordingDepCount,  \* pre-attempt snapshot of recordingDepCount
    lastOutcome             \* outcome of the most recent step

coreVars == <<pipelineBound, logicalAlive, physicalAlive, slotBound,
              boundOffset, recordingLease, inFlightLease,
              recordingDepCount, inFlightDepCount>>
txVars == <<txActive, txFailureStage, txOffsets, txReqDeps>>
snapVars == <<snapSlotBound, snapBoundOffset, snapRecordingLease, snapRecordingDepCount>>
vars == <<pipelineBound, logicalAlive, physicalAlive, slotBound,
          boundOffset, recordingLease, inFlightLease,
          recordingDepCount, inFlightDepCount,
          txActive, txFailureStage, txOffsets, txReqDeps,
          snapSlotBound, snapBoundOffset, snapRecordingLease,
          snapRecordingDepCount, lastOutcome>>

FamilyOrdinal(f) ==
    IF f = "resource" THEN 0
    ELSE IF f = "residency" THEN 1
    ELSE IF f = "swapchain" THEN 2
    ELSE 3

BatchRequestValid(firstSet, setCount, offsetCount, firstOk, secondOk) ==
    /\ pipelineBound
    /\ firstSet = 0
    /\ setCount = 2
    /\ offsetCount = 2
    /\ firstOk
    /\ secondOk
    /\ \A s \in Sets : logicalAlive[s]
    /\ \A s \in Sets : ~inFlightLease[s]
    /\ recordingDepCount["resource"] < MaxBatches

TypeOK ==
    /\ pipelineBound \in BOOLEAN
    /\ logicalAlive \in [Sets -> BOOLEAN]
    /\ physicalAlive \in [Sets -> BOOLEAN]
    /\ slotBound \in [Sets -> BOOLEAN]
    /\ boundOffset \in [Sets -> Offsets]
    /\ recordingLease \in [Sets -> BOOLEAN]
    /\ inFlightLease \in [Sets -> BOOLEAN]
    /\ recordingDepCount \in [Families -> 0..(MaxBatches + 1)]
    /\ inFlightDepCount \in [Families -> 0..MaxBatches]
    /\ txActive \in BOOLEAN
    /\ txFailureStage \in FailureStages
    /\ txOffsets \in [Sets -> Offsets]
    /\ txReqDeps \in [Families -> BOOLEAN]
    /\ snapSlotBound \in [Sets -> BOOLEAN]
    /\ snapBoundOffset \in [Sets -> Offsets]
    /\ snapRecordingLease \in [Sets -> BOOLEAN]
    /\ snapRecordingDepCount \in [Families -> 0..(MaxBatches + 1)]
    /\ lastOutcome \in Outcomes

Init ==
    /\ pipelineBound = TRUE
    /\ logicalAlive = [s \in Sets |-> TRUE]
    /\ physicalAlive = [s \in Sets |-> TRUE]
    /\ slotBound = [s \in Sets |-> FALSE]
    /\ boundOffset = [s \in Sets |-> 0]
    /\ recordingLease = [s \in Sets |-> FALSE]
    /\ inFlightLease = [s \in Sets |-> FALSE]
    /\ recordingDepCount = [f \in Families |-> 0]
    /\ inFlightDepCount = [f \in Families |-> 0]
    /\ txActive = FALSE
    /\ txFailureStage = 4
    /\ txOffsets = [s \in Sets |-> 0]
    /\ txReqDeps = [f \in Families |-> FALSE]
    /\ snapSlotBound = [s \in Sets |-> FALSE]
    /\ snapBoundOffset = [s \in Sets |-> 0]
    /\ snapRecordingLease = [s \in Sets |-> FALSE]
    /\ snapRecordingDepCount = [f \in Families |-> 0]
    /\ lastOutcome = "none"

ResetBatchSnapshot ==
    /\ txFailureStage' = 4
    /\ txOffsets' = [s \in Sets |-> 0]
    /\ txReqDeps' = [f \in Families |-> FALSE]
    /\ snapSlotBound' = [s \in Sets |-> FALSE]
    /\ snapBoundOffset' = [s \in Sets |-> 0]
    /\ snapRecordingLease' = [s \in Sets |-> FALSE]
    /\ snapRecordingDepCount' = [f \in Families |-> 0]

BeginBatch(firstSet, setCount, offsetCount, firstOk, secondOk,
           off0, off1, hasResidency, hasSwapchain, hasSparse, failureStage) ==
    /\ ~txActive
    /\ LET valid == BatchRequestValid(firstSet, setCount, offsetCount, firstOk, secondOk)
           offsets == [s \in Sets |-> IF s = 0 THEN off0 ELSE off1]
           reqDeps == [f \in Families |->
                         \/ f = "resource"
                         \/ (f = "residency" /\ hasResidency)
                         \/ (f = "swapchain" /\ hasSwapchain)
                         \/ (f = "sparse" /\ hasSparse)]
           prefixCount == [f \in Families |->
                             recordingDepCount[f]
                             + (IF reqDeps[f] /\ FamilyOrdinal(f) < failureStage THEN 1 ELSE 0)]
       IN /\ snapSlotBound' = slotBound
          /\ snapBoundOffset' = boundOffset
          /\ snapRecordingLease' = recordingLease
          /\ snapRecordingDepCount' = recordingDepCount
          /\ IF ~valid \/ failureStage = 0
             THEN /\ txActive' = FALSE
                  /\ txFailureStage' = 4
                  /\ txOffsets' = [s \in Sets |-> 0]
                  /\ txReqDeps' = [f \in Families |-> FALSE]
                  /\ lastOutcome' = "rejected"
                  /\ UNCHANGED coreVars
             ELSE /\ txActive' = TRUE
                  /\ txFailureStage' = failureStage
                  /\ txOffsets' = offsets
                  /\ txReqDeps' = reqDeps
                  /\ lastOutcome' = "pending"
                  /\ recordingDepCount' = prefixCount
                  /\ UNCHANGED <<pipelineBound, logicalAlive, physicalAlive,
                                 slotBound, boundOffset, recordingLease,
                                 inFlightLease, inFlightDepCount>>

FinishBatch ==
    /\ txActive
    /\ txActive' = FALSE
    /\ IF txFailureStage < 4
       THEN /\ lastOutcome' = "rejected"
            /\ recordingDepCount' =
                 IF RollbackBatchPrefix THEN snapRecordingDepCount ELSE recordingDepCount
            /\ txFailureStage' = 4
            /\ txOffsets' = [s \in Sets |-> 0]
            /\ txReqDeps' = [f \in Families |-> FALSE]
            /\ UNCHANGED <<pipelineBound, logicalAlive, physicalAlive,
                           slotBound, boundOffset, recordingLease,
                           inFlightLease, inFlightDepCount, snapVars>>
       ELSE /\ lastOutcome' = "accepted"
            /\ slotBound' = [s \in Sets |-> TRUE]
            /\ boundOffset' = txOffsets
            /\ recordingLease' = [s \in Sets |-> TRUE]
            /\ txFailureStage' = 4
            /\ UNCHANGED <<pipelineBound, logicalAlive, physicalAlive,
                           inFlightLease, recordingDepCount, inFlightDepCount,
                           txOffsets, txReqDeps, snapVars>>

BindPipeline ==
    /\ ~txActive
    /\ \A s \in Sets : ~inFlightLease[s]
    /\ pipelineBound' = TRUE
    /\ slotBound' = [s \in Sets |-> FALSE]
    /\ boundOffset' = [s \in Sets |-> 0]
    /\ lastOutcome' = "none"
    /\ ResetBatchSnapshot
    /\ UNCHANGED <<logicalAlive, physicalAlive, recordingLease, inFlightLease,
                   recordingDepCount, inFlightDepCount, txActive>>

DestroySet(s) ==
    /\ ~txActive
    /\ logicalAlive[s]
    /\ logicalAlive' = [logicalAlive EXCEPT ![s] = FALSE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![s] = recordingLease[s] \/ inFlightLease[s]]
    /\ lastOutcome' = "none"
    /\ ResetBatchSnapshot
    /\ UNCHANGED <<pipelineBound, slotBound, boundOffset, recordingLease,
                   inFlightLease, recordingDepCount, inFlightDepCount, txActive>>

Submit ==
    /\ ~txActive
    /\ \E s \in Sets : recordingLease[s]
    /\ \A s \in Sets : ~inFlightLease[s]
    /\ pipelineBound' = FALSE
    /\ slotBound' = [s \in Sets |-> FALSE]
    /\ boundOffset' = [s \in Sets |-> 0]
    /\ recordingLease' = [s \in Sets |-> FALSE]
    /\ inFlightLease' = recordingLease
    /\ recordingDepCount' = [f \in Families |-> 0]
    /\ inFlightDepCount' = recordingDepCount
    /\ lastOutcome' = "none"
    /\ ResetBatchSnapshot
    /\ UNCHANGED <<logicalAlive, physicalAlive, txActive>>

Complete ==
    /\ ~txActive
    /\ \E s \in Sets : inFlightLease[s]
    /\ physicalAlive' = [s \in Sets |-> logicalAlive[s] \/ recordingLease[s]]
    /\ inFlightLease' = [s \in Sets |-> FALSE]
    /\ inFlightDepCount' = [f \in Families |-> 0]
    /\ lastOutcome' = "none"
    /\ ResetBatchSnapshot
    /\ UNCHANGED <<pipelineBound, logicalAlive, slotBound, boundOffset,
                   recordingLease, recordingDepCount, txActive>>

Abandon ==
    /\ ~txActive
    /\ \A s \in Sets : ~inFlightLease[s]
    /\ pipelineBound' = FALSE
    /\ physicalAlive' = [s \in Sets |-> logicalAlive[s]]
    /\ slotBound' = [s \in Sets |-> FALSE]
    /\ boundOffset' = [s \in Sets |-> 0]
    /\ recordingLease' = [s \in Sets |-> FALSE]
    /\ recordingDepCount' = [f \in Families |-> 0]
    /\ lastOutcome' = "none"
    /\ ResetBatchSnapshot
    /\ UNCHANGED <<logicalAlive, inFlightLease, inFlightDepCount, txActive>>

ChooseValidCandidate ==
    \E off0, off1 \in Offsets,
       hasRes, hasSwap, hasSparse \in BOOLEAN,
       stage \in FailureStages :
        BeginBatch(0, 2, 2, TRUE, TRUE, off0, off1, hasRes, hasSwap, hasSparse, stage)

ChooseInvalidCandidate ==
    \E kind \in InvalidKinds :
        BeginBatch(
            IF kind = 0 THEN 1 ELSE 0,
            IF kind = 1 THEN 1 ELSE 2,
            IF kind = 2 THEN 1 ELSE 2,
            kind # 3,
            kind # 4,
            0, 0, FALSE, FALSE, FALSE, 4)

Next ==
    \/ ChooseValidCandidate
    \/ ChooseInvalidCandidate
    \/ FinishBatch
    \/ BindPipeline
    \/ \E s \in Sets : DestroySet(s)
    \/ Submit
    \/ Complete
    \/ Abandon

Spec == Init /\ [][Next]_vars

\* While a batch dependency transaction is in progress, bindings and leases
\* have not yet been published to portable command state.
ActivePrefixLeavesBindingsUncommitted ==
    txActive =>
        /\ slotBound = snapSlotBound
        /\ boundOffset = snapBoundOffset
        /\ recordingLease = snapRecordingLease

\* Every rejected batch (whether invalid before dependency staging or failed
\* mid-transaction at stage 1..3) restores the exact prior recording state.
RejectedBatchRestoresRecordingState ==
    lastOutcome = "rejected" =>
        /\ ~txActive
        /\ slotBound = snapSlotBound
        /\ boundOffset = snapBoundOffset
        /\ recordingLease = snapRecordingLease
        /\ recordingDepCount = snapRecordingDepCount

\* Every accepted batch binds all sets in flattened order and increments
\* exactly the requested dependency families.
AcceptedBatchCommitsAllFamilies ==
    lastOutcome = "accepted" =>
        /\ ~txActive
        /\ \A s \in Sets : slotBound[s] /\ boundOffset[s] = txOffsets[s] /\ recordingLease[s]
        /\ \A f \in Families :
             recordingDepCount[f] =
                 snapRecordingDepCount[f] + (IF txReqDeps[f] THEN 1 ELSE 0)

BoundSlotsHoldLeases ==
    \A s \in Sets :
        slotBound[s] => recordingLease[s] /\ (txActive \/ recordingDepCount["resource"] > 0)

LeasesRetainPhysicalSets ==
    /\ \A s \in Sets : logicalAlive[s] => physicalAlive[s]
    /\ \A s \in Sets : (recordingLease[s] \/ inFlightLease[s]) => physicalAlive[s]

RecordingAndInFlightDisjoint ==
    /\ \A s \in Sets : ~(recordingLease[s] /\ inFlightLease[s])
    /\ \A f \in Families : ~(recordingDepCount[f] > 0 /\ inFlightDepCount[f] > 0)

Safety ==
    /\ ActivePrefixLeavesBindingsUncommitted
    /\ RejectedBatchRestoresRecordingState
    /\ AcceptedBatchCommitsAllFamilies
    /\ BoundSlotsHoldLeases
    /\ LeasesRetainPhysicalSets
    /\ RecordingAndInFlightDisjoint

StepSafety ==
    [][(~txActive /\ lastOutcome' = "rejected") => coreVars' = coreVars]_vars
=============================================================================
