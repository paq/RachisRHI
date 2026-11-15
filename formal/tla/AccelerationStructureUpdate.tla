-------------------- MODULE AccelerationStructureUpdate ---------------------
(***************************************************************************)
(* Out-of-place BLAS/TLAS update validation, immutable destination version *)
(* publication, source immutability, and endpoint/traversal retention.     *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12AccelerationStructure.cpp (ResourceManager::         *)
(*     updateBottomLevelAccelerationStructure and                          *)
(*     updateTopLevelAccelerationStructure, which enter the shared update  *)
(*     path with inPlace = false; lockAndValidateUpdateTarget)             *)
(*   - src/D3D12/D3D12Binding.cpp (traversal capture of the exact version) *)
(*   - src/D3D12/D3D12Queue.cpp (command and timeline retention)           *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Source AS starts built with content = 1 and sourceUpdateEnabled     *)
(*     varying over BOOLEAN so non-updateable sources are tested.          *)
(*   - Out-of-place update accepts only a live built allowUpdate source,   *)
(*     an unbuilt live distinct destination, compatible geometry/instance  *)
(*     inputs, and sufficient aligned update scratch.                      *)
(*   - Source content remains immutable (1); accepted updates publish an   *)
(*     independent version (content = 2) on the destination that survives  *)
(*     logical destination handle destruction while destination traversal  *)
(*     is recorded or in flight.                                           *)
(*   - Design rule RetainUpdateEndpoints (TRUE in the design) retains both *)
(*     source and destination storage while an update is recorded or       *)
(*     executing on the GPU.                                               *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - D3D12 PERFORM_UPDATE BVH refit math and scratch memory contents.    *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RetainUpdateEndpoints

ASSUME RetainUpdateEndpoints \in BOOLEAN

Contents == 0..2
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    sourceLogical,              \* public handle of source AS is alive
    sourceBuilt,                \* source handle exposes its built version
    sourcePhysical,             \* physical backing of source AS is alive
    sourceUpdateEnabled,        \* whether source AS was built with allowUpdate
    sourceContent,              \* content tag of source AS (1)
    destinationLogical,         \* public handle of destination AS is alive
    destinationPublished,       \* destination AS has ever received an update version
    destinationPhysical,        \* physical backing of destination AS is alive
    destinationContent,         \* content tag of destination AS (0 unbuilt, 2 updated)
    recordedUpdate,             \* TRUE while out-of-place update is recorded
    inFlightUpdate,             \* TRUE while out-of-place update is in flight
    recordedSourceTraversal,    \* TRUE while ray traversal of source is recorded
    inFlightSourceTraversal,    \* TRUE while ray traversal of source is in flight
    recordedDestTraversal,      \* TRUE while ray traversal of destination is recorded
    inFlightDestTraversal,      \* TRUE while ray traversal of destination is in flight
    lastOutcome                 \* outcome of the most recent update attempt

coreVars == <<sourceLogical, sourceBuilt, sourcePhysical, sourceUpdateEnabled,
              sourceContent, destinationLogical, destinationPublished,
              destinationPhysical, destinationContent, recordedUpdate,
              inFlightUpdate, recordedSourceTraversal, inFlightSourceTraversal,
              recordedDestTraversal, inFlightDestTraversal>>
vars == <<sourceLogical, sourceBuilt, sourcePhysical, sourceUpdateEnabled,
          sourceContent, destinationLogical, destinationPublished,
          destinationPhysical, destinationContent, recordedUpdate,
          inFlightUpdate, recordedSourceTraversal, inFlightSourceTraversal,
          recordedDestTraversal, inFlightDestTraversal, lastOutcome>>

AnyRecorded ==
    recordedUpdate \/ recordedSourceTraversal \/ recordedDestTraversal

AnyInFlight ==
    inFlightUpdate \/ inFlightSourceTraversal \/ inFlightDestTraversal

TypeOK ==
    /\ sourceLogical \in BOOLEAN
    /\ sourceBuilt \in BOOLEAN
    /\ sourcePhysical \in BOOLEAN
    /\ sourceUpdateEnabled \in BOOLEAN
    /\ sourceContent \in Contents
    /\ destinationLogical \in BOOLEAN
    /\ destinationPublished \in BOOLEAN
    /\ destinationPhysical \in BOOLEAN
    /\ destinationContent \in Contents
    /\ recordedUpdate \in BOOLEAN
    /\ inFlightUpdate \in BOOLEAN
    /\ recordedSourceTraversal \in BOOLEAN
    /\ inFlightSourceTraversal \in BOOLEAN
    /\ recordedDestTraversal \in BOOLEAN
    /\ inFlightDestTraversal \in BOOLEAN
    /\ lastOutcome \in Outcomes

Init ==
    /\ sourceLogical = TRUE
    /\ sourceBuilt = TRUE
    /\ sourcePhysical = TRUE
    /\ sourceUpdateEnabled \in BOOLEAN
    /\ sourceContent = 1
    /\ destinationLogical = TRUE
    /\ destinationPublished = FALSE
    /\ destinationPhysical = FALSE
    /\ destinationContent = 0
    /\ recordedUpdate = FALSE
    /\ inFlightUpdate = FALSE
    /\ recordedSourceTraversal = FALSE
    /\ inFlightSourceTraversal = FALSE
    /\ recordedDestTraversal = FALSE
    /\ inFlightDestTraversal = FALSE
    /\ lastOutcome = "none"

UpdateValid(distinctHandles, compatibleLayout, scratchLargeEnough) ==
    /\ sourceLogical
    /\ sourceBuilt
    /\ sourcePhysical
    /\ sourceUpdateEnabled
    /\ destinationLogical
    /\ ~destinationPublished
    /\ ~recordedUpdate
    /\ distinctHandles
    /\ compatibleLayout
    /\ scratchLargeEnough

AttemptUpdate(distinctHandles, compatibleLayout, scratchLargeEnough) ==
    IF UpdateValid(distinctHandles, compatibleLayout, scratchLargeEnough)
    THEN /\ destinationPublished' = TRUE
         /\ destinationPhysical' = TRUE
         /\ destinationContent' = sourceContent + 1
         /\ recordedUpdate' = TRUE
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<sourceLogical, sourceBuilt, sourcePhysical,
                        sourceUpdateEnabled, sourceContent, destinationLogical,
                        inFlightUpdate, recordedSourceTraversal,
                        inFlightSourceTraversal, recordedDestTraversal,
                        inFlightDestTraversal>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

RecordSourceTraversal ==
    /\ sourceLogical /\ sourceBuilt /\ sourcePhysical
    /\ ~recordedSourceTraversal
    /\ recordedSourceTraversal' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourceBuilt, sourcePhysical,
                   sourceUpdateEnabled, sourceContent, destinationLogical,
                   destinationPublished, destinationPhysical, destinationContent,
                   recordedUpdate, inFlightUpdate, inFlightSourceTraversal,
                   recordedDestTraversal, inFlightDestTraversal>>

RecordDestinationTraversal ==
    /\ destinationLogical /\ destinationPublished /\ destinationPhysical
    /\ ~recordedDestTraversal
    /\ recordedDestTraversal' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourceBuilt, sourcePhysical,
                   sourceUpdateEnabled, sourceContent, destinationLogical,
                   destinationPublished, destinationPhysical, destinationContent,
                   recordedUpdate, inFlightUpdate, recordedSourceTraversal,
                   inFlightSourceTraversal, inFlightDestTraversal>>

DestroySource ==
    /\ sourceLogical
    /\ sourceLogical' = FALSE
    /\ sourceBuilt' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourcePhysical, sourceUpdateEnabled, sourceContent,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationContent, recordedUpdate, inFlightUpdate,
                   recordedSourceTraversal, inFlightSourceTraversal,
                   recordedDestTraversal, inFlightDestTraversal>>

DestroyDestination ==
    /\ destinationLogical
    /\ destinationLogical' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourceBuilt, sourcePhysical,
                   sourceUpdateEnabled, sourceContent, destinationPublished,
                   destinationPhysical, destinationContent, recordedUpdate,
                   inFlightUpdate, recordedSourceTraversal,
                   inFlightSourceTraversal, recordedDestTraversal,
                   inFlightDestTraversal>>

Submit ==
    /\ AnyRecorded
    /\ ~AnyInFlight
    /\ inFlightUpdate' = recordedUpdate
    /\ inFlightSourceTraversal' = recordedSourceTraversal
    /\ inFlightDestTraversal' = recordedDestTraversal
    /\ recordedUpdate' = FALSE
    /\ recordedSourceTraversal' = FALSE
    /\ recordedDestTraversal' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourceBuilt, sourcePhysical,
                   sourceUpdateEnabled, sourceContent, destinationLogical,
                   destinationPublished, destinationPhysical, destinationContent>>

Complete ==
    /\ AnyInFlight
    /\ inFlightUpdate' = FALSE
    /\ inFlightSourceTraversal' = FALSE
    /\ inFlightDestTraversal' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourceBuilt, sourcePhysical,
                   sourceUpdateEnabled, sourceContent, destinationLogical,
                   destinationPublished, destinationPhysical, destinationContent,
                   recordedUpdate, recordedSourceTraversal, recordedDestTraversal>>

Collect ==
    LET keepSource ==
            \/ (sourceLogical /\ sourceBuilt)
            \/ (RetainUpdateEndpoints /\ (recordedUpdate \/ inFlightUpdate))
            \/ recordedSourceTraversal
            \/ inFlightSourceTraversal
        keepDestination ==
            /\ destinationPublished
            /\ (destinationLogical
                \/ (RetainUpdateEndpoints /\ (recordedUpdate \/ inFlightUpdate))
                \/ recordedDestTraversal
                \/ inFlightDestTraversal)
    IN /\ sourcePhysical' = keepSource
       /\ destinationPhysical' = keepDestination
       /\ lastOutcome' = "none"
       /\ UNCHANGED <<sourceLogical, sourceBuilt, sourceUpdateEnabled,
                      sourceContent, destinationLogical, destinationPublished,
                      destinationContent, recordedUpdate, inFlightUpdate,
                      recordedSourceTraversal, inFlightSourceTraversal,
                      recordedDestTraversal, inFlightDestTraversal>>

Next ==
    \/ \E distinctOk, layoutOk, scratchOk \in BOOLEAN :
          AttemptUpdate(distinctOk, layoutOk, scratchOk)
    \/ RecordSourceTraversal
    \/ RecordDestinationTraversal
    \/ DestroySource
    \/ DestroyDestination
    \/ Submit
    \/ Complete
    \/ Collect

Spec == Init /\ [][Next]_vars

SourceIsNeverMutated ==
    sourceContent = 1

NonUpdateableSourceProducesNoDestination ==
    ~sourceUpdateEnabled => ~destinationPublished /\ destinationContent = 0

\* Even after DestroyDestination, any physically retained destination version
\* preserves its independent updated content without mutating the source.
RetainedDestinationIsIndependent ==
    /\ (destinationLogical /\ destinationPublished => destinationPhysical)
    /\ (destinationPhysical =>
          destinationPublished /\ destinationContent = 2 /\ sourceContent = 1)

UpdatesRetainBothVersions ==
    (recordedUpdate \/ inFlightUpdate) =>
        sourcePhysical /\ destinationPhysical /\ destinationPublished

TraversalsRetainExactVersions ==
    /\ ((recordedSourceTraversal \/ inFlightSourceTraversal) => sourcePhysical)
    /\ ((recordedDestTraversal \/ inFlightDestTraversal) =>
          destinationPhysical /\ destinationContent = 2)

Safety ==
    /\ SourceIsNeverMutated
    /\ NonUpdateableSourceProducesNoDestination
    /\ RetainedDestinationIsIndependent
    /\ UpdatesRetainBothVersions
    /\ TraversalsRetainExactVersions

StepSafety ==
    [][/\ (~destinationPublished /\ destinationPublished') =>
             (sourceLogical /\ sourceBuilt /\ sourcePhysical
              /\ sourceUpdateEnabled /\ destinationLogical)
       /\ sourceContent' = sourceContent
       /\ sourceUpdateEnabled' = sourceUpdateEnabled
       /\ (lastOutcome' = "rejected" => coreVars' = coreVars)]_vars
=============================================================================
