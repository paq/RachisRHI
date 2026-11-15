------------------ MODULE AccelerationStructureCompaction -------------------
(***************************************************************************)
(* Acceleration-structure compacted-size query protocol, immutable compact *)
(* copy publication, flag propagation, and transitive lifetime retention.  *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Query.cpp (QueryManager::                            *)
(*     writeAccelerationStructureCompactedSize,                            *)
(*     resolveAccelerationStructureCompactedSizes)                         *)
(*   - src/D3D12/D3D12AccelerationStructure.cpp (ResourceManager::         *)
(*     compactAccelerationStructure,                                       *)
(*     accelerationStructureCompactionSource; free helper                  *)
(*     makeCompactedVersion)                                               *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::                            *)
(*     retainAccelerationStructureDependencies; command and timeline       *)
(*     retention)                                                          *)
(*                                                                         *)
(* Caller vs. RHI boundary note:                                           *)
(*   - Postbuild compacted-size query completion (queryPhase = 4) and      *)
(*     passing the exact GPU-emitted compacted byte count before recording *)
(*     compactAccelerationStructure are caller GPU-ordering obligations.   *)
(*     On the CPU, D3D12AccelerationStructure.cpp validates distinct live  *)
(*     handles, matching AS types, non-zero compactedSize ==               *)
(*     destination->size, compactedSize <= sourceVersion->resultSize,      *)
(*     built source with allowCompaction, and an unbuilt immutable         *)
(*     destination.                                                        *)
(*   - Compacted destinations copy sourceVersion->buildFlags and           *)
(*     dependencies verbatim (makeCompactedVersion), so                    *)
(*     destinationUpdateEnabled matches sourceUpdateEnabled rather than    *)
(*     unconditionally enabling updates.                                   *)
(*   - Design rule PreserveSourceFlags (TRUE in the design) propagates     *)
(*     sourceUpdateEnabled to destinationUpdateEnabled on compaction.      *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Native D3D12 CopyRaytracingAccelerationStructure byte layout.       *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT PreserveSourceFlags

ASSUME PreserveSourceFlags \in BOOLEAN

QueryPhases == 0..4
CopyPhases == 0..3
TraversalPhases == 0..3
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    sourceLogical,              \* public handle of source AS is alive
    sourcePhysical,             \* physical backing of source AS is retained
    sourceUpdateEnabled,        \* whether source AS was built with allowUpdate
    destinationLogical,         \* public handle of destination AS is alive
    destinationPublished,       \* destination has an immutable compact version
    destinationPhysical,        \* physical backing of destination AS is retained
    destinationEquivalent,      \* destination version preserves geometry/instances
    destinationUpdateEnabled,   \* whether compacted destination allows updates
    dependencyPhysical,         \* transitive BLAS dependency physical lifetime
    destinationRetainsDep,      \* compacted destination version retains BLAS dep
    queryPhase,                 \* 0:idle, 1:written, 2:resolved, 3:submitted, 4:done
    copyPhase,                  \* 0:idle, 1:recorded, 2:submitted, 3:done
    traversalPhase,             \* 0:idle, 1:recorded, 2:submitted, 3:done
    lastOutcome                 \* outcome of the last compaction attempt

coreVars == <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
              destinationLogical, destinationPublished, destinationPhysical,
              destinationEquivalent, destinationUpdateEnabled,
              dependencyPhysical, destinationRetainsDep,
              queryPhase, copyPhase, traversalPhase>>
vars == <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
          destinationLogical, destinationPublished, destinationPhysical,
          destinationEquivalent, destinationUpdateEnabled,
          dependencyPhysical, destinationRetainsDep,
          queryPhase, copyPhase, traversalPhase, lastOutcome>>

TypeOK ==
    /\ sourceLogical \in BOOLEAN
    /\ sourcePhysical \in BOOLEAN
    /\ sourceUpdateEnabled \in BOOLEAN
    /\ destinationLogical \in BOOLEAN
    /\ destinationPublished \in BOOLEAN
    /\ destinationPhysical \in BOOLEAN
    /\ destinationEquivalent \in BOOLEAN
    /\ destinationUpdateEnabled \in BOOLEAN
    /\ dependencyPhysical \in BOOLEAN
    /\ destinationRetainsDep \in BOOLEAN
    /\ queryPhase \in QueryPhases
    /\ copyPhase \in CopyPhases
    /\ traversalPhase \in TraversalPhases
    /\ lastOutcome \in Outcomes

Init ==
    /\ sourceLogical = TRUE
    /\ sourcePhysical = TRUE
    /\ sourceUpdateEnabled \in BOOLEAN
    /\ destinationLogical = TRUE
    /\ destinationPublished = FALSE
    /\ destinationPhysical = FALSE
    /\ destinationEquivalent = FALSE
    /\ destinationUpdateEnabled = FALSE
    /\ dependencyPhysical = TRUE
    /\ destinationRetainsDep = FALSE
    /\ queryPhase = 0
    /\ copyPhase = 0
    /\ traversalPhase = 0
    /\ lastOutcome = "none"

WriteSizeQuery ==
    /\ sourceLogical /\ sourcePhysical /\ queryPhase = 0
    /\ queryPhase' = 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   copyPhase, traversalPhase>>

ResolveSizeQuery ==
    /\ queryPhase = 1
    /\ queryPhase' = 2
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   copyPhase, traversalPhase>>

SubmitSizeQuery ==
    /\ queryPhase = 2
    /\ queryPhase' = 3
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   copyPhase, traversalPhase>>

CompleteSizeQuery ==
    /\ queryPhase = 3
    /\ queryPhase' = 4
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   copyPhase, traversalPhase>>

CompactRequestValid(compactionEnabled, distinctHandles, matchingType, exactSize) ==
    /\ sourceLogical
    /\ sourcePhysical
    /\ destinationLogical
    /\ ~destinationPublished
    /\ queryPhase = 4
    /\ copyPhase = 0
    /\ compactionEnabled
    /\ distinctHandles
    /\ matchingType
    /\ exactSize

AttemptCompact(compactionEnabled, distinctHandles, matchingType, exactSize) ==
    IF CompactRequestValid(compactionEnabled, distinctHandles, matchingType, exactSize)
    THEN /\ destinationPublished' = TRUE
         /\ destinationPhysical' = TRUE
         /\ destinationEquivalent' = TRUE
         /\ destinationUpdateEnabled' = IF PreserveSourceFlags THEN sourceUpdateEnabled ELSE TRUE
         /\ destinationRetainsDep' = TRUE
         /\ copyPhase' = 1
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                        destinationLogical, dependencyPhysical,
                        queryPhase, traversalPhase>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

SubmitCompact ==
    /\ copyPhase = 1
    /\ copyPhase' = 2
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   queryPhase, traversalPhase>>

CompleteCompact ==
    /\ copyPhase = 2
    /\ copyPhase' = 3
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   queryPhase, traversalPhase>>

RecordTraversal ==
    /\ destinationLogical /\ destinationPublished /\ destinationPhysical
    /\ traversalPhase = 0
    /\ traversalPhase' = 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   queryPhase, copyPhase>>

SubmitTraversal ==
    /\ traversalPhase = 1
    /\ traversalPhase' = 2
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   queryPhase, copyPhase>>

CompleteTraversal ==
    /\ traversalPhase = 2
    /\ traversalPhase' = 3
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   queryPhase, copyPhase>>

DestroySource ==
    /\ sourceLogical
    /\ sourceLogical' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourcePhysical, sourceUpdateEnabled,
                   destinationLogical, destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   queryPhase, copyPhase, traversalPhase>>

DestroyDestination ==
    /\ destinationLogical
    /\ destinationLogical' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, sourceUpdateEnabled,
                   destinationPublished, destinationPhysical,
                   destinationEquivalent, destinationUpdateEnabled,
                   dependencyPhysical, destinationRetainsDep,
                   queryPhase, copyPhase, traversalPhase>>

Collect ==
    LET keepSource ==
            \/ sourceLogical
            \/ queryPhase \in {1, 2, 3}
            \/ copyPhase \in {1, 2}
        keepDestination ==
            /\ destinationPublished
            /\ (destinationLogical \/ copyPhase \in {1, 2} \/ traversalPhase \in {1, 2})
        keepDependency ==
            \/ keepSource
            \/ (keepDestination /\ destinationRetainsDep)
    IN /\ sourcePhysical' = keepSource
       /\ destinationPhysical' = keepDestination
       /\ dependencyPhysical' = keepDependency
       /\ lastOutcome' = "none"
       /\ UNCHANGED <<sourceLogical, sourceUpdateEnabled,
                      destinationLogical, destinationPublished,
                      destinationEquivalent, destinationUpdateEnabled,
                      destinationRetainsDep, queryPhase, copyPhase,
                      traversalPhase>>

Next ==
    \/ WriteSizeQuery
    \/ ResolveSizeQuery
    \/ SubmitSizeQuery
    \/ CompleteSizeQuery
    \/ \E compOk, distinctOk, typeOk, sizeOk \in BOOLEAN :
          AttemptCompact(compOk, distinctOk, typeOk, sizeOk)
    \/ SubmitCompact
    \/ CompleteCompact
    \/ RecordTraversal
    \/ SubmitTraversal
    \/ CompleteTraversal
    \/ DestroySource
    \/ DestroyDestination
    \/ Collect

Spec == Init /\ [][Next]_vars

QueryLifetimeSafe ==
    queryPhase \in {1, 2, 3} => sourcePhysical

CompactLifetimeSafe ==
    copyPhase \in {1, 2} => sourcePhysical /\ destinationPhysical

PublishedVersionPreservesContract ==
    destinationPublished =>
        /\ queryPhase = 4
        /\ destinationEquivalent
        /\ (destinationUpdateEnabled <=> sourceUpdateEnabled)
        /\ destinationRetainsDep

TraversalLifetimeSafe ==
    traversalPhase \in {1, 2} => destinationPhysical /\ dependencyPhysical

Safety ==
    /\ QueryLifetimeSafe
    /\ CompactLifetimeSafe
    /\ PublishedVersionPreservesContract
    /\ TraversalLifetimeSafe

\* Failure atomicity on rejected compact attempts and immutability of source flags.
StepSafety ==
    [][/\ (~destinationPublished /\ destinationPublished') =>
             (sourceLogical /\ sourcePhysical /\ destinationLogical /\ queryPhase = 4 /\ copyPhase = 0)
       /\ (lastOutcome' = "rejected" /\ (lastOutcome # "rejected" \/ coreVars' # coreVars)) =>
             coreVars' = coreVars
       /\ sourceUpdateEnabled' = sourceUpdateEnabled]_vars
=============================================================================
