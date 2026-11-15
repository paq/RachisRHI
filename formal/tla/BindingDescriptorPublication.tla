-------------------- MODULE BindingDescriptorPublication --------------------
(***************************************************************************)
(* Copy-on-write binding-set staging reuse, shader-visible table           *)
(* publication, and CAS-gated current-version publication across           *)
(* sequential updates.                                                     *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Binding.cpp (applyBindingWrites,                     *)
(*     stageBindingSetDescriptors, publishBindingSetVersion,               *)
(*     publishUpdatedTables, updateBindingSet)                             *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Across up to MaxUpdates = 2 sequential updateBindingSet attempts on *)
(*     Slots = {0, 1, 2}, applyBindingWrites validates writes on a private *)
(*     copy-on-write candidate before any physical descriptor allocation.  *)
(*   - Invalid writes perform zero CPU staging preparations and allocate   *)
(*     zero GPU tables.                                                    *)
(*   - Valid writes prepare CPU descriptors only for slots listed in       *)
(*     writes (incrementing their stagingEpoch; unwritten slots reuse      *)
(*     prior CPU staging handles unchanged). Note that rewriting a slot    *)
(*     with the same resource value still marks that slot in preparation,  *)
(*     matching D3D12Binding.cpp.                                          *)
(*   - materializeFailed stands for both native failures that follow       *)
(*     preparation: stageBindingSetDescriptors failing to allocate CPU     *)
(*     staging descriptors, and publishUpdatedTables running out of        *)
(*     shader-visible descriptors after the CAS check, which also reverts  *)
(*     the binding-pool commit.                                            *)
(*   - casConflict is publishBindingSetVersion finding, under the manager  *)
(*     mutex, that the stored version is no longer the one the update      *)
(*     started from (stored != current in D3D12Binding.cpp).               *)
(*   - Design rule AtomicCasPublication (TRUE in the design) leaves        *)
(*     currentSources and publishedGpuTable unchanged on CAS conflict.     *)
(*   - publishedGpuTable is the content of the shader-visible table that   *)
(*     the current version binds, wherever that table lives.               *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Where an accepted version's table lives. publishUpdatedTables       *)
(*     rewrites the prepared descriptors into the current table in place   *)
(*     when no command retains the current version, and otherwise copies   *)
(*     the complete staged table into a newly allocated one. A heap whose  *)
(*     descriptors the writes leave unprepared keeps sharing the current   *)
(*     version's table, and sampler tables come from a cache keyed by the  *)
(*     ordered sampler list. The model has one table and counts each       *)
(*     accepted publication once in totalGpuTablesBuilt. That a table a    *)
(*     command retains is never rewritten is not modeled here.             *)
(*   - The native code checks for a CAS conflict before it allocates or    *)
(*     writes any table, so a conflict publishes nothing. The model still  *)
(*     counts a built table for it, an over-approximation that no          *)
(*     property depends on.                                                *)
(*   - Native D3D12 CopyDescriptors hardware descriptor copies.            *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT AtomicCasPublication

ASSUME AtomicCasPublication \in BOOLEAN

Slots == {0, 1, 2}
SourceValues == {1, 2, 3}
MaxUpdates == 2
Outcomes == {"none", "accepted", "invalidWrite", "materializeFailed", "casConflict"}

VARIABLES
    updateAttemptCount,     \* number of update attempts executed (0..MaxUpdates)
    currentSources,         \* logical descriptor sources of current version
    publishedGpuTable,      \* GPU table backing the published current version
    currentStagingEpoch,    \* per-slot CPU staging preparation epoch of current version
    priorSources,           \* snapshot of currentSources before last attempt
    priorStagingEpoch,      \* snapshot of currentStagingEpoch before last attempt
    lastWrittenSlots,       \* subset of Slots passed in writes on last attempt
    totalPreparedCount,     \* cumulative CPU staging preparations across attempts
    totalGpuTablesBuilt,    \* cumulative complete GPU descriptor tables materialized
    lastOutcome             \* outcome classification of the last attempt

coreLogicalVars == <<currentSources, publishedGpuTable, currentStagingEpoch>>
vars == <<updateAttemptCount, currentSources, publishedGpuTable,
          currentStagingEpoch, priorSources, priorStagingEpoch,
          lastWrittenSlots, totalPreparedCount, totalGpuTablesBuilt,
          lastOutcome>>

TypeOK ==
    /\ updateAttemptCount \in 0..MaxUpdates
    /\ currentSources \in [Slots -> SourceValues]
    /\ publishedGpuTable \in [Slots -> SourceValues]
    /\ currentStagingEpoch \in [Slots -> 0..MaxUpdates]
    /\ priorSources \in [Slots -> SourceValues]
    /\ priorStagingEpoch \in [Slots -> 0..MaxUpdates]
    /\ lastWrittenSlots \subseteq Slots
    /\ totalPreparedCount \in 0..(MaxUpdates * Cardinality(Slots))
    /\ totalGpuTablesBuilt \in 0..MaxUpdates
    /\ lastOutcome \in Outcomes

Init ==
    /\ updateAttemptCount = 0
    /\ currentSources = [s \in Slots |-> 1]
    /\ publishedGpuTable = [s \in Slots |-> 1]
    /\ currentStagingEpoch = [s \in Slots |-> 0]
    /\ priorSources = [s \in Slots |-> 1]
    /\ priorStagingEpoch = [s \in Slots |-> 0]
    /\ lastWrittenSlots = {}
    /\ totalPreparedCount = 0
    /\ totalGpuTablesBuilt = 0
    /\ lastOutcome = "none"

AttemptUpdate(writtenSlots, writeValue, writesValid, materializeOk, casConflict) ==
    /\ updateAttemptCount < MaxUpdates
    /\ writtenSlots # {}
    /\ Cardinality(writtenSlots) <= 2
    /\ updateAttemptCount' = updateAttemptCount + 1
    /\ priorSources' = currentSources
    /\ priorStagingEpoch' = currentStagingEpoch
    /\ lastWrittenSlots' = writtenSlots
    /\ LET candidateSources ==
               [s \in Slots |->
                  IF s \in writtenSlots THEN writeValue ELSE currentSources[s]]
           candidateStaging ==
               [s \in Slots |->
                  IF s \in writtenSlots
                  THEN currentStagingEpoch[s] + 1
                  ELSE currentStagingEpoch[s]]
       IN IF ~writesValid
          THEN /\ lastOutcome' = "invalidWrite"
               /\ UNCHANGED <<coreLogicalVars, totalPreparedCount, totalGpuTablesBuilt>>
          ELSE IF ~materializeOk
          THEN /\ lastOutcome' = "materializeFailed"
               /\ totalPreparedCount' = totalPreparedCount + Cardinality(writtenSlots)
               /\ UNCHANGED <<coreLogicalVars, totalGpuTablesBuilt>>
          ELSE IF casConflict
          THEN /\ lastOutcome' = "casConflict"
               /\ totalPreparedCount' = totalPreparedCount + Cardinality(writtenSlots)
               /\ totalGpuTablesBuilt' = totalGpuTablesBuilt + 1
               /\ IF AtomicCasPublication
                  THEN UNCHANGED coreLogicalVars
                  ELSE /\ currentSources' = candidateSources
                       /\ publishedGpuTable' = candidateSources
                       /\ currentStagingEpoch' = candidateStaging
          ELSE /\ lastOutcome' = "accepted"
               /\ totalPreparedCount' = totalPreparedCount + Cardinality(writtenSlots)
               /\ totalGpuTablesBuilt' = totalGpuTablesBuilt + 1
               /\ currentSources' = candidateSources
               /\ publishedGpuTable' = candidateSources
               /\ currentStagingEpoch' = candidateStaging

Next ==
    \E s0 \in Slots, includeNext \in BOOLEAN, val \in SourceValues,
       writesOk, matOk, conflict \in BOOLEAN :
        LET dirty == IF includeNext THEN {s0, (s0 + 1) % 3} ELSE {s0}
        IN AttemptUpdate(dirty, val, writesOk, matOk, conflict)

Spec == Init /\ [][Next]_vars

\* The published GPU descriptor table always matches every slot of currentSources.
PublishedGpuTableMatchesCurrent ==
    publishedGpuTable = currentSources

\* Unwritten slots in an accepted CoW update preserve both their source and
\* their reusable CPU staging descriptor without re-preparation.
UnwrittenStagingIsReused ==
    lastOutcome = "accepted" =>
        \A s \in Slots :
            s \notin lastWrittenSlots =>
                /\ currentSources[s] = priorSources[s]
                /\ currentStagingEpoch[s] = priorStagingEpoch[s]

\* Written slots in an accepted update advance their staging epoch by 1.
WrittenSlotsAdvanceStaging ==
    lastOutcome = "accepted" =>
        \A s \in Slots :
            s \in lastWrittenSlots =>
                currentStagingEpoch[s] = priorStagingEpoch[s] + 1

\* Any invalid write, materialization failure, or CAS conflict leaves the
\* published logical binding set untouched.
ConflictedOrFailedUpdatePreservesCurrent ==
    lastOutcome \in {"invalidWrite", "materializeFailed", "casConflict"} =>
        /\ currentSources = priorSources
        /\ publishedGpuTable = priorSources
        /\ currentStagingEpoch = priorStagingEpoch

Safety ==
    /\ PublishedGpuTableMatchesCurrent
    /\ UnwrittenStagingIsReused
    /\ WrittenSlotsAdvanceStaging
    /\ ConflictedOrFailedUpdatePreservesCurrent

StepSafety ==
    [][/\ (lastOutcome' = "invalidWrite") =>
             /\ coreLogicalVars' = coreLogicalVars
             /\ totalPreparedCount' = totalPreparedCount
             /\ totalGpuTablesBuilt' = totalGpuTablesBuilt
       /\ (lastOutcome' = "materializeFailed") =>
             /\ coreLogicalVars' = coreLogicalVars
             /\ totalGpuTablesBuilt' = totalGpuTablesBuilt
       /\ (AtomicCasPublication /\ lastOutcome' = "casConflict") =>
             coreLogicalVars' = coreLogicalVars]_vars
=============================================================================
