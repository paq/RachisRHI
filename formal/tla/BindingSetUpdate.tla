-------------------------- MODULE BindingSetUpdate --------------------------
(***************************************************************************)
(* Partial copy-on-write binding-set updates, CPU validation before GPU    *)
(* descriptor allocation, CAS base-version verification, and command       *)
(* version retention.                                                      *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Binding.cpp (updateBindingSet, applyBindingWrites,   *)
(*     stageBindingSetDescriptors, publishBindingSetVersion)               *)
(*   - src/D3D12/D3D12CommandRecording.cpp, src/D3D12/D3D12Queue.cpp       *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Each immutable version in Versions = {0, 1, 2} owns a 2-slot        *)
(*     descriptor table (Slots = {0, 1}; 0 = unwritten, 1..2 = valid       *)
(*     descriptor sources).                                                *)
(*   - StartUpdate validates partial writes on CPU state BEFORE allocating *)
(*     any physical GPU table (D3D12Binding.cpp): when                     *)
(*     writesValid = FALSE, physical[version] stays FALSE and no candidate *)
(*     is materialized.                                                    *)
(*   - When writesValid = TRUE, the candidate inherits unwritten slots     *)
(*     from candidateBase, forming a complete 2-slot CoW version.          *)
(*   - FinishUpdate publishes candidateVersion only if logicalSetAlive     *)
(*     still holds and currentVersion still equals candidateBase (when     *)
(*     VerifyCapturedBaseVersion = TRUE).                                  *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Native D3D12 GPU descriptor heap layout and mutex implementation.   *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT VerifyCapturedBaseVersion

ASSUME VerifyCapturedBaseVersion \in BOOLEAN

Versions == {0, 1, 2}
Slots == {0, 1}
DescriptorValues == {0, 1, 2}
Outcomes == {"none", "started", "accepted", "rejected"}

VARIABLES
    logicalSetAlive,        \* public BindingSetHandle is still live
    currentVersion,         \* version exposed to new binds/updates
    descriptors,            \* [Versions -> [Slots -> DescriptorValues]]
    physical,               \* [Versions -> BOOLEAN] GPU table allocated
    recordingVersions,      \* subset of Versions retained by open recordings
    inFlightVersions,       \* subset of Versions retained by submissions
    candidateActive,        \* TRUE while an update is between materialize and CAS
    candidateVersion,       \* version slot held by in-progress update
    candidateBase,          \* currentVersion captured when StartUpdate began
    concurrentWinner,       \* version published by concurrent thread during active candidate (or candidateBase)
    lastOutcome             \* outcome of the most recent update action

coreVars == <<logicalSetAlive, currentVersion, descriptors, physical,
              recordingVersions, inFlightVersions, candidateActive,
              candidateVersion, candidateBase, concurrentWinner>>
vars == <<logicalSetAlive, currentVersion, descriptors, physical,
          recordingVersions, inFlightVersions, candidateActive,
          candidateVersion, candidateBase, concurrentWinner, lastOutcome>>

IsComplete(table) ==
    \A s \in Slots : table[s] # 0

Retained(v, logical, curr, active, cand, base, rec, inflight) ==
    \/ (logical /\ v = curr)
    \/ (active /\ (v = cand \/ v = base))
    \/ v \in rec
    \/ v \in inflight

RecomputePhysical(phys, logical, curr, active, cand, base, rec, inflight) ==
    [v \in Versions |->
        phys[v] /\ Retained(v, logical, curr, active, cand, base, rec, inflight)]

TypeOK ==
    /\ logicalSetAlive \in BOOLEAN
    /\ currentVersion \in Versions
    /\ descriptors \in [Versions -> [Slots -> DescriptorValues]]
    /\ physical \in [Versions -> BOOLEAN]
    /\ recordingVersions \subseteq Versions
    /\ inFlightVersions \subseteq Versions
    /\ candidateActive \in BOOLEAN
    /\ candidateVersion \in Versions
    /\ candidateBase \in Versions
    /\ concurrentWinner \in Versions
    /\ lastOutcome \in Outcomes

Init ==
    /\ logicalSetAlive = TRUE
    /\ currentVersion = 0
    /\ descriptors = [v \in Versions |-> [s \in Slots |-> IF v = 0 THEN 1 ELSE 0]]
    /\ physical = [v \in Versions |-> v = 0]
    /\ recordingVersions = {}
    /\ inFlightVersions = {}
    /\ candidateActive = FALSE
    /\ candidateVersion = 0
    /\ candidateBase = 0
    /\ concurrentWinner = 0
    /\ lastOutcome = "none"

\* Partial CoW update on one slot: CPU write validation precedes GPU allocation.
StartUpdate(version, dirtySlot, writesValid) ==
    /\ logicalSetAlive
    /\ ~candidateActive
    /\ version # currentVersion
    /\ ~physical[version]
    /\ IF ~writesValid
       THEN /\ lastOutcome' = "rejected"
            /\ UNCHANGED coreVars
       ELSE LET inherited == [descriptors[currentVersion] EXCEPT ![dirtySlot] = 2]
            IN /\ descriptors' = [descriptors EXCEPT ![version] = inherited]
               /\ physical' = [physical EXCEPT ![version] = TRUE]
               /\ candidateActive' = TRUE
               /\ candidateVersion' = version
               /\ candidateBase' = currentVersion
               /\ concurrentWinner' = currentVersion
               /\ lastOutcome' = "started"
               /\ UNCHANGED <<logicalSetAlive, currentVersion,
                              recordingVersions, inFlightVersions>>

\* Another thread completes a CoW update while candidateActive is materializing.
PublishConcurrent(version, dirtySlot) ==
    /\ candidateActive
    /\ logicalSetAlive
    /\ version # currentVersion
    /\ version # candidateVersion
    /\ ~physical[version]
    /\ LET table == [descriptors[currentVersion] EXCEPT ![dirtySlot] = 2]
           withNew == [physical EXCEPT ![version] = TRUE]
       IN /\ currentVersion' = version
          /\ concurrentWinner' = version
          /\ descriptors' = [descriptors EXCEPT ![version] = table]
          /\ physical' =
               RecomputePhysical(withNew, logicalSetAlive, version, TRUE,
                                 candidateVersion, candidateBase,
                                 recordingVersions, inFlightVersions)
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<logicalSetAlive, recordingVersions, inFlightVersions,
                         candidateActive, candidateVersion, candidateBase>>

FinishUpdate ==
    /\ candidateActive
    /\ LET baseMatches ==
               IF VerifyCapturedBaseVersion
               THEN currentVersion = candidateBase
               ELSE TRUE
           accepted == logicalSetAlive /\ baseMatches
           nextCurrent == IF accepted THEN candidateVersion ELSE currentVersion
           nextPhys ==
               RecomputePhysical(physical, logicalSetAlive, nextCurrent, FALSE,
                                 candidateVersion, candidateBase,
                                 recordingVersions, inFlightVersions)
       IN /\ currentVersion' = nextCurrent
          /\ physical' = nextPhys
          /\ descriptors' =
               [v \in Versions |->
                  IF nextPhys[v] THEN descriptors[v] ELSE [s \in Slots |-> 0]]
          /\ candidateActive' = FALSE
          /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
          /\ UNCHANGED <<logicalSetAlive, recordingVersions, inFlightVersions,
                         candidateVersion, candidateBase, concurrentWinner>>

Bind ==
    /\ logicalSetAlive
    /\ recordingVersions' = recordingVersions \cup {currentVersion}
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalSetAlive, currentVersion, descriptors, physical,
                   inFlightVersions, candidateActive, candidateVersion,
                   candidateBase, concurrentWinner>>

Submit ==
    /\ recordingVersions # {}
    /\ inFlightVersions' = inFlightVersions \cup recordingVersions
    /\ recordingVersions' = {}
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalSetAlive, currentVersion, descriptors, physical,
                   candidateActive, candidateVersion, candidateBase,
                   concurrentWinner>>

ReleaseRecorded ==
    /\ recordingVersions # {}
    /\ LET nextPhys ==
               RecomputePhysical(physical, logicalSetAlive, currentVersion,
                                 candidateActive, candidateVersion, candidateBase,
                                 {}, inFlightVersions)
       IN /\ recordingVersions' = {}
          /\ physical' = nextPhys
          /\ descriptors' =
               [v \in Versions |->
                  IF nextPhys[v] THEN descriptors[v] ELSE [s \in Slots |-> 0]]
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<logicalSetAlive, currentVersion, inFlightVersions,
                         candidateActive, candidateVersion, candidateBase,
                         concurrentWinner>>

CompleteSubmission ==
    /\ inFlightVersions # {}
    /\ LET nextPhys ==
               RecomputePhysical(physical, logicalSetAlive, currentVersion,
                                 candidateActive, candidateVersion, candidateBase,
                                 recordingVersions, {})
       IN /\ inFlightVersions' = {}
          /\ physical' = nextPhys
          /\ descriptors' =
               [v \in Versions |->
                  IF nextPhys[v] THEN descriptors[v] ELSE [s \in Slots |-> 0]]
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<logicalSetAlive, currentVersion, recordingVersions,
                         candidateActive, candidateVersion, candidateBase,
                         concurrentWinner>>

DestroySet ==
    /\ logicalSetAlive
    /\ LET nextPhys ==
               RecomputePhysical(physical, FALSE, currentVersion,
                                 candidateActive, candidateVersion, candidateBase,
                                 recordingVersions, inFlightVersions)
       IN /\ logicalSetAlive' = FALSE
          /\ physical' = nextPhys
          /\ descriptors' =
               [v \in Versions |->
                  IF nextPhys[v] THEN descriptors[v] ELSE [s \in Slots |-> 0]]
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<currentVersion, recordingVersions, inFlightVersions,
                         candidateActive, candidateVersion, candidateBase,
                         concurrentWinner>>

Next ==
    \/ \E v \in Versions, s \in Slots, ok \in BOOLEAN : StartUpdate(v, s, ok)
    \/ \E v \in Versions, s \in Slots : PublishConcurrent(v, s)
    \/ FinishUpdate
    \/ Bind
    \/ Submit
    \/ ReleaseRecorded
    \/ CompleteSubmission
    \/ DestroySet

Spec == Init /\ [][Next]_vars

\* Every physically live version has a complete descriptor table (invalid writes
\* never allocate a physical GPU version).
PhysicalVersionsAreComplete ==
    \A v \in Versions : physical[v] => IsComplete(descriptors[v])

PublishedVersionIsCompleteAndPhysical ==
    logicalSetAlive =>
        /\ physical[currentVersion]
        /\ IsComplete(descriptors[currentVersion])

RetainedVersionsStayPhysical ==
    \A v \in Versions :
        (v \in recordingVersions \/ v \in inFlightVersions) =>
            /\ physical[v]
            /\ IsComplete(descriptors[v])

ActiveCandidateRetainsBothVersions ==
    candidateActive =>
        /\ physical[candidateVersion]
        /\ IsComplete(descriptors[candidateVersion])
        /\ physical[candidateBase]

\* If a concurrent update won while candidateActive was in progress, finishing
\* the stale candidate must never overwrite concurrentWinner.
ConcurrentWinnerNotOverwrittenByStaleCandidate ==
    (~candidateActive /\ logicalSetAlive /\ concurrentWinner # candidateBase
     /\ lastOutcome \in {"accepted", "rejected"}) =>
        currentVersion = concurrentWinner

Safety ==
    /\ PhysicalVersionsAreComplete
    /\ PublishedVersionIsCompleteAndPhysical
    /\ RetainedVersionsStayPhysical
    /\ ActiveCandidateRetainsBothVersions
    /\ ConcurrentWinnerNotOverwrittenByStaleCandidate

StepSafety ==
    [][/\ (~candidateActive /\ lastOutcome' = "rejected") => coreVars' = coreVars
       /\ (candidateActive /\ ~candidateActive' /\ lastOutcome' = "rejected") =>
             currentVersion' = currentVersion
       /\ (candidateActive /\ ~candidateActive' /\ lastOutcome' = "accepted"
           /\ VerifyCapturedBaseVersion) =>
             (logicalSetAlive /\ currentVersion = candidateBase
              /\ currentVersion' = candidateVersion)]_vars
=============================================================================
