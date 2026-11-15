--------------------- MODULE PartiallyBoundBindingArray ---------------------
(***************************************************************************)
(* Partially bound descriptor-array initialization, required-slot          *)
(* completeness, copy-on-write `BindingSetRecord` version isolation, and   *)
(* version lifetime retention in the D3D12 backend: `createBindingSet`     *)
(* (`initializeDescriptorSources` null-descriptor materialization,         *)
(* `validateEveryDescriptorWritten`), `updateBindingSet`                   *)
(* (`clearDescriptorSource`, `publishBindingSetVersion`), and command      *)
(* snapshots (`nativeBindingSet`) in `src/D3D12/D3D12Binding.cpp`;         *)
(* binding-set batch retention (`prepareBindingSetBatch`,                  *)
(* `retainBindingSetBatchDependencies`) in                                 *)
(* `src/D3D12/D3D12CommandRecording.cpp`; and command retention in         *)
(* `src/D3D12/D3D12Queue.cpp` (`include/Rachis/Binding.h`,                 *)
(* `include/Rachis/Device.h`).                                             *)
(*                                                                         *)
(* Models a binding set with one required non-partial slot (`0`) and two   *)
(* `BindingFlag::partiallyBound` slots (`1`, `2`) backed by a three-slot   *)
(* copy-on-write version pool (`Versions = {0, 1, 2}`). Unbound partial    *)
(* slots are materialized with null descriptors so all slots remain        *)
(* initialized; `attemptUpdate` allocates a fresh immutable version        *)
(* distinct from any recorded command snapshot and rejects clearing        *)
(* required slot `0` or out-of-range writes atomically.                    *)
(*                                                                         *)
(* Abstracted: GPU shader dynamic indexing into unbound slots.             *)
(***************************************************************************)
EXTENDS Naturals

Versions == {0, 1, 2}
Slots == {0, 1, 2}
RequestedSlots == {0, 1, 2, 3}
RequiredSlots == {0}
EmptySlots == [s \in Slots |-> FALSE]
AllInitialized == [s \in Slots |-> TRUE]
Operations == {"none", "create", "bind", "update", "releaseRecording", "destroySet"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    setAlive,               \* TRUE while public BindingSetHandle is live
    currentVersion,         \* published version id in Versions when setAlive
    recording,              \* TRUE while a command list retains a snapshot
    recordedVersion,        \* command-retained version id in Versions
    versionLive,            \* active version mask in [Versions -> BOOLEAN]
    versionBound,           \* per-version bound occupancy [Versions -> [Slots -> BOOLEAN]]
    versionInitialized,     \* per-version descriptor init [Versions -> [Slots -> BOOLEAN]]
    lastOperation,          \* latest operation tag in Operations
    lastOutcome             \* outcome of latest operation in Outcomes

coreVars == <<
    setAlive, currentVersion, recording, recordedVersion,
    versionLive, versionBound, versionInitialized
>>

vars == <<
    setAlive, currentVersion, recording, recordedVersion,
    versionLive, versionBound, versionInitialized,
    lastOperation, lastOutcome
>>

RequiredSlotsBound(bindings) ==
    \A s \in RequiredSlots : bindings[s]

EverySlotInitialized(initialized) ==
    \A s \in Slots : initialized[s]

FreshVersion ==
    CHOOSE v \in Versions : ~versionLive[v]

Init ==
    /\ setAlive = FALSE
    /\ currentVersion = 0
    /\ recording = FALSE
    /\ recordedVersion = 0
    /\ versionLive = [v \in Versions |-> FALSE]
    /\ versionBound = [v \in Versions |-> EmptySlots]
    /\ versionInitialized = [v \in Versions |-> EmptySlots]
    /\ lastOperation = "none"
    /\ lastOutcome = "none"

create(partialOneBound, partialTwoBound) ==
    /\ ~setAlive
    /\ LET nextVer == FreshVersion
           initialBound ==
               [s \in Slots |->
                   IF s = 0 THEN TRUE
                   ELSE IF s = 1 THEN partialOneBound
                   ELSE partialTwoBound]
       IN /\ setAlive' = TRUE
          /\ currentVersion' = nextVer
          /\ versionLive' = [versionLive EXCEPT ![nextVer] = TRUE]
          /\ versionBound' = [versionBound EXCEPT ![nextVer] = initialBound]
          /\ versionInitialized' =
               [versionInitialized EXCEPT ![nextVer] = AllInitialized]
          /\ lastOperation' = "create"
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<recording, recordedVersion>>

bind ==
    /\ setAlive
    /\ (~recording \/ recordedVersion # currentVersion)
    /\ recording' = TRUE
    /\ recordedVersion' = currentVersion
    /\ versionLive' = [v \in Versions |-> v = currentVersion]
    /\ versionBound' =
        [v \in Versions |->
            IF v = currentVersion THEN versionBound[v] ELSE EmptySlots]
    /\ versionInitialized' =
        [v \in Versions |->
            IF v = currentVersion THEN versionInitialized[v] ELSE EmptySlots]
    /\ lastOperation' = "bind"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<setAlive, currentVersion>>

\* `updateBindingSet` prepares a fresh copy-on-write `BindingSetRecord` and
\* rejects clearing non-partial required entries or out-of-bounds slots.
attemptUpdate(slot, makeBound) ==
    LET accepted ==
            /\ setAlive
            /\ slot \in Slots
            /\ (IF slot \in RequiredSlots THEN makeBound ELSE TRUE)
    IN /\ IF accepted
          THEN LET nextVer == FreshVersion
                   updatedBound ==
                       [versionBound[currentVersion] EXCEPT ![slot] = makeBound]
                   nextLive ==
                       [v \in Versions |->
                           v = nextVer \/ (recording /\ v = recordedVersion)]
               IN /\ currentVersion' = nextVer
                  /\ versionLive' = nextLive
                  /\ versionBound' =
                       [v \in Versions |->
                           IF v = nextVer
                           THEN updatedBound
                           ELSE IF nextLive[v]
                                THEN versionBound[v]
                                ELSE EmptySlots]
                  /\ versionInitialized' =
                       [v \in Versions |->
                           IF v = nextVer
                           THEN AllInitialized
                           ELSE IF nextLive[v]
                                THEN versionInitialized[v]
                                ELSE EmptySlots]
                  /\ UNCHANGED <<setAlive, recording, recordedVersion>>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "update"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

releaseRecording ==
    /\ recording
    /\ LET nextLive == [v \in Versions |-> setAlive /\ v = currentVersion]
       IN /\ recording' = FALSE
          /\ recordedVersion' = 0
          /\ versionLive' = nextLive
          /\ versionBound' =
               [v \in Versions |->
                   IF nextLive[v] THEN versionBound[v] ELSE EmptySlots]
          /\ versionInitialized' =
               [v \in Versions |->
                   IF nextLive[v] THEN versionInitialized[v] ELSE EmptySlots]
          /\ lastOperation' = "releaseRecording"
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<setAlive, currentVersion>>

destroySet ==
    /\ setAlive
    /\ LET nextLive == [v \in Versions |-> recording /\ v = recordedVersion]
       IN /\ setAlive' = FALSE
          /\ currentVersion' = 0
          /\ versionLive' = nextLive
          /\ versionBound' =
               [v \in Versions |->
                   IF nextLive[v] THEN versionBound[v] ELSE EmptySlots]
          /\ versionInitialized' =
               [v \in Versions |->
                   IF nextLive[v] THEN versionInitialized[v] ELSE EmptySlots]
          /\ lastOperation' = "destroySet"
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<recording, recordedVersion>>

Next ==
    \/ \E p1 \in BOOLEAN, p2 \in BOOLEAN : create(p1, p2)
    \/ bind
    \/ \E s \in RequestedSlots, mb \in BOOLEAN : attemptUpdate(s, mb)
    \/ releaseRecording
    \/ destroySet

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ setAlive \in BOOLEAN
    /\ currentVersion \in Versions
    /\ recording \in BOOLEAN
    /\ recordedVersion \in Versions
    /\ versionLive \in [Versions -> BOOLEAN]
    /\ versionBound \in [Versions -> [Slots -> BOOLEAN]]
    /\ versionInitialized \in [Versions -> [Slots -> BOOLEAN]]
    /\ lastOperation \in Operations
    /\ lastOutcome \in Outcomes

PublishedVersionCompleteAndInitialized ==
    setAlive =>
        /\ versionLive[currentVersion]
        /\ RequiredSlotsBound(versionBound[currentVersion])
        /\ EverySlotInitialized(versionInitialized[currentVersion])

RecordedVersionCompleteAndInitialized ==
    recording =>
        /\ versionLive[recordedVersion]
        /\ RequiredSlotsBound(versionBound[recordedVersion])
        /\ EverySlotInitialized(versionInitialized[recordedVersion])

VersionLifetimeExact ==
    \A v \in Versions :
        versionLive[v] =
            ((setAlive /\ currentVersion = v) \/ (recording /\ recordedVersion = v))

UpdateAllocatesDistinctVersion ==
    (setAlive /\ recording /\ lastOperation = "update" /\ lastOutcome = "accepted") =>
        currentVersion # recordedVersion

Safety ==
    /\ PublishedVersionCompleteAndInitialized
    /\ RecordedVersionCompleteAndInitialized
    /\ VersionLifetimeExact
    /\ UpdateAllocatesDistinctVersion

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
        /\ ((recording /\ recording' /\ lastOperation' # "bind") =>
                /\ recordedVersion' = recordedVersion
                /\ versionBound'[recordedVersion] = versionBound[recordedVersion]
                /\ versionInitialized'[recordedVersion] =
                    versionInitialized[recordedVersion])
    ]_vars

=============================================================================
