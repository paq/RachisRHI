----------------------- MODULE ResourceMemoryIdentity -----------------------
(***************************************************************************)
(* Monotonic allocation identity issuance, unpublished-identity burning on *)
(* failed native creation, non-reuse across same-slot destroy-and-recreate *)
(* cycles, snapshot non-aliasing across epochs, and atomic exhaustion.     *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/Common/MonotonicIdentitySource.h (acquire: non-zero             *)
(*     monotonic identities, terminal exhaustion after the maximum)        *)
(*   - src/D3D12/D3D12Resource.cpp (reserveResourceMemoryIdentity over     *)
(*     memoryTracking->allocationIdentities, reserved before native        *)
(*     creation and burned on failure; trackResourceMemory;                *)
(*     ResourceManager::captureMemorySnapshot)                             *)
(*   - include/Rachis/Resource.h (ResourceMemoryEntry::allocationId)       *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two reusable resource slots ("left", "right"), MaxIdentity = 4, and   *)
(*   independent monotonic allocation epochs MaxEpoch = 4 distinguish a    *)
(*   recreated resource in the same slot from an earlier snapshot entry.   *)
(***************************************************************************)

EXTENDS Naturals

Slots == {"left", "right"}
MaxIdentity == 4
MaxEpoch == 4
Identities == 1..MaxIdentity
Epochs == 1..MaxEpoch
Operations == {
  "none", "create", "dropUnpublished", "retain",
  "destroy", "release", "captureSnapshot"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  nextIdentity,       \* Next monotonic allocationId to issue (1..MaxIdentity+1)
  nextEpoch,          \* Independent monotonic physical-instance epoch counter
  issuedIdentities,   \* Set of all allocationIds ever issued
  burnedIdentities,   \* Subset of issuedIds reserved then dropped before publish
  retiredIdentities,  \* Subset of issuedIds whose physical resource has retired
  logicalAlive,       \* Public handle liveness per slot
  retained,           \* Deferred in-flight/descriptor retention per slot
  physicalAlive,      \* Native resource storage liveness per slot
  identity,           \* Assigned allocationId per live physical slot (0 if dead)
  allocEpoch,         \* Assigned physical-instance epoch per live slot (0 if dead)
  snapshotValid,      \* Whether a memory snapshot has been captured
  snapshotPresent,    \* Per-slot presence in the latest captured snapshot
  snapshotIdentity,   \* Per-slot allocationId in the latest captured snapshot
  snapshotEpoch,      \* Per-slot physical-instance epoch in the latest snapshot
  lastOp,             \* Most recent operation kind
  lastOutcome         \* Outcome of most recent operation

coreVars == <<
  nextIdentity, nextEpoch,
  issuedIdentities, burnedIdentities, retiredIdentities,
  logicalAlive, retained, physicalAlive, identity, allocEpoch,
  snapshotValid, snapshotPresent, snapshotIdentity, snapshotEpoch
>>

vars == <<
  nextIdentity, nextEpoch,
  issuedIdentities, burnedIdentities, retiredIdentities,
  logicalAlive, retained, physicalAlive, identity, allocEpoch,
  snapshotValid, snapshotPresent, snapshotIdentity, snapshotEpoch,
  lastOp, lastOutcome
>>

Init ==
  /\ nextIdentity = 1
  /\ nextEpoch = 1
  /\ issuedIdentities = {}
  /\ burnedIdentities = {}
  /\ retiredIdentities = {}
  /\ logicalAlive = [s \in Slots |-> FALSE]
  /\ retained = [s \in Slots |-> FALSE]
  /\ physicalAlive = [s \in Slots |-> FALSE]
  /\ identity = [s \in Slots |-> 0]
  /\ allocEpoch = [s \in Slots |-> 0]
  /\ snapshotValid = FALSE
  /\ snapshotPresent = [s \in Slots |-> FALSE]
  /\ snapshotIdentity = [s \in Slots |-> 0]
  /\ snapshotEpoch = [s \in Slots |-> 0]
  /\ lastOp = "none"
  /\ lastOutcome = "none"

AttemptCreate(slot) ==
  LET ok ==
        /\ ~physicalAlive[slot]
        /\ nextIdentity <= MaxIdentity
        /\ nextEpoch <= MaxEpoch
      id == nextIdentity
      ep == nextEpoch
  IN
  /\ IF ok
     THEN /\ nextIdentity' = id + 1
          /\ nextEpoch' = ep + 1
          /\ issuedIdentities' = issuedIdentities \cup {id}
          /\ logicalAlive' = [logicalAlive EXCEPT ![slot] = TRUE]
          /\ retained' = [retained EXCEPT ![slot] = FALSE]
          /\ physicalAlive' = [physicalAlive EXCEPT ![slot] = TRUE]
          /\ identity' = [identity EXCEPT ![slot] = id]
          /\ allocEpoch' = [allocEpoch EXCEPT ![slot] = ep]
          /\ UNCHANGED <<
               burnedIdentities, retiredIdentities,
               snapshotValid, snapshotPresent, snapshotIdentity, snapshotEpoch
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "create"

\* Models reserving an allocationId when native creation subsequently fails:
\* the id is permanently burned and never assigned to a later resource.
AttemptDropUnpublished ==
  LET id == nextIdentity IN
  /\ IF id <= MaxIdentity
     THEN /\ nextIdentity' = id + 1
          /\ issuedIdentities' = issuedIdentities \cup {id}
          /\ burnedIdentities' = burnedIdentities \cup {id}
          /\ UNCHANGED <<
               nextEpoch, retiredIdentities,
               logicalAlive, retained, physicalAlive, identity, allocEpoch,
               snapshotValid, snapshotPresent, snapshotIdentity, snapshotEpoch
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "dropUnpublished"

RetainSlot(slot) ==
  /\ logicalAlive[slot]
  /\ ~retained[slot]
  /\ retained' = [retained EXCEPT ![slot] = TRUE]
  /\ lastOp' = "retain"
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       nextIdentity, nextEpoch,
       issuedIdentities, burnedIdentities, retiredIdentities,
       logicalAlive, physicalAlive, identity, allocEpoch,
       snapshotValid, snapshotPresent, snapshotIdentity, snapshotEpoch
     >>

DestroySlot(slot) ==
  LET staysAlive == retained[slot]
      oldId == identity[slot]
  IN
  /\ logicalAlive[slot]
  /\ logicalAlive' = [logicalAlive EXCEPT ![slot] = FALSE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![slot] = staysAlive]
  /\ identity' = [identity EXCEPT ![slot] = IF staysAlive THEN oldId ELSE 0]
  /\ allocEpoch' = [allocEpoch EXCEPT ![slot] = IF staysAlive THEN allocEpoch[slot] ELSE 0]
  /\ retiredIdentities' =
       IF staysAlive THEN retiredIdentities ELSE retiredIdentities \cup {oldId}
  /\ lastOp' = "destroy"
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       nextIdentity, nextEpoch, issuedIdentities, burnedIdentities,
       retained, snapshotValid, snapshotPresent, snapshotIdentity, snapshotEpoch
     >>

ReleaseSlot(slot) ==
  LET staysAlive == logicalAlive[slot]
      oldId == identity[slot]
  IN
  /\ retained[slot]
  /\ retained' = [retained EXCEPT ![slot] = FALSE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![slot] = staysAlive]
  /\ identity' = [identity EXCEPT ![slot] = IF staysAlive THEN oldId ELSE 0]
  /\ allocEpoch' = [allocEpoch EXCEPT ![slot] = IF staysAlive THEN allocEpoch[slot] ELSE 0]
  /\ retiredIdentities' =
       IF staysAlive THEN retiredIdentities ELSE retiredIdentities \cup {oldId}
  /\ lastOp' = "release"
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       nextIdentity, nextEpoch, issuedIdentities, burnedIdentities,
       logicalAlive, snapshotValid, snapshotPresent, snapshotIdentity, snapshotEpoch
     >>

CaptureSnapshot ==
  /\ snapshotValid' = TRUE
  /\ snapshotPresent' = physicalAlive
  /\ snapshotIdentity' = [s \in Slots |-> IF physicalAlive[s] THEN identity[s] ELSE 0]
  /\ snapshotEpoch' = [s \in Slots |-> IF physicalAlive[s] THEN allocEpoch[s] ELSE 0]
  /\ lastOp' = "captureSnapshot"
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       nextIdentity, nextEpoch,
       issuedIdentities, burnedIdentities, retiredIdentities,
       logicalAlive, retained, physicalAlive, identity, allocEpoch
     >>

Next ==
  \/ \E s \in Slots : AttemptCreate(s)
  \/ AttemptDropUnpublished
  \/ \E s \in Slots : RetainSlot(s)
  \/ \E s \in Slots : DestroySlot(s)
  \/ \E s \in Slots : ReleaseSlot(s)
  \/ CaptureSnapshot

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ nextIdentity \in 1..(MaxIdentity + 1)
  /\ nextEpoch \in 1..(MaxEpoch + 1)
  /\ issuedIdentities \in SUBSET Identities
  /\ burnedIdentities \in SUBSET Identities
  /\ retiredIdentities \in SUBSET Identities
  /\ logicalAlive \in [Slots -> BOOLEAN]
  /\ retained \in [Slots -> BOOLEAN]
  /\ physicalAlive \in [Slots -> BOOLEAN]
  /\ identity \in [Slots -> 0..MaxIdentity]
  /\ allocEpoch \in [Slots -> 0..MaxEpoch]
  /\ snapshotValid \in BOOLEAN
  /\ snapshotPresent \in [Slots -> BOOLEAN]
  /\ snapshotIdentity \in [Slots -> 0..MaxIdentity]
  /\ snapshotEpoch \in [Slots -> 0..MaxEpoch]
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes

PhysicalLifetimeAndIdentityExact ==
  \A s \in Slots :
    /\ physicalAlive[s] = (logicalAlive[s] \/ retained[s])
    /\ physicalAlive[s] = (identity[s] > 0)
    /\ physicalAlive[s] = (allocEpoch[s] > 0)

IssuedBurnedAndRetiredDisjoint ==
  /\ issuedIdentities = 1..(nextIdentity - 1)
  /\ burnedIdentities \subseteq issuedIdentities
  /\ retiredIdentities \subseteq issuedIdentities
  /\ burnedIdentities \cap retiredIdentities = {}
  /\ \A s \in Slots :
       physicalAlive[s] =>
         /\ identity[s] \in issuedIdentities
         /\ identity[s] \notin burnedIdentities
         /\ identity[s] \notin retiredIdentities

PhysicalIdentitiesAreUnique ==
  \A s1, s2 \in Slots :
    (physicalAlive[s1] /\ physicalAlive[s2] /\ identity[s1] = identity[s2]) =>
      s1 = s2

\* Eliminates the same-slot reuse blind spot: matching identity implies BOTH
\* the same slot AND the exact same physical-instance epoch.
SnapshotsNeverAliasDifferentAllocations ==
  snapshotValid =>
    /\ \A s \in Slots :
         snapshotPresent[s] =>
           /\ snapshotIdentity[s] \in issuedIdentities
           /\ snapshotIdentity[s] \notin burnedIdentities
           /\ snapshotEpoch[s] \in Epochs
    /\ \A cur, snap \in Slots :
         (physicalAlive[cur] /\ snapshotPresent[snap]
          /\ identity[cur] = snapshotIdentity[snap]) =>
           /\ cur = snap
           /\ allocEpoch[cur] = snapshotEpoch[snap]

Safety ==
  /\ PhysicalLifetimeAndIdentityExact
  /\ IssuedBurnedAndRetiredDisjoint
  /\ PhysicalIdentitiesAreUnique
  /\ SnapshotsNeverAliasDifferentAllocations

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ nextIdentity' >= nextIdentity
    /\ nextEpoch' >= nextEpoch
    /\ issuedIdentities \subseteq issuedIdentities'
    /\ burnedIdentities \subseteq burnedIdentities'
    /\ retiredIdentities \subseteq retiredIdentities'
    /\ (lastOp' = "dropUnpublished" /\ lastOutcome' = "accepted" =>
          /\ nextIdentity' = nextIdentity + 1
          /\ nextIdentity \in burnedIdentities')
    /\ \A s \in Slots :
         physicalAlive[s] /\ physicalAlive'[s] =>
           /\ identity'[s] = identity[s]
           /\ allocEpoch'[s] = allocEpoch[s]
  ]_vars

=============================================================================
