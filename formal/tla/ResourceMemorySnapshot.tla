----------------------- MODULE ResourceMemorySnapshot -----------------------
(***************************************************************************)
(* Point-in-time ResourceMemorySnapshot accuracy (partitioning allocator-  *)
(* owned resources and ResourceHeap allocations into live vs retained      *)
(* counts and bytes while excluding imported and placed resources) and     *)
(* snapshot immutability across subsequent destroy, release, and reuse.    *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Resource.cpp                                         *)
(*     (ResourceManager::captureMemorySnapshot, trackResourceMemory        *)
(*     weak-pointer memoryTracking table, ResourceMemoryKind::resourceHeap *)
(*     inclusion, imported and placed resource exclusion)                  *)
(*   - src/Device.cpp (Device::captureResourceMemorySnapshot forwarding)   *)
(*   - include/Rachis/Resource.h (ResourceMemorySnapshot,                  *)
(*     ResourceMemoryEntry, ResourceMemoryKind)                            *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Slot 1 (4B resource: owned, imported, or placed in Heap Slot 3),      *)
(*   Slot 2 (8B resource: owned or imported), and Slot 3 (16B owned        *)
(*   ResourceHeap) support slot reuse and verify ground-truth capture      *)
(*   accuracy plus snapshot immutability under later mutations.            *)
(***************************************************************************)

EXTENDS Naturals

ResourceSlotOne == 1
ResourceSlotTwo == 2
HeapSlot == 3
Slots == {ResourceSlotOne, ResourceSlotTwo, HeapSlot}
AllocKinds == {"none", "owned", "imported", "placed"}
MaxTotalBytes == 28
Operations == {
  "none", "createOwned", "importExternal", "createPlaced",
  "createHeap", "retain", "destroy", "release", "capture"
}

SizeOf(s) ==
  CASE s = ResourceSlotOne -> 4
    [] s = ResourceSlotTwo -> 8
    [] s = HeapSlot -> 16

CountOfSet(S) ==
  (IF ResourceSlotOne \in S THEN 1 ELSE 0)
    + (IF ResourceSlotTwo \in S THEN 1 ELSE 0)
    + (IF HeapSlot \in S THEN 1 ELSE 0)

BytesOfSet(S) ==
  (IF ResourceSlotOne \in S THEN SizeOf(ResourceSlotOne) ELSE 0)
    + (IF ResourceSlotTwo \in S THEN SizeOf(ResourceSlotTwo) ELSE 0)
    + (IF HeapSlot \in S THEN SizeOf(HeapSlot) ELSE 0)

VARIABLES
  logicalAlive,          \* Public handle liveness per slot
  retained,              \* Command/descriptor retention per slot
  physicalAlive,         \* Native storage liveness per slot
  allocKind,             \* Provenance: "none", "owned", "imported", or "placed"
  snapshotValid,         \* Whether a snapshot has been captured
  snapshotEntries,       \* Set of slots recorded in latest snapshot
  snapshotLive,          \* Subset of snapshotEntries logically alive at capture
  snapshotRetained,      \* Subset of snapshotEntries retained-only at capture
  snapshotLiveCount,     \* liveAllocationCount of latest snapshot
  snapshotLiveBytes,     \* liveAllocationBytes of latest snapshot
  snapshotRetainedCount, \* retainedAllocationCount of latest snapshot
  snapshotRetainedBytes, \* retainedAllocationBytes of latest snapshot
  expectedLive,          \* Ground-truth owned live set frozen at capture time
  expectedRetained,      \* Ground-truth owned retained set frozen at capture time
  lastOp                 \* Most recent operation kind

snapshotVars == <<
  snapshotValid, snapshotEntries, snapshotLive, snapshotRetained,
  snapshotLiveCount, snapshotLiveBytes,
  snapshotRetainedCount, snapshotRetainedBytes,
  expectedLive, expectedRetained
>>

vars == <<
  logicalAlive, retained, physicalAlive, allocKind,
  snapshotValid, snapshotEntries, snapshotLive, snapshotRetained,
  snapshotLiveCount, snapshotLiveBytes,
  snapshotRetainedCount, snapshotRetainedBytes,
  expectedLive, expectedRetained, lastOp
>>

CurrentOwnedLive ==
  {s \in Slots : physicalAlive[s] /\ allocKind[s] = "owned" /\ logicalAlive[s]}

CurrentOwnedRetained ==
  {s \in Slots : physicalAlive[s] /\ allocKind[s] = "owned" /\ ~logicalAlive[s]}

PlacedChildActive(physMap, kindMap) ==
  physMap[ResourceSlotOne] /\ kindMap[ResourceSlotOne] = "placed"

RecomputePhysical(logMap, retMap, kindMap) ==
  LET s1Phys == logMap[ResourceSlotOne] \/ retMap[ResourceSlotOne]
      s2Phys == logMap[ResourceSlotTwo] \/ retMap[ResourceSlotTwo]
      s1Kind == IF s1Phys THEN kindMap[ResourceSlotOne] ELSE "none"
      heapPhys == logMap[HeapSlot] \/ retMap[HeapSlot] \/ (s1Phys /\ s1Kind = "placed")
  IN [s \in Slots |->
        CASE s = ResourceSlotOne -> s1Phys
          [] s = ResourceSlotTwo -> s2Phys
          [] s = HeapSlot -> heapPhys]

RecomputeKind(nextPhys, kindMap) ==
  [s \in Slots |-> IF nextPhys[s] THEN kindMap[s] ELSE "none"]

Init ==
  /\ logicalAlive = [s \in Slots |-> FALSE]
  /\ retained = [s \in Slots |-> FALSE]
  /\ physicalAlive = [s \in Slots |-> FALSE]
  /\ allocKind = [s \in Slots |-> "none"]
  /\ snapshotValid = FALSE
  /\ snapshotEntries = {}
  /\ snapshotLive = {}
  /\ snapshotRetained = {}
  /\ snapshotLiveCount = 0
  /\ snapshotLiveBytes = 0
  /\ snapshotRetainedCount = 0
  /\ snapshotRetainedBytes = 0
  /\ expectedLive = {}
  /\ expectedRetained = {}
  /\ lastOp = "none"

CreateOwnedResource(s) ==
  /\ s \in {ResourceSlotOne, ResourceSlotTwo}
  /\ ~physicalAlive[s]
  /\ logicalAlive' = [logicalAlive EXCEPT ![s] = TRUE]
  /\ retained' = [retained EXCEPT ![s] = FALSE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![s] = TRUE]
  /\ allocKind' = [allocKind EXCEPT ![s] = "owned"]
  /\ lastOp' = "createOwned"
  /\ UNCHANGED snapshotVars

ImportExternalResource(s) ==
  /\ s \in {ResourceSlotOne, ResourceSlotTwo}
  /\ ~physicalAlive[s]
  /\ logicalAlive' = [logicalAlive EXCEPT ![s] = TRUE]
  /\ retained' = [retained EXCEPT ![s] = FALSE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![s] = TRUE]
  /\ allocKind' = [allocKind EXCEPT ![s] = "imported"]
  /\ lastOp' = "importExternal"
  /\ UNCHANGED snapshotVars

CreateOwnedHeap ==
  /\ ~physicalAlive[HeapSlot]
  /\ logicalAlive' = [logicalAlive EXCEPT ![HeapSlot] = TRUE]
  /\ retained' = [retained EXCEPT ![HeapSlot] = FALSE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![HeapSlot] = TRUE]
  /\ allocKind' = [allocKind EXCEPT ![HeapSlot] = "owned"]
  /\ lastOp' = "createHeap"
  /\ UNCHANGED snapshotVars

CreatePlacedResource ==
  /\ ~physicalAlive[ResourceSlotOne]
  /\ logicalAlive[HeapSlot]
  /\ logicalAlive' = [logicalAlive EXCEPT ![ResourceSlotOne] = TRUE]
  /\ retained' = [retained EXCEPT ![ResourceSlotOne] = FALSE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![ResourceSlotOne] = TRUE]
  /\ allocKind' = [allocKind EXCEPT ![ResourceSlotOne] = "placed"]
  /\ lastOp' = "createPlaced"
  /\ UNCHANGED snapshotVars

RetainSlot(s) ==
  /\ logicalAlive[s]
  /\ ~retained[s]
  /\ retained' = [retained EXCEPT ![s] = TRUE]
  /\ lastOp' = "retain"
  /\ UNCHANGED <<logicalAlive, physicalAlive, allocKind>>
  /\ UNCHANGED snapshotVars

DestroySlot(s) ==
  LET nextLog == [logicalAlive EXCEPT ![s] = FALSE]
      nextPhys == RecomputePhysical(nextLog, retained, allocKind)
      nextKind == RecomputeKind(nextPhys, allocKind)
  IN
  /\ logicalAlive[s]
  /\ logicalAlive' = nextLog
  /\ physicalAlive' = nextPhys
  /\ allocKind' = nextKind
  /\ lastOp' = "destroy"
  /\ UNCHANGED retained
  /\ UNCHANGED snapshotVars

ReleaseSlot(s) ==
  LET nextRet == [retained EXCEPT ![s] = FALSE]
      nextPhys == RecomputePhysical(logicalAlive, nextRet, allocKind)
      nextKind == RecomputeKind(nextPhys, allocKind)
  IN
  /\ retained[s]
  /\ retained' = nextRet
  /\ physicalAlive' = nextPhys
  /\ allocKind' = nextKind
  /\ lastOp' = "release"
  /\ UNCHANGED logicalAlive
  /\ UNCHANGED snapshotVars

CaptureSnapshot ==
  LET live == CurrentOwnedLive
      ret == CurrentOwnedRetained
  IN
  /\ snapshotValid' = TRUE
  /\ snapshotLive' = live
  /\ snapshotRetained' = ret
  /\ snapshotEntries' = live \cup ret
  /\ snapshotLiveCount' = CountOfSet(live)
  /\ snapshotLiveBytes' = BytesOfSet(live)
  /\ snapshotRetainedCount' = CountOfSet(ret)
  /\ snapshotRetainedBytes' = BytesOfSet(ret)
  /\ expectedLive' = live
  /\ expectedRetained' = ret
  /\ lastOp' = "capture"
  /\ UNCHANGED <<logicalAlive, retained, physicalAlive, allocKind>>

Next ==
  \/ \E s \in {ResourceSlotOne, ResourceSlotTwo} : CreateOwnedResource(s)
  \/ \E s \in {ResourceSlotOne, ResourceSlotTwo} : ImportExternalResource(s)
  \/ CreateOwnedHeap
  \/ CreatePlacedResource
  \/ \E s \in Slots : RetainSlot(s)
  \/ \E s \in Slots : DestroySlot(s)
  \/ \E s \in Slots : ReleaseSlot(s)
  \/ CaptureSnapshot

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ logicalAlive \in [Slots -> BOOLEAN]
  /\ retained \in [Slots -> BOOLEAN]
  /\ physicalAlive \in [Slots -> BOOLEAN]
  /\ allocKind \in [Slots -> AllocKinds]
  /\ snapshotValid \in BOOLEAN
  /\ snapshotEntries \in SUBSET Slots
  /\ snapshotLive \in SUBSET Slots
  /\ snapshotRetained \in SUBSET Slots
  /\ snapshotLiveCount \in 0..3
  /\ snapshotLiveBytes \in 0..MaxTotalBytes
  /\ snapshotRetainedCount \in 0..3
  /\ snapshotRetainedBytes \in 0..MaxTotalBytes
  /\ expectedLive \in SUBSET Slots
  /\ expectedRetained \in SUBSET Slots
  /\ lastOp \in Operations

PhysicalLifetimeAndProvenanceExact ==
  /\ physicalAlive = RecomputePhysical(logicalAlive, retained, allocKind)
  /\ \A s \in Slots : physicalAlive[s] = (allocKind[s] # "none")
  /\ PlacedChildActive(physicalAlive, allocKind) => physicalAlive[HeapSlot]
  /\ allocKind[HeapSlot] \in {"none", "owned"}
  /\ allocKind[ResourceSlotTwo] \in {"none", "owned", "imported"}

SnapshotMatchesGroundTruthAndPartition ==
  snapshotValid =>
    /\ snapshotLive = expectedLive
    /\ snapshotRetained = expectedRetained
    /\ snapshotEntries = snapshotLive \cup snapshotRetained
    /\ snapshotLive \cap snapshotRetained = {}
    /\ snapshotLiveCount = CountOfSet(snapshotLive)
    /\ snapshotLiveBytes = BytesOfSet(snapshotLive)
    /\ snapshotRetainedCount = CountOfSet(snapshotRetained)
    /\ snapshotRetainedBytes = BytesOfSet(snapshotRetained)

Safety ==
  /\ PhysicalLifetimeAndProvenanceExact
  /\ SnapshotMatchesGroundTruthAndPartition

StepSafety ==
  [][
    /\ (lastOp' = "capture" =>
          /\ snapshotValid'
          /\ expectedLive' = CurrentOwnedLive
          /\ expectedRetained' = CurrentOwnedRetained
          /\ snapshotLive' = CurrentOwnedLive
          /\ snapshotRetained' = CurrentOwnedRetained
          /\ \A s \in Slots :
               (s \in snapshotEntries') <=>
                 (physicalAlive[s] /\ allocKind[s] = "owned"))
    /\ (lastOp' # "capture" => snapshotVars' = snapshotVars)
    /\ \A s \in Slots :
         physicalAlive[s] /\ physicalAlive'[s] => allocKind'[s] = allocKind[s]
  ]_vars

=============================================================================
