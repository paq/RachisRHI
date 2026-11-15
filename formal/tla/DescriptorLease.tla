-------------------------- MODULE DescriptorLease ---------------------------
(***************************************************************************)
(* Reference-counted descriptor slot leases across lazy backend pools,     *)
(* non-aliasing physical slot ownership, and owning aggregate allocation   *)
(* snapshots.                                                              *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - include/Rachis/Diagnostics.h (BackendAllocationSnapshot)            *)
(*   - src/D3D12/D3D12Descriptor.cpp (CpuDescriptorPool,                   *)
(*     CpuDescriptorLease, GpuDescriptorHeap, GpuDescriptorLease)          *)
(*   - src/D3D12/D3D12Device.cpp (D3D12Device::                            *)
(*     captureBackendAllocationSnapshot, which sums the ResourceManager,   *)
(*     ViewManager, SamplerManager, and BindingManager                     *)
(*     descriptorAllocationStatistics)                                     *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Two lazy capacity-2 descriptor pools (Pools = {0, 1},               *)
(*     Slots = {0, 1, 2, 3}) lease slots to RHI objects                    *)
(*     (Objects = {0, 1}) that may be retained by command/view consumers   *)
(*     (Consumers = {0, 1}).                                               *)
(*   - When DeferFreeUntilConsumersRelease = TRUE, destroying a logical    *)
(*     object keeps its physical record and descriptor slot leased until   *)
(*     all retaining consumers release, preventing premature slot reuse.   *)
(*   - BackendAllocationSnapshot captures aggregate liveDescriptorCount    *)
(*     and reservedDescriptorCount as an owning value that remains         *)
(*     immutable across subsequent lease creations, releases, or pool      *)
(*     initializations until recaptured.                                   *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Contiguous multi-descriptor range fragmentation in                  *)
(*     GpuDescriptorHeap::allocate (D3D12Descriptor.cpp); this model       *)
(*     tracks unit-slot leases.                                            *)
(*   - Caller synchronization required when reading sequential manager     *)
(*     counters as a single device instant: D3D12Device::                  *)
(*     captureBackendAllocationSnapshot (D3D12Device.cpp) reads each       *)
(*     manager's statistics in turn.                                       *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANT DeferFreeUntilConsumersRelease

ASSUME DeferFreeUntilConsumersRelease \in BOOLEAN

Objects == {0, 1}
Consumers == {0, 1}
Pools == {0, 1}
Slots == {0, 1, 2, 3}
Actions == {"none", "mutate", "capture"}

VARIABLES
    logicalAlive,           \* [Objects -> BOOLEAN] public RHI handle alive
    physicalAlive,          \* [Objects -> BOOLEAN] physical descriptor lease
    slotOf,                 \* [Objects -> Slots] slot leased by object
    slotAllocated,          \* [Slots -> BOOLEAN] slot currently leased
    consumerAlive,          \* [Consumers -> BOOLEAN] active retaining consumer
    consumerOwner,          \* [Consumers -> Objects] object retained
    poolInitialized,        \* [Pools -> BOOLEAN] lazy pool backing allocated
    snapshotValid,          \* BOOLEAN: BackendAllocationSnapshot captured
    snapshotLiveCount,      \* 0..Cardinality(Slots)
    snapshotReservedCount,  \* 0..Cardinality(Slots)
    lastAction              \* Actions

vars == <<logicalAlive, physicalAlive, slotOf, slotAllocated, consumerAlive,
          consumerOwner, poolInitialized, snapshotValid, snapshotLiveCount,
          snapshotReservedCount, lastAction>>

PoolForSlot(s) == IF s < 2 THEN 0 ELSE 1

SlotsForPool(p) == {s \in Slots : PoolForSlot(s) = p}

HasConsumer(obj) ==
    \E c \in Consumers : consumerAlive[c] /\ consumerOwner[c] = obj

PoolLiveCount(p) ==
    Cardinality({s \in SlotsForPool(p) : slotAllocated[s]})

PoolReservedCount(p) ==
    IF poolInitialized[p] THEN Cardinality(SlotsForPool(p)) ELSE 0

LiveDescriptorCount ==
    Cardinality({s \in Slots : slotAllocated[s]})

ReservedDescriptorCount ==
    PoolReservedCount(0) + PoolReservedCount(1)

TypeOK ==
    /\ logicalAlive \in [Objects -> BOOLEAN]
    /\ physicalAlive \in [Objects -> BOOLEAN]
    /\ slotOf \in [Objects -> Slots]
    /\ slotAllocated \in [Slots -> BOOLEAN]
    /\ consumerAlive \in [Consumers -> BOOLEAN]
    /\ consumerOwner \in [Consumers -> Objects]
    /\ poolInitialized \in [Pools -> BOOLEAN]
    /\ snapshotValid \in BOOLEAN
    /\ snapshotLiveCount \in 0..Cardinality(Slots)
    /\ snapshotReservedCount \in 0..Cardinality(Slots)
    /\ lastAction \in Actions

Init ==
    /\ logicalAlive = [o \in Objects |-> FALSE]
    /\ physicalAlive = [o \in Objects |-> FALSE]
    /\ slotOf = [o \in Objects |-> 0]
    /\ slotAllocated = [s \in Slots |-> FALSE]
    /\ consumerAlive = [c \in Consumers |-> FALSE]
    /\ consumerOwner = [c \in Consumers |-> 0]
    /\ poolInitialized = [p \in Pools |-> FALSE]
    /\ snapshotValid = FALSE
    /\ snapshotLiveCount = 0
    /\ snapshotReservedCount = 0
    /\ lastAction = "none"

Create(obj, s) ==
    /\ ~logicalAlive[obj]
    /\ ~physicalAlive[obj]
    /\ ~HasConsumer(obj)
    /\ ~slotAllocated[s]
    /\ logicalAlive' = [logicalAlive EXCEPT ![obj] = TRUE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![obj] = TRUE]
    /\ slotOf' = [slotOf EXCEPT ![obj] = s]
    /\ slotAllocated' = [slotAllocated EXCEPT ![s] = TRUE]
    /\ poolInitialized' = [poolInitialized EXCEPT ![PoolForSlot(s)] = TRUE]
    /\ lastAction' = "mutate"
    /\ UNCHANGED <<consumerAlive, consumerOwner, snapshotValid,
                   snapshotLiveCount, snapshotReservedCount>>

Retain(c, obj) ==
    /\ logicalAlive[obj]
    /\ physicalAlive[obj]
    /\ ~consumerAlive[c]
    /\ consumerAlive' = [consumerAlive EXCEPT ![c] = TRUE]
    /\ consumerOwner' = [consumerOwner EXCEPT ![c] = obj]
    /\ lastAction' = "mutate"
    /\ UNCHANGED <<logicalAlive, physicalAlive, slotOf, slotAllocated,
                   poolInitialized, snapshotValid, snapshotLiveCount,
                   snapshotReservedCount>>

Destroy(obj) ==
    /\ logicalAlive[obj]
    /\ LET keepPhysical == DeferFreeUntilConsumersRelease /\ HasConsumer(obj)
       IN /\ logicalAlive' = [logicalAlive EXCEPT ![obj] = FALSE]
          /\ physicalAlive' = [physicalAlive EXCEPT ![obj] = keepPhysical]
          /\ slotAllocated' =
               [s \in Slots |->
                  IF s = slotOf[obj] THEN keepPhysical ELSE slotAllocated[s]]
          /\ lastAction' = "mutate"
          /\ UNCHANGED <<slotOf, consumerAlive, consumerOwner, poolInitialized,
                         snapshotValid, snapshotLiveCount, snapshotReservedCount>>

Release(c) ==
    /\ consumerAlive[c]
    /\ LET remConsumers == {other \in Consumers : other # c /\ consumerAlive[other]}
           nextPhysical ==
               [o \in Objects |->
                  physicalAlive[o]
                  /\ (logicalAlive[o]
                      \/ \E other \in remConsumers : consumerOwner[other] = o)]
       IN /\ consumerAlive' = [consumerAlive EXCEPT ![c] = FALSE]
          /\ consumerOwner' = [consumerOwner EXCEPT ![c] = 0]
          /\ physicalAlive' = nextPhysical
          /\ slotAllocated' =
               [s \in Slots |->
                  \E o \in Objects : nextPhysical[o] /\ slotOf[o] = s]
          /\ lastAction' = "mutate"
          /\ UNCHANGED <<logicalAlive, slotOf, poolInitialized, snapshotValid,
                         snapshotLiveCount, snapshotReservedCount>>

Capture ==
    /\ snapshotValid' = TRUE
    /\ snapshotLiveCount' = LiveDescriptorCount
    /\ snapshotReservedCount' = ReservedDescriptorCount
    /\ lastAction' = "capture"
    /\ UNCHANGED <<logicalAlive, physicalAlive, slotOf, slotAllocated,
                   consumerAlive, consumerOwner, poolInitialized>>

Next ==
    \/ \E o \in Objects, s \in Slots : Create(o, s)
    \/ \E o \in Objects : Destroy(o)
    \/ \E c \in Consumers, o \in Objects : Retain(c, o)
    \/ \E c \in Consumers : Release(c)
    \/ Capture

Spec == Init /\ [][Next]_vars

LogicalObjectsArePhysical ==
    \A o \in Objects : logicalAlive[o] => physicalAlive[o]

PhysicalObjectsBijectAllocatedSlots ==
    /\ (\A o \in Objects : physicalAlive[o] => slotAllocated[slotOf[o]])
    /\ (\A o1, o2 \in Objects :
          (physicalAlive[o1] /\ physicalAlive[o2] /\ o1 # o2) =>
              slotOf[o1] # slotOf[o2])
    /\ LiveDescriptorCount = Cardinality({o \in Objects : physicalAlive[o]})

\* Every active consumer retains a physically alive owner whose leased
\* descriptor slot remains allocated and unaliased.
ConsumersRetainUnaliasedDescriptors ==
    \A c \in Consumers :
        consumerAlive[c] =>
            LET owner == consumerOwner[c] IN
            /\ physicalAlive[owner]
            /\ slotAllocated[slotOf[owner]]
            /\ \A other \in Objects :
                   (physicalAlive[other] /\ other # owner) =>
                       slotOf[other] # slotOf[owner]

DescriptorAccountingSound ==
    /\ \A p \in Pools :
           /\ PoolLiveCount(p) <= PoolReservedCount(p)
           /\ (PoolLiveCount(p) > 0 => poolInitialized[p])
    /\ LiveDescriptorCount = PoolLiveCount(0) + PoolLiveCount(1)
    /\ LiveDescriptorCount <= ReservedDescriptorCount
    /\ (LiveDescriptorCount > 0 => \E p \in Pools : poolInitialized[p])
    /\ (snapshotValid =>
           /\ snapshotLiveCount <= snapshotReservedCount
           /\ (snapshotLiveCount > 0 => snapshotReservedCount > 0))

Safety ==
    /\ LogicalObjectsArePhysical
    /\ PhysicalObjectsBijectAllocatedSlots
    /\ ConsumersRetainUnaliasedDescriptors
    /\ DescriptorAccountingSound

StepSafety ==
    [][(snapshotValid /\ lastAction' # "capture") =>
          /\ snapshotLiveCount' = snapshotLiveCount
          /\ snapshotReservedCount' = snapshotReservedCount]_vars
=============================================================================
