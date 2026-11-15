-------------------------- MODULE ResourceLifetime --------------------------
(***************************************************************************)
(* Exact physical resource lifetime under public handle liveness,          *)
(* descriptor/view references, command-recording retention, and in-flight  *)
(* GPU submission retention, plus exclusion of imported external resources *)
(* from allocator-owned memory accounting.                                 *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Resource.cpp (ResourceManager::createBuffer,         *)
(*     createTexture, importSharedBuffer, importSharedTexture,             *)
(*     importBuffer, importTexture, destroyBuffer, destroyTexture,         *)
(*     trackResourceMemory memoryTracking registration)                    *)
(*   - src/D3D12/D3D12Queue.cpp (recording/in-flight command retention)    *)
(*   - src/Vulkan/VulkanResources.cpp (VulkanNativeKeepAlive physical      *)
(*     retention; imported records excluded from                           *)
(*     captureResourceMemorySnapshot; destroyed buffers/textures leave     *)
(*     the snapshot at logical destruction, so retained-owner              *)
(*     accounting is not mirrored)                                         *)
(*   - src/Metal/MetalResources.mm (createBuffer, createTexture,           *)
(*     destroyBuffer, destroyTexture, MetalAllocationTelemetry; no         *)
(*     external import path) and src/Metal/MetalCommand.mm                 *)
(*     (retainMetalApplicationResource)                                    *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two resource slots with distinct byte sizes (4 and 8 bytes), bounded  *)
(*   descriptor reference counts, recording/in-flight command leases, and  *)
(*   explicit imported provenance verify both directions of physical       *)
(*   liveness (no premature free and no physical leak) and exact owned     *)
(*   byte/count accounting.                                                *)
(***************************************************************************)

EXTENDS Naturals

Resources == {1, 2}
MaxDescriptorRefs == 2
MaxTotalBytes == 12
Operations == {
  "none", "createOwned", "importExternal", "destroy",
  "retainDescriptor", "releaseDescriptor",
  "recordUse", "abandonRecording", "submit", "collectInFlight"
}

SizeOf(r) == IF r = 1 THEN 4 ELSE 8

VARIABLES
  logicalAlive,    \* Public handle liveness per resource slot
  physicalAlive,   \* Native storage liveness per resource slot
  imported,        \* True iff current physical resource was externally imported
  allocatorOwned,  \* True iff currently tracked in allocator memory accounting
  descriptorRefs,  \* Active descriptor/view/SBT reference count per slot
  recordingRef,    \* Whether an open command buffer retains each slot
  inFlightRef,     \* Whether an in-flight GPU submission retains each slot
  allocationCount, \* Tracked allocator-owned allocation count
  allocationBytes, \* Tracked allocator-owned byte total
  lastOp           \* Most recent operation kind

vars == <<
  logicalAlive, physicalAlive, imported, allocatorOwned,
  descriptorRefs, recordingRef, inFlightRef,
  allocationCount, allocationBytes, lastOp
>>

HasReference(r) ==
  \/ descriptorRefs[r] > 0
  \/ recordingRef[r]
  \/ inFlightRef[r]

OwnedCountOf(ownedMap) ==
  (IF ownedMap[1] THEN 1 ELSE 0) + (IF ownedMap[2] THEN 1 ELSE 0)

OwnedBytesOf(ownedMap) ==
  (IF ownedMap[1] THEN SizeOf(1) ELSE 0) + (IF ownedMap[2] THEN SizeOf(2) ELSE 0)

Init ==
  /\ logicalAlive = [r \in Resources |-> FALSE]
  /\ physicalAlive = [r \in Resources |-> FALSE]
  /\ imported = [r \in Resources |-> FALSE]
  /\ allocatorOwned = [r \in Resources |-> FALSE]
  /\ descriptorRefs = [r \in Resources |-> 0]
  /\ recordingRef = [r \in Resources |-> FALSE]
  /\ inFlightRef = [r \in Resources |-> FALSE]
  /\ allocationCount = 0
  /\ allocationBytes = 0
  /\ lastOp = "none"

CreateOwned(r) ==
  /\ ~physicalAlive[r]
  /\ logicalAlive' = [logicalAlive EXCEPT ![r] = TRUE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![r] = TRUE]
  /\ imported' = [imported EXCEPT ![r] = FALSE]
  /\ allocatorOwned' = [allocatorOwned EXCEPT ![r] = TRUE]
  /\ allocationCount' = allocationCount + 1
  /\ allocationBytes' = allocationBytes + SizeOf(r)
  /\ lastOp' = "createOwned"
  /\ UNCHANGED <<descriptorRefs, recordingRef, inFlightRef>>

ImportExternal(r) ==
  /\ ~physicalAlive[r]
  /\ logicalAlive' = [logicalAlive EXCEPT ![r] = TRUE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![r] = TRUE]
  /\ imported' = [imported EXCEPT ![r] = TRUE]
  /\ allocatorOwned' = [allocatorOwned EXCEPT ![r] = FALSE]
  /\ lastOp' = "importExternal"
  /\ UNCHANGED <<
       descriptorRefs, recordingRef, inFlightRef,
       allocationCount, allocationBytes
     >>

Destroy(r) ==
  LET staysAlive == HasReference(r)
      nextOwned == [allocatorOwned EXCEPT ![r] = (staysAlive /\ allocatorOwned[r])]
  IN
  /\ logicalAlive[r]
  /\ logicalAlive' = [logicalAlive EXCEPT ![r] = FALSE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![r] = staysAlive]
  /\ imported' = [imported EXCEPT ![r] = (staysAlive /\ imported[r])]
  /\ allocatorOwned' = nextOwned
  /\ allocationCount' = OwnedCountOf(nextOwned)
  /\ allocationBytes' = OwnedBytesOf(nextOwned)
  /\ lastOp' = "destroy"
  /\ UNCHANGED <<descriptorRefs, recordingRef, inFlightRef>>

RetainDescriptor(r) ==
  /\ logicalAlive[r]
  /\ descriptorRefs[r] < MaxDescriptorRefs
  /\ descriptorRefs' = [descriptorRefs EXCEPT ![r] = descriptorRefs[r] + 1]
  /\ lastOp' = "retainDescriptor"
  /\ UNCHANGED <<
       logicalAlive, physicalAlive, imported, allocatorOwned,
       recordingRef, inFlightRef, allocationCount, allocationBytes
     >>

ReleaseDescriptor(r) ==
  LET nextDesc == descriptorRefs[r] - 1
      staysAlive == logicalAlive[r] \/ nextDesc > 0 \/ recordingRef[r] \/ inFlightRef[r]
      nextOwned == [allocatorOwned EXCEPT ![r] = (staysAlive /\ allocatorOwned[r])]
  IN
  /\ descriptorRefs[r] > 0
  /\ descriptorRefs' = [descriptorRefs EXCEPT ![r] = nextDesc]
  /\ physicalAlive' = [physicalAlive EXCEPT ![r] = staysAlive]
  /\ imported' = [imported EXCEPT ![r] = (staysAlive /\ imported[r])]
  /\ allocatorOwned' = nextOwned
  /\ allocationCount' = OwnedCountOf(nextOwned)
  /\ allocationBytes' = OwnedBytesOf(nextOwned)
  /\ lastOp' = "releaseDescriptor"
  /\ UNCHANGED <<logicalAlive, recordingRef, inFlightRef>>

RecordUse(r) ==
  /\ logicalAlive[r]
  /\ ~recordingRef[r]
  /\ recordingRef' = [recordingRef EXCEPT ![r] = TRUE]
  /\ lastOp' = "recordUse"
  /\ UNCHANGED <<
       logicalAlive, physicalAlive, imported, allocatorOwned,
       descriptorRefs, inFlightRef, allocationCount, allocationBytes
     >>

AbandonRecording ==
  LET nextPhys ==
        [r \in Resources |-> logicalAlive[r] \/ descriptorRefs[r] > 0 \/ inFlightRef[r]]
      nextOwned == [r \in Resources |-> nextPhys[r] /\ allocatorOwned[r]]
  IN
  /\ \E r \in Resources : recordingRef[r]
  /\ recordingRef' = [r \in Resources |-> FALSE]
  /\ physicalAlive' = nextPhys
  /\ imported' = [r \in Resources |-> nextPhys[r] /\ imported[r]]
  /\ allocatorOwned' = nextOwned
  /\ allocationCount' = OwnedCountOf(nextOwned)
  /\ allocationBytes' = OwnedBytesOf(nextOwned)
  /\ lastOp' = "abandonRecording"
  /\ UNCHANGED <<logicalAlive, descriptorRefs, inFlightRef>>

SubmitRecording ==
  /\ \E r \in Resources : recordingRef[r]
  /\ \A r \in Resources : ~inFlightRef[r]
  /\ recordingRef' = [r \in Resources |-> FALSE]
  /\ inFlightRef' = recordingRef
  /\ lastOp' = "submit"
  /\ UNCHANGED <<
       logicalAlive, physicalAlive, imported, allocatorOwned,
       descriptorRefs, allocationCount, allocationBytes
     >>

CollectInFlight ==
  LET nextPhys ==
        [r \in Resources |-> logicalAlive[r] \/ descriptorRefs[r] > 0 \/ recordingRef[r]]
      nextOwned == [r \in Resources |-> nextPhys[r] /\ allocatorOwned[r]]
  IN
  /\ \E r \in Resources : inFlightRef[r]
  /\ inFlightRef' = [r \in Resources |-> FALSE]
  /\ physicalAlive' = nextPhys
  /\ imported' = [r \in Resources |-> nextPhys[r] /\ imported[r]]
  /\ allocatorOwned' = nextOwned
  /\ allocationCount' = OwnedCountOf(nextOwned)
  /\ allocationBytes' = OwnedBytesOf(nextOwned)
  /\ lastOp' = "collectInFlight"
  /\ UNCHANGED <<logicalAlive, descriptorRefs, recordingRef>>

Next ==
  \/ \E r \in Resources : CreateOwned(r)
  \/ \E r \in Resources : ImportExternal(r)
  \/ \E r \in Resources : Destroy(r)
  \/ \E r \in Resources : RetainDescriptor(r)
  \/ \E r \in Resources : ReleaseDescriptor(r)
  \/ \E r \in Resources : RecordUse(r)
  \/ AbandonRecording
  \/ SubmitRecording
  \/ CollectInFlight

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ logicalAlive \in [Resources -> BOOLEAN]
  /\ physicalAlive \in [Resources -> BOOLEAN]
  /\ imported \in [Resources -> BOOLEAN]
  /\ allocatorOwned \in [Resources -> BOOLEAN]
  /\ descriptorRefs \in [Resources -> 0..MaxDescriptorRefs]
  /\ recordingRef \in [Resources -> BOOLEAN]
  /\ inFlightRef \in [Resources -> BOOLEAN]
  /\ allocationCount \in 0..2
  /\ allocationBytes \in 0..MaxTotalBytes
  /\ lastOp \in Operations

\* Exact equivalence: neither premature physical destruction nor physical leak.
PhysicalLifetimeIsExact ==
  \A r \in Resources :
    physicalAlive[r] = (logicalAlive[r] \/ HasReference(r))

ImportedAndAllocatorOwnedPartitionPhysicalResources ==
  \A r \in Resources :
    /\ (imported[r] => physicalAlive[r])
    /\ (allocatorOwned[r] = (physicalAlive[r] /\ ~imported[r]))

AllocatorAccountingMatchesOwnedResources ==
  /\ allocationCount = OwnedCountOf(allocatorOwned)
  /\ allocationBytes = OwnedBytesOf(allocatorOwned)

Safety ==
  /\ PhysicalLifetimeIsExact
  /\ ImportedAndAllocatorOwnedPartitionPhysicalResources
  /\ AllocatorAccountingMatchesOwnedResources

StepSafety ==
  [][
    /\ \A r \in Resources :
         physicalAlive[r] /\ physicalAlive'[r] =>
           /\ imported'[r] = imported[r]
           /\ allocatorOwned'[r] = allocatorOwned[r]
    /\ (lastOp' = "importExternal" =>
          /\ allocationCount' = allocationCount
          /\ allocationBytes' = allocationBytes)
  ]_vars

=============================================================================
