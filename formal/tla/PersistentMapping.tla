------------------------- MODULE PersistentMapping --------------------------
(***************************************************************************)
(* Persistent host-visible buffer mappings, subrange cache synchronization *)
(* (flush/invalidate), submission across live mappings, and deferred       *)
(* physical retention across logical destruction.                          *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - include/Rachis/Device.h (DeviceCapabilities::persistentMapping)     *)
(*   - src/D3D12/D3D12Resource.cpp (mapBuffer, unmapBuffer, destroyBuffer, *)
(*     flushMappedMemoryRanges, invalidateMappedMemoryRanges,              *)
(*     synchronizeMappedMemoryRanges)                                      *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::submit and retainResource:  *)
(*     submission across live mappings and command retention of logically  *)
(*     destroyed buffers)                                                  *)
(*   - persistentMapping capability reporting: src/D3D12/D3D12Device.cpp,  *)
(*     src/Vulkan/VulkanRuntime.cpp (publishMemoryCapabilities),           *)
(*     src/Metal/MetalBootstrap.mm                                         *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two two-cell buffers ("upload" and "readback") and three payload      *)
(*   values cover full and partial subrange mappings, out-of-bounds and    *)
(*   unmapped cache-sync rejection, capability-gated mapped submission,    *)
(*   and physical retention while GPU work is in flight. Hardware PCIe/UMA *)
(*   cache lines and native GPU copy execution are trusted to the driver.  *)
(***************************************************************************)

EXTENDS Naturals

Buffers == {"upload", "readback"}
Cells == {0, 1}
RequestRanges == {{}, {0}, {1}, {0, 1}, {2}}
Values == {0, 1, 2}
Operations == {
  "none", "map", "unmap", "flush", "invalidate",
  "destroy", "write", "submit", "complete", "read"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  persistentMapping, \* Reported DeviceCapabilities::persistentMapping
  logicalAlive,      \* Public BufferHandle validity per buffer
  physicalAlive,     \* Native storage alive (logical handle or in-flight use)
  mappedRange,       \* Active mapped cell set per buffer ({} when unmapped)
  uploadValue,       \* Latest host payload written to upload buffer
  readbackValue,     \* Latest GPU payload retired into readback buffer
  submittedValue,    \* Upload payload captured at queue submission
  inFlight,          \* Whether a GPU copy from upload to readback is active
  completionReady,   \* Whether at least one submission has completed
  hostObserved,      \* Latest value read by the host from readback memory
  lastOp,            \* Most recent operation kind
  lastOutcome,       \* Outcome of most recent operation
  lastBuffer,        \* Target buffer of most recent buffer operation
  lastRange          \* Requested cell range of most recent range operation

coreVars == <<
  persistentMapping, logicalAlive, physicalAlive, mappedRange,
  uploadValue, readbackValue, submittedValue,
  inFlight, completionReady, hostObserved
>>

auditVars == <<lastOp, lastOutcome, lastBuffer, lastRange>>

vars == <<
  persistentMapping, logicalAlive, physicalAlive, mappedRange,
  uploadValue, readbackValue, submittedValue,
  inFlight, completionReady, hostObserved,
  lastOp, lastOutcome, lastBuffer, lastRange
>>

ValidSubrange(range) ==
  /\ range # {}
  /\ range \subseteq Cells

DirectionValid(op, buffer) ==
  \/ (op = "flush" /\ buffer = "upload")
  \/ (op = "invalidate" /\ buffer = "readback")

CanMap(buffer, range) ==
  /\ logicalAlive[buffer]
  /\ mappedRange[buffer] = {}
  /\ ValidSubrange(range)

CanUnmap(buffer, range) ==
  /\ logicalAlive[buffer]
  /\ mappedRange[buffer] # {}
  /\ ValidSubrange(range)

CanCacheSync(op, buffer, range) ==
  /\ logicalAlive[buffer]
  /\ mappedRange[buffer] # {}
  /\ DirectionValid(op, buffer)
  /\ ValidSubrange(range)
  /\ range \subseteq mappedRange[buffer]

CanDestroy(buffer) ==
  /\ logicalAlive[buffer]
  /\ mappedRange[buffer] = {}

CanWrite ==
  /\ logicalAlive["upload"]
  /\ mappedRange["upload"] # {}
  /\ ~inFlight

CanSubmit ==
  /\ logicalAlive["upload"]
  /\ logicalAlive["readback"]
  /\ ~inFlight
  /\ (persistentMapping \/ (mappedRange["upload"] = {} /\ mappedRange["readback"] = {}))

CanRead ==
  /\ logicalAlive["readback"]
  /\ mappedRange["readback"] # {}
  /\ completionReady
  /\ ~inFlight

Init ==
  /\ persistentMapping \in BOOLEAN
  /\ logicalAlive = [b \in Buffers |-> TRUE]
  /\ physicalAlive = [b \in Buffers |-> TRUE]
  /\ mappedRange = [b \in Buffers |-> {}]
  /\ uploadValue = 0
  /\ readbackValue = 0
  /\ submittedValue = 0
  /\ inFlight = FALSE
  /\ completionReady = FALSE
  /\ hostObserved = 0
  /\ lastOp = "none"
  /\ lastOutcome = "none"
  /\ lastBuffer = "upload"
  /\ lastRange = {}

AttemptMap(buffer, range) ==
  /\ IF CanMap(buffer, range)
     THEN /\ mappedRange' = [mappedRange EXCEPT ![buffer] = range]
          /\ UNCHANGED <<
               persistentMapping, logicalAlive, physicalAlive,
               uploadValue, readbackValue, submittedValue,
               inFlight, completionReady, hostObserved
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "map"
  /\ lastBuffer' = buffer
  /\ lastRange' = range

AttemptUnmap(buffer, range) ==
  /\ IF CanUnmap(buffer, range)
     THEN /\ mappedRange' = [mappedRange EXCEPT ![buffer] = {}]
          /\ UNCHANGED <<
               persistentMapping, logicalAlive, physicalAlive,
               uploadValue, readbackValue, submittedValue,
               inFlight, completionReady, hostObserved
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "unmap"
  /\ lastBuffer' = buffer
  /\ lastRange' = range

AttemptCacheSync(op, buffer, range) ==
  /\ UNCHANGED coreVars
  /\ lastOutcome' = IF CanCacheSync(op, buffer, range) THEN "accepted" ELSE "rejected"
  /\ lastOp' = op
  /\ lastBuffer' = buffer
  /\ lastRange' = range

AttemptDestroy(buffer) ==
  /\ IF CanDestroy(buffer)
     THEN /\ logicalAlive' = [logicalAlive EXCEPT ![buffer] = FALSE]
          /\ physicalAlive' = [physicalAlive EXCEPT ![buffer] = inFlight]
          /\ UNCHANGED <<
               persistentMapping, mappedRange,
               uploadValue, readbackValue, submittedValue,
               inFlight, completionReady, hostObserved
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "destroy"
  /\ lastBuffer' = buffer
  /\ lastRange' = {}

AttemptHostWrite(value) ==
  /\ IF CanWrite
     THEN /\ uploadValue' = value
          /\ UNCHANGED <<
               persistentMapping, logicalAlive, physicalAlive, mappedRange,
               readbackValue, submittedValue,
               inFlight, completionReady, hostObserved
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "write"
  /\ lastBuffer' = "upload"
  /\ lastRange' = {}

AttemptSubmit ==
  /\ IF CanSubmit
     THEN /\ submittedValue' = uploadValue
          /\ inFlight' = TRUE
          /\ completionReady' = FALSE
          /\ UNCHANGED <<
               persistentMapping, logicalAlive, physicalAlive, mappedRange,
               uploadValue, readbackValue, hostObserved
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "submit"
  /\ lastBuffer' = "upload"
  /\ lastRange' = {}

CompleteSubmission ==
  /\ inFlight
  /\ physicalAlive' = [b \in Buffers |-> logicalAlive[b]]
  /\ readbackValue' = submittedValue
  /\ inFlight' = FALSE
  /\ completionReady' = TRUE
  /\ UNCHANGED <<
       persistentMapping, logicalAlive, mappedRange,
       uploadValue, submittedValue, hostObserved
     >>
  /\ lastOp' = "complete"
  /\ lastOutcome' = "accepted"
  /\ lastBuffer' = "readback"
  /\ lastRange' = {}

AttemptHostRead ==
  /\ IF CanRead
     THEN /\ hostObserved' = readbackValue
          /\ UNCHANGED <<
               persistentMapping, logicalAlive, physicalAlive, mappedRange,
               uploadValue, readbackValue, submittedValue,
               inFlight, completionReady
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "read"
  /\ lastBuffer' = "readback"
  /\ lastRange' = {}

Next ==
  \/ \E b \in Buffers, r \in RequestRanges :
       \/ AttemptMap(b, r)
       \/ AttemptUnmap(b, r)
  \/ \E op \in {"flush", "invalidate"}, b \in Buffers, r \in RequestRanges :
       AttemptCacheSync(op, b, r)
  \/ \E b \in Buffers : AttemptDestroy(b)
  \/ \E v \in Values : AttemptHostWrite(v)
  \/ AttemptSubmit
  \/ CompleteSubmission
  \/ AttemptHostRead

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ persistentMapping \in BOOLEAN
  /\ logicalAlive \in [Buffers -> BOOLEAN]
  /\ physicalAlive \in [Buffers -> BOOLEAN]
  /\ mappedRange \in [Buffers -> SUBSET Cells]
  /\ uploadValue \in Values
  /\ readbackValue \in Values
  /\ submittedValue \in Values
  /\ inFlight \in BOOLEAN
  /\ completionReady \in BOOLEAN
  /\ hostObserved \in Values
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes
  /\ lastBuffer \in Buffers
  /\ lastRange \in RequestRanges

PointersRequireLiveBuffers ==
  \A b \in Buffers :
    mappedRange[b] # {} => logicalAlive[b]

PhysicalLifetimeExact ==
  \A b \in Buffers :
    physicalAlive[b] = (logicalAlive[b] \/ inFlight)

AcceptedSubmitCapturesUpload ==
  lastOp = "submit" /\ lastOutcome = "accepted" =>
    /\ inFlight
    /\ ~completionReady
    /\ submittedValue = uploadValue
    /\ (persistentMapping \/ (mappedRange["upload"] = {} /\ mappedRange["readback"] = {}))

CompletionPublishesExactReadback ==
  lastOp = "complete" /\ lastOutcome = "accepted" =>
    /\ ~inFlight
    /\ completionReady
    /\ readbackValue = submittedValue

HostReadsOnlyCompletedData ==
  lastOp = "read" /\ lastOutcome = "accepted" =>
    /\ completionReady
    /\ ~inFlight
    /\ hostObserved = readbackValue

AcceptedUnmapInvalidatesPointer ==
  lastOp = "unmap" /\ lastOutcome = "accepted" =>
    mappedRange[lastBuffer] = {}

Safety ==
  /\ PointersRequireLiveBuffers
  /\ PhysicalLifetimeExact
  /\ AcceptedSubmitCapturesUpload
  /\ CompletionPublishesExactReadback
  /\ HostReadsOnlyCompletedData
  /\ AcceptedUnmapInvalidatesPointer

StepSafety ==
  [][
    /\ persistentMapping' = persistentMapping
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (lastOp' \in {"flush", "invalidate"} /\ lastOutcome' = "accepted" =>
          /\ coreVars' = coreVars
          /\ CanCacheSync(lastOp', lastBuffer', lastRange'))
    /\ (lastOp' = "destroy" /\ mappedRange[lastBuffer'] # {} =>
          lastOutcome' = "rejected")
    /\ (lastOp' = "map" =>
          (lastOutcome' = "accepted" <=> CanMap(lastBuffer', lastRange')))
    /\ (lastOp' = "unmap" =>
          (lastOutcome' = "accepted" <=> CanUnmap(lastBuffer', lastRange')))
    /\ (lastOp' = "submit" =>
          (lastOutcome' = "accepted" <=> CanSubmit))
    /\ (lastOp' = "complete" =>
          (inFlight /\ mappedRange' = mappedRange))
    /\ (lastOp' = "read" =>
          (lastOutcome' = "accepted" <=> CanRead))
  ]_vars

=============================================================================
