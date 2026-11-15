------------------------ MODULE RenderPassTimestamp -------------------------
(***************************************************************************)
(* Optional render-pass boundary timestamp writes, transactional           *)
(* attachment/swapchain/pool dependency publication before timestamp state *)
(* mutation, end-slot reservation, unresolved reuse rejection, and         *)
(* aligned 8-byte out-of-pass resolution.                                  *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12CommandRecording.cpp (beginRendering timestamp       *)
(*     validation and transactional dependency publication, endRendering)  *)
(*   - src/D3D12/D3D12Query.cpp (prepareRenderingTimestamps,               *)
(*     commitRenderingTimestamps, endRenderingTimestamps,                  *)
(*     resolveTimestamps)                                                  *)
(*   - src/D3D12/D3D12Queue.cpp (retainCommandDependencies rollback)       *)
(*   - src/Vulkan/VulkanDevice.cpp (beginRendering prepare/commit order)   *)
(*   - src/Vulkan/VulkanGraphics.cpp (attachment retention with reserved   *)
(*     timestamp-pool capacity)                                            *)
(*   - src/Vulkan/VulkanQueries.cpp (prepareRenderingTimestamps,           *)
(*     commitRenderingTimestamps, endRenderingTimestamps,                  *)
(*     resolveTimestamps)                                                  *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Three resources (Pool, Attachment, Destination), optional swapchain   *)
(*   lease, two timestamp query slots, and injected publication failures   *)
(*   at attachment/swapchain/pool stages verify transactional rollback,    *)
(*   end-slot reservation, and deferred physical retirement.               *)
(***************************************************************************)

EXTENDS Integers, FiniteSets

Pool == 0
Attachment == 1
Destination == 2
Resources == {Pool, Attachment, Destination}
Queries == {0, 1}
NoQuery == -1
BoundaryIndices == {NoQuery, 0, 1, 2}
QueryCounts == {0, 1, 2}
DestinationOffsets == {0, 4, 8}
ResultSize == 8
DestinationSize == 16
PublishFaults == {"none", "failAttachment", "failSwapchain", "failPool"}
Operations == {
  "none", "acquireSwapchain", "begin", "end",
  "resolve", "destroy", "submit", "complete", "abandon"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  logicalAlive,          \* Public handle liveness per resource
  physicalAlive,         \* Native object liveness per resource
  recordingReferences,   \* Resources retained by open command recording
  inFlightReferences,    \* Resources retained by in-flight submission
  swapchainAcquired,     \* Whether a swapchain image is currently acquired
  swapchainCommandLease, \* Whether the open recording holds a swapchain lease
  rendering,             \* Whether a render pass is currently open
  timestampScope,        \* Whether the active render pass records timestamps
  activeEndWrite,        \* Reserved end-of-pass query index, or NoQuery
  pendingWrites,         \* Unresolved written/reserved timestamp indices
  inFlight,              \* Whether the command buffer is currently in flight
  lastOp,                \* Most recent operation kind
  lastOutcome,           \* Outcome of most recent operation
  lastResolveRange,      \* Query index set targeted by most recent resolve
  lastResolveOffset      \* Byte offset targeted by most recent resolve

coreVars == <<
  logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
  swapchainAcquired, swapchainCommandLease, rendering,
  timestampScope, activeEndWrite, pendingWrites, inFlight
>>

vars == <<
  logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
  swapchainAcquired, swapchainCommandLease, rendering,
  timestampScope, activeEndWrite, pendingWrites, inFlight,
  lastOp, lastOutcome, lastResolveRange, lastResolveOffset
>>

Referenced(resource) ==
  resource \in recordingReferences \/ resource \in inFlightReferences

QueryRange(firstQuery, queryCount) ==
  IF queryCount = 1 THEN {firstQuery}
  ELSE IF queryCount = 2 THEN {firstQuery, firstQuery + 1}
  ELSE {}

RequestedWrites(beginWrite, endWrite) ==
  {idx \in {beginWrite, endWrite} : idx # NoQuery}

BeginDescriptorValid(attachmentValid, usesSwapchain,
                     hasTimestampWrites, beginWrite, endWrite) ==
  LET writes == RequestedWrites(beginWrite, endWrite) IN
  /\ ~inFlight
  /\ ~rendering
  /\ attachmentValid
  /\ logicalAlive[Attachment]
  /\ (~usesSwapchain \/ swapchainAcquired)
  /\ IF hasTimestampWrites
     THEN /\ logicalAlive[Pool]
          /\ beginWrite \in (Queries \cup {NoQuery})
          /\ endWrite \in (Queries \cup {NoQuery})
          /\ writes # {}
          /\ (beginWrite = NoQuery \/ endWrite = NoQuery \/ beginWrite # endWrite)
          /\ writes \cap pendingWrites = {}
     ELSE TRUE

DependencyPublicationSucceeds(usesSwapchain, hasTimestampWrites, fault) ==
  /\ fault # "failAttachment"
  /\ ~(usesSwapchain /\ fault = "failSwapchain")
  /\ ~(hasTimestampWrites /\ fault = "failPool")

EndValid ==
  /\ ~inFlight
  /\ rendering
  /\ Attachment \in recordingReferences
  /\ (timestampScope => Pool \in recordingReferences /\ physicalAlive[Pool])

ResolveValid(firstQuery, queryCount, destinationOffset, copyDstUsage) ==
  LET querySet == QueryRange(firstQuery, queryCount) IN
  /\ ~inFlight
  /\ ~rendering
  /\ ~timestampScope
  /\ logicalAlive[Pool]
  /\ logicalAlive[Destination]
  /\ copyDstUsage
  /\ queryCount > 0
  /\ firstQuery + queryCount <= Cardinality(Queries)
  /\ querySet \subseteq Queries
  /\ querySet \subseteq pendingWrites
  /\ destinationOffset % ResultSize = 0
  /\ destinationOffset + queryCount * ResultSize <= DestinationSize

Init ==
  /\ logicalAlive = [r \in Resources |-> TRUE]
  /\ physicalAlive = [r \in Resources |-> TRUE]
  /\ recordingReferences = {}
  /\ inFlightReferences = {}
  /\ swapchainAcquired = FALSE
  /\ swapchainCommandLease = FALSE
  /\ rendering = FALSE
  /\ timestampScope = FALSE
  /\ activeEndWrite = NoQuery
  /\ pendingWrites = {}
  /\ inFlight = FALSE
  /\ lastOp = "none"
  /\ lastOutcome = "none"
  /\ lastResolveRange = {}
  /\ lastResolveOffset = 0

AcquireSwapchain ==
  /\ ~swapchainAcquired
  /\ ~inFlight
  /\ swapchainAcquired' = TRUE
  /\ UNCHANGED <<
       logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
       swapchainCommandLease, rendering, timestampScope,
       activeEndWrite, pendingWrites, inFlight
     >>
  /\ lastOp' = "acquireSwapchain"
  /\ lastOutcome' = "accepted"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

\* Attachment, swapchain-lease, and query-pool retention commit as one
\* transaction before rendering or timestamp state is mutated.
AttemptBegin(attachmentValid, usesSwapchain,
             hasTimestampWrites, beginWrite, endWrite, fault) ==
  LET writes == RequestedWrites(beginWrite, endWrite)
      newRefs == IF hasTimestampWrites THEN {Attachment, Pool} ELSE {Attachment}
      canCommit ==
        /\ BeginDescriptorValid(attachmentValid, usesSwapchain,
                                hasTimestampWrites, beginWrite, endWrite)
        /\ DependencyPublicationSucceeds(usesSwapchain, hasTimestampWrites, fault)
  IN
  /\ IF canCommit
     THEN /\ recordingReferences' = recordingReferences \cup newRefs
          /\ swapchainCommandLease' = (swapchainCommandLease \/ usesSwapchain)
          /\ rendering' = TRUE
          /\ timestampScope' = hasTimestampWrites
          /\ activeEndWrite' = IF hasTimestampWrites THEN endWrite ELSE NoQuery
          /\ pendingWrites' =
               IF hasTimestampWrites THEN pendingWrites \cup writes ELSE pendingWrites
          /\ UNCHANGED <<
               logicalAlive, physicalAlive, inFlightReferences,
               swapchainAcquired, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "begin"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

AttemptEnd ==
  /\ IF EndValid
     THEN /\ rendering' = FALSE
          /\ timestampScope' = FALSE
          /\ activeEndWrite' = NoQuery
          /\ UNCHANGED <<
               logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
               swapchainAcquired, swapchainCommandLease, pendingWrites, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "end"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

AttemptResolve(firstQuery, queryCount, destinationOffset, copyDstUsage) ==
  LET querySet == QueryRange(firstQuery, queryCount) IN
  /\ IF ResolveValid(firstQuery, queryCount, destinationOffset, copyDstUsage)
     THEN /\ recordingReferences' = recordingReferences \cup {Pool, Destination}
          /\ pendingWrites' = pendingWrites \ querySet
          /\ UNCHANGED <<
               logicalAlive, physicalAlive, inFlightReferences,
               swapchainAcquired, swapchainCommandLease, rendering,
               timestampScope, activeEndWrite, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "resolve"
  /\ lastResolveRange' = querySet
  /\ lastResolveOffset' = destinationOffset

DestroyResource(resource) ==
  /\ logicalAlive[resource]
  /\ logicalAlive' = [logicalAlive EXCEPT ![resource] = FALSE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![resource] = Referenced(resource)]
  /\ UNCHANGED <<
       recordingReferences, inFlightReferences,
       swapchainAcquired, swapchainCommandLease, rendering,
       timestampScope, activeEndWrite, pendingWrites, inFlight
     >>
  /\ lastOp' = "destroy"
  /\ lastOutcome' = "accepted"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

Submit ==
  /\ ~inFlight
  /\ ~rendering
  /\ ~timestampScope
  /\ recordingReferences # {}
  /\ recordingReferences' = {}
  /\ inFlightReferences' = recordingReferences
  /\ swapchainCommandLease' = FALSE
  /\ pendingWrites' = {}
  /\ inFlight' = TRUE
  /\ UNCHANGED <<
       logicalAlive, physicalAlive, swapchainAcquired,
       rendering, timestampScope, activeEndWrite
     >>
  /\ lastOp' = "submit"
  /\ lastOutcome' = "accepted"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

Complete ==
  /\ inFlight
  /\ physicalAlive' = [r \in Resources |-> logicalAlive[r]]
  /\ inFlightReferences' = {}
  /\ swapchainAcquired' = FALSE
  /\ inFlight' = FALSE
  /\ UNCHANGED <<
       logicalAlive, recordingReferences, swapchainCommandLease,
       rendering, timestampScope, activeEndWrite, pendingWrites
     >>
  /\ lastOp' = "complete"
  /\ lastOutcome' = "accepted"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

Abandon ==
  /\ ~inFlight
  /\ physicalAlive' = [r \in Resources |-> logicalAlive[r]]
  /\ recordingReferences' = {}
  /\ swapchainCommandLease' = FALSE
  /\ rendering' = FALSE
  /\ timestampScope' = FALSE
  /\ activeEndWrite' = NoQuery
  /\ pendingWrites' = {}
  /\ UNCHANGED <<logicalAlive, inFlightReferences, swapchainAcquired, inFlight>>
  /\ lastOp' = "abandon"
  /\ lastOutcome' = "accepted"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

Next ==
  \/ AcquireSwapchain
  \/ \E av, sc \in BOOLEAN, fault \in PublishFaults :
       AttemptBegin(av, sc, FALSE, NoQuery, NoQuery, fault)
  \/ \E sc \in BOOLEAN, bw, ew \in BoundaryIndices, fault \in PublishFaults :
       AttemptBegin(TRUE, sc, TRUE, bw, ew, fault)
  \/ AttemptEnd
  \/ \E fq \in {0, 1, 2}, qc \in QueryCounts,
        off \in DestinationOffsets, usage \in BOOLEAN :
       AttemptResolve(fq, qc, off, usage)
  \/ \E r \in Resources : DestroyResource(r)
  \/ Submit
  \/ Complete
  \/ Abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ logicalAlive \in [Resources -> BOOLEAN]
  /\ physicalAlive \in [Resources -> BOOLEAN]
  /\ recordingReferences \in SUBSET Resources
  /\ inFlightReferences \in SUBSET Resources
  /\ swapchainAcquired \in BOOLEAN
  /\ swapchainCommandLease \in BOOLEAN
  /\ rendering \in BOOLEAN
  /\ timestampScope \in BOOLEAN
  /\ activeEndWrite \in (Queries \cup {NoQuery})
  /\ pendingWrites \in SUBSET Queries
  /\ inFlight \in BOOLEAN
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes
  /\ lastResolveRange \in SUBSET {0, 1, 2, 3}
  /\ lastResolveOffset \in DestinationOffsets

PhysicalLifetimeExact ==
  \A r \in Resources :
    physicalAlive[r] = (logicalAlive[r] \/ Referenced(r))

TimestampScopeIsRenderBound ==
  timestampScope =>
    /\ rendering
    /\ Pool \in recordingReferences
    /\ Attachment \in recordingReferences
    /\ physicalAlive[Pool]
    /\ physicalAlive[Attachment]

RenderingRetainsAttachment ==
  rendering =>
    /\ Attachment \in recordingReferences
    /\ physicalAlive[Attachment]

ActiveEndSlotIsReserved ==
  activeEndWrite \in Queries =>
    /\ timestampScope
    /\ activeEndWrite \in pendingWrites

SwapchainCommandLeaseRequiresAcquisition ==
  swapchainCommandLease =>
    /\ swapchainAcquired
    /\ Attachment \in recordingReferences

AcceptedResolveIsAlignedAndInBounds ==
  lastOp = "resolve" /\ lastOutcome = "accepted" =>
    /\ lastResolveRange # {}
    /\ lastResolveRange \subseteq Queries
    /\ lastResolveOffset % ResultSize = 0
    /\ lastResolveOffset + Cardinality(lastResolveRange) * ResultSize <= DestinationSize
    /\ {Pool, Destination} \subseteq recordingReferences

Safety ==
  /\ PhysicalLifetimeExact
  /\ TimestampScopeIsRenderBound
  /\ RenderingRetainsAttachment
  /\ ActiveEndSlotIsReserved
  /\ SwapchainCommandLeaseRequiresAcquisition
  /\ AcceptedResolveIsAlignedAndInBounds

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (lastOp' = "resolve" /\ lastOutcome' = "accepted" =>
          /\ ~rendering
          /\ ~timestampScope
          /\ lastResolveRange' \subseteq pendingWrites
          /\ pendingWrites' = pendingWrites \ lastResolveRange')
    /\ (lastOp' = "begin" /\ lastOutcome' = "accepted" =>
          /\ ~rendering
          /\ rendering'
          /\ Attachment \in recordingReferences')
  ]_vars

=============================================================================
