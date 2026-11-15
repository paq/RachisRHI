--------------------------- MODULE TimestampQuery ---------------------------
(***************************************************************************)
(* Formal safety model for queue-scoped timestamp query writes, render-    *)
(* pass exclusion on resolves, aligned destination buffer bounds,          *)
(* transactional dependency/residency retention rollback, and resource     *)
(* lifetime retention:                                                     *)
(*   - include/Rachis/Device.h (writeTimestamp, resolveTimestamps) and     *)
(*     include/Rachis/Query.h (TimestampQueryResolveDesc,                  *)
(*     timestampQueryResultSize, timestampQueryResultAlignment)            *)
(*   - src/D3D12/D3D12Query.cpp (QueryManager::writeTimestamp and          *)
(*     QueryManager::resolveTimestamps; retainQueryCommandDependencies)    *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::retainResource and          *)
(*     retainAccelerationStructureDependencies retain owners and residency *)
(*     leases as one unit or roll back)                                    *)
(*   - src/D3D12/D3D12Resource.cpp (ResourceManager::bufferBindingSource   *)
(*     resolves the live destination and its manual-residency record)      *)
(*                                                                         *)
(* src/Vulkan/VulkanQueries.cpp and src/Metal/MetalQueries.mm apply the    *)
(* same queue, range, alignment, and render-pass checks, but retain the    *)
(* pool and destination with sequential pushes and no manual-residency     *)
(* lease, so a late failure can leave an earlier reference retained; they  *)
(* are outside the rollback claim.                                         *)
(*                                                                         *)
(* Eliminates clone-variable tautologies and models multi-resource         *)
(* retention (Pool, Destination, and optional manual Residency) so partial *)
(* retention failure rolls back recordingReferences and pendingWrites      *)
(* atomically without mutating prior command state.                        *)
(*                                                                         *)
(* Trusted boundary: hardware timestamp counter latching and DMA copy      *)
(* execution once query commands are submitted.                            *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

Pool == 0
Destination == 1
Residency == 2
Resources == {Pool, Destination, Residency}

PoolQueue == 0
Queues == {0, 1}
Queries == {0, 1}
QueryIndices == {0, 1, 2}
FirstQueries == {0, 1, 2}
QueryCounts == {0, 1, 2}
DestinationOffsets == {0, 1, 8, 16}
ResultSize == 8
DestinationSize == 16
Phases == {"idle", "retainingWrite", "retainingResolve"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  renderingActive,       \* Whether a render pass is currently open.
  logicalAlive,          \* Per-resource public handle liveness.
  physicalAlive,         \* Per-resource native object/heap liveness.
  manualResidency,       \* Whether Destination uses a manual-residency heap.
  recordingReferences,   \* Resources retained on the open command buffer.
  inFlightReferences,    \* Resources retained on the in-flight GPU submission.
  pendingWrites,         \* Timestamp slots written in this command and not yet resolved.
  inFlight,              \* Whether the command buffer is currently in flight on GPU.
  txPhase,               \* Active write/resolve retention stage.
  stagedReferences,      \* Working copy of recordingReferences during retention.
  txQuerySet,            \* Query slot(s) targeted by active write/resolve transaction.
  txRequiredRefs,        \* Resources that must be retained before commit.
  txResolveOffset,       \* Destination byte offset of active resolve transaction.
  lastResolvedRange,     \* Query slot set resolved by latest accepted resolve.
  lastResolvedOffset,    \* Destination byte offset of latest accepted resolve.
  lastOutcome            \* Outcome of latest write/resolve transaction.

coreVars == <<
  renderingActive,
  logicalAlive,
  physicalAlive,
  manualResidency,
  recordingReferences,
  inFlightReferences,
  pendingWrites,
  inFlight
>>

vars == <<
  renderingActive,
  logicalAlive,
  physicalAlive,
  manualResidency,
  recordingReferences,
  inFlightReferences,
  pendingWrites,
  inFlight,
  txPhase,
  stagedReferences,
  txQuerySet,
  txRequiredRefs,
  txResolveOffset,
  lastResolvedRange,
  lastResolvedOffset,
  lastOutcome
>>

Referenced(resource) ==
  resource \in recordingReferences \/ resource \in inFlightReferences

QueryRange(firstQuery, queryCount) ==
  IF queryCount = 1 THEN {firstQuery}
  ELSE IF queryCount = 2 THEN {firstQuery, firstQuery + 1}
  ELSE {}

WriteValid(commandQueue, queryIndex) ==
  /\ ~inFlight
  /\ logicalAlive[Pool]
  /\ commandQueue = PoolQueue
  /\ queryIndex \in Queries
  /\ ~(queryIndex \in pendingWrites)

ResolveValid(
    commandQueue,
    firstQuery,
    queryCount,
    destinationOffset,
    copyDestinationUsage) ==
  LET querySet == QueryRange(firstQuery, queryCount) IN
  /\ ~inFlight
  /\ ~renderingActive
  /\ logicalAlive[Pool]
  /\ logicalAlive[Destination]
  /\ (~manualResidency \/ logicalAlive[Residency])
  /\ commandQueue = PoolQueue
  /\ copyDestinationUsage
  /\ queryCount > 0
  /\ firstQuery + queryCount <= Cardinality(Queries)
  /\ querySet \subseteq Queries
  /\ querySet \subseteq pendingWrites
  /\ destinationOffset % ResultSize = 0
  /\ destinationOffset + queryCount * ResultSize <= DestinationSize

Init ==
  \E usesManualResidency \in BOOLEAN :
    /\ renderingActive = FALSE
    /\ logicalAlive = [r \in Resources |-> TRUE]
    /\ physicalAlive = [r \in Resources |-> TRUE]
    /\ manualResidency = usesManualResidency
    /\ recordingReferences = {}
    /\ inFlightReferences = {}
    /\ pendingWrites = {}
    /\ inFlight = FALSE
    /\ txPhase = "idle"
    /\ stagedReferences = {}
    /\ txQuerySet = {}
    /\ txRequiredRefs = {}
    /\ txResolveOffset = 0
    /\ lastResolvedRange = {}
    /\ lastResolvedOffset = 0
    /\ lastOutcome = "none"

enterRendering ==
  /\ txPhase = "idle"
  /\ ~inFlight
  /\ ~renderingActive
  /\ renderingActive' = TRUE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       manualResidency,
       recordingReferences,
       inFlightReferences,
       pendingWrites,
       inFlight,
       txPhase,
       stagedReferences,
       txQuerySet,
       txRequiredRefs,
       txResolveOffset,
       lastResolvedRange,
       lastResolvedOffset
     >>

leaveRendering ==
  /\ txPhase = "idle"
  /\ renderingActive
  /\ renderingActive' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       manualResidency,
       recordingReferences,
       inFlightReferences,
       pendingWrites,
       inFlight,
       txPhase,
       stagedReferences,
       txQuerySet,
       txRequiredRefs,
       txResolveOffset,
       lastResolvedRange,
       lastResolvedOffset
     >>

beginWrite(commandQueue, queryIndex) ==
  /\ txPhase = "idle"
  /\ commandQueue \in Queues
  /\ queryIndex \in QueryIndices
  /\ IF WriteValid(commandQueue, queryIndex)
     THEN /\ txPhase' = "retainingWrite"
          /\ stagedReferences' = recordingReferences
          /\ txQuerySet' = {queryIndex}
          /\ txRequiredRefs' = {Pool}
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<coreVars, txResolveOffset, lastResolvedRange, lastResolvedOffset>>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED <<
               coreVars,
               txPhase,
               stagedReferences,
               txQuerySet,
               txRequiredRefs,
               txResolveOffset,
               lastResolvedRange,
               lastResolvedOffset
             >>

beginResolve(
    commandQueue,
    firstQuery,
    queryCount,
    destinationOffset,
    copyDestinationUsage) ==
  /\ txPhase = "idle"
  /\ commandQueue \in Queues
  /\ firstQuery \in FirstQueries
  /\ queryCount \in QueryCounts
  /\ destinationOffset \in DestinationOffsets
  /\ copyDestinationUsage \in BOOLEAN
  /\ IF ResolveValid(
          commandQueue,
          firstQuery,
          queryCount,
          destinationOffset,
          copyDestinationUsage)
     THEN /\ txPhase' = "retainingResolve"
          /\ stagedReferences' = recordingReferences
          /\ txQuerySet' = QueryRange(firstQuery, queryCount)
          /\ txRequiredRefs' =
               {Pool, Destination}
               \cup (IF manualResidency THEN {Residency} ELSE {})
          /\ txResolveOffset' = destinationOffset
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<coreVars, lastResolvedRange, lastResolvedOffset>>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED <<
               coreVars,
               txPhase,
               stagedReferences,
               txQuerySet,
               txRequiredRefs,
               txResolveOffset,
               lastResolvedRange,
               lastResolvedOffset
             >>

stageReference(resource) ==
  /\ txPhase \in {"retainingWrite", "retainingResolve"}
  /\ resource \in txRequiredRefs \ stagedReferences
  /\ stagedReferences' = stagedReferences \cup {resource}
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       coreVars,
       txPhase,
       txQuerySet,
       txRequiredRefs,
       txResolveOffset,
       lastResolvedRange,
       lastResolvedOffset
     >>

\* Any fallible reservation/retention failure rolls back stagedReferences
\* before pendingWrites or recordingReferences are modified.
failRetention ==
  /\ txPhase \in {"retainingWrite", "retainingResolve"}
  /\ txPhase' = "idle"
  /\ stagedReferences' = {}
  /\ txQuerySet' = {}
  /\ txRequiredRefs' = {}
  /\ txResolveOffset' = 0
  /\ lastOutcome' = "rejected"
  /\ UNCHANGED <<coreVars, lastResolvedRange, lastResolvedOffset>>

commitTransaction ==
  /\ txPhase \in {"retainingWrite", "retainingResolve"}
  /\ txRequiredRefs \subseteq stagedReferences
  /\ recordingReferences' = stagedReferences
  /\ IF txPhase = "retainingWrite"
     THEN /\ pendingWrites' = pendingWrites \cup txQuerySet
          /\ UNCHANGED <<lastResolvedRange, lastResolvedOffset>>
     ELSE /\ pendingWrites' = pendingWrites \ txQuerySet
          /\ lastResolvedRange' = txQuerySet
          /\ lastResolvedOffset' = txResolveOffset
  /\ txPhase' = "idle"
  /\ stagedReferences' = {}
  /\ txQuerySet' = {}
  /\ txRequiredRefs' = {}
  /\ txResolveOffset' = 0
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       renderingActive,
       logicalAlive,
       physicalAlive,
       manualResidency,
       inFlightReferences,
       inFlight
     >>

destroyResource(resource) ==
  /\ txPhase = "idle"
  /\ logicalAlive[resource]
  /\ logicalAlive' = [logicalAlive EXCEPT ![resource] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       physicalAlive,
       manualResidency,
       recordingReferences,
       inFlightReferences,
       pendingWrites,
       inFlight,
       txPhase,
       stagedReferences,
       txQuerySet,
       txRequiredRefs,
       txResolveOffset,
       lastResolvedRange,
       lastResolvedOffset
     >>

collectResource(resource) ==
  /\ txPhase = "idle"
  /\ physicalAlive[resource]
  /\ ~logicalAlive[resource]
  /\ ~Referenced(resource)
  /\ physicalAlive' = [physicalAlive EXCEPT ![resource] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       logicalAlive,
       manualResidency,
       recordingReferences,
       inFlightReferences,
       pendingWrites,
       inFlight,
       txPhase,
       stagedReferences,
       txQuerySet,
       txRequiredRefs,
       txResolveOffset,
       lastResolvedRange,
       lastResolvedOffset
     >>

submit ==
  /\ txPhase = "idle"
  /\ ~inFlight
  /\ ~renderingActive
  /\ recordingReferences # {}
  /\ recordingReferences' = {}
  /\ inFlightReferences' = recordingReferences
  /\ pendingWrites' = {}
  /\ inFlight' = TRUE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       logicalAlive,
       physicalAlive,
       manualResidency,
       txPhase,
       stagedReferences,
       txQuerySet,
       txRequiredRefs,
       txResolveOffset,
       lastResolvedRange,
       lastResolvedOffset
     >>

complete ==
  /\ txPhase = "idle"
  /\ inFlight
  /\ inFlightReferences' = {}
  /\ inFlight' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       logicalAlive,
       physicalAlive,
       manualResidency,
       recordingReferences,
       pendingWrites,
       txPhase,
       stagedReferences,
       txQuerySet,
       txRequiredRefs,
       txResolveOffset,
       lastResolvedRange,
       lastResolvedOffset
     >>

abandon ==
  /\ txPhase = "idle"
  /\ ~inFlight
  /\ (renderingActive \/ recordingReferences # {})
  /\ renderingActive' = FALSE
  /\ recordingReferences' = {}
  /\ pendingWrites' = {}
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       manualResidency,
       inFlightReferences,
       inFlight,
       txPhase,
       stagedReferences,
       txQuerySet,
       txRequiredRefs,
       txResolveOffset,
       lastResolvedRange,
       lastResolvedOffset
     >>

Next ==
  \/ enterRendering
  \/ leaveRendering
  \/ \E q \in Queues, idx \in QueryIndices : beginWrite(q, idx)
  \/ \E q \in Queues,
        fq \in FirstQueries,
        cnt \in QueryCounts,
        off \in DestinationOffsets,
        usage \in BOOLEAN :
       beginResolve(q, fq, cnt, off, usage)
  \/ \E r \in Resources : stageReference(r)
  \/ failRetention
  \/ commitTransaction
  \/ \E r \in Resources : destroyResource(r)
  \/ \E r \in Resources : collectResource(r)
  \/ submit
  \/ complete
  \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ renderingActive \in BOOLEAN
  /\ logicalAlive \in [Resources -> BOOLEAN]
  /\ physicalAlive \in [Resources -> BOOLEAN]
  /\ manualResidency \in BOOLEAN
  /\ recordingReferences \subseteq Resources
  /\ inFlightReferences \subseteq Resources
  /\ pendingWrites \subseteq Queries
  /\ inFlight \in BOOLEAN
  /\ txPhase \in Phases
  /\ stagedReferences \subseteq Resources
  /\ txQuerySet \subseteq Queries
  /\ txRequiredRefs \subseteq Resources
  /\ txResolveOffset \in DestinationOffsets
  /\ lastResolvedRange \subseteq Queries
  /\ lastResolvedOffset \in DestinationOffsets
  /\ lastOutcome \in Outcomes

PendingWritesRetainPool ==
  pendingWrites # {} => Pool \in recordingReferences /\ ~inFlight

ResolvedRangeIsAlignedAndInBounds ==
  lastResolvedRange # {} =>
    /\ lastResolvedOffset % ResultSize = 0
    /\ lastResolvedOffset + Cardinality(lastResolvedRange) * ResultSize <= DestinationSize

ResolveTransactionsExcludeRenderPass ==
  txPhase = "retainingResolve" =>
    /\ ~renderingActive
    /\ txQuerySet # {}
    /\ txQuerySet \subseteq pendingWrites

RetainedAndLogicalResourcesRemainPhysical ==
  \A r \in Resources :
    (logicalAlive[r] \/ Referenced(r)) => physicalAlive[r]

RecordingAndSubmissionAreSeparate ==
  /\ ~(recordingReferences # {} /\ inFlight)
  /\ recordingReferences \cap inFlightReferences = {}

Safety ==
  /\ PendingWritesRetainPool
  /\ ResolvedRangeIsAlignedAndInBounds
  /\ ResolveTransactionsExcludeRenderPass
  /\ RetainedAndLogicalResourcesRemainPhysical
  /\ RecordingAndSubmissionAreSeparate

StepSafety ==
  [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
     /\ (txPhase' # "idle" => coreVars' = coreVars)]_vars

=============================================================================
