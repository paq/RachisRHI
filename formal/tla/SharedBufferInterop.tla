------------------------ MODULE SharedBufferInterop -------------------------
(***************************************************************************)
(* Formal safety model for cross-device shared committed buffers and       *)
(* shared timeline-fence handoffs:                                         *)
(*   - include/Rachis/Interop.h, src/Interop.cpp (SharedHandle ownership)  *)
(*   - src/D3D12/D3D12Resource.cpp (createSharedBuffer,                    *)
(*     exportSharedBuffer, importSharedBuffer, validateSharedBufferDesc)   *)
(*   - src/D3D12/D3D12Queue.cpp (exportTimelinePoint,                      *)
(*     enqueueSharedTimelineWait)                                          *)
(*                                                                         *)
(* Models shareability gating (committed device-local automatic-residency  *)
(* vs non-shareable placed/host-visible/manual-residency allocations),     *)
(* transport handle close before and after consumer import/wait, concrete  *)
(* rejected invalid interop requests, and bidirectional GPU ordering.      *)
(*                                                                         *)
(* Trusted boundary: OS kernel handle duplication and native D3D12 shared  *)
(* heap/fence signaling once cross-device waits are enqueued.              *)
(***************************************************************************)
EXTENDS Naturals

Parties == {"producer", "consumer"}
AllocationKinds == {"none", "sharedCommitted", "nonShareable"}
BufferCandidates == {"sharedCommitted", "nonShareable"}
Contents == {"empty", "producer", "consumer"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  logicalAlive,          \* Per-party public buffer handle liveness.
  physicalAlive,         \* Whether underlying shared/local buffer memory exists.
  allocationKind,        \* Buffer storage class (none, sharedCommitted, nonShareable).
  transportOpen,         \* Whether exported shared buffer transport handle is open.
  submitted,             \* Per-party submitted timeline value (0..2).
  completed,             \* Per-party completed timeline value (0..2).
  exported,              \* Per-party exported shared timeline value.
  pending,               \* Per-party whether a submission is currently in flight.
  producerFenceOpen,     \* Whether producer's exported fence transport is open.
  producerFenceRetained, \* Whether consumer queue wait retained producer's fence.
  consumerWaitValue,     \* Producer timeline value awaited by consumer queue.
  consumerFenceOpen,     \* Whether consumer's exported fence transport is open.
  consumerFenceRetained, \* Whether producer queue wait retained consumer's fence.
  producerWaitValue,     \* Consumer timeline value awaited by producer queue.
  content,               \* Current committed GPU payload in buffer storage.
  observed,              \* Payload observed by each party's completed read.
  lastOutcome            \* Outcome of the latest fallible interop request.

coreVars == <<
  logicalAlive,
  physicalAlive,
  allocationKind,
  transportOpen,
  submitted,
  completed,
  exported,
  pending,
  producerFenceOpen,
  producerFenceRetained,
  consumerWaitValue,
  consumerFenceOpen,
  consumerFenceRetained,
  producerWaitValue,
  content,
  observed
>>

vars == <<
  logicalAlive,
  physicalAlive,
  allocationKind,
  transportOpen,
  submitted,
  completed,
  exported,
  pending,
  producerFenceOpen,
  producerFenceRetained,
  consumerWaitValue,
  consumerFenceOpen,
  consumerFenceRetained,
  producerWaitValue,
  content,
  observed,
  lastOutcome
>>

Init ==
  /\ logicalAlive = [party \in Parties |-> FALSE]
  /\ physicalAlive = FALSE
  /\ allocationKind = "none"
  /\ transportOpen = FALSE
  /\ submitted = [party \in Parties |-> 0]
  /\ completed = [party \in Parties |-> 0]
  /\ exported = [party \in Parties |-> 0]
  /\ pending = [party \in Parties |-> FALSE]
  /\ producerFenceOpen = FALSE
  /\ producerFenceRetained = FALSE
  /\ consumerWaitValue = 0
  /\ consumerFenceOpen = FALSE
  /\ consumerFenceRetained = FALSE
  /\ producerWaitValue = 0
  /\ content = "empty"
  /\ observed = [party \in Parties |-> "empty"]
  /\ lastOutcome = "none"

createBuffer(kind) ==
  /\ ~physicalAlive
  /\ kind \in BufferCandidates
  /\ logicalAlive' = [logicalAlive EXCEPT !["producer"] = TRUE]
  /\ physicalAlive' = TRUE
  /\ allocationKind' = kind
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       transportOpen,
       submitted,
       completed,
       exported,
       pending,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

\* Non-shareable buffers (host-visible, manual residency, placed, RTAS) reject export.
attemptExportBuffer ==
  /\ logicalAlive["producer"]
  /\ ~transportOpen
  /\ IF physicalAlive /\ allocationKind = "sharedCommitted"
     THEN /\ transportOpen' = TRUE
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               logicalAlive,
               physicalAlive,
               allocationKind,
               submitted,
               completed,
               exported,
               pending,
               producerFenceOpen,
               producerFenceRetained,
               consumerWaitValue,
               consumerFenceOpen,
               consumerFenceRetained,
               producerWaitValue,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

\* Closed transports or mismatched import descriptors reject atomically.
attemptImportBuffer(descriptorMatches) ==
  /\ ~logicalAlive["consumer"]
  /\ IF transportOpen /\ physicalAlive /\ allocationKind = "sharedCommitted" /\ descriptorMatches
     THEN /\ logicalAlive' = [logicalAlive EXCEPT !["consumer"] = TRUE]
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               physicalAlive,
               allocationKind,
               transportOpen,
               submitted,
               completed,
               exported,
               pending,
               producerFenceOpen,
               producerFenceRetained,
               consumerWaitValue,
               consumerFenceOpen,
               consumerFenceRetained,
               producerWaitValue,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

closeTransport ==
  /\ transportOpen
  /\ transportOpen' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       submitted,
       completed,
       exported,
       pending,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

submitProducerWrite ==
  /\ logicalAlive["producer"]
  /\ physicalAlive
  /\ submitted["producer"] = 0
  /\ ~pending["producer"]
  /\ submitted' = [submitted EXCEPT !["producer"] = 1]
  /\ pending' = [pending EXCEPT !["producer"] = TRUE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       completed,
       exported,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

exportProducerPoint ==
  /\ submitted["producer"] = 1
  /\ exported["producer"] = 0
  /\ ~producerFenceOpen
  /\ consumerWaitValue = 0
  /\ exported' = [exported EXCEPT !["producer"] = 1]
  /\ producerFenceOpen' = TRUE
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       completed,
       pending,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

\* Closing producerFenceOpen before enqueueConsumerWait is permitted and causes
\* any subsequent wait attempt on that closed transport to reject atomically.
closeProducerFence ==
  /\ producerFenceOpen
  /\ producerFenceOpen' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       completed,
       exported,
       pending,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

attemptEnqueueConsumerWait(nonzeroValue) ==
  /\ logicalAlive["consumer"]
  /\ consumerWaitValue = 0
  /\ IF producerFenceOpen /\ exported["producer"] = 1 /\ nonzeroValue
     THEN /\ producerFenceRetained' = TRUE
          /\ consumerWaitValue' = 1
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               logicalAlive,
               physicalAlive,
               allocationKind,
               transportOpen,
               submitted,
               completed,
               exported,
               pending,
               producerFenceOpen,
               consumerFenceOpen,
               consumerFenceRetained,
               producerWaitValue,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

submitConsumerRead ==
  /\ logicalAlive["consumer"]
  /\ physicalAlive
  /\ consumerWaitValue = 1
  /\ submitted["consumer"] = 0
  /\ ~pending["consumer"]
  /\ submitted' = [submitted EXCEPT !["consumer"] = 1]
  /\ pending' = [pending EXCEPT !["consumer"] = TRUE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       completed,
       exported,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

completeProducerWrite ==
  /\ submitted["producer"] = 1
  /\ pending["producer"]
  /\ completed' = [completed EXCEPT !["producer"] = 1]
  /\ pending' = [pending EXCEPT !["producer"] = FALSE]
  /\ content' = "producer"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       exported,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       observed
     >>

completeConsumerRead ==
  /\ submitted["consumer"] = 1
  /\ pending["consumer"]
  /\ completed["producer"] >= consumerWaitValue
  /\ completed' = [completed EXCEPT !["consumer"] = 1]
  /\ pending' = [pending EXCEPT !["consumer"] = FALSE]
  /\ observed' = [observed EXCEPT !["consumer"] = content]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       exported,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content
     >>

submitConsumerWrite ==
  /\ logicalAlive["consumer"]
  /\ physicalAlive
  /\ submitted["consumer"] = 1
  /\ completed["consumer"] = 1
  /\ ~pending["consumer"]
  /\ submitted' = [submitted EXCEPT !["consumer"] = 2]
  /\ pending' = [pending EXCEPT !["consumer"] = TRUE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       completed,
       exported,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

completeConsumerWrite ==
  /\ submitted["consumer"] = 2
  /\ pending["consumer"]
  /\ completed' = [completed EXCEPT !["consumer"] = 2]
  /\ pending' = [pending EXCEPT !["consumer"] = FALSE]
  /\ content' = "consumer"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       exported,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       observed
     >>

exportConsumerPoint ==
  /\ submitted["consumer"] = 2
  /\ exported["consumer"] = 0
  /\ ~consumerFenceOpen
  /\ producerWaitValue = 0
  /\ exported' = [exported EXCEPT !["consumer"] = 2]
  /\ consumerFenceOpen' = TRUE
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       completed,
       pending,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

closeConsumerFence ==
  /\ consumerFenceOpen
  /\ consumerFenceOpen' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       completed,
       exported,
       pending,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

attemptEnqueueProducerWait(nonzeroValue) ==
  /\ logicalAlive["producer"]
  /\ producerWaitValue = 0
  /\ IF consumerFenceOpen /\ exported["consumer"] = 2 /\ nonzeroValue
     THEN /\ consumerFenceRetained' = TRUE
          /\ producerWaitValue' = 2
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               logicalAlive,
               physicalAlive,
               allocationKind,
               transportOpen,
               submitted,
               completed,
               exported,
               pending,
               producerFenceOpen,
               producerFenceRetained,
               consumerWaitValue,
               consumerFenceOpen,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

submitProducerRead ==
  /\ logicalAlive["producer"]
  /\ physicalAlive
  /\ producerWaitValue = 2
  /\ submitted["producer"] = 1
  /\ completed["producer"] = 1
  /\ ~pending["producer"]
  /\ submitted' = [submitted EXCEPT !["producer"] = 2]
  /\ pending' = [pending EXCEPT !["producer"] = TRUE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       completed,
       exported,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

completeProducerRead ==
  /\ submitted["producer"] = 2
  /\ pending["producer"]
  /\ completed["consumer"] >= producerWaitValue
  /\ completed' = [completed EXCEPT !["producer"] = 2]
  /\ pending' = [pending EXCEPT !["producer"] = FALSE]
  /\ observed' = [observed EXCEPT !["producer"] = content]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       exported,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content
     >>

destroyLogical(party) ==
  /\ logicalAlive[party]
  /\ logicalAlive' = [logicalAlive EXCEPT ![party] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       completed,
       exported,
       pending,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

collectPhysical ==
  /\ physicalAlive
  /\ ~transportOpen
  /\ \A party \in Parties : ~logicalAlive[party]
  /\ \A party \in Parties : ~pending[party]
  /\ \A party \in Parties : completed[party] = submitted[party]
  /\ physicalAlive' = FALSE
  /\ allocationKind' = "none"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       transportOpen,
       submitted,
       completed,
       exported,
       pending,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

collectProducerFence ==
  /\ producerFenceRetained
  /\ completed["consumer"] >= consumerWaitValue
  /\ ~pending["consumer"]
  /\ producerFenceOpen' = FALSE
  /\ producerFenceRetained' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       completed,
       exported,
       pending,
       consumerWaitValue,
       consumerFenceOpen,
       consumerFenceRetained,
       producerWaitValue,
       content,
       observed
     >>

collectConsumerFence ==
  /\ consumerFenceRetained
  /\ completed["producer"] >= producerWaitValue
  /\ ~pending["producer"]
  /\ consumerFenceOpen' = FALSE
  /\ consumerFenceRetained' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       transportOpen,
       submitted,
       completed,
       exported,
       pending,
       producerFenceOpen,
       producerFenceRetained,
       consumerWaitValue,
       producerWaitValue,
       content,
       observed
     >>

Next ==
  \/ \E kind \in BufferCandidates : createBuffer(kind)
  \/ attemptExportBuffer
  \/ \E ok \in BOOLEAN : attemptImportBuffer(ok)
  \/ closeTransport
  \/ submitProducerWrite
  \/ exportProducerPoint
  \/ closeProducerFence
  \/ \E ok \in BOOLEAN : attemptEnqueueConsumerWait(ok)
  \/ submitConsumerRead
  \/ completeProducerWrite
  \/ completeConsumerRead
  \/ submitConsumerWrite
  \/ completeConsumerWrite
  \/ exportConsumerPoint
  \/ closeConsumerFence
  \/ \E ok \in BOOLEAN : attemptEnqueueProducerWait(ok)
  \/ submitProducerRead
  \/ completeProducerRead
  \/ \E party \in Parties : destroyLogical(party)
  \/ collectPhysical
  \/ collectProducerFence
  \/ collectConsumerFence

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ logicalAlive \in [Parties -> BOOLEAN]
  /\ physicalAlive \in BOOLEAN
  /\ allocationKind \in AllocationKinds
  /\ transportOpen \in BOOLEAN
  /\ submitted \in [Parties -> 0..2]
  /\ completed \in [Parties -> 0..2]
  /\ exported \in [Parties -> 0..2]
  /\ pending \in [Parties -> BOOLEAN]
  /\ producerFenceOpen \in BOOLEAN
  /\ producerFenceRetained \in BOOLEAN
  /\ consumerWaitValue \in {0, 1}
  /\ consumerFenceOpen \in BOOLEAN
  /\ consumerFenceRetained \in BOOLEAN
  /\ producerWaitValue \in {0, 2}
  /\ content \in Contents
  /\ observed \in [Parties -> Contents]
  /\ lastOutcome \in Outcomes
  /\ \A party \in Parties :
       /\ completed[party] <= submitted[party]
       /\ exported[party] <= submitted[party]

OnlySharedCommittedIsShareable ==
  /\ (physicalAlive <=> allocationKind # "none")
  /\ ((transportOpen \/ logicalAlive["consumer"])
      => allocationKind = "sharedCommitted")

EveryBufferReferenceRetainsStorage ==
  /\ \A party \in Parties : (logicalAlive[party] \/ pending[party]) => physicalAlive
  /\ (transportOpen => physicalAlive)

ExportedValuesAreExact ==
  /\ exported["producer"] \in {0, 1}
  /\ exported["consumer"] \in {0, 2}
  /\ (consumerWaitValue > 0 => exported["producer"] = consumerWaitValue)
  /\ (producerWaitValue > 0 => exported["consumer"] = producerWaitValue)

ClosedFenceHandlesDoNotRevokeWaits ==
  /\ (consumerWaitValue > 0 /\ ~producerFenceOpen /\ completed["consumer"] < 1
      => producerFenceRetained)
  /\ (producerWaitValue > 0 /\ ~consumerFenceOpen /\ completed["producer"] < 2
      => consumerFenceRetained)

ExternalGpuAccessIsOrdered ==
  /\ (completed["consumer"] > 0 =>
        consumerWaitValue = 1 /\ completed["producer"] >= consumerWaitValue)
  /\ (completed["producer"] = 2 =>
        producerWaitValue = 2 /\ completed["consumer"] >= producerWaitValue)

WritesNeverExecuteConcurrently ==
  ~(pending["producer"] /\ submitted["producer"] = 1
    /\ pending["consumer"] /\ submitted["consumer"] = 2)

BidirectionalContentsAreObserved ==
  /\ (observed["consumer"] # "empty" => observed["consumer"] = "producer")
  /\ (observed["producer"] # "empty" => observed["producer"] = "consumer")

Safety ==
  /\ OnlySharedCommittedIsShareable
  /\ EveryBufferReferenceRetainsStorage
  /\ ExportedValuesAreExact
  /\ ClosedFenceHandlesDoNotRevokeWaits
  /\ ExternalGpuAccessIsOrdered
  /\ WritesNeverExecuteConcurrently
  /\ BidirectionalContentsAreObserved

StepSafety ==
  [][lastOutcome' = "rejected" => coreVars' = coreVars]_vars

=============================================================================
