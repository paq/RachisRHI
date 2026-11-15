------------------------ MODULE SparseBufferMapping -------------------------
(***************************************************************************)
(* Formal safety model for capability-gated sparse buffers, serialized     *)
(* page-table updates, caller-selected publication timelines, cross-queue  *)
(* fence wait ordering, and physical heap retention:                       *)
(*   - src/D3D12/D3D12Device.cpp (sparseBuffers capability discovery)      *)
(*   - src/D3D12/D3D12SparseAndFeedback.cpp (createSparseBuffer,           *)
(*     enqueueSparseBufferMappings and its in-use remap rejection,         *)
(*     captureSparseBufferBinding)                                         *)
(*   - src/D3D12/D3D12Resource.cpp (sparse-backing heaps, copy/barrier     *)
(*     mapping capture), src/D3D12/D3D12ResourceCopies.cpp (mapped-tile    *)
(*     copy checks)                                                        *)
(*   - src/D3D12/D3D12Binding.cpp, src/D3D12/D3D12CommandRecording.cpp     *)
(*     (command-time binding capture and retention)                        *)
(*   - src/D3D12/D3D12Queue.cpp (enqueueSparseMappings serialization,      *)
(*     validateSparseMappingDependencies)                                  *)
(*                                                                         *)
(* Models dedicated sparse heaps with independent physical reclamation     *)
(* (heapPhysical), cross-queue GPU fence wait enforcement on both          *)
(* sequential mapping updates and sparse-consuming commands, Tier 2 null   *)
(* shader mappings vs mapped copy requirements, and atomic batch rollback. *)
(*                                                                         *)
(* Trusted boundary: D3D12 UpdateTileMappings hardware MMU programming     *)
(* once serialized queue waits complete.                                   *)
(***************************************************************************)
EXTENDS Naturals

Queues == {0, 1}
Targets == {0, 1, 2}
Unmapped == 2
MaxTimeline == 3
MaxVersion == 2
Kinds == {"none", "ordinary", "sparse"}
AllocatableKinds == {"ordinary", "sparse"}
CommandPhases == {"idle", "recorded", "inFlight"}
CommandKinds == {"none", "copy", "shader"}
RecordableCommandKinds == {"copy", "shader"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  deviceExists,            \* Whether the RHI device has been initialized.
  capable,                 \* Whether Tier 2 sparse buffers are supported.
  heapKind,                \* Allocated heap kind (none, ordinary, sparse).
  heapLogical,             \* Whether the public heap handle is live.
  heapPhysical,            \* Whether native heap memory is physically alive.
  bufferKind,              \* Allocated buffer kind (none, ordinary, sparse).
  mapping,                 \* Current physical tile target (0, 1, or Unmapped=2).
  version,                 \* Current immutable mapping generation (0..MaxVersion).
  publicationQueue,        \* Queue carrying the latest mapping publication signal.
  publicationValue,        \* Fence value signaled for the latest mapping version.
  mappingWaitQueue,        \* Prior publication queue awaited by version > 1.
  mappingWaitValue,        \* Prior publication fence value awaited by version > 1.
  oldRetained,             \* Whether the replaced generation is still retained.
  oldMapping,              \* Physical target owned by the replaced generation.
  oldRetireQueue,          \* Queue whose completion retires oldRetained.
  oldRetireValue,          \* Fence value whose completion retires oldRetained.
  commandPhase,            \* Sparse-using command lifecycle phase.
  commandKind,             \* Kind of sparse operation recorded ("copy" or "shader").
  commandVersion,          \* Mapping generation captured when recording the command.
  commandMapping,          \* Physical tile target captured by the command.
  commandQueue,            \* Queue on which the command is recorded/submitted.
  commandPublicationQueue, \* Mapping publication queue captured by the command.
  commandPublicationValue, \* Mapping publication fence value captured by command.
  commandWaitQueue,        \* Explicit wait queue supplied at submitCommand.
  commandWaitValue,        \* Explicit wait value supplied at submitCommand.
  commandRetireValue,      \* Submission fence value retiring the command.
  submitted,               \* Per-queue submitted timeline frontier.
  completed,               \* Per-queue GPU-completed timeline frontier.
  lastOutcome              \* Outcome of latest fallible mapping/command attempt.

coreVars == <<
  deviceExists,
  capable,
  heapKind,
  heapLogical,
  heapPhysical,
  bufferKind,
  mapping,
  version,
  publicationQueue,
  publicationValue,
  mappingWaitQueue,
  mappingWaitValue,
  oldRetained,
  oldMapping,
  oldRetireQueue,
  oldRetireValue,
  commandPhase,
  commandKind,
  commandVersion,
  commandMapping,
  commandQueue,
  commandPublicationQueue,
  commandPublicationValue,
  commandWaitQueue,
  commandWaitValue,
  commandRetireValue,
  submitted,
  completed
>>

vars == <<
  deviceExists,
  capable,
  heapKind,
  heapLogical,
  heapPhysical,
  bufferKind,
  mapping,
  version,
  publicationQueue,
  publicationValue,
  mappingWaitQueue,
  mappingWaitValue,
  oldRetained,
  oldMapping,
  oldRetireQueue,
  oldRetireValue,
  commandPhase,
  commandKind,
  commandVersion,
  commandMapping,
  commandQueue,
  commandPublicationQueue,
  commandPublicationValue,
  commandWaitQueue,
  commandWaitValue,
  commandRetireValue,
  submitted,
  completed,
  lastOutcome
>>

PublicationComplete(queue, value) ==
  value = 0 \/ completed[queue] >= value

CommandOrdered(waitQueue, waitValue) ==
  \/ commandQueue = commandPublicationQueue
  \/ PublicationComplete(commandPublicationQueue, commandPublicationValue)
  \/ (waitQueue = commandPublicationQueue /\ waitValue >= commandPublicationValue)

HeapHasRetainer ==
  \/ heapLogical
  \/ mapping # Unmapped
  \/ (oldRetained /\ oldMapping # Unmapped)
  \/ (commandPhase # "idle" /\ commandMapping # Unmapped)

Init ==
  /\ deviceExists = FALSE
  /\ capable = FALSE
  /\ heapKind = "none"
  /\ heapLogical = FALSE
  /\ heapPhysical = FALSE
  /\ bufferKind = "none"
  /\ mapping = Unmapped
  /\ version = 0
  /\ publicationQueue = 0
  /\ publicationValue = 0
  /\ mappingWaitQueue = 0
  /\ mappingWaitValue = 0
  /\ oldRetained = FALSE
  /\ oldMapping = Unmapped
  /\ oldRetireQueue = 0
  /\ oldRetireValue = 0
  /\ commandPhase = "idle"
  /\ commandKind = "none"
  /\ commandVersion = 0
  /\ commandMapping = Unmapped
  /\ commandQueue = 0
  /\ commandPublicationQueue = 0
  /\ commandPublicationValue = 0
  /\ commandWaitQueue = 0
  /\ commandWaitValue = 0
  /\ commandRetireValue = 0
  /\ submitted = [q \in Queues |-> 0]
  /\ completed = [q \in Queues |-> 0]
  /\ lastOutcome = "none"

createDevice(supportsSparse) ==
  /\ ~deviceExists
  /\ deviceExists' = TRUE
  /\ capable' = supportsSparse
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       heapKind,
       heapLogical,
       heapPhysical,
       bufferKind,
       mapping,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       oldRetained,
       oldMapping,
       oldRetireQueue,
       oldRetireValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapping,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       submitted,
       completed
     >>

attemptCreateHeap(kind) ==
  /\ deviceExists
  /\ heapKind = "none"
  /\ ~heapPhysical
  /\ IF kind = "ordinary" \/ capable
     THEN /\ heapKind' = kind
          /\ heapLogical' = TRUE
          /\ heapPhysical' = TRUE
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               deviceExists,
               capable,
               bufferKind,
               mapping,
               version,
               publicationQueue,
               publicationValue,
               mappingWaitQueue,
               mappingWaitValue,
               oldRetained,
               oldMapping,
               oldRetireQueue,
               oldRetireValue,
               commandPhase,
               commandKind,
               commandVersion,
               commandMapping,
               commandQueue,
               commandPublicationQueue,
               commandPublicationValue,
               commandWaitQueue,
               commandWaitValue,
               commandRetireValue,
               submitted,
               completed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

destroyHeap ==
  /\ heapLogical
  /\ heapLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       heapKind,
       heapPhysical,
       bufferKind,
       mapping,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       oldRetained,
       oldMapping,
       oldRetireQueue,
       oldRetireValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapping,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       submitted,
       completed
     >>

collectHeap ==
  /\ heapPhysical
  /\ ~HeapHasRetainer
  /\ heapPhysical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       heapKind,
       heapLogical,
       bufferKind,
       mapping,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       oldRetained,
       oldMapping,
       oldRetireQueue,
       oldRetireValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapping,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       submitted,
       completed
     >>

attemptCreateBuffer(kind) ==
  /\ deviceExists
  /\ bufferKind = "none"
  /\ IF kind = "ordinary" \/ capable
     THEN /\ bufferKind' = kind
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               deviceExists,
               capable,
               heapKind,
               heapLogical,
               heapPhysical,
               mapping,
               version,
               publicationQueue,
               publicationValue,
               mappingWaitQueue,
               mappingWaitValue,
               oldRetained,
               oldMapping,
               oldRetireQueue,
               oldRetireValue,
               commandPhase,
               commandKind,
               commandVersion,
               commandMapping,
               commandQueue,
               commandPublicationQueue,
               commandPublicationValue,
               commandWaitQueue,
               commandWaitValue,
               commandRetireValue,
               submitted,
               completed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptBindSparseBuffer(queue, target) ==
  /\ deviceExists
  /\ version < MaxVersion
  /\ submitted[queue] < MaxTimeline
  /\ LET nextValue == submitted[queue] + 1
         physicalTargetAvailable ==
           target = Unmapped \/ mapping = Unmapped \/ target = mapping
         valid ==
           /\ capable
           /\ bufferKind = "sparse"
           /\ (target = Unmapped \/ (heapLogical /\ heapPhysical /\ heapKind = "sparse"))
           /\ physicalTargetAvailable
           /\ (~oldRetained \/ oldMapping = Unmapped)
           /\ (commandPhase = "idle" \/ commandVersion # version)
     IN IF valid
        THEN /\ mapping' = target
             /\ version' = version + 1
             /\ publicationQueue' = queue
             /\ publicationValue' = nextValue
             /\ mappingWaitQueue' = publicationQueue
             /\ mappingWaitValue' = publicationValue
             /\ oldRetained' = TRUE
             /\ oldMapping' = mapping
             /\ oldRetireQueue' = queue
             /\ oldRetireValue' = nextValue
             /\ submitted' = [submitted EXCEPT ![queue] = nextValue]
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  deviceExists,
                  capable,
                  heapKind,
                  heapLogical,
                  heapPhysical,
                  bufferKind,
                  commandPhase,
                  commandKind,
                  commandVersion,
                  commandMapping,
                  commandQueue,
                  commandPublicationQueue,
                  commandPublicationValue,
                  commandWaitQueue,
                  commandWaitValue,
                  commandRetireValue,
                  completed
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

collectOldVersion ==
  /\ oldRetained
  /\ completed[oldRetireQueue] >= oldRetireValue
  /\ (version > 1 /\ publicationQueue # mappingWaitQueue
      => completed[mappingWaitQueue] >= mappingWaitValue)
  /\ ~(commandPhase # "idle" /\ commandVersion + 1 = version)
  /\ oldRetained' = FALSE
  /\ oldMapping' = Unmapped
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       heapKind,
       heapLogical,
       heapPhysical,
       bufferKind,
       mapping,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       oldRetireQueue,
       oldRetireValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapping,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       submitted,
       completed
     >>

attemptRecordCommand(queue, kind) ==
  /\ deviceExists
  /\ commandPhase = "idle"
  /\ IF bufferKind = "sparse" /\ (kind = "shader" \/ mapping # Unmapped)
     THEN /\ commandPhase' = "recorded"
          /\ commandKind' = kind
          /\ commandVersion' = version
          /\ commandMapping' = mapping
          /\ commandQueue' = queue
          /\ commandPublicationQueue' = publicationQueue
          /\ commandPublicationValue' = publicationValue
          /\ commandWaitQueue' = 0
          /\ commandWaitValue' = 0
          /\ commandRetireValue' = 0
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               deviceExists,
               capable,
               heapKind,
               heapLogical,
               heapPhysical,
               bufferKind,
               mapping,
               version,
               publicationQueue,
               publicationValue,
               mappingWaitQueue,
               mappingWaitValue,
               oldRetained,
               oldMapping,
               oldRetireQueue,
               oldRetireValue,
               submitted,
               completed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

abandonCommand ==
  /\ commandPhase = "recorded"
  /\ commandPhase' = "idle"
  /\ commandKind' = "none"
  /\ commandMapping' = Unmapped
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       heapKind,
       heapLogical,
       heapPhysical,
       bufferKind,
       mapping,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       oldRetained,
       oldMapping,
       oldRetireQueue,
       oldRetireValue,
       commandVersion,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       submitted,
       completed
     >>

attemptSubmitCommand(waitQueue, waitValue) ==
  /\ commandPhase = "recorded"
  /\ submitted[commandQueue] < MaxTimeline
  /\ LET nextValue == submitted[commandQueue] + 1
         valid ==
           waitValue <= submitted[waitQueue] /\ CommandOrdered(waitQueue, waitValue)
     IN IF valid
        THEN /\ commandPhase' = "inFlight"
             /\ commandWaitQueue' = waitQueue
             /\ commandWaitValue' = waitValue
             /\ commandRetireValue' = nextValue
             /\ submitted' = [submitted EXCEPT ![commandQueue] = nextValue]
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  deviceExists,
                  capable,
                  heapKind,
                  heapLogical,
                  heapPhysical,
                  bufferKind,
                  mapping,
                  version,
                  publicationQueue,
                  publicationValue,
                  mappingWaitQueue,
                  mappingWaitValue,
                  oldRetained,
                  oldMapping,
                  oldRetireQueue,
                  oldRetireValue,
                  commandKind,
                  commandVersion,
                  commandMapping,
                  commandQueue,
                  commandPublicationQueue,
                  commandPublicationValue,
                  completed
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

\* GPU completion enforces cross-queue fence waits enqueued at nextPoint.
complete(queue) ==
  /\ completed[queue] < submitted[queue]
  /\ LET nextPoint == completed[queue] + 1
     IN /\ (version > 1
            /\ queue = publicationQueue
            /\ nextPoint = publicationValue
            /\ publicationQueue # mappingWaitQueue
            => completed[mappingWaitQueue] >= mappingWaitValue)
        /\ (commandPhase = "inFlight"
            /\ queue = commandQueue
            /\ nextPoint = commandRetireValue
            /\ commandWaitValue > 0
            /\ commandQueue # commandWaitQueue
            => completed[commandWaitQueue] >= commandWaitValue)
        /\ completed' = [completed EXCEPT ![queue] = nextPoint]
        /\ lastOutcome' = "none"
        /\ UNCHANGED <<
             deviceExists,
             capable,
             heapKind,
             heapLogical,
             heapPhysical,
             bufferKind,
             mapping,
             version,
             publicationQueue,
             publicationValue,
             mappingWaitQueue,
             mappingWaitValue,
             oldRetained,
             oldMapping,
             oldRetireQueue,
             oldRetireValue,
             commandPhase,
             commandKind,
             commandVersion,
             commandMapping,
             commandQueue,
             commandPublicationQueue,
             commandPublicationValue,
             commandWaitQueue,
             commandWaitValue,
             commandRetireValue,
             submitted
           >>

collectCommand ==
  /\ commandPhase = "inFlight"
  /\ completed[commandQueue] >= commandRetireValue
  /\ commandPhase' = "idle"
  /\ commandKind' = "none"
  /\ commandMapping' = Unmapped
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       heapKind,
       heapLogical,
       heapPhysical,
       bufferKind,
       mapping,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       oldRetained,
       oldMapping,
       oldRetireQueue,
       oldRetireValue,
       commandVersion,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       submitted,
       completed
     >>

Next ==
  \/ \E supports \in BOOLEAN : createDevice(supports)
  \/ \E kind \in AllocatableKinds : attemptCreateHeap(kind)
  \/ destroyHeap
  \/ collectHeap
  \/ \E kind \in AllocatableKinds : attemptCreateBuffer(kind)
  \/ \E queue \in Queues, target \in Targets : attemptBindSparseBuffer(queue, target)
  \/ collectOldVersion
  \/ \E queue \in Queues, kind \in RecordableCommandKinds : attemptRecordCommand(queue, kind)
  \/ abandonCommand
  \/ \E waitQueue \in Queues, waitValue \in 0..MaxTimeline :
       (waitValue = 0 => waitQueue = 0) /\ attemptSubmitCommand(waitQueue, waitValue)
  \/ \E queue \in Queues : complete(queue)
  \/ collectCommand

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ deviceExists \in BOOLEAN
  /\ capable \in BOOLEAN
  /\ heapKind \in Kinds
  /\ heapLogical \in BOOLEAN
  /\ heapPhysical \in BOOLEAN
  /\ bufferKind \in Kinds
  /\ mapping \in Targets
  /\ version \in 0..MaxVersion
  /\ publicationQueue \in Queues
  /\ publicationValue \in 0..MaxTimeline
  /\ mappingWaitQueue \in Queues
  /\ mappingWaitValue \in 0..MaxTimeline
  /\ oldRetained \in BOOLEAN
  /\ oldMapping \in Targets
  /\ oldRetireQueue \in Queues
  /\ oldRetireValue \in 0..MaxTimeline
  /\ commandPhase \in CommandPhases
  /\ commandKind \in CommandKinds
  /\ commandVersion \in 0..MaxVersion
  /\ commandMapping \in Targets
  /\ commandQueue \in Queues
  /\ commandPublicationQueue \in Queues
  /\ commandPublicationValue \in 0..MaxTimeline
  /\ commandWaitQueue \in Queues
  /\ commandWaitValue \in 0..MaxTimeline
  /\ commandRetireValue \in 0..MaxTimeline
  /\ submitted \in [Queues -> 0..MaxTimeline]
  /\ completed \in [Queues -> 0..MaxTimeline]
  /\ lastOutcome \in Outcomes
  /\ \A q \in Queues : completed[q] <= submitted[q]

SparseObjectsAreCapabilityGated ==
  (heapKind = "sparse" \/ bufferKind = "sparse") => capable

MappingsUseDedicatedSparseObjects ==
  mapping # Unmapped => bufferKind = "sparse" /\ heapKind = "sparse"

PublicationIsExact ==
  version > 0 =>
    /\ publicationValue > 0
    /\ publicationValue <= submitted[publicationQueue]

RetainedVersionsReferencePublishedWork ==
  oldRetained =>
    /\ version > 0
    /\ oldRetireValue > 0
    /\ oldRetireValue <= submitted[oldRetireQueue]

SequentialMappingPublicationIsOrdered ==
  version > 1 =>
    /\ mappingWaitValue > 0
    /\ mappingWaitValue <= submitted[mappingWaitQueue]
    /\ (completed[publicationQueue] >= publicationValue
        => completed[mappingWaitQueue] >= mappingWaitValue)

PhysicalRangesDoNotConflict ==
  (oldRetained /\ oldMapping # Unmapped /\ mapping # Unmapped)
    => oldMapping = mapping

CommandsCaptureImmutableVersions ==
  commandPhase # "idle" =>
    /\ commandKind \in RecordableCommandKinds
    /\ commandVersion = version
    /\ (commandKind = "shader" \/ commandMapping # Unmapped)
    /\ (commandVersion = 0 <=> commandPublicationValue = 0)

IdleCommandsOwnNoGeneration ==
  commandPhase = "idle" => commandKind = "none" /\ commandMapping = Unmapped

CrossQueueUseIsOrdered ==
  commandPhase = "inFlight" =>
    /\ (commandQueue # commandPublicationQueue
        /\ ~PublicationComplete(commandPublicationQueue, commandPublicationValue)
        => commandWaitQueue = commandPublicationQueue
           /\ commandWaitValue >= commandPublicationValue)
    /\ (completed[commandQueue] >= commandRetireValue
        => PublicationComplete(commandPublicationQueue, commandPublicationValue))

MappedOrCapturedStorageRetainsTheHeap ==
  HeapHasRetainer => heapPhysical

Safety ==
  /\ SparseObjectsAreCapabilityGated
  /\ MappingsUseDedicatedSparseObjects
  /\ PublicationIsExact
  /\ RetainedVersionsReferencePublishedWork
  /\ SequentialMappingPublicationIsOrdered
  /\ PhysicalRangesDoNotConflict
  /\ CommandsCaptureImmutableVersions
  /\ IdleCommandsOwnNoGeneration
  /\ CrossQueueUseIsOrdered
  /\ MappedOrCapturedStorageRetainsTheHeap

StepSafety ==
  [][lastOutcome' = "rejected" => coreVars' = coreVars]_vars

=============================================================================
