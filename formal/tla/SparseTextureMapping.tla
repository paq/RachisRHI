------------------------ MODULE SparseTextureMapping ------------------------
(***************************************************************************)
(* Formal safety model for copy-visible sparse 2D textures, standard mip   *)
(* tiles, atomic packed-mip tails, cross-queue fence-ordered publication,  *)
(* and physical heap retention:                                            *)
(*   - src/D3D12/D3D12Device.cpp (sparseTextures capability discovery)     *)
(*   - src/D3D12/D3D12SparseAndFeedback.cpp (createSparseTexture,          *)
(*     querySparseTextureTiling standard-mip grids and packed tails,       *)
(*     querySparseTextureProperties, enqueueSparseTextureMappings and its  *)
(*     in-use remap rejection)                                             *)
(*   - src/D3D12/D3D12Resource.cpp (sparse-backing heaps, copy/barrier     *)
(*     mapping capture), src/D3D12/D3D12ResourceCopies.cpp (mapped-tile    *)
(*     copy checks)                                                        *)
(*   - src/D3D12/D3D12Queue.cpp (enqueueSparseMappings serialization,      *)
(*     validateSparseMappingDependencies)                                  *)
(*                                                                         *)
(* Models dedicated sparse heaps with independent physical reclamation     *)
(* (heapPhysical), a 2-tile packed-mip tail where partial 1-tile requests  *)
(* reject atomically so tailMappedTiles is never partially populated,      *)
(* cross-queue GPU fence wait enforcement on both sequential mapping       *)
(* updates and copy commands, and command abandonment vs completion.       *)
(*                                                                         *)
(* Trusted boundary: D3D12 UpdateTileMappings hardware MMU programming     *)
(* and terminal native device-loss teardown outside this finite model.     *)
(***************************************************************************)
EXTENDS Naturals

Queues == {0, 1}
Units == {"standard", "tail"}
Kinds == {"none", "ordinary", "sparse"}
AllocatableKinds == {"ordinary", "sparse"}
CommandPhases == {"idle", "recorded", "inFlight"}
Outcomes == {"none", "accepted", "rejected"}
TailTileCount == 2
MaxTimeline == 3
MaxVersion == 2

VARIABLES
  deviceExists,            \* Whether the RHI device has been initialized.
  capable,                 \* Whether Tier 2 sparse textures are supported.
  heapKind,                \* Allocated heap kind (none, ordinary, sparse).
  heapLogical,             \* Whether the public heap handle is live.
  heapPhysical,            \* Whether native heap memory is physically alive.
  textureKind,             \* Allocated texture kind (none, ordinary, sparse).
  standardMapped,          \* Whether the standard-mip tile is mapped.
  tailMappedTiles,         \* Count of mapped tiles in packed tail (0 or 2; 1 is partial).
  version,                 \* Current immutable mapping generation (0..MaxVersion).
  publicationQueue,        \* Queue carrying the latest mapping publication signal.
  publicationValue,        \* Fence value signaled for the latest mapping version.
  mappingWaitQueue,        \* Prior publication queue awaited when version > 1.
  mappingWaitValue,        \* Prior publication fence value awaited when version > 1.
  commandPhase,            \* Copy command lifecycle phase.
  commandUnit,             \* Footprint unit copied by the command ("standard" or "tail").
  commandVersion,          \* Mapping generation captured at copy recording.
  commandQueue,            \* Queue on which the copy command is recorded/submitted.
  commandPublicationQueue, \* Mapping publication queue captured by the copy command.
  commandPublicationValue, \* Mapping publication fence value captured by copy.
  commandWaitQueue,        \* Explicit wait queue supplied at submitCopy.
  commandWaitValue,        \* Explicit wait value supplied at submitCopy.
  commandRetireValue,      \* Submission fence value retiring the copy command.
  submitted,               \* Per-queue submitted timeline frontier.
  completed,               \* Per-queue GPU-completed timeline frontier.
  lastOutcome              \* Outcome of the latest fallible request.

coreVars == <<
  deviceExists,
  capable,
  heapKind,
  heapLogical,
  heapPhysical,
  textureKind,
  standardMapped,
  tailMappedTiles,
  version,
  publicationQueue,
  publicationValue,
  mappingWaitQueue,
  mappingWaitValue,
  commandPhase,
  commandUnit,
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

vars == <<
  deviceExists,
  capable,
  heapKind,
  heapLogical,
  heapPhysical,
  textureKind,
  standardMapped,
  tailMappedTiles,
  version,
  publicationQueue,
  publicationValue,
  mappingWaitQueue,
  mappingWaitValue,
  commandPhase,
  commandUnit,
  commandVersion,
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

UnitMapped(unit) ==
  IF unit = "standard"
  THEN standardMapped
  ELSE tailMappedTiles = TailTileCount

PublicationComplete(queue, value) ==
  value = 0 \/ completed[queue] >= value

CommandOrdered(waitQueue, waitValue) ==
  \/ commandQueue = commandPublicationQueue
  \/ PublicationComplete(commandPublicationQueue, commandPublicationValue)
  \/ (waitQueue = commandPublicationQueue /\ waitValue >= commandPublicationValue)

HeapHasRetainer ==
  \/ heapLogical
  \/ standardMapped
  \/ tailMappedTiles > 0
  \/ commandPhase # "idle"

Init ==
  /\ deviceExists = FALSE
  /\ capable = FALSE
  /\ heapKind = "none"
  /\ heapLogical = FALSE
  /\ heapPhysical = FALSE
  /\ textureKind = "none"
  /\ standardMapped = FALSE
  /\ tailMappedTiles = 0
  /\ version = 0
  /\ publicationQueue = 0
  /\ publicationValue = 0
  /\ mappingWaitQueue = 0
  /\ mappingWaitValue = 0
  /\ commandPhase = "idle"
  /\ commandUnit = "standard"
  /\ commandVersion = 0
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
       textureKind,
       standardMapped,
       tailMappedTiles,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandUnit,
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
               textureKind,
               standardMapped,
               tailMappedTiles,
               version,
               publicationQueue,
               publicationValue,
               mappingWaitQueue,
               mappingWaitValue,
               commandPhase,
               commandUnit,
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
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptCreateTexture(kind) ==
  /\ deviceExists
  /\ textureKind = "none"
  /\ IF kind = "ordinary" \/ capable
     THEN /\ textureKind' = kind
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               deviceExists,
               capable,
               heapKind,
               heapLogical,
               heapPhysical,
               standardMapped,
               tailMappedTiles,
               version,
               publicationQueue,
               publicationValue,
               mappingWaitQueue,
               mappingWaitValue,
               commandPhase,
               commandUnit,
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
       textureKind,
       standardMapped,
       tailMappedTiles,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandUnit,
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
       textureKind,
       standardMapped,
       tailMappedTiles,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandUnit,
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

attemptMapStandard(queue, mapped) ==
  /\ deviceExists
  /\ version < MaxVersion
  /\ submitted[queue] < MaxTimeline
  /\ LET nextValue == submitted[queue] + 1
         valid ==
           /\ capable
           /\ textureKind = "sparse"
           /\ (~mapped \/ (heapLogical /\ heapPhysical /\ heapKind = "sparse"))
           /\ commandPhase = "idle"
     IN IF valid
        THEN /\ standardMapped' = mapped
             /\ version' = version + 1
             /\ publicationQueue' = queue
             /\ publicationValue' = nextValue
             /\ mappingWaitQueue' = publicationQueue
             /\ mappingWaitValue' = publicationValue
             /\ submitted' = [submitted EXCEPT ![queue] = nextValue]
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  deviceExists,
                  capable,
                  heapKind,
                  heapLogical,
                  heapPhysical,
                  textureKind,
                  tailMappedTiles,
                  commandPhase,
                  commandUnit,
                  commandVersion,
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

\* A packed-mip tail request specifies 0 (unmap whole tail), 1 (invalid partial
\* tail), or 2 (map whole tail); partial tail requests reject atomically.
attemptMapTail(queue, requestedTiles) ==
  /\ deviceExists
  /\ version < MaxVersion
  /\ submitted[queue] < MaxTimeline
  /\ LET nextValue == submitted[queue] + 1
         valid ==
           /\ capable
           /\ textureKind = "sparse"
           /\ requestedTiles \in {0, TailTileCount}
           /\ (requestedTiles = 0 \/ (heapLogical /\ heapPhysical /\ heapKind = "sparse"))
           /\ commandPhase = "idle"
     IN IF valid
        THEN /\ tailMappedTiles' = requestedTiles
             /\ version' = version + 1
             /\ publicationQueue' = queue
             /\ publicationValue' = nextValue
             /\ mappingWaitQueue' = publicationQueue
             /\ mappingWaitValue' = publicationValue
             /\ submitted' = [submitted EXCEPT ![queue] = nextValue]
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  deviceExists,
                  capable,
                  heapKind,
                  heapLogical,
                  heapPhysical,
                  textureKind,
                  standardMapped,
                  commandPhase,
                  commandUnit,
                  commandVersion,
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

attemptCaptureCopy(queue, unit) ==
  /\ deviceExists
  /\ commandPhase = "idle"
  /\ IF textureKind = "sparse" /\ UnitMapped(unit)
     THEN /\ commandPhase' = "recorded"
          /\ commandUnit' = unit
          /\ commandVersion' = version
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
               textureKind,
               standardMapped,
               tailMappedTiles,
               version,
               publicationQueue,
               publicationValue,
               mappingWaitQueue,
               mappingWaitValue,
               submitted,
               completed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

abandonCopy ==
  /\ commandPhase = "recorded"
  /\ commandPhase' = "idle"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       heapKind,
       heapLogical,
       heapPhysical,
       textureKind,
       standardMapped,
       tailMappedTiles,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandUnit,
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

attemptSubmitCopy(waitQueue, waitValue) ==
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
                  textureKind,
                  standardMapped,
                  tailMappedTiles,
                  version,
                  publicationQueue,
                  publicationValue,
                  mappingWaitQueue,
                  mappingWaitValue,
                  commandUnit,
                  commandVersion,
                  commandQueue,
                  commandPublicationQueue,
                  commandPublicationValue,
                  completed
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

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
             textureKind,
             standardMapped,
             tailMappedTiles,
             version,
             publicationQueue,
             publicationValue,
             mappingWaitQueue,
             mappingWaitValue,
             commandPhase,
             commandUnit,
             commandVersion,
             commandQueue,
             commandPublicationQueue,
             commandPublicationValue,
             commandWaitQueue,
             commandWaitValue,
             commandRetireValue,
             submitted
           >>

collectCopy ==
  /\ commandPhase = "inFlight"
  /\ completed[commandQueue] >= commandRetireValue
  /\ commandPhase' = "idle"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       heapKind,
       heapLogical,
       heapPhysical,
       textureKind,
       standardMapped,
       tailMappedTiles,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandUnit,
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
  \/ \E kind \in AllocatableKinds : attemptCreateTexture(kind)
  \/ destroyHeap
  \/ collectHeap
  \/ \E queue \in Queues, mapped \in BOOLEAN : attemptMapStandard(queue, mapped)
  \/ \E queue \in Queues, tiles \in 0..TailTileCount : attemptMapTail(queue, tiles)
  \/ \E queue \in Queues, unit \in Units : attemptCaptureCopy(queue, unit)
  \/ abandonCopy
  \/ \E waitQueue \in Queues, waitValue \in 0..MaxTimeline :
       (waitValue = 0 => waitQueue = 0) /\ attemptSubmitCopy(waitQueue, waitValue)
  \/ \E queue \in Queues : complete(queue)
  \/ collectCopy

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ deviceExists \in BOOLEAN
  /\ capable \in BOOLEAN
  /\ heapKind \in Kinds
  /\ heapLogical \in BOOLEAN
  /\ heapPhysical \in BOOLEAN
  /\ textureKind \in Kinds
  /\ standardMapped \in BOOLEAN
  /\ tailMappedTiles \in 0..TailTileCount
  /\ version \in 0..MaxVersion
  /\ publicationQueue \in Queues
  /\ publicationValue \in 0..MaxTimeline
  /\ mappingWaitQueue \in Queues
  /\ mappingWaitValue \in 0..MaxTimeline
  /\ commandPhase \in CommandPhases
  /\ commandUnit \in Units
  /\ commandVersion \in 0..MaxVersion
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
  (heapKind = "sparse" \/ textureKind = "sparse") => capable

MappingsUseDedicatedSparseObjects ==
  (standardMapped \/ tailMappedTiles > 0)
    => textureKind = "sparse" /\ heapKind = "sparse"

PackedMipTailIsAtomic ==
  tailMappedTiles \in {0, TailTileCount}

SequentialMappingPublicationIsOrdered ==
  version > 1 =>
    /\ mappingWaitValue > 0
    /\ mappingWaitValue <= submitted[mappingWaitQueue]
    /\ (completed[publicationQueue] >= publicationValue
        => completed[mappingWaitQueue] >= mappingWaitValue)

CopiesCaptureMappedImmutableGenerations ==
  commandPhase # "idle" =>
    /\ commandVersion = version
    /\ UnitMapped(commandUnit)
    /\ commandPublicationValue > 0

CrossQueueCopyIsOrdered ==
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
  /\ PackedMipTailIsAtomic
  /\ SequentialMappingPublicationIsOrdered
  /\ CopiesCaptureMappedImmutableGenerations
  /\ CrossQueueCopyIsOrdered
  /\ MappedOrCapturedStorageRetainsTheHeap

StepSafety ==
  [][lastOutcome' = "rejected" => coreVars' = coreVars]_vars

=============================================================================
