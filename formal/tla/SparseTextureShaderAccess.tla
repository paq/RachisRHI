--------------------- MODULE SparseTextureShaderAccess ----------------------
(***************************************************************************)
(* Formal safety model for reusable sparse texture views, binding sets,    *)
(* command-time generation capture, cross-queue fence ordering, and GPU    *)
(* execution-time Tier 2 null-tile read/write semantics:                   *)
(*   - src/D3D12/D3D12View.cpp (reusable texture views)                    *)
(*   - src/D3D12/D3D12Binding.cpp (nativeBindingSet sparse capture)        *)
(*   - src/D3D12/D3D12SparseAndFeedback.cpp (captureSparseTextureBinding,  *)
(*     enqueueSparseTextureMappings remap rejection)                       *)
(*   - src/D3D12/D3D12CommandRecording.cpp (prepareBindingSetBatch,        *)
(*     retainBindingSetBatchDependencies)                                  *)
(*   - src/D3D12/D3D12Queue.cpp (validateSparseMappingDependencies,        *)
(*     retirement)                                                         *)
(*                                                                         *)
(* Models logical destruction and independent physical reclamation across  *)
(* heap, texture, view, and bindingSet; demonstrates that descriptors      *)
(* created at setCreationVersion do not freeze generations when remapped   *)
(* later; enforces cross-queue fence waits on GPU completion; and          *)
(* evaluates shader loads/stores at GPU completion rather than submit.     *)
(*                                                                         *)
(* Trusted boundary: D3D12 hardware MMU null-tile zero-read and            *)
(* write-discard behavior once page-table updates complete.                *)
(***************************************************************************)
EXTENDS Naturals

Queues == {0, 1}
ObjectKinds == {"heap", "texture", "view", "bindingSet"}
Kinds == {"none", "sample", "store"}
ShaderKinds == {"sample", "store"}
Phases == {"idle", "recorded", "inFlight"}
Outcomes == {"none", "accepted", "rejected"}
MaxTimeline == 2
MaxTotalSubmissions == 3
MaxVersion == 2

VARIABLES
  deviceExists,            \* Whether the RHI device has been initialized.
  capable,                 \* Whether Tier 2 sparse textures are supported.
  created,                 \* Subset of ObjectKinds already allocated once.
  heapLogical,             \* Public sparse-heap handle liveness.
  heapPhysical,            \* Physical sparse-heap memory liveness.
  textureLogical,          \* Public sparse-texture handle liveness.
  texturePhysical,         \* Physical virtual-texture object liveness.
  viewLogical,             \* Public texture-view handle liveness.
  viewPhysical,            \* Physical texture-view descriptor liveness.
  bindingSetLogical,       \* Public binding-set handle liveness.
  bindingSetPhysical,      \* Physical binding-set table liveness.
  setCreationVersion,      \* Mapping generation that was current at createBindingSet.
  mapped,                  \* Whether the tile is mapped in the current generation.
  version,                 \* Current immutable mapping generation (0..MaxVersion).
  publicationQueue,        \* Queue carrying the latest mapping publication signal.
  publicationValue,        \* Fence value signaled for the latest mapping version.
  mappingWaitQueue,        \* Prior publication queue awaited when version > 1.
  mappingWaitValue,        \* Prior publication fence value awaited when version > 1.
  commandPhase,            \* Shader command lifecycle phase.
  commandKind,             \* Recorded shader access kind ("sample" or "store").
  commandVersion,          \* Mapping generation captured when binding the set.
  commandMapped,           \* Tile residency in the command-captured generation.
  commandQueue,            \* Queue on which the shader command is recorded/submitted.
  commandPublicationQueue, \* Mapping publication queue captured at bind time.
  commandPublicationValue, \* Mapping publication fence value captured at bind time.
  commandWaitQueue,        \* Explicit wait queue supplied at submitShader.
  commandWaitValue,        \* Explicit wait value supplied at submitShader.
  commandRetireValue,      \* Submission fence value retiring the shader command.
  physicalValue,           \* Payload currently stored in physical heap tile (0..1).
  preStorePhysicalValue,   \* Heap payload immediately before GPU shader execution.
  sampledValue,            \* Payload observed by the latest completed sample (0..1).
  sampleObserved,          \* Whether the latest completed GPU shader run was a sample.
  storeExecuted,           \* Whether the latest completed GPU shader run was a store.
  executedMapped,          \* Residency of the generation used by latest GPU run.
  submitted,               \* Per-queue submitted timeline frontier.
  completed,               \* Per-queue GPU-completed timeline frontier.
  lastOutcome              \* Outcome of the latest fallible operation.

coreVars == <<
  deviceExists,
  capable,
  created,
  heapLogical,
  heapPhysical,
  textureLogical,
  texturePhysical,
  viewLogical,
  viewPhysical,
  bindingSetLogical,
  bindingSetPhysical,
  setCreationVersion,
  mapped,
  version,
  publicationQueue,
  publicationValue,
  mappingWaitQueue,
  mappingWaitValue,
  commandPhase,
  commandKind,
  commandVersion,
  commandMapped,
  commandQueue,
  commandPublicationQueue,
  commandPublicationValue,
  commandWaitQueue,
  commandWaitValue,
  commandRetireValue,
  physicalValue,
  preStorePhysicalValue,
  sampledValue,
  sampleObserved,
  storeExecuted,
  executedMapped,
  submitted,
  completed
>>

vars == <<
  deviceExists,
  capable,
  created,
  heapLogical,
  heapPhysical,
  textureLogical,
  texturePhysical,
  viewLogical,
  viewPhysical,
  bindingSetLogical,
  bindingSetPhysical,
  setCreationVersion,
  mapped,
  version,
  publicationQueue,
  publicationValue,
  mappingWaitQueue,
  mappingWaitValue,
  commandPhase,
  commandKind,
  commandVersion,
  commandMapped,
  commandQueue,
  commandPublicationQueue,
  commandPublicationValue,
  commandWaitQueue,
  commandWaitValue,
  commandRetireValue,
  physicalValue,
  preStorePhysicalValue,
  sampledValue,
  sampleObserved,
  storeExecuted,
  executedMapped,
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

Init ==
  /\ deviceExists = FALSE
  /\ capable = FALSE
  /\ created = {}
  /\ heapLogical = FALSE
  /\ heapPhysical = FALSE
  /\ textureLogical = FALSE
  /\ texturePhysical = FALSE
  /\ viewLogical = FALSE
  /\ viewPhysical = FALSE
  /\ bindingSetLogical = FALSE
  /\ bindingSetPhysical = FALSE
  /\ setCreationVersion = 0
  /\ mapped = FALSE
  /\ version = 0
  /\ publicationQueue = 0
  /\ publicationValue = 0
  /\ mappingWaitQueue = 0
  /\ mappingWaitValue = 0
  /\ commandPhase = "idle"
  /\ commandKind = "none"
  /\ commandVersion = 0
  /\ commandMapped = FALSE
  /\ commandQueue = 0
  /\ commandPublicationQueue = 0
  /\ commandPublicationValue = 0
  /\ commandWaitQueue = 0
  /\ commandWaitValue = 0
  /\ commandRetireValue = 0
  /\ physicalValue = 0
  /\ preStorePhysicalValue = 0
  /\ sampledValue = 0
  /\ sampleObserved = FALSE
  /\ storeExecuted = FALSE
  /\ executedMapped = FALSE
  /\ submitted = [q \in Queues |-> 0]
  /\ completed = [q \in Queues |-> 0]
  /\ lastOutcome = "none"

createDevice(supportsSparse) ==
  /\ ~deviceExists
  /\ deviceExists' = TRUE
  /\ capable' = supportsSparse
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       created,
       heapLogical,
       heapPhysical,
       textureLogical,
       texturePhysical,
       viewLogical,
       viewPhysical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

attemptCreateHeap ==
  /\ deviceExists
  /\ ~("heap" \in created)
  /\ IF capable
     THEN /\ created' = created \cup {"heap"}
          /\ heapLogical' = TRUE
          /\ heapPhysical' = TRUE
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               deviceExists,
               capable,
               textureLogical,
               texturePhysical,
               viewLogical,
               viewPhysical,
               bindingSetLogical,
               bindingSetPhysical,
               setCreationVersion,
               mapped,
               version,
               publicationQueue,
               publicationValue,
               mappingWaitQueue,
               mappingWaitValue,
               commandPhase,
               commandKind,
               commandVersion,
               commandMapped,
               commandQueue,
               commandPublicationQueue,
               commandPublicationValue,
               commandWaitQueue,
               commandWaitValue,
               commandRetireValue,
               physicalValue,
               preStorePhysicalValue,
               sampledValue,
               sampleObserved,
               storeExecuted,
               executedMapped,
               submitted,
               completed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptCreateTexture ==
  /\ deviceExists
  /\ ~("texture" \in created)
  /\ IF capable
     THEN /\ created' = created \cup {"texture"}
          /\ textureLogical' = TRUE
          /\ texturePhysical' = TRUE
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               deviceExists,
               capable,
               heapLogical,
               heapPhysical,
               viewLogical,
               viewPhysical,
               bindingSetLogical,
               bindingSetPhysical,
               setCreationVersion,
               mapped,
               version,
               publicationQueue,
               publicationValue,
               mappingWaitQueue,
               mappingWaitValue,
               commandPhase,
               commandKind,
               commandVersion,
               commandMapped,
               commandQueue,
               commandPublicationQueue,
               commandPublicationValue,
               commandWaitQueue,
               commandWaitValue,
               commandRetireValue,
               physicalValue,
               preStorePhysicalValue,
               sampledValue,
               sampleObserved,
               storeExecuted,
               executedMapped,
               submitted,
               completed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

createView ==
  /\ textureLogical
  /\ ~("view" \in created)
  /\ created' = created \cup {"view"}
  /\ viewLogical' = TRUE
  /\ viewPhysical' = TRUE
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       heapLogical,
       heapPhysical,
       textureLogical,
       texturePhysical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

\* Records setCreationVersion so later remaps prove descriptors do not freeze it.
createBindingSet ==
  /\ viewLogical
  /\ ~("bindingSet" \in created)
  /\ created' = created \cup {"bindingSet"}
  /\ bindingSetLogical' = TRUE
  /\ bindingSetPhysical' = TRUE
  /\ setCreationVersion' = version
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       heapLogical,
       heapPhysical,
       textureLogical,
       texturePhysical,
       viewLogical,
       viewPhysical,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

destroyBindingSet ==
  /\ bindingSetLogical
  /\ bindingSetLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       created,
       heapLogical,
       heapPhysical,
       textureLogical,
       texturePhysical,
       viewLogical,
       viewPhysical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

collectBindingSet ==
  /\ bindingSetPhysical
  /\ ~bindingSetLogical
  /\ commandPhase = "idle"
  /\ bindingSetPhysical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       created,
       heapLogical,
       heapPhysical,
       textureLogical,
       texturePhysical,
       viewLogical,
       viewPhysical,
       bindingSetLogical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

destroyView ==
  /\ viewLogical
  /\ viewLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       created,
       heapLogical,
       heapPhysical,
       textureLogical,
       texturePhysical,
       viewPhysical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

collectView ==
  /\ viewPhysical
  /\ ~viewLogical
  /\ ~bindingSetPhysical
  /\ viewPhysical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       created,
       heapLogical,
       heapPhysical,
       textureLogical,
       texturePhysical,
       viewLogical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

destroyTexture ==
  /\ textureLogical
  /\ textureLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       created,
       heapLogical,
       heapPhysical,
       texturePhysical,
       viewLogical,
       viewPhysical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

collectTexture ==
  /\ texturePhysical
  /\ ~textureLogical
  /\ ~viewPhysical
  /\ commandPhase = "idle"
  /\ texturePhysical' = FALSE
  /\ mapped' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       created,
       heapLogical,
       heapPhysical,
       textureLogical,
       viewLogical,
       viewPhysical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

destroyHeap ==
  /\ heapLogical
  /\ heapLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       created,
       heapPhysical,
       textureLogical,
       texturePhysical,
       viewLogical,
       viewPhysical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

collectHeap ==
  /\ heapPhysical
  /\ ~heapLogical
  /\ ~mapped
  /\ (commandPhase = "idle" \/ ~commandMapped)
  /\ heapPhysical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       created,
       heapLogical,
       textureLogical,
       texturePhysical,
       viewLogical,
       viewPhysical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       commandPhase,
       commandKind,
       commandVersion,
       commandMapped,
       commandQueue,
       commandPublicationQueue,
       commandPublicationValue,
       commandWaitQueue,
       commandWaitValue,
       commandRetireValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

attemptUpdateMapping(queue, requestedMapped) ==
  /\ deviceExists
  /\ version < MaxVersion
  /\ submitted[queue] < MaxTimeline
  /\ submitted[0] + submitted[1] < MaxTotalSubmissions
  /\ LET nextValue == submitted[queue] + 1
         valid ==
           /\ capable
           /\ textureLogical
           /\ (~requestedMapped \/ (heapLogical /\ heapPhysical))
           /\ commandPhase = "idle"
     IN IF valid
        THEN /\ mapped' = requestedMapped
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
                  created,
                  heapLogical,
                  heapPhysical,
                  textureLogical,
                  texturePhysical,
                  viewLogical,
                  viewPhysical,
                  bindingSetLogical,
                  bindingSetPhysical,
                  setCreationVersion,
                  commandPhase,
                  commandKind,
                  commandVersion,
                  commandMapped,
                  commandQueue,
                  commandPublicationQueue,
                  commandPublicationValue,
                  commandWaitQueue,
                  commandWaitValue,
                  commandRetireValue,
                  physicalValue,
                  preStorePhysicalValue,
                  sampledValue,
                  sampleObserved,
                  storeExecuted,
                  executedMapped,
                  completed
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

\* Binding the set on a command buffer captures the current generation (even
\* when version > setCreationVersion), without executing the shader yet.
recordShader(queue, kind) ==
  /\ bindingSetLogical
  /\ commandPhase = "idle"
  /\ commandPhase' = "recorded"
  /\ commandKind' = kind
  /\ commandVersion' = version
  /\ commandMapped' = mapped
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
       created,
       heapLogical,
       heapPhysical,
       textureLogical,
       texturePhysical,
       viewLogical,
       viewPhysical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

abandonShader ==
  /\ commandPhase = "recorded"
  /\ commandPhase' = "idle"
  /\ commandKind' = "none"
  /\ commandVersion' = 0
  /\ commandMapped' = FALSE
  /\ commandQueue' = 0
  /\ commandPublicationQueue' = 0
  /\ commandPublicationValue' = 0
  /\ commandWaitQueue' = 0
  /\ commandWaitValue' = 0
  /\ commandRetireValue' = 0
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       created,
       heapLogical,
       heapPhysical,
       textureLogical,
       texturePhysical,
       viewLogical,
       viewPhysical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

attemptSubmitShader(waitQueue, waitValue) ==
  /\ commandPhase = "recorded"
  /\ submitted[commandQueue] < MaxTimeline
  /\ submitted[0] + submitted[1] < MaxTotalSubmissions
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
                  created,
                  heapLogical,
                  heapPhysical,
                  textureLogical,
                  texturePhysical,
                  viewLogical,
                  viewPhysical,
                  bindingSetLogical,
                  bindingSetPhysical,
                  setCreationVersion,
                  mapped,
                  version,
                  publicationQueue,
                  publicationValue,
                  mappingWaitQueue,
                  mappingWaitValue,
                  commandKind,
                  commandVersion,
                  commandMapped,
                  commandQueue,
                  commandPublicationQueue,
                  commandPublicationValue,
                  physicalValue,
                  preStorePhysicalValue,
                  sampledValue,
                  sampleObserved,
                  storeExecuted,
                  executedMapped,
                  completed
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

\* Shader load/store executes on the GPU when nextPoint reaches commandRetireValue
\* after all cross-queue waits have completed, not at host submit time.
complete(queue) ==
  /\ completed[queue] < submitted[queue]
  /\ LET nextPoint == completed[queue] + 1
         executesShader ==
           commandPhase = "inFlight"
           /\ queue = commandQueue
           /\ nextPoint = commandRetireValue
     IN /\ (version > 1
            /\ queue = publicationQueue
            /\ nextPoint = publicationValue
            /\ publicationQueue # mappingWaitQueue
            => completed[mappingWaitQueue] >= mappingWaitValue)
        /\ (executesShader
            /\ commandWaitValue > 0
            /\ commandQueue # commandWaitQueue
            => completed[commandWaitQueue] >= commandWaitValue)
        /\ completed' = [completed EXCEPT ![queue] = nextPoint]
        /\ IF executesShader
           THEN /\ preStorePhysicalValue' = physicalValue
                /\ executedMapped' = commandMapped
                /\ IF commandKind = "store"
                   THEN /\ IF commandMapped
                           THEN physicalValue' = 1
                           ELSE UNCHANGED physicalValue
                        /\ sampleObserved' = FALSE
                        /\ storeExecuted' = TRUE
                        /\ UNCHANGED sampledValue
                   ELSE /\ sampledValue' = IF commandMapped THEN physicalValue ELSE 0
                        /\ sampleObserved' = TRUE
                        /\ storeExecuted' = FALSE
                        /\ UNCHANGED physicalValue
           ELSE UNCHANGED <<
                  physicalValue,
                  preStorePhysicalValue,
                  sampledValue,
                  sampleObserved,
                  storeExecuted,
                  executedMapped
                >>
        /\ lastOutcome' = "none"
        /\ UNCHANGED <<
             deviceExists,
             capable,
             created,
             heapLogical,
             heapPhysical,
             textureLogical,
             texturePhysical,
             viewLogical,
             viewPhysical,
             bindingSetLogical,
             bindingSetPhysical,
             setCreationVersion,
             mapped,
             version,
             publicationQueue,
             publicationValue,
             mappingWaitQueue,
             mappingWaitValue,
             commandPhase,
             commandKind,
             commandVersion,
             commandMapped,
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
  /\ commandVersion' = 0
  /\ commandMapped' = FALSE
  /\ commandQueue' = 0
  /\ commandPublicationQueue' = 0
  /\ commandPublicationValue' = 0
  /\ commandWaitQueue' = 0
  /\ commandWaitValue' = 0
  /\ commandRetireValue' = 0
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       deviceExists,
       capable,
       created,
       heapLogical,
       heapPhysical,
       textureLogical,
       texturePhysical,
       viewLogical,
       viewPhysical,
       bindingSetLogical,
       bindingSetPhysical,
       setCreationVersion,
       mapped,
       version,
       publicationQueue,
       publicationValue,
       mappingWaitQueue,
       mappingWaitValue,
       physicalValue,
       preStorePhysicalValue,
       sampledValue,
       sampleObserved,
       storeExecuted,
       executedMapped,
       submitted,
       completed
     >>

Next ==
  \/ \E supports \in BOOLEAN : createDevice(supports)
  \/ attemptCreateHeap
  \/ attemptCreateTexture
  \/ createView
  \/ createBindingSet
  \/ destroyBindingSet
  \/ collectBindingSet
  \/ destroyView
  \/ collectView
  \/ destroyTexture
  \/ collectTexture
  \/ destroyHeap
  \/ collectHeap
  \/ \E queue \in Queues, req \in BOOLEAN : attemptUpdateMapping(queue, req)
  \/ \E queue \in Queues, kind \in ShaderKinds : recordShader(queue, kind)
  \/ abandonShader
  \/ \E waitQueue \in Queues, waitValue \in 0..MaxTimeline :
       (waitValue = 0 => waitQueue = 0) /\ attemptSubmitShader(waitQueue, waitValue)
  \/ \E queue \in Queues : complete(queue)
  \/ collectCommand

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ deviceExists \in BOOLEAN
  /\ capable \in BOOLEAN
  /\ created \subseteq ObjectKinds
  /\ heapLogical \in BOOLEAN
  /\ heapPhysical \in BOOLEAN
  /\ textureLogical \in BOOLEAN
  /\ texturePhysical \in BOOLEAN
  /\ viewLogical \in BOOLEAN
  /\ viewPhysical \in BOOLEAN
  /\ bindingSetLogical \in BOOLEAN
  /\ bindingSetPhysical \in BOOLEAN
  /\ setCreationVersion \in 0..MaxVersion
  /\ mapped \in BOOLEAN
  /\ version \in 0..MaxVersion
  /\ publicationQueue \in Queues
  /\ publicationValue \in 0..MaxTimeline
  /\ mappingWaitQueue \in Queues
  /\ mappingWaitValue \in 0..MaxTimeline
  /\ commandPhase \in Phases
  /\ commandKind \in Kinds
  /\ commandVersion \in 0..MaxVersion
  /\ commandMapped \in BOOLEAN
  /\ commandQueue \in Queues
  /\ commandPublicationQueue \in Queues
  /\ commandPublicationValue \in 0..MaxTimeline
  /\ commandWaitQueue \in Queues
  /\ commandWaitValue \in 0..MaxTimeline
  /\ commandRetireValue \in 0..MaxTimeline
  /\ physicalValue \in 0..1
  /\ preStorePhysicalValue \in 0..1
  /\ sampledValue \in 0..1
  /\ sampleObserved \in BOOLEAN
  /\ storeExecuted \in BOOLEAN
  /\ executedMapped \in BOOLEAN
  /\ submitted \in [Queues -> 0..MaxTimeline]
  /\ completed \in [Queues -> 0..MaxTimeline]
  /\ lastOutcome \in Outcomes
  /\ \A q \in Queues : completed[q] <= submitted[q]

SparseObjectsAreCapabilityGated ==
  (heapPhysical \/ texturePhysical) => capable

VirtualObjectRetention ==
  /\ (heapLogical => heapPhysical)
  /\ (textureLogical => texturePhysical)
  /\ (viewLogical => viewPhysical)
  /\ (bindingSetLogical => bindingSetPhysical)
  /\ (bindingSetPhysical => viewPhysical)
  /\ (viewPhysical => texturePhysical)
  /\ (commandPhase # "idle" => bindingSetPhysical /\ texturePhysical)

DescriptorsDoNotFreezeGenerations ==
  /\ (bindingSetPhysical => setCreationVersion <= version)
  /\ (commandPhase # "idle" =>
        /\ commandKind \in ShaderKinds
        /\ commandVersion <= version
        /\ (commandVersion = 0 <=> commandPublicationValue = 0))

CrossQueueShaderUseIsOrdered ==
  /\ (version > 1 /\ completed[publicationQueue] >= publicationValue
      => completed[mappingWaitQueue] >= mappingWaitValue)
  /\ (commandPhase = "inFlight" =>
        /\ (commandQueue # commandPublicationQueue
            /\ ~PublicationComplete(commandPublicationQueue, commandPublicationValue)
            => commandWaitQueue = commandPublicationQueue
               /\ commandWaitValue >= commandPublicationValue)
        /\ (completed[commandQueue] >= commandRetireValue
            => PublicationComplete(commandPublicationQueue, commandPublicationValue)))

GpuExecutionSemantics ==
  /\ (sampleObserved =>
        sampledValue = (IF executedMapped THEN preStorePhysicalValue ELSE 0))
  /\ (storeExecuted /\ ~executedMapped =>
        physicalValue = preStorePhysicalValue)
  /\ (storeExecuted /\ executedMapped =>
        physicalValue = 1)

MappedOrCapturedTilesRetainPhysicalStorage ==
  (mapped \/ (commandPhase # "idle" /\ commandMapped)) => heapPhysical

Safety ==
  /\ SparseObjectsAreCapabilityGated
  /\ VirtualObjectRetention
  /\ DescriptorsDoNotFreezeGenerations
  /\ CrossQueueShaderUseIsOrdered
  /\ GpuExecutionSemantics
  /\ MappedOrCapturedTilesRetainPhysicalStorage

StepSafety ==
  [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
     /\ (physicalValue' # physicalValue =>
           commandPhase = "inFlight" /\ completed'[commandQueue] >= commandRetireValue)]_vars

=============================================================================
