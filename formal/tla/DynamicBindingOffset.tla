------------------------ MODULE DynamicBindingOffset ------------------------
(***************************************************************************)
(* Dynamic-offset consumption order, alignment and fixed-range validation, *)
(* failure atomicity, and transitive set/buffer lifetime retention in      *)
(* `src/D3D12/D3D12Binding.cpp` (exact offset count, per-bind offset       *)
(* application, and buffer-range validation),                              *)
(* `src/D3D12/D3D12CommandRecording.cpp`                                   *)
(* (`PipelineManager::bindComputeBindingSets` and                          *)
(* `bindGraphicsBindingSets` batch preparation and retention),             *)
(* `src/D3D12/D3D12Pipeline.cpp` (pipeline-layout dynamic-buffer counts    *)
(* and promoted root descriptors), and `src/D3D12/D3D12Queue.cpp`          *)
(* (`include/Rachis/Device.h`).                                            *)
(*                                                                         *)
(* Models one immutable binding set containing binding 0 (a 256-byte       *)
(* aligned uniform buffer backed by buffer 0) followed by binding 1 (a     *)
(* two-element 4-byte aligned storage-buffer array backed by buffers 0 and *)
(* 1). Dynamic offsets are consumed in ascending `(binding, element)`      *)
(* order; invalid counts, misalignments, or out-of-range offsets reject    *)
(* atomically, while accepted binds retain the binding set and its backing *)
(* buffers across logical destruction through submission completion or     *)
(* command abandonment.                                                    *)
(*                                                                         *)
(* Abstracted: root-signature GPU virtual addresses, shader execution, and *)
(* multi-pool descriptor heap suballocation.                               *)
(***************************************************************************)
EXTENDS Naturals

Buffers == {0, 1}
StorageElements == {0, 1}
OffsetCandidates == {0, 2, 4, 256, 512}
CountCandidates == {0, 2, 3, 4}
DynamicCount == 3
LeaseCapacity == 2
BaseOffset == 0
BindingRangeSize == 256
BufferSize == 512
UniformAlignment == 256
StorageAlignment == 4
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    logicalSetAlive,        \* TRUE while the public BindingSetHandle is live
    physicalSetAlive,       \* TRUE while the BindingSetRecord remains allocated
    logicalBufferAlive,     \* per-buffer public handle liveness in [Buffers -> BOOLEAN]
    physicalBufferAlive,    \* per-buffer native resource liveness in [Buffers -> BOOLEAN]
    recordingLeases,        \* number of active recording leases on the binding set
    inFlightLeases,         \* number of submitted in-flight leases on the binding set
    bound,                  \* TRUE while the set is bound in the active recording
    boundUniformOffset,     \* bound dynamic offset for binding 0 (uniform)
    boundStorageOffsets,    \* bound dynamic offsets for binding 1 elements {0, 1}
    lastOutcome             \* outcome of the latest step in Outcomes

coreVars == <<
    logicalSetAlive, physicalSetAlive, logicalBufferAlive, physicalBufferAlive,
    recordingLeases, inFlightLeases, bound, boundUniformOffset, boundStorageOffsets
>>

vars == <<
    logicalSetAlive, physicalSetAlive, logicalBufferAlive, physicalBufferAlive,
    recordingLeases, inFlightLeases, bound, boundUniformOffset, boundStorageOffsets,
    lastOutcome
>>

RangeFits(dynamicOffset) ==
    BaseOffset + dynamicOffset + BindingRangeSize <= BufferSize

\* Offsets are consumed in ascending (binding, arrayElement) order:
\* offset 0 -> binding 0 (uniform), offsets 1..2 -> binding 1 elements 0..1 (storage).
BindRequestValid(providedCount, uniformOffset, storageOffset0, storageOffset1) ==
    /\ logicalSetAlive
    /\ inFlightLeases = 0
    /\ recordingLeases < LeaseCapacity
    /\ providedCount = DynamicCount
    /\ uniformOffset % UniformAlignment = 0
    /\ storageOffset0 % StorageAlignment = 0
    /\ storageOffset1 % StorageAlignment = 0
    /\ RangeFits(uniformOffset)
    /\ RangeFits(storageOffset0)
    /\ RangeFits(storageOffset1)

Init ==
    /\ logicalSetAlive = TRUE
    /\ physicalSetAlive = TRUE
    /\ logicalBufferAlive = [b \in Buffers |-> TRUE]
    /\ physicalBufferAlive = [b \in Buffers |-> TRUE]
    /\ recordingLeases = 0
    /\ inFlightLeases = 0
    /\ bound = FALSE
    /\ boundUniformOffset = 0
    /\ boundStorageOffsets = [i \in StorageElements |-> 0]
    /\ lastOutcome = "none"

attemptBind(providedCount, uniformOffset, storageOffset0, storageOffset1) ==
    LET accepted == BindRequestValid(providedCount, uniformOffset, storageOffset0, storageOffset1)
    IN /\ IF accepted
          THEN /\ recordingLeases' = recordingLeases + 1
               /\ bound' = TRUE
               /\ boundUniformOffset' = uniformOffset
               /\ boundStorageOffsets' =
                    [i \in StorageElements |-> IF i = 0 THEN storageOffset0 ELSE storageOffset1]
               /\ UNCHANGED <<
                    logicalSetAlive, physicalSetAlive,
                    logicalBufferAlive, physicalBufferAlive, inFlightLeases
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroySet ==
    LET retained == recordingLeases > 0 \/ inFlightLeases > 0
    IN /\ logicalSetAlive
       /\ logicalSetAlive' = FALSE
       /\ physicalSetAlive' = retained
       /\ physicalBufferAlive' =
            [b \in Buffers |-> logicalBufferAlive[b] \/ retained]
       /\ lastOutcome' = "none"
       /\ UNCHANGED <<
            logicalBufferAlive, recordingLeases, inFlightLeases,
            bound, boundUniformOffset, boundStorageOffsets
          >>

destroyBuffer(buffer) ==
    /\ logicalBufferAlive[buffer]
    /\ logicalBufferAlive' = [logicalBufferAlive EXCEPT ![buffer] = FALSE]
    /\ physicalBufferAlive' = [physicalBufferAlive EXCEPT ![buffer] = physicalSetAlive]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalSetAlive, physicalSetAlive, recordingLeases, inFlightLeases,
        bound, boundUniformOffset, boundStorageOffsets
       >>

submit ==
    /\ recordingLeases > 0
    /\ inFlightLeases = 0
    /\ recordingLeases' = 0
    /\ inFlightLeases' = recordingLeases
    /\ bound' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalSetAlive, physicalSetAlive, logicalBufferAlive,
        physicalBufferAlive, boundUniformOffset, boundStorageOffsets
       >>

complete ==
    /\ inFlightLeases > 0
    /\ physicalSetAlive' = logicalSetAlive
    /\ physicalBufferAlive' =
        [b \in Buffers |-> logicalBufferAlive[b] \/ logicalSetAlive]
    /\ inFlightLeases' = 0
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalSetAlive, logicalBufferAlive, recordingLeases,
        bound, boundUniformOffset, boundStorageOffsets
       >>

abandon ==
    /\ recordingLeases > 0
    /\ inFlightLeases = 0
    /\ physicalSetAlive' = logicalSetAlive
    /\ physicalBufferAlive' =
        [b \in Buffers |-> logicalBufferAlive[b] \/ logicalSetAlive]
    /\ recordingLeases' = 0
    /\ bound' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalSetAlive, logicalBufferAlive, inFlightLeases,
        boundUniformOffset, boundStorageOffsets
       >>

Next ==
    \/ \E cnt \in CountCandidates,
          uOff \in OffsetCandidates,
          sOff0 \in OffsetCandidates,
          sOff1 \in OffsetCandidates :
            attemptBind(cnt, uOff, sOff0, sOff1)
    \/ destroySet
    \/ \E b \in Buffers : destroyBuffer(b)
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ logicalSetAlive \in BOOLEAN
    /\ physicalSetAlive \in BOOLEAN
    /\ logicalBufferAlive \in [Buffers -> BOOLEAN]
    /\ physicalBufferAlive \in [Buffers -> BOOLEAN]
    /\ recordingLeases \in 0..LeaseCapacity
    /\ inFlightLeases \in 0..LeaseCapacity
    /\ bound \in BOOLEAN
    /\ boundUniformOffset \in OffsetCandidates
    /\ boundStorageOffsets \in [StorageElements -> OffsetCandidates]
    /\ lastOutcome \in Outcomes

BoundOffsetsValid ==
    bound =>
        /\ recordingLeases > 0
        /\ boundUniformOffset % UniformAlignment = 0
        /\ RangeFits(boundUniformOffset)
        /\ \A i \in StorageElements :
            /\ boundStorageOffsets[i] % StorageAlignment = 0
            /\ RangeFits(boundStorageOffsets[i])

SetPhysicalLifetimeExact ==
    physicalSetAlive = (logicalSetAlive \/ recordingLeases > 0 \/ inFlightLeases > 0)

BufferPhysicalLifetimeExact ==
    \A b \in Buffers :
        physicalBufferAlive[b] = (logicalBufferAlive[b] \/ physicalSetAlive)

LeasesSeparated ==
    inFlightLeases > 0 => recordingLeases = 0 /\ ~bound

Safety ==
    /\ BoundOffsetsValid
    /\ SetPhysicalLifetimeExact
    /\ BufferPhysicalLifetimeExact
    /\ LeasesSeparated

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
        /\ (lastOutcome' = "accepted" =>
                /\ logicalSetAlive
                /\ inFlightLeases = 0
                /\ recordingLeases < LeaseCapacity
                /\ recordingLeases' = recordingLeases + 1
                /\ bound')
    ]_vars

=============================================================================
