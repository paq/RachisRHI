----------------- MODULE AccelerationStructureSerialization -----------------
(***************************************************************************)
(* Acceleration-structure serialization-info query protocol, opaque export *)
(* copy validation, host readback ordering, and endpoint retention.        *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Query.cpp (QueryManager::                            *)
(*     writeAccelerationStructureSerializationInfo,                        *)
(*     resolveAccelerationStructureSerializationInfo)                      *)
(*   - src/D3D12/D3D12AccelerationStructure.cpp (ResourceManager::         *)
(*     serializeAccelerationStructure,                                     *)
(*     accelerationStructureSerializationSource)                           *)
(*   - src/D3D12/D3D12Queue.cpp (command and timeline retention)           *)
(*                                                                         *)
(* Caller vs. RHI boundary note:                                           *)
(*   - Postbuild serialization-info query completion (queryPhase = 4) and  *)
(*     supplying the exact GPU-emitted serialized byte size before         *)
(*     recording serializeAccelerationStructure are caller GPU-ordering    *)
(*     obligations. On the CPU, D3D12AccelerationStructure.cpp validates   *)
(*     serializedSize >= sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_  *)
(*     STRUCTURE_HEADER), 256-byte destinationOffset alignment, live built *)
(*     source AS, and an in-bounds device-local destination buffer with    *)
(*     BufferUsage::storage and BufferUsage::copySource.                   *)
(*   - The buffer barrier and host readback copy are ordinary              *)
(*     caller-recorded commands that retain the export buffer like any     *)
(*     other copy.                                                         *)
(*   - Design rule RetainEndpointsInFlight (TRUE in the design) retains    *)
(*     source and destination physical allocations across recorded and     *)
(*     in-flight serialization and readback commands after handle destroy. *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Native D3D12 CopyRaytracingAccelerationStructure SERIALIZE bytes.   *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RetainEndpointsInFlight

ASSUME RetainEndpointsInFlight \in BOOLEAN

QueryPhases == 0..4
WorkPhases == 0..3
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    sourceLogical,          \* public handle of built source AS is alive
    sourcePhysical,         \* physical backing of source AS is retained
    destinationLogical,     \* public handle of destination export buffer is alive
    destinationPhysical,    \* physical backing of destination export buffer is retained
    queryPhase,             \* 0:idle, 1:written, 2:resolved, 3:submitted, 4:completed
    serializationPhase,     \* 0:idle, 1:recorded, 2:submitted, 3:completed
    readbackPhase,          \* 0:idle, 1:recorded, 2:submitted, 3:completed
    opaqueBlobProduced,     \* GPU export copy has completed into destination buffer
    hostBlobAvailable,      \* GPU->host readback copy has completed
    lastOutcome             \* outcome of the most recent serialize attempt

coreVars == <<sourceLogical, sourcePhysical, destinationLogical,
              destinationPhysical, queryPhase, serializationPhase,
              readbackPhase, opaqueBlobProduced, hostBlobAvailable>>
vars == <<sourceLogical, sourcePhysical, destinationLogical,
          destinationPhysical, queryPhase, serializationPhase,
          readbackPhase, opaqueBlobProduced, hostBlobAvailable,
          lastOutcome>>

TypeOK ==
    /\ sourceLogical \in BOOLEAN
    /\ sourcePhysical \in BOOLEAN
    /\ destinationLogical \in BOOLEAN
    /\ destinationPhysical \in BOOLEAN
    /\ queryPhase \in QueryPhases
    /\ serializationPhase \in WorkPhases
    /\ readbackPhase \in WorkPhases
    /\ opaqueBlobProduced \in BOOLEAN
    /\ hostBlobAvailable \in BOOLEAN
    /\ lastOutcome \in Outcomes

Init ==
    /\ sourceLogical = TRUE
    /\ sourcePhysical = TRUE
    /\ destinationLogical = TRUE
    /\ destinationPhysical = TRUE
    /\ queryPhase = 0
    /\ serializationPhase = 0
    /\ readbackPhase = 0
    /\ opaqueBlobProduced = FALSE
    /\ hostBlobAvailable = FALSE
    /\ lastOutcome = "none"

WriteInfoQuery ==
    /\ sourceLogical /\ sourcePhysical /\ queryPhase = 0
    /\ queryPhase' = 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationLogical,
                   destinationPhysical, serializationPhase, readbackPhase,
                   opaqueBlobProduced, hostBlobAvailable>>

ResolveInfoQuery ==
    /\ queryPhase = 1
    /\ queryPhase' = 2
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationLogical,
                   destinationPhysical, serializationPhase, readbackPhase,
                   opaqueBlobProduced, hostBlobAvailable>>

SubmitInfoQuery ==
    /\ queryPhase = 2
    /\ queryPhase' = 3
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationLogical,
                   destinationPhysical, serializationPhase, readbackPhase,
                   opaqueBlobProduced, hostBlobAvailable>>

CompleteInfoQuery ==
    /\ queryPhase = 3
    /\ queryPhase' = 4
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationLogical,
                   destinationPhysical, serializationPhase, readbackPhase,
                   opaqueBlobProduced, hostBlobAvailable>>

SerializeValid(deviceLocalStorage, copySourceUsage, aligned256, validHeaderAndBounds) ==
    /\ sourceLogical
    /\ sourcePhysical
    /\ destinationLogical
    /\ destinationPhysical
    /\ queryPhase = 4
    /\ serializationPhase = 0
    /\ deviceLocalStorage
    /\ copySourceUsage
    /\ aligned256
    /\ validHeaderAndBounds

AttemptSerialize(deviceLocalStorage, copySourceUsage, aligned256, validHeaderAndBounds) ==
    IF SerializeValid(deviceLocalStorage, copySourceUsage, aligned256, validHeaderAndBounds)
    THEN /\ serializationPhase' = 1
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationLogical,
                        destinationPhysical, queryPhase, readbackPhase,
                        opaqueBlobProduced, hostBlobAvailable>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

SubmitSerialize ==
    /\ serializationPhase = 1
    /\ serializationPhase' = 2
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationLogical,
                   destinationPhysical, queryPhase, readbackPhase,
                   opaqueBlobProduced, hostBlobAvailable>>

CompleteSerialize ==
    /\ serializationPhase = 2
    /\ serializationPhase' = 3
    /\ opaqueBlobProduced' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationLogical,
                   destinationPhysical, queryPhase, readbackPhase,
                   hostBlobAvailable>>

RecordReadback ==
    /\ serializationPhase = 3
    /\ opaqueBlobProduced
    /\ destinationLogical /\ destinationPhysical
    /\ readbackPhase = 0
    /\ readbackPhase' = 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationLogical,
                   destinationPhysical, queryPhase, serializationPhase,
                   opaqueBlobProduced, hostBlobAvailable>>

SubmitReadback ==
    /\ readbackPhase = 1
    /\ readbackPhase' = 2
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationLogical,
                   destinationPhysical, queryPhase, serializationPhase,
                   opaqueBlobProduced, hostBlobAvailable>>

CompleteReadback ==
    /\ readbackPhase = 2
    /\ readbackPhase' = 3
    /\ hostBlobAvailable' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationLogical,
                   destinationPhysical, queryPhase, serializationPhase,
                   opaqueBlobProduced>>

DestroySource ==
    /\ sourceLogical
    /\ sourceLogical' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourcePhysical, destinationLogical, destinationPhysical,
                   queryPhase, serializationPhase, readbackPhase,
                   opaqueBlobProduced, hostBlobAvailable>>

DestroyDestination ==
    /\ destinationLogical
    /\ destinationLogical' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<sourceLogical, sourcePhysical, destinationPhysical,
                   queryPhase, serializationPhase, readbackPhase,
                   opaqueBlobProduced, hostBlobAvailable>>

Collect ==
    LET keepSource ==
            \/ sourceLogical
            \/ queryPhase \in {1, 2, 3}
            \/ (RetainEndpointsInFlight /\ serializationPhase \in {1, 2})
        keepDestination ==
            \/ destinationLogical
            \/ (RetainEndpointsInFlight /\ serializationPhase \in {1, 2})
            \/ (RetainEndpointsInFlight /\ readbackPhase \in {1, 2})
    IN /\ sourcePhysical' = keepSource
       /\ destinationPhysical' = keepDestination
       /\ lastOutcome' = "none"
       /\ UNCHANGED <<sourceLogical, destinationLogical, queryPhase,
                      serializationPhase, readbackPhase, opaqueBlobProduced,
                      hostBlobAvailable>>

Next ==
    \/ WriteInfoQuery
    \/ ResolveInfoQuery
    \/ SubmitInfoQuery
    \/ CompleteInfoQuery
    \/ \E storageOk, copySrcOk, alignOk, sizeOk \in BOOLEAN :
          AttemptSerialize(storageOk, copySrcOk, alignOk, sizeOk)
    \/ SubmitSerialize
    \/ CompleteSerialize
    \/ RecordReadback
    \/ SubmitReadback
    \/ CompleteReadback
    \/ DestroySource
    \/ DestroyDestination
    \/ Collect

Spec == Init /\ [][Next]_vars

QueryRetainsSource ==
    queryPhase \in {1, 2, 3} => sourcePhysical

SerializationRetainsEndpoints ==
    serializationPhase \in {1, 2} => sourcePhysical /\ destinationPhysical

ReadbackRetainsDestination ==
    readbackPhase \in {1, 2} => destinationPhysical

HostArchiveRequiresProducedBlob ==
    /\ (opaqueBlobProduced <=> serializationPhase = 3)
    /\ (hostBlobAvailable => opaqueBlobProduced /\ readbackPhase = 3)

SerializationRequiresCompletedQuery ==
    serializationPhase > 0 => queryPhase = 4

Safety ==
    /\ QueryRetainsSource
    /\ SerializationRetainsEndpoints
    /\ ReadbackRetainsDestination
    /\ HostArchiveRequiresProducedBlob
    /\ SerializationRequiresCompletedQuery

StepSafety ==
    [][/\ (serializationPhase = 0 /\ serializationPhase' = 1) =>
             (sourceLogical /\ sourcePhysical /\ destinationLogical
              /\ destinationPhysical /\ queryPhase = 4)
       /\ (lastOutcome' = "rejected" /\ coreVars' # coreVars) => FALSE]_vars
=============================================================================
