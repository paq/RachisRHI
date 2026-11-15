------------------------- MODULE MeshShaderPipeline -------------------------
(***************************************************************************)
(* Mesh/task shader capability discovery, exclusive stage-family pipeline  *)
(* creation, direct and indirect mesh dispatch validation, failure         *)
(* atomicity, and module/buffer physical retention in                      *)
(* `src/D3D12/D3D12Device.cpp`, `src/D3D12/D3D12Pipeline.cpp`              *)
(* (stage-family validation and mesh command signatures),                  *)
(* `src/D3D12/D3D12CommandRecording.cpp` (`PipelineManager::dispatchMesh`  *)
(* and `dispatchMeshIndirect`), and `src/D3D12/D3D12Queue.cpp`             *)
(* (`include/Rachis/Device.h`).                                            *)
(*                                                                         *)
(* Models exclusive mesh-family pipeline validation (rejecting             *)
(* conventional or mixed pre-raster sources, wrong shader stages, or       *)
(* destroyed modules), direct workgroup-limit validation, indirect         *)
(* argument-buffer metadata validation, atomic rejection across all        *)
(* state variables (including indirect argument references), and exact     *)
(* physical lifetime retention through submission completion or command    *)
(* abandonment.                                                            *)
(*                                                                         *)
(* Abstracted: large numeric D3D12 workgroup constants (partitioned into   *)
(* boundary equivalence classes) and GPU-authored indirect counts.         *)
(***************************************************************************)
EXTENDS Naturals

SourceFamilies == {"conventional", "mesh", "mixed"}
GroupCountClasses == {"valid", "zero", "axisOverflow", "totalOverflow"}
ArgumentMetadata == {"valid", "invalidRangeOrStride", "staleOrWrongUsage"}
Operations == {
    "none", "create", "direct", "indirect", "destroyTask", "destroyMesh",
    "destroyArgument", "destroyPipeline", "submit", "complete", "abandon"
}
Outcomes == {"none", "accepted", "rejected"}
AbstractMaxTotalWorkgroups == 4
MaxDispatches == 1

VARIABLES
    nativeSupport,              \* adapter mesh/task shader capability
    reportedMeshSupport,        \* reported mesh-shader capability
    reportedTaskSupport,        \* reported task-shader capability
    nativeMaxTotalWorkgroups,   \* adapter total-workgroup limit
    reportedMaxTotalWorkgroups, \* reported total-workgroup limit
    sourceFamily,               \* pre-raster stage family in SourceFamilies
    taskPresent,                \* whether pipeline descriptor includes a task shader
    taskStageCorrect,           \* whether supplied task module has task stage
    meshStageCorrect,           \* whether supplied mesh module has mesh stage
    taskLogicalAlive,           \* public liveness of optional task module
    taskPhysicalAlive,          \* physical bytecode liveness of task module
    meshLogicalAlive,           \* public liveness of mesh module
    meshPhysicalAlive,          \* physical bytecode liveness of mesh module
    pipelineAlive,              \* public liveness of mesh graphics pipeline
    recordedReference,          \* TRUE while active recording retains pipeline modules
    inFlightReference,          \* TRUE while in-flight submission retains modules
    argumentLogicalAlive,       \* public liveness of indirect argument buffer
    argumentPhysicalAlive,      \* physical liveness of indirect argument buffer
    argumentRecordedReference,  \* TRUE while active recording retains argument buffer
    argumentInFlightReference,  \* TRUE while submission retains argument buffer
    dispatches,                 \* accepted mesh dispatches in active recording
    requestedStateComplete,     \* graphics-state readiness of latest dispatch attempt
    requestedGroupCountClass,   \* workgroup count class of latest direct attempt
    requestedArgumentMetadata,  \* metadata class of latest indirect attempt
    lastOperation,              \* latest operation tag in Operations
    lastOutcome                 \* outcome of latest operation in Outcomes

coreVars == <<
    nativeSupport, reportedMeshSupport, reportedTaskSupport,
    nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
    sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
    taskLogicalAlive, taskPhysicalAlive, meshLogicalAlive, meshPhysicalAlive,
    pipelineAlive, recordedReference, inFlightReference,
    argumentLogicalAlive, argumentPhysicalAlive,
    argumentRecordedReference, argumentInFlightReference, dispatches
>>

vars == <<
    nativeSupport, reportedMeshSupport, reportedTaskSupport,
    nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
    sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
    taskLogicalAlive, taskPhysicalAlive, meshLogicalAlive, meshPhysicalAlive,
    pipelineAlive, recordedReference, inFlightReference,
    argumentLogicalAlive, argumentPhysicalAlive,
    argumentRecordedReference, argumentInFlightReference, dispatches,
    requestedStateComplete, requestedGroupCountClass,
    requestedArgumentMetadata, lastOperation, lastOutcome
>>

CreationValid ==
    /\ ~pipelineAlive
    /\ nativeSupport
    /\ sourceFamily = "mesh"
    /\ meshLogicalAlive
    /\ meshStageCorrect
    /\ (~taskPresent \/ (taskLogicalAlive /\ taskStageCorrect))

DirectValid(stateComplete, groupCountClass) ==
    /\ pipelineAlive
    /\ ~inFlightReference
    /\ stateComplete
    /\ groupCountClass = "valid"

IndirectValid(stateComplete, metadata) ==
    /\ pipelineAlive
    /\ ~inFlightReference
    /\ stateComplete
    /\ argumentLogicalAlive
    /\ metadata = "valid"

ModuleRetainedByCommand ==
    recordedReference \/ inFlightReference

Init ==
    \E support \in BOOLEAN, family \in SourceFamilies,
       hasTask \in BOOLEAN, taskCorrect \in BOOLEAN, meshCorrect \in BOOLEAN :
        /\ nativeSupport = support
        /\ reportedMeshSupport = support
        /\ reportedTaskSupport = support
        /\ nativeMaxTotalWorkgroups = IF support THEN AbstractMaxTotalWorkgroups ELSE 0
        /\ reportedMaxTotalWorkgroups = IF support THEN AbstractMaxTotalWorkgroups ELSE 0
        /\ sourceFamily = family
        /\ taskPresent = hasTask
        /\ taskStageCorrect = taskCorrect
        /\ meshStageCorrect = meshCorrect
        /\ taskLogicalAlive = hasTask
        /\ taskPhysicalAlive = hasTask
        /\ meshLogicalAlive = TRUE
        /\ meshPhysicalAlive = TRUE
        /\ pipelineAlive = FALSE
        /\ recordedReference = FALSE
        /\ inFlightReference = FALSE
        /\ argumentLogicalAlive = TRUE
        /\ argumentPhysicalAlive = TRUE
        /\ argumentRecordedReference = FALSE
        /\ argumentInFlightReference = FALSE
        /\ dispatches = 0
        /\ requestedStateComplete = FALSE
        /\ requestedGroupCountClass = "valid"
        /\ requestedArgumentMetadata = "valid"
        /\ lastOperation = "none"
        /\ lastOutcome = "none"

attemptCreatePipeline ==
    LET accepted == CreationValid
    IN /\ IF accepted
          THEN /\ pipelineAlive' = TRUE
               /\ UNCHANGED <<
                    nativeSupport, reportedMeshSupport, reportedTaskSupport,
                    nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
                    sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
                    taskLogicalAlive, taskPhysicalAlive, meshLogicalAlive,
                    meshPhysicalAlive, recordedReference, inFlightReference,
                    argumentLogicalAlive, argumentPhysicalAlive,
                    argumentRecordedReference, argumentInFlightReference,
                    dispatches
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "create"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
       /\ UNCHANGED <<
            requestedStateComplete, requestedGroupCountClass,
            requestedArgumentMetadata
          >>

attemptDirect(stateComplete, groupCountClass) ==
    LET accepted == DirectValid(stateComplete, groupCountClass)
    IN /\ dispatches < MaxDispatches
       /\ ~inFlightReference
       /\ IF accepted
          THEN /\ recordedReference' = TRUE
               /\ dispatches' = dispatches + 1
               /\ UNCHANGED <<
                    nativeSupport, reportedMeshSupport, reportedTaskSupport,
                    nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
                    sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
                    taskLogicalAlive, taskPhysicalAlive, meshLogicalAlive,
                    meshPhysicalAlive, pipelineAlive, inFlightReference,
                    argumentLogicalAlive, argumentPhysicalAlive,
                    argumentRecordedReference, argumentInFlightReference
                  >>
          ELSE UNCHANGED coreVars
       /\ requestedStateComplete' = stateComplete
       /\ requestedGroupCountClass' = groupCountClass
       /\ lastOperation' = "direct"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
       /\ UNCHANGED requestedArgumentMetadata

attemptIndirect(stateComplete, metadata) ==
    LET accepted == IndirectValid(stateComplete, metadata)
    IN /\ dispatches < MaxDispatches
       /\ ~inFlightReference
       /\ IF accepted
          THEN /\ recordedReference' = TRUE
               /\ argumentRecordedReference' = TRUE
               /\ dispatches' = dispatches + 1
               /\ UNCHANGED <<
                    nativeSupport, reportedMeshSupport, reportedTaskSupport,
                    nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
                    sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
                    taskLogicalAlive, taskPhysicalAlive, meshLogicalAlive,
                    meshPhysicalAlive, pipelineAlive, inFlightReference,
                    argumentLogicalAlive, argumentPhysicalAlive,
                    argumentInFlightReference
                  >>
          ELSE UNCHANGED coreVars
       /\ requestedStateComplete' = stateComplete
       /\ requestedArgumentMetadata' = metadata
       /\ lastOperation' = "indirect"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
       /\ UNCHANGED requestedGroupCountClass

destroyTask ==
    /\ taskLogicalAlive
    /\ taskLogicalAlive' = FALSE
    /\ taskPhysicalAlive' =
        (taskPresent /\ (pipelineAlive \/ ModuleRetainedByCommand))
    /\ lastOperation' = "destroyTask"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedMeshSupport, reportedTaskSupport,
        nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
        sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
        meshLogicalAlive, meshPhysicalAlive, pipelineAlive,
        recordedReference, inFlightReference, argumentLogicalAlive,
        argumentPhysicalAlive, argumentRecordedReference,
        argumentInFlightReference, dispatches, requestedStateComplete,
        requestedGroupCountClass, requestedArgumentMetadata
       >>

destroyMesh ==
    /\ meshLogicalAlive
    /\ meshLogicalAlive' = FALSE
    /\ meshPhysicalAlive' = (pipelineAlive \/ ModuleRetainedByCommand)
    /\ lastOperation' = "destroyMesh"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedMeshSupport, reportedTaskSupport,
        nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
        sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
        taskLogicalAlive, taskPhysicalAlive, pipelineAlive,
        recordedReference, inFlightReference, argumentLogicalAlive,
        argumentPhysicalAlive, argumentRecordedReference,
        argumentInFlightReference, dispatches, requestedStateComplete,
        requestedGroupCountClass, requestedArgumentMetadata
       >>

destroyArgument ==
    /\ argumentLogicalAlive
    /\ argumentLogicalAlive' = FALSE
    /\ argumentPhysicalAlive' =
        (argumentRecordedReference \/ argumentInFlightReference)
    /\ lastOperation' = "destroyArgument"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedMeshSupport, reportedTaskSupport,
        nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
        sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
        taskLogicalAlive, taskPhysicalAlive, meshLogicalAlive,
        meshPhysicalAlive, pipelineAlive, recordedReference,
        inFlightReference, argumentRecordedReference,
        argumentInFlightReference, dispatches, requestedStateComplete,
        requestedGroupCountClass, requestedArgumentMetadata
       >>

destroyPipeline ==
    /\ pipelineAlive
    /\ pipelineAlive' = FALSE
    /\ taskPhysicalAlive' =
        (taskLogicalAlive \/ (taskPresent /\ ModuleRetainedByCommand))
    /\ meshPhysicalAlive' = (meshLogicalAlive \/ ModuleRetainedByCommand)
    /\ lastOperation' = "destroyPipeline"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedMeshSupport, reportedTaskSupport,
        nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
        sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
        taskLogicalAlive, meshLogicalAlive, recordedReference,
        inFlightReference, argumentLogicalAlive, argumentPhysicalAlive,
        argumentRecordedReference, argumentInFlightReference, dispatches,
        requestedStateComplete, requestedGroupCountClass,
        requestedArgumentMetadata
       >>

submit ==
    /\ recordedReference
    /\ ~inFlightReference
    /\ recordedReference' = FALSE
    /\ inFlightReference' = TRUE
    /\ argumentRecordedReference' = FALSE
    /\ argumentInFlightReference' = argumentRecordedReference
    /\ dispatches' = 0
    /\ lastOperation' = "submit"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedMeshSupport, reportedTaskSupport,
        nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
        sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
        taskLogicalAlive, taskPhysicalAlive, meshLogicalAlive,
        meshPhysicalAlive, pipelineAlive, argumentLogicalAlive,
        argumentPhysicalAlive, requestedStateComplete,
        requestedGroupCountClass, requestedArgumentMetadata
       >>

complete ==
    /\ inFlightReference
    /\ inFlightReference' = FALSE
    /\ argumentInFlightReference' = FALSE
    /\ taskPhysicalAlive' =
        (taskLogicalAlive \/ (taskPresent /\ (pipelineAlive \/ recordedReference)))
    /\ meshPhysicalAlive' = (meshLogicalAlive \/ pipelineAlive \/ recordedReference)
    /\ argumentPhysicalAlive' = (argumentLogicalAlive \/ argumentRecordedReference)
    /\ lastOperation' = "complete"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedMeshSupport, reportedTaskSupport,
        nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
        sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
        taskLogicalAlive, meshLogicalAlive, pipelineAlive,
        recordedReference, argumentLogicalAlive, argumentRecordedReference,
        dispatches, requestedStateComplete, requestedGroupCountClass,
        requestedArgumentMetadata
       >>

abandon ==
    /\ recordedReference
    /\ ~inFlightReference
    /\ recordedReference' = FALSE
    /\ argumentRecordedReference' = FALSE
    /\ dispatches' = 0
    /\ taskPhysicalAlive' =
        (taskLogicalAlive \/ (taskPresent /\ (pipelineAlive \/ inFlightReference)))
    /\ meshPhysicalAlive' = (meshLogicalAlive \/ pipelineAlive \/ inFlightReference)
    /\ argumentPhysicalAlive' = (argumentLogicalAlive \/ argumentInFlightReference)
    /\ lastOperation' = "abandon"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedMeshSupport, reportedTaskSupport,
        nativeMaxTotalWorkgroups, reportedMaxTotalWorkgroups,
        sourceFamily, taskPresent, taskStageCorrect, meshStageCorrect,
        taskLogicalAlive, meshLogicalAlive, pipelineAlive,
        inFlightReference, argumentLogicalAlive, argumentInFlightReference,
        requestedStateComplete, requestedGroupCountClass,
        requestedArgumentMetadata
       >>

Next ==
    \/ attemptCreatePipeline
    \/ \E sc \in BOOLEAN, gc \in GroupCountClasses : attemptDirect(sc, gc)
    \/ \E sc \in BOOLEAN, md \in ArgumentMetadata : attemptIndirect(sc, md)
    \/ destroyTask
    \/ destroyMesh
    \/ destroyArgument
    \/ destroyPipeline
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ nativeSupport \in BOOLEAN
    /\ reportedMeshSupport \in BOOLEAN
    /\ reportedTaskSupport \in BOOLEAN
    /\ nativeMaxTotalWorkgroups \in {0, AbstractMaxTotalWorkgroups}
    /\ reportedMaxTotalWorkgroups \in {0, AbstractMaxTotalWorkgroups}
    /\ sourceFamily \in SourceFamilies
    /\ taskPresent \in BOOLEAN
    /\ taskStageCorrect \in BOOLEAN
    /\ meshStageCorrect \in BOOLEAN
    /\ taskLogicalAlive \in BOOLEAN
    /\ taskPhysicalAlive \in BOOLEAN
    /\ meshLogicalAlive \in BOOLEAN
    /\ meshPhysicalAlive \in BOOLEAN
    /\ pipelineAlive \in BOOLEAN
    /\ recordedReference \in BOOLEAN
    /\ inFlightReference \in BOOLEAN
    /\ argumentLogicalAlive \in BOOLEAN
    /\ argumentPhysicalAlive \in BOOLEAN
    /\ argumentRecordedReference \in BOOLEAN
    /\ argumentInFlightReference \in BOOLEAN
    /\ dispatches \in 0..MaxDispatches
    /\ requestedStateComplete \in BOOLEAN
    /\ requestedGroupCountClass \in GroupCountClasses
    /\ requestedArgumentMetadata \in ArgumentMetadata
    /\ lastOperation \in Operations
    /\ lastOutcome \in Outcomes

CapabilityMatchesNative ==
    /\ reportedMeshSupport = nativeSupport
    /\ reportedTaskSupport = nativeSupport
    /\ reportedMaxTotalWorkgroups = nativeMaxTotalWorkgroups
    /\ nativeSupport = (nativeMaxTotalWorkgroups = AbstractMaxTotalWorkgroups)

PipelineRequiresExclusiveMeshStages ==
    (pipelineAlive \/ recordedReference \/ inFlightReference) =>
        /\ nativeSupport
        /\ sourceFamily = "mesh"
        /\ meshStageCorrect
        /\ (taskPresent => taskStageCorrect)

TaskPhysicalLifetimeExact ==
    taskPhysicalAlive =
        (taskLogicalAlive
         \/ (taskPresent /\ (pipelineAlive \/ recordedReference \/ inFlightReference)))

MeshPhysicalLifetimeExact ==
    meshPhysicalAlive =
        (meshLogicalAlive \/ pipelineAlive \/ recordedReference \/ inFlightReference)

ArgumentPhysicalLifetimeExact ==
    argumentPhysicalAlive =
        (argumentLogicalAlive \/ argumentRecordedReference \/ argumentInFlightReference)

Safety ==
    /\ CapabilityMatchesNative
    /\ PipelineRequiresExclusiveMeshStages
    /\ TaskPhysicalLifetimeExact
    /\ MeshPhysicalLifetimeExact
    /\ ArgumentPhysicalLifetimeExact

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
        /\ (lastOperation' = "create" =>
                ((lastOutcome' = "accepted") = CreationValid))
        /\ (lastOperation' = "direct" =>
                ((lastOutcome' = "accepted") =
                    DirectValid(requestedStateComplete', requestedGroupCountClass')))
        /\ (lastOperation' = "indirect" =>
                ((lastOutcome' = "accepted") =
                    IndirectValid(requestedStateComplete', requestedArgumentMetadata')))
    ]_vars

=============================================================================
