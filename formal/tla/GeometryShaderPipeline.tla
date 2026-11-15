----------------------- MODULE GeometryShaderPipeline -----------------------
(***************************************************************************)
(* Geometry-shader capability gating, graphics-pipeline creation, point    *)
(* primitive amplification, and shader-module bytecode retention in        *)
(* `src/D3D12/D3D12Device.cpp`, `src/D3D12/D3D12Pipeline.cpp`,             *)
(* `src/D3D12/D3D12CommandRecording.cpp` (pipeline binding and draw        *)
(* recording), and `src/D3D12/D3D12Queue.cpp` (`include/Rachis/Device.h`). *)
(*                                                                         *)
(* Models capability reporting (`reportedSupport = nativeSupport`),        *)
(* fallible geometry-pipeline creation over shader modules of varying      *)
(* stages, point-to-triangle vertex amplification during draw recording,   *)
(* command submission, completion, and abandonment, and logical vs         *)
(* physical shader-module lifetime across pipeline and command owners.     *)
(*                                                                         *)
(* Abstracted: HLSL/DXIL bytecode contents, rasterizer state, and GPU      *)
(* execution internals.                                                    *)
(***************************************************************************)
EXTENDS Naturals

Stages == {"vertex", "pixel", "geometry"}
PointCounts == {1, 2}
Operations == {
    "none", "create", "destroyModule", "draw",
    "destroyPipeline", "submit", "complete", "abandon"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    nativeSupport,              \* TRUE when the adapter supports geometry shaders
    reportedSupport,            \* capability bit surfaced by DeviceCapabilities
    moduleStage,                \* shader stage of the candidate ShaderModuleHandle
    moduleLogicalAlive,         \* TRUE while the public module handle is live
    modulePhysicalAlive,        \* TRUE while native shader bytecode remains retained
    pipelineAlive,              \* TRUE while the geometry graphics pipeline is live
    recordedReference,          \* TRUE while an unsubmitted command retains bytecode
    inFlightReference,          \* TRUE while an in-flight submission retains bytecode
    generatedVertexCount,       \* vertices amplified by the active recorded draw
    lastOperation,              \* latest operation tag in Operations
    lastOutcome                 \* outcome of the latest operation in Outcomes

coreVars == <<
    nativeSupport, reportedSupport, moduleStage,
    moduleLogicalAlive, modulePhysicalAlive, pipelineAlive,
    recordedReference, inFlightReference, generatedVertexCount
>>

vars == <<
    nativeSupport, reportedSupport, moduleStage,
    moduleLogicalAlive, modulePhysicalAlive, pipelineAlive,
    recordedReference, inFlightReference, generatedVertexCount,
    lastOperation, lastOutcome
>>

CreateValid ==
    /\ ~pipelineAlive
    /\ reportedSupport
    /\ moduleLogicalAlive
    /\ moduleStage = "geometry"

Init ==
    \E support \in BOOLEAN, stage \in Stages :
        /\ nativeSupport = support
        /\ reportedSupport = support
        /\ moduleStage = stage
        /\ moduleLogicalAlive = TRUE
        /\ modulePhysicalAlive = TRUE
        /\ pipelineAlive = FALSE
        /\ recordedReference = FALSE
        /\ inFlightReference = FALSE
        /\ generatedVertexCount = 0
        /\ lastOperation = "none"
        /\ lastOutcome = "none"

attemptCreatePipeline ==
    LET accepted == CreateValid
    IN /\ IF accepted
          THEN /\ pipelineAlive' = TRUE
               /\ UNCHANGED <<
                    nativeSupport, reportedSupport, moduleStage,
                    moduleLogicalAlive, modulePhysicalAlive,
                    recordedReference, inFlightReference, generatedVertexCount
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "create"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroyModule ==
    /\ moduleLogicalAlive
    /\ moduleLogicalAlive' = FALSE
    /\ modulePhysicalAlive' = (pipelineAlive \/ recordedReference \/ inFlightReference)
    /\ lastOperation' = "destroyModule"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedSupport, moduleStage, pipelineAlive,
        recordedReference, inFlightReference, generatedVertexCount
       >>

draw(points) ==
    /\ pipelineAlive
    /\ modulePhysicalAlive
    /\ ~recordedReference
    /\ ~inFlightReference
    /\ recordedReference' = TRUE
    /\ generatedVertexCount' = 3 * points
    /\ lastOperation' = "draw"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedSupport, moduleStage,
        moduleLogicalAlive, modulePhysicalAlive, pipelineAlive, inFlightReference
       >>

destroyPipeline ==
    /\ pipelineAlive
    /\ pipelineAlive' = FALSE
    /\ modulePhysicalAlive' = (moduleLogicalAlive \/ recordedReference \/ inFlightReference)
    /\ lastOperation' = "destroyPipeline"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedSupport, moduleStage,
        moduleLogicalAlive, recordedReference, inFlightReference, generatedVertexCount
       >>

submit ==
    /\ recordedReference
    /\ ~inFlightReference
    /\ modulePhysicalAlive' = TRUE
    /\ recordedReference' = FALSE
    /\ inFlightReference' = TRUE
    /\ lastOperation' = "submit"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedSupport, moduleStage,
        moduleLogicalAlive, pipelineAlive, generatedVertexCount
       >>

complete ==
    /\ inFlightReference
    /\ modulePhysicalAlive' = (moduleLogicalAlive \/ pipelineAlive \/ recordedReference)
    /\ inFlightReference' = FALSE
    /\ generatedVertexCount' = 0
    /\ lastOperation' = "complete"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedSupport, moduleStage,
        moduleLogicalAlive, pipelineAlive, recordedReference
       >>

abandon ==
    /\ recordedReference
    /\ modulePhysicalAlive' = (moduleLogicalAlive \/ pipelineAlive \/ inFlightReference)
    /\ recordedReference' = FALSE
    /\ generatedVertexCount' = 0
    /\ lastOperation' = "abandon"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupport, reportedSupport, moduleStage,
        moduleLogicalAlive, pipelineAlive, inFlightReference
       >>

Next ==
    \/ attemptCreatePipeline
    \/ destroyModule
    \/ \E pts \in PointCounts : draw(pts)
    \/ destroyPipeline
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ nativeSupport \in BOOLEAN
    /\ reportedSupport \in BOOLEAN
    /\ moduleStage \in Stages
    /\ moduleLogicalAlive \in BOOLEAN
    /\ modulePhysicalAlive \in BOOLEAN
    /\ pipelineAlive \in BOOLEAN
    /\ recordedReference \in BOOLEAN
    /\ inFlightReference \in BOOLEAN
    /\ generatedVertexCount \in {0, 3, 6}
    /\ lastOperation \in Operations
    /\ lastOutcome \in Outcomes

CapabilityMatchesNative ==
    reportedSupport = nativeSupport

PipelineRequiresCapableGeometryModule ==
    (pipelineAlive \/ recordedReference \/ inFlightReference) =>
        /\ nativeSupport
        /\ moduleStage = "geometry"

ModulePhysicalLifetimeExact ==
    modulePhysicalAlive =
        (moduleLogicalAlive \/ pipelineAlive \/ recordedReference \/ inFlightReference)

GeometryAmplifiesPointsToTriangles ==
    /\ (recordedReference \/ inFlightReference) => generatedVertexCount \in {3, 6}
    /\ (~recordedReference /\ ~inFlightReference) => generatedVertexCount = 0

Safety ==
    /\ CapabilityMatchesNative
    /\ PipelineRequiresCapableGeometryModule
    /\ ModulePhysicalLifetimeExact
    /\ GeometryAmplifiesPointsToTriangles

StepSafety ==
    [][
        /\ (lastOperation' = "create" =>
                ((lastOutcome' = "accepted") = CreateValid))
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    ]_vars

=============================================================================
