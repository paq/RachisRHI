--------------------- MODULE TessellationShaderPipeline ---------------------
(***************************************************************************)
(* Formal safety model for tessellation capability reporting, exact        *)
(* hull/domain shader stage pairing, patch control-point validation,       *)
(* vertex-to-patch draw division, and module/pipeline retention:           *)
(*   - include/Rachis/Device.h (TessellationCapabilities),                 *)
(*     include/Rachis/Pipeline.h (hull/domain/patchControlPoints fields)   *)
(*   - src/D3D12/D3D12Device.cpp (capability discovery)                    *)
(*   - src/D3D12/D3D12Pipeline.cpp                                         *)
(*     (validateTessellationAndTopologySupport; the pipeline record        *)
(*     retains its hull/domain modules)                                    *)
(*   - src/D3D12/D3D12CommandRecording.cpp (bindGraphicsPipeline           *)
(*     recording retention), src/D3D12/D3D12Queue.cpp (submission)         *)
(*   - src/Vulkan/VulkanRuntime.cpp (capability discovery),                *)
(*     src/Vulkan/VulkanPipelines.cpp (validateTessellationUsage,          *)
(*     validateTessellationShaderPair)                                     *)
(*   - src/Metal/MetalPipelines.mm (metalValidateGraphicsShaderStages      *)
(*     validates the pairing, then rejects tessellation as unsupported)    *)
(*                                                                         *)
(* Separates logical handle destruction from independent physical          *)
(* reclamation of hull/domain shader modules and pipelines, models         *)
(* recording abandonment alongside GPU submission/completion, and checks   *)
(* exact patch-count division (vertexCount / patchControlPoints) on draws. *)
(*                                                                         *)
(* Trusted boundary: native tessellator fixed-function stage and shader    *)
(* bytecode execution after pipeline compilation succeeds.                 *)
(***************************************************************************)
EXTENDS Naturals

Stages == {"hull", "domain", "pixel"}
Topologies == {"ordinary", "patch"}
PatchCounts == {0, 3, 4, 33}
VertexCounts == {0, 3, 4, 6, 8}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  nativeSupport,               \* Hardware tessellation support flag.
  reportedSupport,             \* Publicly reported tessellation capability.
  nativeMaxPatchControlPoints, \* Hardware max control points per patch (0 or 32).
  reportedMaxPatchControlPoints, \* Reported max control points per patch.
  hullStage,                   \* Stage tag of the candidate hull module.
  domainStage,                 \* Stage tag of the candidate domain module.
  hullLogical,                 \* Public hull shader module handle liveness.
  hullPhysical,                \* Native hull shader module object liveness.
  domainLogical,               \* Public domain shader module handle liveness.
  domainPhysical,              \* Native domain shader module object liveness.
  pipelineLogical,             \* Public tessellation pipeline handle liveness.
  pipelinePhysical,            \* Native compiled pipeline state liveness.
  patchControlPoints,          \* Control points per patch on live/retained pipeline.
  recordedReference,           \* Whether an unsubmitted command retains the pipeline.
  inFlightReference,           \* Whether an in-flight GPU submission retains it.
  drawVertexCount,             \* Vertex count of the active recorded/submitted draw.
  generatedPatchCount,         \* Patch count produced by the active draw.
  lastOutcome                  \* Outcome of the latest fallible operation.

coreVars == <<
  nativeSupport,
  reportedSupport,
  nativeMaxPatchControlPoints,
  reportedMaxPatchControlPoints,
  hullStage,
  domainStage,
  hullLogical,
  hullPhysical,
  domainLogical,
  domainPhysical,
  pipelineLogical,
  pipelinePhysical,
  patchControlPoints,
  recordedReference,
  inFlightReference,
  drawVertexCount,
  generatedPatchCount
>>

vars == <<
  nativeSupport,
  reportedSupport,
  nativeMaxPatchControlPoints,
  reportedMaxPatchControlPoints,
  hullStage,
  domainStage,
  hullLogical,
  hullPhysical,
  domainLogical,
  domainPhysical,
  pipelineLogical,
  pipelinePhysical,
  patchControlPoints,
  recordedReference,
  inFlightReference,
  drawVertexCount,
  generatedPatchCount,
  lastOutcome
>>

Init ==
  \E support \in BOOLEAN, initHull \in Stages, initDomain \in Stages :
    LET maxPts == IF support THEN 32 ELSE 0 IN
    /\ nativeSupport = support
    /\ reportedSupport = support
    /\ nativeMaxPatchControlPoints = maxPts
    /\ reportedMaxPatchControlPoints = maxPts
    /\ hullStage = initHull
    /\ domainStage = initDomain
    /\ hullLogical = TRUE
    /\ hullPhysical = TRUE
    /\ domainLogical = TRUE
    /\ domainPhysical = TRUE
    /\ pipelineLogical = FALSE
    /\ pipelinePhysical = FALSE
    /\ patchControlPoints = 0
    /\ recordedReference = FALSE
    /\ inFlightReference = FALSE
    /\ drawVertexCount = 0
    /\ generatedPatchCount = 0
    /\ lastOutcome = "none"

attemptCreatePipeline(topology, requestedPoints) ==
  /\ topology \in Topologies
  /\ requestedPoints \in PatchCounts
  /\ LET valid ==
           /\ ~pipelineLogical
           /\ ~pipelinePhysical
           /\ nativeSupport
           /\ hullLogical
           /\ domainLogical
           /\ hullStage = "hull"
           /\ domainStage = "domain"
           /\ topology = "patch"
           /\ requestedPoints >= 1
           /\ requestedPoints <= nativeMaxPatchControlPoints
     IN IF valid
        THEN /\ pipelineLogical' = TRUE
             /\ pipelinePhysical' = TRUE
             /\ patchControlPoints' = requestedPoints
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  nativeSupport,
                  reportedSupport,
                  nativeMaxPatchControlPoints,
                  reportedMaxPatchControlPoints,
                  hullStage,
                  domainStage,
                  hullLogical,
                  hullPhysical,
                  domainLogical,
                  domainPhysical,
                  recordedReference,
                  inFlightReference,
                  drawVertexCount,
                  generatedPatchCount
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

attemptDraw(vertexCount) ==
  /\ vertexCount \in VertexCounts
  /\ LET valid ==
           /\ pipelineLogical
           /\ ~recordedReference
           /\ ~inFlightReference
           /\ patchControlPoints >= 1
           /\ vertexCount >= patchControlPoints
           /\ vertexCount % patchControlPoints = 0
     IN IF valid
        THEN /\ recordedReference' = TRUE
             /\ drawVertexCount' = vertexCount
             /\ generatedPatchCount' = vertexCount \div patchControlPoints
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  nativeSupport,
                  reportedSupport,
                  nativeMaxPatchControlPoints,
                  reportedMaxPatchControlPoints,
                  hullStage,
                  domainStage,
                  hullLogical,
                  hullPhysical,
                  domainLogical,
                  domainPhysical,
                  pipelineLogical,
                  pipelinePhysical,
                  patchControlPoints,
                  inFlightReference
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

abandonRecording ==
  /\ recordedReference
  /\ ~inFlightReference
  /\ recordedReference' = FALSE
  /\ drawVertexCount' = 0
  /\ generatedPatchCount' = 0
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeSupport,
       reportedSupport,
       nativeMaxPatchControlPoints,
       reportedMaxPatchControlPoints,
       hullStage,
       domainStage,
       hullLogical,
       hullPhysical,
       domainLogical,
       domainPhysical,
       pipelineLogical,
       pipelinePhysical,
       patchControlPoints,
       inFlightReference
     >>

submit ==
  /\ recordedReference
  /\ ~inFlightReference
  /\ recordedReference' = FALSE
  /\ inFlightReference' = TRUE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeSupport,
       reportedSupport,
       nativeMaxPatchControlPoints,
       reportedMaxPatchControlPoints,
       hullStage,
       domainStage,
       hullLogical,
       hullPhysical,
       domainLogical,
       domainPhysical,
       pipelineLogical,
       pipelinePhysical,
       patchControlPoints,
       drawVertexCount,
       generatedPatchCount
     >>

complete ==
  /\ inFlightReference
  /\ inFlightReference' = FALSE
  /\ drawVertexCount' = 0
  /\ generatedPatchCount' = 0
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeSupport,
       reportedSupport,
       nativeMaxPatchControlPoints,
       reportedMaxPatchControlPoints,
       hullStage,
       domainStage,
       hullLogical,
       hullPhysical,
       domainLogical,
       domainPhysical,
       pipelineLogical,
       pipelinePhysical,
       patchControlPoints,
       recordedReference
     >>

destroyPipeline ==
  /\ pipelineLogical
  /\ pipelineLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeSupport,
       reportedSupport,
       nativeMaxPatchControlPoints,
       reportedMaxPatchControlPoints,
       hullStage,
       domainStage,
       hullLogical,
       hullPhysical,
       domainLogical,
       domainPhysical,
       pipelinePhysical,
       patchControlPoints,
       recordedReference,
       inFlightReference,
       drawVertexCount,
       generatedPatchCount
     >>

collectPipeline ==
  /\ pipelinePhysical
  /\ ~pipelineLogical
  /\ ~recordedReference
  /\ ~inFlightReference
  /\ pipelinePhysical' = FALSE
  /\ patchControlPoints' = 0
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeSupport,
       reportedSupport,
       nativeMaxPatchControlPoints,
       reportedMaxPatchControlPoints,
       hullStage,
       domainStage,
       hullLogical,
       hullPhysical,
       domainLogical,
       domainPhysical,
       pipelineLogical,
       recordedReference,
       inFlightReference,
       drawVertexCount,
       generatedPatchCount
     >>

destroyHull ==
  /\ hullLogical
  /\ hullLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeSupport,
       reportedSupport,
       nativeMaxPatchControlPoints,
       reportedMaxPatchControlPoints,
       hullStage,
       domainStage,
       hullPhysical,
       domainLogical,
       domainPhysical,
       pipelineLogical,
       pipelinePhysical,
       patchControlPoints,
       recordedReference,
       inFlightReference,
       drawVertexCount,
       generatedPatchCount
     >>

collectHull ==
  /\ hullPhysical
  /\ ~hullLogical
  /\ ~pipelinePhysical
  /\ hullPhysical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeSupport,
       reportedSupport,
       nativeMaxPatchControlPoints,
       reportedMaxPatchControlPoints,
       hullStage,
       domainStage,
       hullLogical,
       domainLogical,
       domainPhysical,
       pipelineLogical,
       pipelinePhysical,
       patchControlPoints,
       recordedReference,
       inFlightReference,
       drawVertexCount,
       generatedPatchCount
     >>

destroyDomain ==
  /\ domainLogical
  /\ domainLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeSupport,
       reportedSupport,
       nativeMaxPatchControlPoints,
       reportedMaxPatchControlPoints,
       hullStage,
       domainStage,
       hullLogical,
       hullPhysical,
       domainPhysical,
       pipelineLogical,
       pipelinePhysical,
       patchControlPoints,
       recordedReference,
       inFlightReference,
       drawVertexCount,
       generatedPatchCount
     >>

collectDomain ==
  /\ domainPhysical
  /\ ~domainLogical
  /\ ~pipelinePhysical
  /\ domainPhysical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeSupport,
       reportedSupport,
       nativeMaxPatchControlPoints,
       reportedMaxPatchControlPoints,
       hullStage,
       domainStage,
       hullLogical,
       hullPhysical,
       domainLogical,
       pipelineLogical,
       pipelinePhysical,
       patchControlPoints,
       recordedReference,
       inFlightReference,
       drawVertexCount,
       generatedPatchCount
     >>

Next ==
  \/ \E top \in Topologies, pts \in PatchCounts : attemptCreatePipeline(top, pts)
  \/ \E vtx \in VertexCounts : attemptDraw(vtx)
  \/ abandonRecording
  \/ submit
  \/ complete
  \/ destroyPipeline
  \/ collectPipeline
  \/ destroyHull
  \/ collectHull
  \/ destroyDomain
  \/ collectDomain

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ nativeSupport \in BOOLEAN
  /\ reportedSupport \in BOOLEAN
  /\ nativeMaxPatchControlPoints \in {0, 32}
  /\ reportedMaxPatchControlPoints \in {0, 32}
  /\ hullStage \in Stages
  /\ domainStage \in Stages
  /\ hullLogical \in BOOLEAN
  /\ hullPhysical \in BOOLEAN
  /\ domainLogical \in BOOLEAN
  /\ domainPhysical \in BOOLEAN
  /\ pipelineLogical \in BOOLEAN
  /\ pipelinePhysical \in BOOLEAN
  /\ patchControlPoints \in {0, 3, 4}
  /\ recordedReference \in BOOLEAN
  /\ inFlightReference \in BOOLEAN
  /\ drawVertexCount \in VertexCounts
  /\ generatedPatchCount \in {0, 1, 2}
  /\ lastOutcome \in Outcomes

CapabilityMappingIsExact ==
  /\ reportedSupport = nativeSupport
  /\ reportedMaxPatchControlPoints = nativeMaxPatchControlPoints
  /\ (nativeSupport <=> nativeMaxPatchControlPoints = 32)

LivePipelineHasValidStagesAndPatchSize ==
  pipelinePhysical =>
    /\ nativeSupport
    /\ hullStage = "hull"
    /\ domainStage = "domain"
    /\ patchControlPoints >= 1
    /\ patchControlPoints <= nativeMaxPatchControlPoints

RetainedReferencesKeepPipelineAndModulesPhysical ==
  /\ (hullLogical => hullPhysical)
  /\ (domainLogical => domainPhysical)
  /\ (pipelineLogical => pipelinePhysical)
  /\ ((recordedReference \/ inFlightReference) => pipelinePhysical)
  /\ (pipelinePhysical => hullPhysical /\ domainPhysical)
  /\ ~(recordedReference /\ inFlightReference)

DrawPatchCountMatchesControlPoints ==
  (recordedReference \/ inFlightReference) =>
    /\ generatedPatchCount >= 1
    /\ drawVertexCount = generatedPatchCount * patchControlPoints

Safety ==
  /\ CapabilityMappingIsExact
  /\ LivePipelineHasValidStagesAndPatchSize
  /\ RetainedReferencesKeepPipelineAndModulesPhysical
  /\ DrawPatchCountMatchesControlPoints

StepSafety ==
  [][lastOutcome' = "rejected" => coreVars' = coreVars]_vars

=============================================================================
