--------------------- MODULE RayTracingIndirectDispatch ---------------------
(***************************************************************************)
(* DXR 1.1 indirect ray-tracing dispatch capability gating, shader-table   *)
(* and 12-byte indirect-argument record validation (rejecting readback     *)
(* memory), internal transient buffer retention across retained/unretained *)
(* pools, and physical lifetime across logical destruction.                *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Device.cpp (D3D12_RAYTRACING_TIER_1_1                *)
(*     indirectRayDispatch capability discovery)                           *)
(*   - src/D3D12/D3D12Pipeline.cpp (indirectRayTracingDispatchSignature    *)
(*     DXR tier 1.1 gate)                                                  *)
(*   - src/D3D12/D3D12RayTracingPipeline.cpp (dispatchRaysIndirect,        *)
(*     validateRayTracingDispatchState, prepareRayTracingShaderTables)     *)
(*   - src/D3D12/D3D12AccelerationStructure.cpp                            *)
(*     (recordRayTracingIndirectDispatchArguments,                         *)
(*     validateIndirectDispatchSources readback rejection, internal        *)
(*     upload/native-argument retention)                                   *)
(*   - src/D3D12/D3D12Queue.cpp (retainCommandDependencies)                *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Device capability and pool retention mode are fixed per trace in      *)
(*   Init. GPU-authored width/height/depth values inside the indirect      *)
(*   argument buffer are a producer obligation at execution time.          *)
(***************************************************************************)

EXTENDS Naturals

TableBufferSize == 128
ArgumentBufferSize == 32
ArgumentRecordSize == 12
HandleSize == 32
RecordAlignment == 32
BaseAlignment == 64
MaxStride == 64
MaxWork == 2
TableOffsets == {0, 32, 64}
TableSizes == {0, 32, 64}
Strides == {0, 16, 32, 64}
ArgumentOffsets == {0, 2, 4, 20, 24}
Operations == {
  "none", "configure", "dispatch",
  "destroyTable", "destroyArgument",
  "submit", "complete", "abandon"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  supportsIndirect,  \* Immutable DXR 1.1 indirect ray-dispatch capability
  retainApplication, \* Whether the command pool retains application resources
  pipelineBound,     \* Whether a ray-tracing pipeline is bound
  bindingsComplete,  \* Whether all required binding sets are bound
  rendering,         \* Whether an active render pass blocks ray dispatch
  tableLogical,      \* Public BufferHandle liveness of shader-table buffer
  tablePhysical,     \* Native storage liveness of shader-table buffer
  argumentLogical,   \* Public BufferHandle liveness of indirect-arg buffer
  argumentPhysical,  \* Native storage liveness of indirect-arg buffer
  recTableRef,       \* Recording retention of shader-table buffer
  recArgumentRef,    \* Recording retention of indirect-argument buffer
  recInternalCount,  \* Count of retained internal upload/native-arg pairs
  inFlightTableRef,  \* In-flight retention of shader-table buffer
  inFlightArgRef,    \* In-flight retention of indirect-argument buffer
  inFlightInternal,  \* Count of in-flight internal upload/native-arg pairs
  inFlight,          \* Whether the command buffer is currently in flight
  work,              \* Count of accepted indirect ray dispatches recorded
  lastOp,            \* Most recent operation kind
  lastOutcome        \* Outcome of most recent operation

coreVars == <<
  supportsIndirect, retainApplication,
  pipelineBound, bindingsComplete, rendering,
  tableLogical, tablePhysical,
  argumentLogical, argumentPhysical,
  recTableRef, recArgumentRef, recInternalCount,
  inFlightTableRef, inFlightArgRef, inFlightInternal,
  inFlight, work
>>

vars == <<
  supportsIndirect, retainApplication,
  pipelineBound, bindingsComplete, rendering,
  tableLogical, tablePhysical,
  argumentLogical, argumentPhysical,
  recTableRef, recArgumentRef, recInternalCount,
  inFlightTableRef, inFlightArgRef, inFlightInternal,
  inFlight, work, lastOp, lastOutcome
>>

TableRetained == recTableRef \/ inFlightTableRef
ArgumentRetained == recArgumentRef \/ inFlightArgRef
CommandReady == pipelineBound /\ bindingsComplete /\ ~rendering

RequestValid(tableUsage, tableOffset, tableSize, stride,
             argUsage, argIsReadback, argOffset) ==
  /\ ~inFlight
  /\ supportsIndirect
  /\ CommandReady
  /\ tableLogical
  /\ tablePhysical
  /\ argumentLogical
  /\ argumentPhysical
  /\ tableUsage
  /\ tableOffset % BaseAlignment = 0
  /\ tableOffset <= TableBufferSize
  /\ tableSize > 0
  /\ tableSize <= TableBufferSize - tableOffset
  /\ tableSize = stride
  /\ stride >= HandleSize
  /\ stride <= MaxStride
  /\ stride % RecordAlignment = 0
  /\ argUsage
  /\ ~argIsReadback
  /\ argOffset % 4 = 0
  /\ argOffset <= ArgumentBufferSize
  /\ ArgumentRecordSize <= ArgumentBufferSize - argOffset

Init ==
  /\ supportsIndirect \in BOOLEAN
  /\ retainApplication \in BOOLEAN
  /\ pipelineBound = FALSE
  /\ bindingsComplete = FALSE
  /\ rendering = FALSE
  /\ tableLogical = TRUE
  /\ tablePhysical = TRUE
  /\ argumentLogical = TRUE
  /\ argumentPhysical = TRUE
  /\ recTableRef = FALSE
  /\ recArgumentRef = FALSE
  /\ recInternalCount = 0
  /\ inFlightTableRef = FALSE
  /\ inFlightArgRef = FALSE
  /\ inFlightInternal = 0
  /\ inFlight = FALSE
  /\ work = 0
  /\ lastOp = "none"
  /\ lastOutcome = "none"

ConfigureState(pipeline, bindings, insideRendering) ==
  /\ ~inFlight
  /\ pipelineBound' = pipeline
  /\ bindingsComplete' = bindings
  /\ rendering' = insideRendering
  /\ UNCHANGED <<
       supportsIndirect, retainApplication,
       tableLogical, tablePhysical,
       argumentLogical, argumentPhysical,
       recTableRef, recArgumentRef, recInternalCount,
       inFlightTableRef, inFlightArgRef, inFlightInternal,
       inFlight, work
     >>
  /\ lastOp' = "configure"
  /\ lastOutcome' = "accepted"

AttemptDispatch(tableUsage, tableOffset, tableSize, stride,
                argUsage, argIsReadback, argOffset) ==
  /\ ~inFlight
  /\ work < MaxWork
  /\ IF RequestValid(tableUsage, tableOffset, tableSize, stride,
                     argUsage, argIsReadback, argOffset)
     THEN /\ recTableRef' = (recTableRef \/ retainApplication)
          /\ recArgumentRef' = (recArgumentRef \/ retainApplication)
          /\ recInternalCount' = recInternalCount + 1
          /\ work' = work + 1
          /\ UNCHANGED <<
               supportsIndirect, retainApplication,
               pipelineBound, bindingsComplete, rendering,
               tableLogical, tablePhysical,
               argumentLogical, argumentPhysical,
               inFlightTableRef, inFlightArgRef, inFlightInternal,
               inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "dispatch"

\* On unretained pools, the caller is obligated to preserve application buffers
\* until completion; on retained pools, destruction can occur at any time.
DestroyTable ==
  /\ tableLogical
  /\ (retainApplication \/ (work = 0 /\ ~inFlight))
  /\ tableLogical' = FALSE
  /\ tablePhysical' = TableRetained
  /\ UNCHANGED <<
       supportsIndirect, retainApplication,
       pipelineBound, bindingsComplete, rendering,
       argumentLogical, argumentPhysical,
       recTableRef, recArgumentRef, recInternalCount,
       inFlightTableRef, inFlightArgRef, inFlightInternal,
       inFlight, work
     >>
  /\ lastOp' = "destroyTable"
  /\ lastOutcome' = "accepted"

DestroyArgument ==
  /\ argumentLogical
  /\ (retainApplication \/ (work = 0 /\ ~inFlight))
  /\ argumentLogical' = FALSE
  /\ argumentPhysical' = ArgumentRetained
  /\ UNCHANGED <<
       supportsIndirect, retainApplication,
       pipelineBound, bindingsComplete, rendering,
       tableLogical, tablePhysical,
       recTableRef, recArgumentRef, recInternalCount,
       inFlightTableRef, inFlightArgRef, inFlightInternal,
       inFlight, work
     >>
  /\ lastOp' = "destroyArgument"
  /\ lastOutcome' = "accepted"

Submit ==
  /\ ~inFlight
  /\ ~rendering
  /\ work > 0
  /\ pipelineBound' = FALSE
  /\ bindingsComplete' = FALSE
  /\ rendering' = FALSE
  /\ recTableRef' = FALSE
  /\ recArgumentRef' = FALSE
  /\ recInternalCount' = 0
  /\ inFlightTableRef' = recTableRef
  /\ inFlightArgRef' = recArgumentRef
  /\ inFlightInternal' = recInternalCount
  /\ inFlight' = TRUE
  /\ work' = 0
  /\ UNCHANGED <<
       supportsIndirect, retainApplication,
       tableLogical, tablePhysical,
       argumentLogical, argumentPhysical
     >>
  /\ lastOp' = "submit"
  /\ lastOutcome' = "accepted"

Complete ==
  /\ inFlight
  /\ tablePhysical' = tableLogical
  /\ argumentPhysical' = argumentLogical
  /\ inFlightTableRef' = FALSE
  /\ inFlightArgRef' = FALSE
  /\ inFlightInternal' = 0
  /\ inFlight' = FALSE
  /\ UNCHANGED <<
       supportsIndirect, retainApplication,
       pipelineBound, bindingsComplete, rendering,
       tableLogical, argumentLogical,
       recTableRef, recArgumentRef, recInternalCount, work
     >>
  /\ lastOp' = "complete"
  /\ lastOutcome' = "accepted"

Abandon ==
  /\ ~inFlight
  /\ tablePhysical' = tableLogical
  /\ argumentPhysical' = argumentLogical
  /\ pipelineBound' = FALSE
  /\ bindingsComplete' = FALSE
  /\ rendering' = FALSE
  /\ recTableRef' = FALSE
  /\ recArgumentRef' = FALSE
  /\ recInternalCount' = 0
  /\ work' = 0
  /\ UNCHANGED <<
       supportsIndirect, retainApplication,
       tableLogical, argumentLogical,
       inFlightTableRef, inFlightArgRef, inFlightInternal, inFlight
     >>
  /\ lastOp' = "abandon"
  /\ lastOutcome' = "accepted"

Next ==
  \/ \E p, b, r \in BOOLEAN : ConfigureState(p, b, r)
  \/ \E tu \in BOOLEAN, toff \in TableOffsets, tsz \in TableSizes, st \in Strides :
       AttemptDispatch(tu, toff, tsz, st, TRUE, FALSE, 0)
  \/ \E au, rb \in BOOLEAN, aoff \in ArgumentOffsets :
       AttemptDispatch(TRUE, 0, 32, 32, au, rb, aoff)
  \/ DestroyTable
  \/ DestroyArgument
  \/ Submit
  \/ Complete
  \/ Abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ supportsIndirect \in BOOLEAN
  /\ retainApplication \in BOOLEAN
  /\ pipelineBound \in BOOLEAN
  /\ bindingsComplete \in BOOLEAN
  /\ rendering \in BOOLEAN
  /\ tableLogical \in BOOLEAN
  /\ tablePhysical \in BOOLEAN
  /\ argumentLogical \in BOOLEAN
  /\ argumentPhysical \in BOOLEAN
  /\ recTableRef \in BOOLEAN
  /\ recArgumentRef \in BOOLEAN
  /\ recInternalCount \in 0..MaxWork
  /\ inFlightTableRef \in BOOLEAN
  /\ inFlightArgRef \in BOOLEAN
  /\ inFlightInternal \in 0..MaxWork
  /\ inFlight \in BOOLEAN
  /\ work \in 0..MaxWork
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes

PhysicalLifetimesExact ==
  /\ tablePhysical = (tableLogical \/ TableRetained)
  /\ argumentPhysical = (argumentLogical \/ ArgumentRetained)

UnsupportedDeviceNeverRecordsIndirectWork ==
  ~supportsIndirect =>
    /\ work = 0
    /\ recInternalCount = 0
    /\ inFlightInternal = 0
    /\ ~inFlight

InternalTransientsAlwaysMatchWork ==
  /\ recInternalCount = work
  /\ (inFlight <=> inFlightInternal > 0)
  /\ (~retainApplication => (~recTableRef /\ ~recArgumentRef /\ ~inFlightTableRef /\ ~inFlightArgRef))
  /\ (retainApplication /\ work > 0 => recTableRef /\ recArgumentRef)
  /\ (retainApplication /\ inFlight => inFlightTableRef /\ inFlightArgRef)

SubmittedReferencesSeparated ==
  inFlight =>
    /\ ~recTableRef
    /\ ~recArgumentRef
    /\ recInternalCount = 0
    /\ work = 0

Safety ==
  /\ PhysicalLifetimesExact
  /\ UnsupportedDeviceNeverRecordsIndirectWork
  /\ InternalTransientsAlwaysMatchWork
  /\ SubmittedReferencesSeparated

StepSafety ==
  [][
    /\ supportsIndirect' = supportsIndirect
    /\ retainApplication' = retainApplication
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (work' > work =>
          /\ lastOp' = "dispatch"
          /\ lastOutcome' = "accepted"
          /\ supportsIndirect
          /\ CommandReady
          /\ ~inFlight
          /\ tableLogical
          /\ argumentLogical
          /\ recInternalCount' = recInternalCount + 1
          /\ work' = work + 1)
  ]_vars

=============================================================================
