---------------------------- MODULE StreamOutput ----------------------------
(***************************************************************************)
(* Formal safety model for stream-output pipeline declarations, atomic     *)
(* multi-slot output buffer binding, per-stream statistics queries that    *)
(* begin and end on the same side of a rendering-scope boundary, and       *)
(* command resource retention:                                             *)
(*   - src/D3D12/D3D12Pipeline.cpp (stream-output declarations, strides)   *)
(*   - src/D3D12/D3D12CommandRecording.cpp (bindStreamOutputBuffers, draw  *)
(*     readiness requiring every declared slot, endRendering)              *)
(*   - src/D3D12/D3D12Resource.cpp (stream-output buffer lifetime)         *)
(*   - src/D3D12/D3D12Query.cpp (stream-output statistics queries),        *)
(*     src/D3D12/D3D12Queue.cpp (submission retention)                     *)
(*   - src/Device.cpp and src/Internal/PortableValidation.h (endRendering  *)
(*     and the shared rendering-scope query checks)                        *)
(*                                                                         *)
(* Models two stream-output buffer slots (0 and 1) so multi-buffer         *)
(* pipeline declarations and draws requiring every declared slot are       *)
(* exercised; per-stream query scopes (streams 0 and 1); recording         *)
(* abandonment vs submission; independent physical resource reclamation;   *)
(* and transition-checked failure atomicity without clone variables.       *)
(*                                                                         *)
(* Trusted boundary: D3D12 rasterizer/geometry primitive emission and      *)
(* native SOSetTargets DMA counter updates.                                *)
(***************************************************************************)
EXTENDS Integers

Slots == {0, 1}
DeclaredMasks == {{0}, {0, 1}}
Streams == {0, 1}
StreamCandidates == {0, 1, 2}
Output0 == 0
Output1 == 1
Counter == 2
QueryPool == 3
Destination == 4
Resources == 0..4
PrimitiveCandidates == {0, 1}
OffsetCandidates == {0, 4, 16, 24}
Outcomes == {"none", "accepted", "rejected"}
ResultSize == 16
ResultAlignment == 8
DestinationSize == 32
UnwrittenQuery == -1

VARIABLES
  pipelinePublished, \* Whether a stream-output graphics pipeline exists.
  declaredSlots,     \* Non-empty subset of Slots required by the pipeline.
  rendering,         \* Whether a graphics rendering scope is open.
  pipelineBound,     \* Whether the stream-output pipeline is currently bound.
  boundSlots,        \* Subset of Slots currently bound in the rendering scope.
  logicalAlive,      \* Per-resource public handle liveness.
  physicalAlive,     \* Per-resource native storage liveness.
  recordingLease,    \* Per-resource command recording retention flag.
  inFlightLease,     \* Per-resource GPU submission retention flag.
  activeQueries,     \* Per-stream active query scope flag.
  queryInPass,       \* Per-stream flag: the active query began in rendering.
  completedQueries,  \* Per-stream ended-but-unresolved query flag.
  queryWritten,      \* Per-stream recorded written-primitive count (-1..1).
  queryNeeded,       \* Per-stream recorded needed-primitive count (-1..1).
  primitivesWritten, \* Written primitives from the latest accepted draw (0..1).
  primitivesNeeded,  \* Storage-needed primitives from latest accepted draw (0..1).
  recordedWork,      \* Whether the command buffer currently holds recorded work.
  inFlightWork,      \* Whether the command buffer is currently in flight on GPU.
  lastOutcome        \* Outcome of the latest fallible operation.

coreVars == <<
  pipelinePublished,
  declaredSlots,
  rendering,
  pipelineBound,
  boundSlots,
  logicalAlive,
  physicalAlive,
  recordingLease,
  inFlightLease,
  activeQueries,
  queryInPass,
  completedQueries,
  queryWritten,
  queryNeeded,
  primitivesWritten,
  primitivesNeeded,
  recordedWork,
  inFlightWork
>>

vars == <<
  pipelinePublished,
  declaredSlots,
  rendering,
  pipelineBound,
  boundSlots,
  logicalAlive,
  physicalAlive,
  recordingLease,
  inFlightLease,
  activeQueries,
  queryInPass,
  completedQueries,
  queryWritten,
  queryNeeded,
  primitivesWritten,
  primitivesNeeded,
  recordedWork,
  inFlightWork,
  lastOutcome
>>

PoolResource(stream) == QueryPool

AnyQueryActive == \E s \in Streams : activeQueries[s]
AnyQueryUnresolved == \E s \in Streams : completedQueries[s]

Init ==
  /\ pipelinePublished = FALSE
  /\ declaredSlots = {0}
  /\ rendering = FALSE
  /\ pipelineBound = FALSE
  /\ boundSlots = {}
  /\ logicalAlive = [r \in Resources |-> TRUE]
  /\ physicalAlive = [r \in Resources |-> TRUE]
  /\ recordingLease = [r \in Resources |-> FALSE]
  /\ inFlightLease = [r \in Resources |-> FALSE]
  /\ activeQueries = [s \in Streams |-> FALSE]
  /\ queryInPass = [s \in Streams |-> FALSE]
  /\ completedQueries = [s \in Streams |-> FALSE]
  /\ queryWritten = [s \in Streams |-> UnwrittenQuery]
  /\ queryNeeded = [s \in Streams |-> UnwrittenQuery]
  /\ primitivesWritten = 0
  /\ primitivesNeeded = 0
  /\ recordedWork = FALSE
  /\ inFlightWork = FALSE
  /\ lastOutcome = "none"

attemptCreatePipeline(slots, declarationsValid, strideValid) ==
  /\ slots \in DeclaredMasks
  /\ IF ~pipelinePublished /\ declarationsValid /\ strideValid
     THEN /\ pipelinePublished' = TRUE
          /\ declaredSlots' = slots
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               rendering,
               pipelineBound,
               boundSlots,
               logicalAlive,
               physicalAlive,
               recordingLease,
               inFlightLease,
               activeQueries,
               queryInPass,
               completedQueries,
               queryWritten,
               queryNeeded,
               primitivesWritten,
               primitivesNeeded,
               recordedWork,
               inFlightWork
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

beginRendering ==
  /\ pipelinePublished
  /\ ~rendering
  /\ ~inFlightWork
  /\ rendering' = TRUE
  /\ pipelineBound' = FALSE
  /\ boundSlots' = {}
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       pipelinePublished,
       declaredSlots,
       logicalAlive,
       physicalAlive,
       recordingLease,
       inFlightLease,
       activeQueries,
       queryInPass,
       completedQueries,
       queryWritten,
       queryNeeded,
       primitivesWritten,
       primitivesNeeded,
       recordedWork,
       inFlightWork
     >>

bindPipeline ==
  /\ rendering
  /\ pipelinePublished
  /\ pipelineBound' = TRUE
  /\ boundSlots' = {}
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       pipelinePublished,
       declaredSlots,
       rendering,
       logicalAlive,
       physicalAlive,
       recordingLease,
       inFlightLease,
       activeQueries,
       queryInPass,
       completedQueries,
       queryWritten,
       queryNeeded,
       primitivesWritten,
       primitivesNeeded,
       recordedWork,
       inFlightWork
     >>

\* Atomic multi-buffer binding: either every slot in requestedSlots is live and
\* valid and retained together, or prior bindings and leases remain untouched.
attemptBindOutputs(requestedSlots, rangesValid) ==
  /\ requestedSlots \in SUBSET Slots
  /\ LET valid ==
           /\ rendering
           /\ pipelineBound
           /\ ~inFlightWork
           /\ requestedSlots # {}
           /\ rangesValid
           /\ logicalAlive[Counter]
           /\ \A slot \in requestedSlots : logicalAlive[slot]
     IN IF valid
        THEN /\ boundSlots' = requestedSlots
             /\ recordingLease' =
                  [r \in Resources |->
                     recordingLease[r] \/ r = Counter \/ r \in requestedSlots]
             /\ recordedWork' = TRUE
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  pipelinePublished,
                  declaredSlots,
                  rendering,
                  pipelineBound,
                  logicalAlive,
                  physicalAlive,
                  inFlightLease,
                  activeQueries,
                  queryInPass,
                  completedQueries,
                  queryWritten,
                  queryNeeded,
                  primitivesWritten,
                  primitivesNeeded,
                  inFlightWork
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

attemptDraw(primitives, capacity) ==
  /\ primitives \in PrimitiveCandidates
  /\ capacity \in PrimitiveCandidates
  /\ LET valid ==
           /\ rendering
           /\ pipelineBound
           /\ declaredSlots \subseteq boundSlots
           /\ primitives = 1
           /\ ~inFlightWork
         written == IF primitives <= capacity THEN primitives ELSE capacity
     IN IF valid
        THEN /\ primitivesWritten' = written
             /\ primitivesNeeded' = primitives
             /\ recordedWork' = TRUE
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  pipelinePublished,
                  declaredSlots,
                  rendering,
                  pipelineBound,
                  boundSlots,
                  logicalAlive,
                  physicalAlive,
                  recordingLease,
                  inFlightLease,
                  activeQueries,
                  queryInPass,
                  completedQueries,
                  queryWritten,
                  queryNeeded,
                  inFlightWork
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

attemptBeginQuery(stream) ==
  /\ stream \in StreamCandidates
  /\ LET valid ==
           /\ stream \in Streams
           /\ ~inFlightWork
           /\ logicalAlive[PoolResource(stream)]
           /\ ~activeQueries[stream]
           /\ ~completedQueries[stream]
     IN IF valid
        THEN /\ recordingLease' = [recordingLease EXCEPT ![PoolResource(stream)] = TRUE]
             /\ activeQueries' = [activeQueries EXCEPT ![stream] = TRUE]
             /\ queryInPass' = [queryInPass EXCEPT ![stream] = rendering]
             /\ queryWritten' = [queryWritten EXCEPT ![stream] = UnwrittenQuery]
             /\ queryNeeded' = [queryNeeded EXCEPT ![stream] = UnwrittenQuery]
             /\ recordedWork' = TRUE
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  pipelinePublished,
                  declaredSlots,
                  rendering,
                  pipelineBound,
                  boundSlots,
                  logicalAlive,
                  physicalAlive,
                  inFlightLease,
                  completedQueries,
                  primitivesWritten,
                  primitivesNeeded,
                  inFlightWork
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

attemptEndQuery(stream) ==
  /\ stream \in StreamCandidates
  /\ LET valid ==
           /\ stream \in Streams
           /\ ~inFlightWork
           /\ activeQueries[stream]
           /\ rendering = queryInPass[stream]
           /\ physicalAlive[PoolResource(stream)]
           /\ recordingLease[PoolResource(stream)]
     IN IF valid
        THEN /\ activeQueries' = [activeQueries EXCEPT ![stream] = FALSE]
             /\ queryInPass' = [queryInPass EXCEPT ![stream] = FALSE]
             /\ completedQueries' = [completedQueries EXCEPT ![stream] = TRUE]
             /\ queryWritten' = [queryWritten EXCEPT ![stream] = primitivesWritten]
             /\ queryNeeded' = [queryNeeded EXCEPT ![stream] = primitivesNeeded]
             /\ recordedWork' = TRUE
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  pipelinePublished,
                  declaredSlots,
                  rendering,
                  pipelineBound,
                  boundSlots,
                  logicalAlive,
                  physicalAlive,
                  recordingLease,
                  inFlightLease,
                  primitivesWritten,
                  primitivesNeeded,
                  inFlightWork
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

attemptResolve(stream, destinationOffset) ==
  /\ stream \in StreamCandidates
  /\ destinationOffset \in OffsetCandidates
  /\ LET valid ==
           /\ stream \in Streams
           /\ ~inFlightWork
           /\ ~rendering
           /\ ~AnyQueryActive
           /\ completedQueries[stream]
           /\ logicalAlive[PoolResource(stream)]
           /\ logicalAlive[Destination]
           /\ destinationOffset % ResultAlignment = 0
           /\ destinationOffset + ResultSize <= DestinationSize
     IN IF valid
        THEN /\ recordingLease' = [recordingLease EXCEPT ![Destination] = TRUE]
             /\ completedQueries' = [completedQueries EXCEPT ![stream] = FALSE]
             /\ queryWritten' = [queryWritten EXCEPT ![stream] = UnwrittenQuery]
             /\ queryNeeded' = [queryNeeded EXCEPT ![stream] = UnwrittenQuery]
             /\ recordedWork' = TRUE
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  pipelinePublished,
                  declaredSlots,
                  rendering,
                  pipelineBound,
                  boundSlots,
                  logicalAlive,
                  physicalAlive,
                  inFlightLease,
                  activeQueries,
                  queryInPass,
                  primitivesWritten,
                  primitivesNeeded,
                  inFlightWork
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

endRendering ==
  /\ rendering
  /\ ~\E s \in Streams : queryInPass[s]
  /\ rendering' = FALSE
  /\ pipelineBound' = FALSE
  /\ boundSlots' = {}
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       pipelinePublished,
       declaredSlots,
       logicalAlive,
       physicalAlive,
       recordingLease,
       inFlightLease,
       activeQueries,
       queryInPass,
       completedQueries,
       queryWritten,
       queryNeeded,
       primitivesWritten,
       primitivesNeeded,
       recordedWork,
       inFlightWork
     >>

abandonRecording ==
  /\ recordedWork
  /\ ~inFlightWork
  /\ ~rendering
  /\ ~AnyQueryActive
  /\ recordingLease' = [r \in Resources |-> FALSE]
  /\ completedQueries' = [s \in Streams |-> FALSE]
  /\ queryWritten' = [s \in Streams |-> UnwrittenQuery]
  /\ queryNeeded' = [s \in Streams |-> UnwrittenQuery]
  /\ primitivesWritten' = 0
  /\ primitivesNeeded' = 0
  /\ recordedWork' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       pipelinePublished,
       declaredSlots,
       rendering,
       pipelineBound,
       boundSlots,
       logicalAlive,
       physicalAlive,
       inFlightLease,
       activeQueries,
       queryInPass,
       inFlightWork
     >>

submit ==
  /\ ~rendering
  /\ ~AnyQueryActive
  /\ ~AnyQueryUnresolved
  /\ recordedWork
  /\ ~inFlightWork
  /\ pipelineBound' = FALSE
  /\ boundSlots' = {}
  /\ recordingLease' = [r \in Resources |-> FALSE]
  /\ inFlightLease' = recordingLease
  /\ primitivesWritten' = 0
  /\ primitivesNeeded' = 0
  /\ recordedWork' = FALSE
  /\ inFlightWork' = TRUE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       pipelinePublished,
       declaredSlots,
       rendering,
       logicalAlive,
       physicalAlive,
       activeQueries,
       queryInPass,
       completedQueries,
       queryWritten,
       queryNeeded
     >>

complete ==
  /\ inFlightWork
  /\ inFlightLease' = [r \in Resources |-> FALSE]
  /\ inFlightWork' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       pipelinePublished,
       declaredSlots,
       rendering,
       pipelineBound,
       boundSlots,
       logicalAlive,
       physicalAlive,
       recordingLease,
       activeQueries,
       queryInPass,
       completedQueries,
       queryWritten,
       queryNeeded,
       primitivesWritten,
       primitivesNeeded,
       recordedWork
     >>

destroyResource(resource) ==
  /\ logicalAlive[resource]
  /\ logicalAlive' = [logicalAlive EXCEPT ![resource] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       pipelinePublished,
       declaredSlots,
       rendering,
       pipelineBound,
       boundSlots,
       physicalAlive,
       recordingLease,
       inFlightLease,
       activeQueries,
       queryInPass,
       completedQueries,
       queryWritten,
       queryNeeded,
       primitivesWritten,
       primitivesNeeded,
       recordedWork,
       inFlightWork
     >>

collectResource(resource) ==
  /\ physicalAlive[resource]
  /\ ~logicalAlive[resource]
  /\ ~recordingLease[resource]
  /\ ~inFlightLease[resource]
  /\ physicalAlive' = [physicalAlive EXCEPT ![resource] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       pipelinePublished,
       declaredSlots,
       rendering,
       pipelineBound,
       boundSlots,
       logicalAlive,
       recordingLease,
       inFlightLease,
       activeQueries,
       queryInPass,
       completedQueries,
       queryWritten,
       queryNeeded,
       primitivesWritten,
       primitivesNeeded,
       recordedWork,
       inFlightWork
     >>

Next ==
  \/ \E slots \in DeclaredMasks, dOk, sOk \in BOOLEAN :
       attemptCreatePipeline(slots, dOk, sOk)
  \/ beginRendering
  \/ bindPipeline
  \/ \E slots \in SUBSET Slots, ok \in BOOLEAN : attemptBindOutputs(slots, ok)
  \/ \E p, cap \in PrimitiveCandidates : attemptDraw(p, cap)
  \/ \E stream \in StreamCandidates : attemptBeginQuery(stream)
  \/ \E stream \in StreamCandidates : attemptEndQuery(stream)
  \/ \E stream \in StreamCandidates, off \in OffsetCandidates : attemptResolve(stream, off)
  \/ endRendering
  \/ abandonRecording
  \/ submit
  \/ complete
  \/ \E r \in Resources : destroyResource(r)
  \/ \E r \in Resources : collectResource(r)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ pipelinePublished \in BOOLEAN
  /\ declaredSlots \in DeclaredMasks
  /\ rendering \in BOOLEAN
  /\ pipelineBound \in BOOLEAN
  /\ boundSlots \in SUBSET Slots
  /\ logicalAlive \in [Resources -> BOOLEAN]
  /\ physicalAlive \in [Resources -> BOOLEAN]
  /\ recordingLease \in [Resources -> BOOLEAN]
  /\ inFlightLease \in [Resources -> BOOLEAN]
  /\ activeQueries \in [Streams -> BOOLEAN]
  /\ queryInPass \in [Streams -> BOOLEAN]
  /\ completedQueries \in [Streams -> BOOLEAN]
  /\ queryWritten \in [Streams -> {UnwrittenQuery, 0, 1}]
  /\ queryNeeded \in [Streams -> {UnwrittenQuery, 0, 1}]
  /\ primitivesWritten \in PrimitiveCandidates
  /\ primitivesNeeded \in PrimitiveCandidates
  /\ recordedWork \in BOOLEAN
  /\ inFlightWork \in BOOLEAN
  /\ lastOutcome \in Outcomes

BoundSlotsAreRetainedInsideScope ==
  /\ (boundSlots # {} => rendering /\ pipelineBound /\ recordingLease[Counter])
  /\ \A slot \in boundSlots : recordingLease[slot] /\ physicalAlive[slot]

OutputNeverExceedsDemand ==
  /\ primitivesWritten <= primitivesNeeded
  /\ \A s \in Streams :
       completedQueries[s] =>
         queryWritten[s] >= 0 /\ queryWritten[s] <= queryNeeded[s]

ActiveQueriesAreIndependentAndRetained ==
  \A s \in Streams :
    activeQueries[s] =>
      /\ recordingLease[PoolResource(s)]
      /\ physicalAlive[PoolResource(s)]
      /\ ~completedQueries[s]

InPassQueriesAreRenderBound ==
  \A s \in Streams :
    queryInPass[s] => activeQueries[s] /\ rendering

ResourceRetention ==
  \A r \in Resources :
    (logicalAlive[r] \/ recordingLease[r] \/ inFlightLease[r]) => physicalAlive[r]

RecordingAndSubmissionAreSeparate ==
  /\ ~(recordedWork /\ inFlightWork)
  /\ \A r \in Resources : ~(recordingLease[r] /\ inFlightLease[r])

SupportingPipelineRequired ==
  pipelineBound => pipelinePublished

Safety ==
  /\ BoundSlotsAreRetainedInsideScope
  /\ OutputNeverExceedsDemand
  /\ SupportingPipelineRequired
  /\ ActiveQueriesAreIndependentAndRetained
  /\ InPassQueriesAreRenderBound
  /\ ResourceRetention
  /\ RecordingAndSubmissionAreSeparate

StepSafety ==
  [][lastOutcome' = "rejected" => coreVars' = coreVars]_vars

=============================================================================
