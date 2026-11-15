------------------------ MODULE GraphicsCommandState ------------------------
(***************************************************************************)
(* Graphics render-pass state, pipeline attachment-signature matching,     *)
(* pipeline-scoped binding invalidation, dynamic state completeness, and   *)
(* pipeline retention in `src/D3D12/D3D12CommandRecording.cpp` (rendering  *)
(* scopes, `PipelineManager::bindGraphicsPipeline`, dynamic state, and     *)
(* draw readiness), `src/D3D12/D3D12Pipeline.cpp` (graphics pipeline       *)
(* records and destruction), and `src/D3D12/D3D12Queue.cpp`                *)
(* (`include/Rachis/Device.h`).                                            *)
(*                                                                         *)
(* Models two graphics pipelines (`0` and `1`, both targeting attachment   *)
(* signature `0`) and two render-pass attachment signatures (`0` and `1`). *)
(* Binding a graphics pipeline resets `boundPipelineForSlot`, vertex and   *)
(* index buffer bindings, and `stencilReferenceSet` while preserving       *)
(* `viewportSet`, `scissorSet`, and `blendConstantsSet` within the active  *)
(* render pass. Render-pass-only recordings (`recordedWork` without any    *)
(* bound pipeline) may be submitted or abandoned once `endRendering` has   *)
(* closed the pass, and physical pipelines live iff logically live or      *)
(* retained by recording or in-flight work.                                *)
(*                                                                         *)
(* Abstracted: attachment clear/load pixel values, push constants, and     *)
(* native root-signature descriptor heaps.                                 *)
(***************************************************************************)
EXTENDS Integers

Pipelines == {0, 1}
Signatures == {0, 1}
Slots == {0, 1}
VertexSlots == {0, 1}
MaxDraws == 2
None == -1
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    logicalPipelineAlive,       \* public handle liveness in [Pipelines -> BOOLEAN]
    physicalPipelineAlive,      \* native PSO liveness in [Pipelines -> BOOLEAN]
    recordingReferences,        \* pipelines retained by the active recording
    inFlightReferences,         \* pipelines retained by an in-flight submission
    recordedWork,               \* TRUE once any render pass or command was recorded
    inFlight,                   \* TRUE while submitted command buffer is in flight
    renderSignature,            \* active render-pass signature in Signatures, or None
    activePipeline,             \* bound pipeline in Pipelines, or None
    boundPipelineForSlot,       \* pipeline under which each binding slot was bound
    boundVertexBuffers,         \* bound vertex-buffer flags in [VertexSlots -> BOOLEAN]
    indexBufferBound,           \* TRUE when an index buffer is bound for activePipeline
    viewportSet,                \* TRUE once setViewport ran in the current pass
    scissorSet,                 \* TRUE once setScissor ran in the current pass
    blendConstantsRequired,     \* TRUE when activePipeline uses blend constants
    blendConstantsSet,          \* TRUE once setBlendConstants ran in the current pass
    stencilReferenceRequired,   \* TRUE when activePipeline enables stencil test
    stencilReferenceSet,        \* TRUE once setStencilReference ran since bindPipeline
    draws,                      \* accepted draw count in the active recording
    lastOutcome                 \* outcome of the latest step in Outcomes

coreVars == <<
    logicalPipelineAlive, physicalPipelineAlive, recordingReferences,
    inFlightReferences, recordedWork, inFlight, renderSignature,
    activePipeline, boundPipelineForSlot, boundVertexBuffers,
    indexBufferBound, viewportSet, scissorSet, blendConstantsRequired,
    blendConstantsSet, stencilReferenceRequired, stencilReferenceSet, draws
>>

vars == <<
    logicalPipelineAlive, physicalPipelineAlive, recordingReferences,
    inFlightReferences, recordedWork, inFlight, renderSignature,
    activePipeline, boundPipelineForSlot, boundVertexBuffers,
    indexBufferBound, viewportSet, scissorSet, blendConstantsRequired,
    blendConstantsSet, stencilReferenceRequired, stencilReferenceSet,
    draws, lastOutcome
>>

\* Both pipelines target attachment signature 0 so switching 0 <-> 1 is exercised
\* inside signature-0 passes, while signature-1 passes reject incompatible binds.
PipelineSignature(pipeline) == 0

RequiredSet(pipeline, slot) ==
    slot = 0 \/ pipeline = 1

RequiredVertexBuffer(pipeline, slot) ==
    pipeline = 1 /\ slot = 0

BindingsComplete(pipeline) ==
    \A slot \in Slots :
        RequiredSet(pipeline, slot) => boundPipelineForSlot[slot] = pipeline

VertexBuffersComplete(pipeline) ==
    \A slot \in VertexSlots :
        RequiredVertexBuffer(pipeline, slot) => boundVertexBuffers[slot]

GraphicsStateComplete(indexed) ==
    /\ renderSignature \in Signatures
    /\ activePipeline \in Pipelines
    /\ renderSignature = PipelineSignature(activePipeline)
    /\ BindingsComplete(activePipeline)
    /\ VertexBuffersComplete(activePipeline)
    /\ viewportSet
    /\ scissorSet
    /\ (~blendConstantsRequired \/ blendConstantsSet)
    /\ (~stencilReferenceRequired \/ stencilReferenceSet)
    /\ (~indexed \/ indexBufferBound)

Retained(pipeline) ==
    pipeline \in recordingReferences \/ pipeline \in inFlightReferences

Init ==
    /\ logicalPipelineAlive = [p \in Pipelines |-> FALSE]
    /\ physicalPipelineAlive = [p \in Pipelines |-> FALSE]
    /\ recordingReferences = {}
    /\ inFlightReferences = {}
    /\ recordedWork = FALSE
    /\ inFlight = FALSE
    /\ renderSignature = None
    /\ activePipeline = None
    /\ boundPipelineForSlot = [s \in Slots |-> None]
    /\ boundVertexBuffers = [v \in VertexSlots |-> FALSE]
    /\ indexBufferBound = FALSE
    /\ viewportSet = FALSE
    /\ scissorSet = FALSE
    /\ blendConstantsRequired = FALSE
    /\ blendConstantsSet = FALSE
    /\ stencilReferenceRequired = FALSE
    /\ stencilReferenceSet = FALSE
    /\ draws = 0
    /\ lastOutcome = "none"

createPipeline(pipeline) ==
    /\ ~logicalPipelineAlive[pipeline]
    /\ ~physicalPipelineAlive[pipeline]
    /\ logicalPipelineAlive' = [logicalPipelineAlive EXCEPT ![pipeline] = TRUE]
    /\ physicalPipelineAlive' = [physicalPipelineAlive EXCEPT ![pipeline] = TRUE]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        recordingReferences, inFlightReferences, recordedWork, inFlight,
        renderSignature, activePipeline, boundPipelineForSlot, boundVertexBuffers,
        indexBufferBound, viewportSet, scissorSet, blendConstantsRequired,
        blendConstantsSet, stencilReferenceRequired, stencilReferenceSet, draws
       >>

destroyPipeline(pipeline) ==
    /\ logicalPipelineAlive[pipeline]
    /\ logicalPipelineAlive' = [logicalPipelineAlive EXCEPT ![pipeline] = FALSE]
    /\ physicalPipelineAlive' = [physicalPipelineAlive EXCEPT ![pipeline] = Retained(pipeline)]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        recordingReferences, inFlightReferences, recordedWork, inFlight,
        renderSignature, activePipeline, boundPipelineForSlot, boundVertexBuffers,
        indexBufferBound, viewportSet, scissorSet, blendConstantsRequired,
        blendConstantsSet, stencilReferenceRequired, stencilReferenceSet, draws
       >>

beginRendering(signature) ==
    /\ ~inFlight
    /\ renderSignature = None
    /\ recordedWork' = TRUE
    /\ renderSignature' = signature
    /\ activePipeline' = None
    /\ boundPipelineForSlot' = [s \in Slots |-> None]
    /\ boundVertexBuffers' = [v \in VertexSlots |-> FALSE]
    /\ indexBufferBound' = FALSE
    /\ viewportSet' = FALSE
    /\ scissorSet' = FALSE
    /\ blendConstantsRequired' = FALSE
    /\ blendConstantsSet' = FALSE
    /\ stencilReferenceRequired' = FALSE
    /\ stencilReferenceSet' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalPipelineAlive, physicalPipelineAlive,
        recordingReferences, inFlightReferences, inFlight, draws
       >>

endRendering ==
    /\ ~inFlight
    /\ renderSignature \in Signatures
    /\ renderSignature' = None
    /\ activePipeline' = None
    /\ boundPipelineForSlot' = [s \in Slots |-> None]
    /\ boundVertexBuffers' = [v \in VertexSlots |-> FALSE]
    /\ indexBufferBound' = FALSE
    /\ viewportSet' = FALSE
    /\ scissorSet' = FALSE
    /\ blendConstantsRequired' = FALSE
    /\ blendConstantsSet' = FALSE
    /\ stencilReferenceRequired' = FALSE
    /\ stencilReferenceSet' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalPipelineAlive, physicalPipelineAlive, recordingReferences,
        inFlightReferences, recordedWork, inFlight, draws
       >>

\* Matches PipelineManager::bindGraphicsPipeline in D3D12CommandRecording.cpp:
\* resets boundBindingSets, vertex/index buffers, and stencilReferenceSet while
\* preserving viewport, scissor, and blendConstantsSet.
attemptBindPipeline(pipeline) ==
    LET accepted ==
            /\ renderSignature = PipelineSignature(pipeline)
            /\ logicalPipelineAlive[pipeline]
            /\ physicalPipelineAlive[pipeline]
    IN /\ ~inFlight
       /\ renderSignature \in Signatures
       /\ IF accepted
          THEN /\ recordingReferences' = recordingReferences \cup {pipeline}
               /\ recordedWork' = TRUE
               /\ activePipeline' = pipeline
               /\ boundPipelineForSlot' = [s \in Slots |-> None]
               /\ boundVertexBuffers' = [v \in VertexSlots |-> FALSE]
               /\ indexBufferBound' = FALSE
               /\ blendConstantsRequired' = (pipeline = 1)
               /\ stencilReferenceRequired' = (pipeline = 1)
               /\ stencilReferenceSet' = FALSE
               /\ UNCHANGED <<
                    logicalPipelineAlive, physicalPipelineAlive, inFlightReferences,
                    inFlight, renderSignature, viewportSet, scissorSet,
                    blendConstantsSet, draws
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

bindSet(slot) ==
    /\ ~inFlight
    /\ activePipeline \in Pipelines
    /\ RequiredSet(activePipeline, slot)
    /\ boundPipelineForSlot' = [boundPipelineForSlot EXCEPT ![slot] = activePipeline]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalPipelineAlive, physicalPipelineAlive, recordingReferences,
        inFlightReferences, recordedWork, inFlight, renderSignature,
        activePipeline, boundVertexBuffers, indexBufferBound, viewportSet,
        scissorSet, blendConstantsRequired, blendConstantsSet,
        stencilReferenceRequired, stencilReferenceSet, draws
       >>

bindVertexBuffer(slot) ==
    /\ ~inFlight
    /\ activePipeline \in Pipelines
    /\ RequiredVertexBuffer(activePipeline, slot)
    /\ boundVertexBuffers' = [boundVertexBuffers EXCEPT ![slot] = TRUE]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalPipelineAlive, physicalPipelineAlive, recordingReferences,
        inFlightReferences, recordedWork, inFlight, renderSignature,
        activePipeline, boundPipelineForSlot, indexBufferBound, viewportSet,
        scissorSet, blendConstantsRequired, blendConstantsSet,
        stencilReferenceRequired, stencilReferenceSet, draws
       >>

bindIndexBuffer ==
    /\ ~inFlight
    /\ activePipeline \in Pipelines
    /\ indexBufferBound' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalPipelineAlive, physicalPipelineAlive, recordingReferences,
        inFlightReferences, recordedWork, inFlight, renderSignature,
        activePipeline, boundPipelineForSlot, boundVertexBuffers, viewportSet,
        scissorSet, blendConstantsRequired, blendConstantsSet,
        stencilReferenceRequired, stencilReferenceSet, draws
       >>

setViewport ==
    /\ ~inFlight
    /\ renderSignature \in Signatures
    /\ viewportSet' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalPipelineAlive, physicalPipelineAlive, recordingReferences,
        inFlightReferences, recordedWork, inFlight, renderSignature,
        activePipeline, boundPipelineForSlot, boundVertexBuffers,
        indexBufferBound, scissorSet, blendConstantsRequired,
        blendConstantsSet, stencilReferenceRequired, stencilReferenceSet, draws
       >>

setScissor ==
    /\ ~inFlight
    /\ renderSignature \in Signatures
    /\ scissorSet' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalPipelineAlive, physicalPipelineAlive, recordingReferences,
        inFlightReferences, recordedWork, inFlight, renderSignature,
        activePipeline, boundPipelineForSlot, boundVertexBuffers,
        indexBufferBound, viewportSet, blendConstantsRequired,
        blendConstantsSet, stencilReferenceRequired, stencilReferenceSet, draws
       >>

setBlendConstants ==
    /\ ~inFlight
    /\ renderSignature \in Signatures
    /\ blendConstantsSet' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalPipelineAlive, physicalPipelineAlive, recordingReferences,
        inFlightReferences, recordedWork, inFlight, renderSignature,
        activePipeline, boundPipelineForSlot, boundVertexBuffers,
        indexBufferBound, viewportSet, scissorSet, blendConstantsRequired,
        stencilReferenceRequired, stencilReferenceSet, draws
       >>

setStencilReference ==
    /\ ~inFlight
    /\ renderSignature \in Signatures
    /\ stencilReferenceSet' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalPipelineAlive, physicalPipelineAlive, recordingReferences,
        inFlightReferences, recordedWork, inFlight, renderSignature,
        activePipeline, boundPipelineForSlot, boundVertexBuffers,
        indexBufferBound, viewportSet, scissorSet, blendConstantsRequired,
        blendConstantsSet, stencilReferenceRequired, draws
       >>

attemptDraw(indexed) ==
    LET accepted == GraphicsStateComplete(indexed)
    IN /\ ~inFlight
       /\ draws < MaxDraws
       /\ IF accepted
          THEN /\ draws' = draws + 1
               /\ recordedWork' = TRUE
               /\ UNCHANGED <<
                    logicalPipelineAlive, physicalPipelineAlive,
                    recordingReferences, inFlightReferences, inFlight,
                    renderSignature, activePipeline, boundPipelineForSlot,
                    boundVertexBuffers, indexBufferBound, viewportSet,
                    scissorSet, blendConstantsRequired, blendConstantsSet,
                    stencilReferenceRequired, stencilReferenceSet
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

submit ==
    /\ ~inFlight
    /\ renderSignature = None
    /\ recordedWork
    /\ recordingReferences' = {}
    /\ inFlightReferences' = recordingReferences
    /\ recordedWork' = FALSE
    /\ inFlight' = TRUE
    /\ activePipeline' = None
    /\ boundPipelineForSlot' = [s \in Slots |-> None]
    /\ boundVertexBuffers' = [v \in VertexSlots |-> FALSE]
    /\ indexBufferBound' = FALSE
    /\ viewportSet' = FALSE
    /\ scissorSet' = FALSE
    /\ blendConstantsRequired' = FALSE
    /\ blendConstantsSet' = FALSE
    /\ stencilReferenceRequired' = FALSE
    /\ stencilReferenceSet' = FALSE
    /\ draws' = 0
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalPipelineAlive, physicalPipelineAlive, renderSignature>>

complete ==
    /\ inFlight
    /\ physicalPipelineAlive' =
        [p \in Pipelines |-> logicalPipelineAlive[p] \/ p \in recordingReferences]
    /\ inFlightReferences' = {}
    /\ inFlight' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalPipelineAlive, recordingReferences, recordedWork,
        renderSignature, activePipeline, boundPipelineForSlot,
        boundVertexBuffers, indexBufferBound, viewportSet, scissorSet,
        blendConstantsRequired, blendConstantsSet, stencilReferenceRequired,
        stencilReferenceSet, draws
       >>

abandon ==
    /\ ~inFlight
    /\ recordedWork
    /\ physicalPipelineAlive' =
        [p \in Pipelines |-> logicalPipelineAlive[p] \/ p \in inFlightReferences]
    /\ recordingReferences' = {}
    /\ recordedWork' = FALSE
    /\ renderSignature' = None
    /\ activePipeline' = None
    /\ boundPipelineForSlot' = [s \in Slots |-> None]
    /\ boundVertexBuffers' = [v \in VertexSlots |-> FALSE]
    /\ indexBufferBound' = FALSE
    /\ viewportSet' = FALSE
    /\ scissorSet' = FALSE
    /\ blendConstantsRequired' = FALSE
    /\ blendConstantsSet' = FALSE
    /\ stencilReferenceRequired' = FALSE
    /\ stencilReferenceSet' = FALSE
    /\ draws' = 0
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalPipelineAlive, inFlightReferences, inFlight>>

Next ==
    \/ \E p \in Pipelines :
        \/ createPipeline(p)
        \/ destroyPipeline(p)
        \/ attemptBindPipeline(p)
    \/ \E sig \in Signatures : beginRendering(sig)
    \/ endRendering
    \/ \E s \in Slots : bindSet(s)
    \/ \E v \in VertexSlots : bindVertexBuffer(v)
    \/ bindIndexBuffer
    \/ setViewport
    \/ setScissor
    \/ setBlendConstants
    \/ setStencilReference
    \/ \E idx \in BOOLEAN : attemptDraw(idx)
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ logicalPipelineAlive \in [Pipelines -> BOOLEAN]
    /\ physicalPipelineAlive \in [Pipelines -> BOOLEAN]
    /\ recordingReferences \in SUBSET Pipelines
    /\ inFlightReferences \in SUBSET Pipelines
    /\ recordedWork \in BOOLEAN
    /\ inFlight \in BOOLEAN
    /\ renderSignature \in {None} \cup Signatures
    /\ activePipeline \in {None} \cup Pipelines
    /\ boundPipelineForSlot \in [Slots -> {None} \cup Pipelines]
    /\ boundVertexBuffers \in [VertexSlots -> BOOLEAN]
    /\ indexBufferBound \in BOOLEAN
    /\ viewportSet \in BOOLEAN
    /\ scissorSet \in BOOLEAN
    /\ blendConstantsRequired \in BOOLEAN
    /\ blendConstantsSet \in BOOLEAN
    /\ stencilReferenceRequired \in BOOLEAN
    /\ stencilReferenceSet \in BOOLEAN
    /\ draws \in 0..MaxDraws
    /\ lastOutcome \in Outcomes

PipelinePhysicalLifetimeExact ==
    \A p \in Pipelines :
        physicalPipelineAlive[p] = (logicalPipelineAlive[p] \/ Retained(p))

ActivePipelineMatchesRendering ==
    activePipeline \in Pipelines =>
        /\ renderSignature = PipelineSignature(activePipeline)
        /\ activePipeline \in recordingReferences
        /\ recordedWork

SlotsScopedToActivePipeline ==
    \A s \in Slots :
        boundPipelineForSlot[s] # None => boundPipelineForSlot[s] = activePipeline

ClosedRenderingClearsGraphicsState ==
    renderSignature = None =>
        /\ activePipeline = None
        /\ ~indexBufferBound
        /\ ~viewportSet
        /\ ~scissorSet
        /\ ~blendConstantsRequired
        /\ ~blendConstantsSet
        /\ ~stencilReferenceRequired
        /\ ~stencilReferenceSet
        /\ \A s \in Slots : boundPipelineForSlot[s] = None
        /\ \A v \in VertexSlots : ~boundVertexBuffers[v]

SubmittedStateSeparated ==
    /\ (recordingReferences # {} \/ draws > 0) => recordedWork
    /\ inFlight =>
        /\ renderSignature = None
        /\ recordingReferences = {}
        /\ ~recordedWork
        /\ activePipeline = None
        /\ draws = 0

Safety ==
    /\ PipelinePhysicalLifetimeExact
    /\ ActivePipelineMatchesRendering
    /\ SlotsScopedToActivePipeline
    /\ ClosedRenderingClearsGraphicsState
    /\ SubmittedStateSeparated

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
        /\ (draws' > draws => GraphicsStateComplete(FALSE))
    ]_vars

=============================================================================
