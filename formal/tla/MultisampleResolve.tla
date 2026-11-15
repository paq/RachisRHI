------------------------- MODULE MultisampleResolve -------------------------
(***************************************************************************)
(* Multisample color, depth, and packed depth-stencil resolve validation,  *)
(* independent per-plane accumulation, staged-recording rollback on        *)
(* abandonment vs queue-submission commit, failure atomicity, and resource *)
(* retention in the D3D12 backend: `ResourceManager::resolveTexture`       *)
(* (`validateTextureResolve`, `recordColorResolve`,                        *)
(* `recordPackedDepthStencilResolve`, and the D32 full-screen              *)
(* depth-resolve draw) in `src/D3D12/D3D12ResourceCopies.cpp`;             *)
(* multisampled attachment creation (`requireMultisampleSupport`) and the  *)
(* depth-resolve pipeline (`createDepthResolvePipeline`) in                *)
(* `src/D3D12/D3D12Resource.cpp`; multisampled 2D attachment views in      *)
(* `src/D3D12/D3D12View.cpp`; multisampled pipeline attachments            *)
(* (`validateAttachmentSampleCount`) in `src/D3D12/D3D12Pipeline.cpp`;     *)
(* explicit per-plane `resolveSource`/`resolveDestination` barriers in     *)
(* `src/D3D12/D3D12ResourceBarriers.cpp`; and dependency retention         *)
(* (`retainCommandDependencies`), abandonment, and submission in           *)
(* `src/D3D12/D3D12Queue.cpp` (`include/Rachis/Device.h`).                 *)
(*                                                                         *)
(* Models a multisampled attachment (`Source = 0`) resolved into a 1x      *)
(* target (`Destination = 1`) across color average (`1`), standalone       *)
(* depth min/max (`1`/`2`), and packed D24S8 depth min/max (`1`/`2`) and   *)
(* stencil min/max (`1`/`2`). Separate per-aspect destination planes prove *)
(* that packed depth and packed stencil resolves update only their target  *)
(* plane without erasing the sibling plane, unsubmitted recordings roll    *)
(* back on abandonment, invalid resolves fail atomically, and physical     *)
(* lifetimes match logical + command references.                           *)
(*                                                                         *)
(* Abstracted: sample positions and per-pixel filter arithmetic.           *)
(***************************************************************************)
EXTENDS Naturals

Resources == {0, 1}
Source == 0
Destination == 1
PlaneValues == {0, 1, 2}
PackedDepthOperations == {"packedDepthMinimum", "packedDepthMaximum"}
PackedStencilOperations == {"packedStencilMinimum", "packedStencilMaximum"}
ResolveOperations == {
    "colorAverage",
    "depthMinimum",
    "depthMaximum",
    "packedDepthMinimum",
    "packedDepthMaximum",
    "packedStencilMinimum",
    "packedStencilMaximum"
}
OptionalOperations == ResolveOperations \cup {"none"}
ResolveAspects == {"color", "depth", "stencil"}
Layouts == {"undefined", "renderTarget", "resolveSource", "resolveDestination"}
Outcomes == {"none", "accepted", "rejected"}

OperationAspect(operation) ==
    IF operation = "colorAverage"
    THEN "color"
    ELSE IF operation \in PackedStencilOperations
         THEN "stencil"
         ELSE "depth"

OperationValue(operation) ==
    IF operation \in {"depthMaximum", "packedDepthMaximum", "packedStencilMaximum"}
    THEN 2
    ELSE 1

VARIABLES
    logicalAlive,                \* public TextureHandle liveness per resource
    physicalAlive,               \* native texture liveness per resource
    layout,                      \* planned TextureLayout per resource in Layouts
    sourceDefined,               \* planned multisampled Source written flag
    destinationPlane,            \* planned per-aspect resolved value
    resolvedOperations,          \* planned resolve operations accepted so far
    resolvedAspects,             \* planned aspects resolved so far
    committedLayout,             \* committed GPU TextureLayout per resource
    committedSourceDefined,      \* committed GPU Source written flag
    committedDestinationPlane,   \* committed GPU per-aspect resolved value
    committedResolvedOperations, \* committed GPU resolve operations
    committedResolvedAspects,    \* committed GPU resolved aspects
    rendering,                   \* TRUE while a render pass is active
    recordingReferences,         \* resources retained by active recording
    inFlightReferences,          \* resources retained by in-flight submission
    inFlight,                    \* TRUE while command buffer is in flight
    lastOperation,               \* latest accepted resolve operation or "none"
    lastOutcome                  \* outcome of latest step in Outcomes

committedVars == <<
    committedLayout, committedSourceDefined, committedDestinationPlane,
    committedResolvedOperations, committedResolvedAspects
>>

coreVars == <<
    logicalAlive, physicalAlive, layout, sourceDefined, destinationPlane,
    resolvedOperations, resolvedAspects,
    committedLayout, committedSourceDefined, committedDestinationPlane,
    committedResolvedOperations, committedResolvedAspects,
    rendering, recordingReferences, inFlightReferences, inFlight
>>

vars == <<
    logicalAlive, physicalAlive, layout, sourceDefined, destinationPlane,
    resolvedOperations, resolvedAspects,
    committedLayout, committedSourceDefined, committedDestinationPlane,
    committedResolvedOperations, committedResolvedAspects,
    rendering, recordingReferences, inFlightReferences, inFlight,
    lastOperation, lastOutcome
>>

Referenced(resource) ==
    resource \in recordingReferences \/ resource \in inFlightReferences

ResolveValid(compatibleSamples, singleSubresource) ==
    /\ ~rendering
    /\ ~inFlight
    /\ logicalAlive[Source]
    /\ logicalAlive[Destination]
    /\ sourceDefined
    /\ layout[Source] = "resolveSource"
    /\ layout[Destination] = "resolveDestination"
    /\ compatibleSamples
    /\ singleSubresource

Init ==
    /\ logicalAlive = [r \in Resources |-> TRUE]
    /\ physicalAlive = [r \in Resources |-> TRUE]
    /\ layout = [r \in Resources |-> "undefined"]
    /\ sourceDefined = FALSE
    /\ destinationPlane = [a \in ResolveAspects |-> 0]
    /\ resolvedOperations = {}
    /\ resolvedAspects = {}
    /\ committedLayout = [r \in Resources |-> "undefined"]
    /\ committedSourceDefined = FALSE
    /\ committedDestinationPlane = [a \in ResolveAspects |-> 0]
    /\ committedResolvedOperations = {}
    /\ committedResolvedAspects = {}
    /\ rendering = FALSE
    /\ recordingReferences = {}
    /\ inFlightReferences = {}
    /\ inFlight = FALSE
    /\ lastOperation = "none"
    /\ lastOutcome = "none"

transitionSourceToRenderTarget ==
    /\ ~inFlight
    /\ ~rendering
    /\ logicalAlive[Source]
    /\ layout[Source] = "undefined"
    /\ layout' = [layout EXCEPT ![Source] = "renderTarget"]
    /\ recordingReferences' = recordingReferences \cup {Source}
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, physicalAlive, sourceDefined, destinationPlane,
        resolvedOperations, resolvedAspects, committedVars, rendering,
        inFlightReferences, inFlight, lastOperation
       >>

beginRendering ==
    /\ ~inFlight
    /\ ~rendering
    /\ logicalAlive[Source]
    /\ layout[Source] = "renderTarget"
    /\ rendering' = TRUE
    /\ recordingReferences' = recordingReferences \cup {Source}
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, physicalAlive, layout, sourceDefined,
        destinationPlane, resolvedOperations, resolvedAspects,
        committedVars, inFlightReferences, inFlight, lastOperation
       >>

renderAndEnd ==
    /\ rendering
    /\ sourceDefined' = TRUE
    /\ rendering' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, physicalAlive, layout, destinationPlane,
        resolvedOperations, resolvedAspects, committedVars,
        recordingReferences, inFlightReferences, inFlight, lastOperation
       >>

transitionForResolve ==
    /\ ~inFlight
    /\ ~rendering
    /\ logicalAlive[Source]
    /\ logicalAlive[Destination]
    /\ sourceDefined
    /\ layout[Source] = "renderTarget"
    /\ layout[Destination] = "undefined"
    /\ layout' =
        [[layout EXCEPT ![Source] = "resolveSource"]
                 EXCEPT ![Destination] = "resolveDestination"]
    /\ recordingReferences' = recordingReferences \cup Resources
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, physicalAlive, sourceDefined, destinationPlane,
        resolvedOperations, resolvedAspects, committedVars, rendering,
        inFlightReferences, inFlight, lastOperation
       >>

attemptResolve(operation, compatibleSamples, singleSubresource) ==
    LET accepted == ResolveValid(compatibleSamples, singleSubresource)
        aspect == OperationAspect(operation)
        value == OperationValue(operation)
    IN /\ ~inFlight
       /\ IF accepted
          THEN /\ destinationPlane' = [destinationPlane EXCEPT ![aspect] = value]
               /\ resolvedOperations' = resolvedOperations \cup {operation}
               /\ resolvedAspects' = resolvedAspects \cup {aspect}
               /\ recordingReferences' = recordingReferences \cup Resources
               /\ lastOperation' = operation
               /\ UNCHANGED <<
                    logicalAlive, physicalAlive, layout, sourceDefined,
                    committedVars, rendering, inFlightReferences, inFlight
                  >>
          ELSE /\ UNCHANGED coreVars
               /\ UNCHANGED lastOperation
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroy(resource) ==
    /\ logicalAlive[resource]
    /\ ~rendering
    /\ logicalAlive' = [logicalAlive EXCEPT ![resource] = FALSE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![resource] = Referenced(resource)]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        layout, sourceDefined, destinationPlane, resolvedOperations,
        resolvedAspects, committedVars, rendering, recordingReferences,
        inFlightReferences, inFlight, lastOperation
       >>

submit ==
    /\ ~inFlight
    /\ ~rendering
    /\ recordingReferences # {}
    /\ recordingReferences' = {}
    /\ inFlightReferences' = recordingReferences
    /\ committedLayout' = layout
    /\ committedSourceDefined' = sourceDefined
    /\ committedDestinationPlane' = destinationPlane
    /\ committedResolvedOperations' = resolvedOperations
    /\ committedResolvedAspects' = resolvedAspects
    /\ inFlight' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, physicalAlive, layout, sourceDefined,
        destinationPlane, resolvedOperations, resolvedAspects,
        rendering, lastOperation
       >>

complete ==
    /\ inFlight
    /\ physicalAlive' =
        [r \in Resources |-> logicalAlive[r] \/ r \in recordingReferences]
    /\ inFlightReferences' = {}
    /\ inFlight' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, layout, sourceDefined, destinationPlane,
        resolvedOperations, resolvedAspects, committedVars, rendering,
        recordingReferences, lastOperation
       >>

\* Abandoning an open recording discards unsubmitted barrier, rendering, and
\* resolve work, restoring the committed GPU state.
abandon ==
    /\ ~inFlight
    /\ ~rendering
    /\ recordingReferences # {}
    /\ physicalAlive' =
        [r \in Resources |-> logicalAlive[r] \/ r \in inFlightReferences]
    /\ layout' = committedLayout
    /\ sourceDefined' = committedSourceDefined
    /\ destinationPlane' = committedDestinationPlane
    /\ resolvedOperations' = committedResolvedOperations
    /\ resolvedAspects' = committedResolvedAspects
    /\ recordingReferences' = {}
    /\ lastOperation' = "none"
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, committedVars, rendering, inFlightReferences, inFlight
       >>

Next ==
    \/ transitionSourceToRenderTarget
    \/ beginRendering
    \/ renderAndEnd
    \/ transitionForResolve
    \/ \E op \in ResolveOperations, cs \in BOOLEAN, ss \in BOOLEAN :
        attemptResolve(op, cs, ss)
    \/ \E r \in Resources : destroy(r)
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ logicalAlive \in [Resources -> BOOLEAN]
    /\ physicalAlive \in [Resources -> BOOLEAN]
    /\ layout \in [Resources -> Layouts]
    /\ sourceDefined \in BOOLEAN
    /\ destinationPlane \in [ResolveAspects -> PlaneValues]
    /\ resolvedOperations \in SUBSET ResolveOperations
    /\ resolvedAspects \in SUBSET ResolveAspects
    /\ committedLayout \in [Resources -> Layouts]
    /\ committedSourceDefined \in BOOLEAN
    /\ committedDestinationPlane \in [ResolveAspects -> PlaneValues]
    /\ committedResolvedOperations \in SUBSET ResolveOperations
    /\ committedResolvedAspects \in SUBSET ResolveAspects
    /\ rendering \in BOOLEAN
    /\ recordingReferences \in SUBSET Resources
    /\ inFlightReferences \in SUBSET Resources
    /\ inFlight \in BOOLEAN
    /\ lastOperation \in OptionalOperations
    /\ lastOutcome \in Outcomes

UnrecordedStateMatchesCommitted ==
    (recordingReferences = {} /\ ~rendering) =>
        /\ layout = committedLayout
        /\ sourceDefined = committedSourceDefined
        /\ destinationPlane = committedDestinationPlane
        /\ resolvedOperations = committedResolvedOperations
        /\ resolvedAspects = committedResolvedAspects

ResolvedPlanesHaveRenderedProvenance ==
    /\ \A a \in ResolveAspects :
        (destinationPlane[a] > 0) = (a \in resolvedAspects)
    /\ (resolvedAspects # {} => sourceDefined)
    /\ \A a \in ResolveAspects :
        (committedDestinationPlane[a] > 0) = (a \in committedResolvedAspects)
    /\ (committedResolvedAspects # {} => committedSourceDefined)

ResolvedAspectsMatchOperations ==
    /\ resolvedAspects = {OperationAspect(op) : op \in resolvedOperations}
    /\ committedResolvedAspects =
        {OperationAspect(op) : op \in committedResolvedOperations}

PackedPlanesAccumulateIndependently ==
    /\ ((\E op \in PackedDepthOperations : op \in resolvedOperations) =>
            destinationPlane["depth"] \in {1, 2})
    /\ ((\E op \in PackedStencilOperations : op \in resolvedOperations) =>
            destinationPlane["stencil"] \in {1, 2})
    /\ (lastOutcome = "accepted" =>
            /\ lastOperation \in ResolveOperations
            /\ destinationPlane[OperationAspect(lastOperation)] =
                OperationValue(lastOperation))

ResourcePhysicalLifetimeExact ==
    \A r \in Resources :
        physicalAlive[r] = (logicalAlive[r] \/ Referenced(r))

Safety ==
    /\ UnrecordedStateMatchesCommitted
    /\ ResolvedPlanesHaveRenderedProvenance
    /\ ResolvedAspectsMatchOperations
    /\ PackedPlanesAccumulateIndependently
    /\ ResourcePhysicalLifetimeExact

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => UNCHANGED coreVars)
        /\ (lastOutcome' = "accepted" =>
                LET target == OperationAspect(lastOperation')
                IN /\ ResolveValid(TRUE, TRUE)
                   /\ UNCHANGED committedVars
                   /\ destinationPlane'[target] = OperationValue(lastOperation')
                   /\ \A a \in ResolveAspects :
                        a # target => destinationPlane'[a] = destinationPlane[a])
    ]_vars

=============================================================================
