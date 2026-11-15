----------------------- MODULE MetalCommandRetention ------------------------
(***************************************************************************)
(* Retained-pool command recording, encoder lifecycle, resource retention, *)
(* commit-feedback callback gating, and device teardown in the Metal 4     *)
(* backend (`src/Metal/MetalCommand.mm`, `src/Metal/MetalDevice.mm`,       *)
(* `src/Metal/MetalDiagnostics.mm`, and                                    *)
(* `src/Metal/MetalSubmissionFeedback.h`).                                 *)
(*                                                                         *)
(* Models one retained-pool command recording, one application resource,   *)
(* and one commit-feedback callback across recording (`record`), command   *)
(* reset/abandonment (`reset`), pre-commit batch validation failure        *)
(* (`reject`), submission (`submit`), commit-feedback execution            *)
(* (`callback`), timeline garbage collection (`complete`), logical         *)
(* resource destruction (`destroyResource`), and device teardown (`close`).*)
(*                                                                         *)
(* Design rules controlled by CONSTANTS (each TRUE in the design):         *)
(*   - CallbackChecksSink: `MetalDiagnosticSink::publish` checks           *)
(*     `sink->alive` under `sink->mutex` before invoking a user callback   *)
(*     (`MetalCommandRetention-callbacksink.cfg` sets FALSE).              *)
(*   - RejectKeepsRetainers: a failed `submit` batch rejects before        *)
(*     `takeRetained`, preserving the recording's retained resource lease  *)
(*     (`MetalCommandRetention-rejectretainers.cfg` sets FALSE).           *)
(*   - CloseEncoderOnSubmit: `endMetalRecording` closes open MTL4 encoders *)
(*     before `MTL4CommandQueue` commit                                    *)
(*     (`MetalCommandRetention-openencoder.cfg` sets FALSE).               *)
(*                                                                         *)
(* Abstracted: Metal driver internals, unretained command pools, and       *)
(* device-loss fault classification (covered by `MetalDeviceLoss.tla`).    *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    CallbackChecksSink,     \* TRUE: publish checks sinkAlive before user callback
    RejectKeepsRetainers,   \* TRUE: rejected submit keeps commandOwns
    CloseEncoderOnSubmit    \* TRUE: submit closes encoderOpen before commit

VARIABLES
    phase,              \* "empty", "recorded", "submitted", or "closed"
    encoderOpen,        \* TRUE while an MTL4 encoder is open on the recording
    logicalAlive,       \* TRUE while the public resource handle is live
    physicalAlive,      \* TRUE while the underlying MTLBuffer/MTLTexture lives
    commandOwns,        \* TRUE while the recording retains the resource
    pendingOwns,        \* TRUE while an uncollected submission retains it
    sinkAlive,          \* TRUE until ~MetalDevice disconnects MetalDiagnosticSink
    callbackPending,    \* TRUE while MTL4CommitFeedback has not yet completed
    userCallbackSafe,   \* FALSE if user callback ever ran after sinkAlive = FALSE
    lastSubmitOutcome   \* "none", "accepted", or "rejected"

vars == <<
    phase, encoderOpen, logicalAlive, physicalAlive, commandOwns, pendingOwns,
    sinkAlive, callbackPending, userCallbackSafe, lastSubmitOutcome
>>

Phases == {"empty", "recorded", "submitted", "closed"}
SubmitOutcomes == {"none", "accepted", "rejected"}

Init ==
    /\ phase = "empty"
    /\ encoderOpen = FALSE
    /\ logicalAlive = TRUE
    /\ physicalAlive = TRUE
    /\ commandOwns = FALSE
    /\ pendingOwns = FALSE
    /\ sinkAlive = TRUE
    /\ callbackPending = FALSE
    /\ userCallbackSafe = TRUE
    /\ lastSubmitOutcome = "none"

\* Opening a copy/barrier encoder retains the application resource.
record ==
    /\ phase = "empty"
    /\ sinkAlive
    /\ logicalAlive
    /\ physicalAlive
    /\ phase' = "recorded"
    /\ encoderOpen' = TRUE
    /\ physicalAlive' = TRUE
    /\ commandOwns' = TRUE
    /\ lastSubmitOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, pendingOwns, sinkAlive, callbackPending, userCallbackSafe>>

\* resetCommandBuffer or abandoning the lease closes encoders and drops retainers.
reset ==
    /\ phase = "recorded"
    /\ sinkAlive
    /\ phase' = "empty"
    /\ encoderOpen' = FALSE
    /\ commandOwns' = FALSE
    /\ physicalAlive' = (logicalAlive \/ pendingOwns)
    /\ lastSubmitOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, pendingOwns, sinkAlive, callbackPending, userCallbackSafe>>

\* Batch validation failure in submit before takeRetained leaves recording intact.
reject ==
    /\ phase = "recorded"
    /\ sinkAlive
    /\ commandOwns' = IF RejectKeepsRetainers THEN commandOwns ELSE FALSE
    /\ physicalAlive' =
        IF RejectKeepsRetainers
        THEN physicalAlive
        ELSE (logicalAlive \/ pendingOwns)
    /\ lastSubmitOutcome' = "rejected"
    /\ UNCHANGED <<
        phase, encoderOpen, logicalAlive, pendingOwns,
        sinkAlive, callbackPending, userCallbackSafe
       >>

\* Successful submit closes encoders, moves retainers to pending, and registers feedback.
submit ==
    /\ phase = "recorded"
    /\ sinkAlive
    /\ ~callbackPending
    /\ phase' = "submitted"
    /\ encoderOpen' = IF CloseEncoderOnSubmit THEN FALSE ELSE encoderOpen
    /\ physicalAlive' = TRUE
    /\ commandOwns' = FALSE
    /\ pendingOwns' = TRUE
    /\ callbackPending' = TRUE
    /\ lastSubmitOutcome' = "accepted"
    /\ UNCHANGED <<logicalAlive, sinkAlive, userCallbackSafe>>

\* collectMetalQueue retires pending entries only after feedback has completed.
complete ==
    /\ phase = "submitted"
    /\ sinkAlive
    /\ ~callbackPending
    /\ phase' = "empty"
    /\ encoderOpen' = FALSE
    /\ commandOwns' = FALSE
    /\ pendingOwns' = FALSE
    /\ physicalAlive' = logicalAlive
    /\ UNCHANGED <<logicalAlive, sinkAlive, callbackPending, userCallbackSafe, lastSubmitOutcome>>

\* MTL4CommitFeedback handler invokes MetalDiagnosticSink::publish.
callback ==
    /\ callbackPending
    /\ LET invokeUser == IF CallbackChecksSink THEN sinkAlive ELSE TRUE
       IN userCallbackSafe' = (userCallbackSafe /\ (invokeUser => sinkAlive))
    /\ callbackPending' = FALSE
    /\ UNCHANGED <<
        phase, encoderOpen, logicalAlive, physicalAlive,
        commandOwns, pendingOwns, sinkAlive, lastSubmitOutcome
       >>

\* Destroying the public handle keeps native storage alive while retained.
destroyResource ==
    /\ logicalAlive
    /\ sinkAlive
    /\ logicalAlive' = FALSE
    /\ physicalAlive' = (commandOwns \/ pendingOwns)
    /\ UNCHANGED <<
        phase, encoderOpen, commandOwns, pendingOwns,
        sinkAlive, callbackPending, userCallbackSafe, lastSubmitOutcome
       >>

\* ~MetalDevice drains or aborts queue state, then disconnects sink->alive.
close ==
    /\ sinkAlive
    /\ phase' = "closed"
    /\ encoderOpen' = FALSE
    /\ commandOwns' = FALSE
    /\ pendingOwns' = FALSE
    /\ physicalAlive' = logicalAlive
    /\ sinkAlive' = FALSE
    /\ lastSubmitOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, callbackPending, userCallbackSafe>>

Next ==
    \/ record
    \/ reset
    \/ reject
    \/ submit
    \/ complete
    \/ callback
    \/ destroyResource
    \/ close

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ CallbackChecksSink \in BOOLEAN
    /\ RejectKeepsRetainers \in BOOLEAN
    /\ CloseEncoderOnSubmit \in BOOLEAN
    /\ phase \in Phases
    /\ encoderOpen \in BOOLEAN
    /\ logicalAlive \in BOOLEAN
    /\ physicalAlive \in BOOLEAN
    /\ commandOwns \in BOOLEAN
    /\ pendingOwns \in BOOLEAN
    /\ sinkAlive \in BOOLEAN
    /\ callbackPending \in BOOLEAN
    /\ userCallbackSafe \in BOOLEAN
    /\ lastSubmitOutcome \in SubmitOutcomes

CallbackRequiresLiveSink ==
    userCallbackSafe

RecordedRetainsResources ==
    phase = "recorded" => commandOwns /\ encoderOpen /\ physicalAlive

SubmittedClosesEncoder ==
    phase = "submitted" => ~encoderOpen /\ ~commandOwns /\ pendingOwns /\ physicalAlive

PhysicalLifetimeExact ==
    physicalAlive = (logicalAlive \/ commandOwns \/ pendingOwns)

EmptyAndClosedReleaseRetention ==
    /\ phase = "empty" => ~encoderOpen /\ ~commandOwns /\ ~pendingOwns
    /\ (phase = "closed") = ~sinkAlive
    /\ phase = "closed" => ~encoderOpen /\ ~commandOwns /\ ~pendingOwns

Safety ==
    /\ CallbackRequiresLiveSink
    /\ RecordedRetainsResources
    /\ SubmittedClosesEncoder
    /\ PhysicalLifetimeExact
    /\ EmptyAndClosedReleaseRetention

=============================================================================
