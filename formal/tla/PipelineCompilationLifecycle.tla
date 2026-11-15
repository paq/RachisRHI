-------------------- MODULE PipelineCompilationLifecycle --------------------
(***************************************************************************)
(* Asynchronous compute/graphics pipeline-compilation request lifecycle,   *)
(* dependency retention, one-time wait consumption, state polling, and     *)
(* immediate abandonment with unclaimed native PSO destruction.            *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Pipeline.cpp (beginComputePipelineCompilation,       *)
(*     beginGraphicsPipelineCompilation, executeCompilation,               *)
(*     pipelineCompilationState, waitPipelineCompilation,                  *)
(*     destroyPipelineCompilation)                                         *)
(*   - src/Device.cpp (portable async compilation forwarding)              *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Models an async compilation slot that can run alongside already       *)
(*   published pipelines, covering success/failure waits, stale/consumed   *)
(*   handle rejection, abandonment while pending or ready, and background  *)
(*   materialization plus immediate destruction of an unclaimed native     *)
(*   PSO. Native shader compilation internals are abstracted.              *)
(***************************************************************************)

EXTENDS Naturals

Phases == {
  "idle", "pending", "succeeded", "failed",
  "abandonedPending", "abandonedCleanup"
}
MaxRequests == 2
Actions == {
  "init", "begin", "poll", "completeSuccess", "completeFailure",
  "waitSuccess", "waitFailure", "abandonPending", "abandonReady",
  "finishAbandonedFailure", "materializeAbandonedSuccess",
  "destroyUnclaimedNative", "destroyPublished"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  phase,                   \* Lifecycle state of the compilation slot
  requestAlive,            \* Public PipelineCompilationHandle still active
  handleIssued,            \* Whether a caller has a handle from a prior begin
  dependenciesRetained,    \* Shader/layout dependencies held by background job
  readyNativePipeline,     \* Native PSO awaiting wait/abandon on a ready request
  unclaimedNativePipeline, \* Native PSO created after abandonment awaiting teardown
  livePipelines,           \* Count of live published public PipelineHandles
  publicationCount,        \* Monotonic count of pipelines published via wait
  requestsStarted,         \* Monotonic count of compilation requests started
  lastAction,              \* Most recent transition label
  lastOutcome              \* Outcome of caller-facing operations

coreVars == <<
  phase, requestAlive, handleIssued, dependenciesRetained,
  readyNativePipeline, unclaimedNativePipeline,
  livePipelines, publicationCount, requestsStarted
>>

vars == <<
  phase, requestAlive, handleIssued, dependenciesRetained,
  readyNativePipeline, unclaimedNativePipeline,
  livePipelines, publicationCount, requestsStarted,
  lastAction, lastOutcome
>>

Init ==
  /\ phase = "idle"
  /\ requestAlive = FALSE
  /\ handleIssued = FALSE
  /\ dependenciesRetained = FALSE
  /\ readyNativePipeline = FALSE
  /\ unclaimedNativePipeline = FALSE
  /\ livePipelines = 0
  /\ publicationCount = 0
  /\ requestsStarted = 0
  /\ lastAction = "init"
  /\ lastOutcome = "none"

BeginCompilation ==
  /\ phase = "idle"
  /\ requestsStarted < MaxRequests
  /\ phase' = "pending"
  /\ requestAlive' = TRUE
  /\ handleIssued' = TRUE
  /\ dependenciesRetained' = TRUE
  /\ readyNativePipeline' = FALSE
  /\ unclaimedNativePipeline' = FALSE
  /\ requestsStarted' = requestsStarted + 1
  /\ UNCHANGED <<livePipelines, publicationCount>>
  /\ lastAction' = "begin"
  /\ lastOutcome' = "accepted"

AttemptPollState ==
  /\ handleIssued
  /\ UNCHANGED coreVars
  /\ lastAction' = "poll"
  /\ lastOutcome' = IF requestAlive THEN "accepted" ELSE "rejected"

CompleteSuccess ==
  /\ phase = "pending"
  /\ requestAlive
  /\ phase' = "succeeded"
  /\ dependenciesRetained' = FALSE
  /\ readyNativePipeline' = TRUE
  /\ UNCHANGED <<
       requestAlive, handleIssued, unclaimedNativePipeline,
       livePipelines, publicationCount, requestsStarted
     >>
  /\ lastAction' = "completeSuccess"
  /\ lastOutcome' = "accepted"

CompleteFailure ==
  /\ phase = "pending"
  /\ requestAlive
  /\ phase' = "failed"
  /\ dependenciesRetained' = FALSE
  /\ readyNativePipeline' = FALSE
  /\ UNCHANGED <<
       requestAlive, handleIssued, unclaimedNativePipeline,
       livePipelines, publicationCount, requestsStarted
     >>
  /\ lastAction' = "completeFailure"
  /\ lastOutcome' = "accepted"

AttemptWait ==
  /\ handleIssued
  /\ IF phase = "succeeded" /\ requestAlive
     THEN /\ phase' = "idle"
          /\ requestAlive' = FALSE
          /\ readyNativePipeline' = FALSE
          /\ livePipelines' = livePipelines + 1
          /\ publicationCount' = publicationCount + 1
          /\ UNCHANGED <<
               handleIssued, dependenciesRetained,
               unclaimedNativePipeline, requestsStarted
             >>
          /\ lastAction' = "waitSuccess"
          /\ lastOutcome' = "accepted"
     ELSE IF phase = "failed" /\ requestAlive
     THEN /\ phase' = "idle"
          /\ requestAlive' = FALSE
          /\ UNCHANGED <<
               handleIssued, dependenciesRetained, readyNativePipeline,
               unclaimedNativePipeline, livePipelines,
               publicationCount, requestsStarted
             >>
          /\ lastAction' = "waitFailure"
          /\ lastOutcome' = "accepted"
     ELSE /\ phase # "pending"
          /\ UNCHANGED coreVars
          /\ lastAction' = "waitFailure"
          /\ lastOutcome' = "rejected"

AttemptAbandon ==
  /\ handleIssued
  /\ IF phase = "pending" /\ requestAlive
     THEN /\ phase' = "abandonedPending"
          /\ requestAlive' = FALSE
          /\ UNCHANGED <<
               handleIssued, dependenciesRetained, readyNativePipeline,
               unclaimedNativePipeline, livePipelines,
               publicationCount, requestsStarted
             >>
          /\ lastAction' = "abandonPending"
          /\ lastOutcome' = "accepted"
     ELSE IF phase \in {"succeeded", "failed"} /\ requestAlive
     THEN /\ phase' = "idle"
          /\ requestAlive' = FALSE
          /\ readyNativePipeline' = FALSE
          /\ UNCHANGED <<
               handleIssued, dependenciesRetained, unclaimedNativePipeline,
               livePipelines, publicationCount, requestsStarted
             >>
          /\ lastAction' = "abandonReady"
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastAction' = "abandonReady"
          /\ lastOutcome' = "rejected"

\* Background worker finishes a failed compilation after caller abandonment.
FinishAbandonedFailure ==
  /\ phase = "abandonedPending"
  /\ ~requestAlive
  /\ phase' = "idle"
  /\ dependenciesRetained' = FALSE
  /\ UNCHANGED <<
       requestAlive, handleIssued, readyNativePipeline,
       unclaimedNativePipeline, livePipelines,
       publicationCount, requestsStarted
     >>
  /\ lastAction' = "finishAbandonedFailure"
  /\ lastOutcome' = "accepted"

\* Background worker finishes creating a native PSO after abandonment,
\* releasing job dependencies and staging the unclaimed PSO for immediate teardown.
MaterializeAbandonedSuccess ==
  /\ phase = "abandonedPending"
  /\ ~requestAlive
  /\ phase' = "abandonedCleanup"
  /\ dependenciesRetained' = FALSE
  /\ unclaimedNativePipeline' = TRUE
  /\ UNCHANGED <<
       requestAlive, handleIssued, readyNativePipeline,
       livePipelines, publicationCount, requestsStarted
     >>
  /\ lastAction' = "materializeAbandonedSuccess"
  /\ lastOutcome' = "accepted"

DestroyUnclaimedNative ==
  /\ phase = "abandonedCleanup"
  /\ unclaimedNativePipeline
  /\ phase' = "idle"
  /\ unclaimedNativePipeline' = FALSE
  /\ UNCHANGED <<
       requestAlive, handleIssued, dependenciesRetained,
       readyNativePipeline, livePipelines,
       publicationCount, requestsStarted
     >>
  /\ lastAction' = "destroyUnclaimedNative"
  /\ lastOutcome' = "accepted"

DestroyPublishedPipeline ==
  /\ livePipelines > 0
  /\ livePipelines' = livePipelines - 1
  /\ UNCHANGED <<
       phase, requestAlive, handleIssued, dependenciesRetained,
       readyNativePipeline, unclaimedNativePipeline,
       publicationCount, requestsStarted
     >>
  /\ lastAction' = "destroyPublished"
  /\ lastOutcome' = "accepted"

Next ==
  \/ BeginCompilation
  \/ AttemptPollState
  \/ CompleteSuccess
  \/ CompleteFailure
  \/ AttemptWait
  \/ AttemptAbandon
  \/ FinishAbandonedFailure
  \/ MaterializeAbandonedSuccess
  \/ DestroyUnclaimedNative
  \/ DestroyPublishedPipeline

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ phase \in Phases
  /\ requestAlive \in BOOLEAN
  /\ handleIssued \in BOOLEAN
  /\ dependenciesRetained \in BOOLEAN
  /\ readyNativePipeline \in BOOLEAN
  /\ unclaimedNativePipeline \in BOOLEAN
  /\ livePipelines \in 0..MaxRequests
  /\ publicationCount \in 0..MaxRequests
  /\ requestsStarted \in 0..MaxRequests
  /\ lastAction \in Actions
  /\ lastOutcome \in Outcomes

PublicRequestMatchesPhase ==
  requestAlive = (phase \in {"pending", "succeeded", "failed"})

DependenciesCoverPendingWork ==
  dependenciesRetained = (phase \in {"pending", "abandonedPending"})

ReadyNativeOwnedBySucceededRequest ==
  readyNativePipeline = (phase = "succeeded" /\ requestAlive)

UnclaimedNativeOnlyInAbandonedCleanup ==
  unclaimedNativePipeline =
    /\ phase = "abandonedCleanup"
    /\ ~requestAlive
    /\ ~dependenciesRetained
    /\ ~readyNativePipeline

IdleClearsTransientWork ==
  phase = "idle" =>
    /\ ~requestAlive
    /\ ~dependenciesRetained
    /\ ~readyNativePipeline
    /\ ~unclaimedNativePipeline

PublishedPipelinesBounded ==
  /\ livePipelines <= publicationCount
  /\ publicationCount <= requestsStarted

Safety ==
  /\ PublicRequestMatchesPhase
  /\ DependenciesCoverPendingWork
  /\ ReadyNativeOwnedBySucceededRequest
  /\ UnclaimedNativeOnlyInAbandonedCleanup
  /\ IdleClearsTransientWork
  /\ PublishedPipelinesBounded

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (publicationCount' >= publicationCount)
    /\ (requestsStarted' >= requestsStarted)
    /\ (publicationCount' > publicationCount <=>
          /\ lastAction' = "waitSuccess"
          /\ lastOutcome' = "accepted"
          /\ phase = "succeeded"
          /\ requestAlive
          /\ readyNativePipeline
          /\ livePipelines' = livePipelines + 1
          /\ publicationCount' = publicationCount + 1)
    /\ (livePipelines' > livePipelines => publicationCount' = publicationCount + 1)
    /\ (lastAction' \in {
          "abandonPending", "abandonReady",
          "finishAbandonedFailure", "materializeAbandonedSuccess",
          "destroyUnclaimedNative"
        } =>
          /\ livePipelines' = livePipelines
          /\ publicationCount' = publicationCount)
  ]_vars

=============================================================================
