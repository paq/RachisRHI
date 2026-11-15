-------------------------- MODULE CommandPoolReset --------------------------
(***************************************************************************)
(* Individual command-buffer reset opt-in, pool-wide recording reset       *)
(* pre-validation, active-scope protection, and submitted-content safety.  *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Queue.cpp (D3D12CommandBuffer::validateReset,        *)
(*     D3D12CommandBuffer::resetRecording, QueueManager::reset,            *)
(*     QueueManager::resetCommandPool)                                     *)
(*   - src/D3D12/D3D12RuntimePolicies.h                                    *)
(*     (isD3D12CommandPoolIdleFromInFlight, executeD3D12RecordingReset,    *)
(*     resetD3D12IdleCommandPoolEntries)                                   *)
(*   - src/Device.cpp (Device::resetCommandBuffer through                  *)
(*     detail::validateCommandBufferReadyForReset in                       *)
(*     src/Internal/PortableValidation.h)                                  *)
(*                                                                         *)
(* Contract & boundary note:                                               *)
(*   - Slots = {0, 1}: slot 0 belongs to an individual-reset-enabled pool  *)
(*     (CommandPoolFlag::allowIndividualReset), while slot 1 does not.     *)
(*   - Individual reset accepts iff the slot holds a live recording with   *)
(*     allowIndividualReset and all rendering/query/debug scopes closed;   *)
(*     acceptance preserves the open recording lease while discarding its  *)
(*     recorded content.                                                   *)
(*   - Pool-wide reset pre-validates that NO entry in the pool is in       *)
(*     flight (when BlockResetWhileInFlight = TRUE) and every active       *)
(*     recording has all scopes closed before resetting recordings.        *)
(*   - Collected GPU completions record observedContent and retiredPayload *)
(*     so resets on re-acquired slots never corrupt submitted or retired   *)
(*     payloads.                                                           *)
(*   - Native Close / Reset HRESULT failures during individual or          *)
(*     pool-wide reset clear retained recording state, detach the lease,   *)
(*     and permanently quarantine the failed slot while keeping remaining  *)
(*     healthy slots usable and preserving pool partition accounting.      *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT BlockResetWhileInFlight

ASSUME BlockResetWhileInFlight \in BOOLEAN

Slots == {0, 1}
Phases == {"available", "recording", "inFlight", "quarantined"}
MaxContent == 2
MaxTimeline == 3
AllowsIndividualReset == [s \in Slots |-> s = 0]
Outcomes == {"none", "accepted", "rejected", "nativeFailed"}

VARIABLES
    phase,              \* [Slots -> Phases] entry lifecycle state
    content,            \* [Slots -> 0..MaxContent] commands in entry
    submittedContent,   \* [Slots -> 0..MaxContent] expected payload of submission
    observedContent,    \* [Slots -> 0..MaxContent] GPU-observed payload at Collect
    hasCollected,       \* [Slots -> BOOLEAN] whether slot has retired at least once
    lastRetiredPayload, \* [Slots -> 0..MaxContent] expected payload of last retired submission
    openScope,          \* [Slots -> BOOLEAN] active rendering/query/debug scope
    retireAt,           \* [Slots -> 0..MaxTimeline] submission retirement point
    submitted,          \* queue submitted timeline frontier
    completed,          \* queue completed timeline frontier
    lastOutcome         \* outcome of the most recent reset attempt

coreVars == <<phase, content, submittedContent, observedContent,
              hasCollected, lastRetiredPayload, openScope, retireAt,
              submitted, completed>>
vars == <<phase, content, submittedContent, observedContent,
          hasCollected, lastRetiredPayload, openScope, retireAt,
          submitted, completed, lastOutcome>>

IndividualResetValid(s) ==
    /\ phase[s] = "recording"
    /\ AllowsIndividualReset[s]
    /\ ~openScope[s]

PoolResetValid ==
    \A s \in Slots :
        /\ (BlockResetWhileInFlight => phase[s] # "inFlight")
        /\ (phase[s] = "recording" => ~openScope[s])

TypeOK ==
    /\ phase \in [Slots -> Phases]
    /\ content \in [Slots -> 0..MaxContent]
    /\ submittedContent \in [Slots -> 0..MaxContent]
    /\ observedContent \in [Slots -> 0..MaxContent]
    /\ hasCollected \in [Slots -> BOOLEAN]
    /\ lastRetiredPayload \in [Slots -> 0..MaxContent]
    /\ openScope \in [Slots -> BOOLEAN]
    /\ retireAt \in [Slots -> 0..MaxTimeline]
    /\ submitted \in 0..MaxTimeline
    /\ completed \in 0..MaxTimeline
    /\ completed <= submitted
    /\ lastOutcome \in Outcomes

Init ==
    /\ phase = [s \in Slots |-> "available"]
    /\ content = [s \in Slots |-> 0]
    /\ submittedContent = [s \in Slots |-> 0]
    /\ observedContent = [s \in Slots |-> 0]
    /\ hasCollected = [s \in Slots |-> FALSE]
    /\ lastRetiredPayload = [s \in Slots |-> 0]
    /\ openScope = [s \in Slots |-> FALSE]
    /\ retireAt = [s \in Slots |-> 0]
    /\ submitted = 0
    /\ completed = 0
    /\ lastOutcome = "none"

Acquire(s) ==
    /\ phase[s] = "available"
    /\ phase' = [phase EXCEPT ![s] = "recording"]
    /\ content' = [content EXCEPT ![s] = 0]
    /\ openScope' = [openScope EXCEPT ![s] = FALSE]
    /\ retireAt' = [retireAt EXCEPT ![s] = 0]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<submittedContent, observedContent, hasCollected,
                   lastRetiredPayload, submitted, completed>>

Record(s) ==
    /\ phase[s] = "recording"
    /\ ~openScope[s]
    /\ content[s] < MaxContent
    /\ content' = [content EXCEPT ![s] = content[s] + 1]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<phase, submittedContent, observedContent, hasCollected,
                   lastRetiredPayload, openScope, retireAt, submitted, completed>>

BeginScope(s) ==
    /\ phase[s] = "recording"
    /\ ~openScope[s]
    /\ openScope' = [openScope EXCEPT ![s] = TRUE]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<phase, content, submittedContent, observedContent,
                   hasCollected, lastRetiredPayload, retireAt, submitted, completed>>

EndScope(s) ==
    /\ phase[s] = "recording"
    /\ openScope[s]
    /\ openScope' = [openScope EXCEPT ![s] = FALSE]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<phase, content, submittedContent, observedContent,
                   hasCollected, lastRetiredPayload, retireAt, submitted, completed>>

AttemptResetIndividual(s) ==
    IF IndividualResetValid(s)
    THEN /\ content' = [content EXCEPT ![s] = 0]
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<phase, submittedContent, observedContent,
                        hasCollected, lastRetiredPayload, openScope,
                        retireAt, submitted, completed>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

AttemptResetIndividualNativeFailure(s) ==
    /\ IndividualResetValid(s)
    /\ phase' = [phase EXCEPT ![s] = "quarantined"]
    /\ content' = [content EXCEPT ![s] = 0]
    /\ openScope' = [openScope EXCEPT ![s] = FALSE]
    /\ retireAt' = [retireAt EXCEPT ![s] = 0]
    /\ lastOutcome' = "nativeFailed"
    /\ UNCHANGED <<submittedContent, observedContent, hasCollected,
                   lastRetiredPayload, submitted, completed>>

AttemptResetPool ==
    IF PoolResetValid
    THEN /\ content' =
              [s \in Slots |->
                 IF phase[s] = "recording"
                    \/ (~BlockResetWhileInFlight /\ phase[s] = "inFlight")
                 THEN 0
                 ELSE content[s]]
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<phase, submittedContent, observedContent,
                        hasCollected, lastRetiredPayload, openScope,
                        retireAt, submitted, completed>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

AttemptResetPoolNativeFailure(failSlot) ==
    /\ PoolResetValid
    /\ phase[failSlot] \in {"recording", "available"}
    /\ phase' = [phase EXCEPT ![failSlot] = "quarantined"]
    /\ content' =
          [s \in Slots |->
             IF s = failSlot
             THEN 0
             ELSE IF phase[failSlot] = "recording"
                  THEN IF phase[s] = "recording" /\ s < failSlot THEN 0 ELSE content[s]
                  ELSE IF phase[s] = "recording"
                          \/ (~BlockResetWhileInFlight /\ phase[s] = "inFlight")
                       THEN 0
                       ELSE content[s]]
    /\ lastOutcome' = "nativeFailed"
    /\ UNCHANGED <<submittedContent, observedContent, hasCollected,
                   lastRetiredPayload, openScope, retireAt, submitted, completed>>

SubmitSlot(s) ==
    /\ phase[s] = "recording"
    /\ ~openScope[s]
    /\ submitted < MaxTimeline
    /\ phase' = [phase EXCEPT ![s] = "inFlight"]
    /\ submittedContent' = [submittedContent EXCEPT ![s] = content[s]]
    /\ retireAt' = [retireAt EXCEPT ![s] = submitted + 1]
    /\ submitted' = submitted + 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<content, observedContent, hasCollected,
                   lastRetiredPayload, openScope, completed>>

Complete ==
    /\ completed < submitted
    /\ completed' = completed + 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<phase, content, submittedContent, observedContent,
                   hasCollected, lastRetiredPayload, openScope, retireAt, submitted>>

\* GPU executes the allocator contents (`content[s]`) while in flight and
\* retires at `Collect`, where `observedContent[s]` records what actually ran.
Collect(s) ==
    /\ phase[s] = "inFlight"
    /\ retireAt[s] <= completed
    /\ phase' = [phase EXCEPT ![s] = "available"]
    /\ observedContent' = [observedContent EXCEPT ![s] = content[s]]
    /\ hasCollected' = [hasCollected EXCEPT ![s] = TRUE]
    /\ lastRetiredPayload' = [lastRetiredPayload EXCEPT ![s] = submittedContent[s]]
    /\ retireAt' = [retireAt EXCEPT ![s] = 0]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<content, submittedContent, openScope, submitted, completed>>

Next ==
    \/ \E s \in Slots :
          \/ Acquire(s)
          \/ Record(s)
          \/ BeginScope(s)
          \/ EndScope(s)
          \/ AttemptResetIndividual(s)
          \/ AttemptResetIndividualNativeFailure(s)
          \/ AttemptResetPoolNativeFailure(s)
          \/ SubmitSlot(s)
          \/ Collect(s)
    \/ AttemptResetPool
    \/ Complete

Spec == Init /\ [][Next]_vars

NonRecordingSlotsHaveClosedScopes ==
    \A s \in Slots : phase[s] # "recording" => ~openScope[s]

QuarantinedSlotsStayCleanAndDetached ==
    \A s \in Slots :
        phase[s] = "quarantined" =>
            /\ content[s] = 0
            /\ ~openScope[s]
            /\ retireAt[s] = 0

\* In-flight command allocators retain their exact submitted commands without
\* interference from individual or pool-wide resets.
SubmittedContentProtectedFromReset ==
    \A s \in Slots :
        phase[s] = "inFlight" =>
            /\ retireAt[s] > 0
            /\ retireAt[s] <= submitted
            /\ content[s] = submittedContent[s]

\* Retired GPU executions always observe the exact payload that was submitted,
\* even if the slot was subsequently re-acquired and reset.
ObservedContentMatchesRetiredSubmission ==
    \A s \in Slots :
        hasCollected[s] => observedContent[s] = lastRetiredPayload[s]

Safety ==
    /\ NonRecordingSlotsHaveClosedScopes
    /\ QuarantinedSlotsStayCleanAndDetached
    /\ SubmittedContentProtectedFromReset
    /\ ObservedContentMatchesRetiredSubmission

StepSafety ==
    [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
       /\ (\A s \in Slots :
             phase[s] = "quarantined" => phase'[s] = "quarantined")
       /\ (\A s \in Slots :
             (~AllowsIndividualReset[s] /\ phase[s] = "recording"
              /\ phase'[s] \in {"recording", "quarantined"}
              /\ (content'[s] < content[s] \/ phase'[s] = "quarantined")) =>
                \* Non-opt-in slot 1 can only be cleared or reset-quarantined by a valid pool-wide reset!
                (\A other \in Slots :
                    (BlockResetWhileInFlight => phase[other] # "inFlight")
                    /\ (phase[other] = "recording" => ~openScope[other])))
       /\ (\A s \in Slots :
             (phase[s] = "recording" /\ openScope[s]) =>
                /\ phase'[s] = "recording"
                /\ content'[s] = content[s])
       /\ submitted' >= submitted
       /\ completed' >= completed]_vars
=============================================================================
