------------------------ MODULE DeviceTokenLifecycle ------------------------
(***************************************************************************)
(* Monotonic non-zero device identity issuance, zero-sentinel terminal     *)
(* exhaustion, early invalid-descriptor rejection, burned unpublished      *)
(* native-init tokens, and cross-generation command/timeline provenance.   *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/Common/MonotonicIdentitySource.h                                *)
(*     (MonotonicIdentitySource::acquire)                                  *)
(*   - src/Device.cpp (detail::deviceTokenSource process-wide identity     *)
(*     source, createDevice)                                               *)
(*   - src/BackendFactory.cpp (createBackendDevice backend dispatch and    *)
(*     rejection of D3D12-only options on other backends)                  *)
(*   - src/D3D12/D3D12Device.cpp (d3d12::createDevice: enableValidation,   *)
(*     selectAdapter, and native device creation precede                   *)
(*     D3D12Device::initializeQueues, which acquires the token)            *)
(*   - src/D3D12/D3D12Queue.cpp (D3D12CommandBuffer::deviceToken,          *)
(*     requireLiveD3D12Command, QueueManager::submit, and                  *)
(*     validateSubmittedPoint for timeline wait/query)                     *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - MonotonicIdentitySource starts at 1, issues each identity in        *)
(*     1..MaxToken at most once, and wraps nextToken to 0 as its terminal  *)
(*     exhausted sentinel after issuing MaxToken                           *)
(*     (MonotonicIdentitySource.h).                                        *)
(*   - Invalid DeviceDesc rejection in BackendFactory.cpp and in           *)
(*     d3d12::createDevice happens before initializeQueues, so it never    *)
(*     consumes nextToken; exhausted token space (nextToken = 0) also      *)
(*     rejects atomically with ErrorCode::resourceExhausted.               *)
(*   - Native initialization failure after identity acquisition            *)
(*     (QueueManager::initialize or initializeQueries) burns the           *)
(*     unpublished token (when AdvanceTokenOnDestroyAndFailure = TRUE),    *)
(*     and device destruction never recycles a published token.            *)
(*   - Command buffers and timeline points carry their creator's runtime   *)
(*     device token; runtime token comparison accepts an artifact iff it   *)
(*     originated from the exact live device instance.                     *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Full 64-bit UINT64_MAX token range (finitely bounded to             *)
(*     MaxToken = 3 to exhaustively exercise exhaustion and reuse paths).  *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT AdvanceTokenOnDestroyAndFailure

ASSUME AdvanceTokenOnDestroyAndFailure \in BOOLEAN

Slots == {"left", "right"}
MaxToken == 3
ExhaustedToken == 0
Tokens == 1..MaxToken
MaxInstance == 3
Instances == 1..MaxInstance
ArtifactKinds == {"command", "timeline"}
Outcomes == {"none", "accepted", "rejected", "invalidDesc", "exhausted", "initFailed"}

VARIABLES
    live,               \* [Slots -> BOOLEAN]
    slotToken,          \* [Slots -> 0..MaxToken] runtime token on live slot
    slotInstance,       \* [Slots -> 0..MaxInstance] ground-truth instance ID
    nextToken,          \* 0..MaxToken (0 = terminal exhausted sentinel)
    nextInstance,       \* 1..(MaxInstance + 1)
    instToken,          \* [Instances -> 0..MaxToken] token of each instance
    burnedTokens,       \* SUBSET Tokens: unpublished failed-init tokens
    cmdIssued,          \* [Instances -> BOOLEAN] command buffer recorded
    timelineIssued,     \* [Instances -> BOOLEAN] timeline point issued
    lastTarget,         \* Slots
    lastArtifactInst,   \* 0..MaxInstance
    lastOutcome         \* Outcomes

coreVars == <<live, slotToken, slotInstance, nextToken, nextInstance,
              instToken, burnedTokens, cmdIssued, timelineIssued>>
vars == <<live, slotToken, slotInstance, nextToken, nextInstance,
          instToken, burnedTokens, cmdIssued, timelineIssued,
          lastTarget, lastArtifactInst, lastOutcome>>

SuccessorToken(tok) ==
    IF tok = MaxToken THEN ExhaustedToken ELSE tok + 1

ArtifactExists(inst, kind) ==
    IF kind = "command" THEN cmdIssued[inst] ELSE timelineIssued[inst]

TypeOK ==
    /\ live \in [Slots -> BOOLEAN]
    /\ slotToken \in [Slots -> 0..MaxToken]
    /\ slotInstance \in [Slots -> 0..MaxInstance]
    /\ nextToken \in 0..MaxToken
    /\ nextInstance \in 1..(MaxInstance + 1)
    /\ instToken \in [Instances -> 0..MaxToken]
    /\ burnedTokens \subseteq Tokens
    /\ cmdIssued \in [Instances -> BOOLEAN]
    /\ timelineIssued \in [Instances -> BOOLEAN]
    /\ lastTarget \in Slots
    /\ lastArtifactInst \in 0..MaxInstance
    /\ lastOutcome \in Outcomes

Init ==
    /\ live = [s \in Slots |-> FALSE]
    /\ slotToken = [s \in Slots |-> 0]
    /\ slotInstance = [s \in Slots |-> 0]
    /\ nextToken = 1
    /\ nextInstance = 1
    /\ instToken = [i \in Instances |-> 0]
    /\ burnedTokens = {}
    /\ cmdIssued = [i \in Instances |-> FALSE]
    /\ timelineIssued = [i \in Instances |-> FALSE]
    /\ lastTarget = "left"
    /\ lastArtifactInst = 0
    /\ lastOutcome = "none"

RejectInvalidDescriptor(s) ==
    /\ ~live[s]
    /\ lastTarget' = s
    /\ lastArtifactInst' = 0
    /\ lastOutcome' = "invalidDesc"
    /\ UNCHANGED coreVars

RejectExhaustedCreation(s) ==
    /\ ~live[s]
    /\ nextToken = ExhaustedToken
    /\ lastTarget' = s
    /\ lastArtifactInst' = 0
    /\ lastOutcome' = "exhausted"
    /\ UNCHANGED coreVars

DropUnpublishedToken(s) ==
    /\ ~live[s]
    /\ nextToken # ExhaustedToken
    /\ burnedTokens' = burnedTokens \cup {nextToken}
    /\ nextToken' =
         IF AdvanceTokenOnDestroyAndFailure
         THEN SuccessorToken(nextToken)
         ELSE nextToken
    /\ lastTarget' = s
    /\ lastArtifactInst' = 0
    /\ lastOutcome' = "initFailed"
    /\ UNCHANGED <<live, slotToken, slotInstance, nextInstance, instToken,
                   cmdIssued, timelineIssued>>

Create(s) ==
    /\ ~live[s]
    /\ nextToken # ExhaustedToken
    /\ nextInstance <= MaxInstance
    /\ LET inst == nextInstance
           tok == nextToken
       IN /\ live' = [live EXCEPT ![s] = TRUE]
          /\ slotToken' = [slotToken EXCEPT ![s] = tok]
          /\ slotInstance' = [slotInstance EXCEPT ![s] = inst]
          /\ nextToken' = SuccessorToken(tok)
          /\ nextInstance' = inst + 1
          /\ instToken' = [instToken EXCEPT ![inst] = tok]
          /\ lastTarget' = s
          /\ lastArtifactInst' = 0
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<burnedTokens, cmdIssued, timelineIssued>>

Destroy(s) ==
    /\ live[s]
    /\ LET oldTok == slotToken[s] IN
       /\ live' = [live EXCEPT ![s] = FALSE]
       /\ slotToken' = [slotToken EXCEPT ![s] = 0]
       /\ slotInstance' = [slotInstance EXCEPT ![s] = 0]
       /\ nextToken' =
            IF AdvanceTokenOnDestroyAndFailure THEN nextToken ELSE oldTok
       /\ lastTarget' = s
       /\ lastArtifactInst' = 0
       /\ lastOutcome' = "none"
       /\ UNCHANGED <<nextInstance, instToken, burnedTokens,
                      cmdIssued, timelineIssued>>

IssueCommand(s) ==
    /\ live[s]
    /\ cmdIssued' = [cmdIssued EXCEPT ![slotInstance[s]] = TRUE]
    /\ lastTarget' = s
    /\ lastArtifactInst' = 0
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<live, slotToken, slotInstance, nextToken, nextInstance,
                   instToken, burnedTokens, timelineIssued>>

IssueTimeline(s) ==
    /\ live[s]
    /\ timelineIssued' = [timelineIssued EXCEPT ![slotInstance[s]] = TRUE]
    /\ lastTarget' = s
    /\ lastArtifactInst' = 0
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<live, slotToken, slotInstance, nextToken, nextInstance,
                   instToken, burnedTokens, cmdIssued>>

AttemptUse(target, inst, kind) ==
    /\ ArtifactExists(inst, kind)
    /\ LET accepted == live[target] /\ slotToken[target] = instToken[inst] IN
       /\ lastTarget' = target
       /\ lastArtifactInst' = inst
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
       /\ UNCHANGED coreVars

Next ==
    \/ \E s \in Slots :
         \/ RejectInvalidDescriptor(s)
         \/ RejectExhaustedCreation(s)
         \/ DropUnpublishedToken(s)
         \/ Create(s)
         \/ Destroy(s)
         \/ IssueCommand(s)
         \/ IssueTimeline(s)
    \/ \E target \in Slots, inst \in Instances, kind \in ArtifactKinds :
         AttemptUse(target, inst, kind)

Spec == Init /\ [][Next]_vars

LiveSlotsCarryInstanceTokens ==
    \A s \in Slots :
        IF live[s]
        THEN /\ slotInstance[s] \in Instances
             /\ slotToken[s] \in Tokens
             /\ slotToken[s] = instToken[slotInstance[s]]
        ELSE /\ slotInstance[s] = 0
             /\ slotToken[s] = 0

LiveSlotsNeverShareTokens ==
    \A s1, s2 \in Slots :
        (s1 # s2 /\ live[s1] /\ live[s2]) => slotToken[s1] # slotToken[s2]

\* Every created device instance owns a globally unique, unburned token, and
\* runtime token equality accepts only artifacts from the exact live instance.
UniqueDeviceProvenance ==
    /\ (\A i, j \in Instances :
          (i # j /\ instToken[i] # 0 /\ instToken[j] # 0) =>
              instToken[i] # instToken[j])
    /\ (\A i \in Instances : instToken[i] \notin burnedTokens)
    /\ (lastOutcome = "accepted" =>
          /\ live[lastTarget]
          /\ slotInstance[lastTarget] = lastArtifactInst)

Safety ==
    /\ LiveSlotsCarryInstanceTokens
    /\ LiveSlotsNeverShareTokens
    /\ UniqueDeviceProvenance

StepSafety ==
    [][lastOutcome' \in {"rejected", "invalidDesc", "exhausted"} =>
          coreVars' = coreVars]_vars
=============================================================================
