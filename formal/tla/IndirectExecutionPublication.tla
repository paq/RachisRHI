-------------------- MODULE IndirectExecutionPublication --------------------
(***************************************************************************)
(* Transactional two-resource dependency publication and partial-prefix    *)
(* rollback shared by indirect compute, draw, indexed-draw, and mesh       *)
(* dispatch commands in `src/D3D12/D3D12CommandRecording.cpp`              *)
(* (`resolveIndirectCommand`, `retainIndirectBufferDependencies`, and      *)
(* `recordIndirectExecution`), `src/D3D12/D3D12Pipeline.cpp`               *)
(* (command-signature preparation), and `src/D3D12/D3D12Queue.cpp`         *)
(* (`QueueManager::retainCommandDependencies` rollback).                   *)
(*                                                                         *)
(* Models validation and command-signature preparation before dependency   *)
(* publication, zero-count dependency-free recording no-ops, tentative     *)
(* `Argument` dependency publication followed by optional distinct `Count` *)
(* dependency publication, exact rollback of `recordingDependencies` and   *)
(* `work` when `Count` retention fails (`failureStage = 1`), and command   *)
(* reset/retirement across repeated transactions.                          *)
(*                                                                         *)
(* Abstracted: byte-offset/stride arithmetic (covered by                   *)
(* `IndirectDispatchBatch.tla` and `IndirectDraw.tla`) and native          *)
(* `ExecuteIndirect` execution.                                            *)
(***************************************************************************)
EXTENDS Integers

Dependencies == {0, 1}
Argument == 0
Count == 1
FailureStages == {-1, 0, 1}
MaxWork == 2

VARIABLES
    recordingDependencies,          \* dependencies currently published on the recording
    work,                           \* count of committed indirect commands in the recording
    transactionActive,              \* TRUE after Argument prefix published, before Count step
    transactionUsesCount,           \* whether active transaction uses a count buffer
    transactionCountAliasesArgument,\* whether active transaction aliases Count to Argument
    transactionFailureStage,        \* injected failure stage in FailureStages
    transactionBeforeDependencies,  \* pre-transaction snapshot of recordingDependencies
    transactionBeforeWork,          \* pre-transaction snapshot of work
    lastAttemptObserved,            \* TRUE when a completed attempt is ready to inspect
    lastRequestValid,               \* descriptor validity of latest completed attempt
    lastZeroCount,                  \* whether latest completed attempt had maxCount = 0
    lastUsesCount,                  \* whether latest completed attempt used Count
    lastCountAliasesArgument,       \* whether latest completed attempt aliased Count
    lastPreparationSucceeded,       \* whether command-signature preparation succeeded
    lastFailureStage,               \* failure stage of latest completed attempt
    lastAccepted,                   \* whether latest completed attempt committed
    lastBeforeDependencies,         \* pre-attempt dependencies of latest completed attempt
    lastPrefixDependencies,         \* mid-transaction prefix dependencies observed
    lastBeforeWork                  \* pre-attempt work of latest completed attempt

vars == <<
    recordingDependencies, work, transactionActive, transactionUsesCount,
    transactionCountAliasesArgument, transactionFailureStage,
    transactionBeforeDependencies, transactionBeforeWork,
    lastAttemptObserved, lastRequestValid, lastZeroCount, lastUsesCount,
    lastCountAliasesArgument, lastPreparationSucceeded, lastFailureStage,
    lastAccepted, lastBeforeDependencies, lastPrefixDependencies, lastBeforeWork
>>

UsesDistinctCount(usesCount, countAliasesArgument) ==
    usesCount /\ ~countAliasesArgument

RequestedDependencies(usesCount, countAliasesArgument) ==
    IF UsesDistinctCount(usesCount, countAliasesArgument)
    THEN {Argument, Count}
    ELSE {Argument}

Init ==
    /\ recordingDependencies = {}
    /\ work = 0
    /\ transactionActive = FALSE
    /\ transactionUsesCount = FALSE
    /\ transactionCountAliasesArgument = FALSE
    /\ transactionFailureStage = -1
    /\ transactionBeforeDependencies = {}
    /\ transactionBeforeWork = 0
    /\ lastAttemptObserved = FALSE
    /\ lastRequestValid = FALSE
    /\ lastZeroCount = FALSE
    /\ lastUsesCount = FALSE
    /\ lastCountAliasesArgument = FALSE
    /\ lastPreparationSucceeded = FALSE
    /\ lastFailureStage = -1
    /\ lastAccepted = FALSE
    /\ lastBeforeDependencies = {}
    /\ lastPrefixDependencies = {}
    /\ lastBeforeWork = 0

beginAttempt(
    requestValid, zeroCount, usesCount,
    countAliasesArgument, preparationSucceeded, failureStage
) ==
    LET canPublishPrefix ==
            /\ requestValid
            /\ ~zeroCount
            /\ preparationSucceeded
            /\ failureStage # 0
        completesImmediately == ~canPublishPrefix
    IN /\ ~transactionActive
       /\ work < MaxWork
       /\ (countAliasesArgument => usesCount)
       /\ (failureStage = 1 => UsesDistinctCount(usesCount, countAliasesArgument))
       /\ (zeroCount => failureStage = -1)
       /\ recordingDependencies' =
            IF canPublishPrefix
            THEN recordingDependencies \cup {Argument}
            ELSE recordingDependencies
       /\ transactionActive' = canPublishPrefix
       /\ transactionUsesCount' = usesCount
       /\ transactionCountAliasesArgument' = countAliasesArgument
       /\ transactionFailureStage' = failureStage
       /\ transactionBeforeDependencies' = recordingDependencies
       /\ transactionBeforeWork' = work
       /\ lastAttemptObserved' = completesImmediately
       /\ lastRequestValid' = requestValid
       /\ lastZeroCount' = zeroCount
       /\ lastUsesCount' = usesCount
       /\ lastCountAliasesArgument' = countAliasesArgument
       /\ lastPreparationSucceeded' = preparationSucceeded
       /\ lastFailureStage' = failureStage
       /\ lastAccepted' = (completesImmediately /\ requestValid /\ zeroCount)
       /\ lastBeforeDependencies' = recordingDependencies
       /\ lastPrefixDependencies' = recordingDependencies
       /\ lastBeforeWork' = work
       /\ UNCHANGED work

finishTransaction ==
    LET countPublicationFailed == transactionFailureStage = 1
        completedDependencies ==
            IF UsesDistinctCount(transactionUsesCount, transactionCountAliasesArgument)
            THEN recordingDependencies \cup {Count}
            ELSE recordingDependencies
    IN /\ transactionActive
       /\ recordingDependencies' =
            IF countPublicationFailed
            THEN transactionBeforeDependencies
            ELSE completedDependencies
       /\ work' = IF countPublicationFailed THEN work ELSE work + 1
       /\ transactionActive' = FALSE
       /\ lastAttemptObserved' = TRUE
       /\ lastRequestValid' = TRUE
       /\ lastZeroCount' = FALSE
       /\ lastUsesCount' = transactionUsesCount
       /\ lastCountAliasesArgument' = transactionCountAliasesArgument
       /\ lastPreparationSucceeded' = TRUE
       /\ lastFailureStage' = transactionFailureStage
       /\ lastAccepted' = ~countPublicationFailed
       /\ lastBeforeDependencies' = transactionBeforeDependencies
       /\ lastPrefixDependencies' = recordingDependencies
       /\ lastBeforeWork' = transactionBeforeWork
       /\ UNCHANGED <<
            transactionUsesCount, transactionCountAliasesArgument,
            transactionFailureStage, transactionBeforeDependencies,
            transactionBeforeWork
          >>

resetRecording ==
    /\ ~transactionActive
    /\ (work > 0 \/ recordingDependencies # {})
    /\ recordingDependencies' = {}
    /\ work' = 0
    /\ lastAttemptObserved' = FALSE
    /\ UNCHANGED <<
        transactionActive, transactionUsesCount,
        transactionCountAliasesArgument, transactionFailureStage,
        transactionBeforeDependencies, transactionBeforeWork,
        lastRequestValid, lastZeroCount, lastUsesCount,
        lastCountAliasesArgument, lastPreparationSucceeded,
        lastFailureStage, lastAccepted, lastBeforeDependencies,
        lastPrefixDependencies, lastBeforeWork
       >>

Next ==
    \/ \E rv \in BOOLEAN, zc \in BOOLEAN, uc \in BOOLEAN,
          alias \in BOOLEAN, prep \in BOOLEAN, fs \in FailureStages :
            beginAttempt(rv, zc, uc, alias, prep, fs)
    \/ finishTransaction
    \/ resetRecording

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ recordingDependencies \in SUBSET Dependencies
    /\ work \in 0..MaxWork
    /\ transactionActive \in BOOLEAN
    /\ transactionUsesCount \in BOOLEAN
    /\ transactionCountAliasesArgument \in BOOLEAN
    /\ transactionFailureStage \in FailureStages
    /\ transactionBeforeDependencies \in SUBSET Dependencies
    /\ transactionBeforeWork \in 0..(MaxWork - 1)
    /\ lastAttemptObserved \in BOOLEAN
    /\ lastRequestValid \in BOOLEAN
    /\ lastZeroCount \in BOOLEAN
    /\ lastUsesCount \in BOOLEAN
    /\ lastCountAliasesArgument \in BOOLEAN
    /\ lastPreparationSucceeded \in BOOLEAN
    /\ lastFailureStage \in FailureStages
    /\ lastAccepted \in BOOLEAN
    /\ lastBeforeDependencies \in SUBSET Dependencies
    /\ lastPrefixDependencies \in SUBSET Dependencies
    /\ lastBeforeWork \in 0..(MaxWork - 1)
    /\ (transactionCountAliasesArgument => transactionUsesCount)
    /\ (lastCountAliasesArgument => lastUsesCount)

ActivePrefixIsReal ==
    transactionActive =>
        /\ transactionFailureStage # 0
        /\ recordingDependencies = transactionBeforeDependencies \cup {Argument}
        /\ work = transactionBeforeWork

AcceptanceIsExact ==
    lastAttemptObserved =>
        (lastAccepted =
            (/\ lastRequestValid
             /\ (lastZeroCount
                 \/ (lastPreparationSucceeded /\ lastFailureStage = -1))))

ZeroCountIsExactNoOp ==
    (lastAttemptObserved /\ lastRequestValid /\ lastZeroCount) =>
        /\ lastAccepted
        /\ lastPrefixDependencies = lastBeforeDependencies
        /\ recordingDependencies = lastBeforeDependencies
        /\ work = lastBeforeWork

RejectedAttemptIsAtomic ==
    (lastAttemptObserved /\ ~lastAccepted) =>
        /\ recordingDependencies = lastBeforeDependencies
        /\ work = lastBeforeWork

CountFailureRollsBackPrefix ==
    (lastAttemptObserved /\ lastRequestValid /\ ~lastZeroCount
     /\ lastPreparationSucceeded /\ lastFailureStage = 1) =>
        /\ ~lastAccepted
        /\ lastUsesCount
        /\ ~lastCountAliasesArgument
        /\ lastPrefixDependencies = lastBeforeDependencies \cup {Argument}
        /\ recordingDependencies = lastBeforeDependencies
        /\ work = lastBeforeWork

AcceptedPublicationIsComplete ==
    (lastAttemptObserved /\ lastAccepted /\ ~lastZeroCount) =>
        /\ lastPrefixDependencies = lastBeforeDependencies \cup {Argument}
        /\ recordingDependencies =
            lastBeforeDependencies \cup RequestedDependencies(lastUsesCount, lastCountAliasesArgument)
        /\ work = lastBeforeWork + 1

Safety ==
    /\ ActivePrefixIsReal
    /\ AcceptanceIsExact
    /\ ZeroCountIsExactNoOp
    /\ RejectedAttemptIsAtomic
    /\ CountFailureRollsBackPrefix
    /\ AcceptedPublicationIsComplete

=============================================================================
