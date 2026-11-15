-------------- MODULE AccelerationStructureCommandPublication ---------------
(***************************************************************************)
(* Transactional command-dependency publication for DXR build, update,     *)
(* compaction, and deserialization commands.                               *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12AccelerationStructure.cpp (ResourceManager::         *)
(*     buildBottomLevelAccelerationStructure,                              *)
(*     buildTopLevelAccelerationStructure, update*AccelerationStructure*,  *)
(*     compactAccelerationStructure, deserializeAccelerationStructure)     *)
(*   - src/D3D12/D3D12Queue.cpp                                            *)
(*     (QueueManager::retainAccelerationStructureDependencies,             *)
(*     QueueManager::retainCommandDependencies,                            *)
(*     QueueManager::retainResource, discardPoolEntryRecording on reset    *)
(*     and abandonment, quarantineBatchEntries on a rejected submission,   *)
(*     commitAccelerationStructurePublications after ExecuteCommandLists)  *)
(*   - src/Vulkan/VulkanAcceleration.cpp (beginFirstBuild) with            *)
(*     src/Vulkan/VulkanCommands.cpp and src/Vulkan/VulkanQueues.cpp       *)
(*     (rollBackVulkanAccelerationStructurePublications on reset,          *)
(*     abandonment, and a rejected submission; commit in                   *)
(*     publishSubmittedBatch)                                              *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Command-list preparation and destination-version allocation happen  *)
(*     before command dependency vectors are mutated.                      *)
(*   - Stage 0 appends manual-residency leases in both retained and        *)
(*     unretained command pools (D3D12Queue.cpp retainResource calls       *)
(*     appendResidencyLeases whether or not the                            *)
(*     CommandPoolFlag::unretainedResources owner check passes), and       *)
(*     appends the application retention bundle only when the pool retains *)
(*     application resources. Thus stage 0 can fail in unretained pools.   *)
(*   - Stage 1 appends internal private uploads (TLAS instance uploads and *)
(*     deserialization archive uploads) with internal = true even when the *)
(*     pool is unretained.                                                 *)
(*   - Counts (not idempotent sets) are tracked across up to MaxWork = 2   *)
(*     recorded commands so a failure on the second command must pop only  *)
(*     its tentative suffix and preserve the first command's entries.      *)
(*   - Design rule RollbackOnFailure (TRUE in the design) restores exact   *)
(*     prior dependency counts when stage 1 fails after stage 0 appended.  *)
(*   - An accepted command publishes its destination version at record     *)
(*     time and journals it on the recording. Until the queue accepts the  *)
(*     recording, a reset, an abandonment, or a rejected submission        *)
(*     discards the recorded native work. Design rule RollbackOnDiscard    *)
(*     (TRUE in the design) then restores every destination to the version *)
(*     it held before the recording, so a discarded first build leaves the *)
(*     destination unbuilt and buildable again. Queue acceptance commits   *)
(*     the journal and the published versions become durable.              *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - D3D12 driver execution of BuildRaytracingAccelerationStructure and  *)
(*     CopyRaytracingAccelerationStructure after the no-fail commit.       *)
(*   - Work recorded in another command buffer that consumes a version     *)
(*     published by this unsubmitted recording must be submitted after     *)
(*     it. Discarding this recording leaves such work referencing a build  *)
(*     that never runs; that is the existing caller GPU-ordering           *)
(*     obligation and is not modeled here.                                 *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS RollbackOnFailure, RollbackOnDiscard

ASSUME RollbackOnFailure \in BOOLEAN
ASSUME RollbackOnDiscard \in BOOLEAN

MaxWork == 2
FailureStages == {"none", "stage0", "stage1"}
Outcomes == {"none", "pending", "accepted", "rejected", "submitted", "discarded"}
RecordingStates == {"open", "submitted", "discarded"}

VARIABLES
    appOwnerCount,          \* count of application retention bundles on entry
    privateOwnerCount,      \* count of internal private upload owners on entry
    residencyCount,         \* count of manual-residency leases on entry
    nativeWork,             \* recorded native DXR commands committed so far
    destinationVersion,     \* published destination AS versions so far
    txActive,               \* TRUE between stage-0 append and stage-1 commit/rollback
    txRetainsApp,           \* whether the active recording pool retains app owners
    txUsesPrivate,          \* whether the command requires an internal private upload
    txFailureStage,         \* injected failure point for the current attempt
    snapAppOwners,          \* pre-attempt snapshot of appOwnerCount
    snapPrivateOwners,      \* pre-attempt snapshot of privateOwnerCount
    snapResidencies,        \* pre-attempt snapshot of residencyCount
    snapWork,               \* pre-attempt snapshot of nativeWork
    snapVersion,            \* pre-attempt snapshot of destinationVersion
    recording,              \* open, accepted by the queue, or discarded before acceptance
    lastOutcome             \* outcome of the most recent step

coreVars == <<appOwnerCount, privateOwnerCount, residencyCount,
              nativeWork, destinationVersion>>
txVars == <<txActive, txRetainsApp, txUsesPrivate, txFailureStage>>
snapVars == <<snapAppOwners, snapPrivateOwners, snapResidencies,
              snapWork, snapVersion>>
vars == <<appOwnerCount, privateOwnerCount, residencyCount,
          nativeWork, destinationVersion,
          txActive, txRetainsApp, txUsesPrivate, txFailureStage,
          snapAppOwners, snapPrivateOwners, snapResidencies,
          snapWork, snapVersion, recording, lastOutcome>>

Stage0AppOwners(retainsApp, before) ==
    IF retainsApp THEN before + 1 ELSE before

Stage0Residencies(before) == before + 1

CommittedPrivateOwners(usesPrivate, before) ==
    IF usesPrivate THEN before + 1 ELSE before

TypeOK ==
    /\ appOwnerCount \in 1..(1 + MaxWork)
    /\ privateOwnerCount \in 0..MaxWork
    /\ residencyCount \in 1..(1 + MaxWork)
    /\ nativeWork \in 0..MaxWork
    /\ destinationVersion \in 0..MaxWork
    /\ txActive \in BOOLEAN
    /\ txRetainsApp \in BOOLEAN
    /\ txUsesPrivate \in BOOLEAN
    /\ txFailureStage \in FailureStages
    /\ snapAppOwners \in 1..(1 + MaxWork)
    /\ snapPrivateOwners \in 0..MaxWork
    /\ snapResidencies \in 1..(1 + MaxWork)
    /\ snapWork \in 0..MaxWork
    /\ snapVersion \in 0..MaxWork
    /\ recording \in RecordingStates
    /\ lastOutcome \in Outcomes

Init ==
    /\ appOwnerCount = 1
    /\ privateOwnerCount = 0
    /\ residencyCount = 1
    /\ nativeWork = 0
    /\ destinationVersion = 0
    /\ txActive = FALSE
    /\ txRetainsApp = FALSE
    /\ txUsesPrivate = FALSE
    /\ txFailureStage = "none"
    /\ snapAppOwners = 1
    /\ snapPrivateOwners = 0
    /\ snapResidencies = 1
    /\ snapWork = 0
    /\ snapVersion = 0
    /\ recording = "open"
    /\ lastOutcome = "none"

\* Begin an AS command attempt. Note that failureStage = "stage0" is permitted
\* even when retainsApp = FALSE because appendResidencyLeases runs in stage 0
\* regardless of CommandPoolFlag::unretainedResources.
BeginAttempt(requestValid, prepSucceeded, retainsApp, usesPrivate, failureStage) ==
    /\ recording = "open"
    /\ ~txActive
    /\ nativeWork < MaxWork
    /\ failureStage = "stage1" => usesPrivate
    /\ snapAppOwners' = appOwnerCount
    /\ snapPrivateOwners' = privateOwnerCount
    /\ snapResidencies' = residencyCount
    /\ snapWork' = nativeWork
    /\ snapVersion' = destinationVersion
    /\ txRetainsApp' = retainsApp
    /\ txUsesPrivate' = usesPrivate
    /\ txFailureStage' = failureStage
    /\ IF ~requestValid \/ ~prepSucceeded \/ failureStage = "stage0"
       THEN /\ txActive' = FALSE
            /\ lastOutcome' = "rejected"
            /\ UNCHANGED coreVars
       ELSE /\ txActive' = TRUE
            /\ lastOutcome' = "pending"
            /\ appOwnerCount' = Stage0AppOwners(retainsApp, appOwnerCount)
            /\ residencyCount' = Stage0Residencies(residencyCount)
            /\ UNCHANGED <<privateOwnerCount, nativeWork, destinationVersion>>
    /\ UNCHANGED recording

FinishTransaction ==
    /\ txActive
    /\ txActive' = FALSE
    /\ IF txFailureStage = "stage1"
       THEN /\ lastOutcome' = "rejected"
            /\ IF RollbackOnFailure
               THEN /\ appOwnerCount' = snapAppOwners
                    /\ residencyCount' = snapResidencies
               ELSE UNCHANGED <<appOwnerCount, residencyCount>>
            /\ UNCHANGED <<privateOwnerCount, nativeWork, destinationVersion>>
       ELSE /\ lastOutcome' = "accepted"
            /\ privateOwnerCount' = CommittedPrivateOwners(txUsesPrivate, snapPrivateOwners)
            /\ nativeWork' = snapWork + 1
            /\ destinationVersion' = snapVersion + 1
            /\ UNCHANGED <<appOwnerCount, residencyCount>>
    /\ UNCHANGED <<txRetainsApp, txUsesPrivate, txFailureStage, snapVars, recording>>

\* The queue accepts the recording; its journaled publications become durable.
SubmitRecording ==
    /\ recording = "open"
    /\ ~txActive
    /\ recording' = "submitted"
    /\ lastOutcome' = "submitted"
    /\ UNCHANGED <<coreVars, txVars, snapVars>>

\* Reset, abandonment, or a rejected submission drops the recorded native work
\* and every retention; the journal restores each destination it published.
DiscardRecording ==
    /\ recording = "open"
    /\ ~txActive
    /\ recording' = "discarded"
    /\ lastOutcome' = "discarded"
    /\ appOwnerCount' = 1
    /\ privateOwnerCount' = 0
    /\ residencyCount' = 1
    /\ nativeWork' = 0
    /\ destinationVersion' = IF RollbackOnDiscard THEN 0 ELSE destinationVersion
    /\ UNCHANGED <<txVars, snapVars>>

\* The discarded pool entry is reused for a fresh recording of the same destination.
ReopenRecording ==
    /\ recording = "discarded"
    /\ recording' = "open"
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<coreVars, txVars, snapVars>>

Next ==
    \/ \E reqOk, prepOk, retainsApp, usesPrivate \in BOOLEAN,
          stage \in FailureStages :
          BeginAttempt(reqOk, prepOk, retainsApp, usesPrivate, stage)
    \/ FinishTransaction
    \/ SubmitRecording
    \/ DiscardRecording
    \/ ReopenRecording

Spec == Init /\ [][Next]_vars

\* Native command recording and logical destination-version publication commit together.
NativeWorkAndVersionPublishTogether ==
    nativeWork = destinationVersion

\* While the transaction is between stage 0 and stage 1, only the stage-0 prefix
\* is staged and neither native work nor destinationVersion has advanced.
ActivePrefixHasNoNativeMutation ==
    txActive =>
        /\ appOwnerCount = Stage0AppOwners(txRetainsApp, snapAppOwners)
        /\ privateOwnerCount = snapPrivateOwners
        /\ residencyCount = Stage0Residencies(snapResidencies)
        /\ nativeWork = snapWork
        /\ destinationVersion = snapVersion

\* Every rejected attempt (including stage-0 failures in unretained pools and
\* stage-1 failures on the second recorded command) restores exact prior counts.
RejectedAttemptRestoresExactCounts ==
    lastOutcome = "rejected" =>
        /\ ~txActive
        /\ appOwnerCount = snapAppOwners
        /\ privateOwnerCount = snapPrivateOwners
        /\ residencyCount = snapResidencies
        /\ nativeWork = snapWork
        /\ destinationVersion = snapVersion

\* Every accepted attempt publishes the complete owner/residency deltas and one
\* recorded command + version increment.
AcceptedPublicationIsComplete ==
    lastOutcome = "accepted" =>
        /\ ~txActive
        /\ appOwnerCount = Stage0AppOwners(txRetainsApp, snapAppOwners)
        /\ privateOwnerCount = CommittedPrivateOwners(txUsesPrivate, snapPrivateOwners)
        /\ residencyCount = Stage0Residencies(snapResidencies)
        /\ nativeWork = snapWork + 1
        /\ destinationVersion = snapVersion + 1

\* Residency count always advances by exactly one per committed command plus at
\* most one tentative stage-0 entry while txActive holds.
ResidencyMatchesCommittedWork ==
    ~txActive => residencyCount = 1 + nativeWork

\* A discarded recording leaves no published destination version, native work,
\* or retention behind, so the destination is unbuilt and may be built again.
DiscardedRecordingPublishesNothing ==
    recording = "discarded" =>
        /\ destinationVersion = 0
        /\ nativeWork = 0
        /\ appOwnerCount = 1
        /\ privateOwnerCount = 0
        /\ residencyCount = 1

Safety ==
    /\ NativeWorkAndVersionPublishTogether
    /\ ActivePrefixHasNoNativeMutation
    /\ RejectedAttemptRestoresExactCounts
    /\ AcceptedPublicationIsComplete
    /\ ResidencyMatchesCommittedWork
    /\ DiscardedRecordingPublishesNothing

StepSafety ==
    [][/\ (~txActive /\ lastOutcome' = "rejected") => coreVars' = coreVars
       /\ lastOutcome' # "discarded" => nativeWork' >= nativeWork
       /\ lastOutcome' # "discarded" => destinationVersion' >= destinationVersion
       /\ recording = "submitted" => (recording' = "submitted" /\ coreVars' = coreVars)]_vars
=============================================================================
