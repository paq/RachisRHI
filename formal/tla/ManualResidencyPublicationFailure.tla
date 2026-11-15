----------------- MODULE ManualResidencyPublicationFailure ------------------
(***************************************************************************)
(* Post-acceptance `EnqueueMakeResident` publication failure, physical     *)
(* allocation quarantine, queue-idle recovery, and faulted-device teardown *)
(* in `src/D3D12/D3D12Residency.h`, `src/D3D12/D3D12Residency.cpp`,        *)
(* `src/D3D12/D3D12Resource.cpp`, and `src/D3D12/D3D12Queue.cpp`.          *)
(*                                                                         *)
(* Distinguishes validation failure before `EnqueueMakeResident` (which    *)
(* leaves an evicted resource untouched) from timeline publication failure *)
(* after native acceptance (which publishes no usable `TimelinePoint`,     *)
(* marks `transitionUnpublishable = TRUE`, and quarantines physical        *)
(* memory). Logical handle destruction cannot release or reuse a           *)
(* quarantined allocation until faulted-device teardown (`deviceAlive =    *)
(* FALSE`), after which no new resource creation is permitted.             *)
(*                                                                         *)
(* Abstracted: multi-queue cross-fence propagation on healthy transitions  *)
(* (covered by `ManualResidency.tla`).                                     *)
(***************************************************************************)
EXTENDS Naturals

MaxGeneration == 2
States == {"absent", "evicted", "pending", "resident"}
Outcomes == {
    "idle",
    "rejectedBeforeAcceptance",
    "published",
    "completed",
    "failedAfterAcceptance",
    "failedAfterQueueWork",
    "destroyed",
    "destroyedFaulted",
    "deviceTeardown",
    "recreated",
    "evicted",
    "queueIdle"
}

VARIABLES
    deviceAlive,                \* TRUE until faulted-device teardown completes
    handleLive,                 \* TRUE while the public resource handle is live
    physicalAlive,              \* TRUE while the native D3D12 allocation lives
    residency,                  \* residency state in States
    nativeAccepted,             \* TRUE after EnqueueMakeResident succeeded
    publicationPoint,           \* 1 when a portable TimelinePoint was published
    transitionUnpublishable,    \* TRUE when post-acceptance publication failed
    ownerQuarantined,           \* TRUE while physical allocation is quarantined
    queueWorkUnpublished,       \* TRUE while accepted unpublished work is pending
    queueIdleObserved,          \* TRUE immediately after queue idle/teardown drain
    generation,                 \* allocation generation in 1..MaxGeneration
    lastOutcome                 \* latest transition tag in Outcomes

coreVars == <<
    deviceAlive, handleLive, physicalAlive, residency, nativeAccepted,
    publicationPoint, transitionUnpublishable, ownerQuarantined,
    queueWorkUnpublished, generation
>>

vars == <<
    deviceAlive, handleLive, physicalAlive, residency, nativeAccepted,
    publicationPoint, transitionUnpublishable, ownerQuarantined,
    queueWorkUnpublished, queueIdleObserved, generation, lastOutcome
>>

Init ==
    /\ deviceAlive = TRUE
    /\ handleLive = TRUE
    /\ physicalAlive = TRUE
    /\ residency = "evicted"
    /\ nativeAccepted = FALSE
    /\ publicationPoint = 0
    /\ transitionUnpublishable = FALSE
    /\ ownerQuarantined = FALSE
    /\ queueWorkUnpublished = FALSE
    /\ queueIdleObserved = FALSE
    /\ generation = 1
    /\ lastOutcome = "idle"

rejectBeforeAcceptance ==
    /\ deviceAlive
    /\ handleLive
    /\ physicalAlive
    /\ residency = "evicted"
    /\ queueIdleObserved' = FALSE
    /\ lastOutcome' = "rejectedBeforeAcceptance"
    /\ UNCHANGED coreVars

publish ==
    /\ deviceAlive
    /\ handleLive
    /\ physicalAlive
    /\ residency = "evicted"
    /\ residency' = "pending"
    /\ nativeAccepted' = TRUE
    /\ publicationPoint' = 1
    /\ transitionUnpublishable' = FALSE
    /\ ownerQuarantined' = FALSE
    /\ queueWorkUnpublished' = FALSE
    /\ queueIdleObserved' = FALSE
    /\ lastOutcome' = "published"
    /\ UNCHANGED <<deviceAlive, handleLive, physicalAlive, generation>>

complete ==
    /\ deviceAlive
    /\ handleLive
    /\ physicalAlive
    /\ residency = "pending"
    /\ publicationPoint = 1
    /\ ~transitionUnpublishable
    /\ residency' = "resident"
    /\ queueIdleObserved' = FALSE
    /\ lastOutcome' = "completed"
    /\ UNCHANGED <<
        deviceAlive, handleLive, physicalAlive, nativeAccepted,
        publicationPoint, transitionUnpublishable, ownerQuarantined,
        queueWorkUnpublished, generation
       >>

failAfterAcceptance ==
    /\ deviceAlive
    /\ handleLive
    /\ physicalAlive
    /\ residency = "evicted"
    /\ residency' = "pending"
    /\ nativeAccepted' = TRUE
    /\ publicationPoint' = 0
    /\ transitionUnpublishable' = TRUE
    /\ ownerQuarantined' = TRUE
    /\ queueWorkUnpublished' = FALSE
    /\ queueIdleObserved' = FALSE
    /\ lastOutcome' = "failedAfterAcceptance"
    /\ UNCHANGED <<deviceAlive, handleLive, physicalAlive, generation>>

failAfterQueueWork ==
    /\ deviceAlive
    /\ handleLive
    /\ physicalAlive
    /\ residency = "evicted"
    /\ residency' = "pending"
    /\ nativeAccepted' = TRUE
    /\ publicationPoint' = 0
    /\ transitionUnpublishable' = TRUE
    /\ ownerQuarantined' = TRUE
    /\ queueWorkUnpublished' = TRUE
    /\ queueIdleObserved' = FALSE
    /\ lastOutcome' = "failedAfterQueueWork"
    /\ UNCHANGED <<deviceAlive, handleLive, physicalAlive, generation>>

evict ==
    /\ deviceAlive
    /\ handleLive
    /\ physicalAlive
    /\ residency = "resident"
    /\ residency' = "evicted"
    /\ nativeAccepted' = FALSE
    /\ publicationPoint' = 0
    /\ transitionUnpublishable' = FALSE
    /\ ownerQuarantined' = FALSE
    /\ queueWorkUnpublished' = FALSE
    /\ queueIdleObserved' = FALSE
    /\ lastOutcome' = "evicted"
    /\ UNCHANGED <<deviceAlive, handleLive, physicalAlive, generation>>

destroyOrdinary ==
    /\ deviceAlive
    /\ handleLive
    /\ physicalAlive
    /\ ~transitionUnpublishable
    /\ handleLive' = FALSE
    /\ physicalAlive' = FALSE
    /\ residency' = "absent"
    /\ nativeAccepted' = FALSE
    /\ publicationPoint' = 0
    /\ transitionUnpublishable' = FALSE
    /\ ownerQuarantined' = FALSE
    /\ queueWorkUnpublished' = FALSE
    /\ queueIdleObserved' = FALSE
    /\ lastOutcome' = "destroyed"
    /\ UNCHANGED <<deviceAlive, generation>>

destroyFaulted ==
    /\ deviceAlive
    /\ handleLive
    /\ physicalAlive
    /\ transitionUnpublishable
    /\ handleLive' = FALSE
    /\ physicalAlive' = TRUE
    /\ queueIdleObserved' = FALSE
    /\ lastOutcome' = "destroyedFaulted"
    /\ UNCHANGED <<
        deviceAlive, residency, nativeAccepted, publicationPoint,
        transitionUnpublishable, ownerQuarantined, queueWorkUnpublished,
        generation
       >>

teardownFaultedDevice ==
    /\ deviceAlive
    /\ transitionUnpublishable
    /\ ownerQuarantined
    /\ physicalAlive
    /\ deviceAlive' = FALSE
    /\ handleLive' = FALSE
    /\ physicalAlive' = FALSE
    /\ residency' = "absent"
    /\ nativeAccepted' = FALSE
    /\ publicationPoint' = 0
    /\ transitionUnpublishable' = FALSE
    /\ ownerQuarantined' = FALSE
    /\ queueWorkUnpublished' = FALSE
    /\ queueIdleObserved' = TRUE
    /\ lastOutcome' = "deviceTeardown"
    /\ UNCHANGED generation

recreate ==
    /\ deviceAlive
    /\ ~handleLive
    /\ ~physicalAlive
    /\ generation < MaxGeneration
    /\ handleLive' = TRUE
    /\ physicalAlive' = TRUE
    /\ residency' = "evicted"
    /\ nativeAccepted' = FALSE
    /\ publicationPoint' = 0
    /\ transitionUnpublishable' = FALSE
    /\ ownerQuarantined' = FALSE
    /\ queueWorkUnpublished' = FALSE
    /\ queueIdleObserved' = FALSE
    /\ generation' = generation + 1
    /\ lastOutcome' = "recreated"
    /\ UNCHANGED deviceAlive

waitQueueIdle ==
    /\ deviceAlive
    /\ queueWorkUnpublished
    /\ queueWorkUnpublished' = FALSE
    /\ queueIdleObserved' = TRUE
    /\ lastOutcome' = "queueIdle"
    /\ UNCHANGED <<
        deviceAlive, handleLive, physicalAlive, residency, nativeAccepted,
        publicationPoint, transitionUnpublishable, ownerQuarantined,
        generation
       >>

Next ==
    \/ rejectBeforeAcceptance
    \/ publish
    \/ complete
    \/ failAfterAcceptance
    \/ failAfterQueueWork
    \/ evict
    \/ destroyOrdinary
    \/ destroyFaulted
    \/ teardownFaultedDevice
    \/ recreate
    \/ waitQueueIdle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ deviceAlive \in BOOLEAN
    /\ handleLive \in BOOLEAN
    /\ physicalAlive \in BOOLEAN
    /\ residency \in States
    /\ nativeAccepted \in BOOLEAN
    /\ publicationPoint \in {0, 1}
    /\ transitionUnpublishable \in BOOLEAN
    /\ ownerQuarantined \in BOOLEAN
    /\ queueWorkUnpublished \in BOOLEAN
    /\ queueIdleObserved \in BOOLEAN
    /\ generation \in 1..MaxGeneration
    /\ lastOutcome \in Outcomes

PhysicalMatchesResidency ==
    /\ (handleLive => deviceAlive /\ physicalAlive)
    /\ (physicalAlive => deviceAlive)
    /\ ((residency = "absent") = ~physicalAlive)

AcceptedFailureQuarantined ==
    transitionUnpublishable = ownerQuarantined
    /\ (transitionUnpublishable =>
            /\ deviceAlive
            /\ physicalAlive
            /\ residency = "pending"
            /\ nativeAccepted
            /\ publicationPoint = 0)

PublishedTransitionsUsable ==
    publicationPoint = 1 =>
        /\ deviceAlive
        /\ handleLive
        /\ physicalAlive
        /\ nativeAccepted
        /\ ~transitionUnpublishable
        /\ ~ownerQuarantined
        /\ residency \in {"pending", "resident"}

LogicalDestructionPreservesQuarantine ==
    (~handleLive /\ physicalAlive) =>
        /\ deviceAlive
        /\ transitionUnpublishable
        /\ ownerQuarantined
        /\ residency = "pending"

QueueIdleDrainsUnpublishedWork ==
    queueIdleObserved => ~queueWorkUnpublished

TornDownDeviceStaysEmpty ==
    ~deviceAlive =>
        /\ ~handleLive
        /\ ~physicalAlive
        /\ residency = "absent"
        /\ ~ownerQuarantined
        /\ ~queueWorkUnpublished

Safety ==
    /\ PhysicalMatchesResidency
    /\ AcceptedFailureQuarantined
    /\ PublishedTransitionsUsable
    /\ LogicalDestructionPreservesQuarantine
    /\ QueueIdleDrainsUnpublishedWork
    /\ TornDownDeviceStaysEmpty

StepSafety ==
    [][
        /\ (lastOutcome' = "rejectedBeforeAcceptance" => coreVars' = coreVars)
        /\ ((ownerQuarantined /\ deviceAlive') =>
                /\ ownerQuarantined'
                /\ physicalAlive'
                /\ generation' = generation)
        /\ (~deviceAlive => ~deviceAlive' /\ generation' = generation)
        /\ generation <= generation'
    ]_vars

=============================================================================
