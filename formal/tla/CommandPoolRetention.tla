------------------------ MODULE CommandPoolRetention ------------------------
(***************************************************************************)
(* Retained vs. unretained command-pool application resource ownership,    *)
(* manual-residency lease tracking across both modes, and backend-private  *)
(* transient ownership.                                                    *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::retainResource gating       *)
(*     application ownership on CommandPoolFlag::unretainedResources and   *)
(*     internal, retainInternalResource, D3D12CommandBuffer::abandon, and  *)
(*     collectQueue fence retirement through clearPoolEntryRetentions)     *)
(*   - src/D3D12/D3D12Residency.cpp (appendResidencyLeases and             *)
(*     ResidencyLease use counts)                                          *)
(*   - src/D3D12/D3D12Resource.cpp (ResourceManager::evict rejects         *)
(*     resources whose residency use count is non-zero)                    *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Command 0 belongs to a default retained pool; Command 1 belongs to  *)
(*     an unretained pool (CommandPoolFlag::unretainedResources).          *)
(*   - Retained pools retain physical application resources across early   *)
(*     public handle destruction; unretained pools omit application-object *)
(*     shared_ptr ownership, placing destruction safety through exact GPU  *)
(*     completion on the caller.                                           *)
(*   - Manual-residency leases ARE tracked in BOTH retained and unretained *)
(*     pools (when TrackResidencyInUnretainedPools = TRUE, matching        *)
(*     D3D12Queue.cpp running appendResidencyLeases regardless of          *)
(*     retainOwner), blocking eviction until abandonment or fence collect. *)
(*   - Backend-private transients (internal = true) remain physically      *)
(*     owned by the command buffer in both pool modes.                     *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Unretained-pool caller discipline against destroying a referenced   *)
(*     application handle before GPU completion (modeled as Destroy guard).*)
(***************************************************************************)
EXTENDS Naturals

CONSTANT TrackResidencyInUnretainedPools

ASSUME TrackResidencyInUnretainedPools \in BOOLEAN

Commands == {0, 1}
Resources == {0, 1}
RetainsAppResources == [c \in Commands |-> c = 0]
Phases == {"available", "recording", "inFlight"}
MaxTimeline == 2

VARIABLES
    logicalAlive,           \* [Resources -> BOOLEAN] public handle alive
    physicalAlive,          \* [Resources -> BOOLEAN] physical D3D12 resource alive
    manualResidency,        \* [Resources -> BOOLEAN] resource uses manual residency
    resident,               \* [Resources -> BOOLEAN] resource memory is currently resident
    phase,                  \* [Commands -> Phases] command lifecycle phase
    hasUse,                 \* [Commands -> BOOLEAN] command recorded a resource use
    resourceForCommand,     \* [Commands -> Resources] resource referenced by command
    commandOwnsResource,    \* [Commands -> BOOLEAN] entry.retainedResources holds resource
    commandHoldsResidency,  \* [Commands -> BOOLEAN] entry.residencyLeases holds resource
    hasInternalTransient,   \* [Commands -> BOOLEAN] command needed private transient
    internalTransientAlive, \* [Commands -> BOOLEAN] private transient is physically alive
    retireAt,               \* [Commands -> 0..MaxTimeline] submission retirement point
    submitted,              \* latest submitted timeline point
    completed               \* latest completed timeline point

vars == <<logicalAlive, physicalAlive, manualResidency, resident, phase,
          hasUse, resourceForCommand, commandOwnsResource,
          commandHoldsResidency, hasInternalTransient,
          internalTransientAlive, retireAt, submitted, completed>>

Active(c) ==
    phase[c] \in {"recording", "inFlight"}

RetainedByOtherCommand(r, excluded) ==
    \E c \in Commands :
        /\ c # excluded
        /\ Active(c)
        /\ hasUse[c]
        /\ resourceForCommand[c] = r
        /\ commandOwnsResource[c]

RetainedReference(r) ==
    \E c \in Commands :
        /\ Active(c)
        /\ hasUse[c]
        /\ resourceForCommand[c] = r
        /\ commandOwnsResource[c]

UnretainedReference(r) ==
    \E c \in Commands :
        /\ Active(c)
        /\ hasUse[c]
        /\ resourceForCommand[c] = r
        /\ ~commandOwnsResource[c]

ResidencyLeased(r) ==
    \E c \in Commands :
        /\ Active(c)
        /\ hasUse[c]
        /\ resourceForCommand[c] = r
        /\ commandHoldsResidency[c]

TypeOK ==
    /\ logicalAlive \in [Resources -> BOOLEAN]
    /\ physicalAlive \in [Resources -> BOOLEAN]
    /\ manualResidency \in [Resources -> BOOLEAN]
    /\ resident \in [Resources -> BOOLEAN]
    /\ phase \in [Commands -> Phases]
    /\ hasUse \in [Commands -> BOOLEAN]
    /\ resourceForCommand \in [Commands -> Resources]
    /\ commandOwnsResource \in [Commands -> BOOLEAN]
    /\ commandHoldsResidency \in [Commands -> BOOLEAN]
    /\ hasInternalTransient \in [Commands -> BOOLEAN]
    /\ internalTransientAlive \in [Commands -> BOOLEAN]
    /\ retireAt \in [Commands -> 0..MaxTimeline]
    /\ submitted \in 0..MaxTimeline
    /\ completed \in 0..MaxTimeline
    /\ completed <= submitted
    /\ \A c \in Commands : retireAt[c] <= submitted

Init ==
    /\ logicalAlive = [r \in Resources |-> FALSE]
    /\ physicalAlive = [r \in Resources |-> FALSE]
    /\ manualResidency = [r \in Resources |-> FALSE]
    /\ resident = [r \in Resources |-> FALSE]
    /\ phase = [c \in Commands |-> "available"]
    /\ hasUse = [c \in Commands |-> FALSE]
    /\ resourceForCommand = [c \in Commands |-> 0]
    /\ commandOwnsResource = [c \in Commands |-> FALSE]
    /\ commandHoldsResidency = [c \in Commands |-> FALSE]
    /\ hasInternalTransient = [c \in Commands |-> FALSE]
    /\ internalTransientAlive = [c \in Commands |-> FALSE]
    /\ retireAt = [c \in Commands |-> 0]
    /\ submitted = 0
    /\ completed = 0

CreateResource(r, isManual) ==
    /\ ~physicalAlive[r]
    /\ logicalAlive' = [logicalAlive EXCEPT ![r] = TRUE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![r] = TRUE]
    /\ manualResidency' = [manualResidency EXCEPT ![r] = isManual]
    /\ resident' = [resident EXCEPT ![r] = TRUE]
    /\ UNCHANGED <<phase, hasUse, resourceForCommand, commandOwnsResource,
                   commandHoldsResidency, hasInternalTransient,
                   internalTransientAlive, retireAt, submitted, completed>>

EvictResource(r) ==
    /\ logicalAlive[r]
    /\ manualResidency[r]
    /\ resident[r]
    /\ ~ResidencyLeased(r)
    /\ resident' = [resident EXCEPT ![r] = FALSE]
    /\ UNCHANGED <<logicalAlive, physicalAlive, manualResidency, phase,
                   hasUse, resourceForCommand, commandOwnsResource,
                   commandHoldsResidency, hasInternalTransient,
                   internalTransientAlive, retireAt, submitted, completed>>

MakeResident(r) ==
    /\ logicalAlive[r]
    /\ manualResidency[r]
    /\ ~resident[r]
    /\ resident' = [resident EXCEPT ![r] = TRUE]
    /\ UNCHANGED <<logicalAlive, physicalAlive, manualResidency, phase,
                   hasUse, resourceForCommand, commandOwnsResource,
                   commandHoldsResidency, hasInternalTransient,
                   internalTransientAlive, retireAt, submitted, completed>>

Record(c, r, needsInternal) ==
    /\ phase[c] = "available"
    /\ logicalAlive[r]
    /\ physicalAlive[r]
    /\ resident[r]
    /\ LET tracksResidency ==
               manualResidency[r]
               /\ (RetainsAppResources[c] \/ TrackResidencyInUnretainedPools)
       IN /\ phase' = [phase EXCEPT ![c] = "recording"]
          /\ hasUse' = [hasUse EXCEPT ![c] = TRUE]
          /\ resourceForCommand' = [resourceForCommand EXCEPT ![c] = r]
          /\ commandOwnsResource' =
               [commandOwnsResource EXCEPT ![c] = RetainsAppResources[c]]
          /\ commandHoldsResidency' =
               [commandHoldsResidency EXCEPT ![c] = tracksResidency]
          /\ hasInternalTransient' = [hasInternalTransient EXCEPT ![c] = needsInternal]
          /\ internalTransientAlive' = [internalTransientAlive EXCEPT ![c] = needsInternal]
          /\ retireAt' = [retireAt EXCEPT ![c] = 0]
          /\ UNCHANGED <<logicalAlive, physicalAlive, manualResidency,
                         resident, submitted, completed>>

DestroyResource(r) ==
    /\ logicalAlive[r]
    \* Caller lifetime obligation for unretained command pools.
    /\ ~UnretainedReference(r)
    /\ LET keepPhys == RetainedReference(r) IN
       /\ logicalAlive' = [logicalAlive EXCEPT ![r] = FALSE]
       /\ physicalAlive' = [physicalAlive EXCEPT ![r] = keepPhys]
       /\ resident' = [resident EXCEPT ![r] = resident[r] /\ keepPhys]
       /\ UNCHANGED <<manualResidency, phase, hasUse, resourceForCommand,
                      commandOwnsResource, commandHoldsResidency,
                      hasInternalTransient, internalTransientAlive,
                      retireAt, submitted, completed>>

Submit(c) ==
    /\ phase[c] = "recording"
    /\ hasUse[c]
    /\ physicalAlive[resourceForCommand[c]]
    /\ submitted < MaxTimeline
    /\ phase' = [phase EXCEPT ![c] = "inFlight"]
    /\ retireAt' = [retireAt EXCEPT ![c] = submitted + 1]
    /\ submitted' = submitted + 1
    /\ UNCHANGED <<logicalAlive, physicalAlive, manualResidency, resident,
                   hasUse, resourceForCommand, commandOwnsResource,
                   commandHoldsResidency, hasInternalTransient,
                   internalTransientAlive, completed>>

ReleaseCommand(c) ==
    LET r == resourceForCommand[c]
        keepPhys == logicalAlive[r] \/ RetainedByOtherCommand(r, c)
    IN /\ physicalAlive' = [physicalAlive EXCEPT ![r] = keepPhys]
       /\ resident' = [resident EXCEPT ![r] = resident[r] /\ keepPhys]
       /\ phase' = [phase EXCEPT ![c] = "available"]
       /\ hasUse' = [hasUse EXCEPT ![c] = FALSE]
       /\ commandOwnsResource' = [commandOwnsResource EXCEPT ![c] = FALSE]
       /\ commandHoldsResidency' = [commandHoldsResidency EXCEPT ![c] = FALSE]
       /\ hasInternalTransient' = [hasInternalTransient EXCEPT ![c] = FALSE]
       /\ internalTransientAlive' = [internalTransientAlive EXCEPT ![c] = FALSE]
       /\ retireAt' = [retireAt EXCEPT ![c] = 0]
       /\ UNCHANGED <<logicalAlive, manualResidency, resourceForCommand,
                      submitted, completed>>

Abandon(c) ==
    /\ phase[c] = "recording"
    /\ ReleaseCommand(c)

Complete ==
    /\ completed < submitted
    /\ completed' = completed + 1
    /\ UNCHANGED <<logicalAlive, physicalAlive, manualResidency, resident,
                   phase, hasUse, resourceForCommand, commandOwnsResource,
                   commandHoldsResidency, hasInternalTransient,
                   internalTransientAlive, retireAt, submitted>>

Collect(c) ==
    /\ phase[c] = "inFlight"
    /\ retireAt[c] <= completed
    /\ ReleaseCommand(c)

Next ==
    \/ \E r \in Resources, isManual \in BOOLEAN : CreateResource(r, isManual)
    \/ \E r \in Resources : EvictResource(r) \/ MakeResident(r) \/ DestroyResource(r)
    \/ \E c \in Commands, r \in Resources, priv \in BOOLEAN : Record(c, r, priv)
    \/ \E c \in Commands : Submit(c) \/ Abandon(c) \/ Collect(c)
    \/ Complete

Spec == Init /\ [][Next]_vars

LogicalResourcesArePhysical ==
    \A r \in Resources : logicalAlive[r] => physicalAlive[r]

ActiveUsesArePhysical ==
    \A c \in Commands :
        (Active(c) /\ hasUse[c]) => physicalAlive[resourceForCommand[c]]

\* Every active recording or in-flight command keeps its referenced resource
\* resident on the GPU in BOTH retained and unretained pool modes.
ActiveUsesStayResident ==
    \A c \in Commands :
        (Active(c) /\ hasUse[c]) => resident[resourceForCommand[c]]

OwnershipMatchesPoolMode ==
    \A c \in Commands :
        commandOwnsResource[c] <=>
            (Active(c) /\ hasUse[c] /\ RetainsAppResources[c])

ResidencyLeasesTrackedInBothModes ==
    \A c \in Commands :
        (TrackResidencyInUnretainedPools =>
            (commandHoldsResidency[c] <=>
                (Active(c) /\ hasUse[c] /\ manualResidency[resourceForCommand[c]])))

InternalTransientsAlwaysRetained ==
    \A c \in Commands :
        internalTransientAlive[c] <=> (Active(c) /\ hasInternalTransient[c])

InFlightUsesHaveExactRetirement ==
    \A c \in Commands :
        phase[c] = "inFlight" => retireAt[c] > 0 /\ retireAt[c] <= submitted

Safety ==
    /\ LogicalResourcesArePhysical
    /\ ActiveUsesArePhysical
    /\ ActiveUsesStayResident
    /\ OwnershipMatchesPoolMode
    /\ ResidencyLeasesTrackedInBothModes
    /\ InternalTransientsAlwaysRetained
    /\ InFlightUsesHaveExactRetirement

StepSafety ==
    [][submitted' >= submitted /\ completed' >= completed]_vars
=============================================================================
