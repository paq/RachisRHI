-------------------- MODULE MeshPipelineStatisticsQuery ---------------------
(***************************************************************************)
(* Options9 mesh/task pipeline-statistics capability gating, 88-byte vs    *)
(* 112-byte query resolution, scope balance, failure atomicity, and pool   *)
(* retention in `src/D3D12/D3D12Device.cpp` (Options9 discovery),          *)
(* `src/D3D12/D3D12Query.cpp`, `src/D3D12/D3D12Queue.cpp`, and             *)
(* `src/Device.cpp` with `src/Internal/PortableValidation.h`               *)
(* (balanced-scope submission). Result sizes come from                     *)
(* `include/Rachis/Query.h`; the API is declared in                        *)
(* `include/Rachis/Device.h`.                                              *)
(*                                                                         *)
(* Models base (88-byte) and extended mesh/task (112-byte) statistics      *)
(* query pools across capability discovery, creation, begin/end recording, *)
(* aligned out-of-scope resolution, query reset, balanced command          *)
(* submission, GPU completion, command abandonment, and logical pool       *)
(* destruction while retained by recorded or submitted work.               *)
(*                                                                         *)
(* Abstracted: 64-bit hardware counter arithmetic beyond presence/absence  *)
(* of mesh/task counter fields.                                            *)
(***************************************************************************)
EXTENDS Integers

BaseMode == 0
MeshMode == 1
InvalidMode == 2
Modes == {BaseMode, MeshMode}
RequestedModes == Modes \cup {InvalidMode}
QueryCounts == {0, 1}
BufferSizes == {87, 88, 111, 112}
Offsets == {0, 1, 8}
CounterValues == {0, 2}
NoCounter == -1
BaseResultSize == 88
MeshResultSize == 112
ResultAlignment == 8
Operations == {
    "none", "create", "begin", "end", "resolve",
    "reset", "destroyPool", "submit", "complete", "abandon"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    nativeMeshStatistics,       \* hardware Options9 mesh-statistics capability
    reportedMeshStatistics,     \* capability surfaced by DeviceCapabilities
    poolLogicalAlive,           \* TRUE while public QueryPoolHandle is live
    poolPhysicalAlive,          \* TRUE while native query heap remains allocated
    recordingRetainsPool,       \* TRUE while active recording retains the pool
    inFlightRetainsPool,        \* TRUE while in-flight submission retains the pool
    inFlight,                   \* TRUE while command buffer is in flight
    poolMode,                   \* BaseMode, MeshMode, or InvalidMode when absent
    active,                     \* TRUE while query scope is open in recording
    completed,                  \* TRUE after endQuery, before resolve or reset
    resolved,                   \* TRUE after an accepted resolve in the cycle
    resolvedBytes,              \* byte width written by the latest resolve
    taskInvocations,            \* recorded task counter or NoCounter
    meshInvocations,            \* recorded mesh counter or NoCounter
    meshPrimitives,             \* recorded mesh-primitive counter or NoCounter
    lastOperation,              \* latest operation tag in Operations
    lastOutcome                 \* outcome of latest operation in Outcomes

coreVars == <<
    nativeMeshStatistics, reportedMeshStatistics,
    poolLogicalAlive, poolPhysicalAlive, recordingRetainsPool,
    inFlightRetainsPool, inFlight, poolMode, active, completed,
    resolved, resolvedBytes, taskInvocations, meshInvocations, meshPrimitives
>>

vars == <<
    nativeMeshStatistics, reportedMeshStatistics,
    poolLogicalAlive, poolPhysicalAlive, recordingRetainsPool,
    inFlightRetainsPool, inFlight, poolMode, active, completed,
    resolved, resolvedBytes, taskInvocations, meshInvocations,
    meshPrimitives, lastOperation, lastOutcome
>>

ResultSize(mode) ==
    IF mode = MeshMode THEN MeshResultSize ELSE BaseResultSize

CreateValid(mode, queryCount) ==
    /\ ~poolLogicalAlive
    /\ ~poolPhysicalAlive
    /\ mode \in Modes
    /\ queryCount = 1
    /\ (mode = BaseMode \/ reportedMeshStatistics)

BeginValid ==
    /\ ~inFlight
    /\ poolLogicalAlive
    /\ ~active
    /\ ~completed

EndValid ==
    /\ ~inFlight
    /\ poolPhysicalAlive
    /\ active

ResolveValid(destinationOffset, destinationSize) ==
    /\ ~inFlight
    /\ poolLogicalAlive
    /\ completed
    /\ ~active
    /\ destinationOffset % ResultAlignment = 0
    /\ destinationOffset + ResultSize(poolMode) <= destinationSize

ResetValid ==
    /\ ~inFlight
    /\ poolLogicalAlive
    /\ ~active

Init ==
    \E nativeSupport \in BOOLEAN :
        /\ nativeMeshStatistics = nativeSupport
        /\ reportedMeshStatistics = nativeSupport
        /\ poolLogicalAlive = FALSE
        /\ poolPhysicalAlive = FALSE
        /\ recordingRetainsPool = FALSE
        /\ inFlightRetainsPool = FALSE
        /\ inFlight = FALSE
        /\ poolMode = InvalidMode
        /\ active = FALSE
        /\ completed = FALSE
        /\ resolved = FALSE
        /\ resolvedBytes = 0
        /\ taskInvocations = NoCounter
        /\ meshInvocations = NoCounter
        /\ meshPrimitives = NoCounter
        /\ lastOperation = "none"
        /\ lastOutcome = "none"

attemptCreate(mode, queryCount) ==
    LET accepted == CreateValid(mode, queryCount)
    IN /\ IF accepted
          THEN /\ poolLogicalAlive' = TRUE
               /\ poolPhysicalAlive' = TRUE
               /\ poolMode' = mode
               /\ active' = FALSE
               /\ completed' = FALSE
               /\ resolved' = FALSE
               /\ resolvedBytes' = 0
               /\ taskInvocations' = NoCounter
               /\ meshInvocations' = NoCounter
               /\ meshPrimitives' = NoCounter
               /\ UNCHANGED <<
                    nativeMeshStatistics, reportedMeshStatistics,
                    recordingRetainsPool, inFlightRetainsPool, inFlight
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "create"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptBegin ==
    LET accepted == BeginValid
    IN /\ IF accepted
          THEN /\ recordingRetainsPool' = TRUE
               /\ active' = TRUE
               /\ UNCHANGED <<
                    nativeMeshStatistics, reportedMeshStatistics,
                    poolLogicalAlive, poolPhysicalAlive, inFlightRetainsPool,
                    inFlight, poolMode, completed, resolved, resolvedBytes,
                    taskInvocations, meshInvocations, meshPrimitives
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "begin"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptEnd(counterValue) ==
    LET accepted == EndValid
    IN /\ IF accepted
          THEN /\ active' = FALSE
               /\ completed' = TRUE
               /\ taskInvocations' =
                    IF poolMode = MeshMode THEN counterValue ELSE NoCounter
               /\ meshInvocations' =
                    IF poolMode = MeshMode THEN counterValue ELSE NoCounter
               /\ meshPrimitives' =
                    IF poolMode = MeshMode THEN counterValue ELSE NoCounter
               /\ UNCHANGED <<
                    nativeMeshStatistics, reportedMeshStatistics,
                    poolLogicalAlive, poolPhysicalAlive, recordingRetainsPool,
                    inFlightRetainsPool, inFlight, poolMode, resolved,
                    resolvedBytes
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "end"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptResolve(destinationOffset, destinationSize) ==
    LET accepted == ResolveValid(destinationOffset, destinationSize)
    IN /\ IF accepted
          THEN /\ recordingRetainsPool' = TRUE
               /\ completed' = FALSE
               /\ resolved' = TRUE
               /\ resolvedBytes' = ResultSize(poolMode)
               /\ UNCHANGED <<
                    nativeMeshStatistics, reportedMeshStatistics,
                    poolLogicalAlive, poolPhysicalAlive, inFlightRetainsPool,
                    inFlight, poolMode, active, taskInvocations,
                    meshInvocations, meshPrimitives
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "resolve"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptReset ==
    LET accepted == ResetValid
    IN /\ IF accepted
          THEN /\ recordingRetainsPool' = TRUE
               /\ completed' = FALSE
               /\ resolved' = FALSE
               /\ resolvedBytes' = 0
               /\ taskInvocations' = NoCounter
               /\ meshInvocations' = NoCounter
               /\ meshPrimitives' = NoCounter
               /\ UNCHANGED <<
                    nativeMeshStatistics, reportedMeshStatistics,
                    poolLogicalAlive, poolPhysicalAlive, inFlightRetainsPool,
                    inFlight, poolMode, active
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "reset"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroyPool ==
    LET retained == recordingRetainsPool \/ inFlightRetainsPool
    IN /\ poolLogicalAlive
       /\ poolLogicalAlive' = FALSE
       /\ poolPhysicalAlive' = retained
       /\ poolMode' = IF retained THEN poolMode ELSE InvalidMode
       /\ lastOperation' = "destroyPool"
       /\ lastOutcome' = "accepted"
       /\ UNCHANGED <<
            nativeMeshStatistics, reportedMeshStatistics,
            recordingRetainsPool, inFlightRetainsPool, inFlight,
            active, completed, resolved, resolvedBytes,
            taskInvocations, meshInvocations, meshPrimitives
          >>

attemptSubmit ==
    LET accepted == ~inFlight /\ recordingRetainsPool /\ ~active
    IN /\ ~inFlight
       /\ recordingRetainsPool
       /\ IF accepted
          THEN /\ recordingRetainsPool' = FALSE
               /\ inFlightRetainsPool' = TRUE
               /\ inFlight' = TRUE
               /\ completed' = FALSE
               /\ resolved' = FALSE
               /\ resolvedBytes' = 0
               /\ taskInvocations' = NoCounter
               /\ meshInvocations' = NoCounter
               /\ meshPrimitives' = NoCounter
               /\ UNCHANGED <<
                    nativeMeshStatistics, reportedMeshStatistics,
                    poolLogicalAlive, poolPhysicalAlive, poolMode, active
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "submit"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

completeSubmission ==
    /\ inFlight
    /\ poolPhysicalAlive' = (poolLogicalAlive \/ recordingRetainsPool)
    /\ poolMode' =
        IF poolLogicalAlive \/ recordingRetainsPool THEN poolMode ELSE InvalidMode
    /\ inFlightRetainsPool' = FALSE
    /\ inFlight' = FALSE
    /\ lastOperation' = "complete"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeMeshStatistics, reportedMeshStatistics,
        poolLogicalAlive, recordingRetainsPool, active, completed,
        resolved, resolvedBytes, taskInvocations, meshInvocations,
        meshPrimitives
       >>

abandon ==
    /\ ~inFlight
    /\ recordingRetainsPool
    /\ poolPhysicalAlive' = (poolLogicalAlive \/ inFlightRetainsPool)
    /\ poolMode' =
        IF poolLogicalAlive \/ inFlightRetainsPool THEN poolMode ELSE InvalidMode
    /\ recordingRetainsPool' = FALSE
    /\ active' = FALSE
    /\ completed' = FALSE
    /\ resolved' = FALSE
    /\ resolvedBytes' = 0
    /\ taskInvocations' = NoCounter
    /\ meshInvocations' = NoCounter
    /\ meshPrimitives' = NoCounter
    /\ lastOperation' = "abandon"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeMeshStatistics, reportedMeshStatistics,
        poolLogicalAlive, inFlightRetainsPool, inFlight
       >>

Next ==
    \/ \E m \in RequestedModes, qc \in QueryCounts : attemptCreate(m, qc)
    \/ attemptBegin
    \/ \E cv \in CounterValues : attemptEnd(cv)
    \/ \E off \in Offsets, sz \in BufferSizes : attemptResolve(off, sz)
    \/ attemptReset
    \/ destroyPool
    \/ attemptSubmit
    \/ completeSubmission
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ nativeMeshStatistics \in BOOLEAN
    /\ reportedMeshStatistics \in BOOLEAN
    /\ poolLogicalAlive \in BOOLEAN
    /\ poolPhysicalAlive \in BOOLEAN
    /\ recordingRetainsPool \in BOOLEAN
    /\ inFlightRetainsPool \in BOOLEAN
    /\ inFlight \in BOOLEAN
    /\ poolMode \in RequestedModes
    /\ active \in BOOLEAN
    /\ completed \in BOOLEAN
    /\ resolved \in BOOLEAN
    /\ resolvedBytes \in {0, BaseResultSize, MeshResultSize}
    /\ taskInvocations \in CounterValues \cup {NoCounter}
    /\ meshInvocations \in CounterValues \cup {NoCounter}
    /\ meshPrimitives \in CounterValues \cup {NoCounter}
    /\ lastOperation \in Operations
    /\ lastOutcome \in Outcomes

CapabilityMatchesNative ==
    reportedMeshStatistics = nativeMeshStatistics

MeshModeCapabilityGated ==
    /\ (poolPhysicalAlive = (poolMode \in Modes))
    /\ (poolMode = MeshMode => reportedMeshStatistics)

QueryScopeConsistent ==
    /\ (active => poolPhysicalAlive /\ recordingRetainsPool /\ ~completed /\ ~inFlight)
    /\ (completed => poolPhysicalAlive /\ ~active)
    /\ ((completed /\ poolMode = MeshMode) =>
            /\ taskInvocations \in CounterValues
            /\ meshInvocations \in CounterValues
            /\ meshPrimitives \in CounterValues)
    /\ (poolMode # MeshMode =>
            /\ taskInvocations = NoCounter
            /\ meshInvocations = NoCounter
            /\ meshPrimitives = NoCounter)

ResolvedSizeMatchesMode ==
    resolved =>
        /\ poolPhysicalAlive
        /\ resolvedBytes = ResultSize(poolMode)

PoolPhysicalLifetimeExact ==
    poolPhysicalAlive =
        (poolLogicalAlive \/ recordingRetainsPool \/ inFlightRetainsPool)

Safety ==
    /\ CapabilityMatchesNative
    /\ MeshModeCapabilityGated
    /\ QueryScopeConsistent
    /\ ResolvedSizeMatchesMode
    /\ PoolPhysicalLifetimeExact

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
        /\ (lastOperation' = "begin" =>
                ((lastOutcome' = "accepted") = BeginValid))
        /\ (lastOperation' = "end" =>
                ((lastOutcome' = "accepted") = EndValid))
        /\ (lastOperation' = "reset" =>
                ((lastOutcome' = "accepted") = ResetValid))
        /\ (lastOperation' = "submit" =>
                ((lastOutcome' = "accepted") = (~inFlight /\ recordingRetainsPool /\ ~active)))
    ]_vars

=============================================================================
