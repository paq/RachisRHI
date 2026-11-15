--------------------------- MODULE QueryPoolReset ---------------------------
(***************************************************************************)
(* Portable query-range reset across all query-pool kinds, selective       *)
(* clearing of unresolved slots while preserving unselected slots, scope   *)
(* and queue validation, and reset-only pool retention across destruction. *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Query.cpp (resetQueryPool, resetPendingRange)        *)
(*   - src/Vulkan/VulkanQueries.cpp (resetQueryPool, resetPendingRange)    *)
(*   - src/Device.cpp (facade forwarding)                                  *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two query indices, two queues, render-pass / pass-timestamp / query   *)
(*   scopes, and pool destruction cover exact set-difference reset         *)
(*   semantics, failure-atomic invalid resets/writes, and physical pool    *)
(*   retention through submission and completion.                          *)
(***************************************************************************)

EXTENDS Naturals, FiniteSets

Queries == {0, 1}
QueryIndices == {0, 1, 2}
QueryCounts == {0, 1, 2}
Queues == {0, 1}
PoolQueue == 0
Operations == {
  "none", "write", "reset", "beginRendering", "endRendering",
  "openScope", "closeScope", "destroy", "submit", "complete", "abandon"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  logicalPoolAlive,     \* Public QueryPoolHandle liveness
  physicalPoolAlive,    \* Native query pool liveness
  recordingRetainsPool, \* Whether the open command buffer retains the pool
  inFlightRetainsPool,  \* Whether an in-flight submission retains the pool
  pendingQueries,       \* Unresolved written query indices in this recording
  rendering,            \* Whether a render pass is currently open
  passTimestampActive,  \* Whether activePassTimestampWrites is open
  scopeOpen,            \* Whether a query or GPU-profile scope is open
  inFlight,             \* Whether the command buffer is currently in flight
  lastOp,               \* Most recent operation kind
  lastOutcome,          \* Outcome of most recent operation
  lastResetRange        \* Query index set targeted by most recent reset

coreVars == <<
  logicalPoolAlive, physicalPoolAlive,
  recordingRetainsPool, inFlightRetainsPool,
  pendingQueries, rendering, passTimestampActive,
  scopeOpen, inFlight
>>

vars == <<
  logicalPoolAlive, physicalPoolAlive,
  recordingRetainsPool, inFlightRetainsPool,
  pendingQueries, rendering, passTimestampActive,
  scopeOpen, inFlight,
  lastOp, lastOutcome, lastResetRange
>>

QueryRange(firstQuery, queryCount) ==
  IF queryCount = 1 THEN {firstQuery}
  ELSE IF queryCount = 2 THEN {firstQuery, firstQuery + 1}
  ELSE {}

WriteValid(queryIndex) ==
  /\ ~inFlight
  /\ logicalPoolAlive
  /\ queryIndex \in Queries
  /\ queryIndex \notin pendingQueries

ResetValid(commandQueue, firstQuery, queryCount) ==
  LET queries == QueryRange(firstQuery, queryCount) IN
  /\ ~inFlight
  /\ logicalPoolAlive
  /\ commandQueue = PoolQueue
  /\ ~rendering
  /\ ~passTimestampActive
  /\ ~scopeOpen
  /\ queryCount > 0
  /\ firstQuery + queryCount <= Cardinality(Queries)
  /\ queries \subseteq Queries

BeginRenderingValid(withPassTimestamps) ==
  /\ ~inFlight
  /\ ~rendering
  /\ (~withPassTimestamps \/ logicalPoolAlive)

EndRenderingValid ==
  /\ ~inFlight
  /\ rendering
  /\ ~scopeOpen

OpenScopeValid ==
  /\ ~inFlight
  /\ logicalPoolAlive
  /\ ~scopeOpen

CloseScopeValid ==
  /\ ~inFlight
  /\ scopeOpen
  /\ physicalPoolAlive
  /\ recordingRetainsPool

Init ==
  /\ logicalPoolAlive = TRUE
  /\ physicalPoolAlive = TRUE
  /\ recordingRetainsPool = FALSE
  /\ inFlightRetainsPool = FALSE
  /\ pendingQueries = {}
  /\ rendering = FALSE
  /\ passTimestampActive = FALSE
  /\ scopeOpen = FALSE
  /\ inFlight = FALSE
  /\ lastOp = "none"
  /\ lastOutcome = "none"
  /\ lastResetRange = {}

AttemptWrite(queryIndex) ==
  /\ IF WriteValid(queryIndex)
     THEN /\ recordingRetainsPool' = TRUE
          /\ pendingQueries' = pendingQueries \cup {queryIndex}
          /\ UNCHANGED <<
               logicalPoolAlive, physicalPoolAlive, inFlightRetainsPool,
               rendering, passTimestampActive, scopeOpen, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "write"
  /\ lastResetRange' = {}

AttemptReset(commandQueue, firstQuery, queryCount) ==
  LET queries == QueryRange(firstQuery, queryCount) IN
  /\ IF ResetValid(commandQueue, firstQuery, queryCount)
     THEN /\ recordingRetainsPool' = TRUE
          /\ pendingQueries' = pendingQueries \ queries
          /\ UNCHANGED <<
               logicalPoolAlive, physicalPoolAlive, inFlightRetainsPool,
               rendering, passTimestampActive, scopeOpen, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "reset"
  /\ lastResetRange' = queries

AttemptBeginRendering(withPassTimestamps) ==
  /\ IF BeginRenderingValid(withPassTimestamps)
     THEN /\ rendering' = TRUE
          /\ passTimestampActive' = withPassTimestamps
          /\ recordingRetainsPool' = (recordingRetainsPool \/ withPassTimestamps)
          /\ UNCHANGED <<
               logicalPoolAlive, physicalPoolAlive, inFlightRetainsPool,
               pendingQueries, scopeOpen, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "beginRendering"
  /\ lastResetRange' = {}

AttemptEndRendering ==
  /\ IF EndRenderingValid
     THEN /\ rendering' = FALSE
          /\ passTimestampActive' = FALSE
          /\ UNCHANGED <<
               logicalPoolAlive, physicalPoolAlive, recordingRetainsPool,
               inFlightRetainsPool, pendingQueries, scopeOpen, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "endRendering"
  /\ lastResetRange' = {}

AttemptOpenScope ==
  /\ IF OpenScopeValid
     THEN /\ recordingRetainsPool' = TRUE
          /\ scopeOpen' = TRUE
          /\ UNCHANGED <<
               logicalPoolAlive, physicalPoolAlive, inFlightRetainsPool,
               pendingQueries, rendering, passTimestampActive, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "openScope"
  /\ lastResetRange' = {}

AttemptCloseScope ==
  /\ IF CloseScopeValid
     THEN /\ scopeOpen' = FALSE
          /\ UNCHANGED <<
               logicalPoolAlive, physicalPoolAlive, recordingRetainsPool,
               inFlightRetainsPool, pendingQueries, rendering,
               passTimestampActive, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "closeScope"
  /\ lastResetRange' = {}

DestroyPool ==
  /\ logicalPoolAlive
  /\ logicalPoolAlive' = FALSE
  /\ physicalPoolAlive' = (recordingRetainsPool \/ inFlightRetainsPool)
  /\ UNCHANGED <<
       recordingRetainsPool, inFlightRetainsPool, pendingQueries,
       rendering, passTimestampActive, scopeOpen, inFlight
     >>
  /\ lastOp' = "destroy"
  /\ lastOutcome' = "accepted"
  /\ lastResetRange' = {}

Submit ==
  /\ ~inFlight
  /\ ~rendering
  /\ ~passTimestampActive
  /\ ~scopeOpen
  /\ recordingRetainsPool
  /\ recordingRetainsPool' = FALSE
  /\ inFlightRetainsPool' = TRUE
  /\ pendingQueries' = {}
  /\ inFlight' = TRUE
  /\ UNCHANGED <<
       logicalPoolAlive, physicalPoolAlive,
       rendering, passTimestampActive, scopeOpen
     >>
  /\ lastOp' = "submit"
  /\ lastOutcome' = "accepted"
  /\ lastResetRange' = {}

Complete ==
  /\ inFlight
  /\ physicalPoolAlive' = logicalPoolAlive
  /\ inFlightRetainsPool' = FALSE
  /\ inFlight' = FALSE
  /\ UNCHANGED <<
       logicalPoolAlive, recordingRetainsPool, pendingQueries,
       rendering, passTimestampActive, scopeOpen
     >>
  /\ lastOp' = "complete"
  /\ lastOutcome' = "accepted"
  /\ lastResetRange' = {}

Abandon ==
  /\ ~inFlight
  /\ recordingRetainsPool
  /\ physicalPoolAlive' = logicalPoolAlive
  /\ recordingRetainsPool' = FALSE
  /\ pendingQueries' = {}
  /\ rendering' = FALSE
  /\ passTimestampActive' = FALSE
  /\ scopeOpen' = FALSE
  /\ UNCHANGED <<logicalPoolAlive, inFlightRetainsPool, inFlight>>
  /\ lastOp' = "abandon"
  /\ lastOutcome' = "accepted"
  /\ lastResetRange' = {}

Next ==
  \/ \E q \in QueryIndices : AttemptWrite(q)
  \/ \E cq \in Queues, fq \in QueryIndices, qc \in QueryCounts :
       AttemptReset(cq, fq, qc)
  \/ \E ts \in BOOLEAN : AttemptBeginRendering(ts)
  \/ AttemptEndRendering
  \/ AttemptOpenScope
  \/ AttemptCloseScope
  \/ DestroyPool
  \/ Submit
  \/ Complete
  \/ Abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ logicalPoolAlive \in BOOLEAN
  /\ physicalPoolAlive \in BOOLEAN
  /\ recordingRetainsPool \in BOOLEAN
  /\ inFlightRetainsPool \in BOOLEAN
  /\ pendingQueries \in SUBSET Queries
  /\ rendering \in BOOLEAN
  /\ passTimestampActive \in BOOLEAN
  /\ scopeOpen \in BOOLEAN
  /\ inFlight \in BOOLEAN
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes
  /\ lastResetRange \in SUBSET (QueryIndices \cup {3})

PoolPhysicalLifetimeExact ==
  physicalPoolAlive =
    (logicalPoolAlive \/ recordingRetainsPool \/ inFlightRetainsPool)

PassTimestampsRequireRenderingAndRetention ==
  passTimestampActive =>
    /\ rendering
    /\ recordingRetainsPool
    /\ physicalPoolAlive

OpenScopeRetainsPool ==
  scopeOpen =>
    /\ recordingRetainsPool
    /\ physicalPoolAlive

AcceptedResetClearsSelectedRange ==
  lastOp = "reset" /\ lastOutcome = "accepted" =>
    /\ pendingQueries \cap lastResetRange = {}
    /\ recordingRetainsPool
    /\ physicalPoolAlive

Safety ==
  /\ PoolPhysicalLifetimeExact
  /\ PassTimestampsRequireRenderingAndRetention
  /\ OpenScopeRetainsPool
  /\ AcceptedResetClearsSelectedRange

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (lastOp' = "reset" /\ lastOutcome' = "accepted" =>
          /\ ~rendering
          /\ ~passTimestampActive
          /\ ~scopeOpen
          /\ lastResetRange' # {}
          /\ lastResetRange' \subseteq Queries
          /\ pendingQueries' = pendingQueries \ lastResetRange'
          /\ recordingRetainsPool')
    /\ (lastOp' = "write" /\ lastOutcome' = "accepted" =>
          \E q \in Queries :
            /\ q \notin pendingQueries
            /\ pendingQueries' = pendingQueries \cup {q})
  ]_vars

=============================================================================
