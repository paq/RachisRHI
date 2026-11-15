---------------------- MODULE PipelineStatisticsQuery -----------------------
(***************************************************************************)
(* Graphics-queue pipeline-statistics query scopes that begin and end on   *)
(* the same side of a rendering-scope boundary (enclosing whole render     *)
(* passes, or nested inside one), coexistence with render-bound occlusion  *)
(* scopes, unresolved index reuse rejection, aligned 88-byte out-of-pass   *)
(* resolution, and pool/destination retention across logical handle        *)
(* destruction.                                                            *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Query.cpp and src/Vulkan/VulkanQueries.cpp           *)
(*     (beginPipelineStatisticsQuery, endPipelineStatisticsQuery,          *)
(*     resolvePipelineStatisticsQueries)                                   *)
(*   - src/D3D12/D3D12CommandRecording.cpp, src/Vulkan/VulkanGraphics.cpp, *)
(*     and src/Device.cpp (endRendering), with the shared rendering-scope  *)
(*     query checks in src/Internal/PortableValidation.h                   *)
(*   - src/D3D12/D3D12Queue.cpp (command/in-flight resource retention)     *)
(*   - src/Device.cpp (scope-balance checks before queue submission)       *)
(*                                                                         *)
(* EnforceEndRenderingScope keeps endRendering from closing a rendering    *)
(* scope while a statistics query that began inside it is open, and        *)
(* EnforceEndStatisticsScope keeps endStatistics from closing a query on   *)
(* the other side of a rendering-scope boundary from where it began.       *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two query indices, one statistics pool, one 176-byte destination      *)
(*   buffer, and representative aligned/unaligned offsets cover scopes     *)
(*   that enclose or nest in render passes, occlusion nesting,             *)
(*   failure-atomic invalid calls, and deferred physical retirement.       *)
(*   Native counter values are abstracted.                                 *)
(***************************************************************************)

EXTENDS Integers, FiniteSets

CONSTANTS EnforceEndRenderingScope, EnforceEndStatisticsScope

ASSUME EnforceEndRenderingScope \in BOOLEAN
ASSUME EnforceEndStatisticsScope \in BOOLEAN

Pool == 0
Destination == 1
Resources == {Pool, Destination}
Queries == {0, 1}
QueryIndices == {0, 1, 2}
QueryCounts == {0, 1, 2}
DestinationOffsets == {0, 1, 8, 88}
CounterValues == {0, 3}
NoCounter == -1
NoQuery == -1
ResultSize == 88
ResultAlignment == 8
DestinationSize == 176
Operations == {
  "none", "beginStatistics", "endStatistics",
  "beginRendering", "endRendering",
  "beginOcclusion", "endOcclusion",
  "resolve", "destroy", "submit", "complete", "abandon"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  logicalAlive,        \* Public handle liveness for Pool and Destination
  physicalAlive,       \* Native object liveness for Pool and Destination
  recordingReferences, \* Resources retained by open command recording
  inFlightReferences,  \* Resources retained by in-flight submission
  rendering,           \* Whether a graphics render pass is currently open
  occlusionActive,     \* Whether a render-bound occlusion query is open
  activeStatistics,    \* Open statistics query index, or NoQuery
  statisticsInPass,    \* Whether the open statistics query began in rendering
  completedQueries,    \* Ended, unresolved statistics query indices
  aggregateCounters,   \* Recorded counter payload per statistics slot
  inFlight,            \* Whether the command buffer is currently in flight
  lastOp,              \* Most recent operation kind
  lastOutcome,         \* Outcome of most recent operation
  lastResolveRange,    \* Query index set targeted by most recent resolve
  lastResolveOffset    \* Byte offset targeted by most recent resolve

coreVars == <<
  logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
  rendering, occlusionActive, activeStatistics, statisticsInPass,
  completedQueries, aggregateCounters, inFlight
>>

vars == <<
  logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
  rendering, occlusionActive, activeStatistics, statisticsInPass,
  completedQueries, aggregateCounters, inFlight,
  lastOp, lastOutcome, lastResolveRange, lastResolveOffset
>>

Referenced(resource) ==
  resource \in recordingReferences \/ resource \in inFlightReferences

QueryRange(firstQuery, queryCount) ==
  IF queryCount = 1 THEN {firstQuery}
  ELSE IF queryCount = 2 THEN {firstQuery, firstQuery + 1}
  ELSE {}

BeginStatisticsValid(queryIndex) ==
  /\ ~inFlight
  /\ logicalAlive[Pool]
  /\ activeStatistics = NoQuery
  /\ queryIndex \in Queries
  /\ queryIndex \notin completedQueries

EndStatisticsValid ==
  /\ ~inFlight
  /\ activeStatistics \in Queries
  /\ physicalAlive[Pool]
  /\ Pool \in recordingReferences
  /\ (EnforceEndStatisticsScope => rendering = statisticsInPass)

BeginRenderingValid ==
  /\ ~inFlight
  /\ ~rendering

EndRenderingValid ==
  /\ ~inFlight
  /\ rendering
  /\ ~occlusionActive
  /\ (EnforceEndRenderingScope => ~statisticsInPass)

BeginOcclusionValid ==
  /\ ~inFlight
  /\ rendering
  /\ ~occlusionActive

EndOcclusionValid ==
  /\ ~inFlight
  /\ rendering
  /\ occlusionActive

ResolveValid(firstQuery, queryCount, destinationOffset, copyDstUsage) ==
  LET querySet == QueryRange(firstQuery, queryCount) IN
  /\ ~inFlight
  /\ ~rendering
  /\ ~occlusionActive
  /\ activeStatistics = NoQuery
  /\ logicalAlive[Pool]
  /\ logicalAlive[Destination]
  /\ copyDstUsage
  /\ queryCount > 0
  /\ firstQuery + queryCount <= Cardinality(Queries)
  /\ querySet \subseteq Queries
  /\ querySet \subseteq completedQueries
  /\ destinationOffset % ResultAlignment = 0
  /\ destinationOffset + queryCount * ResultSize <= DestinationSize

Init ==
  /\ logicalAlive = [r \in Resources |-> TRUE]
  /\ physicalAlive = [r \in Resources |-> TRUE]
  /\ recordingReferences = {}
  /\ inFlightReferences = {}
  /\ rendering = FALSE
  /\ occlusionActive = FALSE
  /\ activeStatistics = NoQuery
  /\ statisticsInPass = FALSE
  /\ completedQueries = {}
  /\ aggregateCounters = [q \in Queries |-> NoCounter]
  /\ inFlight = FALSE
  /\ lastOp = "none"
  /\ lastOutcome = "none"
  /\ lastResolveRange = {}
  /\ lastResolveOffset = 0

AttemptBeginStatistics(queryIndex) ==
  /\ IF BeginStatisticsValid(queryIndex)
     THEN /\ recordingReferences' = recordingReferences \cup {Pool}
          /\ activeStatistics' = queryIndex
          /\ statisticsInPass' = rendering
          /\ aggregateCounters' = [aggregateCounters EXCEPT ![queryIndex] = NoCounter]
          /\ UNCHANGED <<
               logicalAlive, physicalAlive, inFlightReferences,
               rendering, occlusionActive, completedQueries, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "beginStatistics"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

AttemptEndStatistics(counterValue) ==
  /\ IF EndStatisticsValid
     THEN /\ activeStatistics' = NoQuery
          /\ statisticsInPass' = FALSE
          /\ completedQueries' = completedQueries \cup {activeStatistics}
          /\ aggregateCounters' =
               [aggregateCounters EXCEPT ![activeStatistics] = counterValue]
          /\ UNCHANGED <<
               logicalAlive, physicalAlive, recordingReferences,
               inFlightReferences, rendering, occlusionActive, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "endStatistics"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

AttemptBeginRendering ==
  /\ IF BeginRenderingValid
     THEN /\ rendering' = TRUE
          /\ UNCHANGED <<
               logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
               occlusionActive, activeStatistics, statisticsInPass,
               completedQueries, aggregateCounters, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "beginRendering"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

AttemptEndRendering ==
  /\ IF EndRenderingValid
     THEN /\ rendering' = FALSE
          /\ UNCHANGED <<
               logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
               occlusionActive, activeStatistics, statisticsInPass,
               completedQueries, aggregateCounters, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "endRendering"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

AttemptBeginOcclusion ==
  /\ IF BeginOcclusionValid
     THEN /\ occlusionActive' = TRUE
          /\ UNCHANGED <<
               logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
               rendering, activeStatistics, statisticsInPass,
               completedQueries, aggregateCounters, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "beginOcclusion"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

AttemptEndOcclusion ==
  /\ IF EndOcclusionValid
     THEN /\ occlusionActive' = FALSE
          /\ UNCHANGED <<
               logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
               rendering, activeStatistics, statisticsInPass,
               completedQueries, aggregateCounters, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "endOcclusion"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

AttemptResolve(firstQuery, queryCount, destinationOffset, copyDstUsage) ==
  LET querySet == QueryRange(firstQuery, queryCount) IN
  /\ IF ResolveValid(firstQuery, queryCount, destinationOffset, copyDstUsage)
     THEN /\ recordingReferences' = recordingReferences \cup {Pool, Destination}
          /\ completedQueries' = completedQueries \ querySet
          /\ UNCHANGED <<
               logicalAlive, physicalAlive, inFlightReferences,
               rendering, occlusionActive, activeStatistics,
               statisticsInPass, aggregateCounters, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "resolve"
  /\ lastResolveRange' = querySet
  /\ lastResolveOffset' = destinationOffset

DestroyResource(resource) ==
  /\ logicalAlive[resource]
  /\ logicalAlive' = [logicalAlive EXCEPT ![resource] = FALSE]
  /\ physicalAlive' = [physicalAlive EXCEPT ![resource] = Referenced(resource)]
  /\ UNCHANGED <<
       recordingReferences, inFlightReferences, rendering,
       occlusionActive, activeStatistics, statisticsInPass,
       completedQueries, aggregateCounters, inFlight
     >>
  /\ lastOp' = "destroy"
  /\ lastOutcome' = "accepted"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

Submit ==
  /\ ~inFlight
  /\ ~rendering
  /\ ~occlusionActive
  /\ activeStatistics = NoQuery
  /\ recordingReferences # {}
  /\ recordingReferences' = {}
  /\ inFlightReferences' = recordingReferences
  /\ completedQueries' = {}
  /\ inFlight' = TRUE
  /\ UNCHANGED <<
       logicalAlive, physicalAlive, rendering,
       occlusionActive, activeStatistics, statisticsInPass, aggregateCounters
     >>
  /\ lastOp' = "submit"
  /\ lastOutcome' = "accepted"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

Complete ==
  /\ inFlight
  /\ physicalAlive' = [r \in Resources |-> logicalAlive[r]]
  /\ inFlightReferences' = {}
  /\ inFlight' = FALSE
  /\ UNCHANGED <<
       logicalAlive, recordingReferences, rendering,
       occlusionActive, activeStatistics, statisticsInPass,
       completedQueries, aggregateCounters
     >>
  /\ lastOp' = "complete"
  /\ lastOutcome' = "accepted"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

Abandon ==
  /\ ~inFlight
  /\ physicalAlive' = [r \in Resources |-> logicalAlive[r]]
  /\ recordingReferences' = {}
  /\ rendering' = FALSE
  /\ occlusionActive' = FALSE
  /\ activeStatistics' = NoQuery
  /\ statisticsInPass' = FALSE
  /\ completedQueries' = {}
  /\ UNCHANGED <<logicalAlive, inFlightReferences, aggregateCounters, inFlight>>
  /\ lastOp' = "abandon"
  /\ lastOutcome' = "accepted"
  /\ lastResolveRange' = {}
  /\ lastResolveOffset' = 0

Next ==
  \/ \E q \in QueryIndices : AttemptBeginStatistics(q)
  \/ \E v \in CounterValues : AttemptEndStatistics(v)
  \/ AttemptBeginRendering
  \/ AttemptEndRendering
  \/ AttemptBeginOcclusion
  \/ AttemptEndOcclusion
  \/ \E fq \in QueryIndices,
        qc \in QueryCounts,
        off \in DestinationOffsets,
        usage \in BOOLEAN :
       AttemptResolve(fq, qc, off, usage)
  \/ \E r \in Resources : DestroyResource(r)
  \/ Submit
  \/ Complete
  \/ Abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ logicalAlive \in [Resources -> BOOLEAN]
  /\ physicalAlive \in [Resources -> BOOLEAN]
  /\ recordingReferences \in SUBSET Resources
  /\ inFlightReferences \in SUBSET Resources
  /\ rendering \in BOOLEAN
  /\ occlusionActive \in BOOLEAN
  /\ activeStatistics \in ({NoQuery} \cup Queries)
  /\ statisticsInPass \in BOOLEAN
  /\ completedQueries \in SUBSET Queries
  /\ aggregateCounters \in [Queries -> ({NoCounter} \cup CounterValues)]
  /\ inFlight \in BOOLEAN
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes
  /\ lastResolveRange \in SUBSET (QueryIndices \cup {3})
  /\ lastResolveOffset \in DestinationOffsets

PhysicalLifetimeExact ==
  \A r \in Resources :
    physicalAlive[r] = (logicalAlive[r] \/ Referenced(r))

ActiveStatisticsIsRetainedAndUnresolved ==
  activeStatistics \in Queries =>
    /\ Pool \in recordingReferences
    /\ physicalAlive[Pool]
    /\ activeStatistics \notin completedQueries

OcclusionIsRenderBound ==
  occlusionActive => rendering

InPassStatisticsAreRenderBound ==
  statisticsInPass =>
    /\ activeStatistics \in Queries
    /\ rendering

SubmittedScopesClosed ==
  inFlight =>
    /\ ~rendering
    /\ ~occlusionActive
    /\ activeStatistics = NoQuery
    /\ recordingReferences = {}
    /\ completedQueries = {}

CompletedCountersAreDefined ==
  \A q \in completedQueries :
    aggregateCounters[q] \in CounterValues

AcceptedResolveIsAlignedAndInBounds ==
  lastOp = "resolve" /\ lastOutcome = "accepted" =>
    /\ lastResolveRange # {}
    /\ lastResolveRange \subseteq Queries
    /\ lastResolveOffset % ResultAlignment = 0
    /\ lastResolveOffset + Cardinality(lastResolveRange) * ResultSize <= DestinationSize
    /\ {Pool, Destination} \subseteq recordingReferences

Safety ==
  /\ PhysicalLifetimeExact
  /\ ActiveStatisticsIsRetainedAndUnresolved
  /\ OcclusionIsRenderBound
  /\ InPassStatisticsAreRenderBound
  /\ SubmittedScopesClosed
  /\ CompletedCountersAreDefined
  /\ AcceptedResolveIsAlignedAndInBounds

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (lastOp' = "beginStatistics" /\ lastOutcome' = "accepted" =>
          /\ BeginStatisticsValid(activeStatistics')
          /\ activeStatistics' \in Queries)
    /\ (lastOp' = "endStatistics" /\ lastOutcome' = "accepted" =>
          /\ EndStatisticsValid
          /\ completedQueries' = completedQueries \cup {activeStatistics})
    /\ (lastOp' = "endRendering" /\ lastOutcome' = "accepted" =>
          EndRenderingValid)
    /\ (lastOp' = "resolve" /\ lastOutcome' = "accepted" =>
          /\ ~rendering
          /\ ~occlusionActive
          /\ activeStatistics = NoQuery
          /\ lastResolveRange' \subseteq completedQueries
          /\ completedQueries' = completedQueries \ lastResolveRange')
  ]_vars

StatisticsEndOnBeginSide ==
  [][lastOp' = "endStatistics" /\ lastOutcome' = "accepted" =>
       rendering = statisticsInPass]_vars

=============================================================================
