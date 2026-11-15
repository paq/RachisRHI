--------------------------- MODULE OcclusionQuery ---------------------------
(***************************************************************************)
(* Binary vs precise occlusion query scoping, render-pass balance, aligned *)
(* out-of-pass query resolution, failure atomicity, and pool/destination   *)
(* lifetime retention in the D3D12 backend: `createOcclusionQueryPool`,    *)
(* `beginOcclusionQuery`, `endOcclusionQuery`, `resolveOcclusionQueries`,  *)
(* and `destroyQueryPool` in `src/D3D12/D3D12Query.cpp`; the portable      *)
(* active-query guard in `Device::endRendering` (`src/Device.cpp`); and    *)
(* command dependency retention and active-query submission rejection in   *)
(* `src/D3D12/D3D12Queue.cpp` (`include/Rachis/Device.h`).                 *)
(*                                                                         *)
(* Models a two-slot occlusion query pool (`Pool = 0`) and resolve target  *)
(* buffer (`Destination = 1`) in `Precise` (`0`) or `Binary` (`1`) mode.   *)
(* `attemptBegin`/`attemptEnd` require an open render pass and forbid      *)
(* overlapping queries; `attemptEndRendering` rejects closing a pass with  *)
(* an open query; `attemptResolve` requires completed queries outside      *)
(* rendering with 8-byte alignment and copy-destination usage; rejected    *)
(* operations leave recorded state unchanged.                              *)
(*                                                                         *)
(* Abstracted: rasterizer sample-counting hardware internals.              *)
(***************************************************************************)
EXTENDS Integers

Resources == {0, 1}
Pool == 0
Destination == 1
PoolCapacity == 2
Queries == {0, 1}
QueryIndices == {0, 1, 2}
QueryCounts == {0, 1, 2}
DestinationOffsets == {0, 1, 8, 16}
PassedSamples == {0, 3}
RecordedValues == {-1, 0, 1, 3}
Modes == {0, 1}
Precise == 0
Binary == 1
NoQuery == -1
ResultSize == 8
DestinationSize == 16
Operations == {
    "none", "beginRendering", "beginQuery", "endQuery", "endRendering",
    "resolve", "destroy", "submit", "complete", "abandon"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    mode,                   \* pool occlusion mode: Precise (0) or Binary (1)
    logicalAlive,           \* public handle liveness per resource
    physicalAlive,          \* native allocation liveness per resource
    recordingReferences,    \* resources retained by active recording
    inFlightReferences,     \* resources retained by in-flight submission
    rendering,              \* TRUE while a render pass is active
    activeQuery,            \* open query index in Queries or NoQuery
    completedQueries,       \* queries ended and awaiting resolve in cycle
    queryValues,            \* recorded value per query in RecordedValues
    inFlight,               \* TRUE while command buffer is in flight
    lastOperation,          \* latest operation tag in Operations
    lastOutcome,            \* outcome of latest operation in Outcomes
    lastResolveCount,       \* queryCount of latest accepted resolve
    lastResolveOffset       \* destinationOffset of latest accepted resolve

coreVars == <<
    mode, logicalAlive, physicalAlive, recordingReferences,
    inFlightReferences, rendering, activeQuery, completedQueries,
    queryValues, inFlight
>>

vars == <<
    mode, logicalAlive, physicalAlive, recordingReferences,
    inFlightReferences, rendering, activeQuery, completedQueries,
    queryValues, inFlight, lastOperation, lastOutcome,
    lastResolveCount, lastResolveOffset
>>

Referenced(resource) ==
    resource \in recordingReferences \/ resource \in inFlightReferences

QueryRange(firstQuery, queryCount) ==
    IF queryCount = 1
    THEN {firstQuery}
    ELSE IF queryCount = 2
         THEN {firstQuery, firstQuery + 1}
         ELSE {}

BeginValid(queryIndex) ==
    /\ ~inFlight
    /\ rendering
    /\ activeQuery = NoQuery
    /\ logicalAlive[Pool]
    /\ queryIndex \in Queries
    /\ queryIndex \notin completedQueries

EndValid ==
    /\ ~inFlight
    /\ rendering
    /\ activeQuery \in Queries
    /\ physicalAlive[Pool]
    /\ Pool \in recordingReferences

EndRenderingValid ==
    /\ ~inFlight
    /\ rendering
    /\ activeQuery = NoQuery

ResolveValid(firstQuery, queryCount, destinationOffset, copyDestinationUsage) ==
    LET querySet == QueryRange(firstQuery, queryCount)
    IN /\ ~inFlight
       /\ ~rendering
       /\ activeQuery = NoQuery
       /\ logicalAlive[Pool]
       /\ logicalAlive[Destination]
       /\ copyDestinationUsage
       /\ queryCount > 0
       /\ firstQuery + queryCount <= PoolCapacity
       /\ querySet \subseteq Queries
       /\ querySet \subseteq completedQueries
       /\ destinationOffset % ResultSize = 0
       /\ destinationOffset + queryCount * ResultSize <= DestinationSize

Init ==
    \E selectedMode \in Modes :
        /\ mode = selectedMode
        /\ logicalAlive = [r \in Resources |-> TRUE]
        /\ physicalAlive = [r \in Resources |-> TRUE]
        /\ recordingReferences = {}
        /\ inFlightReferences = {}
        /\ rendering = FALSE
        /\ activeQuery = NoQuery
        /\ completedQueries = {}
        /\ queryValues = [q \in Queries |-> -1]
        /\ inFlight = FALSE
        /\ lastOperation = "none"
        /\ lastOutcome = "none"
        /\ lastResolveCount = 1
        /\ lastResolveOffset = 0

beginRendering ==
    /\ ~inFlight
    /\ ~rendering
    /\ rendering' = TRUE
    /\ lastOperation' = "beginRendering"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        mode, logicalAlive, physicalAlive, recordingReferences,
        inFlightReferences, activeQuery, completedQueries, queryValues,
        inFlight, lastResolveCount, lastResolveOffset
       >>

attemptBegin(queryIndex) ==
    LET accepted == BeginValid(queryIndex)
    IN /\ ~inFlight
       /\ IF accepted
          THEN /\ recordingReferences' = recordingReferences \cup {Pool}
               /\ activeQuery' = queryIndex
               /\ queryValues' = [queryValues EXCEPT ![queryIndex] = -1]
               /\ UNCHANGED <<
                    mode, logicalAlive, physicalAlive, inFlightReferences,
                    rendering, completedQueries, inFlight
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "beginQuery"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
       /\ UNCHANGED <<lastResolveCount, lastResolveOffset>>

attemptEnd(passedSamples) ==
    LET accepted == EndValid
        value ==
            IF mode = Binary
            THEN IF passedSamples = 0 THEN 0 ELSE 1
            ELSE passedSamples
    IN /\ ~inFlight
       /\ IF accepted
          THEN /\ activeQuery' = NoQuery
               /\ completedQueries' = completedQueries \cup {activeQuery}
               /\ queryValues' = [queryValues EXCEPT ![activeQuery] = value]
               /\ UNCHANGED <<
                    mode, logicalAlive, physicalAlive, recordingReferences,
                    inFlightReferences, rendering, inFlight
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "endQuery"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
       /\ UNCHANGED <<lastResolveCount, lastResolveOffset>>

attemptEndRendering ==
    LET accepted == EndRenderingValid
    IN /\ ~inFlight
       /\ IF accepted
          THEN /\ rendering' = FALSE
               /\ UNCHANGED <<
                    mode, logicalAlive, physicalAlive, recordingReferences,
                    inFlightReferences, activeQuery, completedQueries,
                    queryValues, inFlight
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "endRendering"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
       /\ UNCHANGED <<lastResolveCount, lastResolveOffset>>

attemptResolve(firstQuery, queryCount, destinationOffset, copyDestinationUsage) ==
    LET querySet == QueryRange(firstQuery, queryCount)
        accepted ==
            ResolveValid(
                firstQuery, queryCount, destinationOffset, copyDestinationUsage
            )
    IN /\ ~inFlight
       /\ IF accepted
          THEN /\ recordingReferences' =
                    recordingReferences \cup {Pool, Destination}
               /\ completedQueries' = completedQueries \ querySet
               /\ lastResolveCount' = queryCount
               /\ lastResolveOffset' = destinationOffset
               /\ UNCHANGED <<
                    mode, logicalAlive, physicalAlive, inFlightReferences,
                    rendering, activeQuery, queryValues, inFlight
                  >>
          ELSE /\ UNCHANGED coreVars
               /\ UNCHANGED <<lastResolveCount, lastResolveOffset>>
       /\ lastOperation' = "resolve"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroy(resource) ==
    /\ logicalAlive[resource]
    /\ logicalAlive' = [logicalAlive EXCEPT ![resource] = FALSE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![resource] = Referenced(resource)]
    /\ lastOperation' = "destroy"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        mode, recordingReferences, inFlightReferences, rendering,
        activeQuery, completedQueries, queryValues, inFlight,
        lastResolveCount, lastResolveOffset
       >>

submit ==
    /\ ~inFlight
    /\ ~rendering
    /\ activeQuery = NoQuery
    /\ recordingReferences # {}
    /\ recordingReferences' = {}
    /\ inFlightReferences' = recordingReferences
    /\ completedQueries' = {}
    /\ inFlight' = TRUE
    /\ lastOperation' = "submit"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        mode, logicalAlive, physicalAlive, rendering, activeQuery,
        queryValues, lastResolveCount, lastResolveOffset
       >>

complete ==
    /\ inFlight
    /\ physicalAlive' =
        [r \in Resources |-> logicalAlive[r] \/ r \in recordingReferences]
    /\ inFlightReferences' = {}
    /\ inFlight' = FALSE
    /\ lastOperation' = "complete"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        mode, logicalAlive, recordingReferences, rendering, activeQuery,
        completedQueries, queryValues, lastResolveCount, lastResolveOffset
       >>

abandon ==
    /\ ~inFlight
    /\ (rendering \/ recordingReferences # {} \/ completedQueries # {})
    /\ physicalAlive' =
        [r \in Resources |-> logicalAlive[r] \/ r \in inFlightReferences]
    /\ recordingReferences' = {}
    /\ rendering' = FALSE
    /\ activeQuery' = NoQuery
    /\ completedQueries' = {}
    /\ lastOperation' = "abandon"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        mode, logicalAlive, inFlightReferences, queryValues, inFlight,
        lastResolveCount, lastResolveOffset
       >>

Next ==
    \/ beginRendering
    \/ \E qi \in QueryIndices : attemptBegin(qi)
    \/ \E ps \in PassedSamples : attemptEnd(ps)
    \/ attemptEndRendering
    \/ \E fq \in QueryIndices, qc \in QueryCounts,
          off \in DestinationOffsets, cdu \in BOOLEAN :
            attemptResolve(fq, qc, off, cdu)
    \/ \E r \in Resources : destroy(r)
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ mode \in Modes
    /\ logicalAlive \in [Resources -> BOOLEAN]
    /\ physicalAlive \in [Resources -> BOOLEAN]
    /\ recordingReferences \in SUBSET Resources
    /\ inFlightReferences \in SUBSET Resources
    /\ rendering \in BOOLEAN
    /\ activeQuery \in Queries \cup {NoQuery}
    /\ completedQueries \in SUBSET Queries
    /\ queryValues \in [Queries -> RecordedValues]
    /\ inFlight \in BOOLEAN
    /\ lastOperation \in Operations
    /\ lastOutcome \in Outcomes
    /\ lastResolveCount \in {1, 2}
    /\ lastResolveOffset \in DestinationOffsets

ActiveAndCompletedQueriesRetained ==
    /\ (activeQuery \in Queries =>
            /\ rendering
            /\ activeQuery \notin completedQueries
            /\ Pool \in recordingReferences
            /\ physicalAlive[Pool])
    /\ (completedQueries # {} =>
            /\ Pool \in recordingReferences
            /\ physicalAlive[Pool])

CompletedValuesMatchMode ==
    \A q \in completedQueries :
        /\ (mode = Binary => queryValues[q] \in {0, 1})
        /\ (mode = Precise => queryValues[q] \in PassedSamples)

AcceptedResolveAlignedAndInRange ==
    (lastOperation = "resolve" /\ lastOutcome = "accepted") =>
        /\ lastResolveCount \in {1, 2}
        /\ lastResolveOffset % ResultSize = 0
        /\ lastResolveOffset + lastResolveCount * ResultSize <= DestinationSize
        /\ {Pool, Destination} \subseteq recordingReferences

ResourcePhysicalLifetimeExact ==
    \A r \in Resources :
        physicalAlive[r] = (logicalAlive[r] \/ Referenced(r))

Safety ==
    /\ ActiveAndCompletedQueriesRetained
    /\ CompletedValuesMatchMode
    /\ AcceptedResolveAlignedAndInRange
    /\ ResourcePhysicalLifetimeExact

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
        /\ (lastOperation' = "endQuery" =>
                ((lastOutcome' = "accepted") = EndValid))
        /\ (lastOperation' = "endRendering" =>
                ((lastOutcome' = "accepted") = EndRenderingValid))
        /\ ((lastOperation' = "resolve" /\ lastOutcome' = "accepted") =>
                /\ ~rendering
                /\ activeQuery = NoQuery
                /\ logicalAlive[Pool]
                /\ logicalAlive[Destination])
    ]_vars

=============================================================================
