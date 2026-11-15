-------------------------- MODULE GpuProfileScope ---------------------------
(***************************************************************************)
(* Nested GPU timestamp/profile scopes, user debug-group hierarchy         *)
(* interleaving, out-of-rendering query resolution and reset, and pool     *)
(* retention in `src/D3D12/D3D12Query.cpp`,                                *)
(* `src/D3D12/D3D12DebugMarker.cpp`, and `src/D3D12/D3D12Queue.cpp`;       *)
(* balanced-scope submission checks in `src/Device.cpp`                    *)
(* (`detail::validateCommandBufferReadyForSubmit` in                       *)
(* `src/Internal/PortableValidation.h`) and `src/D3D12/D3D12Queue.cpp`.    *)
(*                                                                         *)
(* Models interleaved user debug groups and GPU profile scopes up to depth *)
(* `MaxDepth` across two threads, query-slot reservation and end-boundary  *)
(* completion, render-pass gating on query reset and timestamp resolve,    *)
(* balanced-scope enforcement on submit, failure atomicity on every        *)
(* rejected operation, and physical query-pool retention through GPU       *)
(* completion or command abandonment.                                      *)
(*                                                                         *)
(* Abstracted: 64-bit GPU tick values, string label storage, and native    *)
(* command-list allocator internals.                                       *)
(***************************************************************************)
EXTENDS Integers, FiniteSets

MaxDepth == 2
Slots == {0, 1}
Queries == {0, 1, 2, 3}
QueryIndices == {0, 1, 2, 3, 4}
QueryCounts == {0, 1, 2}
Threads == {0, 1}
NoThread == -1
NoOwner == 0
UserOwner == 1
ProfileOwner == 2
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    recording,              \* TRUE while command buffer is in recording state
    submitted,              \* TRUE while command buffer is submitted in flight
    rendering,              \* TRUE while inside a render pass
    logicalPoolAlive,       \* TRUE while public QueryPoolHandle is live
    physicalPoolAlive,      \* TRUE while native timestamp query heap lives
    recordingRetainsPool,   \* TRUE while active recording retains the query pool
    inFlightRetainsPool,    \* TRUE while submitted work retains the query pool
    debugDepth,             \* current combined debug + profile nesting depth
    profileDepth,           \* current active GPU profile scope depth
    ownerThread,            \* thread owning the open scope stack, or NoThread
    groupOwners,            \* per-depth owner tag in [Slots -> {0, 1, 2}]
    profileEnds,            \* reserved end-timestamp slot per profile depth
    pendingQueries,         \* query indices reserved in the current cycle
    writtenQueries,         \* query indices whose timestamp write has been recorded
    lastOutcome             \* outcome of the latest step in Outcomes

coreVars == <<
    recording, submitted, rendering, logicalPoolAlive, physicalPoolAlive,
    recordingRetainsPool, inFlightRetainsPool, debugDepth, profileDepth,
    ownerThread, groupOwners, profileEnds, pendingQueries, writtenQueries
>>

vars == <<
    recording, submitted, rendering, logicalPoolAlive, physicalPoolAlive,
    recordingRetainsPool, inFlightRetainsPool, debugDepth, profileDepth,
    ownerThread, groupOwners, profileEnds, pendingQueries, writtenQueries,
    lastOutcome
>>

TopOwner ==
    IF debugDepth = 1 THEN groupOwners[0]
    ELSE IF debugDepth = 2 THEN groupOwners[1]
    ELSE NoOwner

ActiveEnds ==
    IF profileDepth = 1 THEN {profileEnds[0]}
    ELSE IF profileDepth = 2 THEN {profileEnds[0], profileEnds[1]}
    ELSE {}

QueryRange(firstQuery, queryCount) ==
    IF queryCount = 1 THEN {firstQuery}
    ELSE IF queryCount = 2 THEN {firstQuery, firstQuery + 1}
    ELSE {}

ThreadValid(thread) ==
    debugDepth = 0 \/ ownerThread = thread

UserBeginValid(thread, labelValid, colorValid) ==
    /\ recording
    /\ labelValid
    /\ colorValid
    /\ debugDepth < MaxDepth
    /\ ThreadValid(thread)

UserEndValid(thread) ==
    /\ recording
    /\ debugDepth > 0
    /\ TopOwner = UserOwner
    /\ ownerThread = thread

ProfileBeginValid(thread, labelValid, colorValid, poolValid, beginQuery, endQuery) ==
    /\ recording
    /\ logicalPoolAlive
    /\ poolValid
    /\ labelValid
    /\ colorValid
    /\ debugDepth < MaxDepth
    /\ profileDepth < MaxDepth
    /\ ThreadValid(thread)
    /\ beginQuery \in Queries
    /\ endQuery \in Queries
    /\ beginQuery # endQuery
    /\ beginQuery \notin pendingQueries
    /\ endQuery \notin pendingQueries

ProfileEndValid(thread) ==
    /\ recording
    /\ profileDepth > 0
    /\ debugDepth > 0
    /\ TopOwner = ProfileOwner
    /\ ownerThread = thread
    /\ physicalPoolAlive

Init ==
    /\ recording = TRUE
    /\ submitted = FALSE
    /\ rendering = FALSE
    /\ logicalPoolAlive = TRUE
    /\ physicalPoolAlive = TRUE
    /\ recordingRetainsPool = FALSE
    /\ inFlightRetainsPool = FALSE
    /\ debugDepth = 0
    /\ profileDepth = 0
    /\ ownerThread = NoThread
    /\ groupOwners = [s \in Slots |-> NoOwner]
    /\ profileEnds = [s \in Slots |-> 0]
    /\ pendingQueries = {}
    /\ writtenQueries = {}
    /\ lastOutcome = "none"

beginRendering ==
    /\ recording
    /\ ~rendering
    /\ rendering' = TRUE
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        recording, submitted, logicalPoolAlive, physicalPoolAlive,
        recordingRetainsPool, inFlightRetainsPool, debugDepth, profileDepth,
        ownerThread, groupOwners, profileEnds, pendingQueries, writtenQueries
       >>

endRendering ==
    /\ recording
    /\ rendering
    /\ rendering' = FALSE
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        recording, submitted, logicalPoolAlive, physicalPoolAlive,
        recordingRetainsPool, inFlightRetainsPool, debugDepth, profileDepth,
        ownerThread, groupOwners, profileEnds, pendingQueries, writtenQueries
       >>

attemptUserBegin(thread, labelValid, colorValid) ==
    LET accepted == UserBeginValid(thread, labelValid, colorValid)
    IN /\ recording
       /\ IF accepted
          THEN /\ debugDepth' = debugDepth + 1
               /\ ownerThread' = IF debugDepth = 0 THEN thread ELSE ownerThread
               /\ groupOwners' = [groupOwners EXCEPT ![debugDepth] = UserOwner]
               /\ UNCHANGED <<
                    recording, submitted, rendering, logicalPoolAlive, physicalPoolAlive,
                    recordingRetainsPool, inFlightRetainsPool, profileDepth,
                    profileEnds, pendingQueries, writtenQueries
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptUserEnd(thread) ==
    LET accepted == UserEndValid(thread)
    IN /\ recording
       /\ IF accepted
          THEN /\ debugDepth' = debugDepth - 1
               /\ ownerThread' = IF debugDepth = 1 THEN NoThread ELSE ownerThread
               /\ groupOwners' = [groupOwners EXCEPT ![debugDepth - 1] = NoOwner]
               /\ UNCHANGED <<
                    recording, submitted, rendering, logicalPoolAlive, physicalPoolAlive,
                    recordingRetainsPool, inFlightRetainsPool, profileDepth,
                    profileEnds, pendingQueries, writtenQueries
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptProfileBegin(thread, labelValid, colorValid, poolValid, beginQuery, endQuery) ==
    LET accepted == ProfileBeginValid(thread, labelValid, colorValid, poolValid, beginQuery, endQuery)
    IN /\ recording
       /\ IF accepted
          THEN /\ recordingRetainsPool' = TRUE
               /\ debugDepth' = debugDepth + 1
               /\ profileDepth' = profileDepth + 1
               /\ ownerThread' = IF debugDepth = 0 THEN thread ELSE ownerThread
               /\ groupOwners' = [groupOwners EXCEPT ![debugDepth] = ProfileOwner]
               /\ profileEnds' = [profileEnds EXCEPT ![profileDepth] = endQuery]
               /\ pendingQueries' = pendingQueries \cup {beginQuery, endQuery}
               /\ writtenQueries' = writtenQueries \cup {beginQuery}
               /\ UNCHANGED <<
                    recording, submitted, rendering, logicalPoolAlive,
                    physicalPoolAlive, inFlightRetainsPool
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptProfileEnd(thread) ==
    LET accepted == ProfileEndValid(thread)
        endQuery == IF profileDepth = 1 THEN profileEnds[0] ELSE profileEnds[1]
    IN /\ recording
       /\ IF accepted
          THEN /\ debugDepth' = debugDepth - 1
               /\ profileDepth' = profileDepth - 1
               /\ ownerThread' = IF debugDepth = 1 THEN NoThread ELSE ownerThread
               /\ groupOwners' = [groupOwners EXCEPT ![debugDepth - 1] = NoOwner]
               /\ profileEnds' = [profileEnds EXCEPT ![profileDepth - 1] = 0]
               /\ writtenQueries' = writtenQueries \cup {endQuery}
               /\ UNCHANGED <<
                    recording, submitted, rendering, logicalPoolAlive, physicalPoolAlive,
                    recordingRetainsPool, inFlightRetainsPool, pendingQueries
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptWriteTimestamp(queryIndex) ==
    LET accepted ==
            /\ recording
            /\ logicalPoolAlive
            /\ queryIndex \in Queries
            /\ queryIndex \notin pendingQueries
    IN /\ recording
       /\ IF accepted
          THEN /\ recordingRetainsPool' = TRUE
               /\ pendingQueries' = pendingQueries \cup {queryIndex}
               /\ writtenQueries' = writtenQueries \cup {queryIndex}
               /\ UNCHANGED <<
                    recording, submitted, rendering, logicalPoolAlive, physicalPoolAlive,
                    inFlightRetainsPool, debugDepth, profileDepth,
                    ownerThread, groupOwners, profileEnds
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

\* Resolve rejects inside rendering and while any profile scope remains open.
attemptResolve(firstQuery, queryCount) ==
    LET queries == QueryRange(firstQuery, queryCount)
        accepted ==
            /\ recording
            /\ ~rendering
            /\ logicalPoolAlive
            /\ profileDepth = 0
            /\ queryCount > 0
            /\ firstQuery + queryCount <= Cardinality(Queries)
            /\ queries \subseteq Queries
            /\ queries \subseteq pendingQueries
            /\ queries \subseteq writtenQueries
    IN /\ recording
       /\ IF accepted
          THEN /\ recordingRetainsPool' = TRUE
               /\ pendingQueries' = pendingQueries \ queries
               /\ writtenQueries' = writtenQueries \ queries
               /\ UNCHANGED <<
                    recording, submitted, rendering, logicalPoolAlive, physicalPoolAlive,
                    inFlightRetainsPool, debugDepth, profileDepth,
                    ownerThread, groupOwners, profileEnds
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

\* Resetting query slots rejects inside rendering or while a profile scope is open.
attemptResetPool(firstQuery, queryCount) ==
    LET queries == QueryRange(firstQuery, queryCount)
        accepted ==
            /\ recording
            /\ ~rendering
            /\ logicalPoolAlive
            /\ profileDepth = 0
            /\ queryCount > 0
            /\ firstQuery + queryCount <= Cardinality(Queries)
            /\ queries \subseteq Queries
    IN /\ recording
       /\ IF accepted
          THEN /\ recordingRetainsPool' = TRUE
               /\ pendingQueries' = pendingQueries \ queries
               /\ writtenQueries' = writtenQueries \ queries
               /\ UNCHANGED <<
                    recording, submitted, rendering, logicalPoolAlive, physicalPoolAlive,
                    inFlightRetainsPool, debugDepth, profileDepth,
                    ownerThread, groupOwners, profileEnds
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroyPool ==
    /\ logicalPoolAlive
    /\ logicalPoolAlive' = FALSE
    /\ physicalPoolAlive' = (recordingRetainsPool \/ inFlightRetainsPool)
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        recording, submitted, rendering, recordingRetainsPool, inFlightRetainsPool,
        debugDepth, profileDepth, ownerThread, groupOwners, profileEnds,
        pendingQueries, writtenQueries
       >>

attemptSubmit ==
    LET accepted == ~rendering /\ debugDepth = 0 /\ profileDepth = 0
    IN /\ recording
       /\ IF accepted
          THEN /\ recording' = FALSE
               /\ submitted' = TRUE
               /\ recordingRetainsPool' = FALSE
               /\ inFlightRetainsPool' = recordingRetainsPool
               /\ pendingQueries' = {}
               /\ writtenQueries' = {}
               /\ UNCHANGED <<
                    rendering, logicalPoolAlive, physicalPoolAlive,
                    debugDepth, profileDepth, ownerThread, groupOwners, profileEnds
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

complete ==
    /\ submitted
    /\ submitted' = FALSE
    /\ physicalPoolAlive' = logicalPoolAlive
    /\ inFlightRetainsPool' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        recording, rendering, logicalPoolAlive, recordingRetainsPool,
        debugDepth, profileDepth, ownerThread, groupOwners, profileEnds,
        pendingQueries, writtenQueries
       >>

abandon ==
    /\ recording
    /\ recording' = FALSE
    /\ submitted' = FALSE
    /\ rendering' = FALSE
    /\ physicalPoolAlive' = logicalPoolAlive
    /\ recordingRetainsPool' = FALSE
    /\ inFlightRetainsPool' = FALSE
    /\ debugDepth' = 0
    /\ profileDepth' = 0
    /\ ownerThread' = NoThread
    /\ groupOwners' = [s \in Slots |-> NoOwner]
    /\ profileEnds' = [s \in Slots |-> 0]
    /\ pendingQueries' = {}
    /\ writtenQueries' = {}
    /\ lastOutcome' = "none"
    /\ UNCHANGED logicalPoolAlive

Next ==
    \/ beginRendering
    \/ endRendering
    \/ \E t \in Threads, lv \in BOOLEAN, cv \in BOOLEAN :
        attemptUserBegin(t, lv, cv)
    \/ \E t \in Threads : attemptUserEnd(t)
    \/ \E t \in Threads, lv \in BOOLEAN, cv \in BOOLEAN, pv \in BOOLEAN,
          bq \in QueryIndices, eq \in QueryIndices :
        attemptProfileBegin(t, lv, cv, pv, bq, eq)
    \/ \E t \in Threads : attemptProfileEnd(t)
    \/ \E q \in QueryIndices : attemptWriteTimestamp(q)
    \/ \E fq \in QueryIndices, qc \in QueryCounts : attemptResolve(fq, qc)
    \/ \E fq \in QueryIndices, qc \in QueryCounts : attemptResetPool(fq, qc)
    \/ destroyPool
    \/ attemptSubmit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

ProfileOwnerCount ==
    (IF groupOwners[0] = ProfileOwner THEN 1 ELSE 0)
    + (IF groupOwners[1] = ProfileOwner THEN 1 ELSE 0)

TypeOK ==
    /\ recording \in BOOLEAN
    /\ submitted \in BOOLEAN
    /\ rendering \in BOOLEAN
    /\ logicalPoolAlive \in BOOLEAN
    /\ physicalPoolAlive \in BOOLEAN
    /\ recordingRetainsPool \in BOOLEAN
    /\ inFlightRetainsPool \in BOOLEAN
    /\ debugDepth \in 0..MaxDepth
    /\ profileDepth \in 0..MaxDepth
    /\ ownerThread \in {NoThread} \cup Threads
    /\ groupOwners \in [Slots -> {NoOwner, UserOwner, ProfileOwner}]
    /\ profileEnds \in [Slots -> Queries]
    /\ pendingQueries \in SUBSET Queries
    /\ writtenQueries \in SUBSET Queries
    /\ lastOutcome \in Outcomes

OwnershipMatchesDepth ==
    /\ (debugDepth = 0) = (ownerThread = NoThread)
    /\ \A s \in Slots : (s < debugDepth) = (groupOwners[s] # NoOwner)
    /\ profileDepth = ProfileOwnerCount
    /\ profileDepth <= debugDepth

ActiveEndsReservedNotWritten ==
    /\ ActiveEnds \subseteq pendingQueries
    /\ ActiveEnds \cap writtenQueries = {}

WrittenQueriesArePending ==
    writtenQueries \subseteq pendingQueries

PoolPhysicalLifetimeExact ==
    physicalPoolAlive = (logicalPoolAlive \/ recordingRetainsPool \/ inFlightRetainsPool)

TerminalRecordingIsBalanced ==
    ~recording =>
        /\ ~rendering
        /\ debugDepth = 0
        /\ profileDepth = 0
        /\ ownerThread = NoThread

Safety ==
    /\ OwnershipMatchesDepth
    /\ ActiveEndsReservedNotWritten
    /\ WrittenQueriesArePending
    /\ PoolPhysicalLifetimeExact
    /\ TerminalRecordingIsBalanced

StepSafety ==
    [][
        lastOutcome' = "rejected" => coreVars' = coreVars
    ]_vars

=============================================================================
