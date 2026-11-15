-------------------------- MODULE ResourceAliasing --------------------------
(***************************************************************************)
(* Strict vs opt-in aliasable heaps, explicit enhanced-barrier alias       *)
(* activation with write-after validation, discard of prior/new alias      *)
(* interpretations on activation, and mandatory first write before read.   *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Resource.cpp (reservePlacement aliasable check)      *)
(*   - src/D3D12/D3D12ResourceBarriers.cpp (activateBufferAlias,           *)
(*     activateTextureAlias)                                               *)
(*   - src/Device.cpp (portable alias-activation facade forwarding)        *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two placed resources, two heap offsets, strict/aliasable heap modes,  *)
(*   enhanced-barrier capability, and payload values {1, 2} (with 0 as the *)
(*   discarded sentinel) verify exclusive active ownership, discard on     *)
(*   activation, and rejection of reads before the mandatory first write.  *)
(***************************************************************************)

EXTENDS Naturals

Resources == {0, 1}
Offsets == {0, 1}
UndefinedValue == 0
PayloadValues == {1, 2}
StoredDomain == {UndefinedValue} \cup PayloadValues
Operations == {"none", "createHeap", "place", "activate", "write", "read"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  heapExists,       \* Whether the resource heap has been created
  aliasable,        \* Whether the heap opted into overlapping placements
  enhancedBarriers, \* Whether enhanced alias-activation barriers exist
  placed,           \* Whether each resource is placed in the heap
  offsetOf,         \* Heap offset of each placed resource
  active,           \* Whether each placed resource is the active owner
  contentsDefined,  \* Whether the active alias has received a valid write
  storedValue,      \* Defined payload (1..2) or UndefinedValue (0)
  lastObserved,     \* Payload observed by the most recent accepted read
  lastOp,           \* Most recent operation kind
  lastOutcome,      \* Outcome of most recent operation
  lastTarget        \* Target resource of most recent resource operation

coreVars == <<
  heapExists, aliasable, enhancedBarriers,
  placed, offsetOf, active, contentsDefined,
  storedValue, lastObserved
>>

vars == <<
  heapExists, aliasable, enhancedBarriers,
  placed, offsetOf, active, contentsDefined,
  storedValue, lastObserved,
  lastOp, lastOutcome, lastTarget
>>

OverlapsExisting(resource, offset) ==
  \E other \in Resources :
    /\ other # resource
    /\ placed[other]
    /\ offsetOf[other] = offset

CanPlace(resource, offset) ==
  /\ heapExists
  /\ ~placed[resource]
  /\ (aliasable \/ ~OverlapsExisting(resource, offset))

CanActivate(before, after, hasWriteAfter) ==
  /\ heapExists
  /\ aliasable
  /\ enhancedBarriers
  /\ before # after
  /\ hasWriteAfter
  /\ placed[before]
  /\ placed[after]
  /\ offsetOf[before] = offsetOf[after]
  /\ active[before]

CanWrite(resource) ==
  /\ placed[resource]
  /\ active[resource]

CanRead(resource) ==
  /\ placed[resource]
  /\ active[resource]
  /\ contentsDefined[resource]

Init ==
  /\ heapExists = FALSE
  /\ aliasable = FALSE
  /\ enhancedBarriers = FALSE
  /\ placed = [r \in Resources |-> FALSE]
  /\ offsetOf = [r \in Resources |-> 0]
  /\ active = [r \in Resources |-> FALSE]
  /\ contentsDefined = [r \in Resources |-> FALSE]
  /\ storedValue = [r \in Resources |-> UndefinedValue]
  /\ lastObserved = UndefinedValue
  /\ lastOp = "none"
  /\ lastOutcome = "none"
  /\ lastTarget = 0

CreateHeap(allowsAliasing, hasEnhanced) ==
  /\ ~heapExists
  /\ heapExists' = TRUE
  /\ aliasable' = allowsAliasing
  /\ enhancedBarriers' = hasEnhanced
  /\ UNCHANGED <<placed, offsetOf, active, contentsDefined, storedValue, lastObserved>>
  /\ lastOp' = "createHeap"
  /\ lastOutcome' = "accepted"
  /\ lastTarget' = 0

AttemptPlace(resource, offset) ==
  LET aliasesExisting == OverlapsExisting(resource, offset) IN
  /\ IF CanPlace(resource, offset)
     THEN /\ placed' = [placed EXCEPT ![resource] = TRUE]
          /\ offsetOf' = [offsetOf EXCEPT ![resource] = offset]
          /\ active' = [active EXCEPT ![resource] = ~aliasesExisting]
          /\ contentsDefined' = [contentsDefined EXCEPT ![resource] = FALSE]
          /\ storedValue' = [storedValue EXCEPT ![resource] = UndefinedValue]
          /\ UNCHANGED <<heapExists, aliasable, enhancedBarriers, lastObserved>>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "place"
  /\ lastTarget' = resource

AttemptActivate(before, after, hasWriteAfter) ==
  /\ IF CanActivate(before, after, hasWriteAfter)
     THEN /\ active' = [[active EXCEPT ![before] = FALSE] EXCEPT ![after] = TRUE]
          /\ contentsDefined' =
               [[contentsDefined EXCEPT ![before] = FALSE] EXCEPT ![after] = FALSE]
          /\ storedValue' =
               [[storedValue EXCEPT ![before] = UndefinedValue]
                             EXCEPT ![after] = UndefinedValue]
          /\ UNCHANGED <<heapExists, aliasable, enhancedBarriers, placed, offsetOf, lastObserved>>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "activate"
  /\ lastTarget' = after

AttemptWrite(resource, value) ==
  /\ IF CanWrite(resource)
     THEN /\ contentsDefined' = [contentsDefined EXCEPT ![resource] = TRUE]
          /\ storedValue' = [storedValue EXCEPT ![resource] = value]
          /\ UNCHANGED <<
               heapExists, aliasable, enhancedBarriers,
               placed, offsetOf, active, lastObserved
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "write"
  /\ lastTarget' = resource

AttemptRead(resource) ==
  /\ IF CanRead(resource)
     THEN /\ lastObserved' = storedValue[resource]
          /\ UNCHANGED <<
               heapExists, aliasable, enhancedBarriers,
               placed, offsetOf, active, contentsDefined, storedValue
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "read"
  /\ lastTarget' = resource

Next ==
  \/ \E a, eb \in BOOLEAN : CreateHeap(a, eb)
  \/ \E r \in Resources, off \in Offsets : AttemptPlace(r, off)
  \/ \E b, a \in Resources, wr \in BOOLEAN : AttemptActivate(b, a, wr)
  \/ \E r \in Resources, v \in PayloadValues : AttemptWrite(r, v)
  \/ \E r \in Resources : AttemptRead(r)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ heapExists \in BOOLEAN
  /\ aliasable \in BOOLEAN
  /\ enhancedBarriers \in BOOLEAN
  /\ placed \in [Resources -> BOOLEAN]
  /\ offsetOf \in [Resources -> Offsets]
  /\ active \in [Resources -> BOOLEAN]
  /\ contentsDefined \in [Resources -> BOOLEAN]
  /\ storedValue \in [Resources -> StoredDomain]
  /\ lastObserved \in StoredDomain
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes
  /\ lastTarget \in Resources

ActiveResourcesArePlaced ==
  \A r \in Resources : active[r] => placed[r]

StrictHeapsNeverOverlap ==
  ~aliasable =>
    \A left, right \in Resources :
      left # right /\ placed[left] /\ placed[right] =>
        offsetOf[left] # offsetOf[right]

OverlappingAliasesHaveSingleOwner ==
  \A left, right \in Resources :
    left # right /\ placed[left] /\ placed[right] /\ offsetOf[left] = offsetOf[right] =>
      ~(active[left] /\ active[right])

DefinedContentsRequireActiveWrittenOwner ==
  \A r \in Resources :
    contentsDefined[r] <=> (active[r] /\ storedValue[r] \in PayloadValues)

AcceptedReadObservesActiveDefinedPayload ==
  lastOp = "read" /\ lastOutcome = "accepted" =>
    /\ placed[lastTarget]
    /\ active[lastTarget]
    /\ contentsDefined[lastTarget]
    /\ lastObserved = storedValue[lastTarget]
    /\ lastObserved \in PayloadValues

Safety ==
  /\ ActiveResourcesArePlaced
  /\ StrictHeapsNeverOverlap
  /\ OverlappingAliasesHaveSingleOwner
  /\ DefinedContentsRequireActiveWrittenOwner
  /\ AcceptedReadObservesActiveDefinedPayload

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (lastOp' = "activate" /\ lastOutcome' = "accepted" =>
          /\ aliasable
          /\ enhancedBarriers
          /\ active'[lastTarget']
          /\ ~contentsDefined'[lastTarget']
          /\ storedValue'[lastTarget'] = UndefinedValue)
    /\ (lastOp' = "read" =>
          (lastOutcome' = "accepted" <=> CanRead(lastTarget')))
    /\ (lastOp' = "write" =>
          (lastOutcome' = "accepted" <=> CanWrite(lastTarget')))
  ]_vars

=============================================================================
