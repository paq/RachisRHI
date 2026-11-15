-------------------------- MODULE PlacedBufferHeap --------------------------
(***************************************************************************)
(* Explicit buffer heaps, aligned multi-unit interval placement leases,    *)
(* atomic rejection of stale/incompatible/unaligned/overlapping ranges,    *)
(* command retention across logical destruction, and exact heap lifetime.  *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Resource.cpp (createResourceHeap,                    *)
(*     destroyResourceHeap, createPlacedBuffer, reservePlacement,          *)
(*     destroyBuffer)                                                      *)
(*   - src/Device.cpp (portable placed-buffer facade forwarding)           *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   One 4-unit buffer heap with 2-unit alignment, two placed buffer       *)
(*   slots, offsets {0, 1, 2}, and sizes {1, 2, 3} cover aligned disjoint  *)
(*   placements, partial multi-unit overlap rejection ([0,3) vs [2,4)),    *)
(*   out-of-bounds/unaligned/stale/incompatible rejection, and deferred    *)
(*   physical retirement. Driver virtual-address mapping is abstracted.    *)
(***************************************************************************)

EXTENDS Naturals

Buffers == {0, 1}
HeapCapacity == 4
Alignment == 2
HeapCells == 0..(HeapCapacity - 1)
RequestOffsets == {0, 1, 2}
RequestSizes == {1, 2, 3}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  heapLogical,    \* Public ResourceHeapHandle liveness
  heapPhysical,   \* Native ID3D12Heap allocation liveness
  bufferLogical,  \* Public BufferHandle liveness per placed buffer slot
  bufferPhysical, \* Native placed-buffer liveness per slot
  bufferRetained, \* Command/in-flight retention per placed buffer slot
  bufferSpan,     \* Occupied heap cell interval per physically live buffer
  lastOutcome     \* Outcome of the most recent placement or lifecycle step

coreVars == <<
  heapLogical, heapPhysical,
  bufferLogical, bufferPhysical, bufferRetained, bufferSpan
>>

vars == <<
  heapLogical, heapPhysical,
  bufferLogical, bufferPhysical, bufferRetained, bufferSpan,
  lastOutcome
>>

AnyPhysical(physMap) ==
  \E b \in Buffers : physMap[b]

SpanOf(offset, size) ==
  offset .. (offset + size - 1)

SpanOverlaps(span) ==
  \E b \in Buffers :
    /\ bufferPhysical[b]
    /\ bufferSpan[b] \cap span # {}

CanPlace(buffer, offset, size, compatible) ==
  LET span == SpanOf(offset, size) IN
  /\ heapLogical
  /\ compatible
  /\ ~bufferPhysical[buffer]
  /\ offset % Alignment = 0
  /\ size > 0
  /\ offset + size <= HeapCapacity
  /\ ~SpanOverlaps(span)

Init ==
  /\ heapLogical = FALSE
  /\ heapPhysical = FALSE
  /\ bufferLogical = [b \in Buffers |-> FALSE]
  /\ bufferPhysical = [b \in Buffers |-> FALSE]
  /\ bufferRetained = [b \in Buffers |-> FALSE]
  /\ bufferSpan = [b \in Buffers |-> {}]
  /\ lastOutcome = "none"

CreateHeap ==
  /\ ~heapPhysical
  /\ ~AnyPhysical(bufferPhysical)
  /\ heapLogical' = TRUE
  /\ heapPhysical' = TRUE
  /\ UNCHANGED <<bufferLogical, bufferPhysical, bufferRetained, bufferSpan>>
  /\ lastOutcome' = "accepted"

DestroyHeap ==
  /\ heapLogical
  /\ heapLogical' = FALSE
  /\ heapPhysical' = AnyPhysical(bufferPhysical)
  /\ UNCHANGED <<bufferLogical, bufferPhysical, bufferRetained, bufferSpan>>
  /\ lastOutcome' = "accepted"

AttemptPlace(buffer, offset, size, compatible) ==
  LET span == SpanOf(offset, size) IN
  /\ IF CanPlace(buffer, offset, size, compatible)
     THEN /\ heapPhysical' = TRUE
          /\ bufferLogical' = [bufferLogical EXCEPT ![buffer] = TRUE]
          /\ bufferPhysical' = [bufferPhysical EXCEPT ![buffer] = TRUE]
          /\ bufferRetained' = [bufferRetained EXCEPT ![buffer] = FALSE]
          /\ bufferSpan' = [bufferSpan EXCEPT ![buffer] = span]
          /\ UNCHANGED heapLogical
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"

RetainBuffer(buffer) ==
  /\ bufferLogical[buffer]
  /\ bufferRetained' = [bufferRetained EXCEPT ![buffer] = TRUE]
  /\ UNCHANGED <<heapLogical, heapPhysical, bufferLogical, bufferPhysical, bufferSpan>>
  /\ lastOutcome' = "accepted"

DestroyBuffer(buffer) ==
  LET staysPhysical == bufferRetained[buffer]
      nextPhysical == [bufferPhysical EXCEPT ![buffer] = staysPhysical]
  IN
  /\ bufferLogical[buffer]
  /\ heapPhysical' = (heapLogical \/ AnyPhysical(nextPhysical))
  /\ bufferLogical' = [bufferLogical EXCEPT ![buffer] = FALSE]
  /\ bufferPhysical' = nextPhysical
  /\ bufferSpan' =
       IF staysPhysical THEN bufferSpan ELSE [bufferSpan EXCEPT ![buffer] = {}]
  /\ UNCHANGED <<heapLogical, bufferRetained>>
  /\ lastOutcome' = "accepted"

ReleaseRetention(buffer) ==
  LET staysPhysical == bufferLogical[buffer]
      nextPhysical == [bufferPhysical EXCEPT ![buffer] = staysPhysical]
  IN
  /\ bufferRetained[buffer]
  /\ heapPhysical' = (heapLogical \/ AnyPhysical(nextPhysical))
  /\ bufferPhysical' = nextPhysical
  /\ bufferRetained' = [bufferRetained EXCEPT ![buffer] = FALSE]
  /\ bufferSpan' =
       IF staysPhysical THEN bufferSpan ELSE [bufferSpan EXCEPT ![buffer] = {}]
  /\ UNCHANGED <<heapLogical, bufferLogical>>
  /\ lastOutcome' = "accepted"

Next ==
  \/ CreateHeap
  \/ DestroyHeap
  \/ \E b \in Buffers, off \in RequestOffsets, sz \in RequestSizes, compat \in BOOLEAN :
       AttemptPlace(b, off, sz, compat)
  \/ \E b \in Buffers :
       \/ RetainBuffer(b)
       \/ DestroyBuffer(b)
       \/ ReleaseRetention(b)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ heapLogical \in BOOLEAN
  /\ heapPhysical \in BOOLEAN
  /\ bufferLogical \in [Buffers -> BOOLEAN]
  /\ bufferPhysical \in [Buffers -> BOOLEAN]
  /\ bufferRetained \in [Buffers -> BOOLEAN]
  /\ bufferSpan \in [Buffers -> SUBSET HeapCells]
  /\ lastOutcome \in Outcomes

BufferLifetimeExact ==
  \A b \in Buffers :
    bufferPhysical[b] = (bufferLogical[b] \/ bufferRetained[b])

HeapLifetimeExact ==
  heapPhysical = (heapLogical \/ AnyPhysical(bufferPhysical))

LiveSpansAreNonEmptyAndClearedWhenDead ==
  \A b \in Buffers :
    bufferPhysical[b] <=> bufferSpan[b] # {}

PlacementsNeverOverlap ==
  \A left, right \in Buffers :
    left # right /\ bufferPhysical[left] /\ bufferPhysical[right] =>
      bufferSpan[left] \cap bufferSpan[right] = {}

Safety ==
  /\ BufferLifetimeExact
  /\ HeapLifetimeExact
  /\ LiveSpansAreNonEmptyAndClearedWhenDead
  /\ PlacementsNeverOverlap

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (\A b \in Buffers :
          ~bufferPhysical[b] /\ bufferPhysical'[b] =>
            /\ heapLogical
            /\ bufferSpan'[b] # {}
            /\ bufferSpan'[b] \subseteq HeapCells)
  ]_vars

=============================================================================
