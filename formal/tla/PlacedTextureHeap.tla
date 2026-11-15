------------------------- MODULE PlacedTextureHeap --------------------------
(***************************************************************************)
(* Typed non-attachment and attachment texture heaps, aligned interval     *)
(* placement leases, class-compatibility enforcement, command retention    *)
(* across logical destruction, and exact heap retirement.                  *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Resource.cpp (createResourceHeap,                    *)
(*     destroyResourceHeap, createPlacedTexture, reservePlacement,         *)
(*     destroyTexture)                                                     *)
(*   - src/Device.cpp (portable placed-texture facade forwarding)          *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   One 4-unit typed heap (nonAttachment or attachment), two placed       *)
(*   texture slots whose class is chosen at placement time, offsets        *)
(*   {0, 1, 2}, and sizes {1, 2, 3} cover simultaneous non-overlapping     *)
(*   same-class placements, partial-overlap/class-mismatch/stale/unaligned *)
(*   rejection, and deferred physical retirement.                          *)
(***************************************************************************)

EXTENDS Naturals

Classes == {"nonAttachment", "attachment"}
Textures == {0, 1}
HeapCapacity == 4
Alignment == 2
HeapCells == 0..(HeapCapacity - 1)
RequestOffsets == {0, 1, 2}
RequestSizes == {1, 2, 3}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  heapLogical,     \* Public ResourceHeapHandle liveness
  heapPhysical,    \* Native ID3D12Heap allocation liveness
  heapClass,       \* Class of the current/most recent heap
  textureLogical,  \* Public TextureHandle liveness per slot
  texturePhysical, \* Native placed-texture liveness per slot
  textureRetained, \* Command/in-flight retention per slot
  textureSpan,     \* Occupied heap cell interval per physically live texture
  textureClass,    \* Class of each placed texture
  lastOutcome      \* Outcome of the most recent placement or lifecycle step

coreVars == <<
  heapLogical, heapPhysical, heapClass,
  textureLogical, texturePhysical, textureRetained,
  textureSpan, textureClass
>>

vars == <<
  heapLogical, heapPhysical, heapClass,
  textureLogical, texturePhysical, textureRetained,
  textureSpan, textureClass, lastOutcome
>>

AnyPhysical(physMap) ==
  \E t \in Textures : physMap[t]

SpanOf(offset, size) ==
  offset .. (offset + size - 1)

SpanOverlaps(span) ==
  \E t \in Textures :
    /\ texturePhysical[t]
    /\ textureSpan[t] \cap span # {}

CanPlace(texture, reqClass, offset, size) ==
  LET span == SpanOf(offset, size) IN
  /\ heapLogical
  /\ reqClass = heapClass
  /\ ~texturePhysical[texture]
  /\ offset % Alignment = 0
  /\ size > 0
  /\ offset + size <= HeapCapacity
  /\ ~SpanOverlaps(span)

Init ==
  /\ heapLogical = FALSE
  /\ heapPhysical = FALSE
  /\ heapClass = "nonAttachment"
  /\ textureLogical = [t \in Textures |-> FALSE]
  /\ texturePhysical = [t \in Textures |-> FALSE]
  /\ textureRetained = [t \in Textures |-> FALSE]
  /\ textureSpan = [t \in Textures |-> {}]
  /\ textureClass = [t \in Textures |-> "nonAttachment"]
  /\ lastOutcome = "none"

CreateHeap(cls) ==
  /\ ~heapPhysical
  /\ ~AnyPhysical(texturePhysical)
  /\ heapLogical' = TRUE
  /\ heapPhysical' = TRUE
  /\ heapClass' = cls
  /\ UNCHANGED <<
       textureLogical, texturePhysical, textureRetained,
       textureSpan, textureClass
     >>
  /\ lastOutcome' = "accepted"

DestroyHeap ==
  /\ heapLogical
  /\ heapLogical' = FALSE
  /\ heapPhysical' = AnyPhysical(texturePhysical)
  /\ UNCHANGED <<
       heapClass, textureLogical, texturePhysical,
       textureRetained, textureSpan, textureClass
     >>
  /\ lastOutcome' = "accepted"

AttemptPlace(texture, reqClass, offset, size) ==
  LET span == SpanOf(offset, size) IN
  /\ IF CanPlace(texture, reqClass, offset, size)
     THEN /\ heapPhysical' = TRUE
          /\ textureLogical' = [textureLogical EXCEPT ![texture] = TRUE]
          /\ texturePhysical' = [texturePhysical EXCEPT ![texture] = TRUE]
          /\ textureRetained' = [textureRetained EXCEPT ![texture] = FALSE]
          /\ textureSpan' = [textureSpan EXCEPT ![texture] = span]
          /\ textureClass' = [textureClass EXCEPT ![texture] = reqClass]
          /\ UNCHANGED <<heapLogical, heapClass>>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"

RetainTexture(texture) ==
  /\ textureLogical[texture]
  /\ textureRetained' = [textureRetained EXCEPT ![texture] = TRUE]
  /\ UNCHANGED <<
       heapLogical, heapPhysical, heapClass,
       textureLogical, texturePhysical, textureSpan, textureClass
     >>
  /\ lastOutcome' = "accepted"

DestroyTexture(texture) ==
  LET staysPhysical == textureRetained[texture]
      nextPhysical == [texturePhysical EXCEPT ![texture] = staysPhysical]
  IN
  /\ textureLogical[texture]
  /\ heapPhysical' = (heapLogical \/ AnyPhysical(nextPhysical))
  /\ textureLogical' = [textureLogical EXCEPT ![texture] = FALSE]
  /\ texturePhysical' = nextPhysical
  /\ textureSpan' =
       IF staysPhysical THEN textureSpan ELSE [textureSpan EXCEPT ![texture] = {}]
  /\ UNCHANGED <<heapLogical, heapClass, textureRetained, textureClass>>
  /\ lastOutcome' = "accepted"

ReleaseRetention(texture) ==
  LET staysPhysical == textureLogical[texture]
      nextPhysical == [texturePhysical EXCEPT ![texture] = staysPhysical]
  IN
  /\ textureRetained[texture]
  /\ heapPhysical' = (heapLogical \/ AnyPhysical(nextPhysical))
  /\ texturePhysical' = nextPhysical
  /\ textureRetained' = [textureRetained EXCEPT ![texture] = FALSE]
  /\ textureSpan' =
       IF staysPhysical THEN textureSpan ELSE [textureSpan EXCEPT ![texture] = {}]
  /\ UNCHANGED <<heapLogical, heapClass, textureLogical, textureClass>>
  /\ lastOutcome' = "accepted"

Next ==
  \/ \E cls \in Classes : CreateHeap(cls)
  \/ DestroyHeap
  \/ \E t \in Textures, cls \in Classes, off \in RequestOffsets, sz \in RequestSizes :
       AttemptPlace(t, cls, off, sz)
  \/ \E t \in Textures :
       \/ RetainTexture(t)
       \/ DestroyTexture(t)
       \/ ReleaseRetention(t)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ heapLogical \in BOOLEAN
  /\ heapPhysical \in BOOLEAN
  /\ heapClass \in Classes
  /\ textureLogical \in [Textures -> BOOLEAN]
  /\ texturePhysical \in [Textures -> BOOLEAN]
  /\ textureRetained \in [Textures -> BOOLEAN]
  /\ textureSpan \in [Textures -> SUBSET HeapCells]
  /\ textureClass \in [Textures -> Classes]
  /\ lastOutcome \in Outcomes

TextureLifetimeExact ==
  \A t \in Textures :
    texturePhysical[t] = (textureLogical[t] \/ textureRetained[t])

HeapLifetimeExact ==
  heapPhysical = (heapLogical \/ AnyPhysical(texturePhysical))

PhysicalTexturesMatchHeapClass ==
  \A t \in Textures :
    texturePhysical[t] => textureClass[t] = heapClass

LiveSpansAreNonEmptyAndClearedWhenDead ==
  \A t \in Textures :
    texturePhysical[t] <=> textureSpan[t] # {}

PlacementsNeverOverlap ==
  \A left, right \in Textures :
    left # right /\ texturePhysical[left] /\ texturePhysical[right] =>
      textureSpan[left] \cap textureSpan[right] = {}

Safety ==
  /\ TextureLifetimeExact
  /\ HeapLifetimeExact
  /\ PhysicalTexturesMatchHeapClass
  /\ LiveSpansAreNonEmptyAndClearedWhenDead
  /\ PlacementsNeverOverlap

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (heapPhysical /\ heapPhysical' => heapClass' = heapClass)
    /\ (\A t \in Textures :
          ~texturePhysical[t] /\ texturePhysical'[t] =>
            /\ heapLogical
            /\ textureClass'[t] = heapClass
            /\ textureSpan'[t] # {}
            /\ textureSpan'[t] \subseteq HeapCells)
  ]_vars

=============================================================================
