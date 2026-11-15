-------------------------- MODULE MipmapGeneration --------------------------
(***************************************************************************)
(* Compute-based mipmap generation validation, predecessor provenance,     *)
(* sRGB linear-storage-view gating, dimension-specific extent halving,     *)
(* caller-selected final layout transition, compute-state invalidation,    *)
(* staged-recording rollback on abandonment vs queue-submission commit,    *)
(* and texture lifetime retention in the D3D12 backend:                    *)
(* `MipmapGenerator::generate` (`planMipChain`, `mipmapFormats`,           *)
(* `orderedMipChainBarriers`, `dispatchMipLevel`) in                       *)
(* `src/D3D12/D3D12Mipmap.cpp`; declared alternate view-format validation  *)
(* (`validateAlternateViewFormats`, `storageValidationFormat`) and         *)
(* `textureViewSource` in `src/D3D12/D3D12Resource.cpp`; per-mip           *)
(* subresource transitions (`prepareNativeTextureBarrierSequence`) in      *)
(* `src/D3D12/D3D12ResourceBarriers.cpp`; and dependency retention         *)
(* (`retainCommandDependencies`), abandonment, and submission in           *)
(* `src/D3D12/D3D12Queue.cpp` (`include/Rachis/Device.h`).                 *)
(*                                                                         *)
(* Models a three-mip chain (`0..2`) across `1DArray`, `2DArray`, and `3D` *)
(* textures in `linear` and `srgb` color spaces. `attemptStartGeneration`  *)
(* validates queue capability, outside-rendering state, seed definition    *)
(* and layout, sRGB linear-storage-view compatibility, and final layout    *)
(* before invalidating compute state and retaining the texture;            *)
(* `generateNext` halves dimension-appropriate extents step-by-step from   *)
(* immediate-predecessor mip `m - 1`; `finishGeneration` transitions the   *)
(* chain to `targetFinalLayout`; `submit` commits the generated chain, and *)
(* `abandon` rolls staged mip state back to the committed GPU state.       *)
(*                                                                         *)
(* Abstracted: filter kernel weights and multi-step dependency rollback    *)
(* (covered by `MipmapPublication.tla`).                                   *)
(***************************************************************************)
EXTENDS Naturals

Mips == {0, 1, 2}
Seed == 0
LastMip == 2
Payload == 7
Extents == {0, 1, 2, 4}
Dimensions == {"1DArray", "2DArray", "3D"}
ColorSpaces == {"linear", "srgb"}
Encodings == {"undefined", "linear", "srgb"}
Layouts == {"undefined", "copyDestination", "shaderResource", "copySource"}
ValidFinalLayouts == {"shaderResource", "copySource"}
RequestedFinalLayouts == {"shaderResource", "copySource", "undefined"}
Outcomes == {"none", "accepted", "rejected"}

HalveExtent(value) ==
    IF value > 1 THEN value \div 2 ELSE 1

SeedWidth == 4

SeedHeight(dim) ==
    IF dim = "1DArray" THEN 1 ELSE 4

SeedDepth(dim) ==
    IF dim = "3D" THEN 4 ELSE 1

NextHeight(dim, prevH) ==
    IF dim = "1DArray" THEN 1 ELSE HalveExtent(prevH)

NextDepth(dim, prevD) ==
    IF dim = "3D" THEN HalveExtent(prevD) ELSE 1

VARIABLES
    dimension,              \* texture dimension in Dimensions
    colorSpace,             \* format encoding in ColorSpaces
    linearStorageView,      \* TRUE when sRGB texture exposes linear UAV view
    logicalAlive,           \* TRUE while public TextureHandle is live
    physicalAlive,          \* TRUE while native texture allocation is live
    layout,                 \* planned per-mip TextureLayout in [Mips -> Layouts]
    defined,                \* planned per-mip initialized flag in [Mips -> BOOLEAN]
    content,                \* planned per-mip symbolic payload in [Mips -> {0, Payload}]
    encoding,               \* planned per-mip color encoding in [Mips -> Encodings]
    mipW,                   \* planned width per mip in [Mips -> Extents]
    mipH,                   \* planned height per mip in [Mips -> Extents]
    mipD,                   \* planned depth per mip in [Mips -> Extents]
    committedLayout,        \* committed GPU per-mip TextureLayout
    committedDefined,       \* committed GPU per-mip initialized flag
    committedContent,       \* committed GPU per-mip symbolic payload
    committedEncoding,      \* committed GPU per-mip color encoding
    committedMipW,          \* committed GPU width per mip
    committedMipH,          \* committed GPU height per mip
    committedMipD,          \* committed GPU depth per mip
    generating,             \* TRUE while mipmap generation pass is open
    nextMip,                \* next destination mip index in 1..3
    targetFinalLayout,      \* caller-requested final layout in ValidFinalLayouts
    computeStateBound,      \* TRUE while caller compute state is bound
    recordingReference,     \* TRUE while active recording retains texture
    inFlightReference,      \* TRUE while in-flight submission retains texture
    lastGenerationSucceeded,\* TRUE once finishGeneration completes in cycle
    lastOutcome             \* outcome of latest step in Outcomes

committedVars == <<
    committedLayout, committedDefined, committedContent,
    committedEncoding, committedMipW, committedMipH, committedMipD
>>

coreVars == <<
    dimension, colorSpace, linearStorageView, logicalAlive, physicalAlive,
    layout, defined, content, encoding, mipW, mipH, mipD,
    committedLayout, committedDefined, committedContent,
    committedEncoding, committedMipW, committedMipH, committedMipD,
    generating, nextMip, targetFinalLayout, computeStateBound,
    recordingReference, inFlightReference, lastGenerationSucceeded
>>

vars == <<
    dimension, colorSpace, linearStorageView, logicalAlive, physicalAlive,
    layout, defined, content, encoding, mipW, mipH, mipD,
    committedLayout, committedDefined, committedContent,
    committedEncoding, committedMipW, committedMipH, committedMipD,
    generating, nextMip, targetFinalLayout, computeStateBound,
    recordingReference, inFlightReference, lastGenerationSucceeded,
    lastOutcome
>>

Referenced ==
    recordingReference \/ inFlightReference

Init ==
    \E dim \in Dimensions, cs \in ColorSpaces, lsv \in BOOLEAN :
        /\ dimension = dim
        /\ colorSpace = cs
        /\ linearStorageView = lsv
        /\ logicalAlive = TRUE
        /\ physicalAlive = TRUE
        /\ layout = [m \in Mips |-> "undefined"]
        /\ defined = [m \in Mips |-> FALSE]
        /\ content = [m \in Mips |-> 0]
        /\ encoding = [m \in Mips |-> "undefined"]
        /\ mipW = [m \in Mips |-> 0]
        /\ mipH = [m \in Mips |-> 0]
        /\ mipD = [m \in Mips |-> 0]
        /\ committedLayout = [m \in Mips |-> "undefined"]
        /\ committedDefined = [m \in Mips |-> FALSE]
        /\ committedContent = [m \in Mips |-> 0]
        /\ committedEncoding = [m \in Mips |-> "undefined"]
        /\ committedMipW = [m \in Mips |-> 0]
        /\ committedMipH = [m \in Mips |-> 0]
        /\ committedMipD = [m \in Mips |-> 0]
        /\ generating = FALSE
        /\ nextMip = 1
        /\ targetFinalLayout = "shaderResource"
        /\ computeStateBound = FALSE
        /\ recordingReference = FALSE
        /\ inFlightReference = FALSE
        /\ lastGenerationSucceeded = FALSE
        /\ lastOutcome = "none"

uploadSeed ==
    /\ logicalAlive
    /\ ~generating
    /\ ~recordingReference
    /\ ~inFlightReference
    /\ layout[Seed] = "undefined"
    /\ layout' = [layout EXCEPT ![Seed] = "copyDestination"]
    /\ defined' = [defined EXCEPT ![Seed] = TRUE]
    /\ content' = [content EXCEPT ![Seed] = Payload]
    /\ encoding' = [encoding EXCEPT ![Seed] = colorSpace]
    /\ mipW' = [mipW EXCEPT ![Seed] = SeedWidth]
    /\ mipH' = [mipH EXCEPT ![Seed] = SeedHeight(dimension)]
    /\ mipD' = [mipD EXCEPT ![Seed] = SeedDepth(dimension)]
    /\ committedLayout' = [committedLayout EXCEPT ![Seed] = "copyDestination"]
    /\ committedDefined' = [committedDefined EXCEPT ![Seed] = TRUE]
    /\ committedContent' = [committedContent EXCEPT ![Seed] = Payload]
    /\ committedEncoding' = [committedEncoding EXCEPT ![Seed] = colorSpace]
    /\ committedMipW' = [committedMipW EXCEPT ![Seed] = SeedWidth]
    /\ committedMipH' = [committedMipH EXCEPT ![Seed] = SeedHeight(dimension)]
    /\ committedMipD' = [committedMipD EXCEPT ![Seed] = SeedDepth(dimension)]
    /\ lastGenerationSucceeded' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        dimension, colorSpace, linearStorageView, logicalAlive,
        physicalAlive, generating, nextMip, targetFinalLayout,
        computeStateBound, recordingReference, inFlightReference
       >>

bindComputeState ==
    /\ logicalAlive
    /\ ~generating
    /\ ~inFlightReference
    /\ ~computeStateBound
    /\ computeStateBound' = TRUE
    /\ lastGenerationSucceeded' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        dimension, colorSpace, linearStorageView, logicalAlive,
        physicalAlive, layout, defined, content, encoding, mipW, mipH,
        mipD, committedVars, generating, nextMip, targetFinalLayout,
        recordingReference, inFlightReference
       >>

attemptStartGeneration(queueValid, insideRendering, requestedFinalLayout) ==
    LET accepted ==
            /\ logicalAlive
            /\ physicalAlive
            /\ ~generating
            /\ ~inFlightReference
            /\ queueValid
            /\ ~insideRendering
            /\ defined[Seed]
            /\ content[Seed] = Payload
            /\ encoding[Seed] = colorSpace
            /\ layout[Seed] = "copyDestination"
            /\ (colorSpace = "linear" \/ linearStorageView)
            /\ requestedFinalLayout \in ValidFinalLayouts
    IN /\ ~generating
       /\ ~inFlightReference
       /\ IF accepted
          THEN /\ layout' = [layout EXCEPT ![Seed] = "shaderResource"]
               /\ generating' = TRUE
               /\ nextMip' = 1
               /\ targetFinalLayout' = requestedFinalLayout
               /\ computeStateBound' = FALSE
               /\ recordingReference' = TRUE
               /\ lastGenerationSucceeded' = FALSE
               /\ UNCHANGED <<
                    dimension, colorSpace, linearStorageView, logicalAlive,
                    physicalAlive, defined, content, encoding, mipW, mipH,
                    mipD, committedVars, inFlightReference
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

generateNext ==
    /\ generating
    /\ nextMip <= LastMip
    /\ LET prev == nextMip - 1
       IN /\ defined[prev]
          /\ encoding[prev] = colorSpace
          /\ layout[prev] = "shaderResource"
          /\ layout[nextMip] = "undefined"
          /\ layout' = [layout EXCEPT ![nextMip] = "shaderResource"]
          /\ defined' = [defined EXCEPT ![nextMip] = TRUE]
          /\ content' = [content EXCEPT ![nextMip] = content[prev]]
          /\ encoding' = [encoding EXCEPT ![nextMip] = colorSpace]
          /\ mipW' = [mipW EXCEPT ![nextMip] = HalveExtent(mipW[prev])]
          /\ mipH' = [mipH EXCEPT ![nextMip] = NextHeight(dimension, mipH[prev])]
          /\ mipD' = [mipD EXCEPT ![nextMip] = NextDepth(dimension, mipD[prev])]
          /\ nextMip' = nextMip + 1
          /\ lastGenerationSucceeded' = FALSE
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<
               dimension, colorSpace, linearStorageView, logicalAlive,
               physicalAlive, committedVars, generating, targetFinalLayout,
               computeStateBound, recordingReference, inFlightReference
             >>

finishGeneration ==
    /\ generating
    /\ nextMip = LastMip + 1
    /\ \A m \in Mips : defined[m]
    /\ layout' = [m \in Mips |-> targetFinalLayout]
    /\ generating' = FALSE
    /\ lastGenerationSucceeded' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        dimension, colorSpace, linearStorageView, logicalAlive,
        physicalAlive, defined, content, encoding, mipW, mipH, mipD,
        committedVars, nextMip, targetFinalLayout, computeStateBound,
        recordingReference, inFlightReference
       >>

destroy ==
    /\ logicalAlive
    /\ ~generating
    /\ logicalAlive' = FALSE
    /\ physicalAlive' = Referenced
    /\ lastGenerationSucceeded' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        dimension, colorSpace, linearStorageView, layout, defined,
        content, encoding, mipW, mipH, mipD, committedVars, generating,
        nextMip, targetFinalLayout, computeStateBound,
        recordingReference, inFlightReference
       >>

submit ==
    /\ ~generating
    /\ recordingReference
    /\ ~inFlightReference
    /\ recordingReference' = FALSE
    /\ inFlightReference' = TRUE
    /\ committedLayout' = layout
    /\ committedDefined' = defined
    /\ committedContent' = content
    /\ committedEncoding' = encoding
    /\ committedMipW' = mipW
    /\ committedMipH' = mipH
    /\ committedMipD' = mipD
    /\ computeStateBound' = FALSE
    /\ lastGenerationSucceeded' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        dimension, colorSpace, linearStorageView, logicalAlive,
        physicalAlive, layout, defined, content, encoding, mipW, mipH,
        mipD, generating, nextMip, targetFinalLayout
       >>

complete ==
    /\ inFlightReference
    /\ physicalAlive' = (logicalAlive \/ recordingReference)
    /\ inFlightReference' = FALSE
    /\ lastGenerationSucceeded' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        dimension, colorSpace, linearStorageView, logicalAlive, layout,
        defined, content, encoding, mipW, mipH, mipD, committedVars,
        generating, nextMip, targetFinalLayout, computeStateBound,
        recordingReference
       >>

\* Abandoning an open recording discards unsubmitted mipmap generation work
\* and restores the committed GPU mip state.
abandon ==
    /\ ~generating
    /\ recordingReference
    /\ ~inFlightReference
    /\ physicalAlive' = (logicalAlive \/ inFlightReference)
    /\ layout' = committedLayout
    /\ defined' = committedDefined
    /\ content' = committedContent
    /\ encoding' = committedEncoding
    /\ mipW' = committedMipW
    /\ mipH' = committedMipH
    /\ mipD' = committedMipD
    /\ nextMip' = 1
    /\ targetFinalLayout' = "shaderResource"
    /\ recordingReference' = FALSE
    /\ computeStateBound' = FALSE
    /\ lastGenerationSucceeded' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        dimension, colorSpace, linearStorageView, logicalAlive,
        committedVars, generating, inFlightReference
       >>

Next ==
    \/ uploadSeed
    \/ bindComputeState
    \/ \E qv \in BOOLEAN, ir \in BOOLEAN, fl \in RequestedFinalLayouts :
        attemptStartGeneration(qv, ir, fl)
    \/ generateNext
    \/ finishGeneration
    \/ destroy
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ dimension \in Dimensions
    /\ colorSpace \in ColorSpaces
    /\ linearStorageView \in BOOLEAN
    /\ logicalAlive \in BOOLEAN
    /\ physicalAlive \in BOOLEAN
    /\ layout \in [Mips -> Layouts]
    /\ defined \in [Mips -> BOOLEAN]
    /\ content \in [Mips -> {0, Payload}]
    /\ encoding \in [Mips -> Encodings]
    /\ mipW \in [Mips -> Extents]
    /\ mipH \in [Mips -> Extents]
    /\ mipD \in [Mips -> Extents]
    /\ committedLayout \in [Mips -> Layouts]
    /\ committedDefined \in [Mips -> BOOLEAN]
    /\ committedContent \in [Mips -> {0, Payload}]
    /\ committedEncoding \in [Mips -> Encodings]
    /\ committedMipW \in [Mips -> Extents]
    /\ committedMipH \in [Mips -> Extents]
    /\ committedMipD \in [Mips -> Extents]
    /\ generating \in BOOLEAN
    /\ nextMip \in {1, 2, 3}
    /\ targetFinalLayout \in ValidFinalLayouts
    /\ computeStateBound \in BOOLEAN
    /\ recordingReference \in BOOLEAN
    /\ inFlightReference \in BOOLEAN
    /\ lastGenerationSucceeded \in BOOLEAN
    /\ lastOutcome \in Outcomes

UnrecordedStateMatchesCommitted ==
    ~recordingReference =>
        /\ layout = committedLayout
        /\ defined = committedDefined
        /\ content = committedContent
        /\ encoding = committedEncoding
        /\ mipW = committedMipW
        /\ mipH = committedMipH
        /\ mipD = committedMipD

GeneratedMipsHavePredecessorProvenance ==
    /\ \A m \in Mips :
        defined[m] =>
            /\ defined[Seed]
            /\ content[m] = Payload
            /\ encoding[m] = colorSpace
            /\ (m > Seed => defined[m - 1] /\ content[m] = content[m - 1])
    /\ \A m \in Mips :
        committedDefined[m] =>
            /\ committedDefined[Seed]
            /\ committedContent[m] = Payload
            /\ committedEncoding[m] = colorSpace
            /\ (m > Seed =>
                    committedDefined[m - 1]
                    /\ committedContent[m] = committedContent[m - 1])

SrgbGenerationRequiresLinearStorageView ==
    (generating \/ lastGenerationSucceeded) =>
        (colorSpace = "linear" \/ linearStorageView)

GeneratedMipsHaveValidExtents ==
    \A m \in Mips :
        defined[m] =>
            /\ mipW[m] > 0
            /\ mipH[m] > 0
            /\ mipD[m] > 0
            /\ (dimension = "1DArray" => mipH[m] = 1 /\ mipD[m] = 1)
            /\ (dimension = "2DArray" => mipD[m] = 1)
            /\ (m > Seed =>
                    /\ mipW[m] <= mipW[m - 1]
                    /\ mipH[m] <= mipH[m - 1]
                    /\ mipD[m] <= mipD[m - 1]
                    /\ (dimension = "3D" => mipD[m] < mipD[m - 1]))

FinishedChainMatchesRequestedFinalLayout ==
    lastGenerationSucceeded =>
        \A m \in Mips :
            /\ defined[m]
            /\ layout[m] = targetFinalLayout

GenerationInvalidatesComputeState ==
    (generating \/ lastGenerationSucceeded) => ~computeStateBound

TexturePhysicalLifetimeExact ==
    /\ physicalAlive = (logicalAlive \/ Referenced)
    /\ (generating => recordingReference /\ physicalAlive)

Safety ==
    /\ UnrecordedStateMatchesCommitted
    /\ GeneratedMipsHavePredecessorProvenance
    /\ SrgbGenerationRequiresLinearStorageView
    /\ GeneratedMipsHaveValidExtents
    /\ FinishedChainMatchesRequestedFinalLayout
    /\ GenerationInvalidatesComputeState
    /\ TexturePhysicalLifetimeExact

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => UNCHANGED coreVars)
        /\ (lastOutcome' = "accepted" =>
                /\ logicalAlive
                /\ defined[Seed]
                /\ layout[Seed] = "copyDestination"
                /\ ~computeStateBound'
                /\ UNCHANGED committedVars)
    ]_vars

=============================================================================
