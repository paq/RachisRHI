---------------------------- MODULE ViewLifetime ----------------------------
(***************************************************************************)
(* Formal safety model for texture-view format compatibility, explicit     *)
(* alternate view-format declaration, and physical texture retention by    *)
(* live texture views:                                                     *)
(*   - include/Rachis/Resource.h (TextureDesc::viewFormats,                *)
(*     TextureViewDesc, isTextureViewFormatCompatible) and                 *)
(*     include/Rachis/Device.h (createTextureView)                         *)
(*   - src/D3D12/D3D12Resource.cpp (validateAlternateViewFormat at texture *)
(*     creation; textureViewSource hands views a shared texture owner;     *)
(*     destroyTexture drops only the handle)                               *)
(*   - src/D3D12/D3D12View.cpp (resolveSupportedViewFormat;                *)
(*     ViewManager::createTextureView keeps the source owner in the view   *)
(*     record)                                                             *)
(*                                                                         *)
(* src/Vulkan/VulkanResources.cpp and src/Metal/MetalResources.mm enforce  *)
(* the same base-or-declared-compatible-alternate rule and keep the parent *)
(* texture alive from each view, but reject destroyTexture while any view  *)
(* is live (liveViews), so they never reach the model's                    *)
(* logical-destruction-before-last-view path.                              *)
(*                                                                         *)
(* Models fallible texture and view creation (rejecting incompatible or    *)
(* undeclared alternate formats atomically) and separates logical handle   *)
(* destruction from independent physical texture collection gated on zero  *)
(* live handles and zero live views.                                       *)
(*                                                                         *)
(* Trusted boundary: native SRV/UAV/RTV/DSV descriptor heap slots while    *)
(* the underlying physical texture allocation remains alive.               *)
(***************************************************************************)
EXTENDS Naturals

Textures == {0, 1}
Views == {0, 1}
\* Formats 0 and 1 share a compatible family; format 2 is standalone.
Formats == {0, 1, 2}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  logicalTextureAlive,  \* Per-texture public handle liveness.
  physicalTextureAlive, \* Per-texture native storage liveness.
  textureFormat,        \* Base format of each physical texture.
  textureHasAlternate,  \* Whether each texture declared its alternate format.
  textureAlternate,     \* Declared alternate format when textureHasAlternate.
  viewAlive,            \* Per-view public/native liveness.
  viewOwner,            \* Owning texture index of each live view.
  viewFormat,           \* Selected view format of each live view.
  lastOutcome           \* Outcome of latest fallible creation request.

coreVars == <<
  logicalTextureAlive,
  physicalTextureAlive,
  textureFormat,
  textureHasAlternate,
  textureAlternate,
  viewAlive,
  viewOwner,
  viewFormat
>>

vars == <<
  logicalTextureAlive,
  physicalTextureAlive,
  textureFormat,
  textureHasAlternate,
  textureAlternate,
  viewAlive,
  viewOwner,
  viewFormat,
  lastOutcome
>>

Compatible(baseFormat, selectedFormat) ==
  \/ baseFormat = selectedFormat
  \/ (baseFormat = 0 /\ selectedFormat = 1)
  \/ (baseFormat = 1 /\ selectedFormat = 0)

AlternateOf(format) == IF format = 0 THEN 1 ELSE 0

HasLiveView(texture) ==
  \E v \in Views : viewAlive[v] /\ viewOwner[v] = texture

Init ==
  /\ logicalTextureAlive = [t \in Textures |-> FALSE]
  /\ physicalTextureAlive = [t \in Textures |-> FALSE]
  /\ textureFormat = [t \in Textures |-> 0]
  /\ textureHasAlternate = [t \in Textures |-> FALSE]
  /\ textureAlternate = [t \in Textures |-> 0]
  /\ viewAlive = [v \in Views |-> FALSE]
  /\ viewOwner = [v \in Views |-> 0]
  /\ viewFormat = [v \in Views |-> 0]
  /\ lastOutcome = "none"

attemptCreateTexture(texture, format, declareAlternate) ==
  /\ ~logicalTextureAlive[texture]
  /\ ~physicalTextureAlive[texture]
  /\ format \in Formats
  /\ declareAlternate \in BOOLEAN
  /\ IF ~declareAlternate \/ format # 2
     THEN /\ logicalTextureAlive' = [logicalTextureAlive EXCEPT ![texture] = TRUE]
          /\ physicalTextureAlive' = [physicalTextureAlive EXCEPT ![texture] = TRUE]
          /\ textureFormat' = [textureFormat EXCEPT ![texture] = format]
          /\ textureHasAlternate' =
               [textureHasAlternate EXCEPT ![texture] = declareAlternate]
          /\ textureAlternate' =
               [textureAlternate EXCEPT ![texture] = AlternateOf(format)]
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<viewAlive, viewOwner, viewFormat>>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptCreateView(view, texture, format) ==
  /\ ~viewAlive[view]
  /\ texture \in Textures
  /\ format \in Formats
  /\ LET valid ==
           /\ logicalTextureAlive[texture]
           /\ physicalTextureAlive[texture]
           /\ Compatible(textureFormat[texture], format)
           /\ (format = textureFormat[texture]
               \/ (textureHasAlternate[texture]
                   /\ format = textureAlternate[texture]))
     IN IF valid
        THEN /\ viewAlive' = [viewAlive EXCEPT ![view] = TRUE]
             /\ viewOwner' = [viewOwner EXCEPT ![view] = texture]
             /\ viewFormat' = [viewFormat EXCEPT ![view] = format]
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  logicalTextureAlive,
                  physicalTextureAlive,
                  textureFormat,
                  textureHasAlternate,
                  textureAlternate
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

destroyTexture(texture) ==
  /\ logicalTextureAlive[texture]
  /\ logicalTextureAlive' = [logicalTextureAlive EXCEPT ![texture] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       physicalTextureAlive,
       textureFormat,
       textureHasAlternate,
       textureAlternate,
       viewAlive,
       viewOwner,
       viewFormat
     >>

destroyView(view) ==
  /\ viewAlive[view]
  /\ viewAlive' = [viewAlive EXCEPT ![view] = FALSE]
  /\ viewOwner' = [viewOwner EXCEPT ![view] = 0]
  /\ viewFormat' = [viewFormat EXCEPT ![view] = 0]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalTextureAlive,
       physicalTextureAlive,
       textureFormat,
       textureHasAlternate,
       textureAlternate
     >>

collectTexture(texture) ==
  /\ physicalTextureAlive[texture]
  /\ ~logicalTextureAlive[texture]
  /\ ~HasLiveView(texture)
  /\ physicalTextureAlive' = [physicalTextureAlive EXCEPT ![texture] = FALSE]
  /\ textureFormat' = [textureFormat EXCEPT ![texture] = 0]
  /\ textureHasAlternate' = [textureHasAlternate EXCEPT ![texture] = FALSE]
  /\ textureAlternate' = [textureAlternate EXCEPT ![texture] = 0]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<logicalTextureAlive, viewAlive, viewOwner, viewFormat>>

Next ==
  \/ \E t \in Textures, f \in Formats, alt \in BOOLEAN :
       attemptCreateTexture(t, f, alt)
  \/ \E v \in Views, t \in Textures, f \in Formats :
       attemptCreateView(v, t, f)
  \/ \E t \in Textures : destroyTexture(t)
  \/ \E v \in Views : destroyView(v)
  \/ \E t \in Textures : collectTexture(t)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ logicalTextureAlive \in [Textures -> BOOLEAN]
  /\ physicalTextureAlive \in [Textures -> BOOLEAN]
  /\ textureFormat \in [Textures -> Formats]
  /\ textureHasAlternate \in [Textures -> BOOLEAN]
  /\ textureAlternate \in [Textures -> Formats]
  /\ viewAlive \in [Views -> BOOLEAN]
  /\ viewOwner \in [Views -> Textures]
  /\ viewFormat \in [Views -> Formats]
  /\ lastOutcome \in Outcomes

LogicalTexturesArePhysical ==
  \A t \in Textures : logicalTextureAlive[t] => physicalTextureAlive[t]

LiveViewsRetainTheirOwner ==
  \A v \in Views : viewAlive[v] => physicalTextureAlive[viewOwner[v]]

LiveViewFormatsAreCompatibleAndDeclared ==
  \A v \in Views :
    viewAlive[v] =>
      /\ Compatible(textureFormat[viewOwner[v]], viewFormat[v])
      /\ (viewFormat[v] = textureFormat[viewOwner[v]]
          \/ (textureHasAlternate[viewOwner[v]]
              /\ viewFormat[v] = textureAlternate[viewOwner[v]]))

AlternateDeclarationsAreCompatible ==
  \A t \in Textures :
    (physicalTextureAlive[t] /\ textureHasAlternate[t]) =>
      Compatible(textureFormat[t], textureAlternate[t])

Safety ==
  /\ LogicalTexturesArePhysical
  /\ LiveViewsRetainTheirOwner
  /\ LiveViewFormatsAreCompatibleAndDeclared
  /\ AlternateDeclarationsAreCompatible

StepSafety ==
  [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
     /\ (\A v \in Views :
           viewAlive[v] /\ viewAlive'[v] =>
             viewOwner'[v] = viewOwner[v] /\ viewFormat'[v] = viewFormat[v])]_vars

=============================================================================
