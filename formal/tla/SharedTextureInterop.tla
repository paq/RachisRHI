------------------------ MODULE SharedTextureInterop ------------------------
(***************************************************************************)
(* Formal safety model for cross-device shared committed textures, common  *)
(* handoff layout discipline, queue ownership transfers, and shared        *)
(* timeline-fence handoffs:                                                *)
(*   - include/Rachis/Interop.h, src/Interop.cpp (SharedHandle ownership)  *)
(*   - src/D3D12/D3D12Resource.cpp (createSharedTexture,                   *)
(*     exportSharedTexture, importSharedTexture)                           *)
(*   - src/D3D12/D3D12Queue.cpp (exportTimelinePoint,                      *)
(*     enqueueSharedTimelineWait)                                          *)
(*   - include/Rachis/Resource.h (TextureBarrier::ownership),              *)
(*     src/Vulkan/VulkanBarriers.cpp (release to and acquire from          *)
(*     VK_QUEUE_FAMILY_EXTERNAL), src/D3D12/D3D12ResourceBarriers.cpp      *)
(*     (validated no-op)                                                   *)
(*                                                                         *)
(* Models active non-common GPU layouts ("renderTarget" for writes and     *)
(* "shaderResource" for reads), mandatory transitions back to "common"     *)
(* before cross-device export/handoff, the release of queue ownership to   *)
(* the external owner before each handoff and its acquire by the receiving *)
(* party, shareability gating, fence handle close before and after wait    *)
(* enqueue, concurrent-write exclusion, and atomic rejection of invalid    *)
(* descriptors/transports/layouts. The common handoff layout and the       *)
(* release are caller barrier obligations: exportSharedTexture checks only *)
(* that the texture was created or imported as shared, and                 *)
(* exportTimelinePoint does not know which resources a handoff carries.    *)
(* Ownership barriers themselves are rejected unless the texture is shared *)
(* and both layouts are "common".                                          *)
(*                                                                         *)
(* Design-rule constant (TRUE in SharedTextureInterop.cfg, FALSE in        *)
(* SharedTextureInterop-noacquire.cfg):                                    *)
(*   - RequireOwnershipTransfer: a party records GPU access or a layout    *)
(*     transition only while it holds queue ownership, so the receiving    *)
(*     party acquires the texture from the external owner first.           *)
(*                                                                         *)
(* Trusted boundary: OS kernel handle duplication and native D3D12 shared  *)
(* texture/fence and Vulkan external-memory execution once cross-device    *)
(* waits are enqueued.                                                     *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RequireOwnershipTransfer

ASSUME RequireOwnershipTransfer \in BOOLEAN

Parties == {"producer", "consumer"}
AllocationKinds == {"none", "sharedCommitted", "nonShareable"}
TextureCandidates == {"sharedCommitted", "nonShareable"}
Layouts == {"none", "common", "renderTarget", "shaderResource"}
ActiveLayouts == {"common", "renderTarget", "shaderResource"}
Owners == {"none", "producer", "consumer", "external"}
PendingKinds == {"none", "write", "read"}
Contents == {"empty", "producer", "consumer"}
Outcomes == {"none", "accepted", "rejected"}
MaxTimeline == 2

TimelineRecords ==
  [submitted : 0..MaxTimeline, completed : 0..MaxTimeline, exported : 0..MaxTimeline]

FenceRecords ==
  [open : BOOLEAN, retained : BOOLEAN, waitValue : 0..MaxTimeline]

VARIABLES
  logicalAlive,         \* Per-party public texture handle liveness.
  physicalAlive,        \* Whether underlying texture storage exists.
  allocationKind,       \* Texture storage class (none, sharedCommitted, nonShareable).
  textureTransportOpen, \* Whether exported shared texture transport handle is open.
  layout,               \* Current texture layout across active GPU and handoff states.
  owner,                \* Queue owner of the texture (none, producer, consumer, external).
  handoffInFlight,      \* True while a cross-device handoff awaits synchronized acquire.
  lastExportLayout,     \* Layout observed at the latest accepted export/handoff.
  timeline,             \* Per-party submitted/completed/exported timeline state.
  producerFence,        \* Fence exported by producer and awaited by consumer.
  consumerFence,        \* Fence exported by consumer and awaited by producer.
  pending,              \* Per-party in-flight GPU access kind ("none", "write", "read").
  content,              \* Current committed GPU payload in the texture.
  observed,             \* Payload observed by each party's completed read.
  lastOutcome           \* Outcome of the latest fallible interop request.

coreVars == <<
  logicalAlive,
  physicalAlive,
  allocationKind,
  textureTransportOpen,
  layout,
  owner,
  handoffInFlight,
  lastExportLayout,
  timeline,
  producerFence,
  consumerFence,
  pending,
  content,
  observed
>>

vars == <<
  logicalAlive,
  physicalAlive,
  allocationKind,
  textureTransportOpen,
  layout,
  owner,
  handoffInFlight,
  lastExportLayout,
  timeline,
  producerFence,
  consumerFence,
  pending,
  content,
  observed,
  lastOutcome
>>

ProducerOwnsLayout ==
  /\ logicalAlive["producer"]
  /\ ~handoffInFlight
  /\ \/ timeline["producer"].exported = 0
     \/ (consumerFence.waitValue = 2 /\ timeline["consumer"].completed >= 2)

ConsumerOwnsLayout ==
  /\ logicalAlive["consumer"]
  /\ ~handoffInFlight
  /\ producerFence.waitValue = 1
  /\ timeline["producer"].completed >= 1
  /\ timeline["consumer"].exported = 0

PartyOwnsLayout(party) ==
  IF party = "producer" THEN ProducerOwnsLayout ELSE ConsumerOwnsLayout

\* GPU access and layout transitions require the party to hold queue ownership.
HoldsOwnership(party) == ~RequireOwnershipTransfer \/ owner = party

Init ==
  /\ logicalAlive = [party \in Parties |-> FALSE]
  /\ physicalAlive = FALSE
  /\ allocationKind = "none"
  /\ textureTransportOpen = FALSE
  /\ layout = "none"
  /\ owner = "none"
  /\ handoffInFlight = FALSE
  /\ lastExportLayout = "none"
  /\ timeline = [party \in Parties |-> [submitted |-> 0, completed |-> 0, exported |-> 0]]
  /\ producerFence = [open |-> FALSE, retained |-> FALSE, waitValue |-> 0]
  /\ consumerFence = [open |-> FALSE, retained |-> FALSE, waitValue |-> 0]
  /\ pending = [party \in Parties |-> "none"]
  /\ content = "empty"
  /\ observed = [party \in Parties |-> "empty"]
  /\ lastOutcome = "none"

\* One texture lifetime: timelines and fences are not reset for a second texture.
createTexture(kind) ==
  /\ ~physicalAlive
  /\ layout = "none"
  /\ kind \in TextureCandidates
  /\ logicalAlive' = [logicalAlive EXCEPT !["producer"] = TRUE]
  /\ physicalAlive' = TRUE
  /\ allocationKind' = kind
  /\ layout' = "common"
  /\ owner' = "producer"
  /\ lastOutcome' = "accepted"
  /\ UNCHANGED <<
       textureTransportOpen,
       handoffInFlight,
       lastExportLayout,
       timeline,
       producerFence,
       consumerFence,
       pending,
       content,
       observed
     >>

transitionLayout(party, nextLayout) ==
  /\ physicalAlive
  /\ nextLayout \in ActiveLayouts
  /\ layout # nextLayout
  /\ pending["producer"] = "none"
  /\ pending["consumer"] = "none"
  /\ PartyOwnsLayout(party)
  /\ HoldsOwnership(party)
  /\ layout' = nextLayout
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       owner,
       handoffInFlight,
       lastExportLayout,
       timeline,
       producerFence,
       consumerFence,
       pending,
       content,
       observed
     >>

\* A release is a common-to-common barrier with OwnershipTransfer::releaseExternal.
\* The barrier API rejects it on a texture that is not shared or at another layout.
attemptReleaseOwnership(party) ==
  /\ physicalAlive
  /\ owner = party
  /\ pending[party] = "none"
  /\ PartyOwnsLayout(party)
  /\ IF allocationKind = "sharedCommitted" /\ layout = "common"
     THEN /\ owner' = "external"
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               logicalAlive,
               physicalAlive,
               allocationKind,
               textureTransportOpen,
               layout,
               handoffInFlight,
               lastExportLayout,
               timeline,
               producerFence,
               consumerFence,
               pending,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

\* An acquire is the matching barrier with OwnershipTransfer::acquireExternal,
\* recorded by the receiving party after its wait for the handoff point.
attemptAcquireOwnership(party) ==
  /\ physicalAlive
  /\ owner = "external"
  /\ pending[party] = "none"
  /\ PartyOwnsLayout(party)
  /\ IF allocationKind = "sharedCommitted" /\ layout = "common"
     THEN /\ owner' = party
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               logicalAlive,
               physicalAlive,
               allocationKind,
               textureTransportOpen,
               layout,
               handoffInFlight,
               lastExportLayout,
               timeline,
               producerFence,
               consumerFence,
               pending,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

\* Exporting requires a shareable committed texture at the common handoff layout.
attemptExportTexture ==
  /\ logicalAlive["producer"]
  /\ ~textureTransportOpen
  /\ IF physicalAlive
        /\ allocationKind = "sharedCommitted"
        /\ layout = "common"
        /\ pending["producer"] = "none"
     THEN /\ textureTransportOpen' = TRUE
          /\ lastExportLayout' = layout
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               logicalAlive,
               physicalAlive,
               allocationKind,
               layout,
               owner,
               handoffInFlight,
               timeline,
               producerFence,
               consumerFence,
               pending,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptImportTexture(descriptorMatches) ==
  /\ ~logicalAlive["consumer"]
  /\ IF textureTransportOpen /\ physicalAlive /\ allocationKind = "sharedCommitted" /\ descriptorMatches
     THEN /\ logicalAlive' = [logicalAlive EXCEPT !["consumer"] = TRUE]
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               physicalAlive,
               allocationKind,
               textureTransportOpen,
               layout,
               owner,
               handoffInFlight,
               lastExportLayout,
               timeline,
               producerFence,
               consumerFence,
               pending,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

closeTextureTransport ==
  /\ textureTransportOpen
  /\ textureTransportOpen' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       layout,
       owner,
       handoffInFlight,
       lastExportLayout,
       timeline,
       producerFence,
       consumerFence,
       pending,
       content,
       observed
     >>

submitProducerWrite ==
  LET current == timeline["producer"] IN
  /\ logicalAlive["producer"]
  /\ physicalAlive
  /\ ProducerOwnsLayout
  /\ HoldsOwnership("producer")
  /\ layout = "renderTarget"
  /\ current.submitted = 0
  /\ pending["producer"] = "none"
  /\ timeline' = [timeline EXCEPT !["producer"] = [
       submitted |-> 1,
       completed |-> current.completed,
       exported |-> current.exported
     ]]
  /\ pending' = [pending EXCEPT !["producer"] = "write"]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       handoffInFlight,
       lastExportLayout,
       producerFence,
       consumerFence,
       content,
       observed
     >>

\* The handoff point must follow the release; exportTimelinePoint cannot check it.
attemptExportProducerPoint ==
  LET current == timeline["producer"] IN
  /\ logicalAlive["producer"]
  /\ current.submitted = 1
  /\ current.exported = 0
  /\ ~producerFence.open
  /\ producerFence.waitValue = 0
  /\ IF layout = "common"
        /\ owner = "external"
        /\ pending["producer"] = "none"
        /\ allocationKind = "sharedCommitted"
     THEN /\ timeline' = [timeline EXCEPT !["producer"] = [
               submitted |-> current.submitted,
               completed |-> current.completed,
               exported |-> current.submitted
             ]]
          /\ producerFence' = [open |-> TRUE, retained |-> FALSE, waitValue |-> 0]
          /\ handoffInFlight' = TRUE
          /\ lastExportLayout' = layout
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               logicalAlive,
               physicalAlive,
               allocationKind,
               textureTransportOpen,
               layout,
               owner,
               consumerFence,
               pending,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptEnqueueConsumerWait(nonzeroValue) ==
  /\ logicalAlive["consumer"]
  /\ producerFence.waitValue = 0
  /\ IF producerFence.open /\ timeline["producer"].exported > 0 /\ nonzeroValue
     THEN /\ producerFence' = [
               open |-> producerFence.open,
               retained |-> TRUE,
               waitValue |-> timeline["producer"].exported
             ]
          /\ handoffInFlight' = (timeline["producer"].completed < timeline["producer"].exported)
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               logicalAlive,
               physicalAlive,
               allocationKind,
               textureTransportOpen,
               layout,
               owner,
               lastExportLayout,
               timeline,
               consumerFence,
               pending,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

closeProducerFenceHandle ==
  /\ producerFence.open
  /\ producerFence' = [
       open |-> FALSE,
       retained |-> producerFence.retained,
       waitValue |-> producerFence.waitValue
     ]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       handoffInFlight,
       lastExportLayout,
       timeline,
       consumerFence,
       pending,
       content,
       observed
     >>

submitConsumerRead ==
  LET current == timeline["consumer"] IN
  /\ logicalAlive["consumer"]
  /\ physicalAlive
  /\ ConsumerOwnsLayout
  /\ HoldsOwnership("consumer")
  /\ layout = "shaderResource"
  /\ current.submitted = 0
  /\ producerFence.waitValue > 0
  /\ pending["consumer"] = "none"
  /\ timeline' = [timeline EXCEPT !["consumer"] = [
       submitted |-> 1,
       completed |-> current.completed,
       exported |-> current.exported
     ]]
  /\ pending' = [pending EXCEPT !["consumer"] = "read"]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       handoffInFlight,
       lastExportLayout,
       producerFence,
       consumerFence,
       content,
       observed
     >>

submitConsumerWrite ==
  LET current == timeline["consumer"] IN
  /\ logicalAlive["consumer"]
  /\ physicalAlive
  /\ ConsumerOwnsLayout
  /\ HoldsOwnership("consumer")
  /\ layout = "renderTarget"
  /\ current.submitted = 1
  /\ current.completed = 1
  /\ pending["consumer"] = "none"
  /\ timeline' = [timeline EXCEPT !["consumer"] = [
       submitted |-> 2,
       completed |-> current.completed,
       exported |-> current.exported
     ]]
  /\ pending' = [pending EXCEPT !["consumer"] = "write"]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       handoffInFlight,
       lastExportLayout,
       producerFence,
       consumerFence,
       content,
       observed
     >>

attemptExportConsumerPoint ==
  LET current == timeline["consumer"] IN
  /\ logicalAlive["consumer"]
  /\ current.submitted = 2
  /\ current.exported = 0
  /\ ~consumerFence.open
  /\ consumerFence.waitValue = 0
  /\ IF layout = "common"
        /\ owner = "external"
        /\ pending["consumer"] = "none"
        /\ allocationKind = "sharedCommitted"
     THEN /\ timeline' = [timeline EXCEPT !["consumer"] = [
               submitted |-> current.submitted,
               completed |-> current.completed,
               exported |-> current.submitted
             ]]
          /\ consumerFence' = [open |-> TRUE, retained |-> FALSE, waitValue |-> 0]
          /\ handoffInFlight' = TRUE
          /\ lastExportLayout' = layout
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               logicalAlive,
               physicalAlive,
               allocationKind,
               textureTransportOpen,
               layout,
               owner,
               producerFence,
               pending,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptEnqueueProducerWait(nonzeroValue) ==
  /\ logicalAlive["producer"]
  /\ consumerFence.waitValue = 0
  /\ IF consumerFence.open /\ timeline["consumer"].exported > 0 /\ nonzeroValue
     THEN /\ consumerFence' = [
               open |-> consumerFence.open,
               retained |-> TRUE,
               waitValue |-> timeline["consumer"].exported
             ]
          /\ handoffInFlight' = (timeline["consumer"].completed < timeline["consumer"].exported)
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               logicalAlive,
               physicalAlive,
               allocationKind,
               textureTransportOpen,
               layout,
               owner,
               lastExportLayout,
               timeline,
               producerFence,
               pending,
               content,
               observed
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

closeConsumerFenceHandle ==
  /\ consumerFence.open
  /\ consumerFence' = [
       open |-> FALSE,
       retained |-> consumerFence.retained,
       waitValue |-> consumerFence.waitValue
     ]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       handoffInFlight,
       lastExportLayout,
       timeline,
       producerFence,
       pending,
       content,
       observed
     >>

submitProducerRead ==
  LET current == timeline["producer"] IN
  /\ logicalAlive["producer"]
  /\ physicalAlive
  /\ ProducerOwnsLayout
  /\ HoldsOwnership("producer")
  /\ layout = "shaderResource"
  /\ current.submitted = 1
  /\ current.completed = 1
  /\ consumerFence.waitValue > 0
  /\ pending["producer"] = "none"
  /\ timeline' = [timeline EXCEPT !["producer"] = [
       submitted |-> 2,
       completed |-> current.completed,
       exported |-> current.exported
     ]]
  /\ pending' = [pending EXCEPT !["producer"] = "read"]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       handoffInFlight,
       lastExportLayout,
       producerFence,
       consumerFence,
       content,
       observed
     >>

completeProducer ==
  LET current == timeline["producer"]
      kind == pending["producer"]
  IN
  /\ kind # "none"
  /\ (kind = "read" =>
        consumerFence.waitValue > 0 /\ timeline["consumer"].completed >= consumerFence.waitValue)
  /\ timeline' = [timeline EXCEPT !["producer"] = [
       submitted |-> current.submitted,
       completed |-> current.submitted,
       exported |-> current.exported
     ]]
  /\ pending' = [pending EXCEPT !["producer"] = "none"]
  /\ IF kind = "write"
     THEN /\ content' = "producer"
          /\ UNCHANGED observed
     ELSE /\ observed' = [observed EXCEPT !["producer"] = content]
          /\ UNCHANGED content
  /\ IF producerFence.waitValue > 0 /\ current.submitted >= producerFence.waitValue
     THEN handoffInFlight' = FALSE
     ELSE UNCHANGED handoffInFlight
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       lastExportLayout,
       producerFence,
       consumerFence
     >>

completeConsumer ==
  LET current == timeline["consumer"]
      kind == pending["consumer"]
  IN
  /\ kind # "none"
  /\ producerFence.waitValue > 0
  /\ timeline["producer"].completed >= producerFence.waitValue
  /\ timeline' = [timeline EXCEPT !["consumer"] = [
       submitted |-> current.submitted,
       completed |-> current.submitted,
       exported |-> current.exported
     ]]
  /\ pending' = [pending EXCEPT !["consumer"] = "none"]
  /\ IF kind = "write"
     THEN /\ content' = "consumer"
          /\ UNCHANGED observed
     ELSE /\ observed' = [observed EXCEPT !["consumer"] = content]
          /\ UNCHANGED content
  /\ IF consumerFence.waitValue > 0 /\ current.submitted >= consumerFence.waitValue
     THEN handoffInFlight' = FALSE
     ELSE UNCHANGED handoffInFlight
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       lastExportLayout,
       producerFence,
       consumerFence
     >>

destroyLogical(party) ==
  /\ logicalAlive[party]
  /\ logicalAlive' = [logicalAlive EXCEPT ![party] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       handoffInFlight,
       lastExportLayout,
       timeline,
       producerFence,
       consumerFence,
       pending,
       content,
       observed
     >>

collectPhysical ==
  /\ physicalAlive
  /\ ~textureTransportOpen
  /\ \A party \in Parties : ~logicalAlive[party]
  /\ \A party \in Parties : pending[party] = "none"
  /\ physicalAlive' = FALSE
  /\ allocationKind' = "none"
  /\ owner' = "none"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       textureTransportOpen,
       layout,
       handoffInFlight,
       lastExportLayout,
       timeline,
       producerFence,
       consumerFence,
       pending,
       content,
       observed
     >>

collectProducerFence ==
  /\ producerFence.retained
  /\ timeline["consumer"].completed > 0
  /\ pending["consumer"] = "none"
  /\ producerFence' = [
       open |-> FALSE,
       retained |-> FALSE,
       waitValue |-> producerFence.waitValue
     ]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       handoffInFlight,
       lastExportLayout,
       timeline,
       consumerFence,
       pending,
       content,
       observed
     >>

collectConsumerFence ==
  /\ consumerFence.retained
  /\ timeline["producer"].completed = 2
  /\ pending["producer"] = "none"
  /\ consumerFence' = [
       open |-> FALSE,
       retained |-> FALSE,
       waitValue |-> consumerFence.waitValue
     ]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       allocationKind,
       textureTransportOpen,
       layout,
       owner,
       handoffInFlight,
       lastExportLayout,
       timeline,
       producerFence,
       pending,
       content,
       observed
     >>

Next ==
  \/ \E kind \in TextureCandidates : createTexture(kind)
  \/ \E party \in Parties, nextLayout \in ActiveLayouts : transitionLayout(party, nextLayout)
  \/ \E party \in Parties : attemptReleaseOwnership(party)
  \/ \E party \in Parties : attemptAcquireOwnership(party)
  \/ attemptExportTexture
  \/ \E ok \in BOOLEAN : attemptImportTexture(ok)
  \/ closeTextureTransport
  \/ submitProducerWrite
  \/ attemptExportProducerPoint
  \/ \E ok \in BOOLEAN : attemptEnqueueConsumerWait(ok)
  \/ closeProducerFenceHandle
  \/ submitConsumerRead
  \/ submitConsumerWrite
  \/ attemptExportConsumerPoint
  \/ \E ok \in BOOLEAN : attemptEnqueueProducerWait(ok)
  \/ closeConsumerFenceHandle
  \/ submitProducerRead
  \/ completeProducer
  \/ completeConsumer
  \/ \E party \in Parties : destroyLogical(party)
  \/ collectPhysical
  \/ collectProducerFence
  \/ collectConsumerFence

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ logicalAlive \in [Parties -> BOOLEAN]
  /\ physicalAlive \in BOOLEAN
  /\ allocationKind \in AllocationKinds
  /\ textureTransportOpen \in BOOLEAN
  /\ layout \in Layouts
  /\ owner \in Owners
  /\ handoffInFlight \in BOOLEAN
  /\ lastExportLayout \in {"none", "common"}
  /\ timeline \in [Parties -> TimelineRecords]
  /\ producerFence \in FenceRecords
  /\ consumerFence \in FenceRecords
  /\ pending \in [Parties -> PendingKinds]
  /\ content \in Contents
  /\ observed \in [Parties -> Contents]
  /\ lastOutcome \in Outcomes
  /\ \A party \in Parties :
       /\ timeline[party].completed <= timeline[party].submitted
       /\ timeline[party].exported <= timeline[party].submitted

OnlySharedCommittedTextureIsShareable ==
  /\ (physicalAlive <=> allocationKind # "none")
  /\ ((textureTransportOpen \/ logicalAlive["consumer"])
      => allocationKind = "sharedCommitted")

EveryTextureReferenceRetainsPhysicalStorage ==
  /\ \A party \in Parties : (logicalAlive[party] \/ pending[party] # "none") => physicalAlive
  /\ (textureTransportOpen => physicalAlive)

ExportsOccurAtCommonHandoffLayout ==
  /\ (handoffInFlight => layout = "common")
  /\ (lastExportLayout # "none" => lastExportLayout = "common")

OwnershipTransfersRequireSharedTexture ==
  /\ (physicalAlive <=> owner # "none")
  /\ (owner \in {"consumer", "external"} => allocationKind = "sharedCommitted")

HandoffsReleaseOwnership ==
  handoffInFlight /\ physicalAlive => owner = "external"

OnlyTheOwnerAccessesTheTexture ==
  \A party \in Parties : pending[party] # "none" => owner = party

ClosedFenceHandlesDoNotRevokeEnqueuedWaits ==
  /\ (producerFence.waitValue > 0 /\ ~producerFence.open /\ timeline["consumer"].completed = 0
      => producerFence.retained)
  /\ (consumerFence.waitValue > 0 /\ ~consumerFence.open /\ timeline["producer"].completed < 2
      => consumerFence.retained)

ExternalGpuAccessIsOrdered ==
  /\ (timeline["consumer"].completed > 0 =>
        producerFence.waitValue > 0
        /\ timeline["producer"].completed >= producerFence.waitValue)
  /\ (timeline["producer"].completed = 2 =>
        consumerFence.waitValue > 0
        /\ timeline["consumer"].completed >= consumerFence.waitValue)

WritesNeverExecuteConcurrently ==
  /\ ~(pending["producer"] = "write" /\ pending["consumer"] # "none")
  /\ ~(pending["consumer"] = "write" /\ pending["producer"] # "none")

BidirectionalContentsAreObserved ==
  /\ (observed["consumer"] # "empty" => observed["consumer"] = "producer")
  /\ (observed["producer"] # "empty" => observed["producer"] = "consumer")

Safety ==
  /\ OnlySharedCommittedTextureIsShareable
  /\ EveryTextureReferenceRetainsPhysicalStorage
  /\ ExportsOccurAtCommonHandoffLayout
  /\ OwnershipTransfersRequireSharedTexture
  /\ HandoffsReleaseOwnership
  /\ OnlyTheOwnerAccessesTheTexture
  /\ ClosedFenceHandlesDoNotRevokeEnqueuedWaits
  /\ ExternalGpuAccessIsOrdered
  /\ WritesNeverExecuteConcurrently
  /\ BidirectionalContentsAreObserved

\* A change of queue owner other than creation or collection.
OwnerChanges ==
  owner' # owner /\ owner # "none" /\ owner' # "none"

StepSafety ==
  [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
     /\ ((~textureTransportOpen /\ textureTransportOpen')
         \/ timeline'["producer"].exported > timeline["producer"].exported
         \/ timeline'["consumer"].exported > timeline["consumer"].exported
         => layout = "common" /\ layout' = "common")
     /\ (OwnerChanges =>
           /\ (owner = "external" \/ owner' = "external")
           /\ layout = "common"
           /\ layout' = "common")]_vars

=============================================================================
