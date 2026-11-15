-------------------------- MODULE VulkanDeviceLoss --------------------------
(***************************************************************************)
(* Formal safety model for terminal device-loss publication, fault-report  *)
(* capture, and destructor queue draining in the Vulkan backend:           *)
(*   - src/Vulkan/VulkanDevice.cpp (~VulkanDevice, publishLoss,            *)
(*     requireActive, deviceLossError, captureDeviceFaultReport)           *)
(*   - src/Vulkan/VulkanQueues.cpp (waitIdle, collectGarbage, shutdown)    *)
(*                                                                         *)
(* Design-rule constants (TRUE in VulkanDeviceLoss.cfg, FALSE in           *)
(* counterexample configurations):                                         *)
(*   - DrainSyntheticLoss: synthetic/injected loss leaves a real healthy   *)
(*     GPU running in-flight work, so teardown must still drain pending    *)
(*     submissions unless the driver itself reported VK_ERROR_DEVICE_LOST. *)
(*   - GuardFaultQuery: captureDeviceFaultReport reads the cached report   *)
(*     or returns empty rather than calling vkGetDeviceFaultInfoEXT on a   *)
(*     healthy or synthetically lost device.                               *)
(*                                                                         *)
(* Trusted boundary: Vulkan ICD execution of vkQueueWaitIdle and native    *)
(* vkGetDeviceFaultInfoEXT payload contents once nativeLost holds.         *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANTS DrainSyntheticLoss, GuardFaultQuery

ASSUME {DrainSyntheticLoss, GuardFaultQuery} \subseteq BOOLEAN

NoReason == 0
SyntheticLoss == 1
DriverLoss == 2
LossReasons == {SyntheticLoss, DriverLoss}
MaxPending == 2
MaxLossEvents == 2

BoundedSeq(S, maxLen) == UNION {[1..n -> S] : n \in 0..maxLen}

VARIABLES
  nativeLost,           \* Whether the Vulkan ICD actually lost the device.
  reported,             \* Whether device loss has been published to callers.
  pending,              \* Count of in-flight GPU submissions not yet retired.
  alive,                \* Whether VulkanDevice has not yet been destroyed.
  lossHistory,          \* Ordered log of observed loss reasons (ground truth).
  storedReason,         \* Cached lossError_ published by publishLoss.
  returnedReason,       \* Reason last observed via deviceLossError/requireActive.
  queriedHealthyDevice, \* Whether vkGetDeviceFaultInfoEXT ran while ~nativeLost.
  faultCaptured         \* Whether captureDeviceFaultReport has been invoked.

vars == <<
  nativeLost,
  reported,
  pending,
  alive,
  lossHistory,
  storedReason,
  returnedReason,
  queriedHealthyDevice,
  faultCaptured
>>

Init ==
  /\ nativeLost = FALSE
  /\ reported = FALSE
  /\ pending = 0
  /\ alive = TRUE
  /\ lossHistory = <<>>
  /\ storedReason = NoReason
  /\ returnedReason = NoReason
  /\ queriedHealthyDevice = FALSE
  /\ faultCaptured = FALSE

submit ==
  /\ alive
  /\ ~reported
  /\ ~nativeLost
  /\ pending < MaxPending
  /\ pending' = pending + 1
  /\ UNCHANGED <<
       nativeLost,
       reported,
       alive,
       lossHistory,
       storedReason,
       returnedReason,
       queriedHealthyDevice,
       faultCaptured
     >>

rejectSubmitWhenLost ==
  /\ alive
  /\ reported
  /\ returnedReason' = storedReason
  /\ UNCHANGED <<
       nativeLost,
       reported,
       pending,
       alive,
       lossHistory,
       storedReason,
       queriedHealthyDevice,
       faultCaptured
     >>

gpuCompletes ==
  /\ alive
  /\ ~nativeLost
  /\ pending > 0
  /\ pending' = pending - 1
  /\ UNCHANGED <<
       nativeLost,
       reported,
       alive,
       lossHistory,
       storedReason,
       returnedReason,
       queriedHealthyDevice,
       faultCaptured
     >>

driverLosesDevice ==
  /\ alive
  /\ ~nativeLost
  /\ nativeLost' = TRUE
  /\ UNCHANGED <<
       reported,
       pending,
       alive,
       lossHistory,
       storedReason,
       returnedReason,
       queriedHealthyDevice,
       faultCaptured
     >>

observeLoss(reason) ==
  /\ alive
  /\ reason \in LossReasons
  /\ reason = SyntheticLoss \/ nativeLost
  /\ Len(lossHistory) < MaxLossEvents
  /\ reported' = TRUE
  /\ lossHistory' = Append(lossHistory, reason)
  /\ IF ~reported
     THEN storedReason' = reason
     ELSE UNCHANGED storedReason
  /\ queriedHealthyDevice' =
       (queriedHealthyDevice \/ (~reported /\ reason = DriverLoss /\ ~nativeLost))
  /\ UNCHANGED <<nativeLost, pending, alive, returnedReason, faultCaptured>>

queryLossError ==
  /\ alive
  /\ returnedReason' = storedReason
  /\ UNCHANGED <<
       nativeLost,
       reported,
       pending,
       alive,
       lossHistory,
       storedReason,
       queriedHealthyDevice,
       faultCaptured
     >>

captureFault ==
  /\ alive
  /\ faultCaptured' = TRUE
  /\ queriedHealthyDevice' =
       (queriedHealthyDevice \/ (~GuardFaultQuery /\ ~nativeLost))
  /\ UNCHANGED <<
       nativeLost,
       reported,
       pending,
       alive,
       lossHistory,
       storedReason,
       returnedReason
     >>

destroy ==
  /\ alive
  /\ (IF DrainSyntheticLoss THEN nativeLost ELSE reported) \/ pending = 0
  /\ alive' = FALSE
  /\ UNCHANGED <<
       nativeLost,
       reported,
       pending,
       lossHistory,
       storedReason,
       returnedReason,
       queriedHealthyDevice,
       faultCaptured
     >>

Next ==
  \/ submit
  \/ rejectSubmitWhenLost
  \/ gpuCompletes
  \/ driverLosesDevice
  \/ \E reason \in LossReasons : observeLoss(reason)
  \/ queryLossError
  \/ captureFault
  \/ destroy

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ nativeLost \in BOOLEAN
  /\ reported \in BOOLEAN
  /\ pending \in 0..MaxPending
  /\ alive \in BOOLEAN
  /\ lossHistory \in BoundedSeq(LossReasons, MaxLossEvents)
  /\ storedReason \in {NoReason} \cup LossReasons
  /\ returnedReason \in {NoReason} \cup LossReasons
  /\ queriedHealthyDevice \in BOOLEAN
  /\ faultCaptured \in BOOLEAN

HealthyDeviceNeverQueried ==
  ~queriedHealthyDevice

DrainedBeforeHealthyDestroy ==
  (~alive /\ ~nativeLost) => pending = 0

FirstReasonPreserved ==
  /\ (reported <=> Len(lossHistory) > 0)
  /\ (~reported => storedReason = NoReason /\ returnedReason = NoReason)
  /\ (reported => storedReason = Head(lossHistory))
  /\ (returnedReason # NoReason => reported /\ returnedReason = Head(lossHistory))

DriverLossRequiresNativeLoss ==
  /\ (storedReason = DriverLoss => nativeLost)
  /\ \A i \in 1..Len(lossHistory) : lossHistory[i] = DriverLoss => nativeLost

Safety ==
  /\ HealthyDeviceNeverQueried
  /\ DrainedBeforeHealthyDestroy
  /\ FirstReasonPreserved
  /\ DriverLossRequiresNativeLoss

StepSafety ==
  [][/\ (reported => reported' /\ storedReason' = storedReason)
     /\ (nativeLost => nativeLost')
     /\ (~alive => ~alive')
     /\ (reported /\ alive' => pending' <= pending)]_vars

=============================================================================
