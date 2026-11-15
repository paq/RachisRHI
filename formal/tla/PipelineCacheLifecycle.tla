----------------------- MODULE PipelineCacheLifecycle -----------------------
(***************************************************************************)
(* Keyed pipeline-cache lookup, populate vs require-hit policy, stale      *)
(* handle and incompatible-key rejection, serialization, restoration, and  *)
(* decoupled pipeline ownership.                                           *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Pipeline.cpp (createPipelineCache restoring          *)
(*     PipelineCacheDesc::initialData through decodePipelineCache,         *)
(*     pipelineCacheData serialization, destroyPipelineCache, and          *)
(*     createComputePipeline and createGraphicsPipeline keyed              *)
(*     PipelineCacheRecord::entries lookup through                         *)
(*     createCachedComputePipeline and createCachedGraphicsPipeline)       *)
(*   - include/Rachis/Pipeline.h (PipelineCacheDesc,                       *)
(*     PipelineCachePolicy::populate, PipelineCachePolicy::requireHit),    *)
(*     include/Rachis/Error.h (ErrorCode::pipelineCacheMiss)               *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two cache keys, two incompatible pipeline descriptors, and two cache  *)
(*   generations model independent multi-entry population, same-key        *)
(*   descriptor collision rejection, distinct cache-miss reporting without *)
(*   mutation, corrupt-blob restore rejection, and pipeline survival past  *)
(*   cache destruction. Driver PSO binary encoding is abstracted.          *)
(***************************************************************************)

EXTENDS Naturals

NoKey == 0
Keys == {1, 2}
RequestKeys == {NoKey} \cup Keys
NoDescriptor == 0
Descriptors == {1, 2}
EntryValues == {NoDescriptor} \cup Descriptors
MaxGenerations == 2
Generations == 0..MaxGenerations
MaxPipelines == 2
Policies == {"populate", "requireHit"}
BlobKinds == {"valid", "corrupt"}
Operations == {"none", "createCache", "restore", "createPipeline", "serialize", "destroy"}
Results == {"none", "accepted", "rejected", "cacheMiss"}

VARIABLES
  cacheAlive,        \* Whether a public PipelineCacheHandle is currently live
  generation,        \* Monotonic cache handle generation counter
  liveGeneration,    \* Generation token of the active handle (0 before first create)
  cacheEntries,      \* Per-key cached descriptor in the live cache
  serializedValid,   \* Whether a valid serialized cache blob has been captured
  serializedEntries, \* Keyed descriptor map stored in the serialized blob
  pipelineCount,     \* Count of published pipelines (independent of cache lifetime)
  lastOp,            \* Most recent operation kind
  lastResult,        \* Result classification ("accepted", "rejected", "cacheMiss")
  lastUseCache,      \* Whether the most recent pipeline creation supplied a cache
  lastReqGen,        \* Handle generation supplied by the caller
  lastKey,           \* Cache key supplied to the most recent pipeline creation
  lastDescriptor,    \* Pipeline descriptor supplied to pipeline creation
  lastPolicy         \* Cache policy supplied to pipeline creation

coreVars == <<
  cacheAlive, generation, liveGeneration, cacheEntries,
  serializedValid, serializedEntries, pipelineCount
>>

vars == <<
  cacheAlive, generation, liveGeneration, cacheEntries,
  serializedValid, serializedEntries, pipelineCount,
  lastOp, lastResult, lastUseCache, lastReqGen,
  lastKey, lastDescriptor, lastPolicy
>>

EmptyEntries == [k \in Keys |-> NoDescriptor]

HandleValid(reqGen) ==
  /\ cacheAlive
  /\ reqGen = liveGeneration

PipelineRequestOutcome(useCache, reqGen, key, descriptor, policy) ==
  IF pipelineCount >= MaxPipelines
  THEN "rejected"
  ELSE IF useCache /\ ~HandleValid(reqGen)
  THEN "rejected"
  ELSE IF ~useCache \/ key = NoKey
  THEN IF policy = "requireHit" THEN "rejected" ELSE "accepted"
  ELSE LET existing == cacheEntries[key] IN
       IF existing = NoDescriptor
       THEN IF policy = "requireHit" THEN "cacheMiss" ELSE "accepted"
       ELSE IF existing = descriptor THEN "accepted" ELSE "rejected"

Init ==
  /\ cacheAlive = FALSE
  /\ generation = 0
  /\ liveGeneration = 0
  /\ cacheEntries = EmptyEntries
  /\ serializedValid = FALSE
  /\ serializedEntries = EmptyEntries
  /\ pipelineCount = 0
  /\ lastOp = "none"
  /\ lastResult = "none"
  /\ lastUseCache = FALSE
  /\ lastReqGen = 0
  /\ lastKey = NoKey
  /\ lastDescriptor = 1
  /\ lastPolicy = "populate"

CreateEmptyCache ==
  /\ ~cacheAlive
  /\ generation < MaxGenerations
  /\ cacheAlive' = TRUE
  /\ generation' = generation + 1
  /\ liveGeneration' = generation + 1
  /\ cacheEntries' = EmptyEntries
  /\ UNCHANGED <<serializedValid, serializedEntries, pipelineCount>>
  /\ lastOp' = "createCache"
  /\ lastResult' = "accepted"
  /\ lastUseCache' = FALSE
  /\ lastReqGen' = generation + 1
  /\ lastKey' = NoKey
  /\ lastDescriptor' = 1
  /\ lastPolicy' = "populate"

AttemptRestoreCache(blobKind) ==
  /\ ~cacheAlive
  /\ generation < MaxGenerations
  /\ IF blobKind = "valid" /\ serializedValid
     THEN /\ cacheAlive' = TRUE
          /\ generation' = generation + 1
          /\ liveGeneration' = generation + 1
          /\ cacheEntries' = serializedEntries
          /\ UNCHANGED <<serializedValid, serializedEntries, pipelineCount>>
          /\ lastResult' = "accepted"
          /\ lastReqGen' = generation + 1
     ELSE /\ UNCHANGED coreVars
          /\ lastResult' = "rejected"
          /\ lastReqGen' = generation
  /\ lastOp' = "restore"
  /\ lastUseCache' = FALSE
  /\ lastKey' = NoKey
  /\ lastDescriptor' = 1
  /\ lastPolicy' = "populate"

AttemptCreatePipeline(useCache, reqGen, key, descriptor, policy) ==
  LET outcome == PipelineRequestOutcome(useCache, reqGen, key, descriptor, policy) IN
  /\ IF outcome = "accepted"
     THEN /\ cacheEntries' =
               IF useCache /\ key \in Keys /\ cacheEntries[key] = NoDescriptor
               THEN [cacheEntries EXCEPT ![key] = descriptor]
               ELSE cacheEntries
          /\ pipelineCount' = pipelineCount + 1
          /\ UNCHANGED <<
               cacheAlive, generation, liveGeneration,
               serializedValid, serializedEntries
             >>
     ELSE UNCHANGED coreVars
  /\ lastOp' = "createPipeline"
  /\ lastResult' = outcome
  /\ lastUseCache' = useCache
  /\ lastReqGen' = reqGen
  /\ lastKey' = key
  /\ lastDescriptor' = descriptor
  /\ lastPolicy' = policy

AttemptSerializeCache(reqGen) ==
  /\ IF HandleValid(reqGen)
     THEN /\ serializedValid' = TRUE
          /\ serializedEntries' = cacheEntries
          /\ UNCHANGED <<cacheAlive, generation, liveGeneration, cacheEntries, pipelineCount>>
          /\ lastResult' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastResult' = "rejected"
  /\ lastOp' = "serialize"
  /\ lastUseCache' = TRUE
  /\ lastReqGen' = reqGen
  /\ lastKey' = NoKey
  /\ lastDescriptor' = 1
  /\ lastPolicy' = "populate"

AttemptDestroyCache(reqGen) ==
  /\ IF HandleValid(reqGen)
     THEN /\ cacheAlive' = FALSE
          /\ cacheEntries' = EmptyEntries
          /\ UNCHANGED <<
               generation, liveGeneration,
               serializedValid, serializedEntries, pipelineCount
             >>
          /\ lastResult' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastResult' = "rejected"
  /\ lastOp' = "destroy"
  /\ lastUseCache' = TRUE
  /\ lastReqGen' = reqGen
  /\ lastKey' = NoKey
  /\ lastDescriptor' = 1
  /\ lastPolicy' = "populate"

Next ==
  \/ CreateEmptyCache
  \/ \E blobKind \in BlobKinds : AttemptRestoreCache(blobKind)
  \/ \E useCache \in BOOLEAN,
        reqGen \in Generations,
        key \in RequestKeys,
        descriptor \in Descriptors,
        policy \in Policies :
       AttemptCreatePipeline(useCache, reqGen, key, descriptor, policy)
  \/ \E reqGen \in Generations :
       \/ AttemptSerializeCache(reqGen)
       \/ AttemptDestroyCache(reqGen)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ cacheAlive \in BOOLEAN
  /\ generation \in Generations
  /\ liveGeneration \in Generations
  /\ cacheEntries \in [Keys -> EntryValues]
  /\ serializedValid \in BOOLEAN
  /\ serializedEntries \in [Keys -> EntryValues]
  /\ pipelineCount \in 0..MaxPipelines
  /\ lastOp \in Operations
  /\ lastResult \in Results
  /\ lastUseCache \in BOOLEAN
  /\ lastReqGen \in Generations
  /\ lastKey \in RequestKeys
  /\ lastDescriptor \in Descriptors
  /\ lastPolicy \in Policies

DeadCachesHaveNoEntries ==
  ~cacheAlive => cacheEntries = EmptyEntries

LiveHandlesMatchGeneration ==
  cacheAlive =>
    /\ liveGeneration = generation
    /\ generation > 0

UnusedBlobIsEmpty ==
  ~serializedValid => serializedEntries = EmptyEntries

CacheMissRequiresLiveKeyedRequireHit ==
  lastResult = "cacheMiss" =>
    /\ lastOp = "createPipeline"
    /\ lastUseCache
    /\ lastKey \in Keys
    /\ lastPolicy = "requireHit"
    /\ cacheAlive
    /\ cacheEntries[lastKey] = NoDescriptor

Safety ==
  /\ DeadCachesHaveNoEntries
  /\ LiveHandlesMatchGeneration
  /\ UnusedBlobIsEmpty
  /\ CacheMissRequiresLiveKeyedRequireHit

StepSafety ==
  [][
    /\ (lastResult' \in {"rejected", "cacheMiss"} => coreVars' = coreVars)
    /\ (lastOp' = "createPipeline" =>
          lastResult' = PipelineRequestOutcome(
            lastUseCache', lastReqGen', lastKey', lastDescriptor', lastPolicy'))
    /\ (lastOp' = "createPipeline" /\ lastResult' = "accepted" =>
          /\ pipelineCount' = pipelineCount + 1
          /\ \A k \in Keys :
               IF lastUseCache' /\ k = lastKey' /\ cacheEntries[k] = NoDescriptor
               THEN cacheEntries'[k] = lastDescriptor'
               ELSE cacheEntries'[k] = cacheEntries[k])
    /\ (lastOp' = "destroy" /\ lastResult' = "accepted" =>
          /\ ~cacheAlive'
          /\ pipelineCount' = pipelineCount
          /\ serializedEntries' = serializedEntries)
    /\ (lastOp' = "restore" /\ lastResult' = "accepted" =>
          /\ serializedValid
          /\ cacheEntries' = serializedEntries)
  ]_vars

=============================================================================
