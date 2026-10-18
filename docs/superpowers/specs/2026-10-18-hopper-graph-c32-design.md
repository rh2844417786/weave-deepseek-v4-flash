# Hopper C32 CUDA Graph integration

User clarified CUDA Graph capture/replay for current Weave up to batch32. Scope is optimized Hopper decode graph execution with existing optimized eager prefill; no persistent-scheduler rewrite. Preserve existing native CUDA sources and eager runtime helpers/hashes. Actual GPU checks and serving measurements remain server work.

## Graph contract

Explicit graph-C32 mode, TP4/EP4/attentionDP1/expertTP1, existing SGLang0.5.17/Torch2.11/CUDA13. Capture every integer decode batch1..32, disable padding, maxrunning32, prefill graph disabled. BothNative andWeave use same config. No speculative/MTP/PP/DPattention/LoRA/TBO/SBO/PDMUX/torchcompile/diagnosticprofile/tracecapture. Reject filtered or different captured bucket sets, unsupportedmode, changedsources and eagerfallback in measured decode. Existing eager andlegacyB1persistent modes remain unchanged.

## Runtime

New separate HopperGraphDecodeRuntime derived from/reusing eagerworkspace construction; keep hopper_decode.py/hopper_finalize.py andbothCUDAextensions unchanged. Allow capture only after an eager successful warmup for the batch. Body is existing FP8Workspace.run → active-slot NCCL SUM onTPdevicegroup → in-place devicecounter increment → orderedTritonfinalizer → cachedBF16originmaskview. Allocate devicecounter andallviews beforecapture. No CUDA sync/D2H/hostdependent routing inforward. Cachedboundweights/TMAmaps survive allcapturedgraphs andlayerreuse. Eagerprefill unchanged, separateworkspace.

Devicecounter records actualdecode slot collectives during replay. Existing per-layer device_calls.add_ is captured and reflects realgraph executions; Pythoncapture/implementation counts mustnever masquerade asreplays. Coverage exposes graph runtime/sourceidentity and devicecounter atidle, notforward.

## Source-pinned native observer

Lazy common hook forbothgroups wraps pinnedFullCudaGraphBackend constructor/capture_one/replay andmodel.forward. Capturesmodelowner, exact32buckets, backendclass, sourceidentities; forwardswarmup/capture/eager counts separately. Replay wrapper recordsactual successfulgraph.replay calls andraw/captured batchhistograms, failcounts; noGPUreads. ForWeave replay, update model_batch_counters aroundsuccessfuldispatch sincePythonmodel.forwardisnotcalled; synchronizeonlyidlecoverage existingendpoint. Native observer also records anyeagerdecode model.forward calls so fallback cannotpass unnoticed.

Dedicated localhost /graph-coverage on30310+rank, CPU-onlysnapshot refusingactiveoperations. Observerupdatessequence andactiveatomically; coveragebaseline afterclientwarmup anddrain. Sourcepins are existingmodel/graphbackend/decoderunner/base pluscanonicalgraphconfig andCLIalignment audit, verifiedagainstinstalledfiles.

## Verification

New four-rank GPU smoke usesactual newruntime andNCCL: all32buckets warmed/captured, repeated changed x/ids/routing, crossbucket order32→1→17→7→32, multipleboundweights sharedworkspace, emptyrankroutes, prefill-interleave thenreplay. Samebuffersstable; counterdelta actualreplays; finite outputs; compareeager vsgraph finite errors record-only, exactmetadata/ownership/knownassemblyprimitive checks remainallowed. No numericalerror cutoff. Preserve old completeprefill GPUproof withunchangedkernel/runtimehashes andrun newgraphsmoke separately.

During service require capture32buckets andactualreplayB32; everyWeave layer device_decode_delta equals graph replaydelta, no capture/recapture/eagerdecode duringwindow, actualdecode slot collective delta equals43*replays. Prefill remains all43newHopperwithseparatecounter. Crossrankreplayhistmatches; rawbatch=capturedbatch(no padding). Native hasactualreplays andzeroeagerdecode inwindow.

## GPQA and delivery

Add --decode-graph disabled|full to run_gpqa_c32.py (disableddefault unchanged). Full selects commongraph-C32 launcherflags/ports+newGPUsmoke beforetwo groups. BothGPQA groups32clients/1200formalwindow/ThinkHigh32768EOS unchanged. Newgraph proof/observer snapshots/coveragechecks publicallowlisted. Parent ownschildren/timeouts/cleanup; graphbaseline HTTP+Weavecoverage share clienthandshake budget. No rawinputs/outputIDs public, no newaccuracygates. Source/data recheckedafter, allreportedmodesexplicit. Existinglocalmodel/image/GitHub-only dependencies. Pushmain andserverprompt, reportCPU vsGPU boundaries.
