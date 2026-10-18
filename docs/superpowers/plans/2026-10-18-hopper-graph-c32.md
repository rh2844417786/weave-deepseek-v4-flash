# Hopper C32 Graph implementation plan

Use superpowers:subagent-driven-development inprimarycodex/hopper-graph-c32. Userexplicitly selectedCUDA Graph, establishedGitHubhandoff authorized.
**Spec:** docs/superpowers/specs/2026-10-18-hopper-graph-c32-design.md
**Goal:** actual full decode graph forB1..32 withoptimizedeagerprefill andmatchedGPQA graph comparison.
**Constraints:** existingofflineimage/model/GPUQ4H100; oldCUDA/eagerhelpersunchanged; record-onlynumerics; no fallback; fixedGPQA32/1200.

## Task1 graph runtime and GPU smoke

Own src/weave/hopper_graph.py, tests/gpu/check_hopper_graph.py, CPUtests. HopperGraphDecodeRuntime(peer,devicegroup,experts256) constructsold workspace andbuffers, warmsbatch beforecapture, run reusesbody+devicecollectivecounter. Runtimehashhelper graph_runtime_source_sha256(), snapshotcollectivesmethod atidle; agree exactfields Task2/3. Nochanges hopper_decode/finalize/prefill/csrc/experimental. SyntheticEP4 H128I128E32 actual32buckets, mutableinputs/routes,2distinctboundhandles,changedbuckets/repeats/prefill interleave; finite/metadata/addresses/counter andrecord-onlyeagergrapherrors, atomic0664 reportsprivateerrors. Crossrankfailureagreement andcoordinatedgraphcollectives. CPUfakecapture/order/counters constraints andliteral noGPUtestclaim.

## Task2 pinned observer + adapter + launcher

Own src/weave/graph_observer.py, graph_hooks.py, configs/sglang_graph_sources.json, timing_hooks/sitecustomize.py, staged_decode config, sglang_backend.py, sglang_ext model ifneeded, run_native_ep.py. Sharedhook bothnative/customFullBackend.__init__/capture_one/replay+model.forwardcounts. Nativeexistingpinnedsourcecopies under/tmp/weave-graph-2948168. Verifyloadedsources, exactbucket/config/no unsupportedparallelmodes, phasevalid. New --graph-c32 +--graph-evidence-port-base30310; requiresfull/32/requesttiming, disjointports, appendcanonical --cuda-graph-config JSON decodefull bs1..32 max32 prefilldisabled, disablepadding. Nativeobservernoeagerfallback, Weavereplay hooksModelBatchCounters. AdapternewruntimewhenexplicitgraphC32, oldB1persistentguard retained. GPUactualcounts onlyatsnapshot. Unitnativehook capture/eager/replay separation, capturefailure/activesnapshot/source/config mismatch, legacyunchanged. CoordinateTask1runtimeAPI/Task3schema.

## Task3 graph evidence and GPQA controller

Own src/weave/graph_evidence.py, scripts/run_gpqa_c32.py, run_c32_comparison.py servicehelper optionalkwargs only, graphgate command/helper asneeded, focusedtests. GraphGPQAflagfull defaultdisabled, identicalbothgroupsconfig, baseline/snapshotbothNative+Weave observers andidlecoverage. NewgraphGPUsmoke ownexistingimagecontainer afteroldproofvalidation beforemodels. Validator4rankexactmatrix/hashes, sourceboundobserveractualcaptures1..32/replays/rawhist/noeagerfallback, Weaveall43devicecounts/countersprefill separate; graph_mode inidentity. Exportnewproof/manifests/graph snapshots/validators; no rawdata. CPUtestsrejectflag-onlyphony/one-rankwrongbatch/padded/eager fallback/stalemissingproof/countermismatch. ExistingGPQAstrictdata/EOS/budget/depletion policies unchanged. No unrelatedC32suite expansion.

## Task4 root integration

AuditpinnedSGLang alignment/graphprep/replay/currentstream; docs/serverprompt, preflightsourceclosure (current graph filesindependentnotoldhash). Runfocused/fullbaseline knownGPQAissues, independenttask/finalreview. Commit/push withpreviousuntrackedhandoffpreserved. NewGPUcompilation/capture/replay/actualserviceevidence pendingserver.
