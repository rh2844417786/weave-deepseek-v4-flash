# Hopper expert candidate implementation plan

> Use superpowers:subagent-driven-development with task and final reviews. User has approved implementation and existing GitHub handoff; work in primary project directory on codex/hopper-expert-candidate to avoid prior worktree permission popups.

**Goal:** A runnable, source-bound FP8 TMA/WGMMA expert candidate tested against BF16 on real B32 traces.
**Architecture:** Separate CUDA extension and pinned offline headers; original raw weights; explicit quantized oracle and BF16 drift; replay first, service promotion later.
**Tech Stack:** ExistingCUDA13/Torch2.11, SM90a, minimalCUTLASSf3fde58372d33e9a5650ba7b80fc48b3b49d40c8 headerclosure, Python3.12.
**Spec:** docs/superpowers/specs/2026-10-17-hopper-expert-candidate-design.md

## Global constraints

Original local weights, EP4, all43 layers, B32 real trace. No serverinternet/dependencyinstall. Productiondefaults/BF16csrcidentityunchanged. No rawdataexport, no modifiedgates, no GPUclaims locally. Preserve unrelateduntrackedhandofffile.

## Task1 native candidate

Own `experimental/hopper_fp8/kernel.cu`, `experimental/hopper_fp8/math_contract.h`, optionallocalheaders and `tests/cpp/test_hopper_contract.cpp`, `tests/cpu/test_hopper_contract.py`. Follow exactworkspaceAPI in spec. GenuineTMA+WGMMA usingpinnedCUTE/CUTLASS header primitives, compactexpertlist, boundedpersistentstorage, nofullweightcopies. Useattachmentreferenceattribution and originalBF16numericsheaderwhereappropriate.

- [x] Write failing CPU/C++ tests for exactnibble→FP8bytes, swizzle/permutation, activeexpert/slotmapping, capacity, scalezero policy.
- [x] ImplementCUDA workspace, immutableboundweights/maps, grouping/FP8packing/3GEMMs/SwiGLU/postroute, resources andstableviews.
- [x] Keep only128threads participating in warpgroup operations; source-audit fullwarpgroupfences/barriers/TMAtransactionbytes/descriptors/OOBmask.
- [x] Run availableCPUtests, verify syntaxcontracts; independentCUDAreview. ActualNVCC/GPUpendingserver.

## Task2 loader and oracle

Own `src/weave/hopper_candidate.py`, `src/weave/hopper_reference.py`, `tests/cpu/test_hopper_candidate.py`.
`source_manifest()`, `source_sha256()`, `extension()` exposeindependentcompileidentity. `extension().FP8Workspace` isTask1API. Candidateonlybuildlist,flags/arch/vendoredmanifest/headerclosure hashedoncebeforeloadandrecheckedafter. Rootvendorsheaders.
Oracle `quantize_fp8_reference(tensor)` returns(dequantized_FP32,fp8codes,scales) per128; `expert_reference_fp8(x,raw,expert,clamp)` usesexplicitK32scaleorderandBF16SwiGLUboundaries. NoimportTorchatmoduleimport forCPUtests. PurePythonFP8encoding/rounding helpers testedagainstknownboundaries/ties/saturation/zero;GPUreferenceusesTorch independentlyofCUDA.

- [x] Testmissing/corruptvendorreject andsourceclosureidentity plusCPUquant edgecases RED→GREEN.
- [x] Implementloader/oracle, preserve oldextensionbehavior; independentreview.

## Task3 GPU replay

Own `tests/gpu/check_hopper_candidate.py`, `src/weave/hopper_evidence.py`, `tests/cpu/test_hopper_evidence.py`.
CLI torchrun4 `--model-path --capture-dir --output-dir --identity-file --warmups3 --repetitions20 --budget-seconds7200`.
Load/verifyprivatecapture usingexistingreplay_trace; perrankCPUreplicas mustmatch. MandatorysyntheticB1/7/17/32 andemptyrank,graphstablebuffer tests; actualall43×2samples. Benchmark`bf16_eager`,`fp8_eager`,`bf16_graph`,`fp8_graph`, same scopeonlinegroup→slotoutputs, noIPC/allreduce. Numericagainstquantizedoracle andBF16outside timing. LogFP8precisionfailuresseparately; don'trelaxgates. Publicperrankreport includes2×43×4points, timingarrays, workload/error/resources, capturesource/parentsource/candidatesourceidentity, statuscomplete/partial/failed. Noresumetiming reuse thisfirstdelivery; freshdirs/partialpointcheckpointforsalvage. Requireallrankcompleteandsamecasesforpromotion.

- [x] WriteCPUevidencevalidator testsforcasecoverage,numericstatus,metricsscope,privacy,graph/eagerpairing.
- [x] ImplementGPUgate/replayusingTask1/Task2API; separateFP8driftfromimplementationerror. Reviewcollectivesandbudgetcancellation.

## Task4 root integration and delivery

Ownvendorheaderdelivery/provenance/licenses, `scripts/run_hopper_candidate.py`, `tests/cpu/test_hopper_launch.py`,docs/serverprompt/validationreport.

- [x] FetchpinnedCUTLASSarchiveonconnecteddev, vendoronlyrecursivequotedincludeclosureofneededSM90headers, hashallandrecordupstreamgitlink/archiveSHA; noinstallationserver.
- [x] Testdriverpreview/readonlymounts/dynamicGPUs/offline/publicexportfailclosed.
- [x] Controller validatesexistingBF16core/currentmodel, existingprivatecapture,image/tokenizer, separatecandidatesource; launchesexistingimage andfreshprivateworkdir, boundedOwnedChildcleanup. Aggregatesonlyfullyvalidatedsameidentityreports; noSGlangservicepromotion.
- [x] Readreturnedservicefailuresintosummaryrecord; preservegenerationgate andreportstagedservingnotmeasured.
- [x] Runfocused/fullCPUbaselineaccounting, reviews,sourcechecks;commit/pushmain;serverpromptpointsatcomplete43-layerfrozenexperiment. Statependingperformance/precision/servergraph gates andnextconditionalserviceintegration.

## Local completion

Candidate, loader/oracle, GPU replay, offline vendor and controller code are implemented and independently reviewed. GPU gates and1376 real measurements are prepared, not executed here.36 focused tests:34 pass and2 server-Torch requirements skip; full303-test CPU run retains pre-existing7 failure/3 error subcases. See reports/hopper_expert_candidate_validation.json for exact source identities and limits.
