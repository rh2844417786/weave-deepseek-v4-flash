# Hopper prefill implementation plan

Use superpowers:subagent-driven-development in primary branch codex/hopper-prefill. User authorized prefill replacement after measured diagnosis.

**Spec:** docs/superpowers/specs/2026-10-18-hopper-prefill-design.md
**Goal:** Replace slow persistent prefill expert execution with compact multi-tile FP8/TMA/WGMMA.
**Constraints:** preserve original CUDA/decode hashes; no native fallback or model numerical gates; existing offline runtime; dynamic4GPU; no rawpublicdata; preserve unrelated handoff.

## Task1 independent CUDA + loader

Own experimental/hopper_prefill/*, src/weave/hopper_prefill_candidate.py, pureCPU contract tests. API PrefillWorkspace(rank,ep,max_batch,h,i,experts,topk), bind_weights, run, finalize, properties slot_outputs/output/group_counts/expert_slots/tile_experts/tile_row_bases/tile_count, resources. B1..8192,H/I multiples128,raw6contract. Deterministic GPU grouping and compact tile capacity ceil(Btopk/64)+localE. Dynamic actual tile guards, tile-major TMA activation maps; expert-major weightmaps. TensorCore arithmetic copied/adapted with attribution from unchanged decoder. Bind lifetime/currentstream/no host routing; preallocatedoutput exactordered origin assembly. Reuse verifiedvendor/headerclosure fromexistingloader but independent extensionname/buildmanifest. Test CPU layout/capacity/unsafe inputs/headerplanner/loader beforeimplementation; no localGPU claim.

## Task2 runtime/adapter/launcher

Own src/weave/hopper_prefill.py, sglang_backend.py, staged_decode.py configonly, run_native_ep.py, customext model configplumbing, focusedCPUtests. Runtime init peer/devicegroup; bind/run ->cachedprefix; exposes candidate_source_sha256/runtime_source_sha256/resources/additional_slot_collectives. Prefillmode persistentdefault orhopper_fp8 explicit. Newprefill requiresdecodehopper/eager/tiled/cache/fixednooldprofiles. Newcoverage prefill_implementation,prefill_source_sha256,prefill_runtime_source_sha256,prefill_resources,additional_prefill_slot_collectives; layercalls hopper_prefill. Existing additional_slot_collectives remains decode counter. Newstallcandidate label hopper_prefill forbothnewflags; oldgroupunchanged. All43bindingsfullrepinput/origin output; reject unsupportedshape withoutfallback. Testsphase dispatch/counters/source/noalloc/collective/errors+legacy.

## Task3 GPU validation + controller

Own tests/gpu/check_hopper_prefill.py, scripts/run_prefill_gate.py, scripts/run_prefill_comparison.py, run_c32_comparison.py, c32_comparison.py and newevidencehelper asneeded; focusedCPUtests. New comparison-suite prefill variants native/hopper_fp8/hopper_prefill. Layercheck fullprefillimplementation identity and all43counts, exactlyoneprefillcollective percall; same decoderidentity. Old1376decodeproof validatednativehashes unchanged. GateexistingHoppersmoke thennewprefillchecks, optionalstalls collector. GPU syntheticB1,33,65,129,8035,8192,shrink; empty/skewedroutes; exactmetadata, finite/numericrecordonly vs unchangeddecodecandidate chunks32 reference; all43realB129 andrealL0B8035/8192. CLI outputdir/modelpath/identity args agreedwithTask1/2. Publicreports all4ranks mode0664/source/rawpolicy/runtime. Validatedeclaredcasecoverageandhashes. Newruntime+kernelhashes sourcebound. Existingcontroller lifecycle/response policy/export reused; no rawpublic and no error cutoffs. Optionaldiagnostics use same3groups but noformalratios.

## Task4 integration/delivery

Root coordinates interfaces, docs/memoryconstraints, sourceaudit and semanticcheck. Writepreflight+comparisonserverprompt; no extra workload changes. Runfocused/fullCPU withknownbaselineGPQA issues reported. Independent task/finalreview, fixes, explicitfilecommit, ffmerge/pushmain. NewGPUresults pendingserver, noclaimprefillfixeduntilreturned.
