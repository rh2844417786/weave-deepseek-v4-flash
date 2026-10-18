# GPQA Main C32 20-minute comparison

User explicitly requested GPQA comparison scripts with batch32 and20minutes per group. Compare native SGLang vs current Hopper FP8 prefill+decode, TP4/EP4/eager, same4allocatedH100. Each formal measurement window is1200seconds; loading, proof verification, warmup and cleanup are recorded outside it. This follows the current C32 measurement convention, distinct from the older batch1 ready-time budget script.

## Dataset and generation

Use original GPQA Main448 offline prepared token IDs, deterministic ordered_indices(448) seed42, natural individual inputs without padding/concatenation/repetition. Preserve Think High, thinking_mode=thinking, reasoning_effort=high, temperature0, max_new_tokens32768, ignore_eos=False. No logprob collection in measured requests; this is the same token-count streaming measurement contract as current C32, without per-token logprob diagnostics. Accuracy is not evaluated.

A dedicated strict loader checks saved record SHA against both manifest and pinned deepseek_length_audit, source gpqa_main SHA, model/revision/pinned tokenizer+encoder identities, observed config policy, exact448 roworder andtoken lengths, prompt commitments, seed/thinking/output/EOS settings. Parent checks observed model metadata against actual readonly checkpoint. Do not weaken generic gpqa loader or silently use custom data. Existing prepared file already had recordsha cdcba49397971a26d71b444c7a00785cff3f4cb305456b2acf70e2c0dc30f888.

## Concurrent client

32workers, servermaxrunning32, record actualbatchhistogram; continuous refill as requests finish naturally. Do not promise everyforward B32. Requests useinput_ids andUUIDwireRID, validatedmeta_info.id. Reuse existingSSEparser/control and WindowMetrics with optional normalEOS policy defaultoff forlegacyclient. GPQA accepts stop/eos andlength32768, rejectsaborts/errors/nonmonotoniccounts/foreignIDs. Three bounds fixed32/1200/32768; noCLIoverrides silentlyalter experiment.

Warmup32requests×up-to8tokens outsidewindow, then positive idleflush beforeformalwindow toremove warmupcache. No per-requestflush whileloadactive. At1200s stopdispatch, cancel/abortonlyownedunfinishedrequests, no token beyonddeadline counted. Persist safe rows byoriginalrow_index withtimestamps/counts/finishreason; no prompts/generatedtexts/tokenIDs/rawRID public. Include incomplete tokens in primary numerator and exclude each firsttoken. Output TTFT/validsingle-tokenITL, completed/unfinished/unattempted counts and commoncompletedrow IDs. Numerical/content agreement doesnotgateperformance.

Record first predeadline inability of a worker to refill separately from complete dataset exhaustion; suppress sustained ratio when either occurs. Merely dispatching finalrow is not depletion while32requests remain active. If448rows exhausted beforedeadline, do notduplicate: waituntilfixeddeadline, mark dataset_exhausted andidleinterval; retainmeasurements but suppress sustainedC32speedupratio. Validserverdata missing timestamps remain missing, notzero. Load failures retain diagnostics andno validratio.

## Controller and prior GPU proof

New separate run_gpqa_c32.py, reuse native service_command, OwnedChild, idleflush, sources/runtime/core validators. Do not rewrite existing customC32 suite. Input --records absolute and --prefill-evidence-dir complete95a27da publicbundle. Verifybundlemanifest exactinventory/no symlinks, statuses/core/oldreplay+assembly+52prefillcases, currentnativeCUDA/decode/prefillruntime/checkpoint/image identities. HistoricalservingPython hashes retained but not requiredcurrent because this changesonly workload/controller; validate all43 actualcurrentservicecoverage anew. Preserveoriginalproof bytes innewpublicproof folder. No rerun unchangedGPUmatrix bydefault; failure/missingproof cannot bypass.

Two sequential groups native/hopper_prefill, fixed1200s. Existingimageoffline client read-onlyprepared directory andreadonlyrepo; parentmodels readonly. Datafile/manifest/source rechecked after. Per-group actualB32 native scheduler/customallrankcoverage required. New prefillsource/runtime and independentprefill/decodecounters enforced. Parentdeadline derivedfromload-window marker, setup/flush/warmup bounded, ownedcleanup oncancel/failure. Publicexports staticallowlist+verifiedproof only; no rawdata. Freshjobdirs/0664 artifacts.

## Delivery

Existing SGLang0.5.17/Torch2.11/CUDA13/Python3.12 image, GitHubonly server, no downloads exceptpinnedauthorGPQA GitHub archive ifdata missing, SHA verified; CPUtokenization viaexistingimage networknone. PreserveGPUQallocation andcachepaths. CPU tests forstrictdata/EOS/window/cancellation/provenance/reuse and independentreview. Pushmain +serverprompt. NoGPQAmeasurements oraccuracyclaims beforeactualserverreturn.
