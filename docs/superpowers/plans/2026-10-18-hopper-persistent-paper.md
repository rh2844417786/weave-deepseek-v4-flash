# Hopper persistent paper implementation plan
Spec: docs/superpowers/specs/2026-10-18-hopper-persistent-paper-design.md
User authorized code/push/prompt, no repeatapproval. Primarybranchcodex/hopper-persistent-paper. Preserveuntrackedhandoff.

Task A device tile: new experimental/hopper_persistent/hopper_tile.cuh; port currentHopper arithmetic asdevicefunctions percontract, hosttensormap. CPUTests ofsource/knownnumericmapping ifmeaningful. NoC++workspace edits.
Task B persistent extension: new experimental/hopper_persistent/megakernel.cu, scheduler.h,tile_scheduler.h; derive existingIPCkernel butM64Hopperonly, oneCTAperSM, safeprioritizedchunkclaims, dynamiccK+steal, mechanismevidencecontract. NooriginalCUDAedits. CoordinateTaskAhostmapsanddevicefunctions. CPUC++testsofclaim/selector/layout wherepossible.
Task C runtime/service: new src/weave/hopper_persistent.py; ops.py optionalextensionfactory; staged_decode.py+sglang_backend.py+run_native_ep.py+sglang_extifneeded; candidateexplicitlabels andpaperentrydynamicprofile,43layercoverage. NoGPQAcontroller edits. Bindnewruntimeconstructor/signaturewithroot+TaskB. Ownfocusedtests.
Task D root: runtimecalibration/GPU mechanism verifier, ownedsafeofflinecontrollerandexport +sourceboundevidence CPUtests, auditpaperrequirements source, independentreview, docs/serverprompt, commit/pushmain. Existingcandidateformalresultsnotpaperresults. ReportGPUpendinghonestly.
