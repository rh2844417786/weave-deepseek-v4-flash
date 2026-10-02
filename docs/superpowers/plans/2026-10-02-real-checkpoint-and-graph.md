# DeepSeek-V4-Flash next acceptance gates

Authorized by the user's request to implement the annotated next steps.

1. Strengthen the expert identity oracle; add device counters for completed stolen tasks.
2. Add an explicit BF16 WMMA implementation alongside the compiled scalar oracle.
3. Check H4096/I2048/E256/top6 with rank-distinct synthetic weights and all 43 original checkpoint layers. Compare against an independent dequantize + FP32 PyTorch reference, including zero-token ranks, remote routes, stealing and graph replay.
4. Add an opt-in SGLang decode graph path restricted to batch one. Record device execution counters per layer, separate capture-time Python calls, and expose idle snapshots on localhost.
5. Add a server acceptance client that requires four ranks, all 43 layers, real prefill/decode execution and graph replay after readiness. Save complete request/response and provenance for sequential native comparison.
6. Calibrate and run GPQA only after the new kernel and full-model gates pass on H100. No local Mac result can establish those gates; new CUDA source must be compiled on the server.

WMMA initially computes a padded single-token tile. It establishes actual Tensor Core execution but does not establish efficient token batching or the paper's tile-level stealing implementation. Measured cost profiles remain specific to their kernel and source identity.
