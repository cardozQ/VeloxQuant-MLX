# VeloxQuant-MLX Performance TODO

## Priority
- **P0** — Blocks other work
- **P1** — High impact, clear path
- **P2** — Nice to have

---

## Phase 1: Profile the Bottleneck

- [ ] **P1** Run `mx.metal.start_capture()` during generation on target model(s)
- [ ] **P1** Identify: is it dispatch-bound or memory-bandwidth-bound?
- [ ] **P1** Run `veloxquant_mlx recommend` for your Mac config
- [ ] **P1** Profile with `py-spy` to measure Python overhead vs GPU time
- [ ] **P2** DTrace: measure kernel dispatch cost per layer

## Phase 2: Enable Fused Metal Kernels (Highest Leverage)

- [ ] **P1** Enable `fused_sdpa.py` as default attention path
- [ ] **P1** Benchmark: fused attend vs dequantize+attend round-trip
- [ ] **P1** Test `_rvq_attend.py` (fused RVQ decode + attention)
- [ ] **P1** Test `_scalar_attend.py` (fused KIVI-style decode + attention)
- [ ] **P2** Write fused Metal kernel for rotation + quantize + dequantize (single dispatch)

## Phase 3: Bit-Packed Direct Storage

- [ ] **P1** Store quantized indices directly — skip round-trip through fp16
- [ ] **P1** Integrate with `mlx_lm.generate()` to read packed format in attention
- [ ] **P2** Benchmark memory savings + throughput impact

## Phase 4: Weight Quantization Tuning

- [ ] **P1** Test 4-bit vs 8-bit weight quantization for your models
- [ ] **P1** Measure the weight bandwidth vs KV cache bandwidth trade-off
- [ ] **P2** Test per-layer mixed-precision (more bits for critical layers)

## Phase 5: Prefill Optimization

- [ ] **P2** Profile prefill phase (compute-bound — different strategy than decode)
- [ ] **P2** Evaluate Flash Attention-style kernels for prefill
- [ ] **P2** SnapKV integration for observation-window token eviction at prefill

## Phase 6: Validation & Production

- [ ] **P1** Test quality (cosine sim, perplexity) at each optimization step
- [ ] **P1** Benchmark throughput (tok/s) at each step
- [ ] **P1** Run full model sweep (Mistral 7B, Qwen3 4B/8B, Llama 3.1 8B)
- [ ] **P2** Add CI benchmark regression tests
