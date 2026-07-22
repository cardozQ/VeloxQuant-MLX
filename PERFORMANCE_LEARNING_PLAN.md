# Performance Engineering — VeloxQuant-MLX

**Your background: systems & hardware → KV-cache quantization perf**

This is an ideal background for this project. The bottlenecks here are
kernel dispatch overhead, memory bandwidth, GPU utilization, and
instruction-level throughput — all classic systems/hardware problems.

---

## 1. What You Need to Build (Mental Model)

You are compressing the **key-value cache** that LLMs store during text
generation. Every new token attends to all previous tokens' keys/values.
Without compression, the KV cache eats GBs of memory for long sequences.

The pipeline for each token:

```
key (fp16) → normalize → rotate → scalar-quantize → bit-pack → store
                                 ↓
query → [decode keys → attend → output]
```

Each step launches a GPU kernel. **Kernel launch overhead** (not math)
is often the dominant cost — exactly the kind of problem hardware folks
excel at.

---

## 2. Domain Concepts (Must Learn)

| Concept | Why It Matters |
|---|---|
| **KV cache** — what it stores, how it grows (2 × layers × heads × seq_len × head_dim × bytes) | Everything you optimize |
| **Scalar quantization** — mapping fp16 to b-bit centroids (Lloyd-Max, boundary-sum) | Core compression primitive |
| **RVQ (Residual Vector Quantization)** — quantize, subtract residual, quantize again | Higher quality at low bits |
| **Hadamard rotation** — orthogonal transform that spreads information evenly across dimensions | Makes quantization more efficient |
| **SDPA (Scaled Dot-Product Attention)** — `softmax(Q @ K^T / sqrt(d)) @ V` | The operation you accelerate |
| **Memory-bandwidth bound vs compute-bound** — Mistral 7B hits ~22 tok/s because of memory, not compute | Dictates which optimizations matter |
| **Unified memory** (Apple Silicon) — CPU + GPU share DRAM, no PCIe bottleneck | Changes optimization strategy vs CUDA |
| **flash-decoding** — split KV-axis across SIMD groups, merge partial results | Key pattern in fused Metal kernels |

---

## 3. System Architecture (Where to Optimize)

```mermaid
flowchart LR
    subgraph Host[Python Host]
        CFG[KVCacheConfig]
        KC[KVCache Wrapper]
        QZ[Quantizer: Python/MLX]
    end

    subgraph GPU[Apple GPU (Metal)]
        MK[Metal Kernel<br/>MSL source embedded in .py]
        UM[(Unified Memory)]
    end

    CFG --> KC
    KC --> QZ

    QZ -- "mx.fast.metal_kernel()" --> MK
    MK <--> UM
    QZ -- "mx.array ops" --> UM

    style MK fill:#f96
    style UM fill:#69f
```

**Key insight**: All Metal shader source code is embedded as Python string
literals in `veloxquant_mlx/metal/_*.py`. There are **no standalone .metal
files**. The kernels are JIT-compiled at runtime via
`mx.fast.metal_kernel(source, ...)` — MLX's bridge to Metal Performance
Shaders.

---

## 4. What's Already Been Optimized

From `OPTIMIZATION_FINDINGS.md` — four bottlenecks already fixed
(**1.2–2.6× speedup**):

| # | Bottleneck | Fix | Speedup |
|---|---|---|---|
| 1 | Per-head Python loop → 256 small kernel launches | Flatten `(B,H,S,D)` → `(B*H*S,D)`, single quantizer call | 1.2–1.85× |
| 2 | O(d²) QR rotation | `HadamardPreconditioner` — O(d log d), single Metal kernel | Large on CPU-headroom models |
| 3 | Broadcast-argmin materialized full `(batch,d,k)` tensor | Boundary-sum: one comparison + sum | Compounding |
| 4 | Redundant fp32↔fp16 casts | Keep fp16, promote only for `linalg.norm` | ~1.05× |

**What's still on the table** (from the doc's "what's next"):
- Fused Metal kernel for rotation + quantize + dequantize (single dispatch)
- Bit-packed direct storage (no fp16 round-trip)
- Per-token renormalization skip (small quality hit, deliberate exclusion)

---

## 5. Learning Roadmap (8 Weeks)

### Week 1 — Foundation (no code)

| Day | Task |
|---|---|
| 1 | Read `README.md`, `CITATIONS.md` (bibliography), `OPTIMIZATION_FINDINGS.md` |
| 2 | Read `veloxquant_mlx/core/abstractions.py` — the 8 ABCs |
| 3 | Read `veloxquant_mlx/core/context.py` — QuantizationContext, EncodedVector |
| 4 | Read `veloxquant_mlx/cache/base.py` — KVCacheConfig, KVCacheFactory, KVCacheBuilder |
| 5 | Read `veloxquant_mlx/integration/mlx_lm_patch.py` — how it wires into mlx_lm |

### Week 2 — One Full Method End-to-End

Focus on **TurboQuant RVQ** — the flagship method.

| Day | Task |
|---|---|
| 1 | `veloxquant_mlx/quantizers/turboquant_rvq.py` — encode, decode, estimate_inner_product |
| 2 | `veloxquant_mlx/cache/turboquant_rvq_cache.py` — append_key, attend |
| 3 | `veloxquant_mlx/handlers/` — trace the pipeline chain |
| 4 | `veloxquant_mlx/codebooks/scalar_codebook.py` — boundary-sum quantize |
| 5 | **Hands-on**: `python -m pytest veloxquant_mlx/tests/quantizers/test_turboquant_rvq.py -v` |

### Week 3 — Metal GPU Kernels

| Day | Task |
|---|---|
| 1 | Read `veloxquant_mlx/metal/kernels.py` — the re-export facade |
| 2 | Read `veloxquant_mlx/metal/_vecinfer.py` — embedded MSL for codebook quantize/dequantize |
| 3 | Read `veloxquant_mlx/metal/_bit_packing.py` + `_scalar_quant.py` |
| 4 | Read `veloxquant_mlx/metal/_rabitq_attend.py` — fused flash-decoding pattern |
| 5 | Run: `python -m veloxquant_mlx.benchmarks.metal_kernel_benchmark --n_iter 10` |

### Week 4 — Benchmarking & Profiling Infrastructure

| Day | Task |
|---|---|
| 1 | Read `veloxquant_mlx/benchmarks/metal_kernel_benchmark.py` — timing helpers, bench functions |
| 2 | Read `veloxquant_mlx/benchmarks/model_kv_benchmark.py` — end-to-end model benchmarks |
| 3 | Run a benchmark: `python -m veloxquant_mlx.benchmarks.metal_kernel_benchmark` |
| 4 | Study `scripts/metal_rabitq_attend_bench.py` — standalone Metal vs MLX comparison |
| 5 | Study `scripts/plot_optimization_journey.py` — optimization tracking |

### Week 5 — Deep Dive: Dispatch Overhead

| Day | Task |
|---|---|
| 1 | Profile the cache `attend()` path: count MLX kernel dispatches per call |
| 2 | Identify which dispatches could fuse into a single Metal kernel |
| 3 | Read `veloxquant_mlx/metal/fused_sdpa.py` — an example of fusing dequant + attention |
| 4 | Study the RVQ fused attend kernel (`_rvq_attend.py + metal._rvq_attend`) |
| 5 | Brainstorm: what's the next fusion candidate? |

### Week 6 — Deep Dive: Memory & Bandwidth

| Day | Task |
|---|---|
| 1 | Measure KV cache memory at different seq lens: `scripts/validate_kv_memory.py` |
| 2 | Understand the fp16 round-trip cost (encode → store fp16 → decode) |
| 3 | Study bit-packed storage: `veloxquant_mlx/dsa/bit_pack.py` |
| 4 | Read `docs/PACKED_STORAGE_ROADMAP.md` |
| 5 | Profile memory bandwidth utilization (use `mx.metal.device_info()` if available) |

### Week 7 — Other Methods & Their Optimization Opportunities

| Day | Task |
|---|---|
| 1 | VecInfer (`quantizers/vecinfer.py`, `cache/vecinfer_cache.py`) — product quantization |
| 2 | RaBitQ (`quantizers/rabitq.py`) — binary packing, Hamming distance scoring |
| 3 | QJL (`quantizers/qjl.py`) — JL sketch + sign quantization |
| 4 | Spectral (`quantizers/spectral.py`, `spectral/`) — spectral decomp + bit allocation |
| 5 | Token eviction methods: SnapKV, StreamingLLM, H2O, TOVA |

### Week 8 — Synthesis & Your First Optimization

| Day | Task |
|---|---|
| 1 | Re-read `OPTIMIZATION_FINDINGS.md` — now it should all make sense |
| 2 | Pick one of the "what's next" items or your own candidate |
| 3 | Prototype: benchmark before, implement, benchmark after |
| 4 | Write up findings (model output quality + throughput numbers) |
| 5 | Submit PR |

---

## 6. Optimization Opportunity Matrix

| Area | Opportunity | Effort | Impact | Notes |
|---|---|---|---|---|
| **Kernel fusion** | Combine rotation + quantize + dequantize into 1 Metal dispatch | High | High | Multi-day project; would push throughput past fp16 on all models |
| **Bit-packed storage** | Store encoded vectors as packed bits instead of fp16 | Medium | High | Requires deeper mlx_lm integration |
| **Per-token renormalization skip** | Skip norm recompute for tokens already normalized | Low | Medium | Small quality hit (~1–2% cosine) |
| **More fused attend kernels** | VecInfer attend, Spectral attend, Polar attend | Medium | Medium | Follow the RVQ attend pattern |
| **Async pipeline** | Overlap quantize with previous layer's attention | High | High | Complex, might need MLX graph capture |
| **Dynamic precision** | Adapt bit-width per layer/head based on attention sparsity | Medium | Medium | Follow RateQuant but at runtime |
| **Precomputed codebooks** | Load Lloyd-Max codebooks from artifact store instead of computing each time | Low | Low | Already partially supported |
| **Wider Metal test coverage** | Remaining 39 methods without Metal kernels | Very High | Very High | Long tail; focus on most-used methods first |
| **Reduce Python overhead** | Profile for remaining Python-level dispatch costs | Low | Medium | Use Python profiler: `python -m cProfile` |

---

## 7. Key Files for Performance Work

| File | Why |
|---|---|
| `veloxquant_mlx/metal/_rabitq_attend.py` | Best model for fused kernel design (flash-decoding pattern) |
| `veloxquant_mlx/metal/_vecinfer.py` | Embedded MSL source — study to write your own kernels |
| `veloxquant_mlx/metal/fused_sdpa.py` | Example of fusing dequant + attention |
| `veloxquant_mlx/cache/base.py` | The factory dispatch — windows to all 46 methods |
| `veloxquant_mlx/quantizers/turboquant_rvq.py` | Reference quantizer: encode → decode → estimate_inner_product |
| `veloxquant_mlx/codebooks/scalar_codebook.py` | The boundary-sum optimization (replaced broadcast-argmin) |
| `veloxquant_mlx/benchmarks/metal_kernel_benchmark.py` | Benchmark harness — model your own benchmarks here |
| `veloxquant_mlx/benchmarks/model_kv_benchmark.py` | End-to-end model benchmark (perplexity + throughput + memory) |
| `OPTIMIZATION_FINDINGS.md` | The optimization journey — understand the methodology |
| `scripts/plot_optimization_journey.py` | How to track progress across optimization iterations |

---

## 8. How to Verify an Optimization

Every optimization must pass **three gates**:

1. **Quality** — cosine similarity or perplexity must not regress
   ```
   python -m pytest veloxquant_mlx/tests/quantizers/ -v
   ```

2. **Speed** — measure before/after with same methodology
   ```
   python -m veloxquant_mlx.benchmarks.metal_kernel_benchmark --n_iter 50
   ```

3. **Correctness** — generated output must be coherent (full token sequence)
   ```
   python -m veloxquant_mlx benchmark --method turboquant_rvq --model mlx-community/Mistral-7B-Instruct-4bit
   ```

---

## 9. Quick Reference: MLX Performance Primitives

| MLX API | What It Does | Performance Note |
|---|---|---|
| `mx.fast.metal_kernel(src, ...)` | JIT-compile & dispatch MSL on GPU | Primary perf lever |
| `mx.hadamard_transform(x)` | Fast Walsh-Hadamard transform, O(d log d) | Use instead of QR matmul |
| `mx.eval(x)` | Force MLX graph evaluation | Needed for accurate timing |
| `mx.linalg.norm(x)` | L2 norm | Only promote to fp32 here |
| `mx.maximum(x, eps)` | Safe clamp (replaces `where + norm` combo) | Fewer kernels |
| `mx.array.astype(dtype)` | Cast | Minimize — each is a kernel launch |

---

## 10. Principles from OPTIMIZATION_FINDINGS.md

1. **Batch heads** — fold head dimension into batch, call quantizer once
2. **Prefer Hadamard** over QR rotation (O(d log d) vs O(d²))
3. **Boundary-sum** instead of broadcast-argmin (avoid materializing large diff tensors)
4. **Minimize dtype casts** — stay in fp16, promote only where precision is needed
5. **Single dispatch is faster than many dispatches** — even if each kernel does less math
6. **On M-series, memory bandwidth is the bottleneck for 7B+ models** — reducing bytes moved matters more than reducing FLOPs
