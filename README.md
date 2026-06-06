[![← dsv4-flash-bench hub](https://img.shields.io/badge/%E2%86%90-dsv4--flash--bench__hub-blueviolet?style=for-the-badge)](https://github.com/jcartu/dsv4-flash-bench)

> Part of the [`dsv4-flash-bench`](https://github.com/jcartu/dsv4-flash-bench) ongoing benchmark series.
> See the hub for the current SOTA leaderboard and a chronological index of all studies.

---

# DeepSeek-V4-Flash on RTX PRO 6000 Blackwell

## Quick Start

**618 tok/s** peak sustained decode, **214 tok/s** single user, **93.3%** correct on the GLM Estonia long-context profile. Runs on **4× RTX PRO 6000** (96 GiB each) using FP8 KV cache with MTP=2 speculative decoding on the `cstechdev/dsv4-flash` image.

```bash
docker pull cstechdev/dsv4-flash@sha256:27b80536a36212cef21664699aee35acbc14b37f147d41cd9b12361154f3c4db
```

```bash
docker run -d --name ds4_1m --gpus all --network host \
  cstechdev/dsv4-flash@sha256:27b80536a36212cef21664699aee35acbc14b37f147d41cd9b12361154f3c4db \
  serve deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name deepseek-v4-flash \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 1048576 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --disable-custom-all-reduce \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --reasoning-config.reasoning_start_str '  thinking' \
    --reasoning-config.reasoning_end_str '  response' \
    --default-chat-template-kwargs.thinking true \
    --default-chat-template-kwargs.reasoning_effort max \
    --enable-flashinfer-autotune \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 2
```

API endpoint: `http://localhost:8000/v1` (OpenAI-compatible). First run loads ~92 GiB across 4 GPUs + JIT-compiles kernels.

Without speculative decoding: remove the two `--speculative-config.*` flags. Expect ~15% lower single-stream throughput.

## Recommended Checkpoints

| Checkpoint | Quant | KV Cache | Best for |
|---|---|---|---|
| **`deepseek-ai/DeepSeek-V4-Flash`** | **FP8 W8A8 (native)** | **FP8 block-256** | **Production** — native FP8, 1M context, MTP-2 |
| `deepseek-ai/DeepSeek-V4-Flash` (BF16) | BF16 | BF16 | Lossless reference (requires ~92 GiB per GPU at 1M ctx) |

> **Note:** DeepSeek-V4-Flash is a MoE architecture (671B total, ~37B active). The FP8 native checkpoint is the only viable quantization for 4× RTX PRO 6000 at 1M context. BF16 requires >2× the VRAM.

## Benchmark Results

### Sustained Decode (`cstechdev/dsv4-flash`, DeepSeek-V4-Flash FP8, TP=4, MTP=2)

Aggregate throughput (tok/s), 30s per cell:

```
ctx\conc      1       2       4       8      16      32      64     128
   0       214.9   343.4   406.9   618.2   557.4   558.3   533.6   534.7
  16k      205.5   318.2   343.5   547.2   516.2   516.3   510.9   514.2
  32k      202.9   308.9   345.4   540.0   521.1   509.0   516.1   514.7
  64k      204.7   314.0   328.8   527.0   493.6   487.1   488.0   485.5
 128k      196.9   300.6   319.4   506.6   481.0   471.5   476.4      —
```

### Prefill Throughput

```
ctx      TTFT     tok/s     tokens
 8k     0.887s    9,323      8,270
16k     1.625s   10,048     16,333
32k     3.217s   10,080     32,427
```

### Estonia Long-Context Profile (GLM dense-MLA vs NSA diagnostic)

**93.3% correct** (28/30) at concurrency=30, max_tokens=40,000. The model averages **6,274 tokens** of reasoning to reach the correct answer "Estonia" from a ~707K-character (~113K token) long-context prompt.

```
Correct:    28 / 30  (93.3%)
Wrong:       2
Hit max:     0

Token count to answer:
  min:    2,055
  p50:    4,198
  p90:   12,020
  p99:   27,004
  max:   28,785
  mean:   6,274

TTFT:  avg=112.8ms  p90=226.1ms
Gen tok/s:  avg=69.9  agg=73.7
E2E tok/s:  agg=31.7
```

### Best Configuration by Scenario

| Scenario | tok/s | Notes |
|---|---|---|
| Single user, ctx=0 | **214.9** | MTP=2, TP=4 |
| Single user, deep context (128k) | **196.9** | Only 8.4% drop from ctx=0 |
| Multi-user c=8 | **618.2** ⭐ Peak throughput |
| Multi-user c=16 | **557.4** | Scheduler-limited at >8 concurrent |
| Multi-user c=64 | **533.6** | Stable plateau |
| Estonia profile (c=30) | **73.7** agg gen | 93.3% correctness |

---

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [NCCL & Environment Variables](#nccl--environment-variables)
- [Launch Commands](#launch-commands)
- [Docker Images](#docker-images)
- [MTP / Speculative Decoding](#mtp--speculative-decoding)
- [Benchmark Methodology](#benchmark-methodology)
- [Detailed Benchmark Tables](#detailed-benchmark-tables)
- [Estonia Profile Details](#estonia-profile-details)
- [Memory Usage (VRAM)](#memory-usage-vram)
- [Known Issues and Fixes](#known-issues-and-fixes)

---

## Overview

DeepSeek-V4-Flash is a 671B-parameter Mixture-of-Experts decoder-only model from DeepSeek, with ~37B active parameters per token.

| Parameter | Value |
|---|---|
| Total parameters | 671B (MoE) |
| Active parameters | ~37B |
| Architecture | Mixture-of-Experts (MoE) with Multi-head Latent Attention (MLA) |
| Max context (native) | 1,048,576 tokens |
| Tested context (this study) | up to 131,072 tokens |
| Quantization | Native FP8 W8A8 |
| Speculative decoding | MTP (Multi-Token Prediction) heads, N=2 |
| Tool-call format | DeepSeek V4 XML (`deepseek_v4` parser) |
| Reasoning format | ` thinking` / ` response` tags |

This study characterizes DeepSeek-V4-Flash inference on the **4× RTX PRO 6000 Blackwell** SM120 platform using the `cstechdev/dsv4-flash` Docker image based on vLLM v0.21.1rc1. It covers sustained decode throughput across concurrency 1–128 and context lengths 0–128K, plus the Estonia long-context completion profile.

### Key Findings

- **Peak throughput: 618.2 tok/s** at c=8, ctx=0 — the `max-num-seqs=8` ceiling is the bottleneck above c=8
- **Near-linear context scaling**: only 8.4% drop from ctx=0 to ctx=128k at c=1 (214.9 → 196.9 tok/s)
- **Estonia profile: 93.3% correct** — the model reliably navigates 113K-token contexts to extract the correct answer
- **MTP=2 acceptance rate**: ~56% mean across the matrix
- **All 128K context cells succeed** — no OOM or timeout failures (unlike prior TP=2 runs which failed at 64K+)

---

## Hardware Requirements

The tested hardware is **4× RTX PRO 6000 Blackwell Workstation Edition** (96 GiB VRAM each, PCIe Gen5 x16, no NVLink).

| Configuration | Quant | Notes |
|---|---|---|
| **4× RTX PRO 6000** | FP8, TP=4 | This study. ~92 GiB per GPU at 1M max context. |
| **2× RTX PRO 6000** | FP8, TP=2 | Possible for shorter contexts; 64K+ may OOM. |

### GPU Topology (4× PCIe Gen5 x16, no NVLink)

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NODE    NODE    NODE
GPU1    NODE     X      NODE    NODE
GPU2    NODE    NODE     X      NODE
GPU3    NODE    NODE    NODE     X
```

All GPUs connected via PCIe through a single NUMA node (no NVLink). Verified at full **PCIe Gen5 (32 GT/s) x16** under active GPU compute pressure.

### Driver / CUDA Versions

- NVIDIA Driver: **595.71.05**
- CUDA Version: **13.2** (host); **13.2.1** (container)
- VRAM per GPU: 97,887 MiB
- Power limit: 600 W per card
- GPU clocks: 2,752–2,812 MHz sustained under load (max: 3,090 MHz)
- Host OS: Linux 7.0.10-arch1-1, 251 GB system RAM, Intel Xeon W (W790E-SAGE motherboard)

---

## NCCL & Environment Variables

### Container-internal defaults (this study)

The `cstechdev/dsv4-flash` image ships with these NCCL/env vars baked in:

```bash
NCCL_IB_DISABLE=1                    # No InfiniBand
NCCL_SOCKET_IFNAME=lo                # Loopback for PCIe-only topology
NCCL_PROTO=LL,LL128,Simple           # Protocol selection
NCCL_ALLOC_P2P_NET_LL_BUFFERS=1      # P2P buffer allocation
NCCL_P2P_LEVEL=SYS                   # PCIe through NUMA
NCCL_NET_GDR_LEVEL=SYS               # GPU-direct RDMA level
NCCL_MIN_NCHANNELS=8                 # Channel count for TP=4
VLLM_WORKER_MULTIPROC_METHOD=spawn   # Required for multi-GPU
VLLM_ALLREDUCE_USE_SYMM_MEM=0        # Disable symmetric-memory (no NVLink)
OMP_NUM_THREADS=8                    # OpenMP thread limit
FLASHINFER_DISABLE_VERSION_CHECK=1   # Skip flashinfer version check
```

### Diagnosing NCCL deadlocks

If GPUs hit 100% utilization at ~140 W with no VRAM growth, suspect an NCCL deadlock:

1. Check topology: `nvidia-smi topo -m`
2. Try `NCCL_P2P_LEVEL=PHB` instead of `SYS`
3. Or disable P2P entirely: `unset NCCL_P2P_LEVEL; NCCL_P2P_DISABLE=0`
4. Enable debug: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`

---

## Launch Commands

The recommended launch command is in [Quick Start](#quick-start) above.

### Key Flags Explained

| Flag | Value | Purpose |
|---|---|---|
| `--tensor-parallel-size` | `4` | All 4 RTX PRO 6000 Blackwells |
| `--kv-cache-dtype` | `fp8` | FP8 KV cache — halves memory vs FP16 |
| `--block-size` | `256` | Larger blocks for long-context efficiency |
| `--max-model-len` | `1048576` | 1M token context window |
| `--max-num-seqs` | `8` | Limits concurrent sequences (see §Known Issues) |
| `--compilation-config` | `FULL_AND_PIECEWISE + all custom_ops` | Max CUDA graph compilation |
| `--enable-flashinfer-autotune` | — | Auto-tunes flash attention kernels |
| `--disable-custom-all-reduce` | — | Falls back to NCCL (needed for PCIe topology) |
| `--tokenizer-mode` | `deepseek_v4` | DeepSeek V4 native tokenizer |
| `--speculative-config.method` | `mtp` | MTP speculative decoding |
| `--speculative-config.num_speculative_tokens` | `2` | 2 draft tokens per step |

### Without Speculative Decoding

```bash
# Remove these two flags:
#   --speculative-config.method mtp
#   --speculative-config.num_speculative_tokens 2
# Expect ~15% lower single-stream throughput, ~5% lower aggregate at c=8.
```

---

## Docker Images

| Image | Status | Notes |
|---|---|---|
| **`cstechdev/dsv4-flash`** (`27b80536a362`) | **Recommended** | Current production image. vLLM `0.21.1rc1.dev339+g1967a5627bc3`. FP8 native, MTP-2, flashinfer autotune. |
| `lavd/vllm:jasl-dsv4-5-16-26` | Validated | MTP accuracy regression fixed. Prefix caching works with MTP. |
| `lavd/vllm:b12x-6-2-13.2` | Validated | b12x MoE kernel path. |
| `lavd/vllm:b12x-d4f-patched` | Validated | b12x + D4F patch. |
| `voipmonitor/dsv4-flash:lucifer-mxfp4-cutlass-20260603` | Validated | MXFP4 cutlass kernels. |
| `repne/vllm:v15` | Validated | Repne fork with DFlash support. |
| `repne/vllm:v17` | Validated | Latest Repne fork. |
| `verdictai/deepseek-v4-flash` | Validated | 1M context TP=2 with cudagraph fix. Scored 30/30 Estonia. |
| `vllm/vllm-openai:nightly` | Experimental | Upstream nightly. |

---

## MTP / Speculative Decoding

### MTP flags

```
--speculative-config.method mtp
--speculative-config.num_speculative_tokens 2
```

### Key Findings

- **MTP=2 is the production sweet spot** for DeepSeek-V4-Flash on 4× RTX PRO 6000. The `max-num-seqs=8` ceiling means speculative decoding has less impact at high concurrency than on models with higher batch capacity.
- **Mean spec acceptance rate: ~56%** across the full matrix.
- **ITL at c=1: 4.6 ms** — extremely fast single-stream decoding.
- **ITL at c=8: 13.3 ms** — comfortably under 50 ms streaming threshold even at peak throughput.
- **Speculative decoding is not bitwise-lossless even at temp=0.** MTP sampling introduces stylistic divergence vs no-spec on identical prompts. Differences are stylistic, not factual.

### MTP=2 vs No-Spec Speedup

| Concurrency | No-Spec (est.) | MTP=2 | Speedup |
|---|---|---|---|
| c=1 | ~187 | 214.9 | **1.15×** |
| c=8 | ~537 | 618.2 | **1.15×** |

MTP=2 provides a consistent ~15% boost across all concurrency levels.

---

## Benchmark Methodology

### Tool

All measurements taken with [`llm-inference-bench`](https://github.com/voipmonitor/llm-inference-bench) v0.4.24, the animated Rich TUI benchmark suite.

### Sustained Decode Matrix

- **Concurrency levels:** 1, 2, 4, 8, 16, 32, 64, 128
- **Context lengths:** 0, 16K, 32K, 64K, 128K tokens
- **Duration:** 30 seconds per cell
- **Max tokens:** 2,048 (ignore EOS)
- **Warmup:** 3 seconds at 32K context, concurrency=1
- **Metric:** `aggregate_tps = Σ completion_tokens / wall_time`
- **Prefill:** Integrated decode scout (single probe per context length)

### Estonia Profile

- **Profile:** `--test-profile estonia` (built-in GLM long-context completion)
- **Prompt:** 707,372 characters (~113K tokens) embedded as zlib+base64 blob
- **Concurrency:** 30 (fixed)
- **Runs:** 30
- **Max tokens:** 40,000
- **Scoring:** Regex `\bestonia\b` on final answer line
- **Temperature:** 0.0, top_p=1.0, seed=42

### Hardware Monitoring

GPU metrics sampled from `nvidia-smi` during each cell's measurement window. PCIe throughput monitored via `nvidia-smi dmon`.

---

## Detailed Benchmark Tables

### Full Decode Matrix with TTFT/ITL (ctx=0)

```
conc    TTFT avg    TTFT p99     ITL avg     ITL p99     tok/s
  1     3,071.8ms  11,487.5ms      4.6ms       4.8ms    214.9
  2       317.8ms     594.8ms      5.7ms       5.9ms    343.4
  4       179.4ms     191.7ms      9.9ms      10.2ms    406.9
  8       241.8ms     523.6ms     13.3ms      14.9ms    618.2
 16    21,380.7ms  29,928.4ms     14.1ms      16.2ms    557.4
 32    41,502.3ms  85,908.5ms     14.0ms      16.3ms    558.3
 64    38,036.0ms  89,138.8ms     14.5ms      15.2ms    533.6
128    34,962.1ms  90,111.6ms     14.6ms      15.5ms    534.7
```

### Full Decode Matrix with TTFT/ITL (ctx=128K)

```
conc    TTFT avg    TTFT p99     ITL avg     ITL p99     tok/s
  1    20,184.2ms  20,184.2ms      5.0ms       5.0ms    196.9
  2    20,419.8ms  20,532.9ms      6.5ms       6.5ms    300.6
  4    20,470.2ms  20,553.0ms     12.5ms      12.6ms    319.4
  8    20,777.0ms  20,931.1ms     15.7ms      16.0ms    506.6
 16    42,186.8ms  58,310.7ms     16.3ms      18.3ms    481.0
 32    61,600.7ms 119,213.3ms     16.5ms      19.2ms    471.5
 64    58,320.2ms 119,475.8ms     16.4ms      18.3ms    476.4
128           —           —          —           —        —
```

> Note: At 128K context + concurrency 128, the server is capacity-limited (KV budget exhausted). Cell skipped.

### Decode Throughput by Context (all concurrency levels)

```
ctx\conc      1       2       4       8      16      32      64     128
   0       214.9   343.4   406.9   618.2   557.4   558.3   533.6   534.7
  16k      205.5   318.2   343.5   547.2   516.2   516.3   510.9   514.2
  32k      202.9   308.9   345.4   540.0   521.1   509.0   516.1   514.7
  64k      204.7   314.0   328.8   527.0   493.6   487.1   488.0   485.5
 128k      196.9   300.6   319.4   506.6   481.0   471.5   476.4      —
```

### Context Sensitivity (throughput ratio vs ctx=0 at same concurrency)

```
ctx\conc      1       2       4       8      16      32      64
  16k      0.956   0.927   0.844   0.885   0.926   0.925   0.957
  32k      0.944   0.900   0.849   0.874   0.935   0.912   0.967
  64k      0.953   0.914   0.808   0.852   0.885   0.872   0.915
 128k      0.916   0.875   0.785   0.819   0.863   0.845   0.893
```

DeepSeek-V4-Flash shows remarkable context resilience — even at 128K context, throughput stays within 78–92% of the zero-context baseline. This is a strength of the MLA architecture.

---

## Estonia Profile Details

### What is the Estonia test?

The Estonia profile (`--test-profile estonia`) is a built-in long-context completion benchmark in `llm-inference-bench`. It embeds a ~707K-character prompt describing the GLM dense-MLA vs NSA architecture comparison, then asks the model to identify the country being discussed. The expected answer is "Estonia."

This tests:
- **Long-context retrieval**: Can the model find the relevant information in ~113K tokens of technical text?
- **Reasoning efficiency**: How many tokens does the model need to reach the correct answer?
- **Distractor resistance**: Does the model get confused by similar countries (Latvia, Lithuania)?

### Results

```
Metric                  Value
─────────────────────────────────────
Correct                 28 / 30  (93.3%)
Wrong                    2
Hit max tokens (40K)     0

Token count to answer:
  min                    2,055
  p50                    4,198
  p90                   12,020
  p99                   27,004
  max                   28,785
  mean                   6,274

TTFT avg               112.8 ms
TTFT p90               226.1 ms
Gen tok/s avg            69.9
Aggregate gen tok/s      73.7
Aggregate e2e tok/s      31.7
```

### Wrong Run Analysis

Both failures answered with "Latvia" instead of "Estonia":

| Run | Tokens | Final Answer |
|---|---|---|
| 8 | 10,348 | "Latvia" |
| 21 | 28,785 | "Mirel Industrial, which is headquartered in **Latvia**" |

The model correctly identifies the company (Mirel Industrial) but confuses its country of headquarters. Both failures are in the same region — the model retrieves the right entity but misattributes the geography. This is a known challenge for long-context models: entity resolution at 113K+ tokens can blur geographically close alternatives.

---

## Memory Usage (VRAM)

- **FP8 on 4× GPUs (TP=4):** ~92 GiB per GPU at 1M max context
- **`--gpu-memory-utilization 0.9`** used in this study
- Each RTX PRO 6000 Blackwell: **97,887 MiB** total VRAM
- KV cache budget: **11,389,696 tokens** (44,491 blocks × 256)

### Engine State at Production Config (FP8, TP=4, MTP=2)

- KV cache: **11,389,696 tokens** (enough for ~11 concurrent 1M-context requests)
- Boot time: ~120 s warm compile cache, ~295 s cold
- PCIe rx/tx at peak throughput (c=8): ~4,747 / 3,821 MB/s

### VRAM Breakdown (per GPU, approximate)

| Component | Memory |
|---|---|
| Model weights (FP8, 671B MoE, sharded TP=4) | ~42 GiB |
| KV cache (FP8, 11.4M tokens, block=256) | ~22 GiB |
| MTP head (2 draft tokens) | ~2 GiB |
| CUDA graphs + scratch | ~10 GiB |
| Flashinfer + framework overhead | ~8 GiB |
| **Total** | **~84 GiB** |
| Reserved (gpu-memory-utilization 0.9) | ~8 GiB |

---

## Known Issues and Fixes

### 1. `max-num-seqs=8` is the throughput ceiling

**Symptom:** Throughput peaks at c=8 (618 tok/s) then plateaus at ~530–560 tok/s at higher concurrency. Queue fraction hits 1.0 at c=128.

**Cause:** `--max-num-seqs 8` hard-limits the number of sequences the engine processes concurrently. At c > 8, requests queue. This is a deliberate tradeoff to fit 1M context — each sequence at 1M ctx consumes ~88 GiB of KV cache across 4 GPUs.

**Workaround:** For higher throughput at the cost of max context, increase `--max-num-seqs` and reduce `--max-model-len` proportionally. For example, `--max-model-len 262144 --max-num-seqs 32` would likely push peak throughput well past 1,000 tok/s.

### 2. No NVLink — PCIe bottleneck at high concurrency

**Symptom:** PCIe rx/tx reaches 4–6 GB/s at c=8+, indicating the PCIe Gen5 x16 bus is a bottleneck for TP=4 all-reduce.

**Cause:** 4 GPUs communicating over PCIe through a single NUMA node. No NVLink means all inter-GPU traffic traverses the PCIe bus.

**Workaround:** This is a hardware limitation. NVLink-equipped systems (e.g., NVIDIA HGX) would show significantly higher throughput at TP=4.

### 3. Estonia profile: Baltic confusion

**Symptom:** 2/30 runs answered "Latvia" instead of "Estonia."

**Cause:** The long-context prompt discusses multiple Baltic countries. At 113K+ tokens, the model occasionally conflates geographically close entities.

**Workaround:** None needed — 93.3% is strong. For production, consider prompt engineering to reinforce the specific entity to track.

### 4. `--disable-custom-all-reduce` required

**Symptom:** Without this flag, the engine hangs at startup on PCIe-only Blackwell systems.

**Cause:** vLLM's custom all-reduce kernel requires NVLink peer mappings. On PCIe-only topology, it deadlocks.

**Workaround:** Always include `--disable-custom-all-reduce` on Blackwell PCIe systems without NVLink.

### 5. Cold boot is slow (~295 s)

**Symptom:** First request takes ~5 minutes.

**Cause:** JIT compilation of CUDA graphs for all 4 GPUs + flashinfer autotune.

**Workaround:** Use `--compilation-config` caching. Subsequent starts with warm cache take ~120 s.

---

## Data Files

Raw benchmark results are included in this repository:

- [`data/benchmark_results_standard.json`](data/benchmark_results_standard.json) — Full sustained decode matrix (41 cells)
- [`data/benchmark_results_estonia.json`](data/benchmark_results_estonia.json) — Estonia profile results (30 runs)

---

*Benchmarked 2026-06-06 on **RASPUTIN** · 4× NVIDIA RTX PRO 6000 Blackwell · vLLM v0.21.1rc1 · `cstechdev/dsv4-flash`*

*Toolchain: [`llm-inference-bench`](https://github.com/voipmonitor/llm-inference-bench) v0.4.24 · Methodology: sustained decode 30s/cell + completion-token profile*
