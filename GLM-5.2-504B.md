# GLM-5.2-504B (0xSero Router-KD NVFP4) on 4× RTX PRO 6000 Blackwell

> A self-contained, reproducible recipe for serving **GLM-5.2** at **262K context** on four PCIe-only RTX PRO 6000 Blackwell GPUs, with **≈95 tok/s** single-stream decode and **96.7%** on the Estonia long-context profile. Everything below — image, weights, launch command, and the two non-obvious fixes — is exactly what produced the numbers in this document.

> Part of the [DeepSeek-V4-Flash on RTX PRO 6000 Blackwell](README.md) study repo — this is a standalone companion writeup for a different model on the same hardware.

## TL;DR — Copy-Paste Quick Start

```bash
# 1. Download the weights (~318 GB, 64 shards). aria2c 16-parallel is the reliable fast path
#    (the hf CLI stalls on Xet init for this repo).
DEST=/models/GLM-5.2-504B          # any path with ~320 GB free
mkdir -p "$DEST"
hf download 0xSero/GLM-5.2-504B --local-dir "$DEST"   # or use aria2c, see §Downloading

# 2. Pull the serving image (b12x + the MTP-DCP shard-draft fix)
docker pull madeby561/vllm:dark-devotion-df8ad3b-b12x5af873a-dcpglobaltopk-cu132-20260621-mtpdcpfix

# 3. Serve (TP4 / DCP4 / MTP5, 262K context, NVFP4). See §Launch Command for the full block.
#    Two things that are NOT optional:
#      --load-format safetensors      (NOT fastsafetensors — see §Gotcha #1)
#      -e VLLM_DCP_SHARD_DRAFT=1       (the deep-MTP fix — see §Gotcha #2)
```

API endpoint: `http://localhost:8000/v1` (OpenAI-compatible). Cold start ~6 min (model load + graph capture).

## What This Is

| | |
|---|---|
| **Model** | [`0xSero/GLM-5.2-504B`](https://huggingface.co/0xSero/GLM-5.2-504B) |
| **Architecture** | `GlmMoeDsaForCausalLM` — MoE, 78 layers (3 dense + 75 MoE) + 1 MTP layer, DeepSeek-style MLA attention with a DSA sparse indexer, hidden size 6144 |
| **Provenance** | 34%-expert-pruned (REAP keep-**168** of 256 routed experts), then **Router-KD recovered** (gate-only distillation to the unpruned GLM-5.2 teacher) |
| **Quantization** | **NVFP4** (modelopt) on routed experts; BF16 router / attention / shared expert |
| **Size on disk** | 318 GB · 64 safetensors shards (63 indexed + 1 MTP-layer overflow) |
| **Context served** | **262,144 tokens** (KV pool 482,816 tokens via DCP=4) |
| **Image** | [`madeby561/vllm:dark-devotion-df8ad3b-b12x5af873a-dcpglobaltopk-cu132-20260621-mtpdcpfix`](https://hub.docker.com/r/madeby561/vllm) |
| **vLLM build** | `0.11.2.dev279+dark.devotion.df8ad3b.b12x5af873a.fi9c5ed7c.dcpglobaltopk.cu132.20260621` |
| **Parallelism** | TP=4 + **DCP=4** (decode-context-parallel) + **MTP=5** (speculative) |

## Downloading the Weights

The repo is **Xet-backed**, and the `hf` CLI can stall on Xet init. The reliable fast path on a PCIe box is **`aria2c` with 16 parallel connections** (saturates the link, fully resumable):

```bash
DEST=/models/GLM-5.2-504B
mkdir -p "$DEST"

# Build a URL list from the HF API, then aria2c it
python3 - <<'PY'
import os, json, urllib.request
tok = os.environ["HF_TOKEN"]                       # export HF_TOKEN=hf_...
h = {"Authorization": f"Bearer {tok}"}
dest = "/models/GLM-5.2-504B"
d = json.load(urllib.request.urlopen(urllib.request.Request(
    "https://huggingface.co/api/models/0xSero/GLM-5.2-504B", headers=h), timeout=30))
lines = []
for s in d["siblings"]:
    f = s["rfilename"]
    lines += [f"https://huggingface.co/0xSero/GLM-5.2-504B/resolve/main/{f}",
              f"  dir={dest}", f"  out={f}", ""]
open("/tmp/glm52-urls.txt", "w").write("\n".join(lines))
PY

aria2c --input-file=/tmp/glm52-urls.txt --dir="$DEST" \
  --max-concurrent-downloads=16 --max-connection-per-server=4 --split=4 \
  --continue=true --auto-file-renaming=false --allow-overwrite=true \
  --header="Authorization: Bearer $HF_TOKEN"
```

Verify the result — 64 shards, and the config is what the loader expects:

```bash
ls "$DEST"/*.safetensors | wc -l        # -> 64
python3 -c "import json;c=json.load(open('$DEST/config.json'));\
print(c['model_type'], c['n_routed_experts'], c['quantization_config']['quant_algo'])"
# -> glm_moe_dsa 168 NVFP4
```

## Launch Command

Run from the image with `--gpus` device selection (this host has **no** `nvidia` docker runtime — use `--gpus`, not `--runtime nvidia`):

```bash
IMG=madeby561/vllm:dark-devotion-df8ad3b-b12x5af873a-dcpglobaltopk-cu132-20260621-mtpdcpfix
MODEL_DIR=/models/GLM-5.2-504B

docker run -d --name glm52 \
  --gpus '"device=0,1,2,3"' --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR":/model:ro \
  -v /root/.cache/glm52:/cache \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e OMP_NUM_THREADS=16 -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_B12X_FP8_GEMM=1 -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_DCP_GLOBAL_TOPK=1 -e VLLM_DCP_SHARD_DRAFT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e USES_B12X=True -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A16_TC_DECODE=1 -e B12X_MOE_FORCE_A16=1 \
  "$IMG" \
  bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE; cd /; \
  exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
    --served-model-name GLM-5.2 --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --decode-context-parallel-size 4 --dcp-comm-backend ag_rs --dcp-kv-cache-interleave-size 1 \
    --enable-chunked-prefill --enable-prefix-caching \
    --max-model-len 262144 \
    --load-format safetensors \
    --async-scheduling -cc.pass_config.fuse_allreduce_rms=True \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 8192 --max-num-seqs 8 --max-cudagraph-capture-size 48 \
    --quantization modelopt_fp4 --attention-backend B12X_MLA_SPARSE --moe-backend b12x --kv-cache-dtype fp8 \
    --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 \
    --hf-overrides '"'"'{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'"'"' \
    --speculative-config '"'"'{"model":"/model","method":"mtp","num_speculative_tokens":5,"moe_backend":"b12x","draft_sample_method":"probabilistic"}'"'"''
```

> **The two annotated lines above are mandatory** — see [§The Two Gotchas](#the-two-gotchas-dont-skip-these):
> - `--load-format safetensors` (Gotcha #1)
> - `-e VLLM_DCP_GLOBAL_TOPK=1 -e VLLM_DCP_SHARD_DRAFT=1` (Gotcha #2)

Smoke-test once it's up (cold start ~6 min):

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"GLM-5.2",
  "messages":[{"role":"user","content":"Write a Python one-liner for nth Fibonacci. Code only."}],
  "max_tokens":4096, "temperature":0.6, "top_p":0.95, "repetition_penalty":1.0,
  "chat_template_kwargs":{"reasoning_effort":"high"}
}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

> **Sampling matters.** Use `temperature 0.6, top_p 0.95, repetition_penalty 1.0`. A repetition penalty **> 1.0** accumulates over GLM-5.2's long reasoning and can spiral it into token-salad — keep it at exactly 1.0.
> **Give it room to think.** GLM-5.2 is a heavy reasoner; pass a generous `max_tokens` (≥2048) or it gets cut off mid-`<think>` and returns empty content.

## The Two Gotchas (don't skip these)

**Gotcha #1 — `--load-format safetensors`, never `fastsafetensors`.**
The 0xSero checkpoint physically duplicates the MTP/layer-78 expert weights across several shards (the index `weight_map` is self-consistent, but the bytes overlap). `fastsafetensors` globs every `.safetensors` file and demands globally-unique keys, so it crashes during weight load:
```
Exception: FilesBufferOnDevice: key model.layers.78.mlp.experts.67.up_proj.weight must be unique among files
→ WorkerProc initialization failed → Engine core initialization failed
```
The standard `safetensors` loader reads each tensor from its canonical shard via `model.safetensors.index.json` and ignores the physical duplicates. Cost: ~30 s slower load. Otherwise harmless.

**Gotcha #2 — `VLLM_DCP_SHARD_DRAFT=1` (+ `VLLM_DCP_GLOBAL_TOPK=1`).**
Under DCP, the MTP draft model needs its sparse-indexer top-k buffer sharded too. Without the fix the draft crashes at speculator cudagraph capture (`B12X sparse indexer DCP requires topk_scores_buffer`). The `...mtpdcpfix` image carries the patch (`deepseek_mtp.py` `topk_scores_buffer`); these two env vars activate it. With it, MTP acceptance climbs from ~2.7 to **~4.0–4.6 accepted tokens/step** (~61–73% draft acceptance) — the single biggest decode-speed lever at DCP=4. vLLM prints them as “unknown environment variable”; that warning is **expected** (they’re read by the b12x/GLM code paths, not the central registry).

## Results

Benchmarked with [`llm-inference-bench`](https://github.com/voipmonitor/llm-inference-bench) on the live endpoint, sized to the model’s real limits (`max-num-seqs=8`, 262K context).

### Sustained Decode — aggregate tok/s, 30 s/cell

Concurrency is capped at `max-num-seqs=8`. Cells marked `—` exceed the 482,816-token KV pool at that (context × concurrency) and are skipped by the scheduler:

```
ctx\conc      1       2       4       8
   0        92.5   125.3   162.1   184.2
  16k       89.9   126.4   149.7   175.4
  32k       81.6    99.1   183.9   195.8
  64k       87.0      —    146.9      —
 128k       54.2    97.8      —       —
 256k       45.2      —       —       —
```

### Prefill — tok/s + TTFT

```
ctx      TTFT      tok/s     tokens
 8k     6.976s    1,175      8,199
16k    13.509s    1,201     16,228
32k    27.034s    1,196     32,321
64k    54.150s    1,191     64,513
128k  108.872s    1,184    128,887
```

Prefill holds a steady **~1,190 tok/s** across 8K→128K; TTFT scales linearly with prompt length (a 128K prompt is ~109 s to first token).

### Estonia Long-Context Profile

**96.7% correct** (29/30) at concurrency=30, **zero truncations** (0 runs hit `max_tokens`) — the Router-KD recovery delivers its anti-loop promise on this hardware. The model averages just **3,634 tokens** of reasoning to reach the answer.

```
Correct:    29 / 30  (96.7%)
Wrong:       1
Errors:      0
Hit max:     0

Token count to answer:
  min:      887
  p50:    3,005
  p90:    6,560
  p99:    8,762
  max:    9,336
  mean:   3,634

TTFT:  avg=177.1s  p90=341.5s   (long-context probes, ~113K-token prompt)
Gen tok/s:  avg=29.9  agg=29.9
E2E tok/s:  agg=12.2
```

### At a Glance

| Scenario | tok/s | Notes |
|---|---|---|
| Single user, ctx=0 | **92.5** | TP4 / DCP4 / MTP5, NVFP4 |
| Single user, 128K context | **54.2** | KV/comm pressure at depth under DCP=4 |
| Single user, **262K** (max) | **45.2** | Full context window, single stream |
| Multi-user c=8 @ 32K | **195.8** ⭐ | Peak aggregate |
| Multi-user c=8 @ ctx=0 | **184.2** | Scales cleanly to the `max-num-seqs=8` cap |
| Estonia (c=30) | **29.9** agg gen | **96.7%** correct, 0 truncations |

### How to Reproduce These Numbers

```bash
git clone https://github.com/voipmonitor/llm-inference-bench && cd llm-inference-bench

# Full decode spread (sized to this model's limits) + prefill, animated TUI
python llm_decode_bench.py --host 127.0.0.1 --port 8000 \
  --model GLM-5.2 --api-key EMPTY \
  --concurrency 1,2,4,8 --contexts 0,16k,32k,64k,128k,256k \
  --duration 30 --max-tokens 2048 --display-mode screen \
  --output decode-spread.json

# Estonia long-context correctness profile
python llm_decode_bench.py --host 127.0.0.1 --port 8000 \
  --model GLM-5.2 --api-key EMPTY \
  --completion-stats --test-profile estonia --display-mode screen \
  --output estonia.json
```

Raw result JSONs are in [`data/`](data/):

- [`data/glm52_0xsero_benchmark_results_standard.json`](data/glm52_0xsero_benchmark_results_standard.json) — sustained decode + prefill matrix
- [`data/glm52_0xsero_benchmark_results_estonia.json`](data/glm52_0xsero_benchmark_results_estonia.json) — Estonia profile (30 runs)

> **Note:** `0xSero/GLM-5.2-504B` had a same-day sibling, `0xSero/GLM-5.2-504B-K` (structurally identical — `glm_moe_dsa`, 168 experts, NVFP4, 318 GB). The numbers above were measured on `0xSero/GLM-5.2-504B`; the launch recipe applies unchanged to either.

---

*Benchmarked on **RASPUTIN** · 4× NVIDIA RTX PRO 6000 Blackwell · vLLM `0.11.2.dev279+dark.devotion...dcpglobaltopk` · `madeby561/vllm:...mtpdcpfix`*
