# Proximal benchmarking

Measures what one **student-day** of an AI tutor costs to serve on a given machine,
using open-weight models on your own GPUs. One command runs every benchmark and
writes a self-contained HTML report you can open in any browser and share.

A student-day, as specified in `config/spec.yaml`:

| Line | Daily volume | Model benchmarked |
| --- | --- | --- |
| 1. ASR (speech to text) | 10 min of audio | Bodhan Indic-Transcribe-core |
| 2. TTS (text to speech) | 10 min of audio | Bodhan Indic-Speak |
| 3. OCR | 10 pages | Bodhan IndicOCR |
| 4. Tutor LLM | 2M cached + 100K new input + 100K output tokens | Gemma 4 31B (4-bit) |

The report converts measured throughput into **GPU-seconds per unit**, prices them at
your machine's $/GPU-hour, and compares the result with hosted-API list prices and a
$2/student-month target. Every $ figure in the report can be repriced in the browser.

---

## What you need

- **Linux with NVIDIA GPUs** and a recent driver (tested: 3× A100 40GB PCIe, driver 595, CUDA 13.2, Ubuntu 24.04).
  - Gemma 4 31B (4-bit) needs one GPU with **40 GB or more**.
  - ASR, TTS and OCR need about 10 GB each and run on another GPU, or on the same GPU afterwards on a 1-GPU machine.
- **About 150 GB of free disk** for Python environments, model weights and test inputs.
- **Internet access** from the machine (Hugging Face, PyPI). No sudo and no Docker needed.
- **A Hugging Face account** that has accepted the terms of the models you'll download.
  At minimum, open the pages of `google/gemma-4-31B-it-qat-w4a16-ct` and of each `bodhan-ai/...` model
  (Indic-Transcribe-core, Indic-Speak, IndicOCR; Indic Open Model License) and accept their terms.
- **tmux**, so a 3-hour run survives a dropped SSH connection.

---

## Step by step

All commands run on the benchmark machine unless a step says otherwise.

### 1. Get the code

```bash
git clone https://github.com/Proximal-Cloud/proximal-benchmarking.git
cd proximal-benchmarking
```

### 2. Choose where the big files go

Everything large goes under `DATA_ROOT` (default `/data`). If you can't write to `/data`,
pick a folder with ~150 GB free and export it **before** sourcing the environment.
Do this in every new shell:

```bash
export DATA_ROOT=/data            # or e.g. $HOME/bench-data
source scripts/env.sh
```

### 3. Add your machine's price

Open `config/spec.yaml` and add your machine under `machines:` with its cost per
GPU-hour (what you pay, or your internal cost):

```yaml
machines:
  cdac:
    label: C-DAC Airawat (cn14, A100 40GB PCIe)
    usd_per_gpu_hour: 0.20
    hosts: ["cn14*"]
  mybox:                          # the name you'll pass as MACHINE=mybox
    label: My lab box (2x L40S)
    usd_per_gpu_hour: 0.65
    hosts: []                     # optional: hostname patterns that mean this machine
```

You can still change the price later without re-running anything (see "Reprice a report").

### 4. Install the Python environments (~15 min)

```bash
scripts/00_prep_machine.sh
```

This checks the GPUs and disk, installs `uv`, then builds two environments:
`$VENVS/vllm` (vLLM 0.23, ~10 GB) and `$VENVS/bench` (the benchmark clients).

### 5. Log in to Hugging Face (once per machine)

Create a **read** token at <https://huggingface.co/settings/tokens>, then:

```bash
$VENVS/bench/bin/python -c "from huggingface_hub import login; login()"
```

Paste the token when asked. It's stored under `$HF_HOME` on the machine, not in this repo.
Don't use `pip install` on the system Python: Ubuntu blocks it ("externally managed"),
and everything needed is already in the environments from step 4.

### 6. Download the models and the public benchmark data (~35 GB)

```bash
scripts/01_download_models.sh
```

This downloads Gemma 4 31B, the three Bodhan models and ShareGPT (the public chat
dataset used for the public LLM benchmark). Anything already on the machine is skipped,
so it's safe to re-run if the connection drops.

### 7. Set up the ASR / TTS / OCR models

```bash
scripts/02_setup_bodhan_envs.sh
```

This adds three small packages to the vLLM environment, and the Bodhan models run from there.
If a Bodhan model later fails because of a package clash with vLLM, re-run with
`SEPARATE_ENVS=1` to give each model its own environment.

### 8. Build the test inputs (ASR recordings, OCR pages)

```bash
$VENVS/tts/bin/python bench/prep_public_inputs.py --out "$INPUTS" --cache "$TF_HOME/dataset-cache"
```

This downloads and trims public datasets: children's and adult read speech in Hindi, Marathi
and Kannada for ASR, and handwritten answers, textbook pages and scanned book pages for OCR.
The summary at the end says how many clips and pages it made. If a source fails, run it
again with `--inspect` to see that dataset's columns, or `--only <name>` to retry just that source.

> **Licences:** some of these datasets are non-commercial or research-only. Use them for
> internal benchmarking only; don't redistribute them or train on them. `$INPUTS/SOURCES.md` lists each one.

If you skip this step, the run still works: ASR uses audio generated by the TTS
benchmark and OCR uses placeholder pages. Both make the numbers less realistic.

### 9. Check each model starts (~10 min)

```bash
scripts/serve_llm.sh gemma31b-w4                   # starts Gemma on GPU 0, waits until ready
scripts/smoke_llm.sh gemma31b-w4 8001              # asks one tutoring question; check the reply
scripts/serve_llm.sh stop gemma31b-w4
scripts/smoke_bodhan.sh                            # one TTS sentence, transcribes it back, OCRs one page
```

If Gemma doesn't start, the reason is in `$TF_HOME/logs/tf-gemma31b-w4.log`
(see Troubleshooting).

### 10. Quick dry run (~30 min)

Short load levels, to catch problems before the full run:

```bash
QUICK=1 MACHINE=mybox scripts/run_all.sh
```

It should end with a line starting `Report:`. Open that file to check every section has numbers.

### 11. Full run (~2.5–3 hours)

```bash
tmux new -s bench
cd proximal-benchmarking && export DATA_ROOT=/data && source scripts/env.sh
scripts/serve_llm.sh stop all
mkdir -p results && MACHINE=mybox scripts/run_all.sh 2>&1 | tee results/run_all_$(date +%Y%m%d-%H%M).log
```

Press **Ctrl-b, then d** to leave it running; you can log out. To check on it later:
`tmux attach -t bench`.

The run goes in this order, and each benchmark's log goes in the run's folder as it finishes:

| Order | Benchmark | Load levels | Time |
| --- | --- | --- | --- |
| 1 | Gemma 4 31B, Socratic tutoring sessions (our spec) | 2, 4, 8, 16 students | ~45 min |
| 2 | Gemma 4 31B, public benchmark (ShareGPT via `vllm bench serve`) | 2, 4, 8, 16 requests at once | ~20 min |
| 3 | TTS, one stream end to end | 1 stream | ~10 min |
| 4 | ASR | batch 1 | ~5 min |
| 5 | OCR | 1, 2, 4, 8 model copies | ~40 min |
| 6 | TTS, batched on vLLM | 1–64 streams | ~20 min |
| 7 | Cost report | | seconds |

A failed benchmark doesn't stop the run; the report marks that line "not run yet".

### 12. Get the report

When the run prints `Report: …/results/report_<machine>-<date-time>.html`, copy it to
your laptop. Run this on the laptop:

```bash
scp '<user>@<server>:<path to proximal-benchmarking>/results/report_<machine>-*.html' .
```

Open it in a browser. All raw results and logs are in `results/<machine>-<date-time>/`.

---

## Reading the report

- **Top bar:** which machine and $/GPU-hour the numbers use. Pick another machine or type
  any price, and every $ figure, ratio and the chart update. **Reset** goes back.
- **Headline cards:** cost per student-day and per student-month at 100% GPU utilization
  ("pure compute") and at the planning utilization (30%), against the $2 target.
- **LLM: Socratic sessions vs public benchmark:** the same model on our long-context tutoring
  workload and on short ShareGPT requests, as $ per 1M output tokens and as a multiple of
  the hosted API price for the same tokens.
- **Unit costs (raw):** GPU-seconds per unit for every line. Multiply by your own $/GPU-second
  to price them yourself.
- **Cost by utilization:** the same costs at 10–100% utilization (share of the day the GPU does useful work).
- **Operating points:** the load each cost is based on (the highest load that met the latency
  targets), with each target and the measured value.
- **Details:** every load level of every benchmark.

## Reprice or rebuild a report without re-running

```bash
source scripts/env.sh
$VENVS/vllm/bin/python bench/cost_report.py --results results/<machine>-<date-time>                 # after editing config
$VENVS/vllm/bin/python bench/cost_report.py --results results/<machine>-<date-time> --machine mybox
$VENVS/vllm/bin/python bench/cost_report.py --results results/<machine>-<date-time> --usd-per-gpu-hour 0.35
```

This rewrites `report.html`, `report.md` and `report.json` inside that run's folder.
Use it after changing the student-day volumes, latency targets, utilization or prices in
`config/spec.yaml`.

## Run options

Set these in front of `scripts/run_all.sh`:

| Variable | Effect |
| --- | --- |
| `MACHINE=<name>` | machine from `spec.yaml` (sets the price; saved with the results) |
| `QUICK=1` | short load levels (~30 min) |
| `MODELS="a b"` | LLMs to run (default: all in `config/llm_targets.yaml`) |
| `SKIP_LLM=1`, `SKIP_SOCRATIC=1`, `SKIP_PUBLIC=1` | skip all LLM benchmarks, or one of the two |
| `SKIP_BODHAN=1`, `SKIP_TTS_VLLM=1` | skip ASR/one-stream TTS/OCR, or the batched TTS |
| `LLM_GPU=0`, `BODHAN_GPU=2` | which GPUs to use (defaults: first GPU for the LLM, last for the rest) |

## Add another LLM

1. Add it under `models:` in `config/llm_targets.yaml` (name, port, Hugging Face repo, hosted price key).
2. Add a matching case in `scripts/serve_llm.sh` (GPU, port, model, memory share).
3. Add its hosted price under `api_reference.llm` in `config/spec.yaml`.
4. Run `scripts/01_download_models.sh`, then `MODELS="<name>" scripts/run_all.sh`.

It becomes its own numbered line in the report (5. …). The total uses the cheapest LLM.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `error: externally-managed-environment` from pip | Don't install into the system Python. Use the environments: `$VENVS/vllm/bin/python`, `$VENVS/bench/bin/python`. |
| Hugging Face 401 / "gated repo" | Accept the model's terms on its Hugging Face page, then log in (step 5). |
| Gemma fails to load with a `head_dim` error | transformers is too new: `VIRTUAL_ENV=$VENVS/vllm uv pip install transformers==5.12.1` |
| vLLM fails with "Could not find nvcc" | Already handled by `VLLM_USE_FLASHINFER_SAMPLER=0` in `scripts/env.sh`; make sure you sourced it. |
| A script exits silently | Source `scripts/env.sh` first, and check `DATA_ROOT` points to a folder you can write to. |
| Downloads keep stalling | Re-run the same script: downloads resume and finished files are skipped. |
| Any benchmark failed | Its log is in `results/<machine>-<date-time>/<benchmark>.log`; vLLM server logs are in `$TF_HOME/logs/`. |

## Repository layout

| Path | What it does |
| --- | --- |
| `config/spec.yaml` | student-day volumes, machine prices, utilization, latency targets, hosted prices |
| `config/llm_targets.yaml` | the LLMs to benchmark, the Socratic session shape, public-benchmark levels |
| `scripts/env.sh` | folder layout and settings; source it in every shell |
| `scripts/00_prep_machine.sh` … `02_setup_bodhan_envs.sh` | one-time setup (steps 4–7) |
| `scripts/serve_llm.sh`, `smoke_llm.sh`, `smoke_bodhan.sh` | start/stop model servers; quick checks |
| `scripts/run_all.sh` | runs every benchmark, then the report |
| `bench/llm_sessions.py` | Socratic sessions: concurrent students sharing one system prompt, with history |
| `bench/standard_bench.py` | public benchmark: `vllm bench serve` on ShareGPT |
| `bench/tts_bench.py`, `asr_bench.py`, `ocr_bench.py` | the Bodhan benchmarks |
| `bench/cost_report.py`, `html_report.py` | results → `report.html` / `report.md` / `report.json` |
| `bench/prep_public_inputs.py` | builds ASR recordings and OCR pages from public datasets |
| `tests/` | end-to-end test against a mock server, no GPU needed: `bash tests/test_end_to_end.sh` |

Benchmark results, logs and downloaded data stay out of git (see `.gitignore`).
