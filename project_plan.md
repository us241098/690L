# Project Plan: To Think or Not to Think — Understanding Task-Specific Reasoning in LLMs

**Authors:** Junyan Li, Utsav Shukla  
**Course:** COMPSCI 690L, UMass Amherst  
**Date:** April 10, 2026

---

## 1. Literature Review

### 1.1 Chain-of-Thought and Thinking Modes

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Wei et al. — *Chain-of-Thought Prompting Elicits Reasoning in LLMs* | 2022 | 2201.11903 | Foundational CoT paper. Few-shot CoT demonstrations dramatically improve reasoning at scale. Benefits emerge only for large models. |
| Kojima et al. — *Large Language Models are Zero-Shot Reasoners* | 2022 | 2205.11916 | "Let's think step by step" elicits reasoning zero-shot across arithmetic, symbolic, and commonsense tasks. |
| Wang et al. — *Self-Consistency Improves Chain of Thought Reasoning* | 2023 | 2203.11171 | Sampling diverse reasoning paths and majority voting yields +17.9% on GSM8K over greedy CoT. |
| OpenAI — *Learning to Reason with LLMs* | 2024 | — | o1 trained via large-scale RL to use internal CoT. 89th percentile Codeforces, top-500 AIME. |
| DeepSeek-AI — *DeepSeek-R1* | 2025 | 2501.12948 | Pure RL (no SFT) incentivizes reasoning. Self-reflection and verification emerge naturally. AIME 2024 pass@1: 71.0%. |
| Qwen Team — *Qwen3 Technical Report* | 2025 | 2505.09388 | Hybrid thinking/non-thinking in a single model via four-stage training pipeline. `<think>` tags for explicit reasoning. |
| Meincke et al. — *Decreasing Value of Chain of Thought in Prompting* | 2025 | 2506.07142 | CoT provides diminishing returns as models become more capable. Modern reasoning models already think step-by-step. |

### 1.2 Overthinking and When Reasoning Hurts

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Aggarwal et al. — *OptimalThinkingBench* | 2025 | 2508.13141 | Independently optimizing for overthinking or underthinking improves one at the expense of the other. |
| Su et al. — *Between Underthinking and Overthinking* | 2025 | 2505.00127 | Inverted-U relationship: accuracy rises then declines with reasoning length. LLMs overthink simple problems. |
| Hassid et al. — *Don't Overthink It* | 2025 | 2505.17813 | Shorter reasoning chains are up to 34.5% more likely to yield correct answers than the longest chain. |
| Yang et al. — *Mirage of Test-Time Scaling in Reasoning Models* | 2025 | 2506.04210 | Extended thinking increases output variance. Parallel thinking (multiple paths + majority vote) beats serial extension by up to 20%. |
| Zhang et al. — *Structural Understanding of LLM Overthinking* | 2025 | 2510.07880 | Two overthinking patterns: "Explorer" (excessive exploration) and "Late Landing" (excessive verification). |
| Wang et al. — *NoWait: Removing Thinking Tokens Improves Efficiency* | 2025 | 2506.08343 | Suppressing self-reflection tokens ("Wait", "Hmm") reduces CoT length by 27–51% without compromising accuracy. |
| Sui et al. — *Stop Overthinking: A Survey* | 2025 | 2503.16419 | First comprehensive survey on efficient reasoning in LLMs. Documents the overthinking phenomenon systematically. |

### 1.3 Budget Forcing and Reasoning Control

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Muennighoff et al. — *s1: Simple Test-Time Scaling* | 2025 | 2501.19393 | Seminal budget forcing paper. Truncate thinking or extend by appending "Wait" tokens. s1-32B exceeds o1-preview on competition math. |
| Aggarwal & Welleck — *L1: Controlling Reasoning Length with RL* | 2025 | 2503.04697 | RL-based length control (LCPO). Smoother than s1's hard truncation. Outperforms budget forcing. |
| Li et al. — *Steering LLM Thinking with Budget Guidance* | 2025 | 2506.13752 | Fine-tuning-free approach using a lightweight predictor that models a Gamma distribution over remaining thinking length. |
| Han et al. — *Token-Budget-Aware LLM Reasoning (TALE)* | 2025 | 2412.18547 | Dynamically adjusts reasoning tokens based on problem complexity. 67% token cost reduction with competitive accuracy. |
| SelfBudgeter | 2025 | 2505.11274 | Trains model to self-estimate required reasoning budget per query. 61% response length compression on math tasks. |

### 1.4 Test-Time Compute Scaling Laws

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Snell et al. — *Scaling LLM Test-Time Compute* | 2024 | 2408.03314 | Landmark paper: allocating more inference compute can be more efficient than scaling model parameters. |
| Yang et al. — *Thinking-Optimal Scaling (TOPS)* | 2025 | 2502.18080 | Optimal reasoning length differs across domains. Letting the LLM decide its own budget per problem is best. |
| Yeo et al. — *Demystifying Long Chain-of-Thought Reasoning* | 2025 | 2502.03373 | Reasoning emerges with training compute but is not guaranteed. Error correction is inherent but hard to incentivize. |
| Art of Scaling Test-Time Compute | 2025 | 2512.02008 | First large-scale TTS study (30B+ tokens, 8 models). No single TTS strategy universally dominates. |

### 1.5 Task-Specific Reasoning Differences

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Marjanovic et al. — *DeepSeek-R1 Thoughtology* | 2025 | 2504.07128 | "Sweet spot" of reasoning length per problem type. Longer chains show substantially lower accuracy past the sweet spot. |
| Zhao et al. — *TTS Not Effective for Knowledge-Intensive Tasks* | 2025 | 2509.06861 | Reasoning scaling doesn't help knowledge tasks and can increase hallucinations. |
| Liu et al. — *DiffAdapt* | 2025 | 2510.19669 | U-shaped entropy pattern: 82.3% of problems benefit from an "Easy" strategy. Different difficulties need fundamentally different approaches. |
| Alomrani et al. — *Reasoning on a Budget: A Survey* | 2025 | 2507.02076 | Comprehensive taxonomy: L1-controllability (fixed budgets) vs. L2-adaptiveness (dynamic scaling). |

### 1.6 Gap and Contribution

No existing work has performed a systematic cross-task analysis at varying reasoning budgets using models of multiple sizes from the same family. Thoughtology (Marjanovic et al.) comes closest but studies only DeepSeek-R1 (single model size) without budget forcing. Our work fills this gap by using the Qwen3.5 family (0.8B to 27B) with explicit budget forcing across 7 diverse benchmarks.

---

## 2. Project Design

### 2.1 Models

| Model | Precision | VRAM (est.) | Rationale |
|-------|-----------|-------------|-----------|
| Qwen3.5-0.8B | FP16 | ~2 GB | Smallest — test if reasoning helps tiny models at all |
| Qwen3.5-4B | FP16 | ~8 GB | Small-medium capacity |
| Qwen3.5-9B | FP16 | ~18 GB | Medium capacity |
| Qwen3.5-27B | FP8 | ~32 GB | Largest that fits on single H100. Native FP8 tensor core support |

All Qwen3.5 models support multimodal inputs (vision built-in) and implement thinking via `<think>...</think>` tags.

### 2.2 Benchmarks

| Category | Benchmark | Source | Subset Size | Metric | Expected Curve |
|----------|-----------|--------|-------------|--------|----------------|
| Math (easy) | GSM8K | `openai/gsm8k` | 500 | Exact match accuracy | Peaks then plateaus/degrades |
| Math (hard) | MATH-500 | `HuggingFaceH4/MATH-500` | 500 | Exact match accuracy | Monotonically increasing |
| Knowledge QA | MMLU-Pro | `TIGER-Lab/MMLU-Pro` | 500 (stratified) | Multiple-choice accuracy | Moderate benefit |
| Code | HumanEval+ | `evalplus` package | 164 (all) | pass@1 | Strong benefit |
| Summarization | XSum | `EdinburghNLP/xsum` | 500 | ROUGE-1/2/L + BERTScore | Flat or declining |
| Instruction Following | IFEval | `google/IFEval` | ~500 (all) | Strict/loose accuracy | Mixed |
| Vision + Math | MathVista | `AI4Math/MathVista` | 500 (testmini) | Accuracy | Strong benefit |

### 2.3 Reasoning Budget Levels

Seven levels of reasoning token budget:

```
budgets = [0, 64, 256, 1024, 2048, 4096, 8192]
```

- **Budget = 0**: Thinking disabled entirely (`enable_thinking=False`)
- **Budget = 64–8192**: Thinking enabled, forcefully truncated via budget forcing at the specified token count

### 2.4 Budget Forcing Implementation

We adopt a LogitsProcessor-based approach:

1. Enable thinking mode (`enable_thinking=True`)
2. Track token count inside `<think>...</think>` during generation
3. At `budget - 1` tokens: suppress all tokens except newline character
4. At `budget` tokens: force the `</think>` token (token ID **248069** for Qwen3.5)
5. Continue generation normally for the final answer

This follows the approach described by Muennighoff et al. (s1) and implemented by Zach Mueller's `ThinkingTokenBudgetProcessor`.

### 2.5 Inference Stack

```
vLLM (FP8 on H100) → OpenAI-compatible API → Python evaluation scripts
```

```bash
vllm serve Qwen/Qwen3.5-27B --port 8000 \
  --quantization fp8 \
  --reasoning-parser qwen3 \
  --max-model-len 32768
```

**Sampling parameters (thinking mode):** `temperature=0.6, top_p=0.95, top_k=20`  
**Sampling parameters (no thinking):** `temperature=0.7, top_p=0.8, top_k=20`

Do NOT use greedy decoding (`temperature=0`) with thinking mode — causes infinite repetition loops.

### 2.6 Codebase Structure

```
690L/
├── configs/
│   ├── models.yaml            # Model names, precisions, vLLM args
│   ├── benchmarks.yaml        # Dataset IDs, subset sizes, metrics
│   └── budgets.yaml           # Reasoning token budget levels
├── src/
│   ├── inference/
│   │   ├── server.py          # vLLM server launcher per model
│   │   ├── client.py          # OpenAI-compatible API client
│   │   └── budget_forcing.py  # LogitsProcessor for token budget control
│   ├── data/
│   │   ├── loader.py          # Unified dataset loader (HF datasets → prompts)
│   │   ├── gsm8k.py           # GSM8K prompt formatting + answer extraction
│   │   ├── math500.py         # MATH-500 formatting
│   │   ├── mmlu_pro.py        # MMLU-Pro formatting
│   │   ├── humaneval.py       # HumanEval+ formatting
│   │   ├── xsum.py            # XSum formatting
│   │   ├── ifeval.py          # IFEval formatting
│   │   └── mathvista.py       # MathVista formatting (with images)
│   ├── eval/
│   │   ├── metrics.py         # Exact match, ROUGE, BERTScore, pass@1
│   │   ├── judge_ifeval.py    # IFEval constraint checker
│   │   └── judge_code.py      # Sandbox code execution (evalplus)
│   └── analysis/
│       ├── curves.py          # Plot Acc(t) curves per task per model
│       ├── heatmap.py         # Model x task x budget heatmaps
│       └── stats.py           # Confidence intervals, bootstrap
├── scripts/
│   ├── run_experiment.py      # Main loop: for model x budget x benchmark
│   └── run_all.sh             # Full experiment sweep
├── results/                   # JSON outputs per (model, benchmark, budget)
└── figures/                   # Generated plots
```

### 2.7 Experiment Loop

```python
for model in [0.8B, 4B, 9B, 27B]:
    start_vllm_server(model)
    for benchmark in [gsm8k, math500, mmlu_pro, humaneval, xsum, ifeval, mathvista]:
        for budget in [0, 64, 256, 1024, 2048, 4096, 8192]:
            results = run_inference(benchmark, budget)
            save_results(model, benchmark, budget, results)
    stop_vllm_server()
```

Outer loop by model to avoid reloading weights. Each model loads once, all benchmarks and budgets run, then next model.

### 2.8 Analysis Plan

For each (model, benchmark) pair, plot the accuracy-reasoning curve Acc(t), where t is the reasoning token budget. Characterize curves as:

- **Monotonically increasing**: Task benefits from more reasoning at all budget levels
- **Inverted-U / saturating**: Task has an optimal reasoning budget beyond which performance plateaus or declines
- **Flat**: Task does not benefit from reasoning tokens
- **Declining**: Additional reasoning actively hurts performance

Cross-model analysis: Compare curve shapes across model sizes to address RQ3 (does model size influence reasoning utility?).

Report bootstrap 95% confidence intervals for all accuracy estimates. With 500 samples at ~50% accuracy, the 95% CI is approximately +/- 4.4 percentage points.

---

## 3. Timeline (20 Days)

| Days | Phase | Work |
|------|-------|------|
| 1–2 | Setup | Rent H100, install vLLM, download models + datasets, test end-to-end inference |
| 3–5 | Core code | Build client, budget forcing, all data loaders, all eval metrics. Test each benchmark with 5 examples on 0.8B |
| 6–8 | Run: small models | Full sweep on Qwen3.5-0.8B and Qwen3.5-4B (all 7 benchmarks x 7 budgets) |
| 9–11 | Run: medium model | Full sweep on Qwen3.5-9B. Begin preliminary analysis of small model results |
| 12–14 | Run: large model | Full sweep on Qwen3.5-27B (FP8). Slowest phase (~20–30 hrs GPU time) |
| 15–17 | Analysis | Generate all accuracy-vs-reasoning curves, heatmaps, identify patterns per RQ |
| 18–19 | Write-up | Literature review section, methodology section, results section with figures |
| 20 | Buffer | Fix issues, re-run failed experiments, polish |

### Compute Estimate

- ~3,500 examples x 7 budgets x 4 models = ~98,000 inference calls
- Small models (0.8B, 4B): ~1–2 sec/call → ~12–24 hrs total
- 9B: ~3–4 sec/call → ~24 hrs
- 27B FP8: ~5–8 sec/call → ~48 hrs
- **Total: ~4–5 days of continuous GPU time**

---

## 4. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| 27B too slow or OOM | Drop to 9B as largest, or use Qwen3.5-35B-A3B (MoE, only 3B active) |
| Budget forcing breaks output quality | Test early on day 3. Fall back to recording natural token counts |
| XSum/summarization metrics are noisy | Use both ROUGE and BERTScore. Acknowledge ROUGE limitations |
| Code eval needs sandboxing | Use evalplus package |
| Some models score ~0 on hard benchmarks | This is itself a finding (small models cannot leverage reasoning) |
| Data contamination | Within-model comparison (varying budget) controls for contamination since all conditions share the same training data |
