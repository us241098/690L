# Project Plan: To Think or Not to Think — Understanding Task-Specific Reasoning in LLMs

**Authors:** Junyan Li, Utsav Shukla
**Course:** COMPSCI 690L, UMass Amherst
**Date:** April 2026

---

## 1. Literature Review

### 1.1 Chain-of-Thought and Thinking Modes

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Wei et al. — *Chain-of-Thought Prompting Elicits Reasoning in LLMs* | 2022 | 2201.11903 | Foundational CoT paper. Few-shot CoT demonstrations dramatically improve reasoning at scale. Benefits emerge only for large models. |
| Kojima et al. — *Large Language Models are Zero-Shot Reasoners* | 2022 | 2205.11916 | "Let's think step by step" elicits reasoning zero-shot across arithmetic, symbolic, and commonsense tasks. |
| Wang et al. — *Self-Consistency Improves Chain of Thought Reasoning* | 2023 | 2203.11171 | Sampling diverse reasoning paths and majority voting yields +17.9% on GSM8K over greedy CoT. |
| Wang & Zhou — *Chain-of-Thought Reasoning Without Prompting* | 2024 | 2402.10200 | LLMs already contain latent CoT paths in top-k decoded outputs, providing mechanistic insight into why thinking budgets work. |
| OpenAI — *Learning to Reason with LLMs* | 2024 | — | o1 trained via large-scale RL to use internal CoT. 89th percentile Codeforces, top-500 AIME. |
| DeepSeek-AI — *DeepSeek-R1* | 2025 | 2501.12948 | Pure RL (no SFT) incentivizes reasoning. Self-reflection and verification emerge naturally. AIME 2024 pass@1: 71.0%. |
| Qwen Team — *Qwen3 Technical Report* | 2025 | 2505.09388 | Hybrid thinking/non-thinking in single model. `<think>` tags, `enable_thinking` parameter, `thinking_budget` control. |
| Kimi Team — *Kimi K1.5* | 2025 | 2501.12599 | RL-based long-CoT training. Long2short technique transfers reasoning from long-CoT to short-CoT models — directly relevant to budget forcing. |
| Microsoft — *Phi-4-reasoning Technical Report* | 2025 | 2504.21318 | Small (14B) reasoning model trained via SFT + RL (DPO/GRPO). Demonstrates strong reasoning at small scale, relevant to model-size scaling analysis. |
| Meincke et al. — *Decreasing Value of Chain of Thought in Prompting* | 2025 | 2506.07142 | CoT provides diminishing returns as models become more capable. Modern reasoning models already think step-by-step. |

### 1.2 Overthinking and When Reasoning Hurts

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Aggarwal et al. — *OptimalThinkingBench* | 2025 | 2508.13141 | Independently optimizing for overthinking or underthinking improves one at the expense of the other. |
| Su et al. — *Between Underthinking and Overthinking* | 2025 | 2505.00127 | Inverted-U relationship: accuracy rises then declines with reasoning length. LLMs overthink simple problems. |
| Hassid et al. — *Don't Overthink It* | 2025 | 2505.17813 | Shorter reasoning chains are up to 34.5% more likely to yield correct answers than the longest chain. |
| Yang et al. — *Mirage of Test-Time Scaling* | 2025 | 2506.04210 | Extended thinking increases output variance. Parallel thinking beats serial extension by up to 20%. |
| Zhang et al. — *Structural Understanding of LLM Overthinking* | 2025 | 2510.07880 | Two overthinking patterns: "Explorer" (excessive exploration) and "Late Landing" (excessive verification). |
| Wang et al. — *NoWait: Removing Thinking Tokens Improves Efficiency* | 2025 | 2506.08343 | Suppressing self-reflection tokens reduces CoT length by 27–51% without compromising accuracy. |
| Sui et al. — *Stop Overthinking: A Survey* | 2025 | 2503.16419 | First comprehensive survey on efficient reasoning in LLMs. |

### 1.3 Budget Forcing and Reasoning Control

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Muennighoff et al. — *s1: Simple Test-Time Scaling* | 2025 | 2501.19393 | Seminal budget forcing paper. Truncate thinking or extend by appending "Wait" tokens. **Methodological foundation for our work.** |
| Aggarwal & Welleck — *L1: Controlling Reasoning Length with RL* | 2025 | 2503.04697 | RL-based length control (LCPO). Smoother than s1's hard truncation. |
| Li et al. — *Steering LLM Thinking with Budget Guidance* | 2025 | 2506.13752 | Fine-tuning-free Gamma distribution over remaining thinking length. |
| Han et al. — *Token-Budget-Aware LLM Reasoning (TALE)* | 2025 | 2412.18547 | Dynamically adjusts reasoning tokens based on problem complexity. 67% token cost reduction. |
| Wen et al. — *BudgetThinker: Budget-aware Reasoning with Control Tokens* | 2025 | 2508.17196 | Inserts control tokens during inference to inform model of remaining budget. |
| SelfBudgeter | 2025 | 2505.11274 | Trains model to self-estimate required reasoning budget per query. 61% length compression. |

### 1.4 Test-Time Compute Scaling Laws

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Snell et al. — *Scaling LLM Test-Time Compute* | 2024 | 2408.03314 | Landmark paper: inference compute can be more efficient than model scaling. Adaptive per-prompt allocation outperforms uniform. |
| Bansal et al. — *Compute-Optimal Inference for Problem-Solving* | 2024 | 2408.00724 | Optimal inference strategy depends heavily on problem difficulty. Supports task-specific thesis. |
| Yang et al. — *Thinking-Optimal Scaling (TOPS)* | 2025 | 2502.18080 | Optimal reasoning length differs across domains. |
| Yeo et al. — *Demystifying Long Chain-of-Thought Reasoning* | 2025 | 2502.03373 | Reasoning emerges with training compute but not guaranteed. |
| Art of Scaling Test-Time Compute | 2025 | 2512.02008 | First large-scale TTS study. No single TTS strategy universally dominates. |

### 1.5 Task-Specific Reasoning Differences

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Marjanovic et al. — *DeepSeek-R1 Thoughtology* | 2025 | 2504.07128 | "Sweet spot" of reasoning length per problem type. **Closest existing work** but only single model. |
| Zhao et al. — *TTS Not Effective for Knowledge-Intensive Tasks* | 2025 | 2509.06861 | Reasoning scaling doesn't help knowledge tasks; can increase hallucinations. |
| Liu et al. — *DiffAdapt* | 2025 | 2510.19669 | U-shaped entropy pattern: 82.3% of problems benefit from "Easy" strategy. |
| Alomrani et al. — *Reasoning on a Budget: A Survey* | 2025 | 2507.02076 | L1-controllability vs. L2-adaptiveness taxonomy. |
| Mirzadeh et al. (Apple) — *GSM-Symbolic* | 2024 | 2410.05229 | LLM math performance drops up to 65% on symbolic GSM8K variations — important critical framing. |
| Kambhampati — *Can LLMs Really Reason and Plan?* | 2024 | 2405.13515 | Argues LLM reasoning is "approximate retrieval" — skeptical framing for our findings. |

### 1.6 Reasoning Evaluation, Verification, and Self-Correction

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Lightman et al. — *Let's Verify Step by Step* | 2023 | 2305.20050 | Process reward models (PRMs) on step-level annotations. Process supervision beats outcome supervision for math reasoning. |
| Wang et al. — *Math-Shepherd* | 2024 | 2312.08935 | Automatic process reward model training without human annotations via Monte Carlo. |
| Madaan et al. — *Self-Refine* | 2023 | 2303.17651 | Iterative self-refinement: 5–40% improvement across tasks. Within-trace self-correction analog of extended thinking. |
| Shinn et al. — *Reflexion* | 2023 | 2303.11366 | Verbal self-reflection as RL-like signal. 91% pass@1 on HumanEval. |
| Suzgun et al. — *BIG-Bench Hard (BBH)* | 2023 | 2210.09261 | Curated 23 challenging BIG-Bench tasks. CoT enables surpassing average human-rater. |
| Rein et al. — *GPQA* | 2023 | 2311.12022 | Graduate-level Google-proof multiple choice. Standard for evaluating expert reasoning. |
| Xu et al. — *LLaVA-CoT* | 2024 | 2411.10440 | Structured stage-wise reasoning for VLMs. Significant gains on MathVista — relevant for our multimodal evaluation. |

### 1.7 Agentic Workflows and Compound AI Systems (motivation for RQ4)

| Paper | Year | ID | Key Takeaway |
|-------|------|----|--------------|
| Wu et al. — *AutoGen* | 2023 | 2308.08155 | Multi-agent conversation framework. Different agents can use different models and reasoning depths. |
| Fourney et al. — *Magentic-One* | 2024 | 2411.04468 | Lead Orchestrator + 4 specialized worker agents. Orchestrator needs deep reasoning; workers more execution-focused. |
| Anthropic — *Building Effective Agents* (blog) | 2024 | — | 6 workflow patterns. Explicitly recommends routing easy queries to small models, hard queries to capable models. |
| OpenAI — *A Practical Guide to Building Agents* | 2025 | — | Recommends tuning `reasoning_effort` per node: reasoning models for complex tasks, fast models for simple ones. |
| Zaharia et al. — *Shift from Models to Compound AI Systems* (BAIR) | 2024 | — | Argues SOTA results increasingly come from compound systems. Motivates per-node optimization. |
| Khattab et al. — *DSPy* | 2023 | 2310.03714 | Compiles declarative LLM pipelines. Can assign different prompting strategies and models per module. |
| Chen et al. — *LLMSelector* | 2025 | 2502.14815 | **Most relevant.** Per-module model selection in compound systems yields 5–70% accuracy gains. Directly parallels our per-node budget allocation idea. |
| Ong et al. — *RouteLLM* | 2024 | 2406.18665 | Routes queries between strong/weak LLMs. 2x+ cost reduction. |
| Chen et al. — *FrugalGPT* | 2023 | 2305.05176 | LLM cascades: try cheap, escalate if uncertain. Up to 98% cost reduction. |
| Ding et al. — *Hybrid LLM* | 2024 | 2404.14618 | Routes queries by predicted difficulty. 40% reduction in large-model calls. |
| Su et al. — *DAAO: Difficulty-Aware Agentic Orchestration* | 2025 | 2509.11079 | VAE for difficulty estimation + operator allocator + LLM router. 11% accuracy improvement, 36% cost reduction. |
| Zhang et al. — *AdaptThink* | 2025 | 2505.13417 | RL-trained model adaptively selects Thinking vs NoThinking. 53% length reduction, 2.4% accuracy improvement. |
| Lou et al. — *AdaCoT* | 2025 | 2505.11896 | PPO-trained CoT triggering. 3.18% trigger rate, 69% token reduction. |
| Damani et al. — *Learning How Hard to Think* | 2024 | 2410.04707 | Predicts which inputs benefit from more compute. 50% compute reduction at maintained quality. |
| Yao et al. — *ReAct* | 2022 | 2210.03629 | Interleaves reasoning and action. Different step types may need different reasoning depths. |
| Yao et al. — *Tree of Thoughts* | 2023 | 2305.10601 | Branching reasoning paths. Generation vs. evaluation nodes have different reasoning needs. |
| Paglieri et al. — *Learning When to Plan* | 2025 | 2509.03581 | Agents learn when to allocate test-time compute for planning. Always-planning is costly; never-planning limits capability. |
| Wang et al. — *Mixture-of-Agents (MoA)* | 2024 | 2406.04692 | Layered MoA: 65.1% on AlpacaEval 2.0. Different layers may benefit from different budgets. |
| Kim et al. — *Cost of Dynamic Reasoning* (HPCA 2026) | 2025 | 2506.04301 | First system-level analysis of agent resource usage. Per-query energy orders of magnitude higher than single-turn. |
| Wang et al. — *Efficient Agents* | 2025 | 2508.02694 | Retains 96.7% of SOTA on GAIA at 42.7% lower cost via complexity reduction. |

### 1.8 Benchmark Source Papers

| Paper | Year | ID | Used For |
|-------|------|----|----------|
| Cobbe et al. — *Training Verifiers to Solve Math Word Problems* | 2021 | 2110.14168 | GSM8K |
| Hendrycks et al. — *Measuring Mathematical Problem Solving (MATH)* | 2021 | 2103.03874 | MATH-500 |
| Wang et al. — *MMLU-Pro* | 2024 | 2406.01574 | Knowledge QA |
| Chen et al. — *Evaluating Large Language Models Trained on Code* | 2021 | 2107.03374 | HumanEval |
| Zhou et al. — *Instruction-Following Evaluation (IFEval)* | 2023 | 2311.07911 | Instruction following |
| Lu et al. — *MathVista* | 2024 | 2310.02255 | Vision + Math |

### 1.9 Gap and Contribution

While the literature establishes that (a) reasoning is not universally beneficial, (b) budget forcing can control reasoning compute, (c) different tasks may have different optimal reasoning strategies, and (d) compound AI systems would benefit from per-node optimization, **no existing work has performed a systematic cross-task analysis at varying reasoning budgets using models of multiple sizes from the same family, with explicit translation to workflow-level allocation guidance.** Thoughtology (Marjanovic et al.) studies only DeepSeek-R1 (single model, no budget forcing). LLMSelector (Chen et al.) does per-node *model* selection but not budget allocation. Our work fills this gap by combining all three axes (task type × model size × reasoning budget) and producing concrete workflow design guidance.

---

## 2. Experimental Design

### 2.1 Models

| Model | Precision | VRAM (est.) | Rationale |
|-------|-----------|-------------|-----------|
| Qwen3.5-0.8B | FP16 | ~2 GB | Smallest — test if reasoning helps tiny models at all |
| Qwen3.5-4B | FP16 | ~8 GB | Small-medium capacity |
| Qwen3.5-9B | FP16 | ~18 GB | Medium capacity |
| Qwen3.5-27B | FP8 | ~32 GB | Largest that fits on single H100 |

All Qwen3.5 models support multimodal inputs (vision built-in) and implement thinking via `<think>...</think>` tags.

### 2.2 Benchmarks

| Category | Benchmark | Source | Subset Size | Metric | Original Paper |
|----------|-----------|--------|-------------|--------|----------------|
| Math (easy) | GSM8K | `openai/gsm8k` | 500 | Exact match | Cobbe et al. 2021 |
| Math (hard) | MATH-500 | `HuggingFaceH4/MATH-500` | 500 | Exact match | Hendrycks et al. 2021 |
| Knowledge QA | MMLU-Pro | `TIGER-Lab/MMLU-Pro` | 500 (stratified) | MC accuracy | Wang et al. 2024 |
| Code | HumanEval+ | `evalplus` package | 164 (all) | pass@1 | Chen et al. 2021 |
| Summarization | XSum | `EdinburghNLP/xsum` | 500 | ROUGE + BERTScore | Narayan et al. 2018 |
| Instruction Following | IFEval | `google/IFEval` | ~500 (all) | Strict/loose accuracy | Zhou et al. 2023 |
| Vision + Math | MathVista | `AI4Math/MathVista` | 500 (testmini) | Accuracy | Lu et al. 2024 |

### 2.3 Reasoning Budget Levels

```
budgets = [0, 64, 256, 1024, 2048, 4096, 8192]
```

- **Budget = 0**: Thinking disabled (`enable_thinking=False`)
- **Budget = 64–8192**: Thinking enabled, truncated via budget forcing

### 2.4 Budget Forcing Implementation

LogitsProcessor-based (Muennighoff et al. 2025):

1. Enable thinking mode (`enable_thinking=True`)
2. Track token count inside `<think>...</think>`
3. At `budget − 1` tokens: suppress all tokens except newline
4. At `budget` tokens: force `</think>` token (ID 248069 for Qwen3.5)
5. Continue generation normally for the final answer

### 2.5 Inference Stack

```bash
vllm serve Qwen/Qwen3.5-{size} --port 8000 \
  --quantization fp8 \           # for 27B only
  --reasoning-parser qwen3 \
  --max-model-len 32768
```

**Sampling parameters (thinking mode):** `temperature=0.6, top_p=0.95, top_k=20`
**Sampling parameters (no thinking):** `temperature=0.7, top_p=0.8, top_k=20`

Avoid greedy decoding with thinking mode (causes infinite loops).

### 2.6 Mapping to Research Questions

Each experimental output explicitly maps to one or more RQs:

| RQ | What it asks | How our experiments answer it |
|----|--------------|-------------------------------|
| **RQ1**: Is reasoning always beneficial? | Whether thinking universally helps | Compare budget=0 vs. budget>0 across all 7 benchmarks. Identify benchmarks where Acc(0) ≥ max_t Acc(t) |
| **RQ2**: How does Acc(t) vary across task types? | Curve shape per task | Plot full Acc(t) curves for all 7 benchmarks. Categorize each as monotonic / inverted-U / flat / declining |
| **RQ3**: Does model size influence reasoning utility? | Does scale change the curve | Compare curves across the 4 model sizes for each task. Compute curve "slope" and "knee" per (task, model) |
| **RQ4**: Implications for workflow allocation? | Practical guidance | Five concrete deliverables (see §3) translating curves into workflow design recommendations |

### 2.7 Evaluation Methodology

- **GSM8K, MATH-500**: Extract final answer, exact match
- **MMLU-Pro**: Multiple-choice accuracy
- **HumanEval+**: Sandboxed execution via `evalplus`, pass@1
- **XSum**: ROUGE-1/2/L + BERTScore
- **IFEval**: Programmatic constraint checker, strict/loose accuracy
- **MathVista**: Official evaluation script

### 2.8 Curve Analysis

For each (model, benchmark) pair we plot Acc(t) and characterize as:

- **Monotonically increasing**: Always benefits from more reasoning
- **Inverted-U / saturating**: Optimal budget exists; beyond it accuracy plateaus or declines
- **Flat**: Reasoning provides no benefit
- **Declining**: Additional reasoning actively hurts

We report bootstrap 95% confidence intervals. With 500 samples at ~50% accuracy, CI is approximately ±4.4 points.

---

## 3. RQ4 Analysis: Workflow Implications (NEW)

This section operationalizes RQ4 with five concrete deliverables that translate our empirical curves into actionable workflow design guidance.

### Deliverable 1: Reasoning Budget Recommendation Table

For each (task category, model size) pair we will produce a recommendation table containing:

- **Recommended thinking budget** (token count or thinking mode: off/low/medium/high)
- **Accuracy at recommended budget** with 95% CI
- **Marginal accuracy loss** if budget is halved or doubled
- **Token cost** at recommended budget

This is directly usable by practitioners building Qwen3.5-based workflows. It fills the gap noted by industry agent guides (Anthropic, OpenAI) that recommend per-node routing without quantitative thresholds.

### Deliverable 2: Simulated Multi-Node Workflow Case Study

We will design 2–3 representative workflows and simulate end-to-end performance under different allocation strategies:

- **Workflow A (RAG pipeline)**: query classification → retrieval → synthesis → verification
- **Workflow B (coding agent)**: issue understanding → code localization → patch generation → test reasoning
- **Workflow C (research assistant)**: query decomposition → multi-hop QA → summarization

Allocation strategies compared:
1. **Uniform**: Same budget for all nodes (baseline)
2. **Curve-informed**: Use our task-type curves to assign optimal per-node budgets
3. **Cost-constrained Pareto**: Given fixed total token budget, derive Pareto-optimal allocation

We expect curve-informed allocation to match uniform accuracy at lower cost, paralleling the LLMSelector methodology (Chen et al. 2025) but for reasoning budgets rather than model selection.

### Deliverable 3: Cost-Performance Pareto Frontier Analysis

For each task type and model size we plot the Pareto frontier of accuracy vs. reasoning token cost, then compute:

- **Cost savings at iso-accuracy** — how much cheaper is curve-informed allocation?
- **Accuracy gains at iso-cost** — how much better at the same total budget?
- **Break-even points** — where does reasoning pay for itself?

This connects directly to the AdaCoT (Lou et al. 2025) and FrugalGPT (Chen et al. 2023) framings.

### Deliverable 4: Decision Heuristics for Workflow Designers

We will derive simple, actionable rules from the curves, such as:

- "For classification on Qwen3.5-{size}, disable thinking entirely (Acc(0) ≈ Acc(8192))"
- "For mathematical reasoning, allocate ≥ N tokens; below this accuracy drops by X%"
- "Crossover point: a smaller model with large budget matches a larger model with small budget for task T at budget B"

Output format: a decision tree / flowchart mapping (task type, model size, cost constraint) → recommended budget. This addresses the qualitative gap in Anthropic's "Building Effective Agents" guide.

### Deliverable 5: Reasoning Elasticity and Per-Task Scaling Laws

We will define **reasoning elasticity** as the percentage change in accuracy per percentage change in reasoning token budget, and fit functional forms (logarithmic, sigmoid) to the curves per (task type, model size). This enables:

- **Extrapolation** to budgets we did not directly test
- **Comparison** across model sizes (does elasticity change with scale?)
- **Predictive budgeting** for new task types based on characteristics

This extends the test-time compute scaling laws (Snell et al. 2024) by providing task-type-stratified relationships rather than aggregate ones.

---

## 4. Codebase Architecture

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
│   │   └── budget_forcing.py  # LogitsProcessor for budget control
│   ├── data/
│   │   ├── loader.py          # Unified dataset loader
│   │   ├── gsm8k.py / math500.py / mmlu_pro.py / humaneval.py
│   │   ├── xsum.py / ifeval.py / mathvista.py
│   ├── eval/
│   │   ├── metrics.py         # Exact match, ROUGE, BERTScore, pass@1
│   │   ├── judge_ifeval.py    # IFEval constraint checker
│   │   └── judge_code.py      # Sandbox code execution (evalplus)
│   ├── analysis/              # RQ1–RQ3
│   │   ├── curves.py          # Plot Acc(t) curves
│   │   ├── heatmap.py         # Model x task x budget heatmaps
│   │   └── stats.py           # Bootstrap CIs
│   └── workflow/              # RQ4 deliverables
│       ├── recommend.py       # Build recommendation table (D1)
│       ├── simulate.py        # Multi-node workflow simulator (D2)
│       ├── pareto.py          # Pareto frontier analysis (D3)
│       ├── heuristics.py      # Decision rule extraction (D4)
│       └── elasticity.py      # Scaling law fitting (D5)
├── scripts/
│   ├── run_experiment.py      # Main loop: model x budget x benchmark
│   └── run_all.sh             # Full sweep
├── results/                   # Raw outputs (thinking + answer) per (model, benchmark, budget)
└── figures/
```

### Experiment Loop

```python
for model in [0.8B, 4B, 9B, 27B]:
    start_vllm_server(model)
    for benchmark in [gsm8k, math500, mmlu_pro, humaneval, xsum, ifeval, mathvista]:
        for budget in [0, 64, 256, 1024, 2048, 4096, 8192]:
            results = run_inference(benchmark, budget)
            save_results(model, benchmark, budget, results)
    stop_vllm_server()
```

---

## 5. Timeline (20 Days)

| Days | Phase | Work |
|------|-------|------|
| 1–2 | Setup | Provision H100, install vLLM, download models + datasets, end-to-end smoke test |
| 3–5 | Core code | Client, budget forcing, all data loaders, all eval metrics |
| 6–8 | Run: small models | Full sweep on 0.8B and 4B |
| 9–11 | Run: medium model | Full sweep on 9B. Begin RQ1–RQ3 analysis on small-model results |
| 12–14 | Run: large model | Full sweep on 27B (FP8) |
| 15–16 | Analysis (RQ1–RQ3) | Curves, heatmaps, characterize per-task curve shapes |
| 17–18 | Analysis (RQ4) | Recommendation table, workflow simulations, Pareto analysis, heuristics, elasticity |
| 19 | Write-up | Lit review, methodology, results, RQ4 implications |
| 20 | Buffer | Re-run failed experiments, polish |

### Compute Estimate

- ~3,500 examples × 7 budgets × 4 models = ~98,000 inference calls
- Estimated total GPU time: 4–5 days continuous on a single H100

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| 27B too slow / OOM | Drop to 9B, or use Qwen3.5-35B-A3B (MoE, only 3B active) |
| Budget forcing degrades quality | Test on day 3. Fall back to recording natural token counts |
| Summarization metrics noisy | Use ROUGE + BERTScore. Acknowledge limitations |
| Code eval needs sandboxing | Use `evalplus` package |
| Small models score ~0 on hard benchmarks | Itself a finding (small models can't leverage reasoning) |
| Data contamination | Within-model comparison controls for it |
| RQ4 simulations not "real" workflows | Frame as design exercise; emphasize the recommendation table is the primary RQ4 output |
