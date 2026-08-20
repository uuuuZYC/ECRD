<h1 align="center">See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs</h1>
 
<p align="center">
  <a href="https://arxiv.org/abs/2602.21497"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
  <a href="https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_See_It_Say_It_Sorted_An_Iterative_Training-Free_Framework_for_CVPR_2026_paper.html"><img src="https://img.shields.io/badge/CVPR-2026-blue" alt="CVPR 2026"></a>
  <a href="https://github.com/uuuuZYC/ECRD"><img src="https://img.shields.io/badge/GitHub-Code-black" alt="GitHub Code"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

<p align="center">
  <b>Yongchang Zhang</b><sup>1,3,*</sup>&emsp;
  <b>Oliver Ma</b><sup>2,*</sup>&emsp;
  <b>Tianyi Liu</b><sup>1,†</sup>&emsp;
  <b>Guangquan Zhou</b><sup>1,†</sup>&emsp;
  <b>Yang Chen</b><sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>Southeast University&emsp;
  <sup>2</sup>University of Oxford&emsp;
  <sup>3</sup>AIIA, Ministry of Education, China
</p>

<p align="center">
  * Equal contribution.&emsp;† Corresponding author.
</p>

<p align="center">
  <b>CVPR 2026</b>
</p>

## News

- **2026-06**: *See It, Say It, Sorted* is published in the CVPR 2026 proceedings.
- **2026-02-25**: The paper is released on arXiv.

## ECRD Overview

<p align="center">
  <img src="assets/fig2_ecrd_overview.png" alt="ECRD overview" width="98%">
</p>

### Abstract

Large vision-language models (LVLMs) can solve complex multimodal tasks with long chain-of-thought reasoning, yet their reasoning can gradually drift away from the visual evidence. Once an intermediate step becomes visually inconsistent, the error may propagate through an otherwise fluent reasoning trace and lead to an incorrect final answer.

We introduce **Evidence-Constrained Reweighting Decoding (ECRD)**, a lightweight, training-free, plug-and-play framework for visually grounded multimodal reasoning. ECRD maintains a textual visual-evidence pool and uses it to supervise token selection during decoding. When the current evidence is insufficient to resolve an uncertain step, a lightweight visual decider extracts an additional piece of image-grounded micro-evidence and appends it to the pool. This iterative process keeps reasoning aligned with the image without SFT, RL, reward-model training, or modification of the LVLM parameters.

Across multiple LVLM backbones and benchmarks, ECRD substantially reduces hallucination while improving reasoning accuracy. The paper reports **16.5%–29.5% improvements on TreeBench** and a **13.7% RH-AUC gain on RH-Bench**, all without additional training.

### Motivation

<p align="center">
  <img src="assets/fig1_reasoning_pattern.png" alt="Reasoning pattern comparison" width="78%">
</p>

Conventional greedy decoding can amplify a single visually unsupported token into a wrong reasoning trajectory. ECRD instead introduces decoding-time visual supervision: existing evidence first reweights plausible next-token candidates, and the model consults the image again only when the evidence pool is insufficient.

## Highlights

- **Training-free:** no SFT, RL, reward model, or parameter update.
- **Plug-and-play:** integrates with HuggingFace generation as a standard `LogitsProcessor`.
- **Evidence-constrained decoding:** candidate tokens are reweighted using textual visual evidence at every reasoning step.
- **Dynamic evidence pool:** newly extracted micro-evidence is cached and reused by later decoding steps.
- **Adaptive visual consultation:** the visual decider is invoked only when uncertainty remains after evidence-based reweighting.
- **Backbone-friendly:** the implementation supports Qwen2.5-VL and can be adapted to LLaVA-OneVision and InternVL-style LVLMs by changing only model-specific visual/chat preprocessing.

## Main Results

ECRD improves visually grounded reasoning without additional model training.

| Benchmark | Reported improvement |
| --- | ---: |
| **TreeBench** | **+16.5% to +29.5%** |
| **RH-Bench** | **+13.7% RH-AUC** |

The key result is not only higher final-answer accuracy, but also a reduction in **reasoning-perception drift**: the generated reasoning trace remains better aligned with the visual evidence as decoding proceeds.

## Usage

### Install

```bash
git clone https://github.com/uuuuZYC/ECRD.git
cd ECRD

pip install -e .
```

For the Qwen2.5-VL / GRIT visual-decider example, install the additional dependencies:

```bash
pip install -r requirements.txt
```

> `qwen-vl-utils` is required only for the Qwen2.5-VL demo and `GRITClient`. The core ECRD supervisor (`EvidenceScorer` + `ECRDLogitsProcessor`) depends only on the standard project dependencies defined in `pyproject.toml`.

### Minimal API Usage

```python
from transformers import LogitsProcessorList
from ecrd import (
    Evidence,
    EvidenceScorer,
    ECRDLogitsProcessor,
    MixedGapTrigger,
    GRITClient,
)

# 1. Initialize the evidence pool with a global visual description.
scorer = EvidenceScorer(
    model=model,
    tokenizer=tokenizer,
    max_prefix_len=128,
)
scorer.add_evidence(
    Evidence(
        id="global-0",
        text=global_description,
        source="global",
        time_step=0,
    )
)

# 2. Attach ECRD as a HuggingFace logits processor.
ecrd = ECRDLogitsProcessor(
    scorer=scorer,
    tokenizer=tokenizer,
    min_k=1,
    max_k=64,
)

# 3. Optional: use GRIT as an uncertainty-triggered visual decider.
grit = GRITClient(model_id="yfan1997/GRIT-20-Qwen2.5-VL-3B")

def grit_hook(image, question, prefix_text, candidates):
    return grit.decide_next_token(
        image=image,
        question=question,
        prefix_text=prefix_text,
        candidates=candidates,
        max_new_tokens=64,
    )

ecrd.set_grit_runtime(
    hook=grit_hook,
    trigger=MixedGapTrigger(
        gap_thresh=0.08,
        min_k=2,
        cooldown=5,
    ),
    evidence_pool=scorer,
    question=question,
    image=image_uri,
)

# 4. Generate with ECRD.
outputs = model.generate(
    **inputs,
    logits_processor=LogitsProcessorList([ecrd]),
    do_sample=False,
    max_new_tokens=512,
)
```

### Qwen2.5-VL Quick Demo

```bash
python examples/qwen2_5_vl_ecrd_demo.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --image /path/to/image.jpg \
  --question "What is shown in the image?" \
  --use-grit
```

For LLaVA-OneVision or InternVL-style models, the ECRD objects remain unchanged. Replace only the backbone-specific image and chat preprocessing before `model.generate(...)`.

## Method

ECRD consists of three lightweight decoding-time components.

### 1. Visual Evidence Pool

ECRD starts from short textual descriptions of image evidence. Instead of repeatedly re-processing all visual information at every token, these evidence sentences are stored in a reusable pool and provide persistent visual constraints throughout the reasoning process.

### 2. Distribution Supervisor

At each decoding step, ECRD:

1. computes the LVLM's original next-token distribution;
2. selects a compact candidate set using knee-based top-*k* selection;
3. scores candidate tokens against the current evidence pool;
4. reweights the candidate distribution using evidence-induced preferences; and
5. checks whether the resulting distribution is sufficiently confident.

This preserves the base model's language distribution while steering ambiguous decisions toward candidates that are better supported by the image evidence.

### 3. Visual Decider

If evidence-constrained reweighting still leaves the model uncertain, ECRD invokes a lightweight visual decider. The decider receives the image, current reasoning context, and candidate set, then returns:

- a visually grounded token decision for the current step; and
- a short micro-evidence sentence.

The new evidence is appended to the pool and can supervise later reasoning steps, allowing ECRD to **see again only when necessary**.

## Repository Layout

```text
.
├── ecrd/
│   ├── evidence.py            # Evidence representation and pool
│   ├── scorer.py              # Textual evidence scorer
│   ├── logits_processor.py    # ECRD distribution supervisor
│   ├── triggers.py            # Uncertainty trigger
│   ├── grit_client.py         # Optional GRIT visual decider
│   └── prompts.py             # Visual-evidence prompts
├── examples/
│   └── qwen2_5_vl_ecrd_demo.py
├── assets/
│   ├── fig1_reasoning_pattern.png
│   └── fig2_ecrd_overview.png
├── requirements.txt
├── pyproject.toml
├── CITATION.bib
└── LICENSE
```

<details>
<summary><b>Advanced configuration</b></summary>

The implementation supports the following optional environment variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `ECRD_MIN_K` | `1` | Minimum knee candidate-set size |
| `ECRD_MAX_K` | `64` | Maximum knee candidate-set size |
| `ECRD_LOGMIX_ALPHA` | `0.6` | Base mixing coefficient before adaptive override |
| `ECRD_ALPHA_ADAPT` | `1` | Use adaptive `alpha = p_top` when enabled |
| `ECRD_MIX_REWEIGHT` | `1` | Mass-match the evidence distribution to base candidate mass |
| `ECRD_DEBUG` | `0` | Print per-step base/evidence/mixed token diagnostics |
| `ECRD_DEBUG_TOP` | `10` | Number of tokens shown in debug output |

</details>

## Notes

- ECRD is a **decoding-time** method and does not modify LVLM parameters.
- The optional `GRITClient` is implemented for Qwen2.5-VL-style visual-decider usage; the distribution supervisor itself is a generic Transformers `LogitsProcessor`.
- This repository focuses on the core ECRD implementation and intentionally omits project-specific experimental utilities, cached runs, and dataset-conversion code.

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Zhang_2026_CVPR,
  author    = {Zhang, Yongchang and Ma, Oliver and Liu, Tianyi and Zhou, Guangquan and Chen, Yang},
  title     = {See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {11933--11942}
}
```

## License

This project is released under the [MIT License](LICENSE).
