# Metal Defect Synthesis

> AI-powered synthetic defect image generation for industrial quality control

[![Hugging Face Spaces](https://img.shields.io/badge/HuggingFace-Spaces-blue)](https://huggingface.co/spaces/Yumi-Park996/metal-defect-synthesis)
[![Hugging Face Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/Yumi-Park996/metal-defect-checkpoints)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U0dUqou_aCxwloasepwBF5jtioOjeMju?usp=sharing)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Motivation

Training AI defect detectors in manufacturing requires a large volume of defect images — but real production lines generate defects only rarely. This data scarcity makes it difficult to build reliable inspection models.

This project addresses that gap by using **generative AI to synthesize defect images**, augmenting training datasets where real examples are limited.

---

## How It Works

```
Image → [VQGAN] → Tokens → [MaskGIT] → Filled Tokens → [VQGAN] → Synthesized Image
```

1. **VQGAN (Tokenizer)** — Compresses an image into 256 discrete tokens (like breaking a photo into a 256-piece puzzle)
2. **MaskGIT (Generator)** — Fills in masked tokens conditioned on a defect class (e.g., "scratches"), learning the visual style of each defect type
3. **Inpainting** — Replaces a region of a normal image with the generated defect patch

---

## Tech Stack

| Component | Description | Role |
|-----------|-------------|------|
| **LlamaGen VQGAN** | Image tokenizer with 8-dim codebook | Image ↔ Token conversion |
| **Halton-MaskGIT** | Low-discrepancy mask scheduling (ICLR 2025) | Token → Image generation |
| **Classifier-Free Guidance** | Blends conditional / unconditional outputs | Generation quality |
| **Adaptive GAN Loss** | Auto-balances reconstruction vs. GAN loss | VQGAN training stability |

---

## Dataset

| Dataset | Images | Source |
|---------|--------|--------|
| NEU-DET | 1,440 | [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |
| SD-saliency-900 | 900 | [Kaggle](https://www.kaggle.com/datasets/alex000kim/sdsaliency900) |
| X-SDD | 319 | [Kaggle](https://www.kaggle.com/datasets/sayelabualigah/x-sdd) |
| **Total** | **2,659** | 8× augmentation → **21,272 samples** |

### Defect Classes (6)

| Class | Description |
|-------|-------------|
| crazing | Fine web-like surface cracks |
| inclusion | Foreign particles embedded in the surface |
| patches | Irregular blotchy regions |
| pitted_surface | Small pits or craters |
| rolled-in_scale | Linear streaks from the rolling process |
| scratches | Linear scratch marks |

---

## Model Specs

### VQGAN (Tokenizer)
- Codebook: 16,384 tokens, 8-dim embeddings
- Downsampling: 16× (256×256 → 16×16 = 256 tokens)
- Fine-tuning result: **Edge IoU +10.6%**

### MaskGIT (Generator)
- Parameters: ~69M (Small config)
- Architecture: 12 layers, 8 heads, 512 hidden dim
- Features: AdaLayerNorm, SwiGLU FFN, QK Norm, Weight Tying

---

## Project Structure

```
Metal-Defect-Synthesis/
├── V0/                              # PoC notebooks (Colab-ready)
│   ├── metal_defect_synthesis(PoCFinal).ipynb        # VQGAN fine-tuning
│   ├── metal_defect_HaltonMaskGIT(PoCFinal).ipynb    # MaskGIT training
│   ├── metal_defect_gradio_demo_LlamaGen_Halton(PoCFinal).ipynb  # Demo
│   └── Metal_Defect_Synthesis_PRD_v2_0.pdf           # Technical spec
│
├── src/metal_defect_synthesis/      # Modularized Python package
│   ├── models/                      # Model architectures
│   │   ├── layers.py                # Transformer building blocks
│   │   ├── maskgit.py               # MaskGIT Transformer
│   │   └── vqgan_wrapper.py         # VQGAN wrapper
│   ├── data/                        # Data pipeline
│   │   ├── dataset.py               # Dataset class
│   │   ├── augmentation.py          # 8× data augmentation
│   │   └── token_cache.py           # Token cache builder
│   ├── training/                    # Training modules
│   │   ├── vqgan_trainer.py         # VQGAN training logic
│   │   ├── maskgit_trainer.py       # MaskGIT training logic
│   │   └── scheduler.py             # LR scheduler
│   ├── sampling/                    # Inference
│   │   ├── halton.py                # Halton sequence
│   │   ├── sampler.py               # Image sampler
│   │   └── inpainting.py            # Defect synthesis
│   └── utils/                       # Utilities
│       ├── image.py                 # Image transforms
│       ├── seed.py                  # Seed management
│       └── metrics.py               # Evaluation metrics
│
├── app/gradio_demo.py               # Gradio demo UI
├── scripts/                         # CLI entry points
│   ├── train_vqgan.py
│   ├── train_maskgit.py
│   └── generate.py
├── configs/                         # YAML configs
│   ├── vqgan.yaml
│   ├── maskgit.yaml
│   └── inference.yaml
└── docs/
    └── portfolio_narrative.md
```

---

## Getting Started

### Run on Google Colab (Recommended)

| Step | Notebook | Est. Time |
|------|----------|-----------|
| 1 | [VQGAN Fine-tuning](https://colab.research.google.com/drive/1U0dUqou_aCxwloasepwBF5jtioOjeMju?usp=sharing) | ~2 hrs |
| 2 | [MaskGIT Training](https://colab.research.google.com/drive/1utaTLpAMD-OXp56mapU7EfrxusC67JUN?usp=sharing) | ~2 hrs |
| 3 | [Gradio Demo](https://colab.research.google.com/drive/1sbftaG4L7rvDh2ZVA7U3EWxtThpxzGZU?usp=sharing) | ~10 min |

### Local Installation

```bash
git clone https://github.com/LimPark996/Metal-Defect-Synthesis.git
cd Metal-Defect-Synthesis
pip install -r requirements.txt
```

### CLI Usage

```bash
# Training
python scripts/train_vqgan.py --config configs/vqgan.yaml
python scripts/train_maskgit.py --config configs/maskgit.yaml

# Image generation
python scripts/generate.py --class scratches --num 10
```

---

## Current Status (PoC)

| Component | Status | Notes |
|-----------|--------|-------|
| VQGAN Fine-tuning | ✅ Done | Edge IoU +10.6% |
| MaskGIT Training | 🔄 Converging | Loss 6.77 (target: ~4.0) |
| Gradio Demo | ✅ Live | Generation quality needs improvement |
| Code Modularization | ✅ Done | `src/` package structure |

### Known Limitations
- Insufficient MaskGIT training data (21K samples vs. recommended 1M+)
- Limited inter-class visual variation in generated outputs
- Texture consistency needs improvement

---

## Roadmap

### Near-term (current architecture)
- [ ] Increase training epochs (100 → 500+)
- [ ] Weighted sampling for class balancing
- [ ] Additional datasets (MVTec AD, GC10-DET)

### Mid/Long-term
- [ ] Two-stage training (unconditional → conditional)
- [ ] Explore Stable Diffusion Inpainting approach
- [ ] Expose as AI Agent tool via MCP

---

## References

| Resource | Link |
|----------|------|
| LlamaGen | [GitHub](https://github.com/FoundationVision/LlamaGen) |
| Halton-MaskGIT (ICLR 2025) | [GitHub](https://github.com/valeoai/Halton-MaskGIT) |
| MaskGIT Paper | [arXiv](https://arxiv.org/abs/2202.04200) |
| PRD v2.0 | [PDF](./V0/Metal_Defect_Synthesis_PRD_v2_0.pdf) |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v2.1 | 2026-04-01 | Code modularization, config management, README rewrite |
| v2.0 | 2024-12-12 | Migrated to LlamaGen VQGAN + Halton-MaskGIT |
| v1.0 | 2024-12-11 | Initial draft (taming VQGAN + custom MaskGIT) |
