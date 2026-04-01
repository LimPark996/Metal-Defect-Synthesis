# Metal Defect Synthesis

> 공장에서 불량품 사진이 부족할 때, AI로 불량 사진을 만들어주는 프로젝트

[![Hugging Face Spaces](https://img.shields.io/badge/HuggingFace-Spaces-blue)](https://huggingface.co/spaces/Yumi-Park996/metal-defect-synthesis)
[![Hugging Face Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/Yumi-Park996/metal-defect-checkpoints)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U0dUqou_aCxwloasepwBF5jtioOjeMju?usp=sharing)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 왜 만들었나

제조 현장에서 AI로 불량을 검출하려면 **불량 사진이 많아야** 합니다.
그런데 실제 공장에서 불량은 드물게 발생합니다. 불량 데이터가 부족하면 AI 학습이 어렵습니다.

이 프로젝트는 **AI가 직접 불량 사진을 생성**하여 학습 데이터를 보강하는 시스템입니다.

---

## 어떻게 동작하나

```
사진 → [VQGAN] → 토큰(코드) → [MaskGIT] → 토큰 채우기 → [VQGAN] → 합성된 사진
```

1. **VQGAN (토크나이저)**: 사진을 256개의 작은 코드로 압축합니다
   - 비유: 사진을 256조각 퍼즐로 만드는 것
2. **MaskGIT (생성기)**: 빈 퍼즐 조각을 AI가 채워넣습니다
   - 비유: "스크래치 불량" 스타일로 빈칸을 채우는 것
3. **인페인팅**: 정상 사진의 일부를 결함으로 바꿔치기합니다

---

## 기술 스택

| 기술 | 설명 | 역할 |
|------|------|------|
| **LlamaGen VQGAN** | 8차원 코드북 기반 이미지 토크나이저 | 사진 ↔ 토큰 변환 |
| **Halton-MaskGIT** | 저불일치 시퀀스 기반 마스크 생성 (ICLR 2025) | 토큰 → 이미지 생성 |
| **Classifier-Free Guidance** | 조건부/무조건부 생성을 혼합 | 생성 품질 향상 |
| **Adaptive GAN Loss** | 복원 loss와 GAN loss 균형 자동 조절 | VQGAN 학습 안정화 |

---

## 데이터셋

| 데이터셋 | 이미지 수 | 출처 |
|----------|-----------|------|
| NEU-DET | 1,440장 | [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |
| SD-saliency-900 | 900장 | [Kaggle](https://www.kaggle.com/datasets/alex000kim/sdsaliency900) |
| X-SDD | 319장 | [Kaggle](https://www.kaggle.com/datasets/sayelabualigah/x-sdd) |
| **합계** | **2,659장** | 8배 증강 → **21,272 샘플** |

### 결함 6종

| 클래스 | 한글명 | 특징 |
|--------|--------|------|
| crazing | 균열 | 거미줄 형태의 미세 균열 |
| inclusion | 개재물 | 표면에 박힌 이물질 |
| patches | 패치 | 불규칙한 얼룩 |
| pitted_surface | 피팅 | 움푹 파인 구멍 |
| rolled-in_scale | 압연 스케일 | 압연 중 생긴 줄무늬 |
| scratches | 스크래치 | 긁힌 선 자국 |

---

## 모델 사양

### VQGAN (토크나이저)
- 코드북: 16,384 토큰, 8차원 임베딩
- 다운샘플링: 16배 (256x256 → 16x16 = 256 토큰)
- Fine-tuning 결과: **Edge IoU +10.6% 개선**

### MaskGIT (생성기)
- 파라미터: ~69M (Small 설정)
- 구조: 12 레이어, 8 헤드, 512 히든
- 특징: AdaLayerNorm, SwiGLU FFN, QK Norm, Weight Tying

---

## 프로젝트 구조

```
Metal-Defect-Synthesis/
├── V0/                              # PoC 노트북 (Colab 실행용)
│   ├── metal_defect_synthesis(PoCFinal).ipynb        # VQGAN Fine-tuning
│   ├── metal_defect_HaltonMaskGIT(PoCFinal).ipynb    # MaskGIT 학습
│   ├── metal_defect_gradio_demo_LlamaGen_Halton(PoCFinal).ipynb  # 데모
│   └── Metal_Defect_Synthesis_PRD_v2_0.pdf           # 기술 명세서
│
├── src/metal_defect_synthesis/      # 모듈화된 Python 패키지
│   ├── models/                      # 모델 아키텍처
│   │   ├── layers.py                # 트랜스포머 빌딩 블록
│   │   ├── maskgit.py               # MaskGIT Transformer
│   │   └── vqgan_wrapper.py         # VQGAN 래퍼
│   ├── data/                        # 데이터 처리
│   │   ├── dataset.py               # 데이터셋 클래스
│   │   ├── augmentation.py          # 8배 데이터 증강
│   │   └── token_cache.py           # 토큰 캐시 생성
│   ├── training/                    # 학습 모듈
│   │   ├── vqgan_trainer.py         # VQGAN 학습 로직
│   │   ├── maskgit_trainer.py       # MaskGIT 학습 로직
│   │   └── scheduler.py            # 학습률 스케줄러
│   ├── sampling/                    # 생성/추론
│   │   ├── halton.py                # Halton 시퀀스
│   │   ├── sampler.py               # 이미지 생성기
│   │   └── inpainting.py            # 결함 합성
│   └── utils/                       # 유틸리티
│       ├── image.py                 # 이미지 변환
│       ├── seed.py                  # 시드 고정
│       └── metrics.py               # 평가 지표
│
├── app/gradio_demo.py               # Gradio 데모 UI
├── scripts/                         # CLI 실행 스크립트
│   ├── train_vqgan.py
│   ├── train_maskgit.py
│   └── generate.py
├── configs/                         # YAML 설정 파일
│   ├── vqgan.yaml
│   ├── maskgit.yaml
│   └── inference.yaml
└── docs/                            # 문서
    └── portfolio_narrative.md       # 포트폴리오 서사
```

---

## 실행 방법

### Google Colab에서 바로 실행 (권장)

| 단계 | 노트북 | 소요 시간 |
|------|--------|----------|
| 1 | [VQGAN Fine-tuning](https://colab.research.google.com/drive/1U0dUqou_aCxwloasepwBF5jtioOjeMju?usp=sharing) | ~2시간 |
| 2 | [MaskGIT 학습](https://colab.research.google.com/drive/1utaTLpAMD-OXp56mapU7EfrxusC67JUN?usp=sharing) | ~2시간 |
| 3 | [Gradio 데모](https://colab.research.google.com/drive/1sbftaG4L7rvDh2ZVA7U3EWxtThpxzGZU?usp=sharing) | ~10분 |

### 로컬 설치

```bash
git clone https://github.com/LimPark996/Metal-Defect-Synthesis.git
cd Metal-Defect-Synthesis
pip install -r requirements.txt
```

### CLI 사용

```bash
# 설정 확인
python scripts/train_vqgan.py --config configs/vqgan.yaml
python scripts/train_maskgit.py --config configs/maskgit.yaml

# 이미지 생성
python scripts/generate.py --class scratches --num 10
```

---

## 현재 상태 (PoC)

| 항목 | 상태 | 비고 |
|------|------|------|
| VQGAN Fine-tuning | 완료 | Edge IoU +10.6% 개선 |
| MaskGIT 학습 | 수렴 중 | Loss 6.77 (목표: ~4.0) |
| Gradio 데모 | 동작 | 생성 품질 개선 필요 |
| 코드 모듈화 | 완료 | src/ 패키지 구조 |

### 알려진 한계점
- MaskGIT 학습 데이터 부족 (21K 샘플 vs 권장 1M+)
- 클래스 간 차이가 미미한 생성 결과
- 텍스처 일관성 개선 필요

---

## 개선 로드맵

### 단기 (현재 아키텍처 유지)
- [ ] 학습 epochs 증가 (100 → 500+)
- [ ] Weighted sampling으로 클래스 밸런싱
- [ ] 추가 데이터셋 확보 (MVTec AD, GC10-DET)

### 중장기
- [ ] Two-stage 학습 (unconditional → conditional)
- [ ] Stable Diffusion Inpainting 기반 접근법 검토
- [ ] AI Agent Tool로 활용 (CLI → MCP Tool 변환)

---

## 참고 자료

| 자료 | 링크 |
|------|------|
| LlamaGen | [GitHub](https://github.com/FoundationVision/LlamaGen) |
| Halton-MaskGIT (ICLR 2025) | [GitHub](https://github.com/valeoai/Halton-MaskGIT) |
| MaskGIT 논문 | [arXiv](https://arxiv.org/abs/2202.04200) |
| PRD v2.0 | [PDF](./V0/Metal_Defect_Synthesis_PRD_v2_0.pdf) |

---

## 버전 이력

| 버전 | 일자 | 변경 내용 |
|------|------|----------|
| v2.1 | 2026-04-01 | 코드 모듈화, 설정 관리, README 재작성 |
| v2.0 | 2024-12-12 | LlamaGen VQGAN + Halton-MaskGIT 전환 |
| v1.0 | 2024-12-11 | 초안 작성 (taming VQGAN + 직접 구현 MaskGIT) |
