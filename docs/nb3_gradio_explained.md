# 노트북 3: Gradio 데모 상세 설명

이 문서는 `V0/metal_defect_gradio_demo_LlamaGen_Halton(PoCFinal).ipynb` 노트북과 이에 대응하는 `.py` 모듈(`inpainting.py`, `gradio_demo.py`, `image.py`)의 **모든 코드**를 한 줄 한 줄 상세히 설명한다.

---

## 1. 상수와 설정 (셀 9)

### 디바이스 설정

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

- **변수 `device`**: 문자열 `"cuda"` 또는 `"cpu"`. GPU가 사용 가능하면 CUDA를 선택한다. 이후 모든 텐서와 모델이 이 디바이스 위에 올라간다.

### 이미지/모델 상수

| 상수 | 값 | 타입 | 의미 |
|------|-----|------|------|
| `IMAGE_SIZE` | 256 | int | 입력 이미지의 가로/세로 픽셀 수 |
| `LATENT_SIZE` | 16 | int | VQGAN이 256x256 이미지를 16x16 잠재 격자로 압축. 256/16 = 16이므로 한 토큰이 16x16 픽셀 영역을 대표 |
| `CODEBOOK_SIZE` | 16384 | int | VQGAN 코드북의 벡터 개수. 토큰 ID는 0~16383 범위 |
| `CODEBOOK_DIM` | 8 | int | 코드북 벡터 하나의 차원. LlamaGen은 8차원 (기존 taming-transformers는 256차원) |
| `SEQ_LEN` | 256 | int | 16x16 = 256, 한 이미지를 나타내는 토큰 시퀀스 길이 |
| `NUM_CLASSES` | 6 | int | 결함 분류 클래스 수 |
| `MASK_TOKEN_ID` | 16384 | int | `[MASK]` 토큰의 ID. 코드북 크기(16384)와 동일한 값이므로, 유효한 코드북 토큰(0~16383)과 겹치지 않는다 |

### CLASS_NAMES vs CLASS_NAMES_KR 차이

```python
CLASS_NAMES = ['inclusion', 'pit_hole', 'scratch_crease', 'spot_stain', 'line_crack', 'fold_scale']
```

- **`CLASS_NAMES`**: 영문 결함명 리스트, 길이 6. 모델 내부에서 인덱스로 변환할 때나 로그 메시지에서 사용한다.

```python
CLASS_NAMES_KR = [
    '0: 개재물 (Inclusion)',
    '1: 피트홀 (Pit Hole)',
    ...
]
```

- **`CLASS_NAMES_KR`**: 한글+인덱스 번호 포함된 표시용 리스트. Gradio 드롭다운에 보여줄 때 사용한다. `CLASS_NAMES_KR.index(selected)` 호출로 사용자가 선택한 항목의 인덱스를 바로 추출할 수 있도록 `0:`, `1:` 등의 접두사를 붙여 놓았다.

### get_mask_preset 함수

```python
def get_mask_preset(preset_name):
```

- **입력**: `preset_name` (str) - 프리셋 이름 (노트북에서는 한글 키, `.py`에서는 영문 키)
- **출력**: `List[int]` - 마스킹할 토큰 인덱스 리스트

**인덱스 계산 원리**: 16x16 격자에서 `(y, x)` 좌표의 1차원 인덱스는 `y * 16 + x`이다.

#### center_small (중앙 작음 6x6)

```python
for y in range(5, 11):   # y = 5, 6, 7, 8, 9, 10 → 6행
    for x in range(5, 11):   # x = 5, 6, 7, 8, 9, 10 → 6열
        presets["center_small"].append(y * 16 + x)
```

- y=5, x=5일 때: `5*16+5 = 85`
- y=10, x=10일 때: `10*16+10 = 170`
- 총 6x6 = **36개** 토큰이 마스킹된다.
- 16x16 격자의 정중앙 부근(행 5~10, 열 5~10)에 위치한다.

#### center_large (중앙 큼 8x8)

```python
for y in range(4, 12):   # y = 4~11 → 8행
    for x in range(4, 12):   # x = 4~11 → 8열
```

- 총 8x8 = **64개** 토큰.
- y=4, x=4 → `4*16+4 = 68`, y=11, x=11 → `11*16+11 = 187`

#### top_left (좌상단 6x6)

```python
for y in range(0, 6):    # y = 0~5
    for x in range(0, 6):    # x = 0~5
```

- 총 **36개** 토큰. 인덱스 범위: `0*16+0=0` ~ `5*16+5=85`

#### bottom_right (우하단 6x6)

```python
for y in range(10, 16):   # y = 10~15
    for x in range(10, 16):   # x = 10~15
```

- 총 **36개** 토큰. 인덱스 범위: `10*16+10=170` ~ `15*16+15=255`

---

## 2. 모델 로딩 (셀 14~16)

### VQGAN 로딩 (셀 14)

```python
from tokenizer.tokenizer_image.vq_model import VQ_models
vqgan = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
```

- LlamaGen 프레임워크의 `VQ-16` 모델 아키텍처를 인스턴스화한다.
- `codebook_size=16384`: 16384개 코드 벡터를 가진 코드북 생성
- `codebook_embed_dim=8`: 각 코드 벡터의 차원은 8

```python
checkpoint = torch.load(VQGAN_CKPT_PATH, map_location=device)
```

- 저장된 체크포인트 파일(.pt)을 로드한다. `map_location=device`로 GPU/CPU 자동 매핑.

```python
if "vqmodel" in checkpoint:
    state_dict = checkpoint["vqmodel"]
else:
    state_dict = checkpoint.get("state_dict", checkpoint)
```

- 체크포인트 구조가 `{"vqmodel": ...}` 형태이면 해당 키에서, 아니면 `"state_dict"` 키에서, 또는 체크포인트 전체를 state_dict로 사용한다.

```python
vqgan.load_state_dict(state_dict, strict=True)
vqgan = vqgan.to(device)
vqgan.eval()
for param in vqgan.parameters():
    param.requires_grad = False
```

- `strict=True`: 모든 파라미터가 정확히 일치해야 한다.
- `eval()`: 배치정규화, 드롭아웃 등을 추론 모드로 설정.
- `requires_grad = False`: VQGAN의 가중치는 고정(freeze)하여 학습하지 않는다.

### MaskGIT 로딩 (셀 15~16)

```python
ckpt = torch.load(MASKGIT_CKPT_PATH, map_location=device, weights_only=False)
model_config = ckpt['config'].copy()
```

- 체크포인트에서 `config` 딕셔너리를 꺼낸다. 학습 시 저장한 하이퍼파라미터(vocab_size, hidden_dim, num_layers 등)가 들어 있다.

```python
keys_to_remove = ['model_size']
for key in keys_to_remove:
    model_config.pop(key, None)
```

- `model_size` 같은 `MaskGITTransformer`의 `__init__`이 받지 않는 키를 제거한다.

```python
maskgit = MaskGITTransformer(**model_config).to(device)
maskgit.load_state_dict(ckpt['model_state_dict'])
maskgit.eval()
```

- config를 키워드 인자로 풀어서 모델 생성 후, 학습된 가중치를 로드하고 추론 모드로 전환한다.

**노트북에서 모델 클래스를 다시 정의하는 이유**: Colab 환경에서는 별도의 패키지 설치 없이 단일 노트북에서 모든 코드를 실행해야 하므로, `MaskGITTransformer` 클래스와 그 구성 블록(RMSNorm, SwiGLU, Attention 등)을 셀 11~12에서 직접 정의한다. `.py` 모듈 구조에서는 `src/metal_defect_synthesis/models/maskgit.py`에 별도 파일로 분리되어 있다.

---

## 3. 인코딩/디코딩 함수 (셀 18)

### encode_to_tokens

```python
@torch.no_grad()
def encode_to_tokens(images, vqgan_model):
    _, _, (_, _, indices) = vqgan_model.encode(images)
    batch_size = images.shape[0]
    tokens = indices.view(batch_size, LATENT_SIZE, LATENT_SIZE)
    return tokens
```

**줄별 설명:**

1. `@torch.no_grad()`: 그래디언트 계산을 비활성화하여 메모리를 절약한다.
2. `vqgan_model.encode(images)`: VQGAN 인코더에 이미지를 넣는다. 반환값은 `(z_q, loss, (perplexity, min_encodings, indices))` 형태.
   - `z_q`: 양자화된 잠재 벡터 `[B, 8, 16, 16]`
   - `indices`: 코드북 인덱스 `[B*16*16]` = `[B*256]`, 각 값은 0~16383
3. `indices.view(batch_size, LATENT_SIZE, LATENT_SIZE)`: 1차원 인덱스를 `[B, 16, 16]` 격자로 재구성.

- **입력**: `images` - `torch.Tensor`, 형태 `[B, 3, 256, 256]`, 값 범위 `[-1, 1]`
- **출력**: `tokens` - `torch.Tensor`, 형태 `[B, 16, 16]`, 각 원소는 0~16383의 정수

**구체적 예시**: 배치 크기 1인 256x256 금속 이미지를 넣으면, `[1, 16, 16]` 텐서가 나온다. 예를 들어 `tokens[0, 3, 7] = 5012`이면, 격자 위치 `(3, 7)`의 16x16 픽셀 패치가 코드북의 5012번 벡터로 표현된다는 뜻이다.

### decode_from_tokens

```python
@torch.no_grad()
def decode_from_tokens(tokens, vqgan_model):
    batch_size = tokens.shape[0]
    tokens_flat = tokens.view(-1)
    tokens_flat = torch.clamp(tokens_flat, 0, CODEBOOK_SIZE - 1)
    embed_dim = vqgan_model.quantize.embedding.weight.shape[1]  # 8
    images = vqgan_model.decode_code(
        tokens_flat,
        shape=(batch_size, embed_dim, LATENT_SIZE, LATENT_SIZE)
    )
    return images
```

**줄별 설명:**

1. `tokens.view(-1)`: `[B, 16, 16]` → `[B*256]`으로 평탄화.
2. `torch.clamp(tokens_flat, 0, CODEBOOK_SIZE - 1)`: 값이 0~16383 범위를 벗어나지 않도록 클램핑. 특히 `[MASK]`(16384)가 남아 있을 경우 16383으로 처리.
3. `embed_dim = vqgan_model.quantize.embedding.weight.shape[1]`: 코드북 임베딩 차원(8)을 동적으로 읽는다.
4. `vqgan_model.decode_code(...)`: 토큰 인덱스를 코드북에서 룩업하여 `[B, 8, 16, 16]` 잠재 텐서를 만들고, 디코더를 통해 `[B, 3, 256, 256]` 이미지로 복원.

- **입력**: `tokens` - `[B, 16, 16]`, 정수 텐서
- **출력**: `images` - `[B, 3, 256, 256]`, 값 범위 `[-1, 1]`

---

## 4. 인페인팅 함수 (셀 19) -- 가장 중요!

### cosine_schedule 함수

```python
def cosine_schedule(t):
    return torch.cos(t * torch.pi / 2)
```

- **입력**: `t` - `torch.Tensor` 스칼라, 범위 `[0, 1]`. 현재 디코딩 진행률.
- **출력**: `torch.Tensor` 스칼라, 범위 `[0, 1]`.

**동작 원리:**
- `t = 0`일 때: `cos(0) = 1.0` → 마스킹된 토큰 100%를 유지 (아직 아무것도 확정하지 않음)
- `t = 0.5`일 때: `cos(pi/4) ≈ 0.707` → 약 70%를 유지
- `t = 1`일 때: `cos(pi/2) = 0.0` → 마스킹된 토큰 0%를 유지 (모두 확정)

**왜 코사인 스케줄인가?** 초기에는 천천히 확정하고(모델이 문맥 정보가 적으므로 신중히), 후반으로 갈수록 빠르게 확정한다. 코사인 곡선은 `t=0` 근처에서 기울기가 완만하고 `t=1` 근처에서 급격히 감소하므로, 초반에 소수의 "가장 확신 높은" 토큰만 확정하고 후반에 대량으로 확정하는 효과를 준다. 이렇게 하면 초반의 확정된 토큰이 나머지 토큰 예측의 문맥으로 활용되어 전체 품질이 향상된다.

### inpaint_image 함수 전체 알고리즘

```python
@torch.no_grad()
def inpaint_image(original_image, mask_region, target_class,
                  maskgit_model, vqgan_model, num_steps=8,
                  temperature=1.0, device='cuda'):
```

**파라미터 상세:**

| 파라미터 | 타입 | 의미 |
|----------|------|------|
| `original_image` | `torch.Tensor [1, 3, 256, 256]` | 원본 금속 표면 이미지, 값 `[-1, 1]` |
| `mask_region` | `List[int]` 또는 `torch.Tensor` | 마스킹할 토큰의 1D 인덱스 (0~255) |
| `target_class` | `int` (0~5) | 어떤 결함으로 채울지 |
| `maskgit_model` | `MaskGITTransformer` | 학습된 MaskGIT 모델 |
| `vqgan_model` | VQGAN 모델 | 인코더/디코더 |
| `num_steps` | `int`, 기본 8 | 반복 디코딩 스텝 수 |
| `temperature` | `float`, 기본 1.0 | 샘플링 온도. 높으면 다양성 증가, 낮으면 안정성 증가 |
| `device` | `str` | `"cuda"` 또는 `"cpu"` |

#### Step 1: 원본 이미지를 토큰으로 인코딩

```python
if original_image.dim() == 3:
    original_image = original_image.unsqueeze(0)
original_image = original_image.to(device)
```

- 3차원(`[3, 256, 256]`)이면 배치 차원을 추가하여 `[1, 3, 256, 256]`으로 만든다.

```python
mask_token_id = maskgit_model.mask_token_id   # 16384
seq_len = maskgit_model.seq_len                # 256
```

- 모델에서 `[MASK]` ID와 시퀀스 길이를 가져온다.

```python
if isinstance(mask_region, list):
    mask_region = torch.tensor(mask_region, device=device)
num_mask_tokens = len(mask_region)
```

- `mask_region`이 파이썬 리스트이면 텐서로 변환.
- `num_mask_tokens`: 마스킹할 토큰 수 (center_small이면 36).

```python
original_tokens = encode_to_tokens(original_image, vqgan_model)
```

- `original_tokens`: `[1, 16, 16]` 정수 텐서. 원본 이미지의 완전한 토큰 맵.

#### Step 2: 마스킹할 영역의 토큰을 [MASK]로 교체

```python
tokens = original_tokens.clone().view(1, -1)     # [1, 256]
tokens[:, mask_region] = mask_token_id            # 마스크 영역만 16384로
```

- `clone()`으로 원본을 보존하고, `view(1, -1)`로 `[1, 256]`으로 평탄화한다.
- `mask_region`에 해당하는 위치만 `[MASK]`(16384)로 교체. 나머지 위치는 원본 토큰 그대로 유지된다.

**예시**: center_small 마스크라면, 인덱스 85, 86, ..., 170 등 36곳이 16384로 바뀐다. 나머지 220개 위치는 원본 값(0~16383)이 그대로 남는다.

#### Step 3: 클래스 라벨 준비

```python
labels = torch.tensor([target_class], device=device)      # [1]
drop_label = torch.zeros(1, dtype=torch.bool, device=device)  # [1], False
```

- `labels`: 결함 클래스 (예: `[3]`이면 얼룩).
- `drop_label`: `False`이면 클래스 조건을 사용. `True`이면 무조건부(CFG용). 인페인팅에서는 항상 `False`.

#### Step 4: 반복 디코딩 루프

```python
for step in range(num_steps):    # 기본 8스텝
```

각 스텝에서 수행하는 작업:

**(a) 현재 마스킹된 위치 찾기:**

```python
is_masked = (tokens == mask_token_id)   # [1, 256] bool 텐서
num_masked = is_masked.sum().item()      # 남은 마스크 수 (int)
if num_masked == 0:
    break
```

**(b) 코사인 스케줄로 이번 스텝에서 확정할 개수 계산:**

```python
t = torch.tensor(step / num_steps, device=device)
ratio = cosine_schedule(t)                           # 남겨야 할 비율
num_to_keep_masked = (ratio * num_mask_tokens).long().item()
num_to_unmask = max(1, num_masked - num_to_keep_masked)
```

- `t`: 현재 진행률. step=0이면 `t=0`, step=4(8스텝 중)이면 `t=0.5`.
- `ratio`: 코사인 스케줄 값. `t=0`이면 1.0, `t=0.5`이면 약 0.707.
- `num_to_keep_masked`: 이번 스텝 후 마스킹 상태로 남겨야 할 토큰 수. 예: `0.707 * 36 ≈ 25`.
- `num_to_unmask`: 이번 스텝에서 확정할 토큰 수. 예: `36 - 25 = 11`. 최소 1개.

**구체적 계산 예시 (center_small, num_steps=8, num_mask_tokens=36):**

| step | t | cos(t*pi/2) | keep_masked | unmask |
|------|------|-------------|-------------|--------|
| 0 | 0.000 | 1.000 | 36 | 1 (min 보정) |
| 1 | 0.125 | 0.981 | 35 | 1 |
| 2 | 0.250 | 0.924 | 33 | 3 |
| 3 | 0.375 | 0.831 | 29 | 4 |
| 4 | 0.500 | 0.707 | 25 | 4 |
| 5 | 0.625 | 0.556 | 20 | 5 |
| 6 | 0.750 | 0.383 | 13 | 7 |
| 7 | 0.875 | 0.195 | 7 | 6 |

초반에는 1~3개씩 천천히, 후반에는 5~7개씩 빠르게 확정한다.

**(c) MaskGIT 모델로 예측:**

```python
logits = maskgit_model(tokens, labels, drop_label)   # [1, 256, 16385]
logits = logits / temperature                         # temperature scaling
probs = F.softmax(logits, dim=-1)                     # [1, 256, 16385] 확률
max_probs, predicted_tokens = probs.max(dim=-1)       # 각 위치의 최대 확률과 예측 토큰
```

- `logits`: 각 위치(256개)에 대해 16385개 토큰(16384 코드북 + 1 마스크)의 로짓.
- `temperature`로 나누면: 값이 1보다 작을 때 분포가 더 뾰족해지고(확신 있는 선택), 1보다 클 때 분포가 평평해진다(다양한 선택).
- `max_probs`: `[1, 256]`, 각 위치에서 가장 높은 확률값.
- `predicted_tokens`: `[1, 256]`, 각 위치에서 가장 확률이 높은 토큰 ID.

**(d) 마스킹된 위치만 고려하여 상위 k개 확정:**

```python
max_probs = max_probs * is_masked.float()
```

- 마스킹되지 않은 위치의 확률을 0으로 만든다. 이미 확정된 토큰은 건드리지 않겠다는 뜻.

```python
_, top_k = max_probs.view(-1).topk(num_to_unmask)
tokens.view(-1)[top_k] = predicted_tokens.view(-1)[top_k]
```

- `topk(num_to_unmask)`: 마스킹된 위치 중 모델이 가장 확신하는 상위 `num_to_unmask`개의 인덱스를 선택.
- 해당 위치의 `[MASK]` 토큰을 예측된 토큰으로 교체하여 확정한다.

#### Step 5: 남은 [MASK] 강제 채움

```python
is_masked = (tokens == mask_token_id)
if is_masked.any():
    logits = maskgit_model(tokens, labels, drop_label)
    probs = F.softmax(logits / temperature, dim=-1)
    _, predicted_tokens = probs.max(dim=-1)
    tokens[is_masked] = predicted_tokens[is_masked]
```

- 반복 루프가 끝난 후에도 `[MASK]`가 남아 있을 수 있다(코사인 스케줄의 반올림 오차 등). 이 경우 마지막으로 한 번 더 예측하여 **모든** 남은 마스크를 강제로 채운다.

#### Step 6: 토큰을 이미지로 디코딩

```python
tokens = tokens.view(1, LATENT_SIZE, LATENT_SIZE)   # [1, 256] → [1, 16, 16]
result_image = decode_from_tokens(tokens, vqgan_model)  # [1, 3, 256, 256]
return result_image
```

- 확정된 토큰 맵을 VQGAN 디코더에 넣어 최종 이미지를 복원한다.

---

## 5. Halton 시퀀스와 샘플러 (셀 20~21)

### halton_sequence 알고리즘

```python
def halton_sequence(base, n_samples):
    result = []
    for i in range(1, n_samples + 1):
        f = 1.0
        r = 0.0
        n = i
        while n > 0:
            f /= base
            r += f * (n % base)
            n //= base
        result.append(r)
    return result
```

- **입력**: `base` (int) - 사용할 기저(소수), `n_samples` (int) - 생성할 샘플 수
- **출력**: `List[float]` - `[0, 1)` 범위의 준난수(quasi-random) 시퀀스

**알고리즘 (base=2 예시):**

정수 `i`를 base 진법으로 변환한 뒤 소수점 아래로 뒤집는다.

| i (10진) | i (2진) | 뒤집기 | Halton 값 |
|----------|---------|--------|-----------|
| 1 | 1 | 0.1 | 0.5 |
| 2 | 10 | 0.01 | 0.25 |
| 3 | 11 | 0.11 | 0.75 |
| 4 | 100 | 0.001 | 0.125 |
| 5 | 101 | 0.101 | 0.625 |

코드에서 `f /= base`로 소수점 자릿수를 계산하고, `n % base`로 현재 자릿수의 값을 구하고, `n //= base`로 다음 자릿수로 넘어간다.

**핵심 특성**: 일반 난수(무작위)와 달리, Halton 시퀀스는 공간을 **균일하게** 채운다. 처음 4개 값만 봐도 `[0.5, 0.25, 0.75, 0.125]`으로 `[0, 1)` 구간을 고르게 분할한다.

### build_halton_mask 함수

```python
def build_halton_mask(input_size, n_points=10000):
    x = halton_sequence(2, n_points)    # base 2로 x좌표
    y = halton_sequence(3, n_points)    # base 3으로 y좌표
```

- 서로 다른 소수 기저(2, 3)를 사용하여 x, y 좌표를 독립적으로 생성한다. 같은 기저를 쓰면 대각선에 점이 몰리므로 반드시 다른 소수를 사용해야 한다.

```python
    coords = []
    seen = set()
    for xi, yi in zip(x, y):
        row = int(xi * input_size)      # [0,1) → [0,16) → 정수
        col = int(yi * input_size)
        row = min(row, input_size - 1)  # 경계 보정
        col = min(col, input_size - 1)
        if (row, col) not in seen:      # 중복 제거
            seen.add((row, col))
            coords.append([row, col])
        if len(coords) >= input_size ** 2:   # 256개 모이면 종료
            break
    return torch.tensor(coords, dtype=torch.long)
```

- **출력**: `[256, 2]` 텐서. 각 행은 `(row, col)` 좌표. 16x16 격자의 모든 256개 위치를 Halton 순서대로 나열한 것이다.
- `n_points=10000`으로 충분히 많이 생성하는 이유: `[0,1)` 범위를 16개 구간으로 양자화하면 중복이 많이 발생하므로, 넉넉히 생성해서 중복 제거 후 256개를 채운다.

### HaltonSampler.sample 전체 흐름

이 클래스는 **처음부터 이미지를 생성**(인페인팅이 아님)할 때 사용한다.

```python
code = torch.full((B, H, W), MASK_TOKEN_ID, device=device)   # 전체 [MASK]
```

1. 모든 256개 토큰을 `[MASK]`로 초기화.

2. CFG(Classifier-Free Guidance)를 위해 조건부/무조건부 라벨을 준비.

3. Halton 마스크를 샘플별로 랜덤 오프셋(`torch.roll`)을 적용하여 다양성을 부여.

4. `num_steps`(기본 32) 스텝에 걸쳐 Halton 순서대로 토큰을 채운다:

```python
ratio = (step + 1) / self.num_steps
r = 1 - (math.acos(ratio) / (math.pi * 0.5))
curr_idx = max(step + 1, int(r * seq_len))
```

- `acos` 스케줄: 인페인팅의 `cos` 스케줄과 역관계. 처음에 빠르게, 나중에 느리게 토큰을 채운다.
- `positions = halton_masks[:, prev_idx:curr_idx]`: 이번 스텝에서 채울 위치들.

5. CFG 공식: `logits = (1 + cfg_weight) * logits_cond - cfg_weight * logits_uncond`
   - 조건부 로짓을 강화하고 무조건부 로짓을 빼서, 클래스 특성이 더 강하게 반영되도록 한다.

6. 각 위치에서 `torch.multinomial`로 확률적 샘플링(argmax가 아님).

7. 최종 토큰을 VQGAN 디코더로 이미지 변환.

---

## 6. 이미지 유틸리티 함수 (셀 22)

### preprocess_image 함수

```python
def preprocess_image(pil_image):
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),   # 임의 크기 → 256x256
        T.Grayscale(num_output_channels=3),   # 3채널 그레이스케일로 변환
        T.ToTensor(),                          # PIL → [3, 256, 256], 값 [0, 1]
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [0,1] → [-1,1]
    ])
```

**각 변환 단계:**

1. `T.Resize((256, 256))`: 이미지를 정사각형 256x256으로 리사이즈. 비율이 다르면 왜곡될 수 있다.
2. `T.Grayscale(num_output_channels=3)`: 컬러 이미지를 그레이스케일로 변환하되, 채널을 3개로 유지(R=G=B). 금속 표면 이미지는 대부분 그레이스케일이므로 이렇게 통일한다.
3. `T.ToTensor()`: PIL Image(H, W, C, uint8) → `torch.FloatTensor`(C, H, W), 값 범위 `[0.0, 1.0]`.
4. `T.Normalize([0.5]*3, [0.5]*3)`: `(x - 0.5) / 0.5`로 값 범위를 `[-1.0, 1.0]`으로 정규화. VQGAN이 `[-1, 1]` 범위를 기대한다.

```python
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    tensor = transform(pil_image).unsqueeze(0)   # [3, 256, 256] → [1, 3, 256, 256]
    return tensor.to(device)
```

- **입력**: `pil_image` - `PIL.Image.Image`, 임의 크기/모드
- **출력**: `torch.Tensor`, 형태 `[1, 3, 256, 256]`, 값 범위 `[-1, 1]`, 지정 디바이스

### tensor_to_pil 함수

```python
def tensor_to_pil(tensor):
    tensor = tensor.cpu().detach()
    if tensor.dim() == 4:
        tensor = tensor[0]              # [1, 3, 256, 256] → [3, 256, 256]
    tensor = (tensor + 1) / 2           # [-1, 1] → [0, 1]
    tensor = tensor.clamp(0, 1)         # 안전 클램핑
    numpy_img = tensor.permute(1, 2, 0).numpy()   # [C, H, W] → [H, W, C]
    numpy_img = (numpy_img * 255).astype(np.uint8) # [0, 1] → [0, 255]
    return Image.fromarray(numpy_img)
```

- **역변환 과정**: `[-1, 1]` → `(x+1)/2` → `[0, 1]` → `*255` → `[0, 255]` uint8
- `permute(1, 2, 0)`: PyTorch의 채널-우선(CHW) 형태를 NumPy/PIL의 채널-후방(HWC) 형태로 변환.
- **입력**: `torch.Tensor [1, 3, H, W]` 또는 `[3, H, W]`
- **출력**: `PIL.Image.Image`, RGB, uint8

### visualize_mask_on_image 함수

```python
def visualize_mask_on_image(pil_image, mask_indices, color=(255, 0, 0), alpha=0.4):
```

**파라미터:**
- `pil_image`: 원본 이미지 (PIL)
- `mask_indices`: 마스킹할 토큰 인덱스 리스트
- `color`: 오버레이 색상, 기본값 빨간색 `(255, 0, 0)`
- `alpha`: 투명도, 0.4이면 40% 색상 + 60% 원본

**단계별 설명:**

```python
img_np = np.array(pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32)
```

- PIL 이미지를 256x256으로 리사이즈 후 float32 numpy 배열로 변환. 형태: `[256, 256, 3]`.

```python
mask = np.zeros((LATENT_SIZE, LATENT_SIZE), dtype=np.float32)   # [16, 16] 영행렬
for idx in mask_indices:
    y = idx // LATENT_SIZE    # 행: 인덱스를 16으로 나눈 몫
    x = idx % LATENT_SIZE     # 열: 인덱스를 16으로 나눈 나머지
    mask[y, x] = 1.0
```

- 1D 인덱스를 2D 좌표로 변환하여 16x16 이진 마스크를 생성. 예: 인덱스 85 → y=5, x=5 → `mask[5, 5] = 1.0`.

```python
scale = IMAGE_SIZE // LATENT_SIZE   # 256 // 16 = 16
mask_upscaled = np.kron(mask, np.ones((scale, scale)))   # [16, 16] → [256, 256]
```

- `np.kron` (크로네커 곱): 16x16 마스크의 각 원소를 16x16 블록으로 확장한다. `mask[y, x] = 1.0`이면 해당 16x16 픽셀 영역 전체가 1.0이 된다.
- 결과: `[256, 256]` 이진 마스크.

```python
mask_3d = np.stack([mask_upscaled] * 3, axis=-1)   # [256, 256, 3]
color_overlay = np.array(color, dtype=np.float32).reshape(1, 1, 3)   # [1, 1, 3]
overlay = img_np * (1 - mask_3d * alpha) + color_overlay * mask_3d * alpha
```

- 알파 블렌딩 공식: `결과 = 원본 * (1 - mask * alpha) + 색상 * mask * alpha`
- 마스크가 0인 영역: `결과 = 원본 * 1 + 색상 * 0 = 원본` (변화 없음)
- 마스크가 1인 영역: `결과 = 원본 * 0.6 + 빨간색 * 0.4` (60% 원본 + 40% 빨간색)

---

## 7. Gradio UI (셀 24~27)

### gradio_inpaint 콜백 함수

```python
def gradio_inpaint(image, defect_type, mask_preset, num_steps, temperature):
```

이 함수는 사용자가 "Inpainting 실행" 버튼을 클릭하면 호출된다.

**전체 흐름:**

1. **입력 검증**: `image`가 `None`이면 에러 메시지 반환.

2. **numpy → PIL 변환**:
   ```python
   if isinstance(image, np.ndarray):
       pil_image = Image.fromarray(image)
   ```
   Gradio는 이미지를 numpy 배열로 전달할 수 있으므로 PIL로 변환한다.

3. **클래스 인덱스 추출**:
   ```python
   class_idx = CLASS_NAMES_KR.index(defect_type)
   ```
   예: `"3: 얼룩 (Spot/Stain)"` → `index = 3`

4. **마스크 생성**:
   - 노트북: `get_mask_preset(mask_preset)` - 한글 키를 직접 사용
   - `.py`: `MASK_PRESETS_KR` 딕셔너리로 한글→영문 키 변환 후 호출

5. **전처리**: `preprocess_image(pil_image)` → `[1, 3, 256, 256]` 텐서

6. **VQGAN 재구성** (품질 확인용):
   ```python
   tokens = encode_to_tokens(input_tensor, vqgan)
   reconstructed = decode_from_tokens(tokens, vqgan)
   ```
   원본 → 토큰 → 복원. 이 과정에서 양자화 손실이 발생하므로, 사용자에게 VQGAN 자체의 재구성 품질을 보여준다.

7. **인페인팅 실행**: `inpaint_image(...)` 호출.

8. **결과 변환**: 3개의 텐서를 각각 PIL로 변환하고, 마스크 시각화 이미지를 생성.

9. **반환**: `(mask_preview, reconstructed_pil, inpainted_pil, status)` 4-튜플.

### UI 구성요소

```python
with gr.Blocks(css=css, title="금속 결함 합성기") as demo:
```

- `gr.Blocks`: Gradio의 레이아웃 빌더. CSS로 폰트, 정렬 등을 커스터마이즈.

**입력 위젯 (왼쪽 컬럼, scale=1):**

| 위젯 | 타입 | 역할 |
|------|------|------|
| `input_image` | `gr.Image(type="pil")` | 사용자가 금속 표면 이미지를 업로드. `type="pil"`이면 PIL Image로 전달 |
| `defect_dropdown` | `gr.Dropdown` | 6가지 결함 타입 중 선택. 기본값: `CLASS_NAMES_KR[5]` (접힘/스케일) |
| `mask_dropdown` | `gr.Dropdown` | 4가지 마스크 프리셋 중 선택. 기본값: "중앙 (작음) 6x6" |
| `steps_slider` | `gr.Slider(4~16, 기본 8)` | 디코딩 반복 스텝 수. 많을수록 품질 높지만 느림 |
| `temp_slider` | `gr.Slider(0.5~2.0, 기본 1.0)` | 샘플링 온도. 낮으면 안정적, 높으면 다양함 |
| `inpaint_btn` | `gr.Button(variant="primary")` | 실행 버튼. 클릭 시 `gradio_inpaint` 호출 |

**출력 위젯 (오른쪽 컬럼, scale=2):**

| 위젯 | 역할 |
|------|------|
| `status_text` | 처리 상태 메시지 (완료/에러) 표시 |
| `mask_preview` | 원본 이미지 위에 빨간색으로 마스크 영역을 오버레이한 미리보기 |
| `reconstructed_output` | VQGAN encode→decode 재구성 결과. 인페인팅 없이 양자화 품질만 확인 |
| `inpainted_output` | 최종 인페인팅 결과. 마스크 영역에 지정된 결함이 합성된 이미지 |

**이벤트 바인딩:**

```python
inpaint_btn.click(
    fn=gradio_inpaint,
    inputs=[input_image, defect_dropdown, mask_dropdown, steps_slider, temp_slider],
    outputs=[mask_preview, reconstructed_output, inpainted_output, status_text],
)
```

- 버튼 클릭 → `gradio_inpaint` 함수에 5개 입력 전달 → 4개 출력을 UI에 표시.

**앱 실행:**

```python
demo.launch(share=True, debug=True)
```

- `share=True`: Gradio가 공개 URL(72시간 유효)을 생성하여 외부에서도 접속 가능.
- `debug=True`: 에러 발생 시 상세 트레이스백을 표시.

---

## 8. .py 모듈 대조

### inpainting.py vs 노트북 셀 19

**파일**: `src/metal_defect_synthesis/sampling/inpainting.py`

| 항목 | 노트북 (셀 19) | .py 모듈 |
|------|----------------|----------|
| 함수 시그니처 | `encode_to_tokens`, `decode_from_tokens`를 전역 함수로 직접 호출 | `from ..models.vqgan_wrapper import encode_to_tokens, decode_from_tokens`로 임포트 |
| `cosine_schedule` | 동일한 구현 | 동일한 구현, 타입 힌트 `torch.Tensor` 추가 |
| `inpaint_image` 파라미터 | `device='cuda'` 기본값, `latent_size` 파라미터 없음 (전역 `LATENT_SIZE` 사용) | `latent_size: int = 16` 파라미터 추가, `device: str = 'cuda'` |
| `encode_to_tokens` 호출 | `encode_to_tokens(original_image, vqgan_model)` - 2인자 | `encode_to_tokens(original_image, vqgan_model, latent_size)` - 3인자 |
| `decode_from_tokens` 호출 | `decode_from_tokens(tokens, vqgan_model)` - 2인자 | `decode_from_tokens(tokens, vqgan_model, latent_size)` - 3인자 |
| 타입 힌트 | 없음 | `Union[List[int], torch.Tensor]`, `int`, `float` 등 완전한 타입 힌트 |
| 전역 변수 의존 | `LATENT_SIZE`, `CODEBOOK_SIZE` 등 전역 상수에 의존 | 전역 변수 의존 없음. 모든 값을 파라미터로 받음 |
| `@torch.no_grad()` | 있음 | 있음 |

**알고리즘 로직은 완전히 동일하다.** 차이점은 오직 모듈화(전역 상수 제거, 타입 힌트 추가, 임포트 경로)에 있다.

### gradio_demo.py vs 노트북 셀 24~27

**파일**: `app/gradio_demo.py`

| 항목 | 노트북 (셀 24~25) | .py 모듈 |
|------|-------------------|----------|
| 마스크 프리셋 키 | 한글 키 (`"중앙 (작음) 6x6"`) 직접 사용 | `MASK_PRESETS_KR` 딕셔너리로 한글→영문 매핑 후 `get_mask_preset(영문키)` 호출 |
| `get_mask_preset` 함수 | 셀 9에 정의, 한글 키 사용 | `image.py`에서 임포트, 영문 키 사용 (`"center_small"` 등) |
| `gradio_inpaint` 내 `encode/decode` | 전역 함수 직접 호출 | `from src...vqgan_wrapper import encode_to_tokens, decode_from_tokens`로 임포트 |
| `decode_from_tokens` 호출 | `decode_from_tokens(tokens, vqgan)` - 2인자 | `decode_from_tokens(tokens, vqgan, LATENT_SIZE, CODEBOOK_SIZE)` - 4인자 |
| 에러 메시지 | 이모지 포함 (`"❌ 에러: ..."`) | 이모지 없음 (`"에러: ..."`) |
| 상태 메시지 | 이모지 포함 (`"✅ 완료!"`) | `"완료!"` |
| 모델 로드 | 셀 14~16에서 직접 로드 | `if __name__ == "__main__"` 블록에서 안내 메시지만 출력 (실제 로드 없음) |
| CSS | 동일 | 동일 |
| `demo.launch()` | `share=True, debug=True` | 주석 처리 (`# demo.launch(...)`) |
| 하단 테이블 | 결함별 시각적 특징 설명 테이블과 Tips 포함 | 없음 (간소화) |

**핵심 차이**: 노트북은 Colab에서 바로 실행 가능한 self-contained 코드이고, `.py` 모듈은 패키지 구조에 맞게 임포트 경로를 정리하고 전역 변수 의존성을 제거했다. `.py`는 체크포인트 경로가 환경에 따라 달라지므로 모델 로드를 자동으로 수행하지 않는다.

### image.py vs 노트북 셀 22

**파일**: `src/metal_defect_synthesis/utils/image.py`

| 함수 | 노트북 (셀 22) | .py 모듈 |
|------|----------------|----------|
| `preprocess_image` | `def preprocess_image(pil_image):`, 전역 `IMAGE_SIZE`, `device` 사용 | `def preprocess_image(pil_image, image_size=256, device="cuda"):` 파라미터화 |
| `tensor_to_pil` | 동일한 구현 | 동일한 구현, 타입 힌트 추가 |
| `visualize_mask_on_image` | `(pil_image, mask_indices, color, alpha)`, 전역 `IMAGE_SIZE`, `LATENT_SIZE` 사용 | `(pil_image, mask_indices, image_size=256, latent_size=16, color, alpha)` 파라미터화 |
| `get_mask_preset` | 한글 키 사용 | 영문 키 사용 (`"center_small"` 등) |
| `denormalize` | 없음 | 추가됨. `[-1,1]→[0,1]→numpy` 변환 유틸리티 |
| 타입 힌트 | 없음 | `List[int]`, `Tuple[int, int, int]`, `torch.Tensor` 등 완전한 타입 힌트 |
| 전역 변수 | `IMAGE_SIZE`, `LATENT_SIZE` 직접 참조 | 전역 변수 의존 없음 |

**`denormalize` 함수** (`.py`에만 존재):

```python
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5    # [-1, 1] → [0, 1]
    tensor = tensor.clamp(0, 1)
    if tensor.dim() == 4:
        tensor = tensor[0]
    return tensor.permute(1, 2, 0).cpu().numpy()
```

- `tensor_to_pil`과 유사하지만 numpy 배열(float, `[0, 1]`)을 반환한다. PIL로 변환하지 않으므로 matplotlib 등에서 직접 사용할 때 편리하다.

---

## 요약

이 노트북의 핵심 파이프라인은 다음과 같다:

1. **사용자 입력**: 이미지 업로드 + 결함 타입 선택 + 마스크 영역 선택
2. **전처리**: PIL → 256x256 그레이스케일 → `[-1, 1]` 정규화 텐서
3. **VQGAN 인코딩**: `[1, 3, 256, 256]` → `[1, 16, 16]` 토큰 맵
4. **마스킹**: 선택된 영역의 토큰을 `[MASK]`(16384)로 교체
5. **MaskGIT 반복 디코딩**: 코사인 스케줄에 따라 확신 높은 순서대로 점진적 확정
6. **VQGAN 디코딩**: `[1, 16, 16]` → `[1, 3, 256, 256]` 이미지 복원
7. **후처리**: 텐서 → PIL → Gradio UI에 표시

`.py` 모듈은 이 노트북의 코드를 **전역 변수 의존 제거**, **타입 힌트 추가**, **패키지 구조 임포트**로 리팩토링한 것이며, 알고리즘 로직 자체는 동일하다.
