# 노트북 2: Halton-MaskGIT 학습 상세 설명

이 문서는 `V0/metal_defect_HaltonMaskGIT(PoCFinal).ipynb` 노트북과 그에 대응하는 `src/metal_defect_synthesis/` 패키지의 모든 코드를 코드를 전혀 읽지 못하는 사람도 이해할 수 있도록 극도로 상세하게 설명합니다.

---

## 1. 핵심 상수와 설정

노트북 셀 9에서 전체 파이프라인에 사용되는 상수들을 정의합니다. 이 값들은 이후 모든 단계에서 반복적으로 참조되므로, 각각의 의미를 정확히 이해하는 것이 중요합니다.

### 각 상수 설명

| 상수 | 값 | 의미 |
|------|-----|------|
| `CODEBOOK_SIZE` | 16384 | VQGAN 코드북의 크기입니다. 이미지를 토큰으로 변환할 때, 각 토큰은 0부터 16383까지의 정수 중 하나가 됩니다. 16384개의 "시각적 단어"가 있는 사전이라고 생각하면 됩니다. LlamaGen VQGAN이 사용하는 기본값입니다. |
| `CODEBOOK_DIM` | 8 | 코드북의 각 항목을 나타내는 벡터의 차원입니다. 기존 taming-transformers는 256차원이었지만, LlamaGen은 8차원만 사용합니다. 차원이 낮을수록 메모리 효율이 좋고 연산이 빠릅니다. L2 정규화 덕분에 낮은 차원에서도 성능이 유지됩니다. |
| `DOWNSAMPLE_FACTOR` | 16 | VQGAN 인코더가 이미지를 얼마나 축소하는지 나타냅니다. 256x256 이미지가 16x16 격자로 줄어듭니다. 즉 가로세로 각각 16분의 1로 축소됩니다. |
| `LATENT_SIZE` | 16 | 잠재 공간(latent space)의 한 변 크기입니다. 256 / 16 = 16이므로, 인코딩된 결과는 16x16 격자입니다. |
| `SEQ_LEN` | 256 | 16 x 16 = 256. 토큰 시퀀스의 길이입니다. 2차원 16x16 격자를 1차원으로 펼치면 256개의 토큰이 됩니다. 트랜스포머는 이 256개의 토큰을 시퀀스로 처리합니다. |
| `MASK_TOKEN_ID` | 16384 | 마스크 토큰의 ID입니다. 코드북 인덱스가 0~16383이므로, 16384는 코드북에 없는 특별한 토큰입니다. "이 위치의 토큰을 아직 모릅니다"라는 의미로 사용됩니다. |
| `NUM_CLASSES` | 6 | 금속 결함의 종류 수입니다. |

### 클래스 이름 (`CLASS_NAMES`)

| 인덱스 | 영문 | 한국어 | 설명 |
|--------|------|--------|------|
| 0 | crazing | 균열 | 금속 표면에 가는 금이 간 결함 |
| 1 | inclusion | 개재물 | 금속 내부에 이물질이 포함된 결함 |
| 2 | patches | 패치 | 표면에 불규칙한 얼룩이 생긴 결함 |
| 3 | pitted_surface | 피팅 | 표면에 작은 구멍이 생긴 결함 |
| 4 | rolled-in_scale | 압연 스케일 | 압연 공정에서 산화물이 눌려 들어간 결함 |
| 5 | scratches | 스크래치 | 표면에 긁힌 자국이 있는 결함 |

### 시드 고정 (`set_seed`)

**노트북 셀 8** / **`src/metal_defect_synthesis/utils/seed.py`**

실험의 재현성을 보장하기 위해 모든 난수 생성기의 시드를 42로 고정합니다.

- `random.seed(seed)`: Python 기본 난수 생성기의 시드 고정
- `np.random.seed(seed)`: NumPy 난수 생성기의 시드 고정
- `torch.manual_seed(seed)`: PyTorch CPU 난수 시드 고정
- `torch.cuda.manual_seed_all(seed)`: PyTorch GPU 난수 시드 고정 (멀티 GPU 포함)
- `torch.backends.cudnn.deterministic = True`: cuDNN이 결정론적 알고리즘만 사용하도록 설정
- `torch.backends.cudnn.benchmark = False`: cuDNN의 자동 최적화 끔 (재현성을 위해)

`.py` 파일(`utils/seed.py`)은 노트북의 `set_seed` 함수와 완전히 동일한 로직입니다.

---

## 2. VQGAN 래퍼 함수

**노트북 셀 12, 13** / **`src/metal_defect_synthesis/models/vqgan_wrapper.py`**

VQGAN(Vector Quantized Generative Adversarial Network)은 이미지를 이산 토큰으로 변환하고, 토큰을 다시 이미지로 복원하는 모델입니다. 이 프로젝트에서 VQGAN 자체는 이미 학습이 끝난 상태이며, 여기서는 "도구"로만 사용합니다.

### `load_vqgan` 함수

이 함수는 사전 학습된 VQGAN 모델을 메모리에 로드합니다.

**단계별 설명:**

1. **모델 생성** (`VQ_models["VQ-16"](...)`):
   - LlamaGen 라이브러리에서 VQ-16 아키텍처를 가져옵니다.
   - `codebook_size=16384`: 코드북에 16384개의 벡터를 갖도록 설정합니다.
   - `codebook_embed_dim=8`: 각 코드북 벡터의 차원을 8로 설정합니다.
   - 이 시점에서 모델은 랜덤 가중치를 가지고 있습니다.

2. **체크포인트 로드** (`torch.load(...)`):
   - 디스크에 저장된 학습 완료된 가중치 파일을 불러옵니다.
   - 체크포인트 파일의 구조에 따라 `"vqmodel"`, `"state_dict"`, 또는 최상위 딕셔너리에서 가중치를 찾습니다.

3. **가중치 적용** (`load_state_dict(..., strict=True)`):
   - 랜덤 가중치를 학습된 가중치로 교체합니다.
   - `strict=True`는 모든 파라미터가 정확히 일치해야 함을 의미합니다.

4. **디바이스 이동 + eval 모드**:
   - `.to(device)`: 모델을 GPU로 이동합니다.
   - `.eval()`: 평가 모드로 전환합니다. Dropout이 비활성화되고, BatchNorm이 고정됩니다.

5. **파라미터 고정** (`requires_grad = False`):
   - 모든 파라미터의 기울기 계산을 끕니다.
   - VQGAN은 이미 학습이 끝났으므로 추가 학습하지 않습니다.
   - 이렇게 하면 메모리도 절약됩니다.

**노트북과 `.py`의 차이**: `.py` 파일은 `codebook_size`와 `codebook_dim`을 함수 인자로 받지만, 노트북은 전역 상수 `CODEBOOK_SIZE`와 `CODEBOOK_DIM`을 직접 사용합니다. 또한 `.py`에는 `logging`을 통한 로드 완료 메시지가 있습니다.

### `encode_to_tokens` 함수

이미지를 VQGAN 토큰으로 변환합니다.

**입출력 텐서 형상:**
- 입력: `images` - 형상 `[B, 3, 256, 256]`, 값 범위 [-1, 1]
  - B: 배치 크기 (동시에 처리하는 이미지 수)
  - 3: RGB 채널
  - 256x256: 이미지 해상도
- 출력: `tokens` - 형상 `[B, 16, 16]`, 각 값은 0~16383의 정수

**내부 동작:**
1. `vqgan_model.encode(images)`를 호출합니다.
   - 반환값: `(z_quantized, codebook_loss, (perplexity, min_encodings, indices))`
   - 우리가 필요한 것은 `indices` (코드북 인덱스)뿐입니다.
   - `_,  _, (_, _, indices)`로 나머지를 무시하고 `indices`만 추출합니다.
2. `indices`는 1차원 벡터 `[B*16*16]` 형태입니다.
3. `.view(batch_size, 16, 16)`으로 2차원 격자 형태 `[B, 16, 16]`으로 재구성합니다.

`@torch.no_grad()` 데코레이터는 이 함수 내부에서 기울기 계산을 하지 않도록 합니다 (인코딩은 추론이므로).

**노트북과 `.py`의 차이**: `.py`는 `latent_size`를 인자로 받고, 노트북은 전역 상수 `LATENT_SIZE`를 사용합니다. 로직은 동일합니다.

### `decode_from_tokens` 함수

토큰을 다시 이미지로 변환합니다.

**입출력 텐서 형상:**
- 입력: `tokens` - 형상 `[B, 16, 16]`, 각 값은 0~16383
- 출력: `images` - 형상 `[B, 3, 256, 256]`

**내부 동작:**
1. `tokens.view(-1)`: 토큰을 완전히 1차원으로 펼칩니다 → `[B*256]`
2. `torch.clamp(tokens_flat, 0, CODEBOOK_SIZE - 1)`: 토큰 값을 0~16383 범위로 제한합니다. 혹시 마스크 토큰(16384)이 남아 있으면 16383으로 잘라냅니다.
3. `embed_dim = vqgan_model.quantize.embedding.weight.shape[1]`: 코드북 임베딩의 차원(8)을 가져옵니다.
4. `vqgan_model.decode_code(tokens_flat, shape=(...))`: 토큰 인덱스를 코드북에서 찾아 벡터로 변환한 후, 디코더를 통과시켜 이미지를 복원합니다. `shape` 파라미터로 `(batch_size, 8, 16, 16)` 형태를 알려줍니다.

**노트북과 `.py`의 차이**: `.py`는 `latent_size`와 `codebook_size`를 함수 인자로 받습니다. 노트북은 전역 상수를 사용합니다.

---

## 3. 데이터 증강

**노트북 셀 17** / **`src/metal_defect_synthesis/data/augmentation.py`**

### `get_augmented_versions` 함수

하나의 이미지에서 8가지 변형을 만들어 데이터를 8배로 늘립니다.

**입력**: `img_tensor` - 형상 `[3, 256, 256]`, 단일 이미지 텐서

**출력**: 8개의 `[3, 256, 256]` 텐서 리스트

**8가지 증강 각각의 의미:**

1. **원본**: 아무 변환 없이 그대로 사용합니다.
2. **좌우반전 (hflip)**: 이미지를 거울처럼 좌우로 뒤집습니다. 왼쪽에 있던 결함이 오른쪽으로 이동합니다.
3. **상하반전 (vflip)**: 이미지를 위아래로 뒤집습니다. 위에 있던 결함이 아래로 이동합니다.
4. **90도 회전**: 이미지를 반시계 방향으로 90도 돌립니다.
5. **180도 회전**: 이미지를 180도 돌립니다 (좌우반전 + 상하반전과 같은 효과).
6. **270도 회전**: 이미지를 반시계 방향으로 270도 돌립니다 (시계 방향 90도와 동일).
7. **90도 회전 + 좌우반전**: 먼저 90도 회전한 후, 좌우로 뒤집습니다.
8. **90도 회전 + 상하반전**: 먼저 90도 회전한 후, 위아래로 뒤집습니다.

이 8가지는 정사각형의 대칭군(Dihedral group D4)에 해당하며, 겹치는 변환이 없습니다. 금속 결함은 방향에 의존하지 않으므로 (스크래치가 왼쪽에 있든 오른쪽에 있든 스크래치임), 이런 기하학적 증강이 효과적입니다.

**노트북과 `.py`의 차이**: 완전히 동일합니다. 함수 시그니처에 `.py`만 타입 힌트(`torch.Tensor`, `List[torch.Tensor]`)가 추가되어 있습니다.

---

## 4. 토큰 캐시

**노트북 셀 18, 19** / **`src/metal_defect_synthesis/data/token_cache.py`**

### `create_token_cache` 함수

전체 데이터셋의 모든 이미지를 VQGAN 토큰으로 미리 변환하여 파일로 저장합니다. 이렇게 하면 학습 중 매번 VQGAN 인코딩을 수행할 필요가 없어 속도가 크게 향상됩니다.

**전체 흐름:**

1. **빈 리스트 준비**: `all_tokens`(토큰 저장용)와 `all_labels`(라벨 저장용)를 빈 리스트로 만듭니다.

2. **DataLoader 생성**: `batch_size=1`로 설정하여 이미지를 하나씩 가져옵니다.

3. **루프 (각 이미지에 대해)**:
   - `img.to(device).squeeze(0)`: GPU로 이미지를 보내고, 배치 차원을 제거합니다. `[1, 3, 256, 256]` -> `[3, 256, 256]`
   - 증강이 활성화된 경우 (`augment=True`):
     - `get_augmented_versions(img)`를 호출하여 8가지 변형을 만듭니다.
     - 각 변형에 대해:
       - `.unsqueeze(0)`: 배치 차원 추가 → `[1, 3, 256, 256]`
       - `encode_to_tokens(aug_img, vqgan_model)`: VQGAN으로 인코딩 → `[1, 16, 16]`
       - `.cpu()`: GPU에서 CPU로 이동 (메모리 절약)
       - 토큰과 라벨을 리스트에 추가
   - 증강 없는 경우: 원본 이미지만 인코딩하여 추가

4. **결합**: `torch.cat`으로 모든 토큰을 하나의 텐서로 합칩니다.
   - `all_tokens`: `[N, 16, 16]` (N = 총 이미지 수 x 8)
   - `all_labels`: `[N]`

5. **캐시 딕셔너리 생성**:
   ```
   cache_data = {
       'tokens':       [N, 16, 16] 텐서 - 모든 토큰
       'labels':       [N] 텐서 - 각 토큰의 클래스 라벨
       'class_names':  ['crazing', 'inclusion', ...] - 클래스 이름 리스트
       'num_classes':  6 - 클래스 수
       'codebook_size': 16384 - 코드북 크기
       'seq_len':      256 - 시퀀스 길이
       'latent_size':  16 - 잠재 공간 크기
   }
   ```

6. **저장**: `torch.save`로 `.pt` 파일에 저장합니다.

노트북에서는 캐시 파일이 이미 존재하면 `torch.load`로 불러오고, 없으면 새로 생성합니다.

**노트북과 `.py`의 차이**: `.py`는 `encode_fn`(인코딩 함수)을 인자로 받아 유연성이 높습니다. 노트북은 전역의 `encode_to_tokens` 함수를 직접 호출합니다. 캐시 딕셔너리 구조와 증강 로직은 동일합니다.

### `TokenDataset` 클래스

**노트북 셀 19** (`.py`에는 별도 파일 없음, 노트북 내에서만 정의)

PyTorch의 `Dataset`을 상속하여 Halton-MaskGIT이 기대하는 형식으로 데이터를 제공합니다.

- **`__init__`**: `tokens`(`[N, 16, 16]`)과 `labels`(`[N]`)를 저장합니다.
- **`__len__`**: 데이터셋 크기(N)를 반환합니다.
- **`__getitem__`**: 인덱스 `idx`에 해당하는 데이터를 딕셔너리로 반환합니다.
  - `"code"`: `[16, 16]` - 해당 이미지의 VQGAN 토큰
  - `"y"`: 정수 - 해당 이미지의 클래스 라벨 (0~5)

이 딕셔너리 형식(`"code"`와 `"y"` 키)은 Halton-MaskGIT의 학습 코드가 기대하는 형식과 일치합니다. DataLoader는 이 딕셔너리들을 배치로 묶어 `{"code": [B, 16, 16], "y": [B]}`를 반환합니다.

---

## 5. 트랜스포머 아키텍처 (가장 중요!)

**노트북 셀 21, 22** / **`src/metal_defect_synthesis/models/layers.py`** 및 **`src/metal_defect_synthesis/models/maskgit.py`**

이 부분이 Halton-MaskGIT의 핵심입니다. 마스킹된 토큰을 입력으로 받아, 각 마스크 위치에 어떤 토큰이 와야 하는지 예측하는 트랜스포머 모델입니다.

### `modulate` 함수

**파일**: `layers.py` 라인 12-14 / 노트북 셀 21

AdaLayerNorm에서 사용하는 변조(modulation) 연산입니다.

**수학적 표현**: `output = x * (1 + scale) + shift`

**변수 설명:**
- `x`: 입력 텐서, 형상 `[B, N, C]` (B=배치, N=시퀀스길이(256), C=히든차원(512))
- `shift`: 이동 파라미터, 형상 `[B, C]` → `.unsqueeze(1)` 후 `[B, 1, C]`
- `scale`: 스케일 파라미터, 형상 `[B, C]` → `.unsqueeze(1)` 후 `[B, 1, C]`

**무엇을 하는가**: 정규화된 값을 클래스 조건에 따라 크기를 조절(scale)하고 위치를 이동(shift)합니다. `scale=0, shift=0`이면 입력이 그대로 통과합니다. 이것이 AdaLN의 핵심 아이디어입니다: 클래스 정보가 scale/shift 값을 결정하여, 같은 트랜스포머 블록이 클래스에 따라 다르게 동작하도록 합니다.

### `RMSNorm` 클래스

**파일**: `layers.py` 라인 17-26 / 노트북 셀 21

Root Mean Square Layer Normalization입니다.

**초기화 파라미터:**
- `dim`: 정규화할 차원 (512)
- `eps`: 0으로 나누기 방지용 매우 작은 값 (1e-5)
- `self.weight`: 학습 가능한 스케일 파라미터, 형상 `[dim]`, 초기값 1

**Forward 동작 (줄 단위):**
1. `x.pow(2)`: 입력의 각 값을 제곱합니다 → `[B, N, C]`
2. `.mean(-1, keepdim=True)`: 마지막 차원(C)에 대한 평균을 구합니다 → `[B, N, 1]`. 이것이 "Root Mean Square"의 "Mean Square" 부분입니다.
3. `+ self.eps`: 0으로 나누기를 방지하기 위해 아주 작은 값을 더합니다.
4. `torch.rsqrt(...)`: 역제곱근(1/sqrt)을 구합니다 → `[B, N, 1]`
5. `x * ...`: 원래 입력에 역제곱근을 곱합니다. 이것으로 각 위치의 벡터 크기가 약 1로 정규화됩니다.
6. `self.weight * norm`: 학습 가능한 가중치를 곱합니다 → `[B, N, C]`

**왜 LayerNorm이 아닌 RMSNorm인가?** LayerNorm은 평균을 빼고 분산으로 나누지만, RMSNorm은 평균 빼기를 생략합니다. 연산이 더 간단하면서도 성능은 비슷합니다. LLaMA 등 최신 대형 언어모델에서 널리 사용됩니다.

### `SwiGLU` 클래스

**파일**: `layers.py` 라인 29-48 / 노트북 셀 21

LLaMA 스타일의 피드포워드 네트워크(FFN)입니다.

**초기화 파라미터:**
- `dim`: 입력/출력 차원 (512)
- `hidden_dim`: 중간 확장 차원 (입력의 4배 = 2048이 전달됨)
- `dropout`: 드롭아웃 비율

**hidden_dim 조정 과정:**
1. `int(2 * hidden_dim / 3)`: SwiGLU는 게이트 메커니즘으로 인해 실질적 파라미터가 1.5배이므로, 보상을 위해 2/3로 줄입니다. 2048 * 2/3 = 1365
2. `256 * ((hidden_dim + 255) // 256)`: 256의 배수로 올림합니다. 1365 -> 1536. 이는 GPU의 텐서 코어가 256의 배수에서 가장 효율적으로 동작하기 때문입니다.

**레이어 구성:**
- `self.w1`: `nn.Linear(512, 1536, bias=False)` - 게이트 경로
- `self.w3`: `nn.Linear(512, 1536, bias=False)` - 값 경로
- `self.w2`: `nn.Linear(1536, 512, bias=False)` - 출력 투영

**Forward 동작:**
1. `F.silu(self.w1(x))`: 입력을 W1으로 선형 변환한 후 SiLU 활성화 함수(x * sigmoid(x))를 적용합니다 → `[B, N, 1536]`
2. `self.w3(x)`: 입력을 W3으로 별도 선형 변환합니다 → `[B, N, 1536]`
3. 두 결과를 **원소별 곱셈**합니다. 이것이 "Gated" 메커니즘입니다. W3의 출력이 게이트 역할을 하여 정보를 선택적으로 통과시킵니다.
4. 드롭아웃을 적용합니다.
5. `self.w2(...)`: 다시 원래 차원(512)으로 투영합니다 → `[B, N, 512]`

**수학적 표현**: `SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))`

### `QKNorm` 클래스

**파일**: `layers.py` 라인 51-59 / 노트북 셀 21

Query와 Key를 각각 정규화하는 모듈입니다.

- `self.q_norm`: Query용 RMSNorm (차원: 512)
- `self.k_norm`: Key용 RMSNorm (차원: 512)

**왜 Q와 K를 따로 정규화하는가?** 어텐션에서 Q와 K의 내적이 너무 크면 softmax가 극단적으로 되어 학습이 불안정해집니다. Q와 K를 각각 정규화하면 내적의 크기가 적절히 유지되어 학습이 안정적이 됩니다. 특히 딥 트랜스포머(12층 이상)에서 효과가 큽니다.

### `Attention` 클래스

**파일**: `layers.py` 라인 62-102 / 노트북 셀 21

Multi-Head Self-Attention을 구현합니다.

**초기화 파라미터:**
- `dim`: 512 (입력/출력 차원)
- `num_heads`: 8 (어텐션 헤드 수)
- `head_dim`: `dim // num_heads` = 512 / 8 = 64 (각 헤드의 차원)
- `scale`: `head_dim ** -0.5` = 1/8 = 0.125 (어텐션 스코어 스케일링)

**선형 변환 레이어 (모두 bias 없음):**
- `self.wq`: `nn.Linear(512, 512)` - Query 투영
- `self.wk`: `nn.Linear(512, 512)` - Key 투영
- `self.wv`: `nn.Linear(512, 512)` - Value 투영
- `self.wo`: `nn.Linear(512, 512)` - 출력 투영

**Forward 동작 (텐서 형상 변화):**

1. 입력: `x` 형상 `[B, 256, 512]`

2. **Q, K, V 투영**:
   - `q = self.wq(x)` → `[B, 256, 512]`
   - `k = self.wk(x)` → `[B, 256, 512]`
   - `v = self.wv(x)` → `[B, 256, 512]`

3. **QK 정규화**:
   - `q, k = self.qk_norm(q, k)`: 각각 RMSNorm 적용 → 형상 유지 `[B, 256, 512]`

4. **멀티헤드용 재구성**:
   - `q.view(B, 256, 8, 64)`: 512를 8개 헤드 x 64차원으로 분할
   - `.transpose(1, 2)`: `[B, 8, 256, 64]` - 헤드 차원을 앞으로 이동
   - K, V도 동일하게 → 각각 `[B, 8, 256, 64]`

5. **Scaled Dot-Product Attention** (Flash Attention):
   - PyTorch 2.0+의 `F.scaled_dot_product_attention` 사용
   - 내부적으로: `softmax(Q * K^T / sqrt(64)) * V`를 계산
   - Flash Attention은 메모리 효율적인 구현으로, O(N^2) 메모리 대신 O(N)만 사용
   - 출력: `[B, 8, 256, 64]`

6. **재구성**:
   - `.transpose(1, 2)` → `[B, 256, 8, 64]`
   - `.contiguous().view(B, 256, 512)` → `[B, 256, 512]` (8개 헤드를 다시 합침)

7. **출력 투영**:
   - `self.wo(out)` → `[B, 256, 512]`

### `TransformerBlock` 클래스

**파일**: `layers.py` 라인 105-145 / 노트북 셀 21

하나의 트랜스포머 블록입니다. 모델은 이 블록을 12개 쌓아 사용합니다.

**초기화 구성:**
- `self.adaLN_modulation`: 클래스 조건에서 6개의 변조 파라미터를 생성하는 MLP
  - `nn.SiLU()`: 활성화 함수
  - `nn.Linear(512, 512*6)`: 512차원 입력에서 3072차원(512x6) 출력
- `self.norm1`: 어텐션 전 RMSNorm
- `self.attn`: Multi-Head Self-Attention
- `self.norm2`: FFN 전 RMSNorm
- `self.ffn`: SwiGLU FFN

**Forward 동작 (AdaLN의 핵심):**

입력: `x` 형상 `[B, 256, 512]`, `cond` 형상 `[B, 512]` (클래스 임베딩)

1. **6개 변조 파라미터 계산**:
   - `self.adaLN_modulation(cond)`: `[B, 512]` → `[B, 3072]`
   - `.chunk(6, dim=-1)`: 6등분하여 각각 `[B, 512]`
   - `gamma1, beta1, alpha1`: 어텐션 브랜치용
   - `gamma2, beta2, alpha2`: FFN 브랜치용

2. **어텐션 브랜치**:
   - `self.norm1(x)`: RMSNorm 적용 → `[B, 256, 512]`
   - `modulate(..., beta1, gamma1)`: `x * (1 + gamma1) + beta1` 적용 (beta1=shift, gamma1=scale)
   - `self.attn(...)`: 어텐션 계산 → `[B, 256, 512]`
   - `alpha1.unsqueeze(1) * ...`: 어텐션 출력에 게이트 적용 (alpha1이 0에 가까우면 어텐션 결과가 거의 무시됨)
   - `x = x + ...`: 잔차 연결 (residual connection)

3. **FFN 브랜치**:
   - 어텐션과 동일한 패턴으로 `norm2 → modulate(beta2, gamma2) → ffn → alpha2 게이트 → 잔차 연결`

**gamma, beta, alpha 각각의 역할:**
- `gamma` (scale): 정규화된 값의 크기를 조절합니다. "이 특성을 더 강조해라/줄여라"
- `beta` (shift): 정규화된 값의 위치를 이동합니다. "이 특성의 기준점을 옮겨라"
- `alpha` (gate): 어텐션/FFN의 출력을 얼마나 반영할지 결정합니다. "이 블록의 기여도를 조절해라"

### `AdaNorm` 클래스

**파일**: `layers.py` 라인 148-161 / 노트북 셀 21

트랜스포머 블록들을 모두 통과한 후, 출력 헤드 직전에 적용되는 최종 정규화 레이어입니다.

- TransformerBlock의 adaLN이 6개 파라미터를 쓰는 것과 달리, 여기는 2개 파라미터(shift, scale)만 사용합니다. alpha(게이트)는 필요 없습니다.
- `self.adaLN`: `nn.SiLU() + nn.Linear(512, 1024)` → `.chunk(2)` → shift와 scale 각각 `[B, 512]`

### `MaskGITTransformer` 클래스

**파일**: `src/metal_defect_synthesis/models/maskgit.py` / 노트북 셀 22

전체 MaskGIT 트랜스포머 모델을 하나로 묶는 메인 클래스입니다.

**초기화 파라미터와 각 멤버 변수:**

| 파라미터 | 기본값 | 저장 변수 | 설명 |
|---------|--------|-----------|------|
| `vocab_size` | 16384 | `self.vocab_size` | 코드북 크기 |
| `seq_len` | 256 | `self.seq_len` | 토큰 시퀀스 길이 |
| `hidden_dim` | 512 | `self.hidden_dim` | 트랜스포머 히든 차원 |
| `num_layers` | 12 | (blocks 수) | 트랜스포머 블록 수 |
| `num_heads` | 8 | (블록 내부) | 어텐션 헤드 수 |
| `mlp_ratio` | 4.0 | (블록 내부) | FFN 확장 비율 |
| `dropout` | 0.1 | (블록 내부) | 드롭아웃 비율 |
| `num_classes` | 6 | `self.num_classes` | 결함 클래스 수 |
| - | 16384 | `self.mask_token_id` | `vocab_size`와 동일 |

**임베딩 레이어:**

1. `self.tok_emb = nn.Embedding(16385, 512)`:
   - 토큰 임베딩 테이블입니다.
   - 16385 = 16384(코드북) + 1([MASK] 토큰)
   - 각 토큰 ID를 512차원 벡터로 변환합니다.

2. `self.pos_emb = nn.Embedding(256, 512)`:
   - 위치 임베딩 테이블입니다.
   - 0번부터 255번까지 256개 위치 각각에 고유한 512차원 벡터를 부여합니다.
   - 트랜스포머는 위치 정보가 없으면 토큰의 순서를 알 수 없으므로 필수적입니다.

3. `self.cls_emb = nn.Embedding(7, 512)`:
   - 클래스 임베딩 테이블입니다.
   - 7 = 6(결함 클래스) + 1(CFG용 null 클래스)
   - 인덱스 0~5: 각 결함 유형
   - 인덱스 6: "클래스 조건 없음" (Classifier-Free Guidance에서 unconditional 생성 시 사용)

**트랜스포머 블록:**
- `self.blocks`: 12개의 `TransformerBlock`을 리스트로 보관합니다.

**출력 레이어:**
- `self.final_norm`: `AdaNorm(512)` - 최종 정규화
- `self.head`: `nn.Linear(512, 16385, bias=False)` - 각 위치에서 어떤 토큰이 올지 예측
- **Weight Tying**: `self.head.weight = self.tok_emb.weight` - 출력 헤드의 가중치를 토큰 임베딩의 가중치와 공유합니다. 이렇게 하면 파라미터 수가 줄고, 입출력 공간이 일관성을 갖습니다. 토큰 임베딩이 "토큰 → 벡터" 매핑이라면, 출력 헤드는 "벡터 → 토큰" 매핑이므로 같은 가중치를 사용하는 것이 합리적입니다.

**`_init_weights` 메서드 (가중치 초기화):**

- `nn.init.normal_(self.tok_emb.weight, std=0.02)`: 토큰 임베딩을 평균 0, 표준편차 0.02의 정규분포로 초기화합니다.
- `nn.init.normal_(self.pos_emb.weight, std=0.02)`: 위치 임베딩도 동일하게 초기화합니다.
- `nn.init.normal_(self.cls_emb.weight, std=0.02)`: 클래스 임베딩도 동일하게 초기화합니다.
- **AdaLN modulation 제로 초기화**: 각 블록의 `adaLN_modulation`의 Linear 레이어(인덱스 1)의 weight와 bias를 0으로 초기화합니다. 이렇게 하면 학습 초기에 gamma=0, beta=0, alpha=0이 되어 modulate가 항등 함수가 되고, 트랜스포머 블록의 출력이 잔차 연결만 남게 됩니다. 이것은 깊은 모델의 학습 안정성을 크게 향상시키는 기법입니다.

**`forward` 메서드 (순전파 - 단계별):**

입력: `x` 형상 `[B, 16, 16]`, `y` 형상 `[B]`, `drop_label` 형상 `[B]` (bool)

1. **입력 재구성**: `x.dim() == 3`이면 `x.view(B, -1)` → `[B, 256]`

2. **클래스 조건 처리**:
   - `drop_label`이 주어진 경우, `drop_label`이 True인 위치의 라벨을 `num_classes`(=6)로 교체합니다. 이것이 CFG(Classifier-Free Guidance)를 위한 것입니다.
   - `cond = self.cls_emb(y)`: 클래스 인덱스를 임베딩 벡터로 변환 → `[B, 512]`

3. **토큰 + 위치 임베딩**:
   - `pos = torch.arange(256, device=x.device)`: `[0, 1, 2, ..., 255]` 생성
   - `self.tok_emb(x)`: `[B, 256]` → `[B, 256, 512]` (각 토큰 ID를 벡터로)
   - `self.pos_emb(pos)`: `[256]` → `[256, 512]` (각 위치를 벡터로, 브로드캐스팅됨)
   - 두 임베딩을 더합니다 → `[B, 256, 512]`

4. **12개 트랜스포머 블록 통과**:
   - `for block in self.blocks: x = block(x, cond)`
   - 각 블록은 `x`(`[B, 256, 512]`)와 `cond`(`[B, 512]`)를 받아 같은 형상 `[B, 256, 512]`를 반환합니다.

5. **최종 정규화**:
   - `x = self.final_norm(x, cond)` → `[B, 256, 512]`

6. **출력 헤드**:
   - `logits = self.head(x)` → `[B, 256, 16385]`
   - 각 위치(256개)에 대해 16385개 토큰 중 어느 것이 올 확률이 높은지를 나타내는 로짓

**노트북과 `.py`의 차이**: 로직은 완전히 동일합니다. `.py`에서는 `layers.py`에서 `TransformerBlock`과 `AdaNorm`을 import하고, 노트북에서는 같은 셀 또는 이전 셀에서 직접 정의합니다. `.py`에는 타입 힌트(`Optional`, `Dict`)가 추가되어 있습니다.

### `get_model_config` 함수

**파일**: `maskgit.py` 라인 126-142 / 노트북 셀 23

모델 크기별 설정을 딕셔너리로 반환합니다.

| 크기 | hidden_dim | num_layers | num_heads | 대략적 파라미터 수 |
|------|-----------|-----------|----------|----------------|
| tiny | 384 | 6 | 6 | 23M |
| small | 512 | 12 | 8 | 69M |
| base | 768 | 12 | 12 | 142M |
| large | 1024 | 24 | 16 | 480M |

이 프로젝트에서는 약 9000장(증강 포함)의 데이터에 적합한 `small` 크기를 사용합니다.

---

## 6. 마스킹 함수

**노트북 셀 26** / **`src/metal_defect_synthesis/training/maskgit_trainer.py`** 내부

### `get_mask_schedule` 함수

마스킹 비율을 결정하는 스케줄 함수입니다.

**입력**: `r` - 형상 `[B]`, 0에서 1 사이의 균등 분포 랜덤 값

**출력**: `mask_ratio` - 형상 `[B]`, 실제로 마스킹할 비율 (0~1)

**arccos 스케줄 (기본값):**
- `torch.arccos(r) / (pi / 2)`
- r=0일 때: arccos(0) = pi/2이므로 mask_ratio = 1 (100% 마스킹)
- r=1일 때: arccos(1) = 0이므로 mask_ratio = 0 (0% 마스킹)
- r=0.5일 때: arccos(0.5) = pi/3이므로 mask_ratio = 2/3 (약 67% 마스킹)

arccos 스케줄의 특징은 높은 마스킹 비율(80~100%)에서 더 많은 시간을 보내도록 분포가 편향되어 있다는 것입니다. 이는 모델이 거의 모든 토큰이 마스킹된 어려운 상황에서도 잘 예측하도록 학습하게 합니다. 기존 MaskGIT의 cosine 스케줄보다 더 균일한 난이도 분포를 제공합니다.

**다른 스케줄 옵션:**
- `cosine`: `cos(r * pi/2)` - 기존 MaskGIT 방식
- `linear`: `1 - r` - 가장 단순
- `square`: `1 - r^2` - 낮은 마스킹 비율에 편향

### `mask_tokens` 함수

실제로 토큰에 마스킹을 수행합니다.

**입력:**
- `tokens`: `[B, H, W]` 원본 토큰 (예: `[32, 16, 16]`)
- `mask_token_id`: 16384
- `mode`: `'arccos'`

**출력:**
- `masked_tokens`: `[B, H, W]` 마스킹된 토큰 (일부가 16384로 교체됨)
- `mask`: `[B, H, W]` bool 텐서 (True = 해당 위치가 마스킹됨)

**단계별 동작:**

1. `r = torch.rand(B, device=device)`: 각 배치 샘플에 대해 0~1 사이 랜덤 값 생성 → `[B]`

2. `mask_ratios = get_mask_schedule(r, mode)`: arccos 스케줄로 마스킹 비율 계산 → `[B]`

3. `num_to_mask = (mask_ratios * seq_len).long()`: 비율을 실제 토큰 수로 변환. 예: 0.67 * 256 = 171개

4. `torch.clamp(num_to_mask, min=1, max=seq_len-1)`: 최소 1개, 최대 255개로 제한. 0개(마스킹 없음)나 256개(전부 마스킹)는 학습에 도움이 안 되므로 제외합니다.

5. `tokens_flat = tokens.view(B, -1)`: `[B, 16, 16]` → `[B, 256]`으로 평탄화

6. **배치의 각 샘플에 대해 반복:**
   - `torch.randperm(seq_len, device=device)`: 0~255를 랜덤하게 섞은 순열 생성
   - `perm[:num_mask]`: 앞에서 `num_mask`개를 선택 → 마스킹할 위치
   - 선택된 위치의 토큰을 `mask_token_id`(16384)로 교체
   - 해당 위치의 마스크를 True로 설정

7. 다시 `[B, H, W]` 형태로 복원

---

## 7. 학습 루프

**노트북 셀 27, 28, 34** / **`src/metal_defect_synthesis/training/maskgit_trainer.py`** 및 **`src/metal_defect_synthesis/training/scheduler.py`**

### Optimizer와 Scheduler

**AdamW Optimizer:**
- `lr=1e-4`: 학습률. 한 번에 가중치를 얼마나 업데이트할지 결정합니다.
- `betas=(0.9, 0.999)`: 모멘텀 파라미터. 0.9는 1차 모멘트(평균), 0.999는 2차 모멘트(분산)의 지수이동평균 비율입니다.
- `weight_decay=0.03`: 가중치 감쇠. 가중치가 너무 커지지 않도록 규제합니다. AdamW는 Adam과 달리 weight decay를 기울기 업데이트와 분리하여 적용합니다.

**Warmup + Cosine Annealing Scheduler:**

파일: `src/metal_defect_synthesis/training/scheduler.py`의 `get_lr_scheduler` 함수

`lr_lambda` 함수가 에폭별 학습률 배율을 결정합니다:

- **Warmup 구간** (에폭 0~4, 총 5에폭):
  - `(epoch + 1) / warmup_epochs`
  - 에폭 0: 1/5 = 0.2배, 에폭 1: 2/5 = 0.4배, ..., 에폭 4: 5/5 = 1.0배
  - 학습 초반에 학습률을 천천히 올려 불안정한 초기 업데이트를 방지합니다.

- **Cosine Annealing 구간** (에폭 5~99):
  - `progress = (epoch - 5) / (100 - 5)` → 0에서 1로 증가
  - `0.5 * (1 + cos(pi * progress))` → 1에서 0으로 코사인 커브를 따라 감소
  - 학습 후반부에 학습률을 자연스럽게 0으로 줄여 세밀한 수렴을 돕습니다.

**노트북과 `.py`의 차이**: `.py`는 `get_lr_scheduler` 함수가 별도 파일(`scheduler.py`)에 있고, `MaskGITTrainer` 클래스 내부에서 import하여 사용합니다. 노트북은 같은 셀에서 직접 정의합니다. 수식은 동일합니다.

### `train_one_epoch` 함수 / `MaskGITTrainer.train_one_epoch`

한 에폭의 학습을 수행합니다.

**전체 흐름 (각 배치에 대해):**

1. **데이터 준비:**
   - `code = batch['code'].to(device)`: 토큰 `[B, 16, 16]`을 GPU로 이동
   - `y = batch['y'].to(device)`: 라벨 `[B]`를 GPU로 이동

2. **CFG용 라벨 드롭:**
   - `drop_label = (torch.rand(B, device=device) < 0.1)`: 각 샘플에 10% 확률로 True
   - True인 샘플은 "클래스 조건 없이" 학습됩니다.
   - 이것이 CFG의 핵심: 학습 시 일부 샘플에서 조건을 제거하여, 모델이 조건부/무조건부 생성을 모두 배우게 합니다.

3. **마스킹 적용:**
   - `mask_tokens(code, MASK_TOKEN_ID, mode='arccos')`
   - 반환: `masked_code`(`[B, 16, 16]`, 일부가 16384), `mask`(`[B, 16, 16]`, bool)

4. **Forward:**
   - `logits = model(masked_code, y, drop_label)` → `[B, 256, 16385]`
   - 모델은 마스킹된 토큰 시퀀스를 보고, 각 위치에 어떤 토큰이 와야 하는지 예측합니다.

5. **Loss 계산 (마스킹된 위치에서만):**
   - `logits_flat = logits.view(B * 256, 16385)`: 배치와 위치를 합쳐 평탄화
   - `target_flat = code.view(-1)`: 원본 토큰도 평탄화 → `[B*256]`
   - `mask_flat = mask.view(-1)`: 마스크도 평탄화 → `[B*256]`
   - `target_masked = target_flat.clone()`: 타겟 복사
   - `target_masked[~mask_flat] = -100`: **마스킹되지 않은 위치**의 타겟을 -100으로 설정
   - `criterion(logits_flat, target_masked)`: CrossEntropyLoss 계산
   - `ignore_index=-100`이므로 마스킹되지 않은 위치는 loss 계산에서 제외됩니다.
   - **왜 마스킹된 위치에서만?** 마스킹되지 않은 위치는 모델에게 정답이 이미 주어진 것이므로, 예측하는 것이 의미 없습니다. BERT의 Masked Language Model과 같은 원리입니다.

6. **Backward:**
   - `optimizer.zero_grad()`: 이전 기울기를 초기화합니다.
   - `loss.backward()`: 역전파로 기울기를 계산합니다.
   - `clip_grad_norm_(model.parameters(), 1.0)`: 기울기 크기를 최대 1.0으로 제한합니다. 기울기가 폭발하여 학습이 발산하는 것을 방지합니다.
   - `optimizer.step()`: 기울기를 사용하여 가중치를 업데이트합니다.

7. **정확도 계산:**
   - `pred = logits_flat.argmax(dim=-1)`: 각 위치에서 가장 높은 확률의 토큰을 선택
   - `correct = (pred == target_flat) & mask_flat`: 마스킹된 위치에서 정답을 맞춘 수
   - `acc = correct.sum() / mask_flat.sum()`: 마스킹 위치 중 정답 비율

**노트북과 `.py`의 차이**: `.py`는 `MaskGITTrainer` 클래스로 래핑되어 있으며 `self.model`, `self.optimizer` 등을 멤버로 가집니다. 노트북은 함수형으로 `model`, `dataloader`, `optimizer`, `criterion`을 인자로 받습니다. 학습 로직은 동일합니다. `.py`의 `MaskGITTrainer`에는 `train()` 메서드가 있어 전체 에폭 루프와 체크포인트 저장을 포함합니다.

---

## 8. Halton Sequence

**노트북 셀 30** / **`src/metal_defect_synthesis/sampling/halton.py`**

### `halton_sequence` 함수

1차원 Halton 시퀀스를 생성합니다.

**Halton 시퀀스란?** 0과 1 사이에서 "준무작위(quasi-random)" 수열을 만드는 알고리즘입니다. 일반 난수와 달리 공간을 더 균일하게 채웁니다.

**알고리즘 (base=2 예시):**

숫자 i를 base로 반복 나누며 소수점 이하를 구성합니다.

- i=1: 이진법 1 → 반전 0.1(이진) = 0.5
- i=2: 이진법 10 → 반전 0.01(이진) = 0.25
- i=3: 이진법 11 → 반전 0.11(이진) = 0.75
- i=4: 이진법 100 → 반전 0.001(이진) = 0.125

**코드의 while 루프 설명:**
```
n = i (현재 숫자)
f = 1.0, r = 0.0
반복:
  f /= base     → f가 1/2, 1/4, 1/8, ... 으로 줄어듦
  r += f * (n % base)  → n의 마지막 자릿수에 f를 곱해 누적
  n //= base    → n에서 마지막 자릿수를 제거
```

결과적으로, 숫자 i를 base진법으로 표현한 후 소수점을 기준으로 "뒤집은" 값이 됩니다.

**왜 소수(prime) 기저를 사용하는가?** 서로 다른 소수 기저(2, 3)를 사용하면 두 시퀀스 간의 상관관계가 없어 2D 공간을 더 균일하게 채울 수 있습니다.

### `build_halton_mask` 함수

2D Halton 마스크를 생성합니다. 이 마스크가 이미지 생성 시 토큰을 채워넣는 순서를 결정합니다.

**단계별 동작:**

1. `x = halton_sequence(2, 10000)`: base-2 Halton 시퀀스 10000개 생성 (x 좌표용)
2. `y = halton_sequence(3, 10000)`: base-3 Halton 시퀀스 10000개 생성 (y 좌표용)

3. **좌표 변환 루프:**
   - 각 (xi, yi) 쌍에 대해:
   - `row = int(xi * 16)`: [0, 1) 범위를 [0, 16) 범위로 스케일링 후 정수로 변환
   - `col = int(yi * 16)`: 같은 방식
   - `min(row, 15)`, `min(col, 15)`: 범위를 벗어나지 않도록 제한

4. **중복 제거:**
   - `seen` 집합으로 이미 나온 (row, col) 쌍을 추적
   - 같은 좌표가 나오면 건너뜁니다
   - 256개(= 16x16)가 모이면 종료

5. 결과: `[256, 2]` 텐서, 각 행이 (row, col) 좌표

**왜 10000개나 생성하는가?** Halton 시퀀스를 16x16 격자에 매핑하면 초반에 중복이 많이 발생합니다. 256개의 고유 좌표를 모으려면 충분히 많은 후보가 필요합니다.

**노트북과 `.py`의 차이**: 완전히 동일한 로직입니다.

---

## 9. HaltonSampler 클래스

**노트북 셀 31** / **`src/metal_defect_synthesis/sampling/sampler.py`**

### 초기화

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `num_steps` | 32 | 총 생성 스텝 수. 256개 토큰을 32단계에 걸쳐 채웁니다. |
| `cfg_weight` | 2.0 | CFG 가중치. 클래스 조건을 얼마나 강하게 반영할지 결정합니다. |
| `temperature` | 1.0 | Softmax 온도. 낮을수록 확신이 높은 토큰만 선택하고, 높을수록 다양한 토큰을 선택합니다. |
| `randomize` | True | Halton 시퀀스에 랜덤 오프셋을 줄지 여부 |

### `sample` 메서드

이미지를 처음부터 생성합니다.

**전체 생성 루프 (단계별):**

**1단계: 초기화**
- `code = torch.full((B, 16, 16), MASK_TOKEN_ID, device=device)`: 모든 위치를 [MASK](16384)로 채운 격자 생성
- 이것은 "아직 아무것도 생성하지 않은 빈 캔버스"입니다.

**2단계: CFG용 라벨 준비**
- `drop_label_cond = torch.zeros(B, dtype=torch.bool)`: 전부 False → 조건부
- `drop_label_uncond = torch.ones(B, dtype=torch.bool)`: 전부 True → 무조건부

**3단계: Halton mask 준비**
- `randomize=True`인 경우:
  - 각 샘플에 랜덤 오프셋을 부여합니다.
  - `torch.roll(halton_mask, shifts=offset, dims=0)`: Halton 순서를 offset만큼 회전합니다.
  - 이렇게 하면 같은 클래스라도 다른 순서로 토큰을 채워 다양한 이미지가 생성됩니다.
- 결과: `halton_masks` 형상 `[B, 256, 2]`

**4단계: 스텝별 생성 (32 스텝)**

각 스텝에서:

1. **예측할 토큰 수 계산:**
   - `ratio = (step + 1) / 32`: 현재 진행률 (0.03125 ~ 1.0)
   - `r = 1 - (arccos(ratio) / (pi/2))`: arccos 스케줄 적용
   - `curr_idx = max(step + 1, int(r * 256))`: 누적 토큰 수
   - 초반 스텝에서는 적은 토큰을, 후반에서는 많은 토큰을 한번에 채웁니다.

2. **이번 스텝에서 채울 위치 선택:**
   - `positions = halton_masks[:, prev_idx:curr_idx]`: `[B, num_new, 2]`
   - Halton 순서에 따라 공간적으로 균일하게 분포된 위치들입니다.

3. **CFG Forward:**
   - 배치를 두 배로 복제합니다: `code_double = [code; code]` → `[2B, 16, 16]`
   - 라벨도 복제: `labels_double = [labels; labels]`
   - 드롭 플래그: `drop_double = [False...; True...]` → 앞 절반은 조건부, 뒤 절반은 무조건부
   - 모델에 한 번만 Forward하여 두 종류의 logits를 동시에 구합니다.
   - `logits_double.chunk(2)` → `logits_cond`(조건부), `logits_uncond`(무조건부)

4. **CFG 공식 적용:**
   - `logits = (1 + cfg_weight) * logits_cond - cfg_weight * logits_uncond`
   - cfg_weight=2.0이면: `3 * logits_cond - 2 * logits_uncond`
   - 조건부 logits에서 무조건부 logits를 빼면, "클래스 조건이 기여하는 부분"만 남습니다. 이것을 증폭시켜 클래스 특성이 더 뚜렷한 이미지를 생성합니다.

5. **Softmax + Multinomial 샘플링:**
   - `logits.view(B, H, W, -1)` → `[B, 16, 16, 16385]`
   - `F.softmax(logits / temperature, dim=-1)`: 확률 분포로 변환
   - 각 배치의 각 위치에 대해:
     - `prob = probs[b, row, col, :CODEBOOK_SIZE]`: [MASK] 토큰을 제외한 확률만 사용
     - `prob / prob.sum()`: 재정규화 (합이 1이 되도록)
     - `torch.multinomial(prob, 1)`: 확률에 따라 토큰 하나를 랜덤 샘플링
     - 샘플링된 토큰으로 해당 위치를 채웁니다.

6. `prev_idx = curr_idx`: 다음 스텝을 위해 인덱스 업데이트

**5단계: VQGAN 디코딩**
- `code = torch.clamp(code, 0, CODEBOOK_SIZE - 1)`: 혹시 남은 마스크 토큰 처리
- `decode_from_tokens(code, vqgan)`: 256개 토큰을 256x256 이미지로 복원
- `torch.clamp(images, -1, 1)`: 이미지 값 범위 제한

**노트북과 `.py`의 차이:**
- `.py`는 `latent_size`, `codebook_size`, `mask_token_id`를 생성자 인자로 받아 더 일반적입니다. 내부에서 `build_halton_mask`를 호출합니다.
- 노트북은 전역 상수 `LATENT_SIZE`, `CODEBOOK_SIZE`, `MASK_TOKEN_ID`와 전역 변수 `HALTON_MASK`를 직접 참조합니다.
- 생성 로직 자체는 동일합니다.

---

## 10. .py 모듈 대조 요약

아래 표는 노트북의 각 부분과 대응하는 `.py` 파일을 비교합니다.

### `utils/seed.py` vs 노트북 셀 8

| 항목 | 노트북 | .py |
|------|--------|-----|
| 함수명 | `set_seed` | `set_seed` |
| 인자 | `seed=42` | `seed: int = 42` |
| 로직 | 동일 | 동일 (타입힌트 추가) |

### `models/vqgan_wrapper.py` vs 노트북 셀 12, 13

| 항목 | 노트북 | .py |
|------|--------|-----|
| `load_vqgan` 인자 | `ckpt_path, device` | `ckpt_path, device, codebook_size=16384, codebook_dim=8` |
| 전역 상수 참조 | `CODEBOOK_SIZE`, `CODEBOOK_DIM` 직접 사용 | 함수 인자로 받음 |
| 로깅 | `print` | `logging.info` |
| `encode_to_tokens` | 전역 `LATENT_SIZE` 사용 | `latent_size` 인자 |
| `decode_from_tokens` | 전역 `CODEBOOK_SIZE`, `LATENT_SIZE` 사용 | `latent_size`, `codebook_size` 인자 |

### `data/augmentation.py` vs 노트북 셀 17

| 항목 | 노트북 | .py |
|------|--------|-----|
| 로직 | 완전 동일 | 완전 동일 |
| 차이 | 타입힌트 없음 | `torch.Tensor`, `List[torch.Tensor]` 타입힌트 |

### `data/token_cache.py` vs 노트북 셀 18

| 항목 | 노트북 | .py |
|------|--------|-----|
| `create_token_cache` 인자 | `dataset, vqgan_model, cache_path, augment` | `dataset, vqgan_model, encode_fn, cache_path, augment, codebook_size, seq_len, latent_size, device` |
| 인코딩 호출 | `encode_to_tokens(aug_img, vqgan_model)` 직접 호출 | `encode_fn(aug_img, vqgan_model)` 인자로 받은 함수 호출 |
| 전역 상수 | `CODEBOOK_SIZE`, `SEQ_LEN`, `LATENT_SIZE` 직접 참조 | 모두 함수 인자로 받음 |
| `TokenDataset` | 노트북 셀 19에서 정의 | `.py`에 없음 (노트북 전용) |

### `models/layers.py` vs 노트북 셀 21

| 항목 | 노트북 | .py |
|------|--------|-----|
| `modulate` | 동일 | 동일 (타입힌트 추가) |
| `RMSNorm` | 동일 | 동일 (타입힌트 추가) |
| `SwiGLU` | 동일 | 동일 |
| `QKNorm` | 동일 | 동일 |
| `Attention` | 동일 | 동일 |
| `TransformerBlock` | 동일 | 동일 |
| `AdaNorm` | 동일 | 동일 |
| `einops` import | 있음 (미사용) | 없음 |

### `models/maskgit.py` vs 노트북 셀 22, 23

| 항목 | 노트북 | .py |
|------|--------|-----|
| `MaskGITTransformer` | 동일 | 동일 (타입힌트 추가) |
| `get_model_config` | 동일 | 동일 |
| TransformerBlock 참조 | 같은 셀/이전 셀에서 정의 | `from .layers import` |

### `training/maskgit_trainer.py` vs 노트북 셀 26, 27, 28, 34

| 항목 | 노트북 | .py |
|------|--------|-----|
| `get_mask_schedule` | 함수 | 동일한 함수 |
| `mask_tokens` | 함수 | 동일한 함수 |
| 학습 루프 | `train_one_epoch` 함수 + 메인 루프 | `MaskGITTrainer` 클래스 |
| Optimizer 생성 | 메인 루프에서 직접 | `MaskGITTrainer.__init__`에서 |
| Scheduler | 메인 루프에서 직접 | `from .scheduler import get_lr_scheduler` |
| 체크포인트 저장 | 메인 루프에서 | `_save_checkpoint` 메서드 |
| 샘플 생성 | 메인 루프에서 `generate_and_visualize` 호출 | 포함 안 됨 |
| `generate_and_visualize` | 노트북 셀 32 | `.py`에 없음 (노트북 전용) |

### `training/scheduler.py` vs 노트북 셀 27

| 항목 | 노트북 | .py |
|------|--------|-----|
| `get_lr_scheduler` | 셀 27에서 정의 | 별도 파일 |
| 수식 | 동일 | 동일 |

### `sampling/halton.py` vs 노트북 셀 30

| 항목 | 노트북 | .py |
|------|--------|-----|
| `halton_sequence` | 동일 | 동일 (타입힌트 추가) |
| `build_halton_mask` | 동일 | 동일 (타입힌트 추가) |

### `sampling/sampler.py` vs 노트북 셀 31

| 항목 | 노트북 | .py |
|------|--------|-----|
| 생성자 인자 | `num_steps, cfg_weight, temperature, randomize` | 위 4개 + `latent_size, codebook_size, mask_token_id` |
| Halton mask | 전역 `HALTON_MASK` 참조 | `build_halton_mask(latent_size)` 호출 |
| VQGAN 디코딩 | 전역 `decode_from_tokens` 직접 호출 | `from ..models.vqgan_wrapper import decode_from_tokens` |
| 전역 상수 | `LATENT_SIZE`, `MASK_TOKEN_ID`, `CODEBOOK_SIZE` 직접 참조 | 인자로 받은 값 사용 |
| 생성 로직 | 동일 | 동일 |

### 요약: 노트북에만 있고 .py에 없는 것들

1. **`TokenDataset` 클래스**: 노트북 셀 19에서 정의. `.py` 패키지에는 없습니다.
2. **`generate_and_visualize` 함수**: 노트북 셀 32에서 정의. matplotlib을 사용한 시각화 로직으로 `.py`에는 없습니다.
3. **VQGAN 인코딩/디코딩 테스트** (셀 14): 더미 이미지로 VQGAN이 올바르게 동작하는지 확인하는 코드.
4. **모델 Forward 테스트** (셀 24): 더미 토큰으로 MaskGITTransformer의 Forward가 올바른 형상을 반환하는지 확인하는 코드.
5. **데이터셋 통계 출력** (셀 16): 클래스별 이미지 수를 출력하는 확인용 코드.
6. **마스킹 테스트** (셀 26 하단): 마스킹 함수가 올바르게 동작하는지 확인하는 코드.

### 요약: .py에만 있고 노트북에 없는 것들

1. **`logging` 기반 로그**: `.py`는 `print` 대신 Python 표준 `logging`을 사용합니다.
2. **타입 힌트**: `.py`는 함수 인자와 반환 타입에 타입 힌트가 달려 있습니다.
3. **`MaskGITTrainer` 클래스**: 학습 로직을 클래스로 캡슐화합니다. 노트북은 함수와 메인 루프로 분리되어 있습니다.
4. **`_save_checkpoint` 메서드**: `MaskGITTrainer` 내부에 있으며 노트북 메인 루프의 `torch.save`와 동일한 동작을 합니다. 단, 노트북은 추가로 `config` 딕셔너리를 체크포인트에 포함합니다.

---

## 부록: 전체 텐서 흐름 요약

```
원본 이미지 [B, 3, 256, 256]
    ↓ VQGAN encode
토큰 [B, 16, 16] (값: 0~16383)
    ↓ mask_tokens (arccos schedule)
마스킹된 토큰 [B, 16, 16] (일부가 16384)
    ↓ view → [B, 256]
    ↓ tok_emb → [B, 256, 512]
    ↓ + pos_emb [256, 512] → [B, 256, 512]
    ↓ cls_emb(y) → cond [B, 512]
    ↓ TransformerBlock x 12 (x, cond)
        ↓ adaLN_modulation(cond) → 6 params [B, 512] each
        ↓ norm1 → modulate(gamma1, beta1) → attn → * alpha1 → + residual
        ↓ norm2 → modulate(gamma2, beta2) → ffn  → * alpha2 → + residual
    ↓ final_norm(x, cond) [B, 256, 512]
    ↓ head (weight tying) → [B, 256, 16385]
    ↓ CrossEntropyLoss (마스킹 위치만, ignore_index=-100)
Loss (스칼라)
```

```
생성 시:
전체 [MASK] [B, 16, 16]
    ↓ 32 스텝 반복:
        ↓ Halton 순서로 위치 선택
        ↓ CFG: 조건부 + 무조건부 동시 forward
        ↓ logits = (1+w)*cond - w*uncond
        ↓ softmax / temperature → multinomial sampling
        ↓ 선택된 위치에 토큰 배치
    ↓ 완성된 토큰 [B, 16, 16]
    ↓ VQGAN decode
생성된 이미지 [B, 3, 256, 256]
```
