# 노트북 1: VQGAN Fine-tuning 상세 설명

이 문서는 `V0/metal_defect_synthesis(PoCFinal).ipynb` 노트북의 VQGAN Fine-tuning 파트와 대응하는 `.py` 모듈들을 코드를 전혀 읽지 못하는 사람도 이해할 수 있도록 상세히 설명합니다.

---

## 1. 전처리 파이프라인

### 1.1 transform 변수

`transform`은 원본 금속 결함 이미지를 VQGAN 모델이 받아들일 수 있는 형태로 변환하는 일련의 처리 단계를 순서대로 묶어놓은 파이프라인입니다. `T.Compose`를 사용하여 4개의 변환 단계를 체인처럼 연결합니다.

**1단계: `T.Resize((256, 256))`**
- NEU-DET 원본 이미지는 200x200 픽셀입니다.
- VQGAN은 256x256 크기의 입력을 기대하므로 크기를 키웁니다.
- 보간법(interpolation)을 사용하여 56픽셀만큼 양방향으로 늘립니다.
- 입력: 200x200 PIL Image / 출력: 256x256 PIL Image

**2단계: `T.Grayscale(num_output_channels=3)`**
- NEU-DET 이미지는 흑백(1채널)이지만, VQGAN은 RGB(3채널) 입력을 기대합니다.
- `num_output_channels=3`을 지정하면 동일한 흑백 값을 R, G, B 세 채널에 복사합니다.
- 예: 픽셀 값 128이면 (128, 128, 128)이 됩니다.
- 입력: 256x256 1채널 / 출력: 256x256 3채널

**3단계: `T.ToTensor()`**
- PIL Image 객체를 PyTorch 텐서로 변환합니다.
- 동시에 정수 범위 [0, 255]를 실수 범위 [0.0, 1.0]으로 스케일링합니다.
- 축 순서도 변경됩니다: [높이, 너비, 채널] --> [채널, 높이, 너비]
- 입력: 256x256x3 PIL Image (값 0~255) / 출력: [3, 256, 256] 텐서 (값 0.0~1.0)

**4단계: `T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`**
- 각 채널에 대해 `(값 - 평균) / 표준편차` 공식을 적용합니다.
- 평균=0.5, 표준편차=0.5이므로: `(값 - 0.5) / 0.5 = 값 * 2 - 1`
- 결과적으로 [0, 1] 범위가 [-1, 1] 범위로 변환됩니다.
- 이 범위는 VQGAN이 학습될 때 사용된 범위와 동일합니다.
- 입력: [3, 256, 256] 텐서 (값 0.0~1.0) / 출력: [3, 256, 256] 텐서 (값 -1.0~1.0)

최종적으로 하나의 이미지는 `[3, 256, 256]` 형태의 텐서가 되며, 배치로 묶이면 `[B, 3, 256, 256]` (예: `[8, 3, 256, 256]`)이 됩니다.

### 1.2 denormalize 함수

`denormalize`는 모델이 출력한 텐서를 다시 사람이 볼 수 있는 이미지(numpy 배열)로 되돌리는 역변환 함수입니다. transform의 정규화 단계를 정확히 거꾸로 수행합니다.

**변수 및 단계 설명:**

- `tensor = tensor.clone()`: 원본 텐서를 수정하지 않기 위해 복사본을 만듭니다. clone()을 하지 않으면 원본 데이터가 변경되어 이후 연산에 영향을 줄 수 있습니다.

- `tensor = tensor * 0.5 + 0.5`: 정규화의 역연산입니다. 정규화 공식이 `(x - 0.5) / 0.5`였으므로, 역연산은 `x * 0.5 + 0.5`입니다. [-1, 1] 범위가 [0, 1] 범위로 돌아옵니다. 예: -1 * 0.5 + 0.5 = 0, 1 * 0.5 + 0.5 = 1.

- `tensor = tensor.clamp(0, 1)`: 혹시 모델 출력이 [-1, 1]을 약간 벗어난 경우를 대비하여 값을 0~1 범위로 강제 제한합니다. 예를 들어 1.02 같은 값이 나오면 1.0으로, -0.03이면 0.0으로 잘라냅니다.

- `if tensor.dim() == 4: tensor = tensor[0]`: 텐서가 배치 차원을 포함한 4차원(예: [1, 3, 256, 256])이면, 첫 번째 이미지만 선택하여 3차원([3, 256, 256])으로 만듭니다. dim()은 텐서의 차원 수를 반환합니다.

- `tensor.permute(1, 2, 0).cpu().numpy()`: PyTorch 텐서는 [채널, 높이, 너비] 순서이지만, matplotlib이나 일반 이미지 라이브러리는 [높이, 너비, 채널] 순서를 기대합니다. permute(1, 2, 0)으로 축 순서를 변경합니다. `.cpu()`는 GPU에 있는 텐서를 CPU 메모리로 옮기고, `.numpy()`는 PyTorch 텐서를 numpy 배열로 변환합니다.

- 입력: `[1, 3, 256, 256]` 또는 `[3, 256, 256]` 텐서 (값 -1~1)
- 출력: `[256, 256, 3]` numpy 배열 (값 0~1)

---

## 2. NEUDataset 클래스

PyTorch의 `Dataset` 클래스를 상속받아 NEU-DET 데이터셋을 모델 학습에 사용할 수 있는 형태로 감싸는 클래스입니다. Dataset을 상속받으면 반드시 `__len__`과 `__getitem__` 두 가지 메서드를 구현해야 합니다.

### 2.1 `__init__` 메서드 (초기화)

객체가 생성될 때 호출됩니다. 디스크에 있는 이미지 파일들의 경로를 모두 수집하고, 각 이미지에 클래스 라벨을 부여합니다.

**매개변수:**
- `root_dir` (str): NEU-DET 데이터가 저장된 최상위 폴더 경로. 예: `/content/drive/MyDrive/metal-defect/data/NEU-DET`
- `split` (str): `"train"` 또는 `"validation"`. 학습용인지 검증용인지 선택합니다.
- `transform` (Callable, 선택): 이미지에 적용할 전처리 함수. None이면 기본 전처리를 사용합니다.

**인스턴스 변수:**

- `self.image_paths` (List[str]): 모든 이미지 파일의 전체 경로를 저장하는 리스트. 예: `["/content/.../crazing/img001.jpg", "/content/.../crazing/img002.jpg", ...]`. 최종적으로 1800개(클래스당 300장 x 6개 클래스)의 경로가 들어갑니다.

- `self.labels` (List[int]): 각 이미지의 클래스 번호. image_paths와 동일한 인덱스를 가지며, 0~5 사이의 정수가 들어갑니다. 예: crazing=0, inclusion=1, patches=2, pitted_surface=3, rolled-in_scale=4, scratches=5.

- `self.class_names` (List[str]): 발견된 클래스 이름 목록. 예: `['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']`.

**폴더 탐색 알고리즘:**

1. `images_dir` 변수를 만듭니다: `root_dir/split/images` 경로를 조합합니다. 예: `NEU-DET/train/images`.
2. `os.path.exists(images_dir)`로 경로 존재 여부를 확인합니다. 없으면 에러를 발생시킵니다.
3. `sorted(os.listdir(images_dir))`로 images 폴더 안의 하위 폴더 목록을 알파벳 순으로 정렬하여 가져옵니다. sorted()를 사용하는 이유는, 운영체제마다 파일 목록 순서가 다를 수 있어서 항상 일관된 클래스 번호를 부여하기 위함입니다.
4. `enumerate()`로 각 폴더에 0부터 시작하는 인덱스(class_idx)를 부여합니다.
5. `os.path.isdir(class_dir)`로 실제 디렉토리인지 확인합니다 (숨김 파일 등을 제외).
6. 디렉토리라면 class_names에 이름을 추가하고, 그 안의 모든 파일 경로와 해당 class_idx를 각각 image_paths와 labels에 추가합니다.

**기본 transform 설정:**
transform이 None으로 전달되면, 위 1.1절에서 설명한 것과 동일한 4단계 전처리 파이프라인이 자동으로 생성됩니다 (Resize -> Grayscale -> ToTensor -> Normalize).

### 2.2 `__len__` 메서드

`len(dataset)`을 호출하면 자동으로 실행됩니다. `self.image_paths` 리스트의 길이, 즉 전체 이미지 수를 반환합니다. NEU-DET train 기준으로 1800을 반환합니다.

### 2.3 `__getitem__` 메서드

`dataset[0]`처럼 인덱스로 접근하면 자동으로 실행됩니다. DataLoader가 내부적으로 이 메서드를 반복 호출하여 배치를 구성합니다.

**동작 과정:**
1. `Image.open(self.image_paths[idx])`: 해당 인덱스의 이미지 파일을 PIL Image 객체로 엽니다. 이 시점에서는 아직 디스크에서 메모리로 로드되지 않을 수 있습니다(lazy loading).
2. `self.transform(img)`: 전처리 파이프라인을 적용합니다. 200x200 흑백 이미지가 [3, 256, 256] 텐서(값 -1~1)로 변환됩니다.
3. `self.labels[idx]`: 해당 이미지의 클래스 번호(int)를 가져옵니다.
4. `(img, label)` 튜플을 반환합니다.

- 입력: `idx` = 정수 인덱스 (예: 0)
- 출력: `([3, 256, 256] 텐서, int)` 튜플. 예: `(tensor([[[−0.12, ...], ...], ...]), 0)`

DataLoader는 여러 개의 `__getitem__` 결과를 모아서 배치를 만듭니다. batch_size=8이면 `[8, 3, 256, 256]` 텐서와 `[8]` 라벨 텐서의 튜플이 됩니다.

---

## 3. VQGAN 모델 구조

VQGAN(Vector Quantized Generative Adversarial Network)은 이미지를 이산적인(discrete) 코드로 압축했다가 다시 복원하는 모델입니다. taming-transformers 라이브러리의 `VQModel` 클래스를 사용합니다. 사전학습된 체크포인트 `vqgan_imagenet_f16_16384.ckpt`를 로드합니다 (약 1.4GB, ImageNet으로 학습된 가중치).

### 3.1 전체 흐름

이미지 --> Encoder --> quant_conv --> Quantize --> post_quant_conv --> Decoder --> 재구성 이미지

### 3.2 각 컴포넌트 상세

**Encoder (인코더)**
- 역할: 입력 이미지를 저차원 잠재 공간(latent space)으로 압축합니다.
- 입력: `[B, 3, 256, 256]` - 배치 B장의 256x256 RGB 이미지
- 출력: `[B, 256, 16, 16]` - 공간 해상도가 16배 축소(256/16=16), 채널은 256개로 증가
- 구조: 여러 겹의 합성곱(convolution) 레이어와 다운샘플링으로 구성됩니다. ResNet 블록과 어텐션 레이어를 포함합니다.
- 수학적 의미: 256x256 = 65,536개의 픽셀 정보를 16x16x256 = 65,536개의 잠재 특징으로 변환합니다. 공간적으로는 16배 압축되었지만, 채널 방향으로는 더 풍부한 표현을 가집니다.

**quant_conv (양자화 전 합성곱)**
- 역할: 인코더 출력의 채널 수를 코드북 임베딩 차원에 맞춥니다.
- 입력: `[B, 256, 16, 16]`
- 출력: `[B, 256, 16, 16]` (이 config에서는 임베딩 차원도 256이므로 크기 동일)
- 1x1 합성곱 레이어로 구현됩니다.

**quantize (벡터 양자화)**
- 역할: 연속적인 잠재 벡터를 코드북의 가장 가까운 이산 벡터로 매핑합니다.
- 코드북: 16,384개의 256차원 벡터가 저장된 사전(dictionary)입니다. 각 벡터는 하나의 "패턴"을 나타냅니다.
- 입력: `[B, 256, 16, 16]` 연속 잠재 텐서
- 출력 3가지:
  1. `z_q` `[B, 256, 16, 16]`: 양자화된 텐서. 각 16x16 위치의 256차원 벡터가 코드북에서 가장 가까운 벡터로 교체됩니다.
  2. `codebook_loss` (스칼라): 양자화 전후의 벡터 거리를 측정하는 손실. commitment loss와 embedding loss의 합입니다.
  3. `info` (튜플): perplexity, min_encodings, min_encoding_indices 등 디버깅 정보.
- 동작 알고리즘: 16x16 = 256개 위치 각각에서, 입력 벡터와 코드북의 16,384개 벡터 사이의 유클리드 거리를 계산하고, 가장 가까운 코드북 벡터의 인덱스를 선택합니다. straight-through estimator를 사용하여 양자화의 불연속성을 우회합니다(forward에서는 양자화된 값을 사용하지만, backward에서는 그래디언트를 그대로 통과시킵니다).

**post_quant_conv (양자화 후 합성곱)**
- 역할: 양자화된 잠재 벡터를 디코더 입력에 맞게 변환합니다.
- 입력: `[B, 256, 16, 16]`
- 출력: `[B, 256, 16, 16]`
- quant_conv와 대칭되는 1x1 합성곱 레이어입니다.

**Decoder (디코더)**
- 역할: 잠재 공간의 표현을 다시 원본 해상도 이미지로 복원합니다.
- 입력: `[B, 256, 16, 16]`
- 출력: `[B, 3, 256, 256]` - 재구성된 RGB 이미지
- 구조: 인코더의 역순으로, 업샘플링과 합성곱 레이어로 구성됩니다. ResNet 블록과 어텐션 레이어를 포함합니다.
- `conv_out`: 디코더의 최종 합성곱 레이어. 이 레이어의 가중치(`model.decoder.conv_out.weight`)가 adaptive weight 계산의 기준점으로 사용됩니다.

**Discriminator (판별기)**
- 역할: 입력 이미지가 진짜(원본)인지 가짜(재구성)인지 판별합니다. Generator(VQGAN)와 적대적으로 학습합니다.
- 구조: `NLayerDiscriminator(input_nc=3, n_layers=2, ndf=64)` - 3채널 입력, 2개 레이어, 기본 필터 64개.
- `weights_init` 함수를 apply하여 가중치를 초기화합니다.
- 체크포인트에서 `loss.discriminator.`로 시작하는 키를 추출하여 가중치를 로드합니다.
- 입력: `[B, 3, 256, 256]` / 출력: `[B, 1, H', W']` (공간적 logits 맵)

**Perceptual Loss (LPIPS)**
- 역할: 두 이미지 사이의 "지각적 유사도"를 측정합니다. 단순한 픽셀 차이가 아니라, VGG 같은 사전학습 네트워크의 중간 특징을 비교합니다.
- 평가 모드(`.eval()`)로 고정하고, 모든 파라미터의 `requires_grad`를 False로 설정하여 학습하지 않습니다.
- 입력: 원본 이미지 `[B, 3, 256, 256]`, 재구성 이미지 `[B, 3, 256, 256]`
- 출력: 지각적 손실 값 (스칼라 텐서)

---

## 4. 손실 함수들

### 4.1 adopt_weight 함수

```
def adopt_weight(weight, global_step, threshold=0, value=0.):
```

**목적:** 학습 초반에는 GAN loss를 비활성화하여 모델이 먼저 기본적인 재구성(reconstruction)을 학습하도록 합니다. 충분한 스텝이 지난 후에야 GAN loss를 활성화합니다.

**매개변수:**
- `weight` (float): 활성화 시 사용할 가중치. 보통 `disc_factor` 값(1.0)이 들어옵니다.
- `global_step` (int): 현재까지의 총 학습 스텝 수. 배치 하나를 처리할 때마다 1씩 증가합니다.
- `threshold` (int): GAN loss를 활성화할 스텝. 기본값 0이면 처음부터 활성화. 이 프로젝트에서는 `disc_start=1000`을 사용합니다.
- `value` (float): threshold 이전에 사용할 값. 기본값 0.0.

**동작:**
- `global_step < threshold`이면 (예: 현재 500스텝, threshold 1000): weight를 value(0.0)로 덮어씁니다. 결과적으로 GAN loss에 0이 곱해져서 무효화됩니다.
- `global_step >= threshold`이면: weight를 그대로 반환합니다. GAN loss가 활성화됩니다.

**존재 이유:** GAN 학습은 불안정합니다. 재구성이 아직 엉망인 상태에서 Discriminator까지 동시에 학습하면 모델이 수렴하지 못합니다. 먼저 재구성 품질을 어느 정도 확보한 후에 GAN loss를 도입하는 것이 안정적입니다.

### 4.2 hinge_d_loss 함수

```
def hinge_d_loss(logits_real, logits_fake):
```

**목적:** Discriminator를 학습시키기 위한 Hinge Loss를 계산합니다.

**매개변수:**
- `logits_real` (Tensor): Discriminator가 진짜 이미지에 대해 출력한 값. 양수가 클수록 "진짜"로 판단한 것입니다.
- `logits_fake` (Tensor): Discriminator가 가짜(재구성) 이미지에 대해 출력한 값. 음수가 클수록 "가짜"로 판단한 것입니다.

**변수 및 알고리즘:**

- `loss_real = torch.mean(F.relu(1. - logits_real))`:
  - `1. - logits_real`: logits_real이 1보다 크면 음수, 작으면 양수가 됩니다.
  - `F.relu(...)`: 음수는 0으로 잘라냅니다. 즉, logits_real이 1보다 크면 손실이 0이고(이미 잘 판별), 1보다 작으면 페널티를 줍니다.
  - `torch.mean(...)`: 전체 배치의 평균을 구합니다.
  - 의미: Discriminator가 진짜 이미지에 대해 1 이상의 점수를 주도록 유도합니다.

- `loss_fake = torch.mean(F.relu(1. + logits_fake))`:
  - `1. + logits_fake`: logits_fake가 -1보다 작으면 음수, 크면 양수가 됩니다.
  - `F.relu(...)`: 음수는 0으로 잘라냅니다. 즉, logits_fake가 -1보다 작으면 손실이 0(이미 잘 판별), -1보다 크면 페널티를 줍니다.
  - 의미: Discriminator가 가짜 이미지에 대해 -1 이하의 점수를 주도록 유도합니다.

- `d_loss = 0.5 * (loss_real + loss_fake)`: 두 손실의 평균을 최종 Discriminator 손실로 사용합니다.

**Hinge Loss의 특징:** 일정 마진(여기서는 1) 이상으로 잘 구분하면 더 이상 손실을 주지 않습니다. 이로 인해 Discriminator가 지나치게 강해지는 것을 방지합니다. Wasserstein GAN 계열보다 학습이 안정적인 경우가 많습니다.

### 4.3 calculate_adaptive_weight 함수

이 함수는 전체 코드에서 가장 핵심적인 함수입니다. Reconstruction loss와 GAN loss가 모델에 미치는 영향력을 자동으로 균형 맞춥니다.

```
def calculate_adaptive_weight(nll_loss, g_loss, last_layer, disc_weight=1.0):
```

**매개변수:**
- `nll_loss` (Tensor): 재구성 손실. L1 loss + perceptual loss의 평균. 스칼라 텐서입니다.
- `g_loss` (Tensor): Generator의 GAN 손실. `-mean(logits_fake)`. 스칼라 텐서입니다.
- `last_layer` (Tensor): `model.decoder.conv_out.weight` - 디코더의 마지막 합성곱 레이어의 가중치 텐서. 이 레이어를 기준점으로 선택하는 이유는, 최종 출력에 가장 직접적인 영향을 미치는 레이어이기 때문입니다.
- `disc_weight` (float): 기본 discriminator 가중치. config에서 0.8로 설정합니다.

**변수 및 알고리즘 (가장 중요):**

1. `nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]`:
   - `torch.autograd.grad`: 역전파를 수행하지 않고, 특정 텐서에 대한 특정 손실의 그래디언트만 계산합니다.
   - nll_loss가 last_layer의 가중치에 미치는 그래디언트를 구합니다.
   - `retain_graph=True`: 계산 그래프를 유지합니다. 이후 g_loss에 대해서도 그래디언트를 계산해야 하므로 그래프를 지우면 안 됩니다.
   - `[0]`: grad 함수는 튜플을 반환하므로 첫 번째 원소를 가져옵니다.
   - 결과: last_layer와 같은 형태의 그래디언트 텐서.

2. `g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]`:
   - 같은 방식으로 g_loss의 그래디언트를 구합니다.

3. `d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)`:
   - `torch.norm()`: L2 노름(벡터의 크기)을 계산합니다. 모든 원소의 제곱합의 제곱근입니다.
   - 분자: nll_loss가 last_layer에 미치는 그래디언트의 크기.
   - 분모: g_loss가 last_layer에 미치는 그래디언트의 크기 + epsilon(1e-4, 0으로 나누기 방지).
   - **핵심 아이디어:**
     - nll_grads가 크고 g_grads가 작으면: d_weight가 커집니다. 즉, GAN loss의 영향력을 키워서 균형을 맞춥니다.
     - nll_grads가 작고 g_grads가 크면: d_weight가 작아집니다. GAN loss가 이미 충분히 강하므로 줄입니다.
   - 이렇게 하면 두 손실이 모델의 마지막 레이어에 미치는 영향력이 비슷해집니다.

4. `d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()`:
   - `torch.clamp(0.0, 1e4)`: 극단적인 값을 방지합니다. 최소 0, 최대 10,000으로 제한합니다.
   - `.detach()`: 이 텐서를 계산 그래프에서 분리합니다. d_weight 자체는 학습 대상이 아니라 하이퍼파라미터처럼 사용되므로, 이 값에 대해 역전파할 필요가 없습니다.

5. `d_weight = d_weight * disc_weight`:
   - config에서 설정한 기본 가중치(0.8)를 곱합니다. 전체적인 GAN loss의 세기를 추가로 조절하는 역할입니다.

- 반환값: `d_weight` (스칼라 텐서) - GAN loss에 곱해줄 adaptive 가중치.

---

## 5. 학습 루프 상세

### 5.1 하이퍼파라미터

| 변수 | 값 | 의미 |
|------|------|------|
| `num_epochs` | 50 | 전체 데이터를 50번 반복 학습 |
| `save_every` | 5 | 5 에폭마다 체크포인트 저장 |
| `codebook_weight` | 1.0 | codebook loss에 곱하는 가중치 |
| `perceptual_weight` | 1.0 | perceptual loss에 곱하는 가중치 |
| `disc_weight` | 0.8 | adaptive weight 계산 시 기본 가중치 |
| `disc_factor` | 1.0 | GAN loss 전체에 곱하는 factor |
| `disc_start` | 1000 | 1000 스텝 이후부터 GAN loss 활성화 |
| `batch_size` | 8 | 한 번에 8장 처리 |

NEU-DET 1800장, 배치 8개이면 1 에폭 = 225 스텝. disc_start=1000이면 약 4.4 에폭 후에 GAN loss가 활성화됩니다.

### 5.2 Optimizer 설정

**Generator (VQModel) Optimizer:**
```
optimizer_g = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-7},
    {'params': model.quant_conv.parameters(), 'lr': 1e-7},
    {'params': model.quantize.parameters(), 'lr': 4.5e-6},
    {'params': model.post_quant_conv.parameters(), 'lr': 4.5e-6},
    {'params': model.decoder.parameters(), 'lr': 4.5e-6},
], betas=(0.5, 0.9))
```

- 파라미터 그룹별 학습률 차등 적용:
  - Encoder, quant_conv: `1e-7` (매우 낮음). 사전학습된 인코더의 특징 추출 능력을 최대한 보존합니다. ImageNet에서 학습한 일반적인 시각적 특징은 금속 결함에도 유용하므로 크게 바꾸지 않습니다.
  - Quantize, post_quant_conv, Decoder: `4.5e-6` (상대적으로 높음). 양자화 코드북과 디코더는 금속 결함 도메인에 맞게 더 적극적으로 적응시킵니다.

- `betas=(0.5, 0.9)`:
  - Adam 옵티마이저의 모멘텀 관련 하이퍼파라미터입니다.
  - `beta1=0.5`: 1차 모멘텀(그래디언트의 이동 평균)의 감쇠 계수. 일반적으로 0.9를 사용하지만, GAN 학습에서는 0.5를 사용하는 것이 관행입니다. 더 빠르게 최신 그래디언트에 적응합니다.
  - `beta2=0.9`: 2차 모멘텀(그래디언트 제곱의 이동 평균)의 감쇠 계수.

**Discriminator Optimizer:**
```
optimizer_d = optim.Adam(discriminator.parameters(), lr=4.5e-6, betas=(0.5, 0.9))
```
- 모든 파라미터에 동일한 학습률 4.5e-6을 적용합니다.
- betas는 Generator와 동일합니다.

### 5.3 Forward Pass (순전파)

배치 하나(`[8, 3, 256, 256]`)가 모델을 통과하는 과정입니다.

| 단계 | 연산 | 입력 형태 | 출력 형태 |
|------|------|-----------|-----------|
| 1 | `model.encoder(images)` | `[8, 3, 256, 256]` | `[8, 256, 16, 16]` |
| 2 | `model.quant_conv(z)` | `[8, 256, 16, 16]` | `[8, 256, 16, 16]` |
| 3 | `model.quantize(z)` | `[8, 256, 16, 16]` | `[8, 256, 16, 16]` + codebook_loss |
| 4 | `model.post_quant_conv(z_q)` | `[8, 256, 16, 16]` | `[8, 256, 16, 16]` |
| 5 | `model.decoder(z_q)` | `[8, 256, 16, 16]` | `[8, 3, 256, 256]` |

- 단계 1에서 256x256 이미지가 16x16로 16배 압축됩니다 (공간 해상도 기준).
- 단계 3에서 연속 벡터가 이산 코드북 벡터로 교체됩니다. 이 과정에서 정보 손실이 발생하며, codebook_loss가 이 손실을 측정합니다.
- 단계 5에서 16x16을 다시 256x256으로 복원합니다.

### 5.4 Generator 업데이트

Generator의 총 손실은 세 가지 항의 합입니다.

**1. NLL Loss (nll_loss) = L1 + Perceptual:**

```
rec_loss = torch.abs(images.contiguous() - reconstructions.contiguous())
```
- `torch.abs(원본 - 재구성)`: 픽셀별 절대 차이(L1 loss). 각 픽셀에서 원본과 재구성 값의 차이의 절대값을 구합니다.
- `.contiguous()`: 텐서의 메모리 레이아웃을 연속적으로 만듭니다. 일부 연산은 메모리가 연속적이지 않으면 에러가 발생합니다.
- 결과 형태: `[8, 3, 256, 256]` - 각 픽셀의 L1 오차.

```
p_loss = perceptual_loss(images.contiguous(), reconstructions.contiguous())
rec_loss = rec_loss + perceptual_weight * p_loss
```
- LPIPS 네트워크가 두 이미지의 지각적 차이를 계산합니다. 사람의 눈에 보이는 차이에 더 가까운 메트릭입니다.
- `perceptual_weight=1.0`이므로 L1과 동일한 비중으로 더합니다.

```
nll_loss = torch.mean(rec_loss)
```
- 전체 배치, 모든 채널, 모든 픽셀에 걸쳐 평균을 구합니다. 스칼라 값이 됩니다.

**2. GAN Loss (g_loss):**

```
logits_fake = discriminator(reconstructions.contiguous())
g_loss = -torch.mean(logits_fake)
```
- Discriminator에 재구성 이미지를 넣어서 "진짜 점수"를 얻습니다.
- Generator 입장에서는 이 점수가 높을수록 좋습니다 (Discriminator를 속이는 데 성공한 것).
- `-mean(logits_fake)`: 점수가 높으면 loss가 낮아지도록 부호를 뒤집습니다.

**3. Total Generator Loss:**

```
loss_g = nll_loss + d_weight * current_disc_factor * g_loss + codebook_weight * codebook_loss.mean()
```
- `nll_loss`: 재구성 품질. 원본과 얼마나 비슷한지.
- `d_weight * current_disc_factor * g_loss`: GAN 품질. 얼마나 진짜같은지. d_weight는 adaptive weight, current_disc_factor는 adopt_weight의 결과(0 또는 1).
- `codebook_weight * codebook_loss.mean()`: 코드북 학습. 양자화 전후 벡터가 얼마나 가까운지.

**업데이트:**
```
optimizer_g.zero_grad()  # 이전 그래디언트 초기화
loss_g.backward()        # 역전파로 그래디언트 계산
optimizer_g.step()       # 가중치 업데이트
```

### 5.5 Discriminator 업데이트

Discriminator는 `current_disc_factor > 0`일 때만(즉, disc_start 스텝 이후에만) 학습됩니다.

```
logits_real = discriminator(images.contiguous().detach())
logits_fake = discriminator(reconstructions.contiguous().detach())
```
- `.detach()`: 매우 중요한 호출입니다. 이 텐서에서 계산 그래프를 끊어서, Discriminator의 역전파가 Generator까지 전파되지 않도록 합니다. Discriminator 업데이트 시에는 Generator의 가중치가 변하면 안 되기 때문입니다.
- `logits_real`: 진짜 이미지에 대한 점수. 높을수록 "진짜"로 판단.
- `logits_fake`: 재구성 이미지에 대한 점수. 낮을수록 "가짜"로 판단.

```
d_loss = current_disc_factor * hinge_d_loss(logits_real, logits_fake)
```
- hinge_d_loss로 Discriminator 손실을 계산합니다 (4.2절 참조).
- current_disc_factor를 곱합니다 (이 시점에서 1.0).

```
optimizer_d.zero_grad()
d_loss.backward()
optimizer_d.step()
```
- Discriminator만 업데이트합니다. Generator 가중치는 영향받지 않습니다.

### 5.6 체크포인트 저장

```
torch.save({
    'epoch': epoch + 1,
    'global_step': global_step,
    'vqmodel': model.state_dict(),
    'discriminator': discriminator.state_dict(),
    'optimizer_g': optimizer_g.state_dict(),
    'optimizer_d': optimizer_d.state_dict(),
}, save_path)
```

저장하는 항목:
- `epoch`: 현재 에폭 번호. 학습 재개 시 어디서부터 시작할지 알기 위함.
- `global_step`: 누적 스텝 수. adopt_weight의 threshold 비교에 필요.
- `vqmodel`: VQGAN 모델의 모든 가중치.
- `discriminator`: Discriminator의 모든 가중치.
- `optimizer_g`, `optimizer_d`: 옵티마이저 상태(모멘텀 등). 학습 재개 시 옵티마이저가 이전 상태를 이어받아야 안정적입니다.

---

## 6. 평가 지표

### 6.1 PSNR (Peak Signal-to-Noise Ratio)

```
psnr_value = psnr(original, reconstructed, data_range=1.0)
```

- 의미: 신호 대 잡음비의 최댓값. 원본과 재구성 이미지 사이의 차이를 데시벨(dB) 단위로 측정합니다.
- 계산 방식: `PSNR = 10 * log10(MAX^2 / MSE)`. MAX는 픽셀 값의 최대 범위(여기서 1.0), MSE는 평균 제곱 오차.
- 해석:
  - 30dB 이상: 좋은 품질
  - 40dB 이상: 매우 좋은 품질
  - 높을수록 원본에 가깝습니다.
- `data_range=1.0`: 픽셀 값의 범위가 [0, 1]임을 명시합니다.
- 입력: `[H, W, C]` numpy 배열 두 개 (값 0~1)
- 출력: float (dB 단위)

### 6.2 SSIM (Structural Similarity Index)

```
ssim_value = ssim(original, reconstructed, data_range=1.0, channel_axis=2)
```

- 의미: 구조적 유사도. 밝기, 대비, 구조 세 가지 측면에서 두 이미지를 비교합니다. PSNR보다 사람의 시각적 인식에 더 가까운 메트릭입니다.
- 계산 방식: 슬라이딩 윈도우로 국소 영역의 평균, 분산, 공분산을 계산하여 유사도를 구합니다.
- 해석:
  - 0~1 범위
  - 1에 가까울수록 원본과 구조적으로 유사
  - 0.9 이상이면 좋은 품질
- `channel_axis=2`: 채널 축이 3번째(인덱스 2)임을 명시합니다.
- 입력: `[H, W, C]` numpy 배열 두 개
- 출력: float (0~1)

### 6.3 Edge IoU (Edge Intersection over Union)

```
orig_gray = (original.mean(axis=2) * 255).astype(np.uint8)
edges_orig = cv2.Canny(orig_gray, 50, 150)
```

- 의미: 엣지(경계선) 보존 정도를 측정합니다. 금속 결함 검출에서 결함의 경계선이 잘 보존되는지가 매우 중요합니다.
- 계산 과정:
  1. RGB 이미지를 그레이스케일로 변환: 3채널 평균 후 [0,255] 정수로 변환.
  2. Canny 엣지 검출: 낮은 임계값 50, 높은 임계값 150으로 엣지 맵 생성. 결과는 0(엣지 아님) 또는 255(엣지)인 이진 이미지.
  3. 원본 엣지와 재구성 엣지의 교집합(AND) / 합집합(OR)을 구합니다.
  4. `intersection / (union + 1e-6)`: IoU 계산. 1e-6은 엣지가 전혀 없는 경우 0으로 나누기 방지.
- 해석:
  - 0~1 범위
  - 1에 가까울수록 엣지가 잘 보존됨
  - 결함의 경계가 흐려지지 않았음을 의미
- 입력: `[H, W, C]` numpy 배열 두 개 (값 0~1)
- 출력: float (0~1)

---

## 7. .py 모듈 대조

노트북의 코드를 재사용 가능한 `.py` 모듈로 리팩토링한 결과입니다. 각 모듈별로 노트북과의 차이점을 상세히 비교합니다.

### 7.1 vqgan_trainer.py (학습 루프)

**파일 경로:** `src/metal_defect_synthesis/training/vqgan_trainer.py`

**구조적 차이:**
- 노트북: 모든 코드가 하나의 셀(Cell 55)에 인라인으로 작성됨. 전역 변수(`model`, `discriminator`, `dataloader` 등)를 직접 참조.
- .py: `VQGANTrainer` 클래스로 캡슐화됨. 모든 의존성을 `__init__`의 매개변수로 주입받음.

**손실 함수 비교:**

- `adopt_weight`: 노트북과 .py 모두 동일한 로직. .py에서는 타입 힌트(`weight: float`, `global_step: int` 등)가 추가됨.
- `hinge_d_loss`: 동일한 로직. .py에서는 입출력 타입 힌트(`torch.Tensor`)가 추가됨.
- `calculate_adaptive_weight`: 동일한 로직. .py에서는 상세한 docstring과 Parameters 섹션이 추가됨.

**학습 루프 비교:**

| 항목 | 노트북 (Cell 55) | vqgan_trainer.py |
|------|-----------------|------------------|
| 전체 구조 | for 루프가 글로벌 스코프에 있음 | `train()` 메서드와 `train_epoch()` 메서드로 분리 |
| 변수 관리 | 전역 변수 `global_step`, `train_logs` | 인스턴스 변수 `self.global_step`, `self.train_logs` |
| 모델 참조 | 직접 `model`, `discriminator` 사용 | `self.model`, `self.discriminator` 사용 |
| 체크포인트 저장 | 인라인 torch.save | `_save_checkpoint()` 메서드로 분리 |
| 로깅 | print문 + tqdm | logging 모듈 + tqdm |
| pbar 표시 항목 | loss, nll, cb, d_w, D 5가지 | loss, nll, d_w 3가지 (cb와 D는 생략) |
| 에폭 완료 출력 | 6줄의 상세 print문 | logger.info 1줄 |
| learning_rate | 전역 변수에서 참조 | `__init__`에서 optimizer에 직접 설정 |

**Optimizer 설정:**
- 노트북과 .py 모두 동일한 파라미터 그룹 구성:
  - encoder, quant_conv: lr=1e-7
  - quantize, post_quant_conv, decoder: lr=4.5e-6
  - betas=(0.5, 0.9)
- Discriminator optimizer도 동일: lr=4.5e-6, betas=(0.5, 0.9)

**Forward Pass:**
- 노트북과 .py 모두 동일한 5단계: encoder -> quant_conv -> quantize -> post_quant_conv -> decoder
- Generator 업데이트 공식 동일: `nll_loss + d_weight * disc_factor * g_loss + codebook_weight * codebook_loss.mean()`
- Discriminator 업데이트도 동일: `.detach()` 적용, hinge_d_loss 사용

**노트북에만 있는 것:**
- Cell 52-53: 초기 간단한 MSE 기반 학습 코드 (실제 사용되지 않고, Cell 55에서 정식 학습 수행)
- Cell 56: 학습 곡선 시각화 (matplotlib)
- Cell 57: Fine-tuning 전후 비교 시각화
- Cell 58: 원본 모델 vs Fine-tuned 모델 정량 비교

### 7.2 dataset.py (데이터셋)

**파일 경로:** `src/metal_defect_synthesis/data/dataset.py`

**NEUDataset 클래스 비교:**

| 항목 | 노트북 (Cell 47) | dataset.py |
|------|-----------------|------------|
| 타입 힌트 | 없음 (`def __init__(self, root_dir, split="train", transform=None)`) | 있음 (`root_dir: str, split: str = "train", transform: Optional[Callable] = None`) |
| 반환 타입 | 명시 안 됨 | `Tuple[torch.Tensor, int]` |
| 데이터셋 정보 출력 | `print()` 사용 | `logger.info()` 사용 |
| import 방식 | 셀 상단에서 전역 import | 모듈 내 자체 import |

- 핵심 로직(폴더 탐색, 이미지 로드, transform 적용)은 완전히 동일합니다.
- 기본 transform도 동일합니다: Resize(256) -> Grayscale(3) -> ToTensor -> Normalize([0.5]*3, [0.5]*3).

**TokenDataset 클래스 (dataset.py에만 존재):**
- 노트북에는 직접 대응하는 코드가 없습니다 (MaskGIT 학습용으로 별도 추가됨).
- VQGAN이 생성한 토큰 `[N, 16, 16]`과 라벨 `[N]`을 받아서, MaskGIT이 기대하는 `{"code": [16, 16], "y": int}` 딕셔너리 형태로 반환합니다.

### 7.3 image.py (이미지 유틸리티)

**파일 경로:** `src/metal_defect_synthesis/utils/image.py`

**denormalize 함수 비교:**

| 항목 | 노트북 (Cell 26) | image.py |
|------|-----------------|----------|
| 역정규화 공식 | `tensor * 0.5 + 0.5` | `tensor * 0.5 + 0.5` (동일) |
| 반환 타입 명시 | 없음 | `-> np.ndarray` |

- 노트북 Cell 57에는 약간 다른 버전의 denormalize가 있습니다: `(tensor + 1) / 2` 사용. 수학적으로 `tensor * 0.5 + 0.5`와 동일한 결과입니다.

**image.py에만 있는 함수들:**

1. `preprocess_image(pil_image, image_size=256, device="cuda") -> torch.Tensor`:
   - 노트북에서는 transform + unsqueeze(0) + to(device)를 매번 수동으로 호출했지만, 이 함수는 이를 하나로 통합합니다.
   - 추가로 이미지 모드가 RGB가 아닌 경우 변환하는 안전 로직이 있습니다.
   - 입력: PIL Image / 출력: `[1, 3, 256, 256]` 텐서 (GPU)

2. `tensor_to_pil(tensor) -> Image.Image`:
   - 텐서를 PIL Image로 직접 변환합니다. denormalize와 비슷하지만 numpy 대신 PIL Image를 반환합니다.
   - 0~1 값을 0~255 정수로 변환하고 `Image.fromarray()`를 호출합니다.

3. `visualize_mask_on_image(pil_image, mask_indices, ...)`:
   - MaskGIT 마스킹 시각화용. 16x16 잠재 공간의 특정 위치를 원본 이미지 위에 색칠하여 보여줍니다.
   - `mask_indices`: 마스킹할 위치의 1D 인덱스 리스트. 예: [85, 86, 101, 102]는 16x16 그리드에서 (5,5), (5,6), (6,5), (6,6) 위치.
   - `np.kron`: 크로네커 곱을 사용하여 16x16 마스크를 256x256으로 업스케일합니다. 각 16x16 셀이 16x16 픽셀 블록으로 확장됩니다.

4. `get_mask_preset(preset_name) -> List[int]`:
   - 사전 정의된 마스크 패턴을 반환합니다: center_small(6x6), center_large(8x8), top_left(6x6), bottom_right(6x6).
   - 인덱스 계산: `y * 16 + x`로 2D 좌표를 1D 인덱스로 변환합니다.

### 7.4 metrics.py (평가 지표)

**파일 경로:** `src/metal_defect_synthesis/utils/metrics.py`

**compute_metrics 함수 vs 노트북 Cell 45/58:**

| 항목 | 노트북 | metrics.py |
|------|--------|-----------|
| 함수 구성 | 인라인 코드 (함수 아님) | `compute_metrics()` 단일 함수 |
| 반환 형태 | 개별 변수 (mse, psnr_value, ssim_value) | `Dict[str, float]` ({"psnr", "ssim", "edge_iou"}) |
| MSE 포함 여부 | Cell 45에서 별도 계산 | 미포함 (PSNR이 MSE를 내포하므로) |
| 입력 클리핑 | Cell 58에서 `np.clip` | 함수 시작부에서 `np.clip` |
| Edge IoU | Cell 58에서 인라인 | 함수 내 통합 |

- PSNR, SSIM 계산은 동일한 라이브러리(`skimage.metrics`)와 동일한 매개변수(`data_range=1.0`, `channel_axis=2`)를 사용합니다.
- Edge IoU 계산도 동일합니다: 그레이스케일 변환 -> Canny(50, 150) -> AND/OR 비율.
- 노트북 Cell 58에서는 배치 전체를 순회하며 Before/After를 비교하지만, metrics.py의 compute_metrics는 이미지 한 쌍에 대해서만 계산합니다. 배치 처리는 호출하는 측에서 루프를 돌아야 합니다.

---

## 부록: 전체 학습 흐름 요약

1. **데이터 준비**: NEUDataset이 NEU-DET 폴더를 탐색하여 1800장의 이미지 경로와 라벨을 수집합니다.
2. **전처리**: DataLoader가 배치 단위(8장)로 이미지를 로드하고, transform으로 [8, 3, 256, 256] 텐서를 만듭니다.
3. **모델 로드**: 사전학습된 VQGAN(ImageNet), Discriminator, LPIPS를 체크포인트에서 로드합니다.
4. **학습 (50 에폭)**:
   - 매 배치마다 Forward Pass (Encoder -> Quantize -> Decoder)
   - Generator 손실 계산: L1 + Perceptual + Adaptive GAN + Codebook
   - Generator 가중치 업데이트
   - (1000스텝 이후) Discriminator 손실 계산 및 업데이트
   - 5 에폭마다 체크포인트 저장
5. **평가**: PSNR, SSIM, Edge IoU로 재구성 품질을 정량 평가합니다.
