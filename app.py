"""
Metal Defect Synthesizer - HuggingFace Spaces용 Gradio 앱
"""

import os
import sys
import subprocess

# ============================================
# 0. Gradio 5.x + Python 3.13 버그 패치
# gradio_client/utils.py의 get_type()에서 schema가 bool일 때 크래시 방지
# ============================================
try:
    import gradio_client.utils as _gcu
    _orig_inner = _gcu._json_schema_to_python_type
    def _patched_inner(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_inner(schema, defs)
    _gcu._json_schema_to_python_type = _patched_inner

    _orig_outer = _gcu.json_schema_to_python_type
    def _patched_outer(schema):
        try:
            return _orig_outer(schema)
        except Exception:
            return "Any"
    _gcu.json_schema_to_python_type = _patched_outer
except Exception:
    pass

# ============================================
# 1. LlamaGen 설치 (VQGAN 코드 의존성)
# ============================================
if not os.path.exists("LlamaGen"):
    subprocess.run(["git", "clone", "https://github.com/FoundationVision/LlamaGen.git"], check=True)

sys.path.insert(0, "LlamaGen")

# ============================================
# 2. 체크포인트 다운로드 (HuggingFace Hub)
# ============================================
from huggingface_hub import hf_hub_download

REPO_ID = "Yumi-Park996/metal-defect-checkpoints"
os.makedirs("checkpoints", exist_ok=True)

VQGAN_CKPT = hf_hub_download(repo_id=REPO_ID, filename="vqgan_finetune_up_epoch50.pt", cache_dir="checkpoints")
MASKGIT_CKPT = hf_hub_download(repo_id=REPO_ID, filename="maskgit_epoch100.pt", cache_dir="checkpoints")

# ============================================
# 3. 모델 임포트 및 로드
# ============================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- VQGAN 로드 ---
from tokenizer.tokenizer_image.vq_model import VQ_models

CODEBOOK_SIZE = 16384
CODEBOOK_DIM = 8
LATENT_SIZE = 16
IMAGE_SIZE = 256
SEQ_LEN = 256
MASK_TOKEN_ID = 16384
NUM_CLASSES = 6

vqgan = VQ_models["VQ-16"](codebook_size=CODEBOOK_SIZE, codebook_embed_dim=CODEBOOK_DIM)
ckpt = torch.load(VQGAN_CKPT, map_location=device, weights_only=False)
if "vqmodel" in ckpt:
    vqgan.load_state_dict(ckpt["vqmodel"], strict=True)
elif "state_dict" in ckpt:
    vqgan.load_state_dict(ckpt["state_dict"], strict=True)
else:
    vqgan.load_state_dict(ckpt, strict=True)
vqgan = vqgan.to(device).eval()
for p in vqgan.parameters():
    p.requires_grad = False
print("VQGAN loaded")

# --- MaskGIT 모델 정의 (src에서 임포트) ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.metal_defect_synthesis.models.maskgit import MaskGITTransformer

ckpt_mg = torch.load(MASKGIT_CKPT, map_location=device, weights_only=False)
model_config = ckpt_mg["config"].copy()
model_config.pop("model_size", None)
maskgit = MaskGITTransformer(**model_config).to(device)
maskgit.load_state_dict(ckpt_mg["model_state_dict"])
maskgit.eval()
print("MaskGIT loaded")

# ============================================
# 4. 핵심 함수들
# ============================================

@torch.no_grad()
def encode_to_tokens(images, vqgan_model):
    _, _, (_, _, indices) = vqgan_model.encode(images)
    return indices.view(images.shape[0], LATENT_SIZE, LATENT_SIZE)

@torch.no_grad()
def decode_from_tokens(tokens, vqgan_model):
    batch_size = tokens.shape[0]
    tokens_flat = tokens.view(-1)
    tokens_flat = torch.clamp(tokens_flat, 0, CODEBOOK_SIZE - 1)
    embed_dim = vqgan_model.quantize.embedding.weight.shape[1]
    return vqgan_model.decode_code(tokens_flat, shape=(batch_size, embed_dim, LATENT_SIZE, LATENT_SIZE))

def cosine_schedule(t):
    return torch.cos(t * torch.pi / 2)

@torch.no_grad()
def inpaint_image(original_image, mask_region, target_class, num_steps=8, temperature=1.0):
    if original_image.dim() == 3:
        original_image = original_image.unsqueeze(0)
    original_image = original_image.to(device)

    if isinstance(mask_region, list):
        mask_region = torch.tensor(mask_region, device=device)

    num_mask_tokens = len(mask_region)
    original_tokens = encode_to_tokens(original_image, vqgan)
    tokens = original_tokens.clone().view(1, -1)
    tokens[:, mask_region] = MASK_TOKEN_ID

    labels = torch.tensor([target_class], device=device)
    drop_label = torch.zeros(1, dtype=torch.bool, device=device)

    for step in range(num_steps):
        is_masked = (tokens == MASK_TOKEN_ID)
        num_masked = is_masked.sum().item()
        if num_masked == 0:
            break

        t = torch.tensor(step / num_steps, device=device)
        ratio = cosine_schedule(t)
        num_to_keep_masked = (ratio * num_mask_tokens).long().item()
        num_to_unmask = max(1, num_masked - num_to_keep_masked)

        logits = maskgit(tokens, labels, drop_label)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        max_probs, predicted_tokens = probs.max(dim=-1)
        max_probs = max_probs * is_masked.float()
        _, top_k = max_probs.view(-1).topk(num_to_unmask)
        tokens.view(-1)[top_k] = predicted_tokens.view(-1)[top_k]

    is_masked = (tokens == MASK_TOKEN_ID)
    if is_masked.any():
        logits = maskgit(tokens, labels, drop_label)
        probs = F.softmax(logits / temperature, dim=-1)
        _, predicted_tokens = probs.max(dim=-1)
        tokens[is_masked] = predicted_tokens[is_masked]

    tokens = tokens.view(1, LATENT_SIZE, LATENT_SIZE)
    return decode_from_tokens(tokens, vqgan)

def preprocess_image(pil_image):
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return transform(pil_image).unsqueeze(0).to(device)

def tensor_to_pil(tensor):
    tensor = tensor.cpu().detach()
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = ((tensor + 1) / 2).clamp(0, 1)
    numpy_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(numpy_img)

def visualize_mask_on_image(pil_image, mask_indices, color=(255, 0, 0), alpha=0.4):
    img_np = np.array(pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32)
    mask = np.zeros((LATENT_SIZE, LATENT_SIZE), dtype=np.float32)
    for idx in mask_indices:
        mask[idx // LATENT_SIZE, idx % LATENT_SIZE] = 1.0
    scale = IMAGE_SIZE // LATENT_SIZE
    mask_upscaled = np.kron(mask, np.ones((scale, scale)))
    mask_3d = np.stack([mask_upscaled] * 3, axis=-1)
    color_overlay = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    overlay = img_np * (1 - mask_3d * alpha) + color_overlay * mask_3d * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

def get_mask_preset(preset_name):
    presets = {"center_small": [], "center_large": [], "top_left": [], "bottom_right": []}
    for y in range(5, 11):
        for x in range(5, 11):
            presets["center_small"].append(y * 16 + x)
    for y in range(4, 12):
        for x in range(4, 12):
            presets["center_large"].append(y * 16 + x)
    for y in range(0, 6):
        for x in range(0, 6):
            presets["top_left"].append(y * 16 + x)
    for y in range(10, 16):
        for x in range(10, 16):
            presets["bottom_right"].append(y * 16 + x)
    return presets.get(preset_name, presets["center_small"])

# ============================================
# 5. Gradio 앱
# ============================================
import gradio as gr

CLASS_NAMES = ['inclusion', 'pit_hole', 'scratch_crease', 'spot_stain', 'line_crack', 'fold_scale']
CLASS_NAMES_KR = [
    '0: 개재물 (Inclusion)',
    '1: 피트홀 (Pit Hole)',
    '2: 스크래치/주름 (Scratch/Crease)',
    '3: 얼룩 (Spot/Stain)',
    '4: 균열 (Line/Crack)',
    '5: 접힘/스케일 (Fold/Scale)',
]
MASK_PRESETS_KR = {
    "중앙 (작음) 6x6": "center_small",
    "중앙 (큼) 8x8": "center_large",
    "좌상단 6x6": "top_left",
    "우하단 6x6": "bottom_right",
}

def gradio_inpaint(image, defect_type, mask_preset, num_steps, temperature):
    if image is None:
        return None, None, None, "이미지를 업로드해주세요!"
    try:
        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        class_idx = CLASS_NAMES_KR.index(defect_type)
        mask_indices = get_mask_preset(MASK_PRESETS_KR.get(mask_preset, "center_small"))

        input_tensor = preprocess_image(pil_image)

        with torch.no_grad():
            tokens = encode_to_tokens(input_tensor, vqgan)
            reconstructed = decode_from_tokens(tokens, vqgan)

        inpainted = inpaint_image(input_tensor, mask_indices, class_idx, int(num_steps), temperature)

        return (
            visualize_mask_on_image(tensor_to_pil(input_tensor), mask_indices),
            tensor_to_pil(reconstructed),
            tensor_to_pil(inpainted),
            f"완료! 결함: {CLASS_NAMES[class_idx]}, 마스크: {len(mask_indices)}개 토큰"
        )
    except Exception as e:
        return None, None, None, f"에러: {str(e)}"

with gr.Blocks(title="금속 결함 합성기") as demo:
    gr.Markdown("# Metal Defect Synthesizer\n**NEU-DET / X-SDD / SD-saliency-900** 기반 금속 표면 결함 Inpainting")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="원본 이미지 업로드", type="pil")
            defect_dropdown = gr.Dropdown(choices=CLASS_NAMES_KR, value=CLASS_NAMES_KR[5], label="결함 타입")
            mask_dropdown = gr.Dropdown(choices=list(MASK_PRESETS_KR.keys()), value="중앙 (작음) 6x6", label="마스크 영역")
            with gr.Row():
                steps_slider = gr.Slider(minimum=4, maximum=16, value=8, step=1, label="디코딩 스텝")
                temp_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Temperature")
            inpaint_btn = gr.Button("Inpainting 실행", variant="primary")
        with gr.Column(scale=2):
            status_text = gr.Textbox(label="상태", interactive=False)
            with gr.Row():
                mask_preview = gr.Image(label="마스크 미리보기")
                reconstructed_output = gr.Image(label="VQGAN 재구성")
                inpainted_output = gr.Image(label="Inpainting 결과")

    inpaint_btn.click(
        fn=gradio_inpaint,
        inputs=[input_image, defect_dropdown, mask_dropdown, steps_slider, temp_slider],
        outputs=[mask_preview, reconstructed_output, inpainted_output, status_text],
    )

    gr.Markdown("""
---
### 결함 타입 설명
| 결함 | 특징 |
|------|------|
| 개재물 | 표면에 박힌 어두운 점이나 덩어리 |
| 피트홀 | 표면에 움푹 파인 작은 구멍들 |
| 스크래치/주름 | 길게 긁힌 선 자국이나 접혀서 생긴 주름 |
| 얼룩 | 물, 기름 등이 마르면서 생긴 영역 |
| 균열 | 거미줄처럼 퍼진 미세 균열 |
| 접힘/스케일 | 압연 중 산화막이 눌려 들어간 줄무늬 |
""")

demo.launch(ssr_mode=False)
