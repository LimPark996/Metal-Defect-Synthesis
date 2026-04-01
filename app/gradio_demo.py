"""
Gradio 데모 - 금속 결함 합성기
인터랙티브 웹 UI로 결함 인페인팅을 수행

Usage:
    python app/gradio_demo.py
"""

import sys
import os
import numpy as np
from PIL import Image

import torch
import gradio as gr

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metal_defect_synthesis.models.maskgit import MaskGITTransformer
from src.metal_defect_synthesis.models.vqgan_wrapper import load_vqgan, encode_to_tokens, decode_from_tokens
from src.metal_defect_synthesis.sampling.inpainting import inpaint_image
from src.metal_defect_synthesis.utils.image import preprocess_image, tensor_to_pil, visualize_mask_on_image, get_mask_preset

# 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 256
LATENT_SIZE = 16
CODEBOOK_SIZE = 16384
CODEBOOK_DIM = 8

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
    """Gradio에서 호출하는 Inpainting 함수"""
    if image is None:
        return None, None, None, "이미지를 업로드해주세요!"

    try:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        class_idx = CLASS_NAMES_KR.index(defect_type)
        preset_key = MASK_PRESETS_KR.get(mask_preset, "center_small")
        mask_indices = get_mask_preset(preset_key)

        input_tensor = preprocess_image(pil_image, device=device)

        with torch.no_grad():
            tokens = encode_to_tokens(input_tensor, vqgan, LATENT_SIZE)
            reconstructed = decode_from_tokens(tokens, vqgan, LATENT_SIZE, CODEBOOK_SIZE)

        inpainted = inpaint_image(
            original_image=input_tensor,
            mask_region=mask_indices,
            target_class=class_idx,
            maskgit_model=maskgit,
            vqgan_model=vqgan,
            num_steps=int(num_steps),
            temperature=temperature,
            latent_size=LATENT_SIZE,
            device=device,
        )

        original_pil = tensor_to_pil(input_tensor)
        reconstructed_pil = tensor_to_pil(reconstructed)
        inpainted_pil = tensor_to_pil(inpainted)
        mask_preview = visualize_mask_on_image(original_pil, mask_indices)

        status = f"완료! 결함: {CLASS_NAMES[class_idx]}, 마스크: {len(mask_indices)}개 토큰"
        return mask_preview, reconstructed_pil, inpainted_pil, status

    except Exception as e:
        return None, None, None, f"에러: {str(e)}"


# Gradio UI
css = """
.gradio-container { font-family: 'Noto Sans KR', sans-serif; }
h1 { text-align: center; }
"""

with gr.Blocks(css=css, title="금속 결함 합성기") as demo:
    gr.Markdown("""
# Metal Defect Synthesizer

**NEU-DET / X-SDD / SD-saliency-900** 데이터셋 기반 금속 표면 결함 Inpainting 도구
""")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="원본 이미지 업로드", type="pil")

            defect_dropdown = gr.Dropdown(
                choices=CLASS_NAMES_KR,
                value=CLASS_NAMES_KR[5],
                label="추가할 결함 타입",
            )

            mask_dropdown = gr.Dropdown(
                choices=list(MASK_PRESETS_KR.keys()),
                value="중앙 (작음) 6x6",
                label="마스크 영역",
            )

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


if __name__ == "__main__":
    # 모델 로드 (체크포인트 경로는 환경에 맞게 수정)
    print("모델 로드는 GPU 환경에서 체크포인트 경로 설정 후 실행하세요.")
    print("Colab에서 실행: V0/metal_defect_gradio_demo_LlamaGen_Halton(PoCFinal).ipynb")
    # demo.launch(share=True, debug=True)
