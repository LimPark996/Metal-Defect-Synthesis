"""
이미지 생성/인페인팅 실행 스크립트

Usage:
    python scripts/generate.py --config configs/inference.yaml --class scratches --num 4
    python scripts/generate.py --config configs/inference.yaml --mode inpaint --image input.png --class scratches
"""

import argparse
import logging

from src.metal_defect_synthesis.config.defaults import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def main():
    parser = argparse.ArgumentParser(description="Metal Defect Image Generation")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Config 파일 경로")
    parser.add_argument("--mode", type=str, default="generate", choices=["generate", "inpaint"], help="생성 모드")
    parser.add_argument("--class", dest="defect_class", type=str, default="scratches", help="결함 클래스")
    parser.add_argument("--num", type=int, default=4, help="생성할 이미지 수")
    parser.add_argument("--image", type=str, default=None, help="인페인팅 원본 이미지 경로")
    parser.add_argument("--output-dir", type=str, default="outputs", help="출력 디렉토리")
    args = parser.parse_args()

    config = load_config(args.config)

    logger.info("=" * 50)
    logger.info("Metal Defect Generation")
    logger.info("=" * 50)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Class: {args.defect_class}")
    logger.info(f"Num Images: {args.num}")

    if args.defect_class not in CLASS_NAMES:
        logger.error(f"Invalid class: {args.defect_class}. Choose from: {CLASS_NAMES}")
        return

    # NOTE: 실제 생성은 GPU 환경에서 체크포인트 로드 후 실행해야 합니다.
    #
    # 생성 코드: src/metal_defect_synthesis/sampling/sampler.py (HaltonSampler)
    # 인페인팅: src/metal_defect_synthesis/sampling/inpainting.py (inpaint_image)
    # Gradio 데모: app/gradio_demo.py

    logger.info("GPU 환경에서 체크포인트 로드 후 실행하세요.")
    logger.info("Gradio 데모: python app/gradio_demo.py")


if __name__ == "__main__":
    main()
