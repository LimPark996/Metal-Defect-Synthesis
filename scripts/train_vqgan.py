"""
VQGAN Fine-tuning 실행 스크립트

Usage:
    python scripts/train_vqgan.py --config configs/vqgan.yaml
"""

import argparse
import logging

from src.metal_defect_synthesis.config.defaults import load_config
from src.metal_defect_synthesis.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="VQGAN Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/vqgan.yaml", help="Config 파일 경로")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)

    logger.info("=" * 50)
    logger.info("VQGAN Fine-tuning")
    logger.info("=" * 50)
    logger.info(f"Config: {args.config}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch Size: {config.data.batch_size}")

    # NOTE: 실제 학습은 GPU 환경(Colab 등)에서 실행해야 합니다.
    # 이 스크립트는 로컬에서 config를 확인하고 학습을 시작하는 진입점입니다.
    #
    # 전체 학습 코드는 V0/metal_defect_synthesis(PoCFinal).ipynb 참조
    # 모듈화된 코드: src/metal_defect_synthesis/training/vqgan_trainer.py

    logger.info("학습 코드는 src.metal_defect_synthesis.training.vqgan_trainer.VQGANTrainer를 사용하세요.")
    logger.info("GPU 환경(Colab A100 권장)에서 실행 필요")


if __name__ == "__main__":
    main()
