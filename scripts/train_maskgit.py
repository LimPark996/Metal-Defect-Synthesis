"""
MaskGIT 학습 실행 스크립트

Usage:
    python scripts/train_maskgit.py --config configs/maskgit.yaml
"""

import argparse
import logging

from src.metal_defect_synthesis.config.defaults import load_config
from src.metal_defect_synthesis.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MaskGIT Training")
    parser.add_argument("--config", type=str, default="configs/maskgit.yaml", help="Config 파일 경로")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)

    logger.info("=" * 50)
    logger.info("Halton-MaskGIT Training")
    logger.info("=" * 50)
    logger.info(f"Config: {args.config}")
    logger.info(f"Model Size: {config.model.size}")
    logger.info(f"Hidden Dim: {config.model.hidden_dim}")
    logger.info(f"Layers: {config.model.num_layers}")
    logger.info(f"Epochs: {config.training.epochs}")

    # NOTE: 실제 학습은 GPU 환경(Colab 등)에서 실행해야 합니다.
    #
    # 전체 학습 코드는 V0/metal_defect_HaltonMaskGIT(PoCFinal).ipynb 참조
    # 모듈화된 코드: src/metal_defect_synthesis/training/maskgit_trainer.py

    logger.info("학습 코드는 src.metal_defect_synthesis.training.maskgit_trainer.MaskGITTrainer를 사용하세요.")
    logger.info("GPU 환경(Colab A100 권장)에서 실행 필요")


if __name__ == "__main__":
    main()
