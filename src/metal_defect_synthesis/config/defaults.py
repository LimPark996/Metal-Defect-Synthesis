"""Config 로더 - YAML 설정 파일 로드"""

from pathlib import Path
from omegaconf import OmegaConf


def load_config(config_path: str):
    """YAML config 파일을 OmegaConf로 로드"""
    return OmegaConf.load(config_path)


def get_project_root() -> Path:
    """프로젝트 루트 디렉토리 반환"""
    return Path(__file__).parent.parent.parent.parent
