from setuptools import setup, find_packages

setup(
    name="metal-defect-synthesis",
    version="0.1.0",
    description="AI 기반 금속 표면 결함 이미지 합성",
    author="Yumi Park",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "einops>=0.6.0",
        "omegaconf>=2.3.0",
        "gradio>=4.0.0",
        "matplotlib>=3.7.0",
        "tqdm",
        "scikit-image",
        "opencv-python",
    ],
)
