from setuptools import setup, find_packages

setup(
    name="evoformer-nc",
    version="0.1.0",
    description="Multi-Scale Transformer for Noncoding Variant Effect Prediction",
    author="[Your Name]",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "biopython>=1.81",
    ],
)
