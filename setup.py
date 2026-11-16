from setuptools import setup, find_packages

setup(
    name="deepfake-detection",
    version="1.0.0",
    description="CNN and Frequency-Based Explainable DeepFake Detection — Final Year Project",
    author="Akash",
    python_requires=">=3.12",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.6,<3",
        "torchvision>=0.21,<1",
        "numpy>=1.26,<3",
        "Pillow>=10,<13",
        "opencv-python-headless>=4.10,<5",
        "scikit-learn>=1.5,<2",
        "matplotlib>=3.8,<4",
        "tqdm>=4.66,<5",
        "gradio>=5,<7",
        "transformers>=4.46,<5",
    ],
    extras_require={
        "dev": ["pytest>=8,<10"],
    },
)
