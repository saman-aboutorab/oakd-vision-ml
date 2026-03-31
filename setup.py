from setuptools import setup, find_packages

setup(
    name="oakd_vision",
    version="0.1.0",
    description="OAK-D Lite vision and ML library for robotics perception",
    author="Saman Aboutorab",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "opencv-python>=4.8",
        "torch>=2.0",
        "torchvision>=0.15",
        "ultralytics>=8.0",
        "wandb>=0.16",
        "onnxruntime>=1.16",
        "PyYAML>=6.0",
        "scipy>=1.11",
    ],
    extras_require={
        "depthai": ["depthai>=2.24"],
        "openvino": ["openvino>=2023.3"],
        "blob": ["blobconverter>=1.4"],
    },
)
