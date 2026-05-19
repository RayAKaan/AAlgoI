from setuptools import setup, find_packages

setup(
    name="aalgoi",
    version="1.0.5",
    description="Artificial Algorithm Intelligence with RL - Self-Adaptive Algorithm Selection",
    packages=find_packages(),
    py_modules=["pipeline"],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "networkx>=3.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.8",
)
