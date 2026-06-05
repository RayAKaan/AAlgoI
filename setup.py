from setuptools import setup, find_packages

setup(
    name="aalgoi",
    version="2.1.0",
    description="Artificial Algorithm Intelligence with RL - Self-Adaptive Algorithm Selection",
    packages=find_packages(include=["aalgoi*"]),
    # py_modules moved under aalgoi/ namespace
    install_requires=[
        "numpy>=1.20.0",
        "networkx>=3.0",
        "psutil>=5.8.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "click>=8.1.0",
    ],
    python_requires=">=3.8",
)
