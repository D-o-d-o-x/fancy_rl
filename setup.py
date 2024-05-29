
from setuptools import setup, find_packages

setup(
    name="fancy_rl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchrl",
        "gymnasium",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "fancy_rl=fancy_rl.example:main",
        ],
    },
)
