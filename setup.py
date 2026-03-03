from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="islib",
    version="2025.1.0",
    description="Instance Selection Library — optimal training-period identification for regression models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.8",
    py_modules=["islib"],
    install_requires=[
        "matplotlib>=3.5",
        "numpy>=1.22",
        "pandas>=1.4",
        "scipy>=1.8",
        "scikit-learn>=1.1",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "notebook",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
