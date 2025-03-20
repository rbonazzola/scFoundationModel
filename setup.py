from setuptools import setup, find_packages

setup(
    name="scFoundationModel",
    version="0.0.1",
    author="Rodrigo Bonazzola (rbonazzola)",
    author_email="rodbonazzola@gmail.com",
    description="A brief description of your project",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/rbonazzola/scFoundationModel",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
    install_requires=[
        "scanpy",
        "torch",
        "torchvision",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "anndata",
        "transformers",
        "pytorch-lightning",
        "mlflow",
        "einops",
        "local-attention",
        "scikit-misc"
    ],
)
