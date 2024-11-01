from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="panns-inference",
    version="0.1.1",
    author="Qiuqiang Kong",
    author_email="qiuqiangkong@gmail.com",
    description="panns_inference: audio tagging and sound event detection inference toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiuqiangkong/panns_inference",
    packages=find_packages(),
    package_data={
        'panns_inference': ['data/*.csv'],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'matplotlib',
        'librosa',
        'torchlibrosa',
        'torch>=1.0.0',
        'numpy<2.0.0',  
        'certifi',
    ],
    python_requires='>=3.6',
)