import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ranger_adabelief", 
    version="0.0.9",
    author="Juntang Zhuang",
    author_email="j.zhuang@yale.edu",
    description="PyTorch implementation of AdaBelief Optimizer",
    long_description="PyTorch implementation of AdaBelief Optimizer",
    long_description_content_type="text/markdown",
    url="https://juntang-zhuang.github.io/adabelief/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'torch>=0.4.0',
      ],
    python_requires='>=3.6',
)