import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="adabelief_tf", 
    version="0.2.1",
    author="Juntang Zhuang",
    author_email="j.zhuang@yale.edu",
    description="Tensorflow implementation of AdaBelief Optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://juntang-zhuang.github.io/adabelief/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'tensorflow>=2.0.0,<2.16.0',
      ],
    python_requires='>=3.6',
)
