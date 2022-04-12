import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="boss", # Replace with your own username
    version="0.0.1",
    author="Henry Moss",
    author_email="h.moss@lancaster.ac.uk",
    description="Bayesian Optimization for String Spaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/henrymoss/BOSS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.9',
    tests_require=['pytest>=5.4.3'],
    install_requires=[
          'emukit>=0.4.7',
          'PyDOE>=0.3.8',
          'GPy>=1.9.9',
          'numpy>=1.18.5',
          'matplotlib>=3.2.1',
          'nltk>=3.5'
      ]
)