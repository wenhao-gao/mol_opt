import os
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as f:
    long_description = f.read()

# read the contents of requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()



setuptools.setup(
    name = 'mol_opt',
    version = '0.0.1',
    author = 'Wenhao Gao, Tianfan Fu, Jimeng Sun, Connor Coley',
    author_email = 'gaowh19@gmail.com',
    description = 'mol_opt: A Python Package for Molecular Optimization',
    url = 'https://github.com/wenhao-gao/mol_opt',
    keywords=['molecular optimization', 'drug discovery', 'drug design', 'artificial intelligence', 'deep learning', 'machine learning'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

