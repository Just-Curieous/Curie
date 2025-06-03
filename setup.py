# filepath: /home/ubuntu/Curie/setup.py
from setuptools import setup, find_packages

setup(
    name="curie-ai",  
    version="0.1.3",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        "psutil",
        "argparse", 
    ],
    entry_points={
        'console_scripts': [
            'curie=curie.experiment:experiment',  
        ],
    },
    author="Jiachen Liu",
    author_email="amberjcjj@gmail.com",
    description="A package for scientific research experimentation agent",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/Just-Curieous/Curie",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
