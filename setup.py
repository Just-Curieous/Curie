# filepath: /home/ubuntu/Curie/setup.py
from setuptools import setup, find_packages

setup(
    name="curie",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        "psutil",
        "argparse",
        # Add other dependencies
    ],
    entry_points={
        'console_scripts': [
            'curie=curie.main:main',  
        ],
    },
    author="Jiachen Liu",
    author_email="amberljc@umich.edu",
    description="A package for scientific research experimentation agent",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/Just-Curieous/Curie",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
