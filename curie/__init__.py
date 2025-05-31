"""
Curie - A scientific research experimentation agent

This package provides tools for automated scientific research experimentation,
including experiment design, execution, analysis, and visualization.

Main usage:
    import curie
    curie.experiment(
        dataset_dir="path/to/dataset",
        workspace_name="experiment_name",
        question_file="path/to/questions.txt"
    )
"""

__version__ = "0.1.1"

from .main import experiment 

__all__ = [
    "experiment",
]
