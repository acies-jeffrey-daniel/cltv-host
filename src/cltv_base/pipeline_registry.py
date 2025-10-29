"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from cltv_base.pipeline import create_pipeline as create_full_project_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    full_project_pipeline = create_full_project_pipeline()
    

    return {
        "full_pipeline": full_project_pipeline,
        "__default__": full_project_pipeline,
    }

