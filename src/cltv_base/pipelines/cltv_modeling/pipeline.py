# src/cltv_base/pipelines/cltv_modeling/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import predict_cltv_bgf_ggf, predict_cltv_xgboost, predict_cltv_lightgbm
def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for CLTV modeling using BG/NBD and Gamma-Gamma models.
    """
    return Pipeline(
        [
            node(
                func=predict_cltv_bgf_ggf,
                inputs="transactions_typed",
                outputs="predicted_cltv_df",
                name="predict_cltv_bgf_ggf",
            ),
            node(
                func=predict_cltv_xgboost,
                inputs=["customer_level_merged_data_engagement_score", "predicted_churn_probabilities"],
                outputs="cltv_prediction_xgboost",
                name="cltv_prediction_xgboost",
            ),
            node(
                func=predict_cltv_lightgbm,
                inputs=["customer_level_merged_data_engagement_score","predicted_churn_probabilities"],
                outputs="cltv_prediction_lightgbm",
                name="cltv_prediction_lightgbm",
            ),
        ],
        tags="cltv_modeling"
    )

