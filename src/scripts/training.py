from typing import Dict

import pandas as pd

from src.models.regression_models import MultiLinearRegression, RidgeRegression, LassoRegression, \
    RandomForestRegression, GradientBoostingRegression, KnnRegression, DecisionTreeRegression, \
    HistGradientBoostingRegression, XGBoostRegression
from src.data.data_validation import DataValidation
from src.data.config import Config

config = Config()


def models_training(property_type: str = 'house'):
    """
    Train regression models for the specified property type.

    Args:
        property_type (str): Type of property ('house' or 'apartment'). Default is 'house'.

    Returns:
        Dict[str, RegressionModels]: A dictionary containing trained regression models.
    """
    regressors = {
        'Multiple Linear Regression': MultiLinearRegression(property_type=property_type),
        'Ridge': RidgeRegression(property_type=property_type),
        'Lasso': LassoRegression(property_type=property_type),
        'Random Forest': RandomForestRegression(property_type=property_type),
        'Gradient Boosting': GradientBoostingRegression(property_type=property_type),
        'knn': KnnRegression(property_type=property_type),
        'Decision Tree': DecisionTreeRegression(property_type=property_type),
        'Histogram Gradient Boosting': HistGradientBoostingRegression(property_type=property_type),
        'XGBoost': XGBoostRegression(property_type=property_type)
    }
    return regressors


def k_fold_cross_val(regressors: Dict, property_type: str = 'house') -> pd.DataFrame:
    """
    Perform k-fold cross-validation for regression models.

    Args:
        regressors (Dict[str, RegressionModels]): Dictionary containing trained regression models.
        property_type (str): Type of property ('house' or 'apartment'). Default is 'house'.

    Returns:
        pd.DataFrame: DataFrame containing the results of k-fold cross-validation.
    """
    X = regressors['XGBoost'].X
    y = regressors['XGBoost'].y

    output_file = config.h_k_fold_cross_val_df
    if property_type == 'apartment':
        output_file = config.ap_k_fold_cross_val_df

    df = DataValidation.create_k_fold_cross_val_df(X, y, regressors=regressors, output_file=output_file)
    return df
