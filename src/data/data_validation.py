from typing import Dict
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np


class DataValidation:
    """
    A class to perform data validation operations using cross-validation.

    Methods:
        cross_validation(model: object, X: pd.DataFrame, y: pd.DataFrame, k_folds: int = 5,
                         score_method: str = 'r2', mean: bool = False) -> float or np.ndarray:
            Perform cross-validation for a given model.
        create_k_fold_cross_val_df(X: pd.DataFrame, y: pd.DataFrame, regressors: Dict, k_folds: int = 5,
                                   score_method: str = 'r2', output_file: str = None) -> pd.DataFrame:
            Create a DataFrame containing cross-validation scores for multiple regressors.

    """
    @classmethod
    def cross_validation(cls, model: object, X: pd.DataFrame, y: pd.DataFrame, k_folds: int = 5,
                         score_method: str = 'r2', mean: bool = False) -> float | np.ndarray:
        """
        Perform cross-validation for a given model.

        Args:
            model (object): Regression model to be validated.
            X (pd.DataFrame): Input features.
            y (pd.DataFrame): Target variable.
            k_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
            score_method (str, optional): Scoring method. Defaults to 'r2'.
            mean (bool, optional): Whether to return mean score or array of scores. Defaults to False.

        Returns:
            float or np.ndarray: Mean cross-validation score if mean=True, else array of scores.
        """
        y = np.ravel(y)
        if mean:
            cv = cross_val_score(model, X, y, cv=k_folds, scoring=score_method)
            return np.mean(cv)
        else:
            return cross_val_score(model, X, y, cv=k_folds, scoring=score_method)

    @classmethod
    def create_k_fold_cross_val_df(cls, X: pd.DataFrame, y: pd.DataFrame, regressors: Dict, k_folds: int = 5,
                                   score_method: str = 'r2', output_file: str = None) -> pd.DataFrame:
        """
        Create a DataFrame containing cross-validation scores for multiple regressors.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.DataFrame): Target variable.
            regressors (Dict): Dictionary of regressor names and corresponding models.
            k_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
            score_method (str, optional): Scoring method. Defaults to 'r2'.
            output_file (str, optional): Path to save the DataFrame as CSV file. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing cross-validation scores.
        """

        df = pd.DataFrame(columns=regressors.keys(), index=range(k_folds))
        df = df.fillna(np.nan).infer_objects(copy=False)

        y = np.ravel(y)

        for name, repressor in regressors.items():
            scores = cls.cross_validation(repressor.model, X, y, k_folds=k_folds, score_method=score_method)
            df.loc[:, name] = scores

        if output_file is not None:
            df.to_csv(output_file, index=False)
        else:
            return df
