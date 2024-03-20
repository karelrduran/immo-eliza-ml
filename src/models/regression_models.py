from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

import pickle
import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

from ..data.data_validation import DataValidation
from src.data.data_transformation import Transformation
from ..data.config import Config

config = Config()


class RegressionModels:
    """
    A class to train, evaluate, and use regression models for predictive analysis.

    Methods:
        load_model(model_path: str):
            Load a trained model from a file.
        save_model(model_path: str):
            Save the trained model to a file.
        fit():
            Train the regression model.
        predict():
            Make predictions using the trained model.
        score():
            Evaluate the performance of the trained model.
        cross_val(mean: bool = False):
            Perform cross-validation to evaluate the model's performance.

    """

    def __init__(self, property_type: str = 'house', X: pd.DataFrame = None, y: pd.Series = None):
        """
        Initialize the RegressionModels object.

        Args:
            property_type (str, optional): Type of property to consider (either 'house' or 'apartment'). Defaults to 'house'.

        Raises:
            ValueError: If an invalid property_type is provided.
        """
        if (X is None) and (y is None):
            if property_type.lower() == 'house':
                try:
                    self.X_train = pd.read_csv(config.h_X_train_path)
                    self.X_test = pd.read_csv(config.h_X_test_path)
                    self.y_train = pd.read_csv(config.h_y_train_path)
                    self.y_test = pd.read_csv(config.h_y_test_path)
                except FileNotFoundError as e:
                    print(f"File Not Found: {e}")
            elif property_type.lower() == 'apartment':
                try:
                    self.X_train = pd.read_csv(config.ap_X_train_path)
                    self.X_test = pd.read_csv(config.ap_X_test_path)
                    self.y_train = pd.read_csv(config.ap_y_train_path)
                    self.y_test = pd.read_csv(config.ap_y_test_path)
                except FileNotFoundError as e:
                    print(f"File Not Found: {e}")
            else:
                raise ValueError(f"Invalid property_type: {property_type}")
        else:
            if (property_type.lower() == 'house') or (property_type.lower() == 'house'):
                self.X_train, self.X_test, self.y_train, self.y_test = Transformation.split_data(X=X, y=y)
            else:
                raise ValueError(f"Invalid property_type: {property_type}")

        self.property_type = property_type.lower()

        self.X = pd.concat([self.X_train, self.X_test])

        self.y = pd.concat([self.y_train, self.y_test])

        self.model = None
        self.y_pred = None
        self.accuracy = None

    def load_model(self, model_path: str):
        """
        Load a trained model from a file.

        Args:
            model_path (str): Path to the saved model file.

        Raises:
            FileNotFoundError: If the specified file is not found.
        """
        try:
            with open(model_path, 'rb') as archivo:
                self.model = pickle.load(archivo)
        except FileNotFoundError as e:
            print(f"File Not Found: {e}")

    def save_model(self, model_path: str):
        """
        Save the trained model to a file.

        Args:
            model_path (str): Path to save the trained model.

        Raises:
            FileNotFoundError: If the specified path is not found.
        """
        try:
            with open(model_path, 'wb') as archivo:
                pickle.dump(self.model, archivo)
        except FileNotFoundError as e:
            print(f"File Not found: {e}")

    def fit(self):
        """Train the regression model."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """Make predictions using the trained model."""
        self.y_pred = self.model.predict(self.X_test)

    def score(self):
        """Evaluate the performance of the trained model."""
        self.accuracy = self.model.score(self.X_test, self.y_test)

    def cross_val(self, mean: bool = False):
        """
        Perform cross-validation to evaluate the model's performance.

        Args:
            mean (bool, optional): If True, returns the mean cross-validation score. Defaults to False.

        Returns:
            float or np.ndarray: Mean cross-validation score if mean=True, else array of scores.
        """
        return DataValidation.cross_validation(model=self.model, X=self.X, y=self.y, k_folds=5, score_method='r2',
                                               mean=mean)


class MultiLinearRegression(RegressionModels):
    """
    A class representing MultiLinearRegression model.

    Inherits:
        RegressionModels

    Methods:
        __init__(model_path: str = None, property_type: str = 'house'):
            Initialize the MultiLinearRegression object.
    """

    def __init__(self, model_path: str = None, property_type: str = 'house'):
        """
        Initialize the MultiLinearRegression object.

        Args:
            model_path (str, optional): Path to a saved model file. Defaults to None.
            property_type (str, optional): Type of property to consider (either 'house' or 'apartment'). Defaults to 'house'.
        """
        super().__init__(property_type)
        if model_path is not None:
            self.load_model(model_path=model_path)
        else:
            self.model = LinearRegression()

        self.fit()
        self.predict()
        self.score()


class RidgeRegression(RegressionModels):
    """
    A class representing RidgeRegression model.

    Inherits:
        RegressionModels

    Methods:
        __init__(model_path: str = None, property_type: str = 'house'):
            Initialize the RidgeRegression object.
    """

    def __init__(self, model_path: str = None, property_type: str = 'house'):
        """
        Initialize the RidgeRegression object.

        Args:
            model_path (str, optional): Path to a saved model file. Defaults to None.
            property_type (str, optional): Type of property to consider (either 'house' or 'apartment'). Defaults to 'house'.
        """
        super().__init__(property_type)
        if model_path is not None:
            self.load_model(model_path=model_path)
        else:
            self.model = Ridge(alpha=0.1)

        self.fit()
        self.predict()
        self.score()


class LassoRegression(RegressionModels):
    """
    A class representing LassoRegression model.

    Inherits:
        RegressionModels

    Methods:
        __init__(model_path: str = None, property_type: str = 'house'):
            Initialize the LassoRegression object.
    """

    def __init__(self, model_path: str = None, property_type: str = 'house'):
        """
        Initialize the LassoRegression object.

        Args:
            model_path (str, optional): Path to a saved model file. Defaults to None.
            property_type (str, optional): Type of property to consider (either 'house' or 'apartment'). Defaults to 'house'.
        """
        super().__init__(property_type)
        if model_path is not None:
            self.load_model(model_path=model_path)
        else:
            self.model = Lasso(alpha=100)

        self.fit()
        self.predict()
        self.score()


class RandomForestRegression(RegressionModels):
    """
    A class representing RandomForestRegression model.

    Inherits:
        RegressionModels

    Methods:
        __init__(model_path: str = None, property_type: str = 'house'):
            Initialize the RandomForestRegression object.
    """

    def __init__(self, model_path: str = None, property_type: str = 'house'):
        """
        Initialize the RandomForestRegression object.

        Args:
            model_path (str, optional): Path to a saved model file. Defaults to None.
            property_type (str, optional): Type of property to consider (either 'house' or 'apartment'). Defaults to 'house'.
        """
        super().__init__(property_type)

        self.y_train = np.ravel(self.y_train)

        if model_path is not None:
            self.load_model(model_path=model_path)
        else:
            self.model = RandomForestRegressor(n_estimators=400, random_state=42)

        self.fit()
        self.predict()
        self.score()


class GradientBoostingRegression(RegressionModels):
    """
    A class representing GradientBoostingRegression model.

    Inherits:
        RegressionModels

    Methods:
        __init__(model_path: str = None, property_type: str = 'house'):
            Initialize the GradientBoostingRegression object.
    """

    def __init__(self, model_path: str = None, property_type: str = 'house'):
        """
        Initialize the GradientBoostingRegression object.

        Args:
            model_path (str, optional): Path to a saved model file. Defaults to None.
            property_type (str, optional): Type of property to consider (either 'house' or 'apartment'). Defaults to 'house'.
        """
        super().__init__(property_type)

        self.y_train = np.ravel(self.y_train)

        if model_path is not None:
            self.load_model(model_path=model_path)
        else:
            self.model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, random_state=42)

        self.fit()
        self.predict()
        self.score()


class KnnRegression(RegressionModels):
    """
    A class representing KnnRegression model.

    Inherits:
        RegressionModels

    Methods:
        __init__(model_path: str = None, property_type: str = 'house'):
            Initialize the KnnRegression object.
    """

    def __init__(self, model_path: str = None, property_type: str = 'house'):
        """
        Initialize the KnnRegression object.

        Args:
            model_path (str, optional): Path to a saved model file. Defaults to None.
            property_type (str, optional): Type of property to consider (either 'house' or 'apartment'). Defaults to 'house'.
        """
        super().__init__(property_type)

        if model_path is not None:
            self.load_model(model_path=model_path)
        else:
            self.model = KNeighborsRegressor(n_neighbors=10)

        self.fit()
        self.predict()
        self.score()


class DecisionTreeRegression(RegressionModels):
    """
    A class representing DecisionTreeRegression model.

    Inherits:
        RegressionModels

    Methods:
        __init__(model_path: str = None, property_type: str = 'house'):
            Initialize the DecisionTreeRegression object.
    """

    def __init__(self, model_path: str = None, property_type: str = 'house'):
        """
        Initialize the DecisionTreeRegression object.

        Args:
            model_path (str, optional): Path to a saved model file. Defaults to None.
            property_type (str, optional): Type of property to consider (either 'house' or 'apartment'). Defaults to 'house'.
        """
        super().__init__(property_type)

        if model_path is not None:
            self.load_model(model_path=model_path)
        else:
            self.model = DecisionTreeRegressor(max_depth=500)

        self.fit()
        self.predict()
        self.score()


class HistGradientBoostingRegression(RegressionModels):
    """
    A class representing HistGradientBoostingRegression model.

    Inherits:
        RegressionModels

    Methods:
        __init__(model_path: str = None, property_type: str = 'house'):
            Initialize the HistGradientBoostingRegression object.
    """

    def __init__(self, model_path: str = None, property_type: str = 'house'):
        """
        Initialize the HistGradientBoostingRegression object.

        Args:
            model_path (str, optional): Path to a saved model file. Defaults to None.
            property_type (str, optional): Type of property to consider (either 'house' or 'apartment'). Defaults to 'house'.
        """
        super().__init__(property_type)

        self.y_train = np.ravel(self.y_train)

        if model_path is not None:
            self.load_model(model_path=model_path)
        else:
            self.model = HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.1, random_state=42)

        self.fit()
        self.predict()
        self.score()


class XGBoostRegression(RegressionModels):
    """
    A class representing XGBoostRegression model.

    Inherits:
        RegressionModels

    Methods:
        __init__(model_path: str = None, property_type: str = 'house'):
            Initialize the XGBoostRegression object.
    """

    def __init__(self, model_path: str = None, property_type: str = 'house'):
        """
        Initialize the XGBoostRegression object.

        Args:
            model_path (str, optional): Path to a saved model file. Defaults to None.
            property_type (str, optional): Type of property to consider (either 'house' or 'apartment'). Defaults to 'house'.
        """
        super().__init__(property_type)

        self.y_train = np.ravel(self.y_train)

        if model_path is not None:
            self.load_model(model_path=model_path)
        else:
            self.model = XGBRegressor(learning_rate=0.1, n_estimators=150, booster='dart')

        self.fit()
        self.predict()
        self.score()
