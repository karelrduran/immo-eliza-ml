from src.data.new_data_generator import DataGenerator
from src.models.regression_models import *
from src.data.data_exploration import Exploration
from src.data.config import Config

config = Config()


def new_predict(regressor: object = None, generate_new_data: bool = False,
                property_type: str = 'house') -> pd.DataFrame:
    """
    Predicts property prices using a regression model.

    Args:
        regressor (object, optional): The regression model to use for prediction. Defaults to None.
        generate_new_data (bool, optional): Whether to generate new dummy data before prediction. Defaults to False.
        property_type (str, optional): The type of property for which to make predictions. Defaults to 'house'.

    Returns:
        pd.DataFrame: A DataFrame containing predicted property prices.
    """

    if generate_new_data:
        DataGenerator.generate_dummy_data()
    if regressor is None:
        regressor = XGBoostRegression()

    if property_type == 'house':
        new_data = Exploration.load_data_frame(config.h_fake_data_path)
    else:
        new_data = Exploration.load_data_frame(config.ap_fake_data_path)

    regressor.predict(new_data=new_data)

    return regressor.y_pred
