from ..data.config import Config
from ..data.data_exploration import Exploration
from ..data.data_visualization import Visualization

config = Config()


def visualize(property_type: str = 'house'):
    """
    Visualize the results of k-fold cross-validation for regression models.

    Args:
        property_type (str): Type of property ('house' or 'apartment'). Default is 'house'.
    """
    cross_val_df_path = config.h_k_fold_cross_val_df
    if property_type == 'apartment':
        cross_val_df_path = config.ap_k_fold_cross_val_df
    df_cross_val = Exploration.load_data_frame(cross_val_df_path)
    Visualization.k_fold_score_linear_plot(df_cross_val)
    Visualization.score_mean_bar_plot(df_cross_val)
