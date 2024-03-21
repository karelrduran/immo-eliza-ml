from src.scripts.exploration import exploration
from src.scripts.transformation import house_data_transformation, apartment_data_transformation
from src.scripts.training import models_training, k_fold_cross_val
from src.scripts.predict import new_predict
from src.scripts.visualization import visualize

if __name__ == "__main__":
    # exploration()
    # house_data_transformation()
    # apartment_data_transformation()
    # h_regressors = models_training(property_type='house')
    # ap_regressors = models_training(property_type='apartment')
    # k_fold_cross_val(regressors=h_regressors, property_type='house')
    # k_fold_cross_val(regressors=ap_regressors, property_type='apartment')
    # visualize(property_type='house')
    # visualize(property_type='apartment')
    print(new_predict())

