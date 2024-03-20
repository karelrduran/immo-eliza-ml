import os
import json


class Config:
    """
    A class to handle project configurations and settings.

    Methods:
        load_settings(): Load configurations from the settings JSON file.
        save_settings(path: str, key: str, value: str): Update and save settings to the settings JSON file.
    """

    def __init__(self):
        """
        Initialize Config with project directory and load settings.
        """
        # Get project dir path
        self.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.settings_path = os.path.join(self.project_dir, 'settings', 'settings.json')

        self.load_settings()

    def load_settings(self):
        """
        Load configurations from the settings JSON file.
        """
        # Load configurations from settings.json
        with open(self.settings_path, 'r') as f:
            self.settings = json.load(f)

        self.raw_data_path = os.path.join(self.project_dir, self.settings['data_paths']['raw_data'])
        self.post_code_data_path = os.path.join(self.project_dir, self.settings['data_paths']['post_code_data'])
        self.cleaned_data_path = os.path.join(self.project_dir, self.settings['data_paths']['cleaned_data'])
        self.immo_AP_path = os.path.join(self.project_dir, self.settings['data_paths']['immo_AP_path'])
        self.immo_H_path = os.path.join(self.project_dir, self.settings['data_paths']['immo_H_path'])

        self.null_values_houses_path = os.path.join(self.project_dir,
                                                    self.settings['data_paths']['null_values_houses_path'])
        self.null_values_apartments_path = os.path.join(self.project_dir,
                                                        self.settings['data_paths']['null_values_apartments_path'])

        self.training_data_path = os.path.join(self.project_dir, self.settings['data_paths']['training_data'])
        self.testing_data_path = os.path.join(self.project_dir, self.settings['data_paths']['testing_data'])

        self.h_X_train_path = os.path.join(self.project_dir, self.settings['data_paths']['h_X_train'])
        self.h_X_test_path = os.path.join(self.project_dir, self.settings['data_paths']['h_X_test'])
        self.h_y_train_path = os.path.join(self.project_dir, self.settings['data_paths']['h_y_train'])
        self.h_y_test_path = os.path.join(self.project_dir, self.settings['data_paths']['h_y_test'])
        self.ap_X_train_path = os.path.join(self.project_dir, self.settings['data_paths']['ap_X_train'])
        self.ap_X_test_path = os.path.join(self.project_dir, self.settings['data_paths']['ap_X_test'])
        self.ap_y_train_path = os.path.join(self.project_dir, self.settings['data_paths']['ap_y_train'])
        self.ap_y_test_path = os.path.join(self.project_dir, self.settings['data_paths']['ap_y_test'])

        self.h_k_fold_cross_val_df = os.path.join(self.project_dir,
                                                  self.settings['data_paths']['h_k_fold_cross_val_df'])
        self.ap_k_fold_cross_val_df = os.path.join(self.project_dir,
                                                   self.settings['data_paths']['ap_k_fold_cross_val_df'])

        self.estate_of_building_encoder = self.settings['state_of_building_encoder']
        self.kitchen_type_encoder = self.settings['kitchen_type_encoder']
        self.epc_encoder = self.settings['epc_encoder']

    def save_settings(self, path: str, key: str, value: str):
        """
        Update and save settings to the settings JSON file.

        Args:
            path (str): Path in settings JSON to update.
            key (str): Key of the element to update.
            value (str): New value for the element.
        """
        new_element = {
            key: value
        }

        with open(self.settings_path, 'r') as f:
            content = json.load(f)

        content[path].update(new_element)

        with open(self.settings_path, 'w') as f:
            json.dump(content, f, indent=4)
