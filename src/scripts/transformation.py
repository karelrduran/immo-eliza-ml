from sklearn.model_selection import train_test_split
from src.data.config import Config
from src.data.data_exploration import Exploration
from src.data.data_transformation import Transformation

config = Config()


def house_data_transformation():
    """
    Perform data transformation for house dataset.

    This function performs the following steps:
    1. Loads the house dataset.
    2. Checks for duplicated rows.
    3. Deletes the 'Type' column.
    4. Encodes 'Kitchen Type', 'State of Building', and 'EPC' features.
    5. Deletes 'Terrace Surface', 'Subtype', and 'Heating Type' features.
    6. Removes outliers based on province.
    7. Splits the dataset into train and test sets.
    8. Imputes missing values with province feature averages.
    9. Deletes remaining missing values.
    10. Saves the preprocessed train and test datasets.

    Returns:
        None
    """
    immo_H = Exploration.load_data_frame(config.immo_H_path)

    # Checking for duplicated rows
    immo_H = Exploration.check_duplicates(immo_H)

    # Deleting Type column
    del immo_H['Type']

    # Encoding Kitchen 'Type feature'
    immo_H = Transformation.encode_decode(
        df=immo_H,
        column='Kitchen Type',
        encoder=config.kitchen_type_encoder
    )

    # Encoding 'State of Building' feature
    immo_H = Transformation.encode_decode(
        df=immo_H,
        column='State of Building',
        encoder=config.estate_of_building_encoder
    )

    # Encoding 'EPC' feature
    immo_H = Transformation.encode_decode(df=immo_H, column='EPC', encoder=config.epc_encoder)

    # Checking for duplicated rows
    immo_H = Exploration.check_duplicates(immo_H)

    # Deleting 'Terrace Surface', 'Subtype' and 'Heating Type' features from house dataset
    del immo_H['Terrace Surface']
    del immo_H['Subtype']
    del immo_H['Heating Type']

    # Removing the outliers from house dataset
    immo_H = Transformation.remove_outliers_by_province(immo_H)

    # Checking for duplicated rows
    immo_H = Exploration.check_duplicates(immo_H)

    # house target column
    house_y = immo_H['Price']

    immo_H = Transformation.column_one_hot_encoded(df=immo_H, column='Province')

    # Checking for duplicated rows
    immo_H = Exploration.check_duplicates(immo_H)

    # Save house dataset before processing missing values
    Exploration.save_data_frame(df_path=config.null_values_houses_path, df=immo_H)

    # Splitting data in Train and Test
    train_house_set, test_house_set, y_train, y_test = train_test_split(immo_H, house_y, test_size=0.2, random_state=42)

    # Dealing with missing values

    # Group properties by municipality and property type, and within each group allow a price difference of up to
    # 50000 euros between properties. Calculate the average of the selected features by groups and apply each
    # average to the null values of each group.

    train_house_set_imputed = Transformation.impute_null_values_with_province_feature_averages(train_house_set)

    # Deleting remaining missing values
    train_house_set_imputed = train_house_set_imputed.dropna()

    # Checking for duplicated rows
    train_house_set_imputed = Exploration.check_duplicates(train_house_set_imputed)

    # Reset indexes
    train_house_set_imputed.reset_index(drop=True, inplace=True)

    house_features_columns = ['Facades', 'Habitable Surface', 'Land Surface', 'Bedroom Count', 'Bathroom Count',
                              'Toilet Count', 'Room Count', 'Kitchen Type', 'Furnished',
                              'Terrace', 'Garden Exists', 'State of Building', 'Living Surface', 'EPC',
                              'Consumption Per m2', 'ANTWERPEN', 'BRUSSEL', 'HENEGOUWEN', 'LIMBURG',
                              'LUIK', 'LUXEMBURG', 'NAMEN', 'OOST-VLAANDEREN', 'VLAAMS-BRABANT', 'WAALS-BRABANT',
                              'WEST-VLAANDEREN']

    X_train_house = train_house_set_imputed[house_features_columns]

    # Checking for duplicated rows
    X_train_house = Exploration.check_duplicates(X_train_house)

    y_train_house = train_house_set_imputed['Price']

    X_train_house = Exploration.check_duplicates(X_train_house)
    y_train_house = y_train_house.loc[X_train_house.index]

    # Imputing test_house dataset missing values
    test_house_set_imputed = Transformation.impute_null_values_with_province_feature_averages(test_house_set)

    # Deleting remaining missing values
    test_house_set_imputed = test_house_set_imputed.dropna()

    # Reset indexes
    test_house_set_imputed.reset_index(drop=True, inplace=True)

    test_house_set_imputed = Exploration.check_duplicates(test_house_set_imputed)

    X_test_house = test_house_set_imputed[house_features_columns]

    y_test_house = test_house_set_imputed['Price']

    X_test_house = Exploration.check_duplicates(X_test_house)

    y_test_house = y_test_house[X_test_house.index]

    Exploration.save_data_frame(df_path=config.h_X_train_path, df=X_train_house)
    Exploration.save_data_frame(df_path=config.h_y_train_path, df=y_train_house)
    Exploration.save_data_frame(df_path=config.h_X_test_path, df=X_test_house)
    Exploration.save_data_frame(df_path=config.h_y_test_path, df=y_test_house)


def apartment_data_transformation():
    """
    Perform data transformation for apartment dataset.

    This function performs the following steps:
    1. Loads the apartment dataset.
    2. Checks for duplicated rows.
    3. Deletes the 'Type' column.
    4. Encodes 'Kitchen Type', 'State of Building', and 'EPC' features.
    5. Deletes 'Subtype' and 'Heating Type' features.
    6. Removes outliers based on province.
    7. Splits the dataset into train and test sets.
    8. Imputes missing values with province feature averages.
    9. Deletes remaining missing values.
    10. Saves the preprocessed train and test datasets.

    Returns:
        None
    """
    immo_AP = Exploration.load_data_frame(config.immo_AP_path)

    # Checking for duplicated rows
    immo_AP = Exploration.check_duplicates(immo_AP)

    # Deleting Type column
    del immo_AP['Type']

    # Encoding Kitchen 'Type feature'
    immo_AP = Transformation.encode_decode(
        df=immo_AP,
        column='Kitchen Type',
        encoder=config.kitchen_type_encoder
    )

    # Encoding 'State of Building' feature
    immo_AP = Transformation.encode_decode(
        df=immo_AP,
        column='State of Building',
        encoder=config.estate_of_building_encoder
    )

    # Encoding 'EPC' feature
    immo_AP = Transformation.encode_decode(df=immo_AP, column='EPC', encoder=config.epc_encoder)

    # Checking for duplicated rows
    immo_AP = Exploration.check_duplicates(immo_AP)

    # Deleting 'Subtype' and 'Heating Type' features from house dataset
    del immo_AP['Subtype']
    del immo_AP['Heating Type']

    # Removing the outliers from house dataset
    immo_AP = Transformation.remove_outliers_by_province(immo_AP)

    # Checking for duplicated rows
    immo_AP = Exploration.check_duplicates(immo_AP)

    # house target column
    apartment_y = immo_AP['Price']

    immo_AP = Transformation.column_one_hot_encoded(df=immo_AP, column='Province')

    # Checking for duplicated rows
    immo_AP = Exploration.check_duplicates(immo_AP)

    # Save house dataset before processing missing values
    Exploration.save_data_frame(df_path=config.null_values_apartments_path, df=immo_AP)

    # Splitting data in Train and Test
    train_apartment_set, test_apartment_set, y_train, y_test = train_test_split(immo_AP, apartment_y, test_size=0.2,
                                                                                random_state=42)

    del train_apartment_set['Land Surface']
    del test_apartment_set['Land Surface']

    # Dealing with missing values

    # Group properties by municipality and property type, and within each group allow a price difference of up to
    # 50000 euros between properties. Calculate the average of the selected features by groups and apply each
    # average to the null values of each group.

    train_apartment_set_imputed = Transformation.impute_null_values_with_province_feature_averages(train_apartment_set)

    # Deleting remaining missing values
    train_apartment_set_imputed = train_apartment_set_imputed.dropna()

    # Checking for duplicated rows
    train_apartment_set_imputed = Exploration.check_duplicates(train_apartment_set_imputed)

    # Reset indexes
    train_apartment_set_imputed.reset_index(drop=True, inplace=True)

    apartment_features_columns = ['Facades', 'Habitable Surface', 'Bedroom Count', 'Bathroom Count', 'Toilet Count',
                                  'Room Count', 'Kitchen Type',
                                  'Furnished', 'Terrace', 'Terrace Surface', 'Garden Exists', 'State of Building',
                                  'Living Surface', 'EPC', 'Consumption Per m2', 'ANTWERPEN',
                                  'BRUSSEL', 'HENEGOUWEN', 'LIMBURG', 'LUIK', 'LUXEMBURG', 'NAMEN', 'OOST-VLAANDEREN',
                                  'VLAAMS-BRABANT', 'WAALS-BRABANT', 'WEST-VLAANDEREN']

    X_train_apartment = train_apartment_set_imputed[apartment_features_columns]

    # Checking for duplicated rows
    X_train_apartment = Exploration.check_duplicates(X_train_apartment)

    y_train_apartment = train_apartment_set_imputed['Price']

    y_train_apartment = y_train_apartment.loc[X_train_apartment.index]

    # Imputing test_house dataset missing values
    test_apartment_set_imputed = Transformation.impute_null_values_with_province_feature_averages(test_apartment_set)

    # Deleting remaining missing values
    test_apartment_set_imputed = test_apartment_set_imputed.dropna()

    # Reset indexes
    test_apartment_set_imputed.reset_index(drop=True, inplace=True)

    test_apartment_set_imputed = Exploration.check_duplicates(test_apartment_set_imputed)

    X_test_apartment = test_apartment_set_imputed[apartment_features_columns]

    y_test_apartment = test_apartment_set_imputed['Price']

    y_test_apartment = Exploration.check_duplicates(y_test_apartment)

    y_test_apartment = y_test_apartment[X_test_apartment.index]

    Exploration.save_data_frame(df_path=config.ap_X_train_path, df=X_train_apartment)
    Exploration.save_data_frame(df_path=config.ap_y_train_path, df=y_train_apartment)
    Exploration.save_data_frame(df_path=config.ap_X_test_path, df=X_test_apartment)
    Exploration.save_data_frame(df_path=config.ap_y_test_path, df=y_test_apartment)
