import pandas as pd
pd.set_option('future.no_silent_downcasting', False)

from src.data.config import Config
from src.data.data_exploration import Exploration

config = Config()


def exploration():
    """
    Perform data exploration and preprocessing.

    This function performs the following steps:
    1. Loads the BPost Postal Codes Dataset and the raw data.
    2. Checks for duplicate rows in the raw data.
    3. Merges the raw data with Postal Codes and renames columns.
    4. Fills missing data for specific columns with False.
    5. Converts boolean features to the correct data type.
    6. Converts boolean values to integers.
    7. Selects data related to apartments and houses.
    8. Drops columns with high average null values and uncorrelated columns.
    9. Checks for duplicate rows again.
    10. Splits the data into house and apartment datasets.
    11. Saves the house and apartment datasets.

    Note:
        This function assumes the availability of the necessary configurations and data paths.

    Returns:
        None
    """
    # Load the BPost Postal Codes Dataset
    post_code_df = Exploration.load_data_frame(config.post_code_data_path)

    # Load our raw data
    immo_raw_df = Exploration.load_data_frame(config.raw_data_path)

    # checking for duplicate rows
    immo_raw_df = Exploration.check_duplicates(immo_raw_df)

    # Merge our raw data with Postal Codes

    immo_raw_df = pd.merge(immo_raw_df, post_code_df, left_on='Postal Code', right_on='Postcode', how='inner')
    del immo_raw_df['Postcode']
    del immo_raw_df['Plaatsnaam']
    del immo_raw_df['Deelgemeente']

    immo_raw_df.rename(columns={'Hoofdgemeente': 'Municipality', 'Provincie': 'Province'}, inplace=True)

    # checking for duplicate rows
    immo_raw_df = Exploration.check_duplicates(immo_raw_df)

    # Filling all missing data for 'Swimming Pool', 'Has starting Price', 'Is Holiday Property', 'Sewer' and 'Sea
    # view' with False
    immo_raw_df['Swimming Pool'] = immo_raw_df['Swimming Pool'].fillna(False)
    immo_raw_df['Has starting Price'] = immo_raw_df['Has starting Price'].fillna(False)
    immo_raw_df['Is Holiday Property'] = immo_raw_df['Is Holiday Property'].fillna(False)
    immo_raw_df['Sewer'] = immo_raw_df['Sewer'].fillna(False)
    immo_raw_df['Sea view'] = immo_raw_df['Sea view'].fillna(False)

    # Putting correct type in boolean features
    immo_raw_df['Swimming Pool'] = immo_raw_df['Swimming Pool'].astype(bool)
    immo_raw_df['Has starting Price'] = immo_raw_df['Has starting Price'].astype(bool)
    immo_raw_df['Is Holiday Property'] = immo_raw_df['Is Holiday Property'].astype(bool)
    immo_raw_df['Sewer'] = immo_raw_df['Sewer'].astype(bool)
    immo_raw_df['Sea view'] = immo_raw_df['Sea view'].astype(bool)

    # Changing the bool type to int {True: 1, False: 0}

    # Select bool columns
    boolean_columns = immo_raw_df.select_dtypes(include=bool).columns

    # Apply the lambda function to each element of the Boolean columns.
    immo_raw_df[boolean_columns] = immo_raw_df[boolean_columns].apply(lambda x: x.map({True: 1, False: 0}))

    # Change the data type of columns to int
    immo_raw_df[boolean_columns] = immo_raw_df[boolean_columns].astype(int)

    # Getting just APARTMENT AND HOUSE

    immo_AP_H = immo_raw_df[immo_raw_df['Type'].isin(['APARTMENT', 'HOUSE'])]

    interest_columns = ['Price', 'Postal Code', 'Build Year', 'Facades', 'Habitable Surface', 'Land Surface', 'Type',
                        'Subtype', 'Bedroom Count', 'Bathroom Count', 'Toilet Count', 'Room Count', 'Kitchen Surface',
                        'Kitchen Type', 'Furnished', 'Fireplace Count', 'Terrace', 'Terrace Surface', 'Garden Exists',
                        'Garden Surface', 'Swimming Pool', 'State of Building', 'Living Surface', 'EPC',
                        'Consumption Per m2', 'Heating Type', 'Sewer', 'Sea view', 'Parking count inside',
                        'Parking count outside', 'Province']

    immo_AP_H = immo_AP_H[interest_columns]

    # 'Kitchen_surface', 'Fireplace Count', 'Garden Surface', 'Swimming Pool', 'Sea view', 'Parking count inside', 'Parking count outside' have high null values average
    # We will just drop these columns
    high_average_null_columns = ['Kitchen Surface', 'Fireplace Count', 'Garden Surface', 'Swimming Pool', 'Sea view',
                                 'Parking count inside', 'Parking count outside', 'Sewer']

    immo_AP_H = immo_AP_H.drop(columns=high_average_null_columns)

    # Build Year is not correlated with Price. That's why we deleted
    del immo_AP_H['Build Year']

    # Checking for duplicates
    immo_AP_H = Exploration.check_duplicates(immo_AP_H)

    # Splitting into house and apartment datasets
    immo_AP = immo_AP_H[immo_AP_H['Type'] == 'APARTMENT']
    immo_H = immo_AP_H[immo_AP_H['Type'] == 'HOUSE']

    # Save house and apartment datasets
    Exploration.save_data_frame(df_path=config.immo_AP_path, df=immo_AP)
    Exploration.save_data_frame(df_path=config.immo_H_path, df=immo_H)


