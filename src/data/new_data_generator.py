import random

import pandas as pd
from src.data.config import Config
from src.data.data_exploration import Exploration

config = Config()


class DataGenerator:
    """
    Class for generating dummy data for apartment and house listings.

    Methods:
        generate_dummy_data: Generates dummy data for apartment and house listings based on provided configuration paths.
    """

    @classmethod
    def generate_dummy_data(cls):
        """
        Generates dummy data for apartment and house listings.

        Loads data frames for apartment and house listings using the provided configuration paths.
        Generates fake data for apartment and house listings based on loaded data frames and predefined column lists.
        Saves the generated fake data frames to the specified paths.
        """
        immo_apartment = Exploration.load_data_frame(config.ap_X_path)
        immo_house = Exploration.load_data_frame(config.h_X_path)

        h_cols = ['Facades', 'Habitable Surface', 'Land Surface', 'Bedroom Count', 'Bathroom Count', 'Toilet Count',
                  'Room Count', 'Kitchen Type', 'Furnished', 'Terrace', 'Garden Exists', 'State of Building',
                  'Living Surface', 'EPC', 'Consumption Per m2', 'Province']

        ap_cols = ['Facades', 'Habitable Surface', 'Bedroom Count', 'Bathroom Count', 'Toilet Count', 'Room Count',
                   'Kitchen Type', 'Furnished', 'Terrace', 'Terrace Surface', 'Garden Exists', 'State of Building',
                   'Living Surface', 'EPC', 'Consumption Per m2', 'Province']

        provinces = ['ANTWERPEN', 'BRUSSEL', 'HENEGOUWEN', 'LIMBURG', 'LUIK', 'LUXEMBURG', 'NAMEN', 'OOST-VLAANDEREN',
                     'VLAAMS-BRABANT', 'WAALS-BRABANT', 'WEST-VLAANDEREN']

        h_fake_df = pd.DataFrame(columns=h_cols)
        ap_fake_df = pd.DataFrame(columns=ap_cols)

        for r in range(100):
            h_row = {
                'Facades': [random.randint(immo_house['Facades'].min(), immo_house['Facades'].max())],
                'Habitable Surface': [random.uniform(immo_house['Habitable Surface'].min(),
                                                     immo_house['Habitable Surface'].max())],
                'Land Surface': [random.uniform(immo_house['Land Surface'].min(), immo_house['Land Surface'].max())],
                'Bedroom Count': [random.randint(immo_house['Bedroom Count'].min(), immo_house['Bedroom Count'].max())],
                'Bathroom Count': [
                    random.randint(immo_house['Bathroom Count'].min(), immo_house['Bathroom Count'].max())],
                'Toilet Count': [random.randint(immo_house['Toilet Count'].min(), immo_house['Toilet Count'].max())],
                'Room Count': [0],
                'Kitchen Type': [random.randint(immo_house['Kitchen Type'].min(), immo_house['Kitchen Type'].max())],
                'Furnished': [random.choice([0, 1])],
                'Terrace': [random.choice([0, 1])],
                'Garden Exists': [random.choice([0, 1])],
                'State of Building': [random.randint(immo_house['State of Building'].min(),
                                                     immo_house['State of Building'].max())],
                'Living Surface': [
                    random.uniform(immo_house['Living Surface'].min(), immo_house['Living Surface'].max())],
                'EPC': [random.randint(immo_house['EPC'].min(), immo_house['EPC'].max())],
                'Consumption Per m2': [random.uniform(immo_house['Consumption Per m2'].min(),
                                                      immo_house['Consumption Per m2'].max())],
                'Province': [random.randint(0, 10)]

            }
            new_h_row = pd.DataFrame.from_dict(data=h_row)
            new_h_row['Room Count'] = new_h_row['Bedroom Count'] + new_h_row['Bathroom Count'] + new_h_row[
                'Toilet Count']

            h_fake_df = pd.concat([h_fake_df, new_h_row], ignore_index=True)
            ap_row = {
                'Facades': [random.randint(immo_apartment['Facades'].min(), immo_apartment['Facades'].max())],
                'Habitable Surface': [random.uniform(immo_apartment['Habitable Surface'].min(),
                                                     immo_apartment['Habitable Surface'].max())],
                'Bedroom Count': [random.randint(immo_apartment['Bedroom Count'].min(),
                                                 immo_apartment['Bedroom Count'].max())],
                'Bathroom Count': [random.randint(immo_apartment['Bathroom Count'].min(),
                                                  immo_apartment['Bathroom Count'].max())],
                'Toilet Count': [
                    random.randint(immo_apartment['Toilet Count'].min(), immo_apartment['Toilet Count'].max())],
                'Room Count': [0],
                'Kitchen Type': [
                    random.randint(immo_apartment['Kitchen Type'].min(), immo_apartment['Kitchen Type'].max())],
                'Furnished': [random.choice([0, 1])],
                'Terrace': [random.choice([0, 1])],
                'Terrace Surface': [random.uniform(immo_apartment['Terrace Surface'].min(),
                                                   immo_apartment['Terrace Surface'].max())],
                'Garden Exists': [random.choice([0, 1])],
                'State of Building': [random.randint(immo_apartment['State of Building'].min(),
                                                     immo_apartment['State of Building'].max())],
                'Living Surface': [random.uniform(immo_apartment['Living Surface'].min(),
                                                  immo_apartment['Living Surface'].max())],
                'EPC': [random.randint(immo_apartment['EPC'].min(), immo_apartment['EPC'].max())],
                'Consumption Per m2': [random.uniform(immo_apartment['Consumption Per m2'].min(),
                                                      immo_apartment['Consumption Per m2'].max())],
                'Province': [random.randint(0, 10)]

            }
            new_ap_row = pd.DataFrame.from_dict(data=ap_row)

            new_ap_row['Room Count'] = new_ap_row['Bedroom Count'] + new_ap_row['Bathroom Count'] + new_ap_row[
                'Toilet Count']

            ap_fake_df = pd.concat([ap_fake_df, new_ap_row], ignore_index=True)

        h_prov_df = pd.DataFrame(0, index=range(100), columns=provinces)
        ap_prov_df = pd.DataFrame(0, index=range(100), columns=provinces)

        h_fake_df.reset_index(drop=True, inplace=True)
        ap_fake_df.reset_index(drop=True, inplace=True)

        for i in range(100):
            h_prov_df.iloc[i, int(h_fake_df.loc[i, 'Province'])] = 1
            ap_prov_df.iloc[i, int(ap_fake_df.loc[i, 'Province'])] = 1

        for prov in provinces:
            h_fake_df[prov] = h_prov_df[prov]
            ap_fake_df[prov] = ap_prov_df[prov]

        del h_fake_df['Province']
        del ap_fake_df['Province']

        Exploration.save_data_frame(df_path=config.h_fake_data_path, df=h_fake_df)
        Exploration.save_data_frame(df_path=config.ap_fake_data_path, df=ap_fake_df)


DataGenerator.generate_dummy_data()
