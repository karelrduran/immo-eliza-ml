from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.config import Config


class Transformation:
    """
    A class to perform data transformation operations on pandas DataFrames.

    Methods:
        encode_decode(df: pd.DataFrame, column: str, encoder: Dict) -> pd.DataFrame:
            Encode or decode categorical column using provided encoder dictionary.
        remove_outliers_by_province(df: pd.DataFrame) -> pd.DataFrame:
            Remove outliers from 'Price' column grouped by 'Province'.
        column_one_hot_encoded(df: pd.DataFrame, column: str) -> pd.DataFrame:
            Perform one-hot encoding for the specified column.
        impute_null_values_with_province_feature_averages(df: pd.DataFrame) -> pd.DataFrame:
            Impute null values in DataFrame using province-wise feature averages.

    """

    @classmethod
    def encode_decode(cls, df: pd.DataFrame, column: str, encoder: Dict) -> pd.DataFrame:
        """
        Encode or decode categorical column using provided encoder dictionary.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column to encode/decode.
            encoder (Dict): Dictionary containing encoding mapping.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        df[column] = df[column].map(encoder)
        return df

    @classmethod
    def remove_outliers_by_province(cls, df: pd.DataFrame):
        """
        Remove outliers from 'Price' column grouped by 'Province'.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        df_clean = pd.DataFrame()
        for province, group in df.groupby('Province'):
            Q1 = group['Price'].quantile(0.25)
            Q3 = group['Price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_removed = group[(group['Price'] >= lower_bound) & (group['Price'] <= upper_bound)]
            df_clean = pd.concat([df_clean, outliers_removed])
        return df_clean

    @classmethod
    def split_data(cls, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split the input data into training and testing sets.

        Parameters:
            X (pd.DataFrame): The feature dataset.
            y (pd.Series): The target dataset.
            test_size (float, optional): The proportion of the dataset to include in the test split.
                                          Defaults to 0.2.
            random_state (int, optional): Controls the randomness of the training and testing data split.
                                           Defaults to 42.

        Returns:
            Tuple: A tuple containing the training and testing sets for both features and targets,
                   in the order (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    @classmethod
    def column_one_hot_encoded(cls, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
         Perform one-hot encoding for the specified column.

         Args:
             df (pd.DataFrame): Input DataFrame.
             column (str): Column to perform one-hot encoding on.

         Returns:
             pd.DataFrame: DataFrame with one-hot encoded columns.
         """
        # Perform one-hot encoding for the 'Type' column
        one_hot_encoded = pd.get_dummies(df[column])

        # Convert True/False to 1/0
        one_hot_encoded = one_hot_encoded.astype(int)

        # Concatenate the original DataFrame with the one-hot-encoded columns
        df_encoded = pd.concat([df, one_hot_encoded], axis=1)

        # Delete the original 'Type' column
        # immo_AP_H_encoded.drop(columns=['Type'], inplace=True)
        return df_encoded

    @classmethod
    def impute_null_values_with_province_feature_averages(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute null values in DataFrame using province-wise feature averages.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with null values imputed.
        """
        # Define the columns for which you want to calculate the average
        columns_for_means = ['Habitable Surface', 'Living Surface', 'Consumption Per m2']
        # Define the columns for which you want to calculate the median
        columns_for_median = ['Facades', 'Bathroom Count', 'Toilet Count', 'Room Count', 'Kitchen Type',
                              'State of Building', 'EPC']

        price_ranges = list(range(0, int(df['Price'].max()) + 1, 100000))

        df_imputed = df.copy(deep=True)

        df_imputed['Price_range'] = pd.cut(df_imputed['Price'], bins=price_ranges, right=False)

        groups = df_imputed.groupby(['Province', 'Price_range'], observed=False)

        # Calculate the mean of each group for the specified columns
        means_per_group = groups[columns_for_means].mean()
        # Calculate the median of each group for the specified columns
        medians_per_group = groups[columns_for_median].median()

        for group, mean in means_per_group.iterrows():
            mean_mask = (df_imputed['Province'] == group[0]) & (df_imputed['Price_range'] == group[1])
            df_imputed.loc[mean_mask, columns_for_means] = df_imputed.loc[mean_mask, columns_for_means].fillna(mean)

        for group, median in medians_per_group.iterrows():
            median_mask = (df_imputed['Province'] == group[0]) & (df_imputed['Price_range'] == group[1])
            df_imputed.loc[median_mask, columns_for_median] = df_imputed.loc[median_mask, columns_for_median].fillna(
                median)

        return df_imputed

    @classmethod
    def concat_dataframes(cls,property_type: str = 'house'):
        config = Config()

        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_train = pd.DataFrame()
        y_test = pd.DataFrame()

        if property_type == 'house':
            X_train = pd.read_csv(config.h_X_train_path)
            X_test = pd.read_csv(config.h_X_test_path)
            y_train = pd.read_csv(config.h_y_train_path)
            y_test = pd.read_csv(config.h_y_test_path)
        elif property_type == 'apartment':
            X_train = pd.read_csv(config.ap_X_train_path)
            X_test = pd.read_csv(config.ap_X_test_path)
            y_train = pd.read_csv(config.ap_y_train_path)
            y_test = pd.read_csv(config.ap_y_test_path)

        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])

        X.to_csv(config.ap_X_path, index=False)
        y.to_csv(config.ap_y_path, index=False)
