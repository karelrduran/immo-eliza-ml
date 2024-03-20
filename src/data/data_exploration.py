from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Exploration:
    """
    A class to perform exploratory data analysis (EDA) operations on pandas DataFrames.

    Methods:
        parse_year(year): Convert year to datetime format.
        check_duplicates(df: pd.DataFrame): Check for and remove duplicate rows from DataFrame.
        get_correlation_matrix(df: pd.DataFrame, columns: List = None, method: str = 'pearson'):
            Calculate correlation matrix for DataFrame.
        display_correlation_matrix(df: pd.DataFrame, columns: List, method: str = 'pearson'):
            Display heatmap of correlation matrix.
        display_percentage_missing_values(df: pd.DataFrame, column: str):
            Display percentage of missing values for a specific column.
        delete_column(df: pd.DataFrame, column: str): Delete a column from DataFrame.
        load_data_frame(df_path: str): Load DataFrame from CSV file.
        save_data_frame(df_path: str, df: pd.DataFrame): Save DataFrame to CSV file.
    """
    @classmethod
    def parse_year(cls, year):
        """
        Convert year to datetime format.

        Args:
            year: Year to be converted.

        Returns:
            datetime: Year in datetime format.
        """
        return pd.to_datetime(str(year), format='%Y')

    @classmethod
    def check_duplicates(cls, df: pd.DataFrame):
        """
        Check for and remove duplicate rows from DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame without duplicate rows.
        """
        if df.duplicated().sum() != 0:
            return df.drop_duplicates(inplace=True)
        return df

    @classmethod
    def get_correlation_matrix(cls, df: pd.DataFrame, columns: List = None, method: str = 'pearson'):
        """
        Calculate correlation matrix for DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (List, optional): List of column names to consider for correlation. Defaults to None.
            method (str, optional): Method to use for correlation calculation. Defaults to 'pearson'.

        Returns:
            pd.DataFrame: Correlation matrix.
        """
        if columns:
            correlation_matrix = df[columns].corr(method=method)
        else:
            correlation_matrix = df[df.select_dtypes(include=['int', 'float']).columns].corr(method=method)
        return correlation_matrix

    @classmethod
    def display_correlation_matrix(cls, df: pd.DataFrame, columns: List, method: str = 'pearson'):
        """
        Display heatmap of correlation matrix.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (List): List of column names to consider for correlation.
            method (str, optional): Method to use for correlation calculation. Defaults to 'pearson'.
        """

        correlation_matrix = cls.get_correlation_matrix(df=df, columns=columns, method=method)

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

    @classmethod
    def display_percentage_missing_values(cls, df: pd.DataFrame, column: str):
        """
        Display percentage of missing values for a specific column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column name for which missing values are to be checked.
        """
        print(f"Total observations: {df.shape[0]}")
        print(f"Total 'Building Year' missing values: {df[column].isnull().sum()}")
        print(f"Percentage of 'Building Year' missing values: {df[column].isnull().mean()}")

    @classmethod
    def delete_column(cls, df: pd.DataFrame, column: str):
        """
        Delete a column from DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column name to be deleted.
        """
        if column in df.columns:
            del df[column]

    @classmethod
    def load_data_frame(cls, df_path: str):
        """
        Load DataFrame from CSV file.

        Args:
            df_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        return pd.read_csv(df_path)

    @classmethod
    def save_data_frame(cls, df_path: str, df: pd.DataFrame):
        """
        Save DataFrame to CSV file.

        Args:
            df_path (str): Path to save the DataFrame as CSV file.
            df (pd.DataFrame): DataFrame to be saved.
        """
        df.to_csv(df_path, index=False)
