import pandas as pd
from matplotlib import pyplot as plt


class Visualization:
    """
    A class to visualize data using plots.

    Methods:
        score_mean_bar_plot(df: pd.DataFrame):
            Plot the mean scores of models as a bar plot.
        k_fold_score_linear_plot(df: pd.DataFrame, k_folds: int = 5):
            Plot the scores of models across different folds as a linear plot.

    """

    @classmethod
    def score_mean_bar_plot(cls, df: pd.DataFrame):
        """
        Plot the mean scores of models as a bar plot.

        Args:
            df (pd.DataFrame): DataFrame containing scores of models across folds.

        """
        average = df.mean() * 100

        fig, ax = plt.subplots()

        bars = average.plot(kind='bar', ax=ax)

        for bar in bars.patches:
            ax.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.005,
                    round(bar.get_height(), 3), ha='center', va='bottom')

        ax.set_xlabel('Model')
        ax.set_ylabel('score (%)')
        ax.set_title(f'Score averages ({len(df)} folds)')

        plt.show()

    @classmethod
    def k_fold_score_linear_plot(cls, df: pd.DataFrame, k_folds: int = 5):
        """
        Plot the scores of models across different folds as a linear plot.

        Args:
            df (pd.DataFrame): DataFrame containing scores of models across folds.
            k_folds (int, optional): Number of folds. Defaults to 5.

        """
        fig, ax = plt.subplots()

        for model, scores in df.items():
            ax.plot(range(1, k_folds + 1), scores * 100, marker='o', label=model)

        ax.set_xticks(range(1, k_folds + 1))

        ax.set_xlabel('Folds')
        ax.set_ylabel('score (%)')
        ax.set_title(f'Score averages ({k_folds} folds)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1))

        plt.show()
