from typing import List

from math import log10
from numpy import array

from pandas import DataFrame

from feature_engine.discretisers import EqualFrequencyDiscretiser

def _sturges(n: int) -> int:
    """Computes the optimal number of buckets using the Sturges' formula.

    Args:
        n (int): Number of records in the dataset.

    Returns:
        int: Number of buckets computed using Sturges.
    """
    return int(1 + ((10/3) * log10(n)))

def equal_frequency(
    df: DataFrame,
    excluded_columns: List[str] = [],
) -> DataFrame:
    """Subdivides df's columns in bins intervals with the same frequency.

    Args:
        df (DataFrame): DataFrame to discretize.
        bins (int): Number of bins to produce.
        excluded_columns (List[str]): Columns not
        to be discretized. Defaults to [].

    Returns:
        DataFrame: Discretised version of df.
    """
    # Computes the optimal number of bins using the Sturges' formula
    bins: int = _sturges(df.size)
    # Columns to be discretised
    columns: List[str] = list(set(df.columns) - set(excluded_columns))
    # Discretiser object
    discretiser: EqualFrequencyDiscretiser = EqualFrequencyDiscretiser(
        bins,
        variables = columns
    )
    discretiser.fit(df)
    # Discretised version of df
    disc_df: DataFrame = discretiser.transform(df)
    return disc_df

def most_correlated(df: DataFrame, threshold: float = 0.8) -> array:
    """Returns the feature tuples with correlation
    larger than the threshold.

    Args:
        df (DataFrame): DataFrame to misure correlation of
        threshold (float, optional): Threshold above wich
        to show the features. Defaults to 0.8.

    Returns:
        array: Array of tuples of most correlated features.
    """
    # Correlation matrix
    corr: DataFrame = df.corr().abs().unstack()
    # Gets only the part with values greater than the threshold
    return corr[(corr < 1) & (corr >= threshold)]