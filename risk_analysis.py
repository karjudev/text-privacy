from typing import List
from matplotlib.ticker import PercentFormatter
from numpy.core.function_base import linspace
from numpy import zeros

from pandas.core.frame import DataFrame
from pandas.core.series import Series


def combine_max(dataframes: List[DataFrame]) -> DataFrame:
    """Combines a list of risk DataFrames into one,
    taking the maximum for each column.

    Args:
        dataframes (List[DataFrame]): List of objects to combine.

    Returns:
        DataFrame: Single dataframe containing
        the maximum for each column.
    """
    # Output dataframe
    df: DataFrame = dataframes[0]
    for i in range(len(dataframes)):
        for column in df.columns:
            if column in dataframes[i].columns:
                df[column] = df[column].combine(
                    dataframes[i][column],
                    max
                )
    return df

def _risk_and_coverage(risk: Series, r: float) -> float:
    """Computes the Risk and Coverage value for the given risk values.

    Args:
        risk (Series): Series containing the risk values.
        r (float): Value to compute the RAC function of.

    Returns:
        float: Risk And Coverage value for r.
    """
    return (risk <= r).sum() / risk.count()

def plot_rac(df: DataFrame, graph):
    """Plots the RAC curve for the given DataFrame.

    Args:
        df (DataFrame): DataFrame with risk values.
        graph: Object to plot the graph on.
    """
    # Set axes parameters
    graph.set_xlabel("Risk (%)", fontsize=10)
    graph.xaxis.set_major_formatter(PercentFormatter(xmax=1.00))
    graph.set_ylabel("RAC (%)", fontsize=10)
    graph.yaxis.set_major_formatter(PercentFormatter(xmax=1.00))
    # Equally-spaced points
    x = linspace(0, 1, 50)
    # Plot for each column in the dataframe
    for column in df.columns:
        y = [_risk_and_coverage(df[column], r) for r in x]
        graph.scatter(x, y, label = f"BK = {column}")
        graph.plot(x, y)
    graph.legend(fontsize=10)