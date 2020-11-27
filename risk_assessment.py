from typing import Dict, List, Set, Callable
from itertools import combinations

from pandas import DataFrame, Series

# Occurrences for each instance of columns in df
supp: Callable = lambda df, columns: df.groupby(
    columns
)[columns[0]].count()

def _risk_size(
    df: DataFrame,
    user_df: DataFrame,
    features: Set[str],
    size: int
) -> float:
    """Compute risk for a given author for
    a background knowledge of the given size.

    Args:
        df (DataFrame): Whole data matrix to compute risk.

        user_df (DataFrame): Data matrix containing only
        the items belonging to the author taken into account.
        single user.

        features (Set[str]): Set of columns representing features.

        size (int): Size of the background knowledge to consider.

    Returns:
        float: Maximum probability of the user with respect to
        the considered background knowledge size.
    """
    # Maximum risk value for each combination
    risk: float = 0.0
    for columns in combinations(features, size):
        # Converts tuple to list for indexing
        columns = list(columns)
        # Occurrences for the user dataset
        supp_u: Series = supp(user_df, columns)
        # Occurrences for the whole dataset
        supp_d: Series = supp(df, columns)
        # Maximum probability of reidentification
        prob: float = (supp_u / supp_d).max()
        # If probability is one we have found the maximum
        if prob == 1.0:
            return prob
        # Else update the maximum
        risk = max(risk, prob)
    return risk

def assess_risk(
    df: DataFrame,
    excluded_columns: Set[str],
    id_column: str,
    min_size: int = 1,
    max_size: int = None,
    logging: bool = False
) -> DataFrame:
    """Assesses risk for the DataFrame for a backgroun knowledge of
    the desired range of sizes, from minimum to maximum.

    Args:
        df (DataFrame): Data matrix to assess risk.
        excluded_columns (Set[str]): Column names identifying attributes
        that we don't need to include in the risk assessment.

        id_column (str): Column name that identifies a single user.

        min_size (int, optional): Minimum background knowledge size.
        Defaults to 1.

        max_size (int, optional): Maximum background knowledge size.
        When None, is considered as the number of features.
        Defaults to None.

        logging (bool, optional): Flag that signals when to print output.
        Defaults to False.

    Returns:
        DataFrame: Data Matrix where for each author we associate the
        reidentification risk for each background knowledge of size
        between min_size and max_size.
    """
    # Column names whose attributes represents features
    features: Set[str] = set(df.columns) - excluded_columns - {id_column}
    # If max_size is None sets it
    if max_size is None:
        max_size = len(features)
    if logging:
        print(f"Computing risk for sizes {min_size} to {max_size}")
    # Dictionary containing risk values
    risk_dict: Dict[str, List[float]] = {}
    # For each user computes risk for each size
    for user, user_df in df.groupby(id_column):
        # At the start risk list is empty
        risks: List[str] = []
        for size in range(min_size, max_size + 1):
            if logging:
                print(f"[{user}] Background knowledge size: {size}")
            # If the previous computing returned 1 we set 1
            # (risk cannot increase more)
            if len(risks) > 0 and risks[-1] == 1.0:
                risks.append(1.0)
            else:
                # Risk for the given size
                risk: float = _risk_size(df, user_df, features, size)
                risks.append(risk)
            if logging:
                print(
                    f"[{user}] BK = {size} - Completed"
                )
        risk_dict[user] = risks
    risk_df: DataFrame = DataFrame(risk_dict).transpose()
    # Renames columns switching indices by one
    risk_df.rename(
        columns = {col: col + 1 for col in risk_df.columns},
        inplace = True
    )
    return risk_df