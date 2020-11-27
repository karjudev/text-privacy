from typing import List

from typer import echo, run
from pandas import DataFrame, read_pickle

from risk_assessment import assess_risk

def main(
    input_filename: str,
    output_filename: str,
    id_column: str,
    min_size: int = 1,
    excluded_columns: List[str] = [],
    logging: bool = True
):
    # Reads DataFrame from Pickle
    echo("Reading DataFrame")
    df: DataFrame = read_pickle(input_filename)
    echo("Input DataFrame successifully read")
    # Risk computation for the file
    echo("Computing risk")
    risk: DataFrame = assess_risk(
        df = df,
        excluded_columns = set(excluded_columns),
        min_size = min_size,
        id_column = id_column,
        logging = logging
    )
    echo("Risk successifully computed. Saving DataFrame...")
    risk.to_pickle(output_filename)
    echo("Risk DataFrame successifully saved on disk.")

if __name__ == "__main__":
    run(main)
