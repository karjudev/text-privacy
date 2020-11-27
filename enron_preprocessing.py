from email.message import Message
from typing import Set
from email import message_from_string

from pandas import DataFrame, Series

def _parse_body(email: str) -> str:
    """Parses the email body and returns it.

    Args:
        email: Email in MIME format
    
    Returns:
        Email message without headers and forwarded or original message.
    """
    # Email message
    message: Message = message_from_string(email)
    # Body of the message
    body: str = message.get_payload().lower()
    # Index of eventual un-needed footers
    forwarded_index: int = len(body)
    original_index: int = len(body)
    reply_index: int = len(body)
    if "---------------------- forwarded" in body:
        forwarded_index = body.index("---------------------- forwarded")
    if "-----original" in body:
        original_index = body.index("-----original")
    if "______________________________ reply" in body:
        reply_index = body.index("______________________________ reply")
    # Index of the actual body
    index: int = min(forwarded_index, original_index, reply_index)
    return body[:index].strip()

def enron_dataframe(messages: Series) -> DataFrame:
    """Load a CSV for the Enron dataset and return a DataFrame.

    Args:
        messages:   Pandas series with enron emails.
    
    Raises:
        FileNotFoundError: If the file file_name does not exists.

    Returns:
        DataFrame: A DataFrame with email address
        of the author and messages.
    """
    # Features extracted from the messages column
    documents: DataFrame = DataFrame(index = messages.index)
    # Extracts author via regular expression
    documents["Email"] = messages.str.extract(r"From: (.*)")
    # Extracts actual body from message
    documents["Message"] = messages.apply(_parse_body)
    # Drops lenght 0 and 1
    documents.drop(
        documents[documents["Message"].str.len() < 2].index,
        inplace = True
    )
    # Drops duplicates
    documents.drop_duplicates(ignore_index=True, inplace=True)
    return documents

def liwc_dataframe_preprocessing(df: DataFrame):
    """Needed modifications importing data from LIWC processed data.

    Args:
        df (DataFrame): Data read from a LIWC CSV output.
        renaming (Dict[str, str]): Renaming scheme for the columns.
    """
    # Renames email column
    df.rename(
        columns={"B": "Email"},
        inplace=True
    )
    # Drops message column
    del df["C"]
    # Renames index
    df.index.rename("ID", inplace=True)
    # Column names representing float attributes
    attributes: Set[str] = set(df.columns) - {"Email", "WC"}
    # Transforms the attributes into floats
    for attribute in attributes:
        df[attribute] = df[attribute].str.replace(
            ',',
            '.'
        ).astype(float)
    # Drops the records with no words
    df.drop(df[df["WC"] < 1].index, inplace=True)