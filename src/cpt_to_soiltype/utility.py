import pandas as pd
from typing import Callable
from rich.console import Console

def track_sample_num(func: Callable) -> Callable:
    """Tracking number of samples of a dataframe before and after processing.
    """
    console = Console()

    def df_processing(*args: int, **kwargs: int) -> pd.DataFrame:
        res = func(*args, **kwargs)
        for data in args:
            if isinstance(data, pd.DataFrame):
                console.print("--------------------------------")
                console.print(
                    f"Number of samples before processing with {func.__name__} function"
                    f" (rows,cols): {data.shape}"
                )
                console.print(
                    f"Number of samples after processing with {func.__name__} function"
                    f" (rows,cols): {res.shape}"
                )
                console.print("--------------------------------")
        return res

    return df_processing


def info_dataset(
    df_main: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    label: str,
) -> None:
    """Print info about dataset."""
    console = Console()
    console.rule()
    console.print(df_main.info())
    console.rule()
    console.print(
        f"\nA fantastic dataset of {df_main.shape[0]} samples is built :smiley:"
    )
    console.print(f"Num samples trainset: {df_train.shape[0]}")
    console.print(f"Num samples testset: {df_test.shape[0]}")
    console.rule()
    # value counts train
    console.print("\nValue counts trainset:")
    console.print(df_train[label].value_counts())
    # value counts test
    console.print("\nValue counts testset:")
    console.print(df_test[label].value_counts())
