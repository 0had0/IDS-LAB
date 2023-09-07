import glob
import os
import pandas as pd
from prefect import flow

from config import Location


def preprocess_one_file(df):
    """Pre-Process 1 csv file"""
    df.columns = df.columns.str.lstrip()
    return df


@flow
def preprocess(location: Location = Location()):
    """Pre-Process all csvs"""
    RAW_PATH = "data/raw"

    all_files = glob.glob(os.path.join(RAW_PATH, "*.csv"))
    df = pd.concat(
        (preprocess_one_file(pd.read_csv(f)) for f in all_files),
        ignore_index=True,
    )
    df.to_csv(location.processed_data, index=False)


if __name__ == "__main__":
    preprocess()
