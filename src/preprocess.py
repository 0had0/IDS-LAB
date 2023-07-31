import codecs
import glob
import os

import numpy as np
import pandas as pd
from prefect import flow

from src.config import Location


def to_utf8(filename: str, encoding="latin1", blocksize=1048576):
    """Make a file use UTF-8 encoding"""
    tmpfilename = filename + ".tmp"
    with codecs.open(filename, "r", encoding) as source:
        with codecs.open(tmpfilename, "w", "utf-8") as target:
            while True:
                contents = source.read(blocksize)
                if not contents:
                    break
                target.write(contents)

    # replace the original file
    os.rename(tmpfilename, filename)


def renaming_class_label(df: pd.DataFrame):
    """Rename classes labels"""
    labels = {
        "Web Attack \x96 Brute Force": "Web Attack-Brute Force",
        "Web Attack \x96 XSS": "Web Attack-XSS",
        "Web Attack \x96 Sql Injection": "Web Attack-Sql Injection",
    }

    for old_label, new_label in labels.items():
        df.Label.replace(old_label, new_label, inplace=True)


def preprocess_one_file(df):
    """Pre-Process 1 csv file"""
    df.columns = df.columns.str.lstrip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    renaming_class_label(df)
    return df


@flow
def preprocess(location: Location = Location()):
    """Pre-Process all csvs"""
    RAW_PATH = "data/raw"
    to_utf8(
        f"{RAW_PATH}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    )

    all_files = glob.glob(os.path.join(RAW_PATH, "*.csv"))
    df = pd.concat(
        (preprocess_one_file(pd.read_csv(f)) for f in all_files),
        ignore_index=True,
    )
    df.to_csv(location.processed_data, index=False)


if __name__ == "__main__":
    preprocess()
