import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
import numpy as np


def preprocess_raw_dataframe(df, fit = False, num_pipe=None):
    if num_pipe is None:
        num_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0),
                ),
                (
                    "power_scaler",
                    PowerTransformer()
                ),
            ]
        )

    df["BLDAREA"] = df["BLDFRONT"] * df["BLDDEPTH"]
    df["LTAREA"] = df["LTFRONT"] * df["LTDEPTH"]

    numeric_features = [
        "LTFRONT",
        "LTDEPTH",
        "STORIES",
        "AVLAND",
        "AVTOT",
        "EXLAND",
        "EXTOT",
        "BLDFRONT",
        "BLDDEPTH",
        "AVLAND2",
        "AVTOT2",
        "BLDAREA",
        "LTAREA"
    ]


    if fit:
        df[numeric_features] = num_pipe.fit_transform(df[numeric_features].values)
    else:
        df[numeric_features] = num_pipe.transform(df[numeric_features].values)

    return df, num_pipe


if __name__ == '__main__':
    df_train = pd.read_csv("../data/train.csv", sep=",", error_bad_lines=False)
    df_train, num_pipe= preprocess_raw_dataframe(df_train, fit=True)
    df_train.to_csv("../data/train_processed.csv", index=False)

    df_val = pd.read_csv("../data/val.csv", sep=",", error_bad_lines=False)
    df_val, _ = preprocess_raw_dataframe(df_val,num_pipe=num_pipe)
    df_val.to_csv('../data/val_processed.csv', index=False)

    df_test = pd.read_csv("../data/test.csv", sep=",", error_bad_lines=False)
    df_test, _ = preprocess_raw_dataframe(df_test, num_pipe=num_pipe)
    df_test.to_csv('../data/test_processed.csv', index=False)
