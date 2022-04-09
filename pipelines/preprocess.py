from coordinates_cluster import CoordinatesConverter
from adress_to_coordinates import CoordinatesFromAddress
from sklearn.pipeline import make_pipeline
from category_encoder import CategoryEncoder
from category_merger import CategoryMerger
from category_imputer import CategoryImputer
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
import numpy as np
import pickle as pkl


def preprocess_raw_dataframe(df, fit=False, num_pipe=None, cat_pipeline=None):
    if num_pipe is None:
        num_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(missing_values=np.nan,
                                  strategy="constant", fill_value=0),
                ),
                (
                    "power_scaler",
                    PowerTransformer()
                ),
            ]
        )
    if cat_pipeline is None:
        cat_pipeline = make_pipeline(
            CategoryImputer(),
            CategoryMerger(),
            CategoryEncoder()
        )
    coord_extract = CoordinatesFromAddress()

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
        "LTAREA",
        "EXLAND2",
        "EXTOT2",

    ]
    columns_to_delete = [
        'BBLE',
        'POSTCODE',
        'PERIOD',
        'YEAR',
        'VALTYPE',
        'Borough',
        'Community Board',
        'BORO',
        'Council District',
        'Census Tract',
        'BIN',
        'NTA',
        'New Georeferenced Column',
        'BLOCK',
        'BLDGCL', 'EXT', 'EASEMENT', 'EXCD2', 'EXCD1', 'STADDR', 'LOT', 'OWNER', "TAXCLASS"]

    df["BLDAREA"] = df["BLDFRONT"] * df["BLDDEPTH"]
    df["LTAREA"] = df["LTFRONT"] * df["LTDEPTH"]
    if fit:
        print("Numeric feature processing...")
        df[numeric_features] = num_pipe.fit_transform(
            df[numeric_features].values)
        print("Cat feature processing...")
        df = cat_pipeline.fit_transform(df)
        print("Getting coords...")
        df = coord_extract.fit_transform(df)
    else:
        print("Numeric feature processing...")
        df[numeric_features] = num_pipe.transform(df[numeric_features].values)
        print("Cat feature processing...")
        df = cat_pipeline.transform(df)
        print("Getting coords...")
        df = coord_extract.transform(df)

    df = df.drop(columns=columns_to_delete)

    return df, num_pipe, cat_pipeline


if __name__ == '__main__':
    df_train = pd.read_csv("data/train.csv", sep=",", on_bad_lines='skip')
    df_val = pd.read_csv("data/val.csv", sep=",", on_bad_lines='skip')
    df_test = pd.read_csv("data/test.csv", sep=",", on_bad_lines='skip')

    df_train, num_pipe, cat_pipe = preprocess_raw_dataframe(df_train, fit=True)
    with open('pipelines/pickles/num_pipe.pkl', 'wb') as f:
        pkl.dump(num_pipe, f)
    with open('pipelines/pickles/one_hot.pkl', 'wb') as f:
        pkl.dump(cat_pipe[2].one_hot_transformer, f)

    df_val, _, _ = preprocess_raw_dataframe(
        df_val, num_pipe=num_pipe, cat_pipeline=cat_pipe)

    df_test, _, _ = preprocess_raw_dataframe(
        df_test, num_pipe=num_pipe, cat_pipeline=cat_pipe)

    converter = CoordinatesConverter()

    df_train = df_train.assign(
        loc_cluster=converter.convert(df_train)
    ).drop(columns=['Longitude', 'Latitude'])

    df_val = df_val.assign(
        loc_cluster=converter.convert(df_val)
    ).drop(columns=['Longitude', 'Latitude'])

    df_test = df_test.assign(
        loc_cluster=converter.convert(df_test)
    ).drop(columns=['Longitude', 'Latitude'])

    with open('pipelines/pickles/relevant_cols.pkl', 'rb') as f:
        relevant_cols = pkl.load(f).tolist()
    relevant_cols.append('FULLVAL')

    df_train = df_train[relevant_cols]
    df_val = df_val[relevant_cols]
    df_test = df_test[relevant_cols]

    df_train.to_csv("data/train_processed.csv", index=False)
    df_val.to_csv('data/val_processed.csv', index=False)
    df_test.to_csv('data/test_processed.csv', index=False)
