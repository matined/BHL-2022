import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
import numpy as np
import pickle as pkl
from pipelines.category_imputer import CategoryImputer
from pipelines.category_merger import CategoryMerger
from pipelines.category_encoder import CategoryEncoder
from sklearn.pipeline import make_pipeline
from pipelines.adress_to_coordinates import CoordinatesFromAddress


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
        df[numeric_features] = num_pipe.fit_transform(
            df[numeric_features].values)
        df = cat_pipeline.fit_transform(df)
        df = coord_extract.fit_transform(df)
    else:
        df[numeric_features] = num_pipe.transform(df[numeric_features].values)
        df = cat_pipeline.transform(df)
        df = coord_extract.transform(df)

    df = df.drop(columns=columns_to_delete)

    return df, num_pipe, cat_pipeline


class Predictor:

    def __init__(self):
        with open('pipelines/pickles/num_pipe.pkl', 'rb') as f:
            self.num_pipe = pkl.load(f)
        with open('pipelines/pickles/one_hot.pkl', 'rb') as f:
            self.cat_pipe = make_pipeline(
                CategoryImputer(),
                CategoryMerger(),
                CategoryEncoder()
            )
            self.cat_pipe[2].one_hot_transformer = pkl.load(f)
        with open('pipelines/pickles/kmeans_coords.pkl', 'rb') as f:
            self.coords_converter = pkl.load(f)
        with open('pipelines/pickles/relevant_cols.pkl', 'rb') as f:
            self.relevant_cols = pkl.load(f)
        with open('pipelines/pickles/xgboost.pkl', 'rb') as f:
            self.model = pkl.load(f)

    def predict(self, X: pd.DataFrame):

        processed, _, _ = preprocess_raw_dataframe(
            X, num_pipe=self.num_pipe, cat_pipeline=self.cat_pipe)

        processed = processed.assign(
            loc_cluster=self.coords_converter.convert(processed)
        ).drop(columns=['Longitude', 'Latitude'])

        processed = processed[self.relevant_cols].astype('float')
        return self.model.predict(processed)
