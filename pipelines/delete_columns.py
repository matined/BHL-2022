import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

class DeleteColumns(TransformerMixin):

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
        'New Georeferenced Column'
    ]

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **trans_params):
        new_df = df.copy()
        new_df = new_df.drop(columns=DeleteColumns.columns_to_delete)
        return new_df



