import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class CategoryEncoder(TransformerMixin):

    def fit(self, X, y=None, **kwargs):
        self.one_hot_cols = ['BLDGCL', 'EXT', 'EASEMENT', 'EXCD2', 'EXCD1', "TAXCLASS"]
        self.one_hot_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.one_hot_transformer.fit(X[self.one_hot_cols].astype('str'))
        return self

    def map_exemption(self, ex):
        if ex == 'No':
            return 0
        else:
            return ex[1]

    def transform(self, X, y=None, **kwargs):
        X = X.copy()
        X['EXMPTCL'] = X['EXMPTCL'].apply(self.map_exemption)
        one_hot_transformed = self.one_hot_transformer.transform(
            X[self.one_hot_cols].astype('str')
        )
        one_hot_df = pd.DataFrame(one_hot_transformed,
                                  columns=self.one_hot_transformer.get_feature_names(
                                      self.one_hot_transformer.feature_names_in_
                                  )
                                  )
        return pd.concat([X, one_hot_df], axis='columns')
