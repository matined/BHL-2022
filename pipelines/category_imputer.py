from sklearn.base import TransformerMixin


class CategoryImputer(TransformerMixin):

    def __init__(self):
        self.col_to_impute = ['EASEMENT', 'EXT',
                              'EXMPTCL', 'EXCD2', 'EXCD1', "TAXCLASS"]
        self.impute_val = 'No'

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        X[self.col_to_impute] = \
            X[self.col_to_impute].fillna(self.impute_val)
        return X
