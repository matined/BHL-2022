from sklearn.base import TransformerMixin


class CategoryImputer(TransformerMixin):

    def fit(self, X, y=None, **kwargs):
        self.col_to_impute = ['EASEMENT', 'EXT', 'EXMPTCL', 'EXCD2', 'EXCD1', "TAXCLASS"]
        self.impute_val = 'No'
        return self

    def transform(self, X, y=None, **kwargs):
        X[self.col_to_impute] = \
            X[self.col_to_impute].fillna(self.impute_val)
        return X
