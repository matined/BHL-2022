from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

class AreaExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, dimension_pairs):
        self.dimension_pairs = dimension_pairs
        print('\n>>>>>>>init() called.\n')

    def fit(self, X, y = None):
        return self

    def transform(self, df, y = None):
        for pair in self.dimension_pairs:
            prefix = pair[0][:3]
            df[prefix+"_area"] = df[pair[0]] * df[pair[1]]
        return df

class NumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.pipe = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
                  # ('area_extractor', AreaExtractor((('BLDFRONT', 'BLDDEPTH'),
                  #                                   ('LTFRONT', 'LTFRONT')))),
                  ('power_scaler', PowerTransformer())])

    def fit(self, X, y = None):
        self.pipe.fit(X)
        return self

    def transform(self, df, y = None):
        X_transformed = self.pipe.transform(df)
        df[:,:] = X_transformed
        return df

