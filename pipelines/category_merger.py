from sklearn.base import TransformerMixin


class CategoryMerger(TransformerMixin):

    def __init__(self):
        self.cat_mapping = {
            'A': 'Family',
            'C': 'Family',
            'D': 'Family',
            'G': 'Family',
            'L': 'Multifamily',
            'S': 'Multifamily',
            'B': 'Multifamily',
            'E': 'Industrial',
            'F': 'Industrial',
            'O': 'Industrial',
            'H': 'Service',
            'K': 'Service',
            'T': 'Service',
            'I': 'Public use',
            'J': 'Public use',
            'M': 'Public use',
            'N': 'Public use',
            'P': 'Public use',
            'Q': 'Public use',
            'R': 'Public use',
            'U': 'Public use',
            'V': 'Public use',
            'W': 'Public use',
            'Y': 'Public use',
            'Z': 'Public use'
        }

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        X = X.copy()
        # Merge bldl
        X['BLDGCL'] = X['BLDGCL'].apply(lambda s: s[0])
        X['BLDGCL'] = X['BLDGCL'].map(self.cat_mapping)
        return X
