import pickle as pkl


class CoordinatesConverter:

    def __init__(self):
        with open('pipelines/clustering/kmeans_coords.pkl', 'rb') as f:
            self.clustering = pkl.load(f)

    def convert(self, df):
        coordinates = df[['Longitude', 'Latitude']]
        return self.clustering.predict(coordinates)
