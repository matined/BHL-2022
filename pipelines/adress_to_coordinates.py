import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
import requests
import json

class CoordinatesFromAddress(TransformerMixin):

    TOKEN = "pk.eyJ1IjoibWF0aW5lZCIsImEiOiJjbDFxbnMwOHoxNmlxM2VwZHFnOWt0aWRjIn0.TSltvTw-ucpQyItz12SYsw"
    BB_COORDS = '-74.490421,40.927855,-73.351210,40.362712'

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **trans_params):
        new_df = df.copy()

        addresses = np.array(new_df['STADDR']).astype(str)
        latitude = np.array(new_df['Latitude'])
        longitude = np.array(new_df['Longitude'])
        
        for i in range(addresses.shape[0]):
            if np.isnan(latitude[i]) or np.isnan(longitude[i]):
                coords = self.get_coords(addresses[i])
                longitude[i] = coords[0]
                latitude[i] = coords[1]

        new_df['Latitude'] = latitude   
        new_df['Longitude'] = longitude 

        return new_df

    def convert_address(self, address):
        return address.lower().replace(" ", "%")

    def get_coords(self, address):
        response = requests.get(f"https://api.mapbox.com/geocoding/v5/mapbox.places/{self.convert_address(address)}.json?bbox={CoordinatesFromAddress.BB_COORDS}&access_token={CoordinatesFromAddress.TOKEN}")
        if response.status_code == 200:
            response = json.loads(response.content)
            return response['features'][0]['geometry']['coordinates']
        else:
            return [0, 0]
    
