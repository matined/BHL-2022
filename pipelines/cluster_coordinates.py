import pandas as pd
from coordinates_cluster import CoordinatesConverter

df_train = pd.read_csv('data/train_processed.csv')
df_val = pd.read_csv('data/val_processed.csv')
df_test = pd.read_csv('data/test_processed.csv')

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


df_train.to_csv('data/train_processed.csv', index=False)
df_val.to_csv('data/val_processed.csv', index=False)
df_test.to_csv('data/test_processed.csv', index=False)
