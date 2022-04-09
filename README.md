# Przewidywanie cen posiadłości 

## 1. Feature Engineering - pipeline

### address_to_coordinates.CoordinatesFromAddress()
Brakujące współrzędne geograficzne uzupełniamy korzystając z adresu i mapbox API.

### category_encoder.CategoryEncoder()
Zamienia kategoryczne kolumny na kodowanie one-hot.

### category_imputer.CategoryImputer()
Brakujące kategorie uzupełnia "No".

### category_merger.CategoryMerger()
Łączy kategorie w niktórych kolumnach kategorycznych.

### coordinates_cluster.CoordinatesConverter()
Dzieli współrzędne na klastry.

### delete_columns.DeleteColumns()
Usuwa niktóre kolumny.

### numeric_transformers.NumericTransformer()
Standaryzuje numeryczne kolumny.

### numeric_transformers.AreaExtractor()
Dodaje dodatkowe kolumny.

### preprocess
Skrypt preprocesujący całe dane.

## 2. Model
Aby użyć model należy:
+ utworzyć obiekt klay `pipelines.interface.Predictor` 
+ wywołać na nim metode `predict()`.

