# Przewidywanie cen posiadłości 

## 1. Feature Engineering - pipeline

Feature Engineering został rozwiązany za pomocą pipelinów z sklearna. W związku z tym każda z poniższych klas posiada metodę `fit()` i `transform()`. W niektórych pipelinach wywołanie `fit()` nic nie robi.

### `address_to_coordinates.CoordinatesFromAddress`
    Klasa imputuje współrzędne korzystając z adresu korzystając z mapbox API.

### `category_imputer.CategoryImputer`
    Imputacja wartości 'No' w miejsce nulli, ponieważ te braki nie są wywołane błędami.

### `category_encoder.CategoryEncoder`
    One hot encoding kolumn kategorycznych.

### `category_merger.CategoryMerger`
    Łączy kategorie w niktórych kolumnach kategorycznych.

### `coordinates_cluster.CoordinatesConverter`
    Przy użyciu metody k średnich kategoryzuje zmienne odpowiadające współrzędnym geograficznym nieruchomości.

### `delete_columns.DeleteColumns`
    Usuwa niepotrzebne kolumny.

### `numeric_transformers.NumericTransformer`
    Standaryzuje numeryczne kolumny w taki sposób, że mają średnią 0 i odchylenie standardowe 1

### `numeric_transformers.AreaExtractor`
    Dodaje dodatkowe kolumny oznaczające pola posiadłości i działki, na której jest ta nieruchomość.

### `preprocess`
    Skrypt preprocesujący całe dane wykorzystując wszystkie powyższe pipeliny.

## 2. Model
Modelem jest xgboost. Aby użyć model należy skorzystać z klasy `interface.Predictor` i wywołać metodę predict. Wszystkie pipeliny są wczytywane z plików `.pkl`.

