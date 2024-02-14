"""
Este script entrena un modelo utilizando datos preparados y etiquetas de entrenamiento.
Carga datos preparados desde un archivo CSV, obtiene las características categóricas
y entrena un modelo CatBoostRegressor.

Variables:
- prepared_data_path (str): Ruta del archivo CSV que contiene los datos preparados.
- train_data_path (str): Ruta del archivo CSV que contiene los datos de entrenamiento con etiquetas.
- model_path (str): Ruta donde se guardará el modelo entrenado.
- train_data (pandas.DataFrame): DataFrame que contiene los datos preparados.
- train_labels (pandas.Series): Series que contiene las etiquetas de entrenamiento.
- train_features (pandas.DataFrame): DataFrame que contiene las características de entrenamiento.
- cat_features (list): Lista de nombres de características categóricas.
"""
# train.py
from src.model_training import load_prepared_data, train_model, get_categorical_features

# Rutas de archivos y directorios
prepared_data_path = 'data/prep/all_features.csv'
train_data_path = 'data/train.csv'
model_path = 'data/model/model.joblib'  # Asegurarse de que corresponde a model.joblib

# Cargar los datos preparados y las etiquetas de entrenamiento
train_data = load_prepared_data(prepared_data_path)
train_labels = load_prepared_data(train_data_path)['SalePrice']
# Asegurarse de que solo se usan las filas correspondientes a train_labels para train_features
train_features = train_data.iloc[:len(train_labels), :]
cat_features = get_categorical_features(train_features)

# Entrenar el modelo y guardar en la ruta especificada
train_model(train_features, train_labels, cat_features, model_path=model_path)
