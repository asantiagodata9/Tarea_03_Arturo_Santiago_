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

import yaml
from src.model_training import load_prepared_data, train_model, get_categorical_features

# Rutas de archivos y directorios
PREPARED_DATA_PATH = 'data/prep/all_features.csv'
TRAIN_DATA_PATH = 'data/train.csv'
MODEL_PATH = 'data/model/model.joblib'  # Asegurarse de que corresponda a model.joblib

# Cargar el archivo de configuración YAML
with open('config.yml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Extraer los hiperparámetros para el modelo CatBoostRegressor
model_params = config['catboost_regressor_hyperparameters']

# Cargar los datos preparados y las etiquetas de entrenamiento
train_data = load_prepared_data(PREPARED_DATA_PATH)
train_labels = load_prepared_data(TRAIN_DATA_PATH)['SalePrice']
# Asegurarse de que solo se usan las filas correspondientes a train_labels para train_features
train_features = train_data.iloc[:len(train_labels), :]
cat_features = get_categorical_features(train_features)

# Entrenar el modelo y guardar en la ruta especificada
train_model(train_features, train_labels, cat_features, model_params, model_path=MODEL_PATH)
