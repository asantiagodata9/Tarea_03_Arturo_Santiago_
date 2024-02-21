
"""
Módulo para la preparación de los datos.

Este módulo contiene funciones para manejar valores faltantes y preparar los datos
para el entrenamiento y la inferencia de modelos.
"""

import logging
import pandas as pd
from src.log_config import configure_logging

def load_data(train_path, test_path):
    """
    Carga los datos de entrenamiento y prueba desde archivos CSV.

    Parámetros:
    - train_path (str): Ruta del archivo CSV que contiene los datos de entrenamiento.
    - test_path (str): Ruta del archivo CSV que contiene los datos de prueba.

    Retorna:
    - train_data (pandas.DataFrame): DataFrame que contiene los datos de entrenamiento.
    - test_data (pandas.DataFrame): DataFrame que contiene los datos de prueba.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    logging.debug("Datos de entrenamiento cargados de %s con %d filas y %d columnas", 
                  train_path, train_data.shape[0], train_data.shape[1])
    logging.debug("Datos de prueba cargados de %s con %d filas y %d columnas", 
                  test_path, test_data.shape[0], test_data.shape[1])
    return train_data, test_data

def handle_missing(features):
    """Maneja los valores faltantes en el conjunto de características.
    
    Args:
        features (pd.DataFrame): DataFrame con características que pueden tener valores faltantes.
    
    Returns:
        pd.DataFrame: DataFrame con los valores faltantes tratados.
    """
    
    # Registro inicial del número de valores faltantes por columna
    missing_values_count = features.isnull().sum()
    logging.debug("Valores faltantes por columna antes del tratamiento: %s", missing_values_count)

    # Aquí se rellenan los valores faltantes para cada columna específica
    # con un valor predeterminado o un cálculo basado en otros datos.
    features['Functional'] = features['Functional'].fillna('Typ')
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = (features.groupby('MSSubClass')
                         ['MSZoning']
                         .transform(lambda x: x.fillna(x.mode()[0])))
    features["PoolQC"] = features["PoolQC"].fillna("None")

    # Para columnas que representan características de garaje,
    # se rellenan los faltantes con 'None' o 0,
    # dependiendo de si son categóricas o numéricas.
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')

    # Lo mismo se aplica para características del sótano.
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')

    # Para 'LotFrontage', se rellena con la mediana del vecindario correspondiente.
    features['LotFrontage'] = (features.groupby('Neighborhood')
                            ['LotFrontage']
                            .transform(lambda x: x.fillna(x.median())))

    # Para el resto de columnas, se rellenan los faltantes con 'None' o 0,
    # dependiendo de si son categóricas o numéricas.
    objects_cols = features.select_dtypes(include=['object']).columns
    numeric_cols = features.select_dtypes(include=['number']).columns
    features[objects_cols] = features[objects_cols].fillna('None')
    features[numeric_cols] = features[numeric_cols].fillna(0)
    
    # Registro después del tratamiento de valores faltantes
    missing_values_count_post = features.isnull().sum()
    logging.debug("Valores faltantes por columna después del tratamiento: %s",
                  missing_values_count_post)

    return features

def prepare_data(train_path, test_path, output_path):
    """
    Prepara los datos para el entrenamiento y la prueba, 
    y guarda los datos procesados en output_path.

    Args:
        train_path (str): Ruta al archivo de datos de entrenamiento crudos.
        test_path (str): Ruta al archivo de datos de prueba crudos.
        output_path (str): Ruta donde se guardarán los datos procesados.
    """
    logging.info("Preparando los datos. Datos de entrenamiento: %s. Datos de prueba: %s",
                 train_path,
                 test_path)
    # Cargar los datos de entrenamiento y prueba
    train_data, test_data = load_data(train_path, test_path)
    # Combinar características de entrenamiento y prueba
    processed_data = (
        pd.concat([train_data.drop(['SalePrice'], axis=1), test_data])
        .reset_index(drop=True)
    )
    
    logging.debug("Datos combinados con %d filas y %d columnas",
                  processed_data.shape[0],
                  processed_data.shape[1])
    
    # Manejar valores faltantes en las características combinadas
    processed_data = handle_missing(processed_data)
    logging.info("Datos procesados guardados en %s. Tamaño final: %d filas y %d columnas", 
                 output_path, processed_data.shape[0], processed_data.shape[1])
    # Guardar las características preparadas en un archivo CSV
    processed_data.to_csv(output_path, index=False)

# No es necesario ejecutar el script si se importa como un módulo
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
