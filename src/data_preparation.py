
"""
Módulo para la preparación de los datos.

Este módulo contiene funciones para manejar valores faltantes y preparar los datos
para el entrenamiento y la inferencia de modelos.
"""

import pandas as pd

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
    return train_data, test_data

def handle_missing(features):
    """Maneja los valores faltantes en el conjunto de características.
    
    Args:
        features (pd.DataFrame): DataFrame con características que pueden tener valores faltantes.
    
    Returns:
        pd.DataFrame: DataFrame con los valores faltantes tratados.
    """
    # Aquí se rellenan los valores faltantes para cada columna específica con un valor predeterminado o un cálculo basado en otros datos.
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

    # Para columnas que representan características de garaje, se rellenan los faltantes con 'None' o 0, dependiendo de si son categóricas o numéricas.
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

    # Para el resto de columnas, se rellenan los faltantes con 'None' o 0 dependiendo de si son categóricas o numéricas.
    objects_cols = features.select_dtypes(include=['object']).columns
    numeric_cols = features.select_dtypes(include=['number']).columns
    features[objects_cols] = features[objects_cols].fillna('None')
    features[numeric_cols] = features[numeric_cols].fillna(0)
    
    return features

def prepare_data(train_path='data/train.csv', test_path='data/test.csv'):
    """Prepara los datos para el entrenamiento y la prueba.
    
    Args:
        raw_train_path (str): Ruta al archivo de datos de entrenamiento crudos.
        raw_test_path (str): Ruta al archivo de datos de prueba crudos.
    
    Returns:
        tuple: Tupla conteniendo DataFrames de características de entrenamiento, 
               características de prueba y etiquetas de entrenamiento.
    """
    # Cargar los datos de entrenamiento y prueba
    train_data, test_data = load_data(train_path, test_path)
    
    # Combinar características de entrenamiento y prueba
    all_features = (
        pd.concat([train_data.drop(['SalePrice'], axis=1), test_data])
        .reset_index(drop=True)
    )
    
    # Manejar valores faltantes en las características combinadas
    all_features = handle_missing(all_features)
    
    # Guardar las características preparadas en un archivo CSV
    all_features.to_csv('data/prep/all_features.csv', index=False)
