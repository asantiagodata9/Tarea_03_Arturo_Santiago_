
"""Módulo para la preparación de los datos.

Este módulo contiene funciones para manejar valores faltantes y preparar los datos
para el entrenamiento y la inferencia de modelos.
"""

import pandas as pd

def handle_missing(features):
    """Maneja los valores faltantes en el conjunto de características.
    
    Args:
        features (pd.DataFrame): DataFrame con características que pueden tener valores faltantes.
    
    Returns:
        pd.DataFrame: DataFrame con los valores faltantes tratados.
    """
    features['Functional'] = features['Functional'].fillna('Typ')
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    features["PoolQC"] = features["PoolQC"].fillna("None")
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    objects_cols = features.select_dtypes(include=['object']).columns
    numeric_cols = features.select_dtypes(include=['number']).columns
    features[objects_cols] = features[objects_cols].fillna('None')
    features[numeric_cols] = features[numeric_cols].fillna(0)
    return features

def prepare_data(raw_train_path, raw_test_path):
    """Prepara los datos para el entrenamiento y la prueba.
    
    Args:
        raw_train_path (str): Ruta al archivo de datos de entrenamiento crudos.
        raw_test_path (str): Ruta al archivo de datos de prueba crudos.
    
    Returns:
        tuple: Tupla conteniendo DataFrames de características de entrenamiento, características de prueba y etiquetas de entrenamiento.
    """
    train_data = pd.read_csv(raw_train_path)
    test_data = pd.read_csv(raw_test_path)
    train_labels = train_data['SalePrice'].reset_index(drop=True)
    train_features = train_data.drop(['SalePrice'], axis=1)
    test_features = test_data
    all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
    all_features = handle_missing(all_features)
    num_train = train_data.shape[0]
    train_features = all_features[:num_train]
    test_features = all_features[num_train:]
    return train_features, test_features, train_labels
