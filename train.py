
"""
Script para entrenar el modelo de predicción de precios de casas utilizando CatBoost.
"""
from src.data_preparation import load_data
from src.config import PROCESSED_DATA_PATH, MODEL_PATH
from catboost import CatBoostRegressor
import joblib

def train_model(X, y):
    """
    Entrena el modelo CatBoost con los datos proporcionados.

    Parámetros:
    - X (DataFrame): Características/variables independientes.
    - y (Series): Variable objetivo/dependiente.

    Retorna:
    - model: Modelo entrenado.
    """
    
    """Entrenamiento del modelo CatBoost."""
    model = CatBoostRegressor(iterations=1000,
                              learning_rate=0.1,
                              depth=6,
                              loss_function='RMSE',
                              verbose=200)
    model.fit(X, y, verbose=100)
    return model

if __name__ == "__main__":
    # Cargar los datos preprocesados
    df_train = load_data(PROCESSED_DATA_PATH)

    # Preparar los datos para el entrenamiento
    # Asegúrate de ajustar 'SalePrice' a tu columna objetivo si es diferente
    X = df_train.drop(columns=['SalePrice'])
    y = df_train['SalePrice']

    # Entrenar el modelo
    model = train_model(X, y)

    # Guardar el modelo entrenado
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    