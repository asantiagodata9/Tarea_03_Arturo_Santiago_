
"""
Script para realizar inferencias/predicciones utilizando el modelo entrenado de predicción de precios de casas.
"""

from src.data_preparation import load_data, save_data
from src.config import TEST_DATA_PATH, PREDICTIONS_PATH, MODEL_PATH
import pandas as pd
import joblib

def make_predictions(model, X):
    """Realiza predicciones con el modelo entrenado."""
    predictions = model.predict(X)
    return predictions

if __name__ == "__main__":
    # Cargar el modelo entrenado
    model = joblib.load(MODEL_PATH)
    
    # Cargar los datos de inferencia
    df_test = load_data(TEST_DATA_PATH)
    
    # Suponiendo que tu df_test ya está preprocesado y listo para usar
    # Si necesitas realizar preprocesamiento específico de inferencia aquí, asegúrate de implementarlo
    
    # Realizar predicciones
    predictions = make_predictions(model, df_test)
    
    # Crear un DataFrame con las predicciones
   
    predictions_df = pd.DataFrame({
        'Id': df_test.index + 1,  # Asumiendo que el índice del DataFrame comienza en 0
        'Prediction': predictions
    })
    
    # Guardar las predicciones en un archivo CSV
    save_data(predictions_df, PREDICTIONS_PATH)
    print(f"Predicciones guardadas en {PREDICTIONS_PATH}")
    
