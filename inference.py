"""Módulo de inferencia para hacer predicciones con un modelo entrenado.

Este módulo carga un modelo de aprendizaje automático desde un archivo .joblib,
lee datos de inferencia desde un archivo CSV, y escribe las predicciones del modelo
a otro archivo CSV. Está diseñado para ser utilizado como un script independiente
que facilita la aplicación de modelos entrenados a nuevos datos para generar
predicciones.

Ejemplo de uso:
    $ python <nombre_de_este_archivo>.py

Requisitos:
    - El módulo 'model_inference' debe estar presente en el paquete 'src' y debe
      contener la función 'make_predictions'.
    - Los archivos de entrada y salida deben estar correctamente ubicados en
      las rutas especificadas.
"""

import pandas as pd  # Asegurar que pandas está importado
from src.model_inference import load_model, load_inference_data, make_predictions

# Rutas de archivos y directorios
MODEL_PATH = 'data/model/model.joblib'
INFERENCE_DATA_PATH = 'data/inference/inference_data.csv'
PREDICTIONS_PATH = 'data/predictions/predictions.csv'

# Cargar el modelo entrenado
model = load_model(MODEL_PATH)

# Cargar los datos de inferencia
inference_data = load_inference_data(INFERENCE_DATA_PATH)

# Realizar predicciones
predictions = make_predictions(model, inference_data)

# Guardar las predicciones en un archivo CSV
pd.DataFrame(predictions, columns=['Predictions']).to_csv(PREDICTIONS_PATH, index=False)
print(f'Predictions saved to: {PREDICTIONS_PATH}')
