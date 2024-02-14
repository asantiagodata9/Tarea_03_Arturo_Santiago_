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
# inference.py
from src.model_inference import load_model, load_inference_data, make_predictions

# Rutas de archivos y directorios
model_path = 'data/model/model.joblib'
inference_data_path = 'data/inference/inference_data.csv'
predictions_path = 'data/predictions/predictions.csv'

# Cargar el modelo entrenado
model = load_model(model_path)

# Cargar los datos de inferencia
inference_data = load_inference_data(inference_data_path)

# Realizar predicciones
predictions = make_predictions(model, inference_data)

# Guardar las predicciones en un archivo CSV
pd.DataFrame(predictions, columns=['Predictions']).to_csv(predictions_path, index=False)
print(f'Predictions saved to: {predictions_path}')
