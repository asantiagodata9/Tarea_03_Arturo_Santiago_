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
from src.model_inference import make_predictions

if __name__ == "__main__":
    make_predictions('model.joblib',
                     'data/inference/inference_data.csv',
                     'data/predictions/predictions.csv')
