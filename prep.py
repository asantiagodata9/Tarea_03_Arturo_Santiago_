"""Módulo de preparación de datos para análisis o modelado de aprendizaje automático.

Este módulo realiza tareas de preparación de datos sobre conjuntos de datos crudos.
El proceso está diseñado para preparar los datos para un análisis más eficiente y para
ser utilizados en la formación de modelos de aprendizaje automático.

La función 'prepare_data' se encarga de leer los datos desde archivos CSV especificados,
aplicar las transformaciones necesarias, y guardar los datos procesados en una nueva
ubicación para su posterior uso.

Ejemplo de uso:
    $ python <nombre_de_este_archivo>.py

Requisitos:
    - El módulo 'data_preparation' debe estar presente en el paquete 'src' y debe
      contener la función 'prepare_data'.
    - Los archivos de datos crudos deben estar ubicados en las rutas especificadas
      y en el formato adecuado para su procesamiento.
"""
from src.data_preparation import prepare_data

# Llamar a la función prepare_data para preparar los datos
prepare_data()
