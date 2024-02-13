"""
Este módulo contiene funciones para cargar, procesar y guardar datos para el proyecto de predicción de precios de casas.

Funciones:
- load_data: Carga datos desde un archivo CSV.
- preprocess_data: Realiza la limpieza y preparación de los datos.
- save_data: Guarda el DataFrame procesado en un archivo CSV.
"""

# src/data_preparation.py
import pandas as pd

def load_data(filepath):
    """
    Carga los datos desde un archivo CSV.

    Parámetros:
    - filepath (str): Ruta completa al archivo CSV.

    Retorna:
    - DataFrame: Datos cargados en un DataFrame de pandas.
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Realiza la limpieza y preparación de los datos.

    Parámetros:
    - df (DataFrame): DataFrame original a procesar.

    Retorna:
    - DataFrame: DataFrame procesado.
    """
    df.fillna(0, inplace=True)
    return df

def save_data(df, filepath):
    """
    Guarda el DataFrame procesado en un archivo CSV.

    Parámetros:
    - df (DataFrame): DataFrame a guardar.
    - filepath (str): Ruta del archivo donde se guardará el DataFrame.
    """
    df.to_csv(filepath, index=False)
    