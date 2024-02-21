# utils.py
"""
Este módulo contiene utilidades y decoradores comunes para manejo de excepciones y
otros propósitos generales en el proyecto.
"""

import logging
import functools
import pandas as pd

def try_except_log(func):
    """
    Decorador que maneja las excepciones y registra un error para la función decorada.

    Args:
        func (function): Función a la que se le aplicará el manejo de excepciones.

    Returns:
        function: Función envuelta con el manejo de excepciones.
    """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    raise ValueError("El DataFrame está vacío.")
                if result.shape[0] == 0:
                    raise ValueError("El DataFrame no tiene filas.")
                if result.shape[1] == 0:
                    raise ValueError("El DataFrame no tiene columnas.")
            return result
        except Exception as e:
            logging.error("Error al ejecutar la función %s: %s", func.__name__, e, exc_info=True)
            raise
    return wrapper

def exception_handler(func):
    """Decorador para manejar excepciones y registrar errores."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error("Error al ejecutar %s: %s", func.__name__, e, exc_info=True)
            raise
    return wrapper
 
def model_training_exception_handler(func):
    """Decorador para manejar excepciones específicas del entrenamiento de modelos."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error("Error al ejecutar %s: %s. Verifique parámetros e integridad de los datos.",
                          func.__name__, str(e),
                          exc_info=True)
            # Puedes decidir si quieres que el script se detenga o no
            raise e  # Descomentar si deseas que el script se detenga al encontrar un error
    return wrapper
