# utils.py

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
            logging.error(f"Error al ejecutar {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper