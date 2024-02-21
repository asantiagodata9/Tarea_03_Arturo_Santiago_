"""
Módulo de pruebas para las utilidades de manejo de excepciones.

Este módulo contiene pruebas unitarias para verificar el funcionamiento
del decorador de manejo de excepciones en src/utils.py.
"""

import pytest
from src.utils import exception_handler

def test_exception_handler():
    """
    Verifica que el decorador `exception_handler` capture y maneje correctamente las excepciones.
    
    Esta prueba asegura que cuando una función decorada lanza una excepción,
    el decorador `exception_handler` captura esa excepción y la registra correctamente.
    """
   
    @exception_handler
    def error_function():
        raise ValueError("Error intencional")

    with pytest.raises(ValueError) as exc_info:
        error_function()
    assert "Error intencional" in str(exc_info.value)
