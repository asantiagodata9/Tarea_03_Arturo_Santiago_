"""
Este módulo configura el logging para diferentes scripts del proyecto.

Proporciona una función de utilidad para configurar un logger específico para cada script,
incluyendo la configuración del nombre del archivo de log con un timestamp y el nivel de logging.
El logger captura mensajes de debug y superiores
y escribe en un archivo de log dentro del directorio 'logs'.
"""

import logging
import os
from datetime import datetime

def configure_logging(script_name):
    """
    Configura el logging para el script dado.

    Args:
        script_name (str): El nombre del script para el cual
        se está configurando el logging.
    """
    # Configurar el timestamp y el nombre del archivo de log
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")
    log_directory = './logs'
    log_file_name = f"{log_directory}/{date_time}_{script_name}.log"

    # Crear el directorio de logs si no existe
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Configurar logging
    logging.basicConfig(
        filename=log_file_name,
        level=logging.DEBUG,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logging.debug("Logging configurado para %s.", script_name)
    