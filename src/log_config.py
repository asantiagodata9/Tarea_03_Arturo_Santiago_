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
    # Crear el directorio de logs si no existe
    log_directory = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Configurar el timestamp y el nombre del archivo de log
    now = datetime.now()
    date_time = now.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{log_directory}/{date_time}_{script_name}.log"

    # Configurar logging
    logging.basicConfig(
        filename=log_file_name,
        level=logging.DEBUG,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.debug("Logging configurado para %s.", script_name)