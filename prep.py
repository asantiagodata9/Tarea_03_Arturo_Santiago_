"""
Módulo de preparación de datos para análisis o modelado de aprendizaje automático.

Este script utiliza la función 'prepare_data' del módulo 'data_preparation'
para preparar datos crudos. Los usuarios pueden especificar las rutas de entrada
para los archivos de datos de entrenamiento y prueba, y la ruta de salida para
guardar los datos procesados.

Ejemplo de uso:
    $ python prep.py 
    --train_path <ruta_al_archivo_de_entrenamiento>
    --test_path <ruta_al_archivo_de_prueba>
    --output_path <ruta_para_guardar_datos_procesados>
"""

import argparse
import logging
from data_preparation import prepare_data
from src.log_config import configure_logging

# Configuración de logging para este script
configure_logging('prep')

def main():
    """
    Ejecuta la preparación de datos basándose en los argumentos
    de línea de comando proporcionados.
    """
    # Configuración del analizador de argumentos
    parser = argparse.ArgumentParser(
        description='Preparar datos para análisis o modelado de aprendizaje automático.'
    )
    parser.add_argument('--train_path', type=str, required=True, help=
                        'Ruta del archivo de datos de entrenamiento crudos.')
    parser.add_argument('--test_path', type=str, required=True, help=
                        'Ruta del archivo de datos de prueba crudos.')
    parser.add_argument('--output_path', type=str, required=True, help=
                        'Ruta para guardar los datos procesados.')

    args = parser.parse_args()

    # Inicio del proceso de preparación de datos
    logging.info('Iniciando la preparación de datos desde %s y %s.',
                 args.train_path,
                 args.test_path)

    try:
        # Llamar a la función prepare_data para preparar los datos
        prepare_data(train_path=args.train_path,
                     test_path=args.test_path,
                     output_path=args.output_path)
        logging.info('Datos procesados guardados correctamente en %s.',
                     args.output_path)
    except Exception as e:
        logging.error('Ocurrió un error durante la preparación de datos: %s', str(e))
        raise

if __name__ == '__main__':
    main()
