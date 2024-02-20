"""Módulo de preparación de datos para análisis o modelado de aprendizaje automático.

Este script utiliza la función 'prepare_data' del módulo 'data_preparation'
para preparar datos crudos.
Los usuarios pueden especificar las rutas de entrada para los archivos
de datos de entrenamiento y prueba,
y la ruta de salida para guardar los datos procesados.

Ejemplo de uso:
    $ python prep.py 
    --train_path <ruta_al_archivo_de_entrenamiento> 
    --test_path <ruta_al_archivo_de_prueba> 
    --output_path <ruta_para_guardar_datos_procesados>
"""

import logging
import os
import argparse
from data_preparation import prepare_data

# Configuración del logger según lo especificado
logging.basicConfig(
    # Cambié la ruta a './logs/results.log' para seguir tu estructura de directorios
    filename='./logs/results.log',
    level=logging.DEBUG, # Nivel del logging
    filemode='w', #Sobreescribir el archivo del log existente
    format='%(name)s - %(levelname)s - %(message)s'
)

# Crear el directorio de logs si no existe
if not os.path.exists('./logs'):
    os.makedirs('./logs')

def main():
    """Ejecuta la preparación de datos basándose en los argumentos
    de línea de comando proporcionados."""
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

    # Llamar a la función prepare_data para preparar los datos
    prepare_data(train_path=args.train_path, test_path=args.test_path, output_path=args.output_path)

if __name__ == '__main__':
    main()
