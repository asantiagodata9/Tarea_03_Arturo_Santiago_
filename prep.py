"""Módulo de preparación de datos para análisis o modelado de aprendizaje automático.

Este módulo realiza tareas de preparación de datos sobre conjuntos de datos crudos.
El proceso está diseñado para preparar los datos para un análisis más eficiente y para
ser utilizados en la formación de modelos de aprendizaje automático.

La función 'prepare_data' se encarga de leer los datos desde archivos CSV especificados,
aplicar las transformaciones necesarias, y guardar los datos procesados en una nueva
ubicación para su posterior uso.

Este script utiliza la función 'prepare_data' del módulo 'data_preparation'
para preparar datos crudos.
Los usuarios pueden especificar las rutas de entrada para los archivos de datos de
entrenamiento y prueba, y la ruta de salida para guardar los datos procesados.

Ejemplo de uso:
    $ python prep.py --train_path <ruta_al_archivo_de_entrenamiento> 
    --test_path <ruta_al_archivo_de_prueba> --output_path <ruta_para_guardar_datos_procesados>

Requisitos:
    - El módulo 'data_preparation' debe estar presente en el paquete 'src' y debe
      contener la función 'prepare_data'.
    - Los archivos de datos crudos deben estar ubicados en las rutas especificadas
      y en el formato adecuado para su procesamiento.
"""

import argparse
from data_preparation import prepare_data

def main():
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
