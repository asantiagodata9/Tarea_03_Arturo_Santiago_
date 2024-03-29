"""
Este módulo es parte de un sistema de entrenamiento de modelos de machine learning.

Permite entrenar un modelo CatBoostRegressor utilizando datos preparados 
y configuraciones específicas definidas en un archivo YAML. 
El usuario puede especificar las rutas de los archivos de datos preparados, los datos de
entrenamiento con etiquetas, y la ubicación para guardar el modelo entrenado 
a través de argumentos de línea de comandos. 
Este enfoque facilita la automatización y la flexibilidad en el entrenamiento de modelos.
"""

import argparse
import logging
import yaml
from src.model_training import load_prepared_data, train_model, get_categorical_features
from src.log_config import configure_logging

# Configuración de logging para este script
configure_logging('train')

def main():
    """
    Script principal para entrenar un modelo de CatBoostRegressor.
    
    Este script permite entrenar un modelo utilizando datos preparados y etiquetas de entrenamiento.
    Utiliza un archivo de configuración YAML para los hiperparámetros del modelo
    y permite la especificación
    de rutas a través de argumentos de línea de comandos para los datos preparados,
    datos de entrenamiento, y la ubicación para guardar el modelo entrenado.
    
    Ejemplo de uso:
        python train.py --prepared_data_path data/prep/all_features.csv
                        --train_data_path data/train.csv
                        --model_path data/model/model.joblib
    """

    # Configura el analizador de argumentos para aceptar rutas de archivos y directorios
    parser = argparse.ArgumentParser(description="Entrena un modelo de CatBoostRegressor.")
    parser.add_argument('--config_path', type=str, default='config.yml',
                        help='Ruta del archivo de configuración YAML.')
    parser.add_argument('--prepared_data_path', type=str, required=True,
                        help='Ruta del archivo CSV que contiene los datos preparados.')
    parser.add_argument('--train_data_path',
                        type=str,
                        required=True,
                        help='Ruta archivo CSV que con los datos de entrenamiento con etiquetas.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Ruta donde se guardará el modelo entrenado.')

    args = parser.parse_args()

    logging.info("Iniciando proceso de entrenamiento.")

    # Cargar el archivo de configuración YAML para obtener los hiperparámetros del modelo
    logging.info("Cargando configuración del modelo desde %s.", args.config_path)
    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    logging.info("Configuración del modelo cargada correctamente.")

    # Extraer los hiperparámetros para el modelo CatBoostRegressor desde el archivo de configuración
    model_params = config['catboost_regressor_hyperparameters']

    # Cargar los datos preparados y las etiquetas de entrenamiento desde las rutas especificadas
    logging.info("Cargando datos preparados desde %s.", args.prepared_data_path)
    train_data = load_prepared_data(args.prepared_data_path)
    train_labels = load_prepared_data(args.train_data_path)['SalePrice']
    train_features = train_data.iloc[:len(train_labels), :]
    cat_features = get_categorical_features(train_features)
    logging.info("Datos preparados cargados correctamente.")

    # Entrenar el modelo con los datos, hiperparámetros, y características categóricas especificadas
    # Guardar el modelo entrenado en la ruta especificada
    logging.info("Iniciando entrenamiento del modelo.")
    train_model(train_features,
                train_labels,
                cat_features,
                model_params,
                model_path=args.model_path)
    logging.info("Modelo entrenado y guardado en %s.", args.model_path)

if __name__ == '__main__':
    main()
