import argparse
import yaml
from src.model_training import load_prepared_data, train_model, get_categorical_features

def main():
    """
    Script principal para entrenar un modelo de CatBoostRegressor.
    
    Este script permite entrenar un modelo utilizando datos preparados y etiquetas de entrenamiento.
    Utiliza un archivo de configuración YAML para los hiperparámetros del modelo y permite la especificación
    de rutas a través de argumentos de línea de comandos para los datos preparados, datos de entrenamiento,
    y la ubicación para guardar el modelo entrenado.
    
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
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Ruta del archivo CSV que contiene los datos de entrenamiento con etiquetas.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Ruta donde se guardará el modelo entrenado.')

    args = parser.parse_args()

    # Cargar el archivo de configuración YAML para obtener los hiperparámetros del modelo
    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Extraer los hiperparámetros para el modelo CatBoostRegressor desde el archivo de configuración
    model_params = config['catboost_regressor_hyperparameters']

    # Cargar los datos preparados y las etiquetas de entrenamiento desde las rutas especificadas
    train_data = load_prepared_data(args.prepared_data_path)
    train_labels = load_prepared_data(args.train_data_path)['SalePrice']
    train_features = train_data.iloc[:len(train_labels), :]
    cat_features = get_categorical_features(train_features)

    # Entrenar el modelo con los datos, hiperparámetros, y características categóricas especificadas
    # Guardar el modelo entrenado en la ruta especificada
    train_model(train_features, train_labels, cat_features, model_params, model_path=args.model_path)

if __name__ == '__main__':
    main()
