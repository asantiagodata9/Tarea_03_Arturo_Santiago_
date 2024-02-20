"""
Módulo de inferencia para hacer predicciones con un modelo entrenado.

Este script permite cargar un modelo entrenado, leer datos de inferencia
y escribir las predicciones a un archivo CSV.
Las rutas al modelo, a los datos de inferencia y
al archivo de predicciones son parametrizables a través de la línea de comandos.

Ejemplo de uso:
  $ python inference.py --model_path data/model/model.joblib
                        --inference_data_path data/inference/inference_data.csv
                        --predictions_path data/predictions/predictions.csv
"""

import argparse
import pandas as pd
from src.model_inference import load_model, load_inference_data, make_predictions

def main():
    """
    Ejecuta el proceso de inferencia utilizando un modelo entrenado.
    
    Este proceso incluye la carga del modelo especificado, la carga de los datos de inferencia,
    la realización de predicciones sobre esos datos, y finalmente,
    la escritura de las predicciones a un archivo CSV especificado por el usuario.
    """
    # Configurar el analizador de argumentos
    parser = argparse.ArgumentParser(description='Realizar inferencia con un modelo entrenado.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Ruta al modelo entrenado.')
    parser.add_argument('--inference_data_path', type=str, required=True,
                        help='Ruta a los datos de inferencia.')
    parser.add_argument('--predictions_path', type=str, required=True,
                        help='Ruta al archivo para guardar las predicciones.')

    args = parser.parse_args()

    # Cargar el modelo entrenado
    model = load_model(args.model_path)

    # Cargar los datos de inferencia
    inference_data = load_inference_data(args.inference_data_path)

    # Realizar predicciones
    predictions = make_predictions(model, inference_data)

    # Guardar las predicciones en un archivo CSV
    pd.DataFrame(predictions,
                 columns=['Predictions']).to_csv(args.predictions_path, index=False)
    print(f'Predictions saved to: {args.predictions_path}')

if __name__ == '__main__':
    main()
    