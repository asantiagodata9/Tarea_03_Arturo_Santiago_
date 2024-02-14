"""
Script para entrenar un modelo de aprendizaje automático utilizando características categóricas.

Este script importa la función 'train_model' del módulo 'src.model_training'
para entrenar un modelo en los datos proporcionados.
Requiere las rutas al archivo CSV de características de entrenamiento,
al archivo CSV de etiquetas de entrenamiento,
una lista de nombres de características categóricas, 
y la ruta deseada para guardar el modelo entrenado.

Uso:
    $ python train.py

Ejemplo:
    $ python train.py

    Después de una ejecución exitosa, el modelo entrenado se guardará en la ubicación especificada.

Argumentos:
    Ninguno

Devuelve:
    Ninguno

Nota:
    - Asegúrese de que la función 'train_model' de 'src.model_training' 
    esté implementada correctamente.
    - Asegúrese de que las rutas a los archivos de características y
    etiquetas de entrenamiento sean correctas.
    - Actualice 'cat_features' con los nombres reales de las características categóricas.

"""
from src.model_training import train_model

if __name__ == "__main__":
    # Update this list with actual categorical feature names
    cat_features = ['a list of categorical feature names goes here']
    train_model('data/prep/train_features.csv',
                'data/prep/train_labels.csv',
                cat_features,
                'model.joblib')

    # Guardar el modelo entrenado
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
