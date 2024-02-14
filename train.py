
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
