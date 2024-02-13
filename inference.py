
from src.model_inference import make_predictions

if __name__ == "__main__":
    make_predictions('model.joblib', 'data/inference/inference_data.csv', 'data/predictions/predictions.csv')
