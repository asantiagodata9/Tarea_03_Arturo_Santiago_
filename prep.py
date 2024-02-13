# prep.py
from src.data_preparation import load_data, preprocess_data, save_data

if __name__ == "__main__":
    # Asume que ya tienes definidas las rutas de los archivos
    raw_data_path = 'data/raw/train.csv'
    processed_data_path = 'data/prep/train_processed.csv'
    
    df_raw = load_data(raw_data_path)
    df_processed = preprocess_data(df_raw)
    save_data(df_processed, processed_data_path)
    