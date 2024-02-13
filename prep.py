
from src.data_preparation import prepare_data

if __name__ == "__main__":
    train_features, test_features, train_labels = prepare_data('data/raw/train.csv', 'data/raw/test.csv')
    train_features.to_csv('data/prep/train_features.csv', index=False)
    test_features.to_csv('data/prep/test_features.csv', index=False)
    train_labels.to_csv('data/prep/train_labels.csv', index=False)
