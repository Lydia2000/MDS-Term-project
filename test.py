from data_helpers import DataHolder
from preprocessing_pca import fit_pca

data = DataHolder()
train_datasets, test_datasets, expected_RUL_datasets = data.get()

print(train_datasets[0])