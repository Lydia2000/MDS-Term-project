from data_helpers import DataHolder

data = DataHolder()
train_datasets, validation_datasets, test_datasets, expected_RUL_datasets = data.get()

print(train_datasets[0])