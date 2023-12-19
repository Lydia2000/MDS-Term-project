import os
import pandas as pd
import random
import math
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import torch
from torch.utils.data import Dataset

class DataHolder:
   
    def __init__(self, config) -> None:
        torch.manual_seed(8)
        self.data_path = 'data'+ os.sep
        self.units = []
        self.config = config
        self.window_size = self.config.window_size

        # names of files
        self.train_files = [f'train_FD00{i}.txt' for i in range(1,5)]
        self.test_files = [f'test_FD00{i}.txt' for i in range(1,5)]
        self.RUL_files = [f'RUL_FD00{i}.txt' for i in range(1,5)] # Remaining useful life
        self.true_RUL_files = ['x.txt']
        
        # columns name for train, test data
        self.data_columns = ['Unit Number', 'Time (Cycles)', 
                        'OP1', 'OP2', 'OP3', 
                        'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21']
        self.RUL_columns = ['Expected RUL']

        # datasets
        self.train_datasets = []
        self.test_datasets = []
        self.validation_datasets = []
        self.expected_RUL_datasets = []
    
    def load_data(self):

        sep = ' '
        for i in range(len(self.train_files)):
            train_df = pd.read_csv(self.data_path + self.train_files[i], sep=sep, header=None)
            test_df = pd.read_csv(self.data_path + self.test_files[i], sep=sep, header = None)
            RUL_df = pd.read_csv(self.data_path + self.RUL_files[i], sep=sep, header = None)
            
            # Remove NAN
            train_df = train_df.drop(columns = [26, 27])
            test_df = test_df.drop(columns = [26, 27])
            RUL_df = RUL_df.drop(columns = [1])
            
            # Set columns name
            train_df.columns = self.data_columns
            test_df.columns = self.data_columns
            RUL_df.columns = self.RUL_columns

            # set units
            units = set(train_df['Unit Number'])
            self.units.append(units)
            if i == 1:
                units.remove(260) # test_dataset2 has miss value of unit 260
            if i == 3:
                units.remove(249) #  test_dataset4 has miss value of unit 249

            # Add y in train, val 
            train_df = self.calculate_train_RUL(train_df)
            
            # Add y in test_df
            test_df = self.calculate_test_RUL(test_df, RUL_df, units)

            # Split train and validation dataset
            n = train_df.loc[len(train_df)-1,"Unit Number"]
            random_numbers = random.sample(range(n+1), math.floor(n*0.2))
            validation_df = train_df[train_df["Unit Number"].isin(random_numbers)]
            train_df = train_df[~train_df["Unit Number"].isin(random_numbers)]
           
            # Reset the index for both train_set and validation_set
            train_df = train_df.reset_index(drop=True)
            validation_df = validation_df.reset_index(drop=True)

            # Appending dataframe to dataset list
            self.train_datasets.append(train_df)
            self.validation_datasets.append(validation_df)
            self.test_datasets.append(test_df)
            self.expected_RUL_datasets.append(RUL_df)

    def calculate_train_RUL(self, train_df):

        units_id = set(train_df['Unit Number']) # if dataset is 1 training set still can use unit_id:260 to train model
        rul_list = []
        for unit in units_id:
            time_list = np.array(train_df[train_df['Unit Number'] == unit]['Time (Cycles)'])
            length = len(time_list)
            rul = list(length - time_list)
            rul_list += rul
        train_df['Expected RUL'] = rul_list

        return train_df
    
    def calculate_test_RUL(self, test_df, RUL_df, units):
        
        rul_list = []
        for unit in units:
            time_list = np.array(test_df[test_df['Unit Number'] == unit]['Time (Cycles)'])
            length = len(time_list)
            test_rul = RUL_df.iloc[unit-1].item()
            rul = list(length - time_list + test_rul)
            rul_list += rul
        test_df['Expected RUL'] = rul_list

        return test_df
    
    def fit_pca(self):
        self.trainDatasetsCopy = copy.deepcopy(self.train_datasets)
        self.validationDatasetsCopy = copy.deepcopy(self.validation_datasets)
        self.testDatasetsCopy = copy.deepcopy(self.test_datasets)
        
        scaler = []

        for i in range(4):
            sc = StandardScaler()
            scaler.append(sc)

        for i in range(4):
            self.trainDatasetsCopy[i].iloc[:, 2:-1] = scaler[i].fit_transform(self.trainDatasetsCopy[i].iloc[:, 2:-1])
            self.validationDatasetsCopy[i].iloc[:, 2:-1] = scaler[i].transform(self.validationDatasetsCopy[i].iloc[:, 2:-1])
            self.testDatasetsCopy[i].iloc[:, 2:-1] = scaler[i].transform(self.testDatasetsCopy[i].iloc[:, 2:-1])
        
        pca = PCA(n_components = self.config.n_components)

        newColumns = [f'PCA{i}' for i in range(1, self.config.n_components+1)]

        for i in range(4):
            # Finding Principal Components
            temp1 = pca.fit_transform(self.trainDatasetsCopy[i].iloc[:, 2:-1])
            temp2 = pca.transform(self.validationDatasetsCopy[i].iloc[:, 2:-1])
            temp3 = pca.transform(self.testDatasetsCopy[i].iloc[:, 2:-1])

            # Converting to Dataframes
            temp1 = pd.DataFrame(temp1, columns = newColumns)
            temp2 = pd.DataFrame(temp2, columns = newColumns)
            temp3 = pd.DataFrame(temp3, columns = newColumns)

            # Dropping Excess Data
            self.trainDatasetsCopy[i].drop(inplace = True, columns = self.trainDatasetsCopy[i].columns[2:-1])
            self.validationDatasetsCopy[i].drop(inplace = True, columns = self.validationDatasetsCopy[i].columns[2:-1])
            self.testDatasetsCopy[i].drop(inplace = True, columns = self.testDatasetsCopy[i].columns[2:-1])

            # Merging New Data
            self.trainDatasetsCopy[i] = pd.merge(self.trainDatasetsCopy[i], temp1, left_index=True, right_index=True)
            self.validationDatasetsCopy[i] = pd.merge(self.validationDatasetsCopy[i], temp2, left_index=True, right_index=True)
            self.testDatasetsCopy[i] = pd.merge(self.testDatasetsCopy[i], temp3, left_index=True, right_index=True)

        return self.trainDatasetsCopy, self.validationDatasetsCopy, self.testDatasetsCopy
    
    def get(self, dataset_index):

        self.load_data()

        # Access the datasets based on the specified index
        self.train_datasets, self.validation_datasets, self.test_datasets = self.fit_pca()
        train_df = self.train_datasets[dataset_index]
        valid_df = self.validation_datasets[dataset_index]
        test_df = self.test_datasets[dataset_index]

        # Get indices for training and validation datasets based on window_size
        window = self.window_size
        train_indices = list(train_df[(train_df['Expected RUL'] >= (window - 1)) & (train_df['Time (Cycles)'] > 10)].index)
        val_indices = list(valid_df[(valid_df['Expected RUL'] >= (window - 1)) & (valid_df['Time (Cycles)'] > 10)].index)
        
        # Create instances of custom datasets using the obtained indices
        train_dataset = CustomDataset(train_indices, train_df)
        val_dataset = CustomDataset(val_indices, valid_df)
        min_unit_id = min(self.units[dataset_index])
        max_unit_id = max(self.units[dataset_index])
        units = np.arange(min_unit_id, max_unit_id+1)
        test_dataset = TestDataset(units, test_df)

        return train_dataset, val_dataset, test_dataset

    def get_raw_data(self, dataset_index):

        return train_df, valid_df, test_df

class CustomDataset(Dataset):
    """
    Custom dataset class for handling data.

    Args:
        list_indices (list): List of indices to use for the dataset.
        df_train (pandas.DataFrame): Training dataframe.
    """
    def __init__(self, list_indices, df_train):
        
        self.indices = list_indices
        self.df_train = df_train
        
    def __len__(self):
        
        return len(self.indices)
    
    def __getitem__(self, idx):
        
        ind = self.indices[idx]
        
        X_ = self.df_train.iloc[ind : ind + 20, :].copy()
        y_ = self.df_train.iloc[ind + 19]['Expected RUL']
        X_ = X_.drop(['Unit Number','Time (Cycles)','Expected RUL'], axis = 1).to_numpy()

        return X_, y_
    
class TestDataset(Dataset):
    
    def __init__(self, units, df_test):
        
        self.units = units
        self.df_test = df_test
        
    def __len__(self):
        
        return len(self.units)
    
    def __getitem__(self, idx):
        
        n = self.units[idx]
        U = self.df_test[self.df_test['Unit Number'] == n].copy()
        X_ = U.reset_index().iloc[-20:,:].drop(['index','Unit Number','Time (Cycles)','Expected RUL'], axis = 1).copy().to_numpy()
        y_ = U['Expected RUL'].min()
        
        return X_, y_

class fit_ElasticNet:
   
    def __init__(self) -> None:
        data = DataHolder()
        train_datasets, validation_datasets, test_datasets, expected_RUL_datasets = data.get()
        self.train_datasets = train_datasets
        self.validation_datasets = validation_datasets
        self.test_datasets = test_datasets
        self.expected_RUL_datasets = expected_RUL_datasets
    
    def Calculate_RUL(self,df):
        max_cycles = df.groupby('Unit Number')['Time (Cycles)'].max()
        merged = df.merge(max_cycles.to_frame(name='max_time_cycle'), left_on='Unit Number',right_index=True)
        merged["RUL"] = merged["max_time_cycle"] - merged['Time (Cycles)']
        merged = merged.drop("max_time_cycle", axis=1)
        return merged
    
    def en(self):
        
        self.trainDatasetsCopy = copy.deepcopy(self.train_datasets)
        self.validationDatasetsCopy = copy.deepcopy(self.validation_datasets)
        self.testDatasetsCopy = copy.deepcopy(self.test_datasets)

        scaler = []

        for i in range(4):
            sc = StandardScaler()
            scaler.append(sc)
        
        for i in range(4):
            self.trainDatasetsCopy[i].iloc[:, 2:] = scaler[i].fit_transform(self.trainDatasetsCopy[i].iloc[:, 2:])
            self.validationDatasetsCopy[i].iloc[:, 2:] = scaler[i].fit_transform(self.validationDatasetsCopy[i].iloc[:, 2:])
            self.testDatasetsCopy[i].iloc[:, 2: ] = scaler[i].transform(self.testDatasetsCopy[i].iloc[:, 2:])

        for i in range(4):
            self.trainDatasetsCopy[i] = self.Calculate_RUL(self.trainDatasetsCopy[i])
        
        for i in range(4):

            # fit elastic_net
            elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000) # alpha(lamda) = 1, l1_ratio=0.5 is package default
            elastic_net.fit(self.trainDatasetsCopy[i].loc[:,"OP1":"S21"], self.trainDatasetsCopy[i].loc[:,"RUL"])

            # rank feature by absolute value of coefficients
            coef = pd.DataFrame(elastic_net.coef_, index=elastic_net.feature_names_in_,columns=["coefficient"])
            sort_features_name = abs(coef).sort_values(ascending=False, by="coefficient").index.tolist()
            # print(abs(coef).sort_values(ascending=False, by="coefficient"))

            top_10_features = []
            for j in range(10):
                top_10_features.append(sort_features_name[j])

            temp1 = self.trainDatasetsCopy[i].loc[:,top_10_features]
            temp2 = self.validationDatasetsCopy[i].loc[:,top_10_features]
            temp3 = self.testDatasetsCopy[i].loc[:,top_10_features]

            # Converting to Dataframes
            temp1 = pd.DataFrame(temp1, columns = top_10_features)
            temp2 = pd.DataFrame(temp2, columns = top_10_features)
            temp3 = pd.DataFrame(temp3, columns = top_10_features)


            # Dropping Excess Data
            self.trainDatasetsCopy[i].drop(inplace = True, columns = self.trainDatasetsCopy[i].columns[2:])
            self.validationDatasetsCopy[i].drop(inplace = True, columns = self.validationDatasetsCopy[i].columns[2:])
            self.testDatasetsCopy[i].drop(inplace = True, columns = self.testDatasetsCopy[i].columns[2:])

            # Merging New Data
            self.trainDatasetsCopy[i] = pd.merge(self.trainDatasetsCopy[i], temp1, left_index=True, right_index=True)
            self.validationDatasetsCopy[i] = pd.merge(self.validationDatasetsCopy[i], temp2, left_index=True, right_index=True)
            self.testDatasetsCopy[i] = pd.merge(self.testDatasetsCopy[i], temp3, left_index=True, right_index=True)

    def result_df(self):
        self.en()
        return self.trainDatasetsCopy, self.validationDatasetsCopy, self.testDatasetsCopy, self.expected_RUL_datasets

if __name__=="__main__":

    # print(valid_df)
    # train_indices = list(train_data[(train_data['rul'] >= (window - 1)) & (train_data['time'] > 10)].index)
    # val_indices = list(val_data[(val_data['rul'] >= (window - 1)) & (val_data['time'] > 10)].index)

    # print(train_df)
    # print(data_holder.train_datasets[0].shape)
    # print(data_holder.validation_datasets[0].shape)
    # train_dataset = CustomDataset(train_indices, train_df)
    # print(train_dataset[0])
    # print(data_holder.train_datasets[0].shape)
    pass