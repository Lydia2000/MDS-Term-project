import os
import pandas as pd
import random
import math
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
import torch
from torch.utils.data import Dataset
from utils import plot_pca_loading

def smooth_dataframe(df):
    def smooth(s, b = 0.75):

        v = np.zeros(len(s)+1) #v_0 is already 0.
        bc = np.zeros(len(s)+1)

        for i in range(1, len(v)): #v_t = 0.95
            v[i] = (b * v[i-1] + (1-b) * s[i-1]) 
            bc[i] = 1 - b**i

        sm = v[1:] / bc[1:]
        
        return sm
    cols = list(df.columns)
    cols.remove('Unit Number')
    cols.remove('Time (Cycles)')
    cols.remove('Expected RUL')
    unit_ids = set(df['Unit Number'].values)

    for c in cols:
        sm_list = []
        for n in unit_ids:
            s = np.array(df[df['Unit Number'] == n][c].copy())
            sm = list(smooth(s))
            sm_list += sm
        
        df[c+'_smoothed'] = sm_list

    for c in cols:
        if 'smoothed' not in c:
            df[c] = df[c+'_smoothed']
            df.drop(c+'_smoothed', axis = 1, inplace = True)
    
    return df 

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
    
    def fit_en(self):
        
        self.trainDatasetsCopy = copy.deepcopy(self.train_datasets)
        self.validationDatasetsCopy = copy.deepcopy(self.validation_datasets)
        self.testDatasetsCopy = copy.deepcopy(self.test_datasets)

        scaler = []

        for i in range(4):
            sc = StandardScaler()
            scaler.append(sc)
        
        for i in range(4):
            self.trainDatasetsCopy[i].loc[:, "OP1":"S21"] = scaler[i].fit_transform(self.trainDatasetsCopy[i].loc[:, "OP1":"S21"])
            self.validationDatasetsCopy[i].loc[:, "OP1":"S21"] = scaler[i].fit_transform(self.validationDatasetsCopy[i].loc[:, "OP1":"S21"])
            self.testDatasetsCopy[i].loc[:, "OP1":"S21"] = scaler[i].transform(self.testDatasetsCopy[i].loc[:, "OP1":"S21"])

        for i in range(4):

            # fit elastic_net
            elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000) # alpha(lamda) = 1, l1_ratio=0.5 is package default
            elastic_net.fit(self.trainDatasetsCopy[i].loc[:,"OP1":"S21"], self.trainDatasetsCopy[i].loc[:,"Expected RUL"])

            # rank feature by absolute value of coefficients
            coef = pd.DataFrame(elastic_net.coef_, index=elastic_net.feature_names_in_,columns=["coefficient"])
            sort_features_name = abs(coef).sort_values(ascending=False, by="coefficient").index.tolist()
            # print(abs(coef).sort_values(ascending=False, by="coefficient"))

            n_features = self.config.n_features
            top_10_features = []
            for j in range(n_features):
                top_10_features.append(sort_features_name[j])

            temp1 = self.trainDatasetsCopy[i].loc[:,top_10_features]
            temp2 = self.validationDatasetsCopy[i].loc[:,top_10_features]
            temp3 = self.testDatasetsCopy[i].loc[:,top_10_features]

            # Converting to Dataframes
            temp1 = pd.DataFrame(temp1, columns = top_10_features)
            temp2 = pd.DataFrame(temp2, columns = top_10_features)
            temp3 = pd.DataFrame(temp3, columns = top_10_features)

            # Dropping Excess Data
            self.trainDatasetsCopy[i].drop(inplace = True, columns = self.trainDatasetsCopy[i].columns[2:-1])
            self.validationDatasetsCopy[i].drop(inplace = True, columns = self.validationDatasetsCopy[i].columns[2:-1])
            self.testDatasetsCopy[i].drop(inplace = True, columns = self.testDatasetsCopy[i].columns[2:-1])

            # Merging New Data
            self.trainDatasetsCopy[i] = pd.merge(self.trainDatasetsCopy[i], temp1, left_index=True, right_index=True)
            self.validationDatasetsCopy[i] = pd.merge(self.validationDatasetsCopy[i], temp2, left_index=True, right_index=True)
            self.testDatasetsCopy[i] = pd.merge(self.testDatasetsCopy[i], temp3, left_index=True, right_index=True)

        return self.trainDatasetsCopy, self.validationDatasetsCopy, self.testDatasetsCopy

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

        newColumns = [f'PC{i}' for i in range(1, self.config.n_components+1)]

        for i in range(4):

            cols, start, end = self.get_features_name(self.train_datasets[i])

            # Finding Principal Components
            temp1 = pca.fit_transform(self.trainDatasetsCopy[i].loc[:, start:end])
            temp2 = pca.transform(self.validationDatasetsCopy[i].loc[:, start:end])
            temp3 = pca.transform(self.testDatasetsCopy[i].loc[:, start:end])

            # Converting to Dataframes
            temp1 = pd.DataFrame(temp1, columns = newColumns)
            temp2 = pd.DataFrame(temp2, columns = newColumns)
            temp3 = pd.DataFrame(temp3, columns = newColumns)

            # Dropping Excess Data
            self.trainDatasetsCopy[i].drop(inplace = True, columns = cols)
            self.validationDatasetsCopy[i].drop(inplace = True, columns = cols)
            self.testDatasetsCopy[i].drop(inplace = True, columns = cols)

            # Merging New Data
            self.trainDatasetsCopy[i] = pd.merge(self.trainDatasetsCopy[i], temp1, left_index=True, right_index=True)
            self.validationDatasetsCopy[i] = pd.merge(self.validationDatasetsCopy[i], temp2, left_index=True, right_index=True)
            self.testDatasetsCopy[i] = pd.merge(self.testDatasetsCopy[i], temp3, left_index=True, right_index=True)

        return self.trainDatasetsCopy, self.validationDatasetsCopy, self.testDatasetsCopy
    
    def get_features_name(self, df):
        cols = list(df.columns)
        cols.remove('Unit Number')
        cols.remove('Time (Cycles)')
        cols.remove('Expected RUL')
        start = cols[0]
        end = cols[-1]
        return cols, start, end
    
    def fit_transform_df(self, train_df, val_df, test_df):

        sc = StandardScaler() # MinMaxScaler 學不到任何資訊
        train_df.loc[:, "OP1":"S21"] = sc.fit_transform(train_df.loc[:, "OP1":"S21"])
        val_df.loc[:, "OP1":"S21"] = sc.fit_transform(val_df.loc[:, "OP1":"S21"])
        test_df.loc[:, "OP1":"S21"] = sc.fit_transform(test_df.loc[:, "OP1":"S21"])
        return train_df, val_df, test_df
    
    def fit_transform(self):

        self.trainDatasetsCopy = copy.deepcopy(self.train_datasets)
        self.validationDatasetsCopy = copy.deepcopy(self.validation_datasets)
        self.testDatasetsCopy = copy.deepcopy(self.test_datasets)

        scaler = []

        for i in range(4):
            sc = StandardScaler() # MinMaxScaler 學不到任何資訊
            scaler.append(sc)
        
        for i in range(4):
            self.trainDatasetsCopy[i].loc[:, "OP1":"S21"] = scaler[i].fit_transform(self.trainDatasetsCopy[i].loc[:, "OP1":"S21"])
            self.validationDatasetsCopy[i].loc[:, "OP1":"S21"] = scaler[i].fit_transform(self.validationDatasetsCopy[i].loc[:, "OP1":"S21"])
            self.testDatasetsCopy[i].loc[:, "OP1":"S21"] = scaler[i].transform(self.testDatasetsCopy[i].loc[:, "OP1":"S21"])

        return self.trainDatasetsCopy, self.validationDatasetsCopy, self.testDatasetsCopy

    def get(self, dataset_index: int):

        self.load_data()

        # Access the datasets based on the specified index
        if self.config.preprocessing_method == 'en':
            print("Do elastic net to select important features")   
            self.train_datasets, self.validation_datasets, self.test_datasets = self.fit_en()
            print("Done.")

        elif self.config.preprocessing_method == 'pca':
            print("Do PCA")
            self.train_datasets, self.validation_datasets, self.test_datasets = self.fit_pca()
            print("Done.")

        # else:
        #     self.train_datasets, self.validation_datasets, self.test_datasets = self.fit_transform()

        train_df = self.train_datasets[dataset_index]
        valid_df = self.validation_datasets[dataset_index]
        test_df = self.test_datasets[dataset_index]
        
        # train_df = self.smooth_dataframe(train_df)
        # valid_df = self.smooth_dataframe(valid_df)
        # test_df = self.smooth_dataframe(test_df)
        train_df, valid_df, test_df = self.fit_transform_df(train_df, valid_df, test_df)

        # select PC
        if self.config.preprocessing_method == 'pca':
            PC_to_drop = [f'PC{i}' for i in range(self.config.n_features+1,11)]
            train_df.drop(PC_to_drop, axis=1, inplace=True)
            valid_df.drop(PC_to_drop, axis=1, inplace=True)
            test_df.drop(PC_to_drop, axis=1, inplace=True)
            
        cols, start, end = self.get_features_name(train_df)
        print("Choose",cols,'as health index')
        
        # Get indices for training and validation datasets based on window_size
        window = self.window_size

        train_indices = list(train_df[(train_df['Expected RUL'] >= (window - 1)) & (train_df['Time (Cycles)'] > 10)].index)
        val_indices = list(valid_df[(valid_df['Expected RUL'] >= (window - 1)) & (valid_df['Time (Cycles)'] > 10)].index)
        
        # Create instances of custom datasets using the obtained indices
        train_dataset = CustomDataset(train_indices, train_df, window)
        val_dataset = CustomDataset(val_indices, valid_df, window)
        min_unit_id = min(self.units[dataset_index])
        max_unit_id = max(self.units[dataset_index])
        units = np.arange(min_unit_id, max_unit_id+1)
        test_dataset = TestDataset(units, test_df, window)

        return train_dataset, val_dataset, test_dataset

class CustomDataset(Dataset):
    """
    Custom dataset class for handling data.

    Args:
        list_indices (list): List of indices to use for the dataset.
        df_train (pandas.DataFrame): Training dataframe.
    """
    def __init__(self, list_indices, df_train, window_size):
        self.window_size = window_size
        self.indices = list_indices
        self.df_train = df_train
        
    def __len__(self):
        
        return len(self.indices)
    
    def __getitem__(self, idx):
        
        ind = self.indices[idx]
        
        X_ = self.df_train.iloc[ind : ind + self.window_size, :].copy()
        X_ = smooth_dataframe(X_)
        y_ = self.df_train.iloc[ind + self.window_size-1]['Expected RUL']
        X_ = X_.drop(['Unit Number','Time (Cycles)','Expected RUL'], axis = 1).to_numpy()
        

        return X_, y_
    
class TestDataset(Dataset):
    
    def __init__(self, units, df_test, window_size):
        self.window_size = window_size
        self.units = units
        self.df_test = df_test
        
    def __len__(self):
        
        return len(self.units)
    
    def __getitem__(self, idx):
        
        n = self.units[idx]
        U = self.df_test[self.df_test['Unit Number'] == n].copy()
        X_ = U.reset_index().iloc[-self.window_size:,:].drop(['index','Unit Number','Time (Cycles)','Expected RUL'], axis = 1).copy().to_numpy()
        X_ = smooth_dataframe(X_)
        y_ = U['Expected RUL'].min()
        
        
        return X_, y_

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