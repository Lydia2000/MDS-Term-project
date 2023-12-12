import os
import pandas as pd
import random
import math
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet


class DataHolder:
   
    def __init__(self) -> None:
        
        self.data_path = 'data'+ os.sep

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

            # Split train and validation dataset
            n = train_df.loc[len(train_df)-1,"Unit Number"]
            random_numbers = random.sample(range(n+1), math.floor(n*0.2))

            validation_df = train_df[train_df["Unit Number"].isin(random_numbers)]
            train_df = train_df[~train_df["Unit Number"].isin(random_numbers)]
            # print(validation_df)
            # Reset the index for both train_set and validation_set
            train_df = train_df.reset_index(drop=True)
            validation_df = validation_df.reset_index(drop=True)

            # Appending dataframe to dataset list
            self.train_datasets.append(train_df)
            self.test_datasets.append(test_df)
            self.validation_datasets.append(validation_df)
            self.expected_RUL_datasets.append(RUL_df)

    def get(self):
        self.load_data()
        return self.train_datasets, self.validation_datasets, self.test_datasets, self.expected_RUL_datasets
    
# data = DataHolder()
# train_datasets, validation_datasets, test_datasets, expected_RUL_datasets = data.get()

class fit_pca:
   
    def __init__(self) -> None:
        data = DataHolder()
        train_datasets, validation_datasets, test_datasets, expected_RUL_datasets = data.get()
        self.train_datasets = train_datasets
        self.validation_datasets = validation_datasets
        self.test_datasets = test_datasets
        self.expected_RUL_datasets = expected_RUL_datasets
        
    
    def pc(self):
        
        self.trainDatasetsCopy = copy.deepcopy(self.train_datasets)
        self.validationDatasetsCopy = copy.deepcopy(self.validation_datasets)
        self.testDatasetsCopy = copy.deepcopy(self.test_datasets)
        # print(self.trainDatasetsCopy)
        # print(self.validationDatasetsCopy)

        
        scaler = []

        for i in range(4):
            sc = StandardScaler()
            scaler.append(sc)

        for i in range(4):
            self.trainDatasetsCopy[i].iloc[:, 2:] = scaler[i].fit_transform(self.trainDatasetsCopy[i].iloc[:, 2:])
            self.validationDatasetsCopy[i].iloc[:, 2:] = scaler[i].transform(self.validationDatasetsCopy[i].iloc[:, 2:])
            self.testDatasetsCopy[i].iloc[:, 2: ] = scaler[i].transform(self.testDatasetsCopy[i].iloc[:, 2:])
        
        pca = PCA(n_components = 10)

        newColumns = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10']

        for i in range(4):
            # Finding Principal Components
            temp1 = pca.fit_transform(self.trainDatasetsCopy[i].iloc[:, 2:])
            temp2 = pca.transform(self.validationDatasetsCopy[i].iloc[:, 2:])
            temp3 = pca.transform(self.testDatasetsCopy[i].iloc[:, 2:])

            # Converting to Dataframes
            temp1 = pd.DataFrame(temp1, columns = newColumns)
            temp2 = pd.DataFrame(temp2, columns = newColumns)
            temp3 = pd.DataFrame(temp3, columns = newColumns)

            # Dropping Excess Data
            self.trainDatasetsCopy[i].drop(inplace = True, columns = self.trainDatasetsCopy[i].columns[2:])
            self.validationDatasetsCopy[i].drop(inplace = True, columns = self.validationDatasetsCopy[i].columns[2:])
            self.testDatasetsCopy[i].drop(inplace = True, columns = self.testDatasetsCopy[i].columns[2:])

            # Merging New Data
            self.trainDatasetsCopy[i] = pd.merge(self.trainDatasetsCopy[i], temp1, left_index=True, right_index=True)
            self.validationDatasetsCopy[i] = pd.merge(self.validationDatasetsCopy[i], temp2, left_index=True, right_index=True)
            self.testDatasetsCopy[i] = pd.merge(self.testDatasetsCopy[i], temp3, left_index=True, right_index=True)

    def result_df(self):
        self.pc()
        return self. trainDatasetsCopy, self.validationDatasetsCopy, self.testDatasetsCopy, self.expected_RUL_datasets
    

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
        return self. trainDatasetsCopy, self.validationDatasetsCopy, self.testDatasetsCopy, self.expected_RUL_datasets

