from data_helpers import DataHolder
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class fit_pca:
   
    def __init__(self, train_datasets, test_datasets, expected_RUL_datasets) -> None:

        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.expected_RUL_datasets = expected_RUL_datasets
        
    
    def pc(self):
        
        self.trainDatasetsCopy = copy.deepcopy(self.train_datasets)
        self.testDatasetsCopy = copy.deepcopy(self.test_datasets)
        
        scaler = []

        for i in range(4):
            sc = StandardScaler()
            scaler.append(sc)

        for i in range(4):
            self.trainDatasetsCopy[i].iloc[:, 2:] = scaler[i].fit_transform(self.trainDatasetsCopy[i].iloc[:, 2:])
            self.testDatasetsCopy[i].iloc[:, 2: ] = scaler[i].transform(self.testDatasetsCopy[i].iloc[:, 2:])
        
        pca = PCA(n_components = 10)

        newColumns = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10']

        for i in range(4):
            # Finding Principal Components
            temp1 = pca.fit_transform(self.trainDatasetsCopy[i].iloc[:, 2:])
            temp2 = pca.transform(self.testDatasetsCopy[i].iloc[:, 2:])

            # Converting to Dataframes
            temp1 = pd.DataFrame(temp1, columns = newColumns)
            temp2 = pd.DataFrame(temp2, columns = newColumns)

            # Dropping Excess Data
            self.trainDatasetsCopy[i].drop(inplace = True, columns = self.trainDatasetsCopy[i].columns[2:])
            self.testDatasetsCopy[i].drop(inplace = True, columns = self.testDatasetsCopy[i].columns[2:])

            # Merging New Data
            self.trainDatasetsCopy[i] = pd.merge(self.trainDatasetsCopy[i], temp1, left_index=True, right_index=True)
            self.testDatasetsCopy[i] = pd.merge(self.testDatasetsCopy[i], temp2, left_index=True, right_index=True)

    def result_df(self):
        self.pc()
        return self. trainDatasetsCopy, self.testDatasetsCopy, self.expected_RUL_datasets