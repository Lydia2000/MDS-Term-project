from data_helpers import DataHolder
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

class fit_ElasticNet:
   
    def __init__(self, train_datasets, test_datasets, expected_RUL_datasets) -> None:

        self.train_datasets = train_datasets
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
        self.testDatasetsCopy = copy.deepcopy(self.test_datasets)

        scaler = []

        for i in range(4):
            sc = StandardScaler()
            scaler.append(sc)
        
        for i in range(4):
            self.trainDatasetsCopy[i].iloc[:, 2:] = scaler[i].fit_transform(self.trainDatasetsCopy[i].iloc[:, 2:])
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
            temp2 = self.testDatasetsCopy[i].loc[:,top_10_features]

            # Converting to Dataframes
            temp1 = pd.DataFrame(temp1, columns = top_10_features)
            temp2 = pd.DataFrame(temp2, columns = top_10_features)

            # Dropping Excess Data
            self.trainDatasetsCopy[i].drop(inplace = True, columns = self.trainDatasetsCopy[i].columns[2:])
            self.testDatasetsCopy[i].drop(inplace = True, columns = self.testDatasetsCopy[i].columns[2:])

            # Merging New Data
            self.trainDatasetsCopy[i] = pd.merge(self.trainDatasetsCopy[i], temp1, left_index=True, right_index=True)
            self.testDatasetsCopy[i] = pd.merge(self.testDatasetsCopy[i], temp2, left_index=True, right_index=True)

    def result_df(self):
        self.en()
        return self. trainDatasetsCopy, self.testDatasetsCopy, self.expected_RUL_datasets