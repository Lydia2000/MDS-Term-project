import os
import pandas as pd
from sklearn.model_selection import train_test_split

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
        # self.validation_datasets = []
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
            # train_df, validation_df = train_test_split(train_df, test_size=0.2, random_state=42)

            # Reset the index for both train_set and validation_set
            train_df = train_df.reset_index(drop=True)
            # validation_df = validation_df.reset_index(drop=True)

            # Appending dataframe to dataset list
            self.train_datasets.append(train_df)
            self.test_datasets.append(test_df)
            # self.validation_datasets.append(validation_df)
            self.expected_RUL_datasets.append(RUL_df)


    def get(self):
        self.load_data()
        return self.train_datasets, self.test_datasets, self.expected_RUL_datasets

