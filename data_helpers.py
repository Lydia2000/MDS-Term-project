import os

class DataHolder:
   
    def __init__(self) -> None:
        
        self.data_path = 'data'+ os.sep

        # names of files
        self.train_files = [f'train_FD00{i}.txt' for i in range(1,5)]
        self.test_files = [f'test_FD00{i}.txt' for i in range(1,5)]
        self.RUL_files = [f'RUL_FD00{i}.txt' for i in range(1,5)] # Remaining useful life
        self.true_RUL_files = ['x.txt']
        
        # columns name for train, test data
        self.columns = ['Unit Number', 'Time (Cycles)', 
                        'OP1', 'OP2', 'OP3', 
                        'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21']
        
# def P1(X:pd.DataFrame, y:..., c:list = None, lambda_:list = [0.85, 0.5], M2:int = 100, M3:int = 100):

    def load_data():
        
        


class preprocessor:
    pass