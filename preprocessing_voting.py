"""
Use Voting: Ada Boost, Random Forest Regression, Stepwise Regression.
For Feature Selection
"""
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
import numpy as np
import pandas as pd
import copy

class FitFeatureSelection:

    def __init__(self, train_datasets, test_datasets, expected_RUL_datasets) -> None:
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.expected_RUL_datasets = expected_RUL_datasets

    def feature_selection(self):
        train_datasets_copy = copy.deepcopy(self.train_datasets)
        test_datasets_copy = copy.deepcopy(self.test_datasets)

        regressors = []

        for i in range(4):
            regressor = VotingRegressor(estimators=[
                ('ada_boost', AdaBoostRegressor(n_estimators=50)),
                ('random_forest', RandomForestRegressor(n_estimators=50)),
                ('stepwise_regression', self.stepwise_regression())
            ])
            regressors.append(regressor)

        for i in range(4):
            train_data = train_datasets_copy[i].iloc[:, 2:]
            test_data = test_datasets_copy[i].iloc[:, 2:]

            # Fit the regressors
            regressors[i].fit(train_data, self.expected_RUL_datasets[i])

            # Transform the datasets
            train_datasets_copy[i].iloc[:, 2:] = regressors[i].predict(train_data)
            test_datasets_copy[i].iloc[:, 2:] = regressors[i].predict(test_data)

    def stepwise_regression(self):
        # Implement your stepwise regression here
        # You can use a library like statsmodels or sklearn
        # For simplicity, let's assume we are using sklearn's RFE with LinearRegression
        return RFE(LinearRegression(), n_features_to_select=10)

    def result_df(self):
        self.feature_selection()
        return self.train_datasets_copy, self.test_datasets_copy, self.expected_RUL_datasets


