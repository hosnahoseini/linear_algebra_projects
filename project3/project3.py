import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

all_data = pd.read_csv('covid_cases.csv')

class RegressionModel:
    
    def __init__(self, all_data, order=1):
        self.all_data = all_data
        self.order = order
        self.train_data = all_data.sample(frac = 0.9)
        self.test_data = all_data.drop(self.train_data.index) 
        self.coefficient = None
        
    def train(self):
        x_train = self.train_data['World'].index
        y_train = self.train_data['World']   

        A = np.ones((len(self.train_data), self.order + 1))
        A[:, 1] = x_train.to_numpy()
        if self.order == 2:
            A[:, 2] = x_train ** 2

        y = y_train.to_numpy()
        self.coefficient = np.linalg.inv((np.transpose(A) @ A)) @ np.transpose(A) @ y
        
    def test(self):
        x_test = self.test_data['World'].index
        y_valid = self.test_data['World'].to_numpy() 

        A = np.ones((len(self.test_data), self.order + 1))
        A[:, 1] = x_test.to_numpy()
        if self.order == 2:
            A[:, 2] = x_test ** 2

        y = A @ self.coefficient 

        random_result = random.sample(range(0, len(self.test_data)), 5)
        for i in random_result:
            print(f'Real value: {y_valid[i]}')
            print(f'Estimated value: {y[i]}')
            print(f'Error: {abs(y[i] - y_valid[i])}') 
            print('-' * 20)
            
    def draw_graph(self):
        x = all_data.index.to_numpy()
        
        A = np.ones((len(all_data), self.order + 1))
        A[:, 1] = x
        if self.order == 2:
            A[:, 2] = x ** 2

        y_estimate = A @ self.coefficient 
        y_valid = all_data['World'].to_numpy()
        
        plt.plot(x, y_estimate)
        plt.plot(x, y_valid)        



linearRegressionModel = RegressionModel(all_data, order=1)
linearRegressionModel.train()
linearRegressionModel.test()
linearRegressionModel.draw_graph()

polynomialRegressionModel = RegressionModel(all_data, order=2)
polynomialRegressionModel.train()
polynomialRegressionModel.test()
polynomialRegressionModel.draw_graph()
