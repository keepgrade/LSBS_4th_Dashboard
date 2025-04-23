import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy as sp
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris
from sklearn import linear_model
from tqdm import tqdm
warnings.filterwarnings('ignore')



class DataLoader:
    def __init__(self):
        self.data = pd.read_csv('../data/ames.csv')
        

    def load_data(self):
        self.data['Date'] = pd.to_datetime(
            self.data['YrSold'].astype(str) + '-' + self.data['MoSold'].astype(str),
            format='%Y-%m'
        )
        return self.data
    