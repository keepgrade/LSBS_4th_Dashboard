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



def risk_material(material):
    materials = ["CompShg", "Tar&Grv", "WdShake", "WdShngl", "Metal", "Roll", "Membran"]
    risk_point = [2,2,5,4,1,3,3]
    mat_risk = {mat: risk for mat, risk in zip(materials, risk_point)}

    if material in mat_risk.keys():
        return mat_risk[material]
    else:
        return 0
    
class DataLoader:
    def __init__(self):
        self.data = pd.read_csv('../data/ames.csv')
        

    def load_data(self):
        self.data['Date'] = pd.to_datetime(
            self.data['YrSold'].astype(str) + '-' + self.data['MoSold'].astype(str),
            format='%Y-%m'
        )
        
        self.data['Risk_RoofMatl'] = self.data['RoofMatl'].apply(risk_material) 

        
        return self.data
    