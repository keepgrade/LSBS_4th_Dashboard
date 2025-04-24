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
from dataloader import DataLoader
warnings.filterwarnings('ignore')

os.chdir('../src')
dataloader = DataLoader()

dataset = dataloader.load_data()
dataset.columns
dataset['1stFlrSF']
dataset['PricePerArea'] = dataset['SalePrice'] / dataset['LotArea']
# ---------------------------
# 💰 지역별 '평단가' 기반 등급 분류 (5단계)
# ---------------------------
price_per_area_by_neigh = dataset['PricePerArea']
q20 = price_per_area_by_neigh.quantile(0.20)
q40 = price_per_area_by_neigh.quantile(0.40)
q60 = price_per_area_by_neigh.quantile(0.60)
q80 = price_per_area_by_neigh.quantile(0.80)

def classify_price_grade(price):
    if price <= q20:
        return 1
    elif price <= q40:
        return 2
    elif price <= q60:
        return 3
    elif price <= q80:
        return 4
    else:
        return 5

# dataset['PriceGrade'] = dataset['PricePerArea'].apply(classify_price_grade)

#  위험도 평균 열 생성
dataset['Risk_Avg'] = (
    dataset['Risk_RoofMatl'] * 0.30 +
    dataset['Risk_Exterior1st'] * 0.30 +
    dataset['Risk_Exterior2nd'] * 0.10 +
    dataset['Risk_MasVnrType'] * 0.10 +
    dataset['Risk_WoodDeckSF'] * 0.2
)

# 위험도 평균을 5단계로 그룹화
dataset['Risk_Level'] = dataset['Risk_Avg'].round()
dataset['Risk_Level'].value_counts().sort_index()
dataset.groupby('Risk_Level')['PricePerArea'].mean()
# 결측값 제거
dataset = dataset.dropna(subset=['PricePerArea'])

import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('PricePerArea ~ C(Risk_Level)',data=dataset).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

# 해당 그림이 0을 기준으로 잘 분포되어있어야함 (잔차의 정규성)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.scatter(model.fittedvalues, model.resid)

# 애는 만족안함
import scipy.stats as sp
W, p = sp.shapiro(model.resid)
print(f'검정통계량: {W:.3f}, 유의확률: {p:.3f}')

# 애는 아님
from scipy.stats import probplot
plt.figure(figsize=(6, 6))
probplot(model.resid, dist="norm", plot=plt)


# 등분산성 검정 (만족)
from scipy.stats import bartlett
from scipy.stats import kruskal
groups = [1, 2, 3, 4, 5]
grouped_residuals = [model.resid[dataset['Risk_Level'] == group] for group in groups]
test_statistic, p_value = bartlett(*grouped_residuals)
print(f"검정통계량: {test_statistic}, p-value: {p_value}")

# 변수명을 dataset으로 바꾸고 Kruskal-Wallis 검정 다시 실행

# 그룹 나누기
grouped = [group['PricePerArea'].values for name, group in dataset.groupby('Risk_Level')]

# Kruskal-Wallis 검정
kruskal_stat, kruskal_p = kruskal(*grouped)

# 결과 반환
kruskal_result = {
    "검정통계량 (H)": kruskal_stat,
    "p-value": kruskal_p,
    "결론": "✔️ 그룹 간 차이가 유의함 (p < 0.05)" if kruskal_p < 0.05 else "❌ 유의한 차이 없음 (p ≥ 0.05)"
}

kruskal_result


# 비모수 사후검정
import scikit_posthocs as sp
posthoc = sp.posthoc_dunn(dataset, val_col='PricePerArea', group_col='Risk_Level', p_adjust='bonferroni')
posthoc



import pandas as pd
import plotly.graph_objects as go

# 색상 설정
color_map = {
    1: 'white', 2: 'gray', 3: 'yellow', 4: 'orange', 5: 'red'
}

# 소방서 위치
fire_stations = pd.DataFrame({
    'Name': ['Station 1', 'Station 2', 'Station 3'],
    'Latitude': [42.034862, 42.021596, 42.001115],
    'Longitude': [-93.615031, -93.649759, -93.609166]
})

# 지도 레이아웃
layout_mapbox = dict(
    mapbox=dict(style="open-street-map", center=dict(lat=42.0345, lon=-93.62), zoom=11),
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    title='Ames 시 위험도 기반 주택 시각화 & 소방서 위치'
)

# 주택 마커
traces = []
for level, color in color_map.items():
    df = dataset[dataset['Risk_Level'] == level]
    traces.append(go.Scattermapbox(
        lat=df['Latitude'], lon=df['Longitude'],
        mode='markers',
        marker=dict(size=7, color=color, opacity=0.6),
        text='가격: $' + df['SalePrice'].astype(str) + '<br>위험도: ' + df['Risk_Level'].astype(str),
        name=f'위험도 {level}'
    ))

# 소방서 마커
fire_trace = go.Scattermapbox(
    lat=fire_stations['Latitude'],
    lon=fire_stations['Longitude'],
    mode='markers+text',
    marker=dict(size=12, color='black'),
    text=fire_stations['Name'],
    name='소방서',
    textposition='top right'
)

# 시각화
fig = go.Figure(data=traces + [fire_trace], layout=layout_mapbox)
fig.show()

correlation = dataset[['PricePerArea', 'Risk_Avg']].corr()