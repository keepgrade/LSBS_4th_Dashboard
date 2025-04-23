import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy.stats as st
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import json

warnings.filterwarnings('ignore')

# ---------------------------
# 📂 데이터 로딩
# ---------------------------
# DataLoader 사용
from dataloader import DataLoader
dataloader = DataLoader()
dataset = dataloader.load_data()

# ---------------------------
# 🚒 소방서 위치 데이터
# ---------------------------
fire_stations = pd.DataFrame({
    'Name': ['Station 1', 'Station 2', 'Station 3'],
    'Address': [
        '1300 Burnett Ave, Ames, IA 50010',
        '132 Welch Ave, Ames, IA 50014',
        '2400 S Duff Ave, Ames, IA 50010'
    ],
    'Latitude': [42.034862, 42.021596, 42.001115],
    'Longitude': [-93.615031, -93.649759, -93.609166]
})

# ---------------------------
# 💰 가격 등급 분류 (5단계)
# ---------------------------
price_by_neigh = dataset.groupby('Neighborhood')['SalePrice'].mean()
q20 = price_by_neigh.quantile(0.20)
q40 = price_by_neigh.quantile(0.40)
q60 = price_by_neigh.quantile(0.60)
q80 = price_by_neigh.quantile(0.80)

def classify_price_grade(neighborhood):
    price = price_by_neigh[neighborhood]
    if price <= q20:
        return 'Very Low'
    elif price <= q40:
        return 'Low'
    elif price <= q60:
        return 'Normal'
    elif price <= q80:
        return 'High'
    else:
        return 'Very High'

dataset['PriceGrade'] = dataset['Neighborhood'].apply(classify_price_grade)

# ---------------------------
# 🎨 색상 매핑
# ---------------------------
color_map = {
    'Very Low': 'indigo',
    'Low': 'purple',
    'Normal': 'gray',
    'High': 'blue',
    'Very High': 'navy'
}

# ---------------------------
# 🧭 지도 설정
# ---------------------------
layout_mapbox = dict(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=42.034534, lon=-93.620369),
        zoom=11
    ),
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    title='Ames 시 주택 가격대 & 소방서 위치'
)

# ---------------------------
# 🏡 주택 마커 (등급별 분리)
# ---------------------------
home_traces = []
for grade, color in color_map.items():
    subset = dataset[dataset['PriceGrade'] == grade]
    trace = go.Scattermapbox(
        lat=subset['Latitude'],
        lon=subset['Longitude'],
        mode='markers',
        marker=dict(size=7, color=color, opacity=0.6),
        text=subset['Neighborhood'] + '<br>$' + subset['SalePrice'].astype(int).astype(str),
        name=f"{grade} Area"
    )
    home_traces.append(trace)

# ---------------------------
# 🚒 소방서 마커
# ---------------------------
fire_trace = go.Scattermapbox(
    lat=fire_stations['Latitude'],
    lon=fire_stations['Longitude'],
    mode='markers+text',
    marker=dict(size=12, color='red'),
    text=fire_stations['Name'],
    name='소방서',
    textposition='top right'
)

# ---------------------------
# 🧱 GeoJSON 경계선
# ---------------------------
with open('../data/ames_boundary.geojson') as f:
    geojson = json.load(f)

# 지도 객체 생성
fig = go.Figure(data=home_traces + [fire_trace], layout=layout_mapbox)

# GeoJSON 레이어 추가
fig.update_layout(
    mapbox_layers=[
        {
            "source": {
                "type": "FeatureCollection",
                "features": [geojson] if geojson["type"] != "FeatureCollection" else geojson["features"]
            },
            "type": "line",
            "color": "black",
            "line": {"width": 2}
        }
    ]
)

# 지도 출력
fig.show()



import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy.stats as st
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import json

warnings.filterwarnings('ignore')

# ---------------------------
# 📂 데이터 로딩
# ---------------------------
from dataloader import DataLoader
dataloader = DataLoader()
dataset = dataloader.load_data()

# ---------------------------
# 📏 평단가 컬럼 생성
# ---------------------------
dataset['PricePerArea'] = dataset['SalePrice'] / dataset['GrLivArea']

# ---------------------------
# 🚒 소방서 위치 데이터
# ---------------------------
fire_stations = pd.DataFrame({
    'Name': ['Station 1', 'Station 2', 'Station 3'],
    'Address': [
        '1300 Burnett Ave, Ames, IA 50010',
        '132 Welch Ave, Ames, IA 50014',
        '2400 S Duff Ave, Ames, IA 50010'
    ],
    'Latitude': [42.034862, 42.021596, 42.001115],
    'Longitude': [-93.615031, -93.649759, -93.609166]
})

# ---------------------------
# 💰 지역별 '평단가' 기반 등급 분류 (5단계)
# ---------------------------
price_per_area_by_neigh = dataset.groupby('Neighborhood')['PricePerArea'].mean()
q20 = price_per_area_by_neigh.quantile(0.20)
q40 = price_per_area_by_neigh.quantile(0.40)
q60 = price_per_area_by_neigh.quantile(0.60)
q80 = price_per_area_by_neigh.quantile(0.80)

def classify_price_grade(neigh):
    price = price_per_area_by_neigh[neigh]
    if price <= q20:
        return 'Very Low'
    elif price <= q40:
        return 'Low'
    elif price <= q60:
        return 'Normal'
    elif price <= q80:
        return 'High'
    else:
        return 'Very High'

dataset['PriceGrade'] = dataset['Neighborhood'].apply(classify_price_grade)

# ---------------------------
# 🎨 색상 매핑
# ---------------------------
color_map = {
    'Very Low': 'indigo',
    'Low': 'purple',
    'Normal': 'gray',
    'High': 'blue',
    'Very High': 'navy'
}

# ---------------------------
# 🧭 지도 설정
# ---------------------------
layout_mapbox = dict(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=42.034534, lon=-93.620369),
        zoom=11
    ),
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    title='Ames 시 평단가 기준 주택 등급 & 소방서 위치'
)

# ---------------------------
# 🏡 주택 마커 (등급별 분리)
# ---------------------------
home_traces = []
for grade, color in color_map.items():
    subset = dataset[dataset['PriceGrade'] == grade]
    trace = go.Scattermapbox(
        lat=subset['Latitude'],
        lon=subset['Longitude'],
        mode='markers',
        marker=dict(size=7, color=color, opacity=0.6),
        text=subset['Neighborhood'] + '<br>총가:$' + subset['SalePrice'].astype(int).astype(str) +
             '<br>1평당:$' + subset['PricePerArea'].round(1).astype(str),
        name=f"{grade} Area"
    )
    home_traces.append(trace)

# ---------------------------
# 🚒 소방서 마커
# ---------------------------
fire_trace = go.Scattermapbox(
    lat=fire_stations['Latitude'],
    lon=fire_stations['Longitude'],
    mode='markers+text',
    marker=dict(size=12, color='red'),
    text=fire_stations['Name'],
    name='소방서',
    textposition='top right'
)

# ---------------------------
# 🧱 GeoJSON 경계선
# ---------------------------
with open('../data/ames_boundary.geojson') as f:
    geojson = json.load(f)

# 지도 객체 생성
fig = go.Figure(data=home_traces + [fire_trace], layout=layout_mapbox)

# GeoJSON 레이어 추가
fig.update_layout(
    mapbox_layers=[
        {
            "source": {
                "type": "FeatureCollection",
                "features": [geojson] if geojson["type"] != "FeatureCollection" else geojson["features"]
            },
            "type": "line",
            "color": "black",
            "line": {"width": 2}
        }
    ]
)

# 지도 출력
fig.show()
