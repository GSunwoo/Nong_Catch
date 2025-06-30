'''
날씨-생산량 학습시킬 데이터 numpy로 저장
'''

import numpy as np
from  sklearn.model_selection import train_test_split
import pandas as pd

# csv 불러오기
df_str = pd.read_csv('./data/날씨-생산량-딸기.csv')
df_gal = pd.read_csv('./data/날씨-생산량-마늘.csv')
df_pec = pd.read_csv('./data/날씨-생산량-복숭아.csv')
df_oni = pd.read_csv('./data/날씨-생산량-양파.csv')

# 생산량이 0이거나 null인 행 삭제
df_str = df_str[~((df_str['생산량'] == 0) | (df_str['생산량'].isna())|(df_str['생산량']<100))]
df_gal = df_gal[~((df_str['생산량'] == 0) | (df_gal['생산량'].isna())|(df_gal['생산량']<100))]
df_pec = df_pec[~((df_str['생산량'] == 0) | (df_pec['생산량'].isna())|(df_pec['생산량']<100))]
df_oni = df_oni[~((df_str['생산량'] == 0) | (df_oni['생산량'].isna())|(df_oni['생산량']<100))]

# data와 label로 분리
str_data = df_str[['평균기온(°C)','일강수량(mm)','평균 상대습도(%)','합계 일조시간(hr)']]
str_label = df_str['생산량']
gal_data = df_gal[['평균기온(°C)','일강수량(mm)','평균 상대습도(%)','합계 일조시간(hr)']]
gal_label = df_gal['생산량']
pec_data = df_pec[['평균기온(°C)','일강수량(mm)','평균 상대습도(%)','합계 일조시간(hr)']]
pec_label = df_pec['생산량']
oni_data = df_oni[['평균기온(°C)','일강수량(mm)','평균 상대습도(%)','합계 일조시간(hr)']]
oni_label = df_oni['생산량']

# 학습데이터와 테스트데이터로 분리
train_data_str, test_data_str, train_label_str, test_label_str = train_test_split(str_data, str_label, test_size=0.1, random_state=42)
train_data_gal, test_data_gal, train_label_gal, test_label_gal = train_test_split(gal_data, gal_label, test_size=0.1, random_state=42)
train_data_pec, test_data_pec, train_label_pec, test_label_pec = train_test_split(pec_data, pec_label, test_size=0.1, random_state=42)
train_data_oni, test_data_oni, train_label_oni, test_label_oni = train_test_split(oni_data, oni_label, test_size=0.1, random_state=42)

# npz파일로 저장
np.savez('./trainData/str_train_data_clprod.npz', X_train=train_data_str, X_test=test_data_str, Y_train=train_label_str, Y_test=test_label_str)
np.savez('./trainData/gal_train_data_clprod.npz', X_train=train_data_gal, X_test=test_data_gal, Y_train=train_label_gal, Y_test=test_label_gal)
np.savez('./trainData/pec_train_data_clprod.npz', X_train=train_data_pec, X_test=test_data_pec, Y_train=train_label_pec, Y_test=test_label_pec)
np.savez('./trainData/oni_train_data_clprod.npz', X_train=train_data_oni, X_test=test_data_oni, Y_train=train_label_oni, Y_test=test_label_oni)
print('Task Finished..!!')