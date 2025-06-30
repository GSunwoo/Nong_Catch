'''
가격이나 생산량이 0일땐 패스
각 날짜별 품목 데이터 나누기
'''

import pandas as pd

# 1) CSV 파일 불러오기
df = pd.read_csv('../resData/가격_피벗_데이터.csv')

df['깐마늘(국산)'] = df['깐마늘(국산)'].astype(int)
df['딸기'] = df['딸기'].astype(int)
df['복숭아'] = df['복숭아'].astype(int)
df['양파'] = df['양파'].astype(int)


straw, peach, onion, garlic = [], [], [], []
straw.append(df['날짜' '딸기'],)
peach.append(df['날짜' '복숭아'])
onion.append(df['날짜' '양파'])
garlic.append(df['날짜' '깐마늘(국산)'])

df_straw = pd.DataFrame(straw)
df_peach = pd.DataFrame(peach)
df_onion = pd.DataFrame(onion)
df_garlic =
