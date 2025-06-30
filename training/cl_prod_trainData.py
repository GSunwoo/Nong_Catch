'''
기상 요소와 생산량의 관계를 학습하기 위한 csv 생성
품목별로 시간차 부여
- 양파 : 8개월
- 마늘 : 6개월
- 딸기 : 2개월
- 복숭아 : 2개월
'''

import pandas as pd

# CSV 읽기
df_prod = pd.read_csv('../saveFiles/harvestdata_t.csv') # 생산량
df_cl = pd.read_csv('../saveFiles/2003~2024년 전라남도 평균 기상요소.csv', index_col=0) # 날씨

# 각 데이터프레임의 날짜 컬럼을 datetime 타입으로 변경
df_cl.index = pd.to_datetime(df_cl.index, format='%Y-%m-%d')
df_prod['날짜'] = pd.to_datetime(df_prod['날짜'], format='%Y-%m-%d')
# print(df_prod)
# print(df_cl)

# 날씨 데이터프레임을 품목별 데이터프레임으로 복사
df_oni = df_cl.copy()
df_gal = df_cl.copy()
df_str = df_cl.copy()
df_pec = df_cl.copy()

# 품목별로 n개월 후 날짜 컬럼 지정
df_oni['날짜_8개월후'] = df_oni.index + pd.DateOffset(months=8)
df_gal['날짜_6개월후'] = df_gal.index + pd.DateOffset(months=6)
df_str['날짜_2개월후'] = df_str.index + pd.DateOffset(months=2)
df_pec['날짜_2개월후'] = df_pec.index + pd.DateOffset(months=2)

# 생산량 데이터프레임의 날짜 컬럼을 인덱스로 지정
after_day = df_prod.set_index('날짜')

# 품목별로 n개월 후의 생산량을 컬럼에 추가
df_oni['생산량'] = df_oni['날짜_8개월후'].map(after_day['양파'])
df_gal['생산량'] = df_gal['날짜_6개월후'].map(after_day['마늘'])
df_str['생산량'] = df_str['날짜_2개월후'].map(after_day['딸기'])
df_pec['생산량'] = df_pec['날짜_2개월후'].map(after_day['복숭아'])

# n개월 후 컬럼 삭제
df_oni = df_oni.drop(['날짜_8개월후'], axis=1)
df_gal = df_gal.drop(['날짜_6개월후'], axis=1)
df_str = df_str.drop(['날짜_2개월후'], axis=1)
df_pec = df_pec.drop(['날짜_2개월후'], axis=1)

# CSV 파일로 전환
df_oni.to_csv('./data/날씨-생산량-양파.csv')
df_gal.to_csv('./data/날씨-생산량-마늘.csv')
df_str.to_csv('./data/날씨-생산량-딸기.csv')
df_pec.to_csv('./data/날씨-생산량-복숭아.csv')

print(df_oni.iloc[2000:2010])
print(df_gal.iloc[2000:2010])
print(df_str.iloc[2000:2010])
print(df_pec.iloc[2000:2010])
