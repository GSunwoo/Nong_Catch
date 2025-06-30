import pandas as pd
import os

# 1. CSV 읽기
df_prod = pd.read_csv('../saveFiles/harvestdata_t.csv')  # 생산량
df_cl = pd.read_csv('../saveFiles/2003~2024년 전라남도 평균 기상요소.csv', index_col=0)  # 날씨

# 2. 날짜 처리
df_cl.index = pd.to_datetime(df_cl.index, format='%Y-%m-%d')
df_prod['날짜'] = pd.to_datetime(df_prod['날짜'], format='%Y-%m-%d')

# 3. 기상 데이터: 연도별 평균
df_cl['year'] = df_cl.index.year
df_cl_yearly = df_cl.groupby('year').mean(numeric_only=True)

# 4. 생산량 데이터: 연도별 총합
df_prod['year'] = df_prod['날짜'].dt.year
df_prod_yearly = df_prod.groupby('year').sum(numeric_only=True)

# 5. 품목별로 기상 + 생산량 결합 후 저장
os.makedirs('./data', exist_ok=True)

crops = {
    '양파': '날씨-생산량-양파(연도).csv',
    '마늘': '날씨-생산량-마늘(연도).csv',
    '딸기': '날씨-생산량-딸기(연도).csv',
    '복숭아': '날씨-생산량-복숭아(연도).csv'
}

for crop, filename in crops.items():
    df_combined = df_cl_yearly.copy()
    df_combined['생산량'] = df_prod_yearly[crop]
    df_combined.dropna(inplace=True)  # 둘 중 하나라도 없는 연도는 제거
    df_combined.to_csv(f'./data/{filename}', encoding='utf-8-sig')
    print(f'{filename} 저장 완료')