import pandas as pd
from collections import defaultdict

df = pd.read_csv('../resData/물류기기 품목 시도별 물량이동량-.csv', header=0)
#print(df.head())
df = df.fillna(method='ffill')
print(df.head())

# 'LOAD_ETPS_CTNP_NM' -> 컬럼명
mask = (df['LOAD_ETPS_CTNP_NM']=='전라남도') | (df['LOAD_ETPS_CTNP_NM']=='광주광역시')
df_jn = df[mask]

# 컬럼이름
agg = defaultdict(lambda: {'양파': 0, '마늘': 0, '딸기': 0, '복숭아': 0})

for idx, row in df.iterrows():
    # idx: 행 인덱스
    # row: 판다스 Series, 열 이름으로 값 접근 가능
    date = row['UNLDCG_YMD']  # 하차일자
    origin = row['LOAD_ETPS_CTNP_NM']  # 출발지
    qty = row['TOT_DLNG_VOLM']  # 물량
    crop = row['AGRI_NM']  # 품목

    if crop == '양파':
        agg[date]['양파'] += qty
    elif crop == '깐마늘': # 조건
        agg[date]['마늘'] += qty # 우리가 만든 컬럼
    elif crop == '딸기':
        agg[date]['딸기'] += qty
    elif crop == '복숭아':
        agg[date]['복숭아'] += qty

df = pd.DataFrame(agg)

# 데이터프레임을 csv파일로 저장
df.to_csv("./saveFiles/harvestdata2.csv")

