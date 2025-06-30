# havestdata 행/열 전환

import pandas as pd

# CSV 읽기
df_prod = pd.read_csv('../saveFiles/harvestdata2.csv', index_col=0)

# 행/열 전환
df_prod_t = df_prod.T

df_prod_t.index = pd.to_datetime(df_prod_t.index, format='%Y%m%d')

df_prod_t.to_csv('../saveFiles/harvestdata_t.csv')