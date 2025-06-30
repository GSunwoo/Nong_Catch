import pandas as pd

# 1) CSV 파일 불러오기
df_must_trans = pd.read_csv('../resData/harvestdata2.csv', index_col=0)

# 2) 행과 열 바꾸기 (Transpose)
df = df_must_trans.T

# 기존 DataFrame이 df라고 가정
df = df.reset_index()
df = df.rename(columns={'index': '날짜'})
df['날짜'] = pd.to_datetime(df['날짜'], format='%Y%m%d')

# 3) 품목별로 0이 아닌 데이터만 필터링
df_garlic = df[df['마늘'] != 0][['날짜', '마늘']].copy()
df_straw = df[df['딸기'] != 0][['날짜', '딸기']].copy()
df_peach = df[df['복숭아'] != 0][['날짜', '복숭아']].copy()
df_onion = df[df['양파'] != 0][['날짜', '양파']].copy()

# 4) 필요 시 날짜 인덱스로 설정
df_garlic.set_index('날짜', inplace=True)
df_straw.set_index('날짜', inplace=True)
df_peach.set_index('날짜', inplace=True)
df_onion.set_index('날짜', inplace=True)

df_straw = df_straw.rename(columns={'딸기': '생산량'})
df_peach = df_peach.rename(columns={'복숭아' : '생산량'})
df_onion = df_onion.rename(columns={'양파' : '생산량'})
df_garlic = df_garlic.rename(columns={'마늘' : '생산량'})

# 5) 확인
print("딸기 데이터:", df_straw.head())
print("복숭아 데이터:", df_peach.head())  # 복숭아는 비어있을 수도 있음
print("양파 데이터:", df_onion.head())
print("마늘 데이터:", df_garlic.head())

df_garlic.to_csv('../saveFiles/prod/마늘_생산데이터.csv', encoding='utf-8-sig')
df_peach.to_csv('../saveFiles/prod/복숭아_생산데이터.csv', encoding='utf-8-sig')
df_onion.to_csv('../saveFiles/prod/양파_생산데이터.csv', encoding='utf-8-sig')
df_straw.to_csv('../saveFiles/prod/딸기_생산데이터.csv', encoding='utf-8-sig')