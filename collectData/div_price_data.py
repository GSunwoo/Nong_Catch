import pandas as pd

# 1) CSV 파일 불러오기
df = pd.read_csv('../resData/가격_피벗_데이터.csv')

# 2) 데이터 형 변환
cols = ['깐마늘(국산)', '딸기', '복숭아', '양파']
for col in cols:
    df[col] = df[col].astype(int)

# 3) 품목별로 0이 아닌 데이터만 필터링
df_garlic = df[df['깐마늘(국산)'] != 0][['날짜', '깐마늘(국산)']].copy()
df_straw = df[df['딸기'] != 0][['날짜', '딸기']].copy()
df_peach = df[df['복숭아'] != 0][['날짜', '복숭아']].copy()
df_onion = df[df['양파'] != 0][['날짜', '양파']].copy()

# 4) 필요 시 날짜 인덱스로 설정
df_garlic.set_index('날짜', inplace=True)
df_straw.set_index('날짜', inplace=True)
df_peach.set_index('날짜', inplace=True)
df_onion.set_index('날짜', inplace=True)

# 5) 확인
print("딸기 데이터:", df_straw.head())
print("복숭아 데이터:", df_peach.head())  # 복숭아는 비어있을 수도 있음
print("양파 데이터:", df_onion.head())
print("깐마늘 데이터:", df_garlic.head())

df_garlic.to_csv('../saveFiles/price/마늘_가격데이터.csv', encoding='utf-8-sig')
df_peach.to_csv('../saveFiles/price/복숭아_가격데이터.csv', encoding='utf-8-sig')
df_onion.to_csv('../saveFiles/price/양파_가격데이터.csv', encoding='utf-8-sig')
df_straw.to_csv('../saveFiles/price/딸기_가격데이터.csv', encoding='utf-8-sig')