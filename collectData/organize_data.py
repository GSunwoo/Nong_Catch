import pandas as pd

# 1) CSV 파일 불러오기
df = pd.read_csv('../resData/가격데이터_1차가공.csv')

# 2) 날짜 변환 (yyyymmdd 형식 가정)
df['날짜'] = df['날짜'].astype(int)
df['날짜'] = df['날짜'].astype(str)
df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce').dt.date

# 🧪 확인: 날짜 변환 성공 여부
print("❓ 변환 실패 날짜 수:", df['날짜'].isna().sum())

# 3) 가격이 문자열이면 숫자형으로 변환
df['가격'] = df['가격'].astype(str).str.replace(',', '').astype(float)

# 🧪 확인: 가격 숫자 변환 여부
print("🎯 가격 필드 예시:", df['가격'].head())

# 4) 피벗 테이블 생성
pivot_df = df.pivot_table(index='날짜', columns='품목명', values='가격', aggfunc='mean')

# 5) 결측값은 0으로 채움
pivot_df = pivot_df.fillna(0)

# 결과 확인
print("\n📊 피벗 데이터 (상위 5행):")

