import pandas as pd
import os
import glob

# 경로 설정
price_dir = '../savefiles/price'
yield_dir = '../savefiles/prod'
save_dir = '../training/data'
os.makedirs(save_dir, exist_ok=True)

# 가격 파일 목록 가져오기
price_files = glob.glob(os.path.join(price_dir, '*.csv'))

for price_file in price_files:
    # 1. 품목명 추출 (파일명에서 확장자 제외하고, '_' 기준 첫 부분)
    base = os.path.splitext(os.path.basename(price_file))[0]  # 예: '딸기_가격데이터'
    item_name = base.split('_')[0]                           # 예: '딸기'

    # 2. 대응하는 생산량 파일 경로
    yield_file = os.path.join(yield_dir, f'{item_name}_생산데이터.csv')

    # 3. 생산량 파일이 없으면 건너뜀
    if not os.path.exists(yield_file):
        print(f'⚠️ {item_name} 생산량 파일이 존재하지 않음, 건너뜀')
        continue

    # 4. 데이터 읽기
    price_df = pd.read_csv(price_file)
    yield_df = pd.read_csv(yield_file)

    # 5. 컬럼명 공백 제거 (안정성)
    price_df.columns = price_df.columns.str.strip()
    yield_df.columns = yield_df.columns.str.strip()

    # 6. 컬럼명 '품목명' → '가격' / '생산량'으로 변경
    price_df = price_df.rename(columns={item_name: '가격'})
    yield_df = yield_df.rename(columns={item_name: '생산량'})

    # 7. 날짜 형식 통일 (필요하면)
    price_df['날짜'] = pd.to_datetime(price_df['날짜'])
    yield_df['날짜'] = pd.to_datetime(yield_df['날짜'])

    # 8. 날짜 기준 병합 (inner join)
    merged_df = pd.merge(yield_df, price_df, on='날짜', how='inner')

    # 9. 결측치 제거
    merged_df = merged_df.dropna(subset=['가격', '생산량'])

    # 10. 가격 또는 생산량이 0인 행 제거
    merged_df = merged_df[(merged_df['가격'] != 0) & (merged_df['생산량'] != 0)]

    # 11. 정렬 및 컬럼 순서 맞춤
    result_df = merged_df[['날짜', '생산량', '가격']].sort_values(by='날짜')

    # 12. 데이터가 없으면 건너뜀
    if result_df.empty:
        print(f'⚠️ {item_name} 유효한 데이터 없음 (0 또는 NaN만 존재)')
        continue

    # 13. CSV 저장 (utf-8-sig로 한글 깨짐 방지)
    filename = f'{item_name}_생산량_가격_데이터.csv'
    result_df.to_csv(os.path.join(save_dir, filename), index=False, encoding='utf-8-sig')

    print(f'✅ 저장 완료: {filename}')