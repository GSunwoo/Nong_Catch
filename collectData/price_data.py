import pandas as pd
import glob
import os

# CSV 파일들이 있는 폴더 경로
folder_path = 'C:/Users/kosmo/Downloads/새 폴더 (2)'

# 해당 폴더의 모든 CSV 파일 경로를 가져오기
file_list = glob.glob(os.path.join(folder_path, '*.csv'))

df_list = []

# 파일 하나씩 읽어서 리스트에 저장 (인코딩 문제 대비)
for file in file_list:
    try:
        df = pd.read_csv(file, encoding='cp949', low_memory=False)  # 윈도우 한글 인코딩 시도
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding='utf-8')  # 실패 시 utf-8로 재시도
    df_list.append(df)

# 데이터프레임 하나로 합치기
merged_df = pd.concat(df_list, ignore_index=True)

data = []

# 각 행을 딕셔너리 형태로 순회
for jd in merged_df.to_dict('records'):
    PRCE_REG_YMD = jd.get('PRCE_REG_YMD', '')
    CTNP_NM = jd.get('CTNP_NM', '')
    PDLT_NM = jd.get('PDLT_NM', '')
    EXMN_SE_NM = jd.get('EXMN_SE_NM', '')
    BULK_GRAD_NM = jd.get('BULK_GRAD_NM', '')
    PDLT_PRCE = jd.get('PDLT_PRCE', '')

    if EXMN_SE_NM == '소매' and CTNP_NM == '광주' and BULK_GRAD_NM == '상품' and PDLT_NM in ['양파', '깐마늘(국산)', '딸기', '복숭아']:
        try:
            price_int = int(float(str(PDLT_PRCE).replace(',', '').strip()))
        except:
            price_int = 0  # 가격 변환 실패 시 0으로 처리

        data.append({
            '날짜': PRCE_REG_YMD,
            '시도명': CTNP_NM,
            '품목명': PDLT_NM,
            '조사구분명': EXMN_SE_NM,
            '산물등급': BULK_GRAD_NM,
            '가격': price_int
        })

# 리스트를 DataFrame으로 변환
df_data = pd.DataFrame(data)

# 그룹바이 후 평균 가격 계산 및 정수 변환
result = df_data.groupby(['날짜', '품목명'])['가격'].mean().reset_index()
result['가격'] = result['가격'].round().astype(int)

# 결과 출력
print("\n📊 최종 데이터:")
print(result.head())

# 결과를 CSV 파일로 저장
save_path = os.path.join(os.path.dirname(folder_path), '가격데이터_1차가공.csv')
result.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"✅ CSV 저장 완료: {save_path}")
