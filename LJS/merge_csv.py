import pandas as pd
import glob
import os

# CSV 파일들이 있는 폴더 경로
folder_path = './resData/'

# 폴더 내의 모든 CSV 파일 경로 가져오기
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# 빈 데이터프레임 생성
df_list = []

# 각 CSV 파일 읽어서 리스트에 추가
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

# 데이터프레임 하나로 합치기
merged_df = pd.concat(df_list, ignore_index=True)

# 필요하면 시간순 정렬
merged_df = merged_df.sort_values(by='PRCE_REG_YMD')

# CSV 파일로 저장
merged_df.to_csv('./merged_data.csv', index=False)

print('데이터 합치기 완료!')