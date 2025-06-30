import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager,rc

font_path = "../resData/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

target_crops = ['양파', '딸기', '마늘', '복숭아']
data_folder = '../training/data'

# 품목별 데이터 읽어 합치기 (각 데이터에 '작물명' 컬럼 추가)
df_list = []
for crop in target_crops:
    file_path = os.path.join(data_folder, f"{crop}_생산량_가격_데이터.csv")  # ex) ../resData/품목별데이터/양파.csv
    temp_df = pd.read_csv(file_path, encoding='utf-8')
    temp_df['날짜'] = pd.to_datetime(temp_df['날짜'], errors='coerce')
    temp_df['작물명'] = crop
    df_list.append(temp_df)

df_all = pd.concat(df_list, ignore_index=True)

# 연도 컬럼 추가
df_all['연도'] = df_all['날짜'].dt.year
df_all['월'] = df_all['날짜'].dt.month

# 사용할 연도 범위 설정
selected_year = 2023
df_year = df_all[df_all['연도'] == selected_year]

# 피벗테이블 생성 (작물명 x 연도별 생산량, 가격)
pivot_production = df_all.pivot_table(index='작물명', columns='월', values='생산량', aggfunc='mean')
pivot_price = df_all.pivot_table(index='작물명', columns='월', values='가격', aggfunc='mean')

# x축 월 리스트
months = list(range(1, 13))

# 그래프 그리기
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = axes.flatten()
colors = ['green', 'blue', 'red', 'orange']

for i, crop in enumerate(target_crops):
    ax = axes[i]
    ax2 = ax.twinx()  # 오른쪽 y축 생성

    # 왼쪽 y축: 생산량
    ax.plot(months, pivot_production.loc[crop], marker='o', color=colors[i], label='생산량', linewidth=2)
    ax.set_ylabel('생산량', color=colors[i])
    ax.tick_params(axis='y', labelcolor=colors[i])

    # 오른쪽 y축: 가격
    ax2.plot(months, pivot_price.loc[crop], marker='x', color='red', linestyle='--', label='가격', linewidth=2)
    ax2.set_ylabel('가격', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_title(f"{crop} - 생산량과 가격 추이", fontsize=16)
    ax.set_xticks(months)
    ax.set_xticklabels(months , rotation=45)

    # 범례 따로 붙이기
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.show()
