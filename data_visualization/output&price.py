import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager, rc

# ✅ 폰트 설정 (한글 깨짐 방지)
font_path = "../resData/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# ✅ 작물 리스트와 데이터 폴더 설정
target_crops = ['양파', '딸기', '마늘', '복숭아']
data_folder = '../training/data'

# ✅ CSV 데이터 읽고 병합
df_list = []
for crop in target_crops:
    file_path = os.path.join(data_folder, f"{crop}_생산량_가격_데이터.csv")
    temp_df = pd.read_csv(file_path, encoding='utf-8')
    temp_df['날짜'] = pd.to_datetime(temp_df['날짜'], errors='coerce')
    temp_df['작물명'] = crop
    df_list.append(temp_df)

df_all = pd.concat(df_list, ignore_index=True)
df_all['연도'] = df_all['날짜'].dt.year
df_all['월'] = df_all['날짜'].dt.month

# ✅ 연도 반복: 2003~2023
for selected_year in range(2003, 2024):
    df_year = df_all[df_all['연도'] == selected_year]
    if df_year.empty:
        continue  # 해당 연도 데이터 없으면 건너뜀

    pivot_production = df_year.pivot_table(index='작물명', columns='월', values='생산량', aggfunc='mean')
    pivot_price = df_year.pivot_table(index='작물명', columns='월', values='가격', aggfunc='mean')

    months = list(range(1, 13))  # 1월 ~ 12월
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    colors = ['green', 'blue', 'red', 'orange']

    for i, crop in enumerate(target_crops):
        ax = axes[i]
        ax2 = ax.twinx()

        if crop in pivot_production.index and crop in pivot_price.index:
            # ✅ 월별 데이터 누락 보완: NaN으로 채움
            y_production = pivot_production.loc[crop].reindex(months)
            y_price = pivot_price.loc[crop].reindex(months)

            ax.plot(months, y_production, marker='o', color=colors[i], label='생산량', linewidth=2)
            ax.set_ylabel('생산량', color=colors[i])
            ax.tick_params(axis='y', labelcolor=colors[i])

            ax2.plot(months, y_price, marker='x', color='red', linestyle='--', label='가격', linewidth=2)
            ax2.set_ylabel('가격', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            ax.set_title(f"{selected_year}년 {crop} - 생산량과 가격 추이", fontsize=16)
            ax.set_xticks(months)
            ax.set_xticklabels(months, rotation=45)

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    output_filename = f"{selected_year}.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"{output_filename} 저장 완료")