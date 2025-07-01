# 모듈 사용
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글깨짐처리
font_path = "./data/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 모든 그래프를 무조건 기본 설정으로 시작하게 하려면 코드 맨 앞에 두면 편하다.
plt.style.use('ggplot')
print(plt.rcParams['font.family'])

# 딸기
sd = pd.read_csv('./data/날씨-생산량-딸기(연도).csv')
# 마늘
gd = pd.read_csv('./data/날씨-생산량-마늘(연도).csv')
# 복숭아
ped = pd.read_csv('./data/날씨-생산량-복숭아(연도).csv')
# 양파
od = pd.read_csv('./data/날씨-생산량-양파(연도).csv')

# 년도 int형으로 바꿔줌
year = sd['year'].astype(int)
# 2행 2열짜리 그래프 4개를 한꺼번에 만든다.
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.flatten()

# 딸기
df_sd = pd.read_csv('../saveFiles/2003~2024년 전라남도 평균 기상요소.csv')
# 마늘
df_gd = pd.read_csv('../saveFiles/2003~2024년 전라남도 평균 기상요소.csv')
# 복숭아
df_ped = pd.read_csv('../saveFiles/2003~2024년 전라남도 평균 기상요소.csv')
# 양파
df_od = pd.read_csv('../saveFiles/2003~2024년 전라남도 평균 기상요소.csv')

print(year)
# 딸기
axes[0].set_xticks(year)
axes[0].set_xticklabels([str(int(y)) for y in year], rotation=45)
# 생산량 막대그래프
axes[0].set_xlabel('년도')
axes[0].set_ylabel('평균 생산량 (kg)', color='blue')
axes[0].bar(year, sd['생산량'], color='blue', label='평균 생산량')
axes[0].tick_params(axis='y', labelcolor='blue')
# 강수량 꺽은선그래프
ax1 = axes[0].twinx()
ax1.set_ylabel('연간 강수량 (mm)', color='green')
ax1.plot(year, sd['일강수량(mm)'], color='green', marker='o',label='연간 강수량')
ax1.tick_params(axis='y', labelcolor='green')

# 복숭아
axes[1].set_xticks(year)
axes[1].set_xticklabels([str(int(y)) for y in year], rotation=45)
# 생산량 막대그래프
axes[1].set_xlabel('년도')
axes[1].set_ylabel('평균 생산량 (kg)', color='blue')
axes[1].bar(year, ped['생산량'], color='blue', label='평균 생산량')
axes[1].tick_params(axis='y', labelcolor='blue')
# 강수량 꺽은선그래프
ax2 = axes[1].twinx()
ax2.set_ylabel('연간 강수량 (mm)', color='green')
ax2.plot(year, ped['일강수량(mm)'], color='green', marker='o',label='연간 강수량')
ax2.tick_params(axis='y', labelcolor='green')

# 마늘
axes[2].set_xticks(year)
axes[2].set_xticklabels([str(int(y)) for y in year], rotation=45)
# 생산량 막대그래프
axes[2].set_xlabel('년도')
axes[2].set_ylabel('평균 생산량 (kg)', color='blue')
axes[2].bar(year, gd['생산량'], color='blue', label='평균 생산량')
axes[2].tick_params(axis='y', labelcolor='blue')
# 강수량 꺽은선그래프
ax3 = axes[2].twinx()
ax3.set_ylabel('연간 강수량 (mm)', color='green')
ax3.plot(year, gd['일강수량(mm)'], color='green', marker='o',label='연간 강수량')
ax3.tick_params(axis='y', labelcolor='green')

# 양파
axes[3].set_xticks(year)
axes[3].set_xticklabels([str(int(y)) for y in year], rotation=45)
# 생산량 막대그래프
axes[3].set_xlabel('년도')
axes[3].set_ylabel('평균 생산량 (kg)', color='blue')
axes[3].bar(year, od['생산량'], color='blue', label='평균 생산량')
axes[3].tick_params(axis='y', labelcolor='blue')
# 강수량 꺽은선그래프
ax4 = axes[3].twinx()
ax4.set_ylabel('연간 강수량 (mm)', color='green')
ax4.plot(year, od['일강수량(mm)'], color='green', marker='o',label='연간 강수량')
ax4.tick_params(axis='y', labelcolor='green')


# 해당 그래프의 타이틀부분
axes[0].set_title('딸기')
axes[1].set_title('마늘')
axes[2].set_title('복숭아')
axes[3].set_title('양파')


plt.tight_layout()
plt.show()
