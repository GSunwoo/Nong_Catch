# 모듈 사용
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = "./font/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 모든 그래프를 무조건 기본 설정으로 시작하게 하려면 코드 맨 앞에 두면 편하다.
plt.style.use('default')

# 딸기
sd = pd.read_csv('./data/날씨-생산량-딸기(연도).csv')
# 마늘
gd = pd.read_csv('./data/날씨-생산량-마늘(연도).csv')
# 복숭아
ped = pd.read_csv('./data/날씨-생산량-복숭아(연도).csv')
# 양파
od = pd.read_csv('./data/날씨-생산량-양파(연도).csv')

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

# 생산량 그래프를 그려주는 부분
sd.plot(kind='bar', x='year', y='생산량', ax=axes[0], color='royalblue')
gd.plot(kind='bar', x='year', y='생산량', ax=axes[1], color='royalblue')
ped.plot(kind='bar', x='year', y='생산량', ax=axes[2], color='royalblue')
od.plot(kind='bar', x='year', y='생산량', ax=axes[3], color='royalblue')

sd.plot(kind='bar', x='year', y='생산량', ax=axes[0], color='royalblue')
gd.plot(kind='bar', x='year', y='생산량', ax=axes[1], color='royalblue')
ped.plot(kind='bar', x='year', y='생산량', ax=axes[2], color='royalblue')
od.plot(kind='bar', x='year', y='생산량', ax=axes[3], color='royalblue')


# 강수량 그래프 그리는 부분
sd.plot(kind='line', x='year', y='일강수량(mm)',
            ax=axes[0], color='royalblue', marker='o', markersize=2)

gd.plot(kind='line', x='year', y='일강수량(mm)',
            ax=axes[1], color='royalblue', marker='o', markersize=2)

ped.plot(kind='line', x='year', y='일강수량(mm)',
            ax=axes[2], color='royalblue', marker='o', markersize=2)

od.plot(kind='line', x='year', y='일강수량(mm)',
            ax=axes[3], color='royalblue', marker='o', markersize=2)

# 해당 그래프의 타이틀부분
axes[0].set_title('딸기 (년도별)')
axes[1].set_title('마늘 (년도별)')
axes[2].set_title('복숭아 (년도별)')
axes[3].set_title('양파 (년도별)')
for ax in [axes[0], axes[1], axes[2], axes[3]]:
    ax.set_xlabel('년도')
    ax.set_ylabel('생산량')

plt.tight_layout()
plt.show()
