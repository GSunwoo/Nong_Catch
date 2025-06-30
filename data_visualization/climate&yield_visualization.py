# 모듈 사용
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
'''
사용할 그래프 -> 산점도(Scatter plot)
: 연속되는 값을 갖는 서로 다른 두 변수 사이의 관계를 나타낸다.
  x,y축에 변수를 두고 데이터가 위치한 좌표를 찾아 점으로 표시한다.
'''

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

# 해당 그래프를 그려주는 부분
sd.plot(kind='bar', x='year', y='생산량', ax=axes[0], color='royalblue')
gd.plot(kind='bar', x='year', y='생산량', ax=axes[1], color='royalblue')
ped.plot(kind='bar', x='year', y='생산량', ax=axes[2], color='royalblue')
od.plot(kind='bar', x='year', y='생산량', ax=axes[3], color='royalblue')

# 그래프 그리는 부분
# df_sd .plot(kind='bar', x='일강수량(mm)', y='생산량',
#             ax=axe1, color='royalblue', marker='o', markersize=2)
#
# df_gd .plot(kind='bar', x='일강수량(mm)', y='생산량',
#             ax=axe2, color='royalblue', marker='o', markersize=2)
#
# df_ped.plot(kind='bar', x='일강수량(mm)', y='생산량',
#             ax=axe3, color='royalblue', marker='o', markersize=2)
#
# df_od .plot(kind='bar', x='일강수량(mm)', y='생산량',
#             ax=axe4, color='royalblue', marker='o', markersize=2)

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