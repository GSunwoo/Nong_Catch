import requests, json
import pandas as pd
from collections import defaultdict

# 공공데이터포털에서 제공하는 OpenAPI 사용
url = 'https://api.odcloud.kr/api/15089196/v1/uddi:5d3baac6-67ed-4d26-9973-c407c6f5a621'
agg = defaultdict(lambda: {'양파': 0, '마늘': 0, '딸기': 0, '복숭아': 0})

p = 1
while True:
    params = dict(
        page=p,
        # 타입을 json타입으로
        Type='json',
        # 사이즈를 300으로 만들겠다는 말
        perPage='1000',
        serviceKey='API_KEY')
    # JSON 데이터 읽어오기
    raw_data = requests.get(url=url, params=params)
    binary_data = raw_data.content
    json_data = json.loads(binary_data)
    print(json_data)
    if json_data['currentCount']==0:
        print("종료되었습니다.")
        break

    # 제공된 데이터의 갯수만큼 반복
    for jd in json_data['data']:
        date  = jd['하차일자'] #하차일자
        origin = jd['출발지'] #출발지
        qty = jd['물량'] #물량
        crop  = jd['품목'] #품목
        if not (origin == '광주광역시' or origin == '전라남도'):
            continue
        if crop == '양파':
            agg[date]['양파'] += qty
        elif crop == '마늘':
            agg[date]['마늘'] += qty
        elif crop == '딸기':
            agg[date]['딸기'] += qty
        elif crop == '복숭아':
            agg[date]['복숭아'] += qty
    print(p)
    p += 1



df = pd.DataFrame(agg)

# 데이터프레임을 csv파일로 저장
df.to_csv("./saveFiles/harvestdata.csv")
