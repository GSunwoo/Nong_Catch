import requests
import json
import pandas as pd

url = 'https://api.odcloud.kr/api/15072357/v1/uddi:bb1ac3a7-84e3-4fc0-b3f2-bc22f1d7964b'

page = 1
data = []  # ✅ 전체 데이터 저장용 리스트

while True:
    params = dict(
        page=page,
        perPage=1000,
        serviceKey='K8fiPYPNvfZolnzG684+FWyGua6JLhEmM2mgfRqqMbHvgb3vzXEIl3mZqFzBJtsmptxYN++AblYOUtOPVewccA=='
    )
    response = requests.get(url=url, params=params)

    if response.status_code != 200:
        print(f"❌ 요청 실패 (status: {response.status_code})")
        break

    try:
        json_data = response.json()
    except json.JSONDecodeError:
        print("❌ JSON 디코딩 실패")
        break

    rows = json_data.get('data', [])
    if not rows:
        print("✅ 모든 페이지 로드 완료")
        break

    # 예시: 어떤 품목명이 있는지 먼저 확인해보기
    print(set([row.get("품목명") for row in rows]))
    quit()


    for jd in rows:
        PRCE_REG_YMD = jd.get('가격등록일자', '')
        MRKT_NM = jd.get('시장', '')
        CTNP_NM = jd.get('시도명', '')
        PDLT_NM = jd.get('품목명', '')
        BULK_GRAD_NM = jd.get('산물등급명', '')
        PDLT_PRCE = jd.get('품목가격', '')

        if MRKT_NM == '대인' and CTNP_NM == '광주' and PDLT_NM in ['양파', '깐마늘', '딸기', '복숭아']:
            data.append({
                '날짜': PRCE_REG_YMD,
                '시장명': MRKT_NM,
                '시도명': CTNP_NM,
                '품목명': PDLT_NM,
                '산물등급': BULK_GRAD_NM,
                '가격': PDLT_PRCE
            })

    print(f"📄 페이지 {page} 처리 완료 (누적 {len(data)}건)")
    page += 1

# ✅ Pandas로 변환 후 출력 및 저장
df = pd.DataFrame(data)
print("\n📊 최종 데이터:")
print(df.head())

# ✅ 엑셀 저장 (선택)
# df.to_excel("가격데이터 1차가공.xlsx", index=False)
df.to_csv("가격데이터 1차가공.csv", index=False)
print("✅ 엑셀 저장 완료")


