# HTML파일을 템플릿으로 사용하기 위한 모듈 임포트
from flask import Flask, render_template, request
# 화면이동, 세션처리 등을 위한 모듈 임포트
from flask import redirect, session, url_for
# 문자열 깨짐 방지를 위한 인코딩 처리를 위한 모듈 임포트
from markupsafe import escape
import pandas as pd
from _datetime import datetime
import requests

# 플라스크 앱 초기화
app = Flask(__name__)

#-----------------------------------------------api 추가------------------------------------
ITEMS = {
    "양파": {"category": "200", "code": "245"},
    "마늘": {"category": "200", "code": "248"},
    "딸기": {"category": "200", "code": "226"},
    "복숭아": {"category": "400", "code": "413"}
}

# API 키
API_KEY = "f42c857e-d5bc-47e7-a59e-5d2de8725e9a"
API_ID = "dudns5552"
BASE_URL = "http://www.kamis.or.kr/service/price/xml.do?action=dailyPriceByCategoryList"
#-----------------------------------------------api 추가 끝-------------------------------------







# 앱을 최초로 실행했을때의 화면. 주로 index화면이라고 한다.
@app.route('/')
def root():
    # 📌 1. 카드 데이터 (연간 예측)
    df_cards = pd.read_csv('static/predict_year/품목별_연간예측_데이터.csv')
    cards = []
    for _, row in df_cards.iterrows():
        cards.append({
            'name': row['품목명'],
            'production': f"{int(row['총생산량']):,}kg",
            'price': f"{int(row['연평균가격']):,}원/kg"
        })

    #############################################################################
    #생산량
    # 📌 2. 생산량 그래프 데이터 (일별 → 연평균)
    df_harvest = pd.read_csv('static/data/havestdata_t.csv')
    df_harvest['연도'] = pd.to_datetime(df_harvest['날짜']).dt.year
    grouped = df_harvest.groupby('연도')[['양파', '마늘', '딸기', '복숭아']].mean().round(1)

    years = grouped.index.tolist()
    onion = grouped['양파'].tolist()
    garlic = grouped['마늘'].tolist()
    strawberry = grouped['딸기'].tolist()
    peach = grouped['복숭아'].tolist()

    ######################################################################
    # 가격
    # 📌 가격 데이터 로딩
    df_price = pd.read_csv('static/data/가격_피벗_데이터.csv')
    df_price['연도'] = pd.to_datetime(df_price['날짜']).dt.year

    # ✅ '깐마늘(국산)'을 '마늘'로 통일
    df_price.rename(columns={'깐마늘(국산)': '마늘'}, inplace=True)

    # 📌 연도별 평균 계산
    grouped_price = df_price.groupby('연도')[['마늘', '딸기', '복숭아', '양파']].mean().round(1)

    # 📌 리스트로 변환
    price_years = grouped_price.index.tolist()
    price_onion = grouped_price['양파'].tolist()
    price_garlic = grouped_price['마늘'].tolist()
    price_strawberry = grouped_price['딸기'].tolist()
    price_peach = grouped_price['복숭아'].tolist()

    # ✅ 품목별: 연도별 생산량 합계 + 가격 평균
    item_files = {
        '양파': 'static/data/양파_생산량_가격_데이터.csv',
        '마늘': 'static/data/마늘_생산량_가격_데이터.csv',
        '딸기': 'static/data/딸기_생산량_가격_데이터.csv',
        '복숭아': 'static/data/복숭아_생산량_가격_데이터.csv'
    }

    production_price_data = {}
    for item, filepath in item_files.items():
        df = pd.read_csv(filepath)
        df['연도'] = pd.to_datetime(df['날짜']).dt.year
        grouped = df.groupby('연도').agg({'생산량': 'sum', '가격': 'mean'}).round(1)
        production_price_data[item] = {
            'years': grouped.index.tolist(),
            'production': grouped['생산량'].tolist(),
            'price': grouped['가격'].tolist()
        }

    ###############################################################################
    #기후데이터
    # 📌 기후 데이터 (기본 연도: 2023)
    df_weather = pd.read_csv('static/data/2003~2024년 전라남도 평균 기상요소.csv')
    df_weather['일시'] = pd.to_datetime(df_weather['일시'])
    df_weather['연도'] = df_weather['일시'].dt.year
    df_weather['월'] = df_weather['일시'].dt.month

    weather_data_by_year = {}
    for year in range(2003, 2025):
        df_year = df_weather[df_weather['연도'] == year]
        monthly_avg = df_year.groupby('월').mean(numeric_only=True).round(2)
        weather_data_by_year[year] = {
            'temperature': monthly_avg['평균기온(°C)'].tolist(),
            'rainfall': monthly_avg['일강수량(mm)'].tolist(),
            'humidity': monthly_avg['평균 상대습도(%)'].tolist(),
            'sunshine': monthly_avg['합계 일조시간(hr)'].tolist()
        }

    # 📌 템플릿으로 모든 데이터 전달
    return render_template('main_dashboard.html',
                           cards=cards,
                           years=years,
                           onion=onion,
                           garlic=garlic,
                           strawberry=strawberry,
                           peach=peach,
                           price_years=price_years,
                           price_onion=price_onion,
                           price_garlic=price_garlic,
                           price_strawberry=price_strawberry,
                           price_peach=price_peach,
                           production_price_data=production_price_data,
                           weather_data_by_year=weather_data_by_year,
                           default_weather_year=2023)

@app.route('/visual')
def show_visual():
    return render_template('visual.html')

@app.route('/dashboard')
def dashboard():
    return render_template('Nong-catch.html')
# 기상&생산량 대시보드로 갈 수 있는 경로
@app.route('/dashboard/cy')
def cy_dashboard():
    return render_template('cli&yie_dashboard.html')

# Page not found 에러 발생시 핸들링
@app.errorhandler(404)
def page_not_found(error):
    print("오류 로그:", error)  # 서버콘솔에 출력
    return render_template('404.html'), 404

#------------------------------------------------가격 동향 페이지 추가---------------------------------------

@app.route("/price")
def index():
    results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for name, info in ITEMS.items():
        params = {
            "p_cert_key": API_KEY,
            "p_cert_id": API_ID,
            "p_returntype": "json",
            "p_product_cls_code": "01",
            "p_item_category_code": info["category"],
            "p_country_code": "2401",
            "p_regday": today,
            "p_convert_kg_yn": "N"
        }

        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()

            # 필터링: 해당 itemcode + 등급='상품'
            for item in data["price"]:
                if item["itemcode"] == info["code"] and item["rank"] == "상품":
                    price_today = float(item["dpr1"] or 0)
                    price_yesterday = float(item["dpr2"] or 0)
                    price_week = float(item["dpr3"] or 0)
                    price_normal = float(item["dpr7"] or 0)

                    # 값 보정
                    actual_price = price_today if price_today else price_yesterday

                    # 퍼센트 계산
                    diff_normal = round(((actual_price - price_normal) / price_normal) * 100, 1) if price_normal else 0
                    diff_week = round(((actual_price - price_week) / price_week) * 100, 1) if price_week else 0

                    results[name] = {
                        "prices": {
                            "평년": price_normal,
                            "1주일전": price_week,
                            "오늘": actual_price
                        },
                        "percent": {
                            "평년": diff_normal,
                            "1주일전": diff_week
                        }
                    }
                    break
            else:
                results[name] = None  # 수확철 아님
        except Exception as e:
            results[name] = None  # 오류 발생 시 예외 처리

    return render_template("index.html", results=results)
#------------------------------------------------가격 동향 페이지 끝-------------------------------------

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
