'''
품목코드(p_product_cls_code) 양파(245), 마늘(248), 딸기(226), 복숭아(413)
지역코드(p_country_code) 광주(2401)
날짜코드(p_regday) yyyy-mm-dd
당일데이터가 비어있으면 어제 데이터를 넣도록
카테고리코드(p_item_category_code) 채소(200) 과일(400) 복숭아만 400에 있음
조사단위(p_convert_kg_yn) 정보조사 단위표시(N)
도소매구분(p_product_cls_code) 소매(01)
반환타입(p_returntype) json

요청자(p_cert_id) dudns5552
p_cert_key= 'f42c857e-d5bc-47e7-a59e-5d2de8725e9a'

복숭아와 딸기는 계절에만 나오므로 가격이 있을때만 불러오는 식으로
처리해야될듯
'''

url = 'http://www.kamis.or.kr/service/price/xml.do?action=dailyPriceByCategoryList'
key = 'f42c857e-d5bc-47e7-a59e-5d2de8725e9a'