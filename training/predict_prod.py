import pandas as pd
import numpy as np
import joblib
from scipy import stats
from cl_prod_test import EnhancedCropYieldPreprocessor

# ===== 1. 예측용 데이터 로드 =====
new_data = pd.read_csv('./data/weath-prod/future_weath.csv')  # '일시', '평균기온(°C)', ...
new_data['일시'] = pd.to_datetime(new_data['일시'])  # 일시 형 변환

# ===== 2. 전처리 객체 생성 =====
processor = EnhancedCropYieldPreprocessor()
crop_types = ['딸기','마늘','복숭아','양파']

# ===== 3. 사용할 날씨 컬럼 설정 =====
weather_cols = ['평균기온(°C)', '일강수량(mm)', '평균 상대습도(%)', '합계 일조시간(hr)']
available_weather_cols = [col for col in weather_cols if col in new_data.columns]

for crop_name in crop_types:
    # ===== 4. 동일한 전처리 흐름 적용 =====
    new_data = processor.create_enhanced_time_features(new_data)
    new_data = processor.create_seasonal_weather_features(new_data, available_weather_cols)
    new_data = processor.create_advanced_time_series_features(new_data, available_weather_cols)
    new_data = processor.create_crop_specific_features(new_data, crop_name)
    new_data = processor.create_weather_stress_indicators(new_data, available_weather_cols)
    new_data = processor.create_advanced_interactions(new_data, available_weather_cols)

    # ===== 6. 모델과 스케일러 불러오기 =====
    rf_model = joblib.load(f'./trainedModel/{crop_name}_rf_model.pkl')
    gb_model = joblib.load(f'./trainedModel/{crop_name}_gb_model.pkl')
    scaler = joblib.load(f'./trainedModel/{crop_name}_scaler.pkl')
    feature_cols = joblib.load(f'./trainedModel/{crop_name}_feature_cols.pkl')

    X_new = new_data[feature_cols]
    X_new = X_new.fillna(0)

    # ===== 7. 스케일링 및 예측 =====
    X_new_scaled = scaler.transform(X_new)
    ensemble_pred = 0.6 * rf_model.predict(X_new_scaled) + 0.4 * gb_model.predict(X_new_scaled)

    # ===== 8. 결과 출력 =====
    print("✅ 예측된 수확량 목록:")
    print(ensemble_pred)

    # (선택) 평균값 출력
    print(f"\n📊 예측 평균 수확량: {np.mean(ensemble_pred):.2f}")

    rounded_preds = np.round(ensemble_pred, 2)

    date_array = new_data['일시'].values

    df_pred = pd.DataFrame({
        '날짜': date_array,       # 날짜 정보 컬럼
        '생산량': rounded_preds
    })

    df_pred.to_csv(f'./data/pred/{crop_name}_생산량_예측결과.csv', index=False, encoding='utf-8')
print("✅ 예측값이 '예측결과.csv'로 저장되었습니다.")
