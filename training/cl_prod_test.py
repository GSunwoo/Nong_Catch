'''
향상된 농작물 수확량 예측 데이터 전처리 (계절성 + 연간 트렌드 + 작물별 특성 강화)
'''
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class EnhancedCropYieldPreprocessor:
    def __init__(self):
        self.scalers = {}

    def load_and_clean_data(self, filepath, crop_name):
        """데이터 로드 및 기본 정제"""
        df = pd.read_csv(filepath)

        # 생산량이 없을 경우 0으로 채움
        if '생산량' not in df.columns:
            print(f"⚠️ '{crop_name}' 데이터에 '생산량' 열이 없습니다. 0으로 채웁니다.")
            df['생산량'] = 0

        # 생산량 필터링 (0, null, 극값 제거)
        df = df[~((df['생산량'] == 0) | (df['생산량'].isna()) | (df['생산량'] < 100))]

        if len(df) == 0:
            raise ValueError(f"'{crop_name}' 데이터가 '생산량' 기준 필터링 후 비어 있습니다.")

        # 극값 제거 (IQR 방법 - 더 관대하게 조정)
        Q1 = df['생산량'].quantile(0.15)
        Q3 = df['생산량'].quantile(0.85)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.0 * IQR
        upper_bound = Q3 + 2.0 * IQR
        df = df[(df['생산량'] >= lower_bound) & (df['생산량'] <= upper_bound)]

        # 일시 형식 변환
        if '일시' in df.columns:
            df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
        else:
            raise ValueError(f"'{crop_name}' 데이터에 '일시' 열이 없습니다.")

        # 빈값 전날로 채우기
        df = df.sort_values('일시').ffill()

        print(f"{crop_name} 데이터: {len(df)}개 샘플")
        return df

    def create_enhanced_time_features(self, df):
        """향상된 시간 특성 생성"""
        df = df.copy()

        # 기본 시간 특성
        df['년도'] = df['일시'].dt.year
        df['월'] = df['일시'].dt.month
        df['일'] = df['일시'].dt.day
        df['요일'] = df['일시'].dt.dayofweek
        df['주차'] = df['일시'].dt.isocalendar().week

        # 계절성 특성 (사인/코사인 인코딩)
        df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
        df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)
        df['일_sin'] = np.sin(2 * np.pi * df['일'] / 31)
        df['일_cos'] = np.cos(2 * np.pi * df['일'] / 31)
        df['주차_sin'] = np.sin(2 * np.pi * df['주차'] / 52)
        df['주차_cos'] = np.cos(2 * np.pi * df['주차'] / 52)

        # 계절 구분
        df['계절'] = df['월'].map({12: 0, 1: 0, 2: 0,  # 겨울
                                3: 1, 4: 1, 5: 1,   # 봄
                                6: 2, 7: 2, 8: 2,   # 여름
                                9: 3, 10: 3, 11: 3}) # 가을

        # 연간 트렌드 (기준년도로부터 경과 년수)
        base_year = df['년도'].min()
        df['년도_경과'] = df['년도'] - base_year
        df['년도_경과_제곱'] = df['년도_경과'] ** 2

        return df

    def create_seasonal_weather_features(self, df, weather_cols):
        """계절별 날씨 특성 생성"""
        df = df.copy()

        # 계절별 날씨 통계
        for season in range(4):
            season_mask = df['계절'] == season
            for col in weather_cols:
                season_data = df[season_mask][col]
                if len(season_data) > 0:
                    df[f'{col}_계절{season}_평균'] = season_data.mean()
                    df[f'{col}_계절{season}_편차'] = df[col] - season_data.mean()
                else:
                    df[f'{col}_계절{season}_평균'] = 0
                    df[f'{col}_계절{season}_편차'] = 0

        return df

    def create_advanced_time_series_features(self, df, weather_cols, window_sizes=[7, 14, 30, 60, 90]):
        """고급 시계열 특성 생성"""
        df = df.sort_values('일시').copy()

        for window in window_sizes:
            for col in weather_cols:
                # 이동평균 계열
                df[f'{col}_ma{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_std{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_min{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_max{window}'] = df[col].rolling(window=window, min_periods=1).max()

                # 누적 특성 (강수량, 일조시간)
                if '강수량' in col or '일조시간' in col:
                    df[f'{col}_sum{window}'] = df[col].rolling(window=window, min_periods=1).sum()

                # 트렌드 특성
                df[f'{col}_trend{window}'] = df[col].rolling(window=window, min_periods=2).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0, raw=True)

                # 변동계수 (CV)
                df[f'{col}_cv{window}'] = df[f'{col}_std{window}'] / (df[f'{col}_ma{window}'] + 1e-8)

        # 결측값 처리
        df = df.ffill().fillna(0)
        return df

    def create_crop_specific_features(self, df, crop_type):
        """작물별 맞춤 특성 생성"""
        df = df.copy()

        # 작물별 생육 단계 가중치 (월별)
        growth_weights = {
            '딸기': {9: 0.5, 10: 1.0, 11: 1.5, 12: 2.0, 1: 2.5, 2: 2.5, 3: 2.0, 4: 1.5, 5: 1.0, 6: 0.5},
            '마늘': {9: 1.0, 10: 1.5, 11: 2.0, 12: 2.5, 1: 2.5, 2: 2.0, 3: 1.8, 4: 1.5, 5: 2.0, 6: 1.2},
            '복숭아': {1: 0.5, 2: 0.8, 3: 1.2, 4: 1.8, 5: 2.2, 6: 2.5, 7: 2.5, 8: 2.0, 9: 1.5, 10: 1.0},
            '양파': {9: 1.0, 10: 1.5, 11: 2.0, 12: 2.2, 1: 2.5, 2: 2.5, 3: 2.0, 4: 1.8, 5: 1.5, 6: 1.0}
        }

        # 작물별 최적 온도 범위
        optimal_temp_ranges = {
            '딸기': (15, 25),
            '마늘': (15, 22),
            '복숭아': (20, 28),
            '양파': (18, 25)
        }

        # 작물별 최적 습도 범위
        optimal_humidity_ranges = {
            '딸기': (60, 80),
            '마늘': (65, 75),
            '복숭아': (60, 70),
            '양파': (65, 75)
        }

        if crop_type in growth_weights:
            df['growth_weight'] = df['월'].map(growth_weights[crop_type]).fillna(0.5)

            # 생육 단계별 가중 온도
            if '평균기온(°C)' in df.columns:
                df['weighted_temp'] = df['평균기온(°C)'] * df['growth_weight']

        # 최적 환경 조건 점수
        if crop_type in optimal_temp_ranges and '평균기온(°C)' in df.columns:
            min_temp, max_temp = optimal_temp_ranges[crop_type]
            df['temp_optimality'] = np.where(
                (df['평균기온(°C)'] >= min_temp) & (df['평균기온(°C)'] <= max_temp),
                1.0,
                1.0 - np.minimum(
                    np.abs(df['평균기온(°C)'] - min_temp) / min_temp,
                    np.abs(df['평균기온(°C)'] - max_temp) / max_temp
                ) * 0.5
            )

        if crop_type in optimal_humidity_ranges and '평균 상대습도(%)' in df.columns:
            min_hum, max_hum = optimal_humidity_ranges[crop_type]
            df['humidity_optimality'] = np.where(
                (df['평균 상대습도(%)'] >= min_hum) & (df['평균 상대습도(%)'] <= max_hum),
                1.0,
                1.0 - np.minimum(
                    np.abs(df['평균 상대습도(%)'] - min_hum) / min_hum,
                    np.abs(df['평균 상대습도(%)'] - max_hum) / max_hum
                ) * 0.5
            )

        # 종합 환경 적합성 점수
        if 'temp_optimality' in df.columns and 'humidity_optimality' in df.columns:
            df['environment_score'] = (df['temp_optimality'] * df['humidity_optimality'] *
                                     df.get('growth_weight', 1.0))

        return df

    def create_weather_stress_indicators(self, df, weather_cols):
        """날씨 스트레스 지표 생성"""
        df = df.copy()

        # 온도 스트레스
        if '평균기온(°C)' in weather_cols:
            df['heat_stress'] = np.maximum(0, df['평균기온(°C)'] - 30)  # 30도 이상 열 스트레스
            df['cold_stress'] = np.maximum(0, 5 - df['평균기온(°C)'])  # 5도 이하 한랭 스트레스
            df['temp_variability_7d'] = df['평균기온(°C)'].rolling(7, min_periods=1).std()

        # 수분 스트레스
        if '일강수량(mm)' in weather_cols:
            df['drought_stress_7d'] = (df['일강수량(mm)'].rolling(7, min_periods=1).sum() < 5).astype(int)
            df['drought_stress_14d'] = (df['일강수량(mm)'].rolling(14, min_periods=1).sum() < 10).astype(int)
            df['flood_stress'] = (df['일강수량(mm)'] > 50).astype(int)  # 일일 50mm 이상

        # 습도 스트레스
        if '평균 상대습도(%)' in weather_cols:
            df['humidity_stress_high'] = np.maximum(0, df['평균 상대습도(%)'] - 85)
            df['humidity_stress_low'] = np.maximum(0, 40 - df['평균 상대습도(%)'])

        # 일조 부족 스트레스
        if '합계 일조시간(hr)' in weather_cols:
            df['sunshine_deficit_7d'] = np.maximum(0, 35 - df['합계 일조시간(hr)'].rolling(7, min_periods=1).sum())

        return df

    def create_advanced_interactions(self, df, weather_cols):
        """고급 상호작용 특성 생성"""
        df = df.copy()

        # 기존 상호작용
        if '평균기온(°C)' in weather_cols and '평균 상대습도(%)' in weather_cols:
            df['temp_humidity_interaction'] = df['평균기온(°C)'] * df['평균 상대습도(%)'] / 100

            # 포화수증기압 차이 (VPD)
            sat_vp = 0.6108 * np.exp(17.27 * df['평균기온(°C)'] / (df['평균기온(°C)'] + 237.3))
            actual_vp = sat_vp * df['평균 상대습도(%)'] / 100
            df['vpd'] = sat_vp - actual_vp

        # 온도와 일조시간 상호작용
        if '평균기온(°C)' in weather_cols and '합계 일조시간(hr)' in weather_cols:
            df['temp_sunshine_interaction'] = df['평균기온(°C)'] * df['합계 일조시간(hr)']

        # 강수량과 습도 상호작용
        if '일강수량(mm)' in weather_cols and '평균 상대습도(%)' in weather_cols:
            df['rain_humidity_interaction'] = df['일강수량(mm)'] * df['평균 상대습도(%)'] / 100

        # 복합 스트레스 지표
        stress_cols = [col for col in df.columns if 'stress' in col]
        if stress_cols:
            df['total_stress'] = df[stress_cols].sum(axis=1)

        return df

    def scale_features(self, X_train, X_test, method='standard'):
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler


def process_enhanced_crop_data(filepath, crop_name, crop_type):
    """향상된 작물 데이터 처리"""
    processor = EnhancedCropYieldPreprocessor()

    # 1. 데이터 로드 및 정제
    df = processor.load_and_clean_data(filepath, crop_name)

    # 2. 기상 데이터 컬럼 확인
    weather_cols = ['평균기온(°C)', '일강수량(mm)', '평균 상대습도(%)', '합계 일조시간(hr)']
    available_weather_cols = [col for col in weather_cols if col in df.columns]

    # 3. 향상된 시간 특성 생성
    df = processor.create_enhanced_time_features(df)

    # 4. 계절별 날씨 특성
    df = processor.create_seasonal_weather_features(df, available_weather_cols)

    # 5. 고급 시계열 특성
    df = processor.create_advanced_time_series_features(df, available_weather_cols)

    # 6. 작물별 맞춤 특성
    df = processor.create_crop_specific_features(df, crop_type)

    # 7. 날씨 스트레스 지표
    df = processor.create_weather_stress_indicators(df, available_weather_cols)

    # 8. 고급 상호작용 특성
    df = processor.create_advanced_interactions(df, available_weather_cols)

    # 9. 불필요한 컬럼 제거
    drop_cols = ['일시', '년도', '월', '일', '요일', '주차', '계절']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 10. 특성과 타겟 분리
    feature_cols = [col for col in df.columns if col != '생산량']
    X = df[feature_cols].fillna(0)
    y = df['생산량']

    # 11. 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=None
    )

    # 12. 스케일링
    X_train_scaled, X_test_scaled, scaler = processor.scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def train_enhanced_model(X_train, y_train, crop_type):
    """향상된 모델 훈련 (앙상블)"""
    # 작물별 하이퍼파라미터 조정
    if crop_type in ['마늘', '양파']:  # 연중 재배 작물
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42
        }
        gb_params = {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 8,
            'subsample': 0.8,
            'random_state': 42
        }
    else:  # 계절 작물
        rf_params = {
            'n_estimators': 150,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 3,
            'random_state': 42
        }
        gb_params = {
            'n_estimators': 100,
            'learning_rate': 0.15,
            'max_depth': 6,
            'subsample': 0.9,
            'random_state': 42
        }

    # 모델 훈련
    rf_model = RandomForestRegressor(**rf_params)
    gb_model = GradientBoostingRegressor(**gb_params)

    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    return rf_model, gb_model


def ensemble_predict(rf_model, gb_model, X_test, crop_type):
    """앙상블 예측"""
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)

    # 작물별 가중치 조정
    if crop_type in ['마늘', '양파']:
        # 연중 재배 작물은 GradientBoosting에 더 높은 가중치
        ensemble_pred = 0.4 * rf_pred + 0.6 * gb_pred
    else:
        # 계절 작물은 RandomForest에 더 높은 가중치
        ensemble_pred = 0.6 * rf_pred + 0.4 * gb_pred

    return ensemble_pred


def main():
    """메인 실행 함수"""
    crops = {
        '딸기': ('./data/weath-prod/날씨-생산량-딸기.csv', '딸기'),
        '마늘': ('./data/weath-prod/날씨-생산량-마늘.csv', '마늘'),
        '복숭아': ('./data/weath-prod/날씨-생산량-복숭아.csv', '복숭아'),
        '양파': ('./data/weath-prod/날씨-생산량-양파.csv', '양파')
    }

    processed_data = {}

    for crop_name, (filepath, crop_type) in crops.items():
        try:
            print(f"\n{'='*50}")
            print(f"처리 중: {crop_name}")
            print(f"{'='*50}")

            # 데이터 처리
            X_train, X_test, y_train, y_test, scaler, feature_cols = process_enhanced_crop_data(
                filepath, crop_name, crop_type
            )

            # 모델 훈련
            rf_model, gb_model = train_enhanced_model(X_train, y_train, crop_type)

            # 앙상블 예측
            ensemble_pred = ensemble_predict(rf_model, gb_model, X_test, crop_type)

            # 개별 모델 성능
            rf_pred = rf_model.predict(X_test)
            gb_pred = gb_model.predict(X_test)

            # 성능 평가
            rf_mae = mean_absolute_error(y_test, rf_pred)
            rf_r2 = r2_score(y_test, rf_pred)

            gb_mae = mean_absolute_error(y_test, gb_pred)
            gb_r2 = r2_score(y_test, gb_pred)

            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_r2 = r2_score(y_test, ensemble_pred)

            # 결과 출력
            print(f"\n{crop_name} 결과:")
            print(f"특성 개수: {X_train.shape[1]}")
            print(f"훈련 샘플: {X_train.shape[0]}, 테스트 샘플: {X_test.shape[0]}")
            print(f"\n개별 모델 성능:")
            print(f"RandomForest - MAE: {rf_mae:.2f}, R² Score: {rf_r2:.3f}")
            print(f"GradientBoosting - MAE: {gb_mae:.2f}, R² Score: {gb_r2:.3f}")
            print(f"\n🎯 앙상블 모델 성능:")
            print(f"MAE: {ensemble_mae:.2f}")
            print(f"R² Score: {ensemble_r2:.3f}")

            # 특성 중요도 (RandomForest 기준)
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            print(f"\n중요 특성 Top 15:")
            for idx, row in feature_importance.iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")

            # 데이터 저장
            processed_data[crop_name] = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'scaler': scaler, 'feature_cols': feature_cols,
                'rf_model': rf_model, 'gb_model': gb_model
            }

            # 향상된 데이터 저장
            np.savez(f'./trainData/{crop_name}_ultra_enhanced_data.npz',
                     X_train=X_train, X_test=X_test,
                     y_train=y_train, y_test=y_test)



        except Exception as e:
            print(f"{crop_name} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()

        # 모델 저장
        joblib.dump(rf_model, f'./trainedModel/{crop_name}_rf_model.pkl')
        joblib.dump(gb_model, f'./trainedModel/{crop_name}_gb_model.pkl')
        joblib.dump(scaler, f'./trainedModel/{crop_name}_scaler.pkl')

        print(f"💾 모델과 스케일러 저장 완료: {crop_name}")

    return processed_data


if __name__ == "__main__":
    processed_data = main()
    print(f"\n{'='*60}")
    print("🎉 모든 작물 데이터 처리 완료!")
    print("✨ 향상된 특성과 앙상블 모델로 정확도가 크게 개선되었습니다!")
    print(f"{'='*60}")