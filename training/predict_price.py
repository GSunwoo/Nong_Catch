import pandas as pd
import numpy as np
import os
import json
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')


class FruitPricePredictor:
    def __init__(self, fruit_type, model_dir='./models'):
        self.fruit_type = fruit_type
        self.model_dir = model_dir
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.meta_info = None
        self.feature_names = None

    def load_model_components(self):
        """저장된 모델과 스케일러, 메타정보를 불러옵니다."""
        try:
            # 모델 파일명 매핑
            item_mapping = {'마늘': 'gar', '양파': 'oni', '딸기': 'str', '복숭아': 'pch'}
            item_eng = item_mapping.get(self.fruit_type, self.fruit_type)

            # 모델 로드
            model_path = os.path.join(self.model_dir, f'{item_eng}_model.h5')
            self.model = load_model(model_path)

            # 스케일러 로드
            scaler_X_path = os.path.join(self.model_dir, f'{self.fruit_type}_scaler_X.pkl')
            scaler_y_path = os.path.join(self.model_dir, f'{self.fruit_type}_scaler_y.pkl')
            self.scaler_X = joblib.load(scaler_X_path)
            self.scaler_y = joblib.load(scaler_y_path)

            # 메타정보 로드
            meta_path = os.path.join(self.model_dir, f'{self.fruit_type}_meta.json')
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.meta_info = json.load(f)

            self.feature_names = self.meta_info['feature_names']
            self.sequence_length = self.meta_info['sequence_length']

            print(f"✅ {self.fruit_type} 모델 구성요소 로드 완료")
            return True

        except Exception as e:
            print(f"❌ {self.fruit_type} 모델 로드 실패: {str(e)}")
            return False

    def load_production_data(self, data_dir='./data/pred'):
        """생산량 예측 데이터를 불러옵니다."""
        try:
            file_path = os.path.join(data_dir, f'{self.fruit_type}_생산량_예측결과.csv')
            production_df = pd.read_csv(file_path, encoding='utf-8-sig')

            # 날짜 컬럼 확인 및 변환
            date_cols = ['날짜', 'date', '일자', 'Date']
            date_col = None
            for col in date_cols:
                if col in production_df.columns:
                    date_col = col
                    break

            if date_col is None:
                print(f"❌ {self.fruit_type} 생산량 데이터에서 날짜 컬럼을 찾을 수 없습니다.")
                print(f"사용 가능한 컬럼: {production_df.columns.tolist()}")
                return None

            production_df['날짜'] = pd.to_datetime(production_df[date_col])

            # 생산량 컬럼 확인
            prod_cols = ['생산량', 'production', '예측생산량', '생산량예측']
            prod_col = None
            for col in prod_cols:
                if col in production_df.columns:
                    prod_col = col
                    break

            if prod_col is None:
                print(f"❌ {self.fruit_type} 생산량 데이터에서 생산량 컬럼을 찾을 수 없습니다.")
                print(f"사용 가능한 컬럼: {production_df.columns.tolist()}")
                return None

            production_df['생산량'] = pd.to_numeric(production_df[prod_col], errors='coerce')

            # 필요한 컬럼만 선택
            production_df = production_df[['날짜', '생산량']].dropna()
            production_df = production_df.sort_values('날짜').reset_index(drop=True)

            print(f"✅ {self.fruit_type} 생산량 데이터 로드 완료: {len(production_df)}행")
            return production_df

        except Exception as e:
            print(f"❌ {self.fruit_type} 생산량 데이터 로드 실패: {str(e)}")
            return None

    def create_fruit_specific_features(self, df):
        """과일별 특화 특성을 생성합니다."""
        df = df.copy()

        df['년'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        df['일'] = df['날짜'].dt.day
        df['요일'] = df['날짜'].dt.dayofweek
        df['년중일차'] = df['날짜'].dt.dayofyear

        if self.fruit_type == '딸기':
            df['딸기시즌'] = ((df['월'] >= 11) | (df['월'] <= 4)).astype(int)
            df['딸기성수기'] = ((df['월'] >= 12) & (df['월'] <= 2)).astype(int)
            df['딸기비수기'] = ((df['월'] >= 5) & (df['월'] <= 10)).astype(int)
            df['딸기_주기_sin'] = np.sin(2 * np.pi * df['년중일차'] / 365 * 4)
            df['딸기_주기_cos'] = np.cos(2 * np.pi * df['년중일차'] / 365 * 4)

        elif self.fruit_type == '복숭아':
            df['복숭아시즌'] = ((df['월'] >= 6) & (df['월'] <= 9)).astype(int)
            df['복숭아성수기'] = ((df['월'] >= 7) & (df['월'] <= 8)).astype(int)
            df['복숭아비수기'] = ((df['월'] <= 5) | (df['월'] >= 10)).astype(int)
            df['복숭아_주기_sin'] = np.sin(2 * np.pi * (df['년중일차'] - 180) / 365 * 2)
            df['복숭아_주기_cos'] = np.cos(2 * np.pi * (df['년중일차'] - 180) / 365 * 2)

        df['온도민감도'] = np.abs(np.sin(2 * np.pi * df['년중일차'] / 365)) * df['월']
        df['연말연시'] = ((df['월'] == 12) | (df['월'] == 1)).astype(int)
        df['추석시즌'] = ((df['월'] == 9) & (df['일'] >= 10)).astype(int)

        return df

    def create_lag_features(self, df, target_col='가격'):
        """지연 특성을 생성합니다."""
        df = df.copy()

        # 기본 가격을 생산량 기반으로 생성 (가격 예측의 시작점)
        if target_col not in df.columns:
            # 생산량에 반비례하는 기본 가격 패턴 생성
            base_price = 1000 if self.fruit_type in ['딸기', '복숭아'] else 500
            df[target_col] = base_price * (df['생산량'].max() / (df['생산량'] + 1))

        # 기본 지연 특성
        for lag in [1, 2, 3, 7]:
            df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
            df[f'생산량_lag{lag}'] = df['생산량'].shift(lag)

        # 계절적 지연 특성
        for lag in [30, 60]:
            df[f'{target_col}_seasonal_lag{lag}'] = df[target_col].shift(lag)

        # 이동평균
        for window in [3, 7, 14]:
            df[f'{target_col}_ma{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()

        # 표준편차
        df[f'{target_col}_std_3d'] = df[target_col].rolling(3, min_periods=1).std().fillna(0)
        df[f'{target_col}_std_7d'] = df[target_col].rolling(7, min_periods=1).std().fillna(0)

        # 차분
        df[f'{target_col}_diff1'] = df[target_col].diff().fillna(0)
        df[f'{target_col}_diff2'] = df[f'{target_col}_diff1'].diff().fillna(0)

        return df

    def safe_log_transform(self, series):
        """안전한 로그 변환을 적용합니다."""
        return np.log1p(np.maximum(series, 0))

    def preprocess_data(self, df):
        """데이터 전처리를 수행합니다."""
        df = df.sort_values('날짜').reset_index(drop=True)

        # 수치형 변환
        df['생산량'] = pd.to_numeric(df['생산량'], errors='coerce')

        # 결측치 제거
        df = df.dropna(subset=['생산량'])

        if len(df) == 0:
            raise ValueError("유효한 데이터가 없습니다.")

        # 특성 생성
        df = self.create_fruit_specific_features(df)
        df = self.create_lag_features(df)

        # 결측치 처리
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 원본 데이터 보존
        df['생산량_원본'] = df['생산량'].copy()
        if '가격' in df.columns:
            df['가격_원본'] = df['가격'].copy()

        # 로그 변환
        df['생산량'] = self.safe_log_transform(df['생산량'])
        if '가격' in df.columns:
            df['가격'] = self.safe_log_transform(df['가격'])

        # 지연 특성에도 로그 변환 적용
        price_lag_cols = [col for col in df.columns if '가격_lag' in col or '가격_seasonal' in col or '가격_ma' in col]
        production_lag_cols = [col for col in df.columns if '생산량_lag' in col]

        for col in price_lag_cols:
            df[col] = self.safe_log_transform(df[col])
        for col in production_lag_cols:
            df[col] = self.safe_log_transform(df[col])

        return df

    def create_sequences(self, data):
        """LSTM용 시퀀스 데이터를 생성합니다."""
        X = []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
        return np.array(X)

    def predict_prices(self, production_df):
        """생산량 데이터를 기반으로 가격을 예측합니다."""
        if not self.model:
            print("❌ 모델이 로드되지 않았습니다.")
            return None

        try:
            # 데이터 전처리
            df_processed = self.preprocess_data(production_df)
            print(f"전처리 완료: {df_processed.shape}")

            # 특성 선택 (학습할 때와 동일한 특성 사용)
            available_features = [col for col in self.feature_names if col in df_processed.columns]

            if len(available_features) != len(self.feature_names):
                print(f"⚠️ 일부 특성이 누락되었습니다. 사용 가능한 특성: {len(available_features)}/{len(self.feature_names)}")
                print(f"누락된 특성: {set(self.feature_names) - set(available_features)}")

            if len(available_features) == 0:
                print("❌ 사용할 수 있는 특성이 없습니다.")
                return None

            X = df_processed[available_features].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # 스케일링
            X_scaled = self.scaler_X.transform(X)

            # 시퀀스 생성
            X_seq = self.create_sequences(X_scaled)
            print(f"시퀀스 생성 완료: {X_seq.shape}")

            if len(X_seq) == 0:
                print("❌ 시퀀스 생성에 실패했습니다. 데이터가 충분하지 않습니다.")
                return None

            # 예측
            y_pred_scaled = self.model.predict(X_seq, verbose=0)
            y_pred = np.expm1(self.scaler_y.inverse_transform(y_pred_scaled).flatten())

            # 음수 제거
            y_pred = np.maximum(y_pred, 0)

            # 예측 결과를 원본 데이터프레임에 매핑
            prediction_df = df_processed.iloc[self.sequence_length:].copy().reset_index(drop=True)
            prediction_df['예측가격'] = y_pred.astype(int)  # int형으로 변환

            # ★ 생산량이 0이면 가격도 0으로 설정 ★
            zero_production_mask = prediction_df['생산량_원본'] <= 0
            prediction_df.loc[zero_production_mask, '예측가격'] = 0

            print(f"생산량 0인 날짜 수: {zero_production_mask.sum()}일")

            # 필요한 컬럼만 선택하여 반환
            result_df = prediction_df[['날짜', '생산량_원본', '예측가격']].copy()
            result_df['생산량_원본'] = result_df['생산량_원본'].astype(int)  # int형으로 변환
            result_df.columns = ['날짜', '생산량', '예측가격']

            print(f"예측 완료: {len(y_pred)}개 예측값 생성")
            return result_df

        except Exception as e:
            print(f"❌ 예측 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def create_annual_summary(daily_results):
    """일별 예측 결과를 연간 요약으로 변환합니다."""
    annual_data = []

    for fruit_type, df in daily_results.items():
        if df is not None and len(df) > 0:
            df_copy = df.copy()
            df_copy['년도'] = df_copy['날짜'].dt.year

            # 과일별 시즌 고려
            if fruit_type == '딸기':
                # 딸기 시즌: 11월~4월
                season_mask = (df_copy['날짜'].dt.month >= 11) | (df_copy['날짜'].dt.month <= 4)
                season_df = df_copy[season_mask]
            elif fruit_type == '복숭아':
                # 복숭아 시즌: 6월~9월
                season_mask = (df_copy['날짜'].dt.month >= 6) & (df_copy['날짜'].dt.month <= 9)
                season_df = df_copy[season_mask]
            else:
                # 마늘, 양파: 연중
                season_df = df_copy

            # 연간 집계
            if len(season_df) > 0:
                yearly_data = []
                for year in season_df['년도'].unique():
                    year_data = season_df[season_df['년도'] == year]

                    # 총 생산량
                    total_production = int(year_data['생산량'].sum())

                    # ★ 가격이 0이 아닌 날만으로 평균 계산 ★
                    non_zero_prices = year_data[year_data['예측가격'] > 0]['예측가격']

                    if len(non_zero_prices) > 0:
                        avg_price = int(non_zero_prices.mean())  # 0이 아닌 가격들의 평균
                        non_zero_days = len(non_zero_prices)
                        zero_days = len(year_data) - non_zero_days

                        print(
                            f"{fruit_type} {year}년: 총 {len(year_data)}일 중 가격 0원인 날 {zero_days}일, 평균계산 대상 {non_zero_days}일")
                    else:
                        avg_price = 0  # 모든 날이 0원인 경우
                        print(f"{fruit_type} {year}년: 모든 날이 0원")

                    yearly_data.append({
                        '품목명': fruit_type,
                        '년도': year,
                        '총생산량': total_production,
                        '연평균가격': avg_price
                    })

                annual_data.extend(yearly_data)

    return pd.DataFrame(annual_data)


def main():
    print("=" * 50)
    print("🔮 농산물 가격 예측 및 CSV 생성")
    print("=" * 50)

    # 결과 저장 폴더 생성
    price_dir = './predict_price'
    year_dir = './predict_year'
    os.makedirs(price_dir, exist_ok=True)
    os.makedirs(year_dir, exist_ok=True)

    items = ['양파', '마늘', '딸기', '복숭아']
    daily_results = {}

    for item in items:
        print(f"\n🔄 {item} 가격 예측 중...")

        try:
            # 예측기 초기화 및 모델 로드
            predictor = FruitPricePredictor(item)
            if not predictor.load_model_components():
                continue

            # 생산량 예측 데이터 로드
            production_data = predictor.load_production_data()
            if production_data is None:
                continue

            # 가격 예측
            result_df = predictor.predict_prices(production_data)

            if result_df is not None and len(result_df) > 0:
                # 1. 일별 가격 예측 결과 저장
                price_filename = f'{item}_가격_예측결과.csv'
                price_path = os.path.join(price_dir, price_filename)
                result_df.to_csv(price_path, index=False, encoding='utf-8-sig')

                daily_results[item] = result_df

                print(f"✅ {item} 일별 가격 예측 완료 - 저장된 데이터: {len(result_df)}행")
                print(f"📁 저장 위치: {price_path}")

                # 결과 미리보기
                print("📊 일별 예측 결과 미리보기:")
                print(result_df.head())

            else:
                print(f"❌ {item} 예측 실패")

        except Exception as e:
            print(f"❌ {item} 처리 중 오류 발생: {str(e)}")
            continue

    # 2. 연간 요약 데이터 생성
    if daily_results:
        print(f"\n🔄 연간 요약 데이터 생성 중...")
        annual_summary = create_annual_summary(daily_results)

        if len(annual_summary) > 0:
            # 품목명을 인덱스로 설정
            annual_summary_grouped = annual_summary.groupby('품목명').agg({
                '총생산량': 'sum',
                '연평균가격': 'mean'
            }).round().astype(int)

            year_filename = '품목별_연간예측_데이터.csv'
            year_path = os.path.join(year_dir, year_filename)
            annual_summary_grouped.to_csv(year_path, encoding='utf-8-sig')

            print(f"✅ 연간 요약 데이터 생성 완료")
            print(f"📁 저장 위치: {year_path}")
            print("📊 연간 요약 데이터:")
            print(annual_summary_grouped)
        else:
            print("⚠️ 연간 요약 데이터를 생성할 수 없습니다.")

    print(f"\n✅ 모든 예측 완료!")
    print(f"📁 일별 가격 예측: {price_dir}")
    print(f"📁 연간 요약 데이터: {year_dir}")


if __name__ == '__main__':
    main()