import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import warnings
import os
import pickle
import json

warnings.filterwarnings('ignore')


class AgricultureDataPreprocessor:
    def __init__(self):
        self.scaler_features = RobustScaler()
        self.scaler_target = RobustScaler()

    def create_time_features(self, df):
        df = df.copy()
        df['년'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        df['일'] = df['날짜'].dt.day
        df['요일'] = df['날짜'].dt.dayofweek
        df['년중일차'] = df['날짜'].dt.dayofyear
        df['분기'] = df['날짜'].dt.quarter

        # 순환 인코딩
        df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
        df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)
        df['일_sin'] = np.sin(2 * np.pi * df['년중일차'] / 365)
        df['일_cos'] = np.cos(2 * np.pi * df['년중일차'] / 365)

        # 계절 특성
        df['봄'] = ((df['월'] >= 3) & (df['월'] <= 5)).astype(int)
        df['여름'] = ((df['월'] >= 6) & (df['월'] <= 8)).astype(int)
        df['가을'] = ((df['월'] >= 9) & (df['월'] <= 11)).astype(int)
        df['겨울'] = ((df['월'] == 12) | (df['월'] <= 2)).astype(int)

        return df

    def create_lag_features(self, df, target_col='가격', lags=[1, 3, 7, 14, 30]):
        df = df.copy()

        # 래그 특성
        for lag in lags:
            df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
            df[f'생산량_lag{lag}'] = df['생산량'].shift(lag)

        # 이동평균
        for window in [7, 14, 30]:
            df[f'{target_col}_ma{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'생산량_ma{window}'] = df['생산량'].rolling(window=window, min_periods=1).mean()

        # 변동성
        for window in [7, 30]:
            df[f'{target_col}_std{window}'] = df[target_col].rolling(window=window, min_periods=1).std()

        return df

    def create_interaction_features(self, df):
        df = df.copy()

        # 계절별 생산량
        df['생산량_봄'] = df['생산량'] * df['봄']
        df['생산량_여름'] = df['생산량'] * df['여름']
        df['생산량_가을'] = df['생산량'] * df['가을']
        df['생산량_겨울'] = df['생산량'] * df['겨울']

        # 비율 특성
        df['생산량_가격비율'] = df['생산량'] / (df['가격'] + 1e-8)
        df['가격_생산량비율'] = df['가격'] / (df['생산량'] + 1e-8)

        return df

    def preprocess_data(self, df):
        df = df.sort_values('날짜').reset_index(drop=True)
        df = self.create_time_features(df)

        # 로그 변환
        df['가격'] = np.log1p(df['가격'])
        df['생산량'] = np.log1p(df['생산량'])

        df = self.create_lag_features(df)
        df = self.create_interaction_features(df)
        df = df.interpolate(method='linear').ffill().bfill()

        return df


class ImprovedLSTMModel:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.feature_names = []

    def create_model(self, n_features):
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                          input_shape=(self.sequence_length, n_features)),
            Dropout(0.3),
            BatchNormalization(),

            LSTM(32, return_sequences=False, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(0.3),
            BatchNormalization(),

            Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        return model

    def create_sequences(self, data, target):
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def prepare_features(self, df):
        # 핵심 특성만 선택
        priority_features = [
            '생산량', '가격_lag1', '가격_lag7', '가격_lag30',
            '월_sin', '월_cos', '일_sin', '일_cos',
            '봄', '여름', '가을', '겨울',
            '생산량_봄', '생산량_여름', '생산량_가을', '생산량_겨울',
            '가격_ma7', '가격_ma14', '가격_ma30',
            '생산량_가격비율', '가격_생산량비율',
            '가격_std7', '가격_std30'
        ]

        available_features = [col for col in priority_features if col in df.columns]
        return available_features

    def prepare_data(self, df, fit_scalers=True):
        self.feature_names = self.prepare_features(df)

        X = df[self.feature_names].values
        y = df['가격'].values

        if fit_scalers:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            X_scaled = self.scaler_X.transform(X)
            y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()

        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        return X_seq, y_seq

    def train_model(self, df, test_size=0.2, item_eng='model'):
        X_seq, y_seq = self.prepare_data(df)

        split_idx = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        val_split_idx = int(len(X_train) * 0.8)
        X_val = X_train[val_split_idx:]
        y_val = y_train[val_split_idx:]
        X_train = X_train[:val_split_idx]
        y_train = y_train[:val_split_idx]

        self.model = self.create_model(X_train.shape[2])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0),
            ModelCheckpoint(f'best_model_{item_eng}.h5', monitor='val_loss', save_best_only=True, verbose=0)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=callbacks,
            verbose=0
        )

        # 테스트 성능 평가
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = np.expm1(self.scaler_y.inverse_transform(y_pred_scaled).flatten())
        y_true = np.expm1(self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten())

        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
            'wmape': wmape,
            'y_true': y_true,
            'y_pred': y_pred
        }

    def save_model(self, model_path, scaler_path):
        """모델과 스케일러 저장"""
        self.model.save(model_path)

        scaler_data = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length
        }
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_data, f)

        print(f"모델 저장 완료: {model_path}")

    def load_model(self, model_path, scaler_path):
        """저장된 모델과 스케일러 로드"""
        try:
            self.model = load_model(model_path)

            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)

            self.scaler_X = scaler_data['scaler_X']
            self.scaler_y = scaler_data['scaler_y']
            self.feature_names = scaler_data['feature_names']
            self.sequence_length = scaler_data['sequence_length']

            print(f"모델 로드 완료: {model_path}")
            return True
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            return False

    def predict_future(self, df, days_ahead=7):
        """미래 가격 예측"""
        X_seq, _ = self.prepare_data(df, fit_scalers=False)

        if len(X_seq) == 0:
            return None

        last_sequence = X_seq[-1:].copy()
        predictions = []

        for i in range(days_ahead):
            pred_scaled = self.model.predict(last_sequence, verbose=0)
            pred_price = np.expm1(self.scaler_y.inverse_transform(pred_scaled).flatten()[0])
            predictions.append(max(pred_price, 0))

            # 시퀀스 업데이트 (실제로는 더 정교한 방법이 필요)
            last_sequence = np.roll(last_sequence, -1, axis=1)

        last_date = df['날짜'].max()
        future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(days_ahead)]

        return pd.DataFrame({
            '날짜': future_dates,
            '예측가격': predictions
        })


def load_or_train_model(item, df_processed, force_retrain=False):
    """모델이 있으면 로드, 없으면 학습 후 저장"""
    name_map = {'양파': 'onion', '마늘': 'garlic', '딸기': 'strawberry', '복숭아': 'peach'}
    item_eng = name_map.get(item, item)

    # 모델 저장 경로
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)

    model_path = f"{model_dir}/{item_eng}_model.h5"
    scaler_path = f"{model_dir}/{item_eng}_scalers.pkl"

    model = ImprovedLSTMModel(sequence_length=30)

    # 모델이 존재하고 재학습을 강제하지 않는 경우 로드 시도
    if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        if model.load_model(model_path, scaler_path):
            # 기존 모델 성능 평가
            X_seq, y_seq = model.prepare_data(df_processed, fit_scalers=False)
            test_size = int(len(X_seq) * 0.2)
            X_test = X_seq[-test_size:]
            y_test = y_seq[-test_size:]

            y_pred_scaled = model.model.predict(X_test, verbose=0)
            y_pred = np.expm1(model.scaler_y.inverse_transform(y_pred_scaled).flatten())
            y_true = np.expm1(model.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten())

            wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

            results = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
                'wmape': wmape,
                'y_true': y_true,
                'y_pred': y_pred
            }

            return model, results, False  # 기존 모델 사용

    # 새 모델 학습
    print(f"{item} 새 모델 학습 중...")
    results = model.train_model(df_processed, test_size=0.2, item_eng=item_eng)

    # 모델 저장
    model.save_model(model_path, scaler_path)

    return model, results, True  # 새 모델 학습됨


def main():
    """메인 실행 함수"""
    items = ['양파', '마늘', '딸기', '복숭아']
    all_results = {}
    all_models = {}

    print("=== 농산물 가격 예측 모델 ===\n")

    for item in items:
        print(f"\n{item} 처리 중...")

        # 데이터 로드 (실제 파일이 없으면 샘플 데이터 생성)
        file_path = f'./data/prod-price/{item}_생산량_가격_데이터.csv'


        df = pd.read_csv(file_path)
        df['날짜'] = pd.to_datetime(df['날짜'])
        print(f"{item} 데이터 로드 완료")


        # 데이터 전처리
        preprocessor = AgricultureDataPreprocessor()
        df_processed = preprocessor.preprocess_data(df)

        # 모델 로드 또는 학습
        model, results, is_new = load_or_train_model(item, df_processed, force_retrain=False)

        all_results[item] = results
        all_models[item] = model

        status = "새로 학습됨" if is_new else "기존 모델 로드됨"
        print(f"{item} - R²: {results['r2']:.4f}, MAPE: {results['mape']:.2f}% ({status})")

        # 미래 예측
        future_pred = model.predict_future(df_processed, days_ahead=7)
        if future_pred is not None:
            print(f"{item} 7일 후 예측 가격: {future_pred['예측가격'].iloc[-1]:.0f}원")

    print("\n=== 전체 성능 요약 ===")
    for item, results in all_results.items():
        print(f"{item} - R²: {results['r2']:.4f}, MAPE: {results['mape']:.2f}%")

    return all_models, all_results


if __name__ == "__main__":
    models, results = main()