import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')


class FruitSpecificPreprocessor:
    def __init__(self, fruit_type):
        self.fruit_type = fruit_type
        self.scaler_features = RobustScaler()
        self.scaler_target = RobustScaler()

    def create_fruit_specific_features(self, df):
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
        df = df.copy()

        for lag in [1, 2, 3, 7]:
            df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
            df[f'생산량_lag{lag}'] = df['생산량'].shift(lag)

        for lag in [30, 60]:
            df[f'{target_col}_seasonal_lag{lag}'] = df[target_col].shift(lag)

        for window in [3, 7, 14]:
            df[f'{target_col}_ma{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()

        df[f'{target_col}_std_3d'] = df[target_col].rolling(3, min_periods=1).std().fillna(0)
        df[f'{target_col}_std_7d'] = df[target_col].rolling(7, min_periods=1).std().fillna(0)

        df[f'{target_col}_diff1'] = df[target_col].diff().fillna(0)
        df[f'{target_col}_diff2'] = df[f'{target_col}_diff1'].diff().fillna(0)

        return df

    def safe_log_transform(self, series):
        return np.log1p(np.maximum(series, 0))

    def preprocess_data(self, df):
        print(f"🔄 {self.fruit_type} 전처리 시작...")

        df = df.sort_values('날짜').reset_index(drop=True)

        df['가격'] = pd.to_numeric(df['가격'], errors='coerce')
        df['생산량'] = pd.to_numeric(df['생산량'], errors='coerce')

        df = df.dropna(subset=['가격', '생산량'])

        if len(df) == 0:
            raise ValueError("유효한 데이터가 없습니다.")

        df = self.create_fruit_specific_features(df)
        df = self.create_lag_features(df)

        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        df['가격_원본'] = df['가격'].copy()
        df['생산량_원본'] = df['생산량'].copy()

        df['가격'] = self.safe_log_transform(df['가격'])
        df['생산량'] = self.safe_log_transform(df['생산량'])

        price_lag_cols = [col for col in df.columns if '가격_lag' in col or '가격_seasonal' in col or '가격_ma' in col]
        production_lag_cols = [col for col in df.columns if '생산량_lag' in col]

        for col in price_lag_cols:
            df[col] = self.safe_log_transform(df[col])
        for col in production_lag_cols:
            df[col] = self.safe_log_transform(df[col])

        print(f"✅ 전처리 완료. 최종 데이터 크기: {df.shape}")
        return df


class FruitSpecificLSTM:
    def __init__(self, fruit_type, sequence_length=21):
        self.fruit_type = fruit_type
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()

    def create_fruit_model(self, n_features):
        if self.fruit_type in ['딸기', '복숭아']:
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(self.sequence_length, n_features)),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                BatchNormalization(),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            optimizer = Adam(learning_rate=0.001)
        else:
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(self.sequence_length, n_features)),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            optimizer = Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def prepare_fruit_data(self, df):
        if self.fruit_type == '딸기':
            feature_cols = [
                '생산량', '딸기시즌', '딸기성수기', '딸기비수기',
                '딸기_주기_sin', '딸기_주기_cos', '온도민감도',
                '가격_lag1', '가격_lag2', '가격_lag3', '가격_lag7',
                '가격_ma3', '가격_ma7', '가격_std_3d', '가격_diff1',
                '연말연시', '추석시즌', '월', '요일'
            ]
        elif self.fruit_type == '복숭아':
            feature_cols = [
                '생산량', '복숭아시즌', '복숭아성수기', '복숭아비수기',
                '복숭아_주기_sin', '복숭아_주기_cos', '온도민감도',
                '가격_lag1', '가격_lag2', '가격_lag3', '가격_lag7',
                '가격_ma3', '가격_ma7', '가격_std_3d', '가격_diff1',
                '월', '요일'
            ]
        else:
            feature_cols = [
                '생산량', '월', '일', '요일',
                '가격_lag1', '가격_lag3', '가격_lag7',
                '가격_ma7', '가격_ma14'
            ]

        available_features = [col for col in feature_cols if col in df.columns]

        if len(available_features) < 3:
            basic_features = ['월', '일', '요일', '년중일차']
            for feat in basic_features:
                if feat in df.columns and feat not in available_features:
                    available_features.append(feat)

        print(f"사용 특성 ({len(available_features)}개): {available_features}")

        X = df[available_features].values
        y = df['가격'].values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        self.feature_names = available_features # ✏️ 특성 리스트를 저장

        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        return X_seq, y_seq, available_features

    def create_sequences(self, data, target):
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def train(self, df, test_size=0.2):
        print(f"🚀 {self.fruit_type} 모델 학습 시작...")

        X_seq, y_seq, feature_names = self.prepare_fruit_data(df)
        print(f"시퀀스 데이터 형태: X={X_seq.shape}, y={y_seq.shape}")

        if len(X_seq) < 50:
            print("⚠️ 데이터가 너무 적습니다. 최소 50개 이상의 시퀀스가 필요합니다.")
            return None

        split_idx = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        self.model = self.create_fruit_model(X_train.shape[2])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=1e-6, verbose=1)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        y_pred_scaled = self.model.predict(X_test)

        y_pred = np.expm1(self.scaler_y.inverse_transform(y_pred_scaled).flatten())
        y_true = np.expm1(self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten())

        y_pred = np.maximum(y_pred, 0)
        y_true = np.maximum(y_true, 0)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        safe_mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

        print(f"\n📊 {self.fruit_type} 모델 성능:")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {safe_mape:.2f}%")

        self.save_model()  # ➕ 학습 완료 후 저장

        return {
            'mse': mse, 'mae': mae, 'r2': r2, 'mape': safe_mape,
            'y_true': y_true, 'y_pred': y_pred, 'history': history
        }

    def save_model(self, save_dir='./models'):
        import os, json, joblib

        os.makedirs(save_dir, exist_ok=True)

        item = {'마늘': 'gar', '양파': 'oni', '딸기':'str', '복숭아':'pch'}
        item_eng = item.get(self.fruit_type, item)

        # 모델 저장
        model_path = os.path.join(save_dir, f'{item_eng}_model.h5')
        self.model.save(model_path)

        # 스케일러 저장
        joblib.dump(self.scaler_X, os.path.join(save_dir, f'{self.fruit_type}_scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(save_dir, f'{self.fruit_type}_scaler_y.pkl'))

        # 메타 정보 저장
        meta = {
            'fruit_type': self.fruit_type,
            'sequence_length': self.sequence_length,
            'feature_names': self.feature_names
        }
        meta_path = os.path.join(save_dir, f'{self.fruit_type}_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)

        print(f"💾 모델 및 구성 저장 완료: {save_dir}")


def main():
    print("=" * 50)
    print("🌾 과일 특화 농산물 가격 예측 AI 모델")
    print("=" * 50)

    all_results = {}
    items = ['양파', '마늘', '딸기', '복숭아']

    for item in items:
        print(f"\n🔄 {item} 모델 학습 중...")

        file_path = f'./data/prod-price/{item}_생산량_가격_데이터.csv'

        try:
            df = pd.read_csv(file_path)
            df['날짜'] = pd.to_datetime(df['날짜'])
            print(f"✅ 데이터 로드 성공: {df.shape}")

            preprocessor = FruitSpecificPreprocessor(item)
            df_processed = preprocessor.preprocess_data(df)

            model = FruitSpecificLSTM(item, sequence_length=21)
            results = model.train(df_processed, test_size=0.2)

            if results:
                all_results[item] = results
                print(f"✅ {item} 모델 완료 - R²: {results['r2']:.4f}, MAPE: {results['mape']:.2f}%")
            else:
                print(f"❌ {item} 모델 학습 실패")

        except FileNotFoundError:
            print(f"⚠️ {item} 데이터 파일을 찾을 수 없습니다: {file_path}")
            continue
        except Exception as e:
            print(f"❌ {item} 처리 중 오류 발생: {str(e)}")
            continue

    if all_results:
        print("\n📊 전체 모델 성능 비교")
        print("=" * 40)
        for item, result in all_results.items():
            print(f"📈 {item:4s} | R²: {result['r2']:6.4f} | MAPE: {result['mape']:6.2f}%")
    else:
        print("❌ 성공적으로 학습된 모델이 없습니다.")

    return all_results


if __name__ == '__main__':
    main()