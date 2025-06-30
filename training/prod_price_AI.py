# ==========================================================================================================
# 📦 개선된 LSTM 농산물 가격 예측 모델 (안정화 버전)
# ==========================================================================================================
# 이 프로그램은 농산물(양파) 가격을 예측하는 딥러닝 모델입니다.
#
# 주요 기능:
# 1. 시계열 데이터의 특성 엔지니어링 (시간, 지연, 상호작용 변수 생성)
# 2. LSTM 신경망을 이용한 가격 예측
# 3. 전처리 파이프라인과 성능 평가
# 4. 결과 시각화
#
# 작성자: AI Assistant
# 버전: 2.0 (안정화 버전)
# ==========================================================================================================

# ----------------------------------------
# 🔹 필수 라이브러리 임포트
# ----------------------------------------
import pandas as pd  # 데이터 조작 및 분석을 위한 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 데이터 정규화를 위한 전처리 도구
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 모델 성능 평가 지표
from tensorflow.keras.models import Sequential, Model  # 딥러닝 모델 구조 정의
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Input, concatenate  # 신경망 레이어
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # 학습 최적화 콜백
from tensorflow.keras.optimizers import Adam  # 최적화 알고리즘
import warnings  # 경고 메시지 제어

# 불필요한 경고 메시지 숨기기 (모델 학습 시 나오는 deprecation warning 등)
warnings.filterwarnings('ignore')

# ----------------------------------------
# 🔹 한글 폰트 설정 (그래프에서 한글 표시를 위함)
# ----------------------------------------
try:
    from matplotlib import font_manager, rc

    # 시스템에 설치된 맑은 고딕 폰트 경로 지정
    font_path = '../resData/malgun.ttf'
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)  # matplotlib에 한글 폰트 설정
except:
    # 한글 폰트가 없을 경우 기본 폰트 사용
    plt.rcParams['font.family'] = ['DejaVu Sans']
    print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")


# ==========================================================================================================
# 🔹 데이터 전처리 클래스
# ==========================================================================================================
class AgricultureDataPreprocessor:
    """
    농산물 가격 예측을 위한 데이터 전처리 클래스

    주요 기능:
    1. 시간 기반 특성 생성 (계절성, 주기성 인코딩)
    2. 지연(lag) 특성 생성 (과거 가격/생산량 정보)
    3. 상호작용 특성 생성 (변수 간 관계)
    4. 이상치 제거 및 결측값 처리
    5. 로그 변환을 통한 데이터 안정화
    """

    def __init__(self):
        """
        클래스 초기화
        - scaler_features: 입력 특성들을 정규화하는 스케일러
        - scaler_target: 예측 대상(가격)을 정규화하는 스케일러
        - feature_names: 생성된 특성 이름들을 저장하는 리스트
        """
        self.scaler_features = StandardScaler()  # 평균 0, 표준편차 1로 정규화
        self.scaler_target = StandardScaler()  # 타겟 변수도 동일하게 정규화
        self.feature_names = []  # 특성 이름 저장용

    def create_time_features(self, df):
        """
        시간 기반 특성을 생성하는 함수

        매개변수:
        - df: 날짜 컬럼이 포함된 DataFrame

        반환값:
        - 시간 특성이 추가된 DataFrame

        생성되는 특성들:
        1. 기본 시간 정보: 년, 월, 일, 요일, 주차, 년중일차
        2. 주기적 인코딩: sin/cos 변환으로 순환 특성 표현
        3. 계절 더미 변수: 봄, 여름, 가을, 겨울
        4. 농업 특수 시기: 명절, 김장철 등
        """
        df = df.copy()  # 원본 데이터 보존을 위한 복사본 생성

        # ====== 기본 시간 특성 생성 ======
        df['년'] = df['날짜'].dt.year  # 연도 (2020, 2021, ...)
        df['월'] = df['날짜'].dt.month  # 월 (1-12)
        df['일'] = df['날짜'].dt.day  # 일 (1-31)
        df['요일'] = df['날짜'].dt.dayofweek  # 요일 (0=월요일, 6=일요일)
        df['주차'] = df['날짜'].dt.isocalendar().week  # 해당 연도의 주차 (1-53)
        df['년중일차'] = df['날짜'].dt.dayofyear  # 1월 1일부터의 일수 (1-365/366)

        # ====== 주기적 인코딩 (Cyclical Encoding) ======
        # 순환하는 시간 특성을 sin/cos로 변환하여 연속성 보장
        # 예: 12월과 1월이 인접하다는 정보를 모델이 학습할 수 있음
        df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)  # 월의 sin 변환
        df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)  # 월의 cos 변환
        df['일_sin'] = np.sin(2 * np.pi * df['년중일차'] / 365)  # 연중 일차의 sin 변환
        df['일_cos'] = np.cos(2 * np.pi * df['년중일차'] / 365)  # 연중 일차의 cos 변환
        df['요일_sin'] = np.sin(2 * np.pi * df['요일'] / 7)  # 요일의 sin 변환
        df['요일_cos'] = np.cos(2 * np.pi * df['요일'] / 7)  # 요일의 cos 변환

        # ====== 계절 더미 변수 생성 ======
        # 각 계절을 0 또는 1로 표현하는 이진 변수
        df['봄'] = ((df['월'] >= 3) & (df['월'] <= 5)).astype(int)  # 3-5월
        df['여름'] = ((df['월'] >= 6) & (df['월'] <= 8)).astype(int)  # 6-8월
        df['가을'] = ((df['월'] >= 9) & (df['월'] <= 11)).astype(int)  # 9-11월
        df['겨울'] = ((df['월'] == 12) | (df['월'] <= 2)).astype(int)  # 12, 1, 2월

        # ====== 농업 특수 시기 특성 ======
        # 한국 농업에서 중요한 시기들을 나타내는 특성
        df['추석시즌'] = ((df['월'] == 9) & (df['일'] >= 10)).astype(int)  # 9월 중순 이후 (추석 전후)
        df['김장시즌'] = ((df['월'] == 11) & (df['일'] >= 1)).astype(int)  # 11월 (김장철)
        df['설날시즌'] = ((df['월'] <= 2)).astype(int)  # 1-2월 (설날 시즌)

        return df

    def create_lag_features(self, df, target_col='가격', lags=[1, 3, 7, 14, 30]):
        """
        지연(lag) 특성을 생성하는 함수

        매개변수:
        - df: 입력 DataFrame
        - target_col: 지연 특성을 생성할 대상 컬럼 (기본값: '가격')
        - lags: 지연시킬 일수 리스트 (기본값: [1, 3, 7, 14, 30])

        반환값:
        - 지연 특성이 추가된 DataFrame

        생성되는 특성들:
        1. 지연 특성: 과거 N일 전의 가격/생산량 정보
        2. 이동평균: 과거 N일간의 평균값
        3. 변동성: 과거 N일간의 표준편차
        """
        df = df.copy()

        # ====== 지연 특성 생성 ======
        # 과거의 가격과 생산량 정보를 현재 예측에 활용
        for lag in lags:
            # 가격의 지연 특성: lag일 전의 가격
            df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
            # 생산량의 지연 특성: lag일 전의 생산량
            df[f'생산량_lag{lag}'] = df['생산량'].shift(lag)

        # ====== 이동평균 특성 생성 ======
        # 최근 window일간의 평균을 계산하여 트렌드 파악
        for window in [7, 14, 30]:  # 1주, 2주, 1달 평균
            # 가격의 이동평균
            df[f'{target_col}_ma{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            # 생산량의 이동평균
            df[f'생산량_ma{window}'] = df['생산량'].rolling(window=window, min_periods=1).mean()

        # ====== 변동성 특성 생성 ======
        # 최근 기간의 변동성을 측정하여 시장 불안정성 파악
        for window in [7, 14]:  # 1주, 2주 변동성
            # 가격의 표준편차 (변동성)
            df[f'{target_col}_std{window}'] = df[target_col].rolling(window=window, min_periods=1).std()

        return df

    def create_interaction_features(self, df):
        """
        상호작용 특성을 생성하는 함수

        매개변수:
        - df: 입력 DataFrame

        반환값:
        - 상호작용 특성이 추가된 DataFrame

        생성되는 특성들:
        1. 생산량-계절 상호작용: 각 계절별 생산량
        2. 비율 특성: 생산량과 가격의 비율
        """
        df = df.copy()

        # ====== 생산량-계절 상호작용 특성 ======
        # 계절별 생산량 패턴을 파악하기 위한 특성
        df['생산량_봄'] = df['생산량'] * df['봄']  # 봄철 생산량
        df['생산량_여름'] = df['생산량'] * df['여름']  # 여름철 생산량
        df['생산량_가을'] = df['생산량'] * df['가을']  # 가을철 생산량 (수확철)
        df['생산량_겨울'] = df['생산량'] * df['겨울']  # 겨울철 생산량

        # ====== 비율 특성 생성 ======
        # 생산량 대비 가격의 비율 (공급량 대비 가격 수준)
        # 1e-8을 더해서 0으로 나누는 오류 방지
        df['생산량_가격비율'] = df['생산량'] / (df['가격'] + 1e-8)

        return df

    def preprocess_data(self, df):
        """
        전체 전처리 파이프라인을 실행하는 메인 함수

        매개변수:
        - df: 원본 DataFrame (날짜, 생산량, 가격 컬럼 포함)

        반환값:
        - 전처리가 완료된 DataFrame

        처리 과정:
        1. 데이터 정렬
        2. 이상치 제거
        3. 시간 특성 생성
        4. 지연 특성 생성
        5. 상호작용 특성 생성
        6. 결측값 처리
        7. 로그 변환
        """
        print("🔄 데이터 전처리 시작...")

        # ====== 1. 날짜 순서로 정렬 ======
        # 시계열 데이터는 시간 순서가 중요하므로 반드시 정렬
        df = df.sort_values('날짜').reset_index(drop=True)

        # ====== 2. 이상치 제거 (IQR 방법) ======
        # Interquartile Range를 이용한 이상치 탐지 및 제거
        Q1 = df['가격'].quantile(0.25)  # 1사분위수 (25퍼센타일)
        Q3 = df['가격'].quantile(0.75)  # 3사분위수 (75퍼센타일)
        IQR = Q3 - Q1  # 사분위수 범위
        lower_bound = Q1 - 1.5 * IQR  # 하한선
        upper_bound = Q3 + 1.5 * IQR  # 상한선

        original_len = len(df)
        # 이상치 제거: 하한선과 상한선을 벗어나는 데이터 제거
        df = df[(df['가격'] >= lower_bound) & (df['가격'] <= upper_bound)]
        print(f"   이상치 제거: {original_len} → {len(df)} 행")

        # ====== 3. 시간 특성 생성 ======
        df = self.create_time_features(df)

        # ====== 4. 지연 특성 생성 ======
        df = self.create_lag_features(df)

        # ====== 5. 상호작용 특성 생성 ======
        df = self.create_interaction_features(df)

        # ====== 6. 결측값 처리 ======
        # 순전파(ffill): 이전 값으로 채우기
        # 역전파(bfill): 다음 값으로 채우기
        df = df.ffill().bfill()

        # ====== 7. 로그 변환 ======
        # 원본 데이터 백업 (나중에 역변환할 때 사용)
        df['가격_원본'] = df['가격'].copy()
        df['생산량_원본'] = df['생산량'].copy()

        # 로그 변환: 데이터의 분산을 줄이고 정규분포에 가깝게 만듦
        # log1p 사용: log(1 + x) 계산으로 0값 처리 안전
        df['가격'] = np.log1p(df['가격'])
        df['생산량'] = np.log1p(df['생산량'])

        # 로그 변환된 특성들도 업데이트
        for col in df.columns:
            if 'lag' in col or 'ma' in col or 'std' in col:
                if '가격' in col:
                    df[col] = np.log1p(df[col])  # 가격 관련 특성 로그 변환
                elif '생산량' in col:
                    df[col] = np.log1p(df[col])  # 생산량 관련 특성 로그 변환

        print(f"✅ 전처리 완료. 최종 데이터 크기: {df.shape}")
        return df


# ==========================================================================================================
# 🔹 LSTM 모델 클래스
# ==========================================================================================================
class EnhancedLSTMModel:
    """
    향상된 LSTM 기반 농산물 가격 예측 모델

    주요 기능:
    1. 다층 LSTM 신경망 구조
    2. 드롭아웃과 배치 정규화를 통한 과적합 방지
    3. 시퀀스 데이터 생성 및 학습
    4. 성능 평가 및 시각화

    LSTM (Long Short-Term Memory):
    - 순환신경망(RNN)의 한 종류
    - 장기 의존성 문제 해결
    - 시계열 데이터 패턴 학습에 특화
    """

    def __init__(self, sequence_length=30):
        """
        모델 초기화

        매개변수:
        - sequence_length: 입력 시퀀스 길이 (기본값: 30일)
                          30일간의 과거 데이터를 보고 다음 날 가격 예측
        """
        self.sequence_length = sequence_length  # 시퀀스 길이 설정
        self.model = None  # 학습된 모델 저장용
        self.scaler_X = StandardScaler()  # 입력 특성 정규화용 스케일러
        self.scaler_y = StandardScaler()  # 타겟 변수 정규화용 스케일러

    def create_model(self, n_features):
        """
        LSTM 모델 구조를 생성하는 함수

        매개변수:
        - n_features: 입력 특성의 개수

        반환값:
        - 컴파일된 Keras 모델

        모델 구조:
        1. 첫 번째 LSTM층 (128 유닛, return_sequences=True)
        2. 두 번째 LSTM층 (64 유닛, return_sequences=True)
        3. 세 번째 LSTM층 (32 유닛, return_sequences=False)
        4. 완전연결층들 (Dense layers)

        각 층 사이에 드롭아웃과 배치 정규화 적용
        """
        model = Sequential([
            # ====== 첫 번째 LSTM층 ======
            # 128개 뉴런, 다음 층으로 시퀀스 전달 (return_sequences=True)
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            Dropout(0.2),  # 20% 뉴런을 무작위로 비활성화 (과적합 방지)
            BatchNormalization(),  # 배치 정규화 (학습 안정화)

            # ====== 두 번째 LSTM층 ======
            # 64개 뉴런, 계속 시퀀스 전달
            LSTM(64, return_sequences=True),
            Dropout(0.2),  # 20% 드롭아웃
            BatchNormalization(),  # 배치 정규화

            # ====== 세 번째 LSTM층 ======
            # 32개 뉴런, 마지막 출력만 전달 (return_sequences=False)
            LSTM(32, return_sequences=False),
            Dropout(0.2),  # 20% 드롭아웃
            BatchNormalization(),  # 배치 정규화

            # ====== 완전연결층 (Dense layers) ======
            Dense(64, activation='relu'),  # 64개 뉴런, ReLU 활성화 함수
            Dropout(0.3),  # 30% 드롭아웃 (더 강한 정규화)
            Dense(32, activation='relu'),  # 32개 뉴런, ReLU 활성화 함수
            Dropout(0.2),  # 20% 드롭아웃
            Dense(1)  # 최종 출력층 (가격 예측값 1개)
        ])

        # ====== 모델 컴파일 ======
        model.compile(
            optimizer=Adam(learning_rate=0.001),  # Adam 옵티마이저, 학습률 0.001
            loss='huber',  # Huber 손실함수 (이상치에 덜 민감)
            metrics=['mae']  # 평가 지표: 평균절대오차
        )

        return model

    def create_sequences(self, data, target):
        """
        시계열 데이터를 LSTM 입력에 맞는 시퀀스 형태로 변환

        매개변수:
        - data: 입력 특성 데이터 (2D 배열)
        - target: 예측 대상 데이터 (1D 배열)

        반환값:
        - X: 3D 시퀀스 데이터 (samples, timesteps, features)
        - y: 해당하는 타겟 값들

        예시:
        sequence_length=3일 때
        원본: [1,2,3,4,5] → X=[[1,2,3],[2,3,4]], y=[4,5]
        """
        X, y = [], []

        # sequence_length만큼의 과거 데이터로 다음 값 예측
        for i in range(self.sequence_length, len(data)):
            # i-sequence_length부터 i까지의 데이터를 입력으로 사용
            X.append(data[i - self.sequence_length:i])
            # i번째 타겟 값을 예측 대상으로 사용
            y.append(target[i])

        return np.array(X), np.array(y)

    def prepare_data(self, df):
        """
        모델 학습을 위한 데이터 준비

        매개변수:
        - df: 전처리된 DataFrame

        반환값:
        - X_seq: 시퀀스 형태의 입력 데이터
        - y_seq: 해당하는 타겟 데이터
        - available_features: 사용된 특성 이름 리스트
        """
        # ====== 특성 선택 ======
        # 모델 학습에 사용할 특성들을 정의
        feature_cols = [
            # 기본 특성
            '생산량',
            # 시간 주기 특성
            '월_sin', '월_cos', '일_sin', '일_cos', '요일_sin', '요일_cos',
            # 계절 특성
            '봄', '여름', '가을', '겨울',
            # 특수 시즌
            '추석시즌', '김장시즌', '설날시즌',
            # 지연 특성
            '가격_lag1', '가격_lag3', '가격_lag7', '생산량_lag1', '생산량_lag3',
            # 이동평균 특성
            '가격_ma7', '가격_ma14', '생산량_ma7', '생산량_ma14',
            # 변동성 및 상호작용 특성
            '가격_std7', '생산량_가격비율', '생산량_봄', '생산량_가을'
        ]

        # 실제 데이터에 존재하는 특성만 선택
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"사용 가능한 특성: {len(available_features)}개")

        # ====== 최소 특성 개수 확인 ======
        if len(available_features) < 10:  # 최소 10개 특성 필요
            print("⚠️ 충분한 특성이 생성되지 않았습니다. 기본 특성을 추가합니다.")
            # 기본 특성 추가
            basic_features = ['월', '일', '요일', '주차']
            for feat in basic_features:
                if feat in df.columns and feat not in available_features:
                    available_features.append(feat)

        # ====== 데이터 추출 ======
        X = df[available_features].values  # 입력 특성 데이터
        y = df['가격'].values  # 예측 대상 (가격)

        # ====== 데이터 정규화 ======
        # StandardScaler: 평균=0, 표준편차=1로 정규화
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # ====== 시퀀스 데이터 생성 ======
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)

        return X_seq, y_seq, available_features

    def train(self, df, test_size=0.2):
        """
        모델 학습 메인 함수

        매개변수:
        - df: 전처리된 DataFrame
        - test_size: 테스트 데이터 비율 (기본값: 20%)

        반환값:
        - 성능 지표와 결과가 포함된 딕셔너리
        """
        print("🚀 모델 학습 시작...")

        # ====== 1. 데이터 준비 ======
        X_seq, y_seq, feature_names = self.prepare_data(df)
        print(f"시퀀스 데이터 형태: X={X_seq.shape}, y={y_seq.shape}")

        # ====== 2. 학습/테스트 데이터 분할 ======
        # 시계열 데이터이므로 시간 순서를 유지하여 분할 (무작위 분할 X)
        split_idx = int(len(X_seq) * (1 - test_size))

        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]  # 입력 데이터 분할
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]  # 타겟 데이터 분할

        print(f"학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

        # ====== 3. 모델 생성 ======
        # X_train.shape[2]: 특성의 개수
        self.model = self.create_model(X_train.shape[2])

        # ====== 4. 콜백 설정 ======
        # 학습 과정을 모니터링하고 최적화하는 콜백들
        callbacks = [
            # 조기 종료: 검증 손실이 15 에포크 동안 개선되지 않으면 학습 중단
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            # 학습률 감소: 검증 손실이 10 에포크 동안 개선되지 않으면 학습률을 절반으로 감소
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
        ]

        # 학습
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # 예측 및 평가
        y_pred_scaled = self.model.predict(X_test)

        # 역변환
        y_pred = np.expm1(self.scaler_y.inverse_transform(y_pred_scaled).flatten())
        y_true = np.expm1(self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten())

        # 성능 계산
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        print(f"\n📊 모델 성능:")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")

        # 시각화
        self.plot_results(y_true, y_pred, history)

        return {
            'mse': mse, 'mae': mae, 'r2': r2, 'mape': mape,
            'y_true': y_true, 'y_pred': y_pred, 'history': history
        }

    def plot_results(self, y_true, y_pred, history):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 예측 결과
        axes[0, 0].plot(y_true, label='실제 가격', alpha=0.8)
        axes[0, 0].plot(y_pred, label='예측 가격', alpha=0.8)
        axes[0, 0].set_title('🎯 가격 예측 결과')
        axes[0, 0].set_xlabel('시간')
        axes[0, 0].set_ylabel('가격')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 산점도
        axes[0, 1].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('실제 가격')
        axes[0, 1].set_ylabel('예측 가격')
        axes[0, 1].set_title('📊 실제 vs 예측 가격')
        axes[0, 1].grid(True, alpha=0.3)

        # 학습 곡선
        axes[1, 0].plot(history.history['loss'], label='Train Loss')
        axes[1, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[1, 0].set_title('📉 학습 곡선')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 오차 분포
        errors = y_true - y_pred
        axes[1, 1].hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('📊 예측 오차 분포')
        axes[1, 1].set_xlabel('오차')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ----------------------------------------
# 🔹 메인 실행 함수
# ----------------------------------------
def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("🌾 농산물 가격 예측 AI 모델 학습")
    print("=" * 50)

    # 데이터 로드
    file_path = './data/prod-price/마늘_생산량_가격_데이터.csv'

    try:
        df = pd.read_csv(file_path)
        print(f"✅ 데이터 로드 성공: {df.shape}")
    except FileNotFoundError:
        print("⚠️ 데이터 파일을 찾을 수 없어 샘플 데이터를 생성합니다.")
        # 샘플 데이터 생성 (더 현실적)
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2024-06-30', freq='D')

        # 계절성이 있는 생산량 (여름/가을에 많음)
        seasonal_pattern = np.sin(np.arange(len(dates)) * 2 * np.pi / 365 + np.pi / 2) * 300
        production = 1000 + seasonal_pattern + np.random.normal(0, 100, len(dates))

        # 생산량과 반비례하는 가격 (계절성 + 노이즈)
        price_seasonal = np.sin(np.arange(len(dates)) * 2 * np.pi / 365 - np.pi / 2) * 400
        price = 2000 + price_seasonal + np.random.normal(0, 200, len(dates))
        # 생산량이 많을 때 가격 하락 효과
        price = price - (production - 1000) * 0.3

        df = pd.DataFrame({
            '날짜': dates,
            '생산량': np.abs(production),
            '가격': np.abs(price)
        })

        print(f"✅ 샘플 데이터 생성: {df.shape}")

    # 날짜 컬럼 변환
    df['날짜'] = pd.to_datetime(df['날짜'])

    # 전처리
    preprocessor = AgricultureDataPreprocessor()
    df_processed = preprocessor.preprocess_data(df)

    # 모델 학습
    model = EnhancedLSTMModel(sequence_length=30)
    results = model.train(df_processed)

    print("=" * 50)
    print("🎉 학습 완료!")
    print(f"최종 R² 점수: {results['r2']:.4f}")
    print(f"최종 MAPE: {results['mape']:.2f}%")
    print("=" * 50)

    return model, preprocessor, results


# 실행
if __name__ == "__main__":
    model, preprocessor, results = main()