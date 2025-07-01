import joblib
import pandas as pd

rf_model = joblib.load('./models/딸기/rf_model.pkl')
gb_model = joblib.load('./models/딸기/gb_model.pkl')
scaler = joblib.load('./models/딸기/scaler.pkl')

X_new = pd.read_csv('./data/weath-prod/future_weath.csv')

# 예측용 입력 데이터가 있다면 스케일링 후 예측
X_new_scaled = scaler.transform(X_new)
ensemble_pred = 0.6 * rf_model.predict(X_new_scaled) + 0.4 * gb_model.predict(X_new_scaled)

pre_df = pd.DataFrame(ensemble_pred)

pre_df.to_csv('./data/weath-prod/future_prod')