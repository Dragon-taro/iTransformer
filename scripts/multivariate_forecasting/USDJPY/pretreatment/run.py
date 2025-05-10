import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib  # joblibをインポート

# Identify CSV file in .data
csv_file = os.path.join(os.path.dirname(__file__), 'row_data', 'USDJPY_2024_M1.csv')
print(f"Using file: {csv_file}")

# Load a small sample to inspect columns
df_sample = pd.read_csv(csv_file, nrows=5)
print("Sample columns:", df_sample.columns.tolist())

# Full load and datetime parsing
df = pd.read_csv(csv_file)

# Dynamic datetime parsing
df['Datetime'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')  # "Gmt time" 列と形式を指定
df = df.drop(columns=['Gmt time'])

# Prepare time series index
df = df.set_index('Datetime').sort_index()

# Resample to 1-minute frequency and forward-fill missing
df = df.resample('1T').ffill()

# Select price columns for normalization
price_cols = ['Open', 'High', 'Low', 'Close']
if not all(col in df.columns for col in price_cols):
    price_cols = df.columns.drop('Volume') if 'Volume' in df.columns else df.columns

data = df[price_cols].values

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Generate sliding windows
window_size = 60  # past 60 minutes
target_size = 10  # next 10 minutes
X, y = [], []
for i in range(len(scaled) - window_size - target_size + 1):
    X.append(scaled[i:i+window_size])
    y.append(scaled[i+window_size:i+window_size+target_size, price_cols.index('Close')])

X = np.array(X)
y = np.array(y)

# Split 80/10/10 for train/validation/test
n = len(X)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Show dataset shapes
print("Train shapes:", X_train.shape, y_train.shape)
print("Val shapes:  ", X_val.shape,   y_val.shape)
print("Test shapes: ", X_test.shape,  y_test.shape)

# Save preprocessed arrays
output_path = os.path.join(os.path.dirname(__file__), 'pretreatment_data', 'usdjpy_windows.npz')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.savez(output_path,
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test,
         scaler_min=scaler.data_min_, scaler_max=scaler.data_max_)

# スケーラーオブジェクトをjoblibで直接保存
scaler_path = os.path.join(os.path.dirname(output_path), 'usdjpy_scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"スケーラーを保存しました: {scaler_path}")

# 追加: 各データセットをiTransformer標準形式でCSV保存
base_dir = os.path.join(os.path.dirname(output_path))  # pretreatment/pretreatment_data
os.makedirs(base_dir, exist_ok=True)
# カラム名
columns = ['date'] + [col for col in price_cols for _ in range(window_size)]
# 各windowの先頭日時をdate列に
window_dates = df.index[:len(X)]
train_dates = window_dates[:train_end]
val_dates = window_dates[train_end:val_end]
test_dates = window_dates[val_end:]

def save_with_date(X, dates, path):
    X_reshaped = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
    if len(X.shape) > 2:
        # 入力データ（X）の場合
        df_out = pd.DataFrame(X_reshaped, columns=[col for col in price_cols for _ in range(window_size)])
    else:
        # 予測データ（y）の場合
        df_out = pd.DataFrame(X_reshaped, columns=[f'Close.{i+1}' for i in range(X.shape[1])])
    df_out.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
    df_out.to_csv(path, index=False)

# 入力データ（X）の保存
save_with_date(X_train, train_dates, os.path.join(base_dir, 'usdjpy_X_train_wdate.csv'))
save_with_date(X_val, val_dates, os.path.join(base_dir, 'usdjpy_X_val_wdate.csv'))
save_with_date(X_test, test_dates, os.path.join(base_dir, 'usdjpy_X_test_wdate.csv'))

# 予測データ（y）の保存
save_with_date(y_train, train_dates, os.path.join(base_dir, 'usdjpy_y_train_wdate.csv'))
save_with_date(y_val, val_dates, os.path.join(base_dir, 'usdjpy_y_val_wdate.csv'))
save_with_date(y_test, test_dates, os.path.join(base_dir, 'usdjpy_y_test_wdate.csv'))

print(f"Saved preprocessed data to {output_path} and CSV files in {base_dir}/")
