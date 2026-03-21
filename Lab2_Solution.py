import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# BÀI 1: DỮ LIỆU HOUSING
# ==========================================
print("--- BÀI 1: HOUSING ---")
df_h = pd.read_csv('ITA105_Lab_2_Housing.csv')
print("Shape:", df_h.shape)
print("Missing:\n", df_h.isnull().sum())

# Boxplot và Scatterplot
df_h.plot.box(subplots=True, figsize=(10,4), title="Boxplot Housing")
plt.show()

df_h.plot.scatter(x='dien_tich', y='gia', title="Scatter: Diện tích vs Giá")
plt.show()

# Tính Z-score và lọc ngoại lệ (|Z| > 3)
z_h = np.abs((df_h - df_h.mean()) / df_h.std())
df_h_clean = df_h[(z_h < 3).all(axis=1)]

print(f"Số lượng dữ liệu ban đầu: {len(df_h)}")
print(f"Số lượng dữ liệu sau làm sạch (Z<3): {len(df_h_clean)}")
df_h_clean.plot.box(subplots=True, figsize=(10,4), title="Boxplot sau làm sạch")
plt.show()


# ==========================================
# BÀI 2: DỮ LIỆU IOT SENSOR
# ==========================================
print("\n--- BÀI 2: IOT ---")
df_i = pd.read_csv('ITA105_Lab_2_Iot.csv')
df_i['timestamp'] = pd.to_datetime(df_i['timestamp'])
df_i.set_index('timestamp', inplace=True)

# Plot nhiệt độ
for s_id, group in df_i.groupby('sensor_id'):
    group['temperature'].plot(label=s_id, figsize=(10,4), title="Nhiệt độ theo thời gian")
plt.legend()
plt.show()

# Phát hiện ngoại lệ bằng Rolling Mean
rm = df_i.groupby('sensor_id')['temperature'].transform(lambda x: x.rolling(10, min_periods=1).mean())
rstd = df_i.groupby('sensor_id')['temperature'].transform(lambda x: x.rolling(10, min_periods=1).std().fillna(0))
df_i['outlier'] = (df_i['temperature'] > rm + 3*rstd) | (df_i['temperature'] < rm - 3*rstd)

# Boxplot theo sensor
df_i.boxplot(column='temperature', by='sensor_id', figsize=(8,4))
plt.show()

# Scatter Pressure vs Temp (Đỏ là ngoại lệ)
colors = df_i['outlier'].map({True: 'red', False: 'blue'})
df_i.plot.scatter(x='pressure', y='temperature', c=colors, title="Pressure vs Temperature")
plt.show()

# Xử lý: Cắt (clip) ngoại lệ
df_i['temp_clean'] = df_i['temperature'].clip(lower=rm - 3*rstd, upper=rm + 3*rstd)


# ==========================================
# BÀI 3: GIAO DỊCH E-COMMERCE
# ==========================================
print("\n--- BÀI 3: E-COMMERCE ---")
df_e = pd.read_csv('ITA105_Lab_2_Ecommerce.csv')
num_cols = ['price', 'quantity', 'rating']

# Boxplot
df_e[num_cols].plot.box(subplots=True, figsize=(12,4))
plt.show()

# Z-score ngoại lệ
z_e = np.abs((df_e[num_cols] - df_e[num_cols].mean()) / df_e[num_cols].std())
print("Số ngoại lệ (Z>3) mỗi cột:\n", (z_e > 3).sum())

df_e.plot.scatter(x='quantity', y='price', title="Price vs Quantity ban đầu")
plt.show()

# Xử lý: Xóa lỗi (price/qty <= 0) & log transform price
df_e_clean = df_e.copy()
df_e_clean = df_e_clean[(df_e_clean['price'] > 0) & (df_e_clean['quantity'] > 0)]
df_e_clean['log_price'] = np.log1p(df_e_clean['price'])


# ==========================================
# BÀI 4 & 5: MULTIVARIATE OUTLIER
# ==========================================
print("\n--- BÀI 4 & 5: MULTIVARIATE ---")
# Phát hiện ngoại lệ kết hợp: Vừa giá cao vừa số lượng cực lớn 
# (Ở đây ta ví dụ điểm nào có Z_price > 2 và Z_quantity > 2 thì bị coi là ngoại lệ đa biến)
df_e_clean['multi_outlier'] = (z_e['price'] > 2) & (z_e['quantity'] > 2)

colors_multi = df_e_clean['multi_outlier'].map({True: 'red', False: 'blue'})
df_e_clean.plot.scatter(x='quantity', y='price', c=colors_multi, 
                        title="Đa biến: Giá vs Số lượng (Đỏ = Ngoại lệ cả 2)")
plt.show()

print("Hoàn tất chạy code!")
