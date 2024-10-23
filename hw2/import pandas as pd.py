# 必要的庫導入
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime

# 讀取 CSV 資料
file_path = r"C:\Users\locke\OneDrive\桌面\hw2\data.csv"
df = pd.read_csv(file_path)

# 資料處理
# 1. 將 Date 欄位轉換為 datetime 格式
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# 2. 移除 y 欄位中的逗號，並轉換為 float
df['y'] = df['y'].replace({',': ''}, regex=True).astype(float)

# 準備 Prophet 模型所需的資料
df = df[['Date', 'y']].rename(columns={'Date': 'ds'})

# 建立 Prophet 模型
model = Prophet(
    changepoint_prior_scale=0.5,  # 設定變化點靈敏度
    interval_width=0.95            # 設定不確定性區間為 95%
)

# 增加每月季節性，Fourier Order 設為 5
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# 模型訓練
model.fit(df)

# 預測未來 60 天的股票價格
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# 畫出預測結果圖
fig, ax = plt.subplots(figsize=(10, 6))

# 畫出實際數據（黑色線條）
ax.plot(df['ds'], df['y'], 'k-', label='Actual Stock Price')

# 畫出預測數據（藍色線條），以及不確定性區間（淺藍色陰影）
ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecasted Stock Price')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2)

# 添加一條水平虛線表示「Historical Average」
historical_avg = df['y'].mean()
ax.axhline(y=historical_avg, color='gray', linestyle='--', label='Historical Average')

# 在黑色線的尾巴（實際數據的最後一個點）添加紅色箭頭
last_date = df['ds'].iloc[-1]
last_price = df['y'].iloc[-1]

# 調整標註文字的位置，將它放到圖表上方，避免重疊
ax.annotate('Last Actual Point', xy=(last_date, last_price), xytext=(last_date, last_price + 50),  # 將文字位置上移
            arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, color='red')

# 格式化 X 軸日期
date_form = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(date_form)

# 標題與圖例
plt.title('Stock Price Forecast with Prophet')
plt.legend()

# 顯示圖表
plt.show()
